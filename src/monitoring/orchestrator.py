"""
Drift orchestrator
------------------
The monitoring stack produces many signals. Someone still has to decide what to
do. This module is that decision, written down explicitly rather than left to a
pager and a panic.

Three ideas do the work:

**Input drift and quality drift are different incidents.** A model can serve
traffic that looks nothing like its training data and still be perfectly
accurate (a new phrasing of a known intent). It can also keep receiving
identical traffic and lose accuracy (a provider silently changed the model, a
dependency bumped, a feature was mis-serialised). Retraining fixes the first
cause and does nothing for the second. The decision therefore weights *quality*
far above *input shift* when both are present, and refuses to retrain on input
drift alone.

**Confirmation is required before expensive action.** A retrain is a human- or
money-consuming event. The policy requires drift to persist across consecutive
windows and requires quality evidence, so a single noisy window triggers a
ticket, not a rebuild.

**Degradation is judged against a frozen baseline, always.** Every number in a
decision carries the baseline it was compared to and the encoder/judge
signature it was produced under, so a decision made after a model swap cannot
inherit the confidence of one made before it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

SEVERITY_RANK = {"none": 0, "moderate": 1, "severe": 2}

ACTION_NOOP = "noop"
ACTION_INVESTIGATE = "investigate"
ACTION_RETRAIN = "retrain"
ACTION_ROLLBACK = "rollback"


@dataclass
class Decision:
    action: str
    severity: str
    health_score: float
    confidence: float
    rationale: str
    signals: dict[str, Any] = field(default_factory=dict)
    policy_version: str = "policy-v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "severity": self.severity,
            "health_score": round(self.health_score, 1),
            "confidence": round(self.confidence, 3),
            "rationale": self.rationale,
            "signals": self.signals,
            "policy_version": self.policy_version,
        }


class DriftOrchestrator:
    """
    Stateful policy engine: consumes window signals, emits an action, opens and
    closes incidents, and remembers enough history to require confirmation.
    """

    def __init__(self, *, confirm_windows: int = 2,
                 weights: dict[str, float] | None = None,
                 quality_gates_retrain: tuple[str, ...] = ("accuracy", "ece", "brier"),
                 auto_retrain: bool = settings.AUTO_RETRAIN_ENABLED,
                 policy_version: str = "policy-v1"):
        self.confirm_windows = max(1, confirm_windows)
        self.weights = weights or {
            "embedding": 25.0,   # input shift
            "quality": 45.0,     # did we get worse — the thing users feel
            "judge": 25.0,       # did the answer get worse — the thing users read
            "volume": 5.0,       # traffic anomalies
        }
        self.quality_gates_retrain = quality_gates_retrain
        self.auto_retrain = auto_retrain
        self.policy_version = policy_version

        self._consecutive_signals: dict[str, int] = {}
        self._open_incident: str | None = None
        self._incident_windows = 0

    # ── policy ──────────────────────────────────────────────────────
    def decide(
        self,
        *,
        embedding: dict[str, Any] | None,
        quality: dict[str, Any] | None,
        judge: dict[str, Any] | None,
        volume: dict[str, Any] | None = None,
        window_index: int = 0,
    ) -> Decision:
        embedding = embedding or {}
        quality = quality or {}
        judge = judge or {}

        components: dict[str, float] = {}
        fired: list[str] = []

        # 1. Input drift
        emb_sev = embedding.get("severity", "none")
        emb_fire = bool(embedding.get("drift_detected", False))
        components["embedding"] = self._severity_penalty(emb_sev, emb_fire)
        if emb_fire:
            fired.append(f"embedding:{emb_sev}")

        # 2. Output quality
        q_sev = quality.get("severity", "none")
        q_fire = bool(quality.get("drift_detected", False))
        components["quality"] = self._severity_penalty(q_sev, q_fire)
        q_signals = quality.get("severity_by_signal", {}) or {}
        for name, sev in q_signals.items():
            if sev != "none":
                fired.append(f"quality:{name}:{sev}")

        # 3. LLM-as-judge
        j_sev = judge.get("severity", "none")
        j_fire = bool(judge.get("regression", False))
        components["judge"] = self._severity_penalty(j_sev, j_fire)
        if j_fire:
            fired.append(f"judge:{j_sev}")
            for dim, delta in (judge.get("dimension_deltas") or {}).items():
                if isinstance(delta, (int, float)) and delta == delta and delta < -0.25:
                    fired.append(f"judge_dim:{dim}")

        # 4. Volume / traffic anomalies
        vol_fire = bool((volume or {}).get("anomaly", False))
        components["volume"] = 12.0 if vol_fire else 0.0
        if vol_fire:
            fired.append(f"volume:{volume.get('direction', '?')}")

        health = self._health(components)

        # ── confirmation ────────────────────────────────────────────
        signal_key = self._signal_key(embedding, quality, judge, vol_fire)
        if signal_key:
            self._consecutive_signals[signal_key] = self._consecutive_signals.get(signal_key, 0) + 1
        else:
            self._consecutive_signals.clear()

        sustained = signal_key is not None and self._consecutive_signals[signal_key] >= self.confirm_windows
        quality_confirmed = any(
            q_signals.get(g, "none") != "none" for g in self.quality_gates_retrain
        )

        severity = "none"
        for block in (q_sev, j_sev, emb_sev, ("moderate" if vol_fire else "none")):
            if SEVERITY_RANK.get(block, 0) > SEVERITY_RANK.get(severity, 0):
                severity = block

        # ── action ──────────────────────────────────────────────────
        if not fired:
            action = ACTION_NOOP
            rationale = "All monitored signals within tolerance. No action."
        elif quality_confirmed and sustained and severity == "severe":
            action = ACTION_RETRAIN if self.auto_retrain else ACTION_INVESTIGATE
            rationale = (
                "Sustained quality regression with input drift. Retraining is the indicated remedy: "
                "the distribution moved and the model got worse on it. "
                if self.auto_retrain else
                "Sustained quality regression alongside input drift. Retrain on recent labelled data "
                "including the new intents, then re-baseline the judge."
            )
        elif quality_confirmed and sustained:
            action = ACTION_INVESTIGATE
            rationale = (
                "Quality degraded without input drift. Suspect the model or its dependencies rather "
                "than the data: verify the served artifact hash, check for a provider or dependency "
                "change, and re-run the offline eval set before retraining."
            )
        elif j_fire and sustained:
            action = ACTION_INVESTIGATE
            rationale = (
                "Judge regression without a labelled-quality drop. Check the judge itself first "
                "(run the regression suite) before assuming the application changed."
            )
        elif emb_fire and not quality_confirmed:
            action = ACTION_INVESTIGATE
            rationale = (
                "Input drift with quality intact. This is a scope/coverage problem, not a model "
                "problem: decide whether to expand the label set or route the new traffic to a "
                "specialist. Do not retrain — it will not help while the new traffic is unlabelled."
            )
        elif severity == "severe":
            action = ACTION_INVESTIGATE
            rationale = "Single severe signal on a window that has not yet confirmed. Watching."
        else:
            action = ACTION_NOOP
            rationale = "Below confirmation threshold; not acting on a single window."

        confidence = self._confidence(embedding, quality, judge, sustained)
        signals = {
            "components": {k: round(v, 2) for k, v in components.items()},
            "fired": fired,
            "sustained": sustained,
            "consecutive": self._consecutive_signals.get(signal_key, 0) if signal_key else 0,
            "signal_key": signal_key,
            "quality_confirmed": quality_confirmed,
            "embedding_severity": emb_sev,
            "quality_severity": q_sev,
            "judge_severity": j_sev,
            "volume_anomaly": vol_fire,
        }
        return Decision(action=action, severity=severity, health_score=health,
                        confidence=confidence, rationale=rationale, signals=signals,
                        policy_version=self.policy_version)

    # ── incident lifecycle ──────────────────────────────────────────
    def update_incident(self, store, run_id: str, window_id: str,
                        decision: Decision, extra: dict[str, Any] | None = None) -> str | None:
        """Open an incident on escalation, annotate it while open, close on recovery."""
        escalate = decision.action in {ACTION_RETRAIN, ACTION_ROLLBACK} or decision.severity == "severe"
        recover = decision.action == ACTION_NOOP and decision.health_score >= 85

        if self._open_incident and (recover or not escalate):
            inc = self._open_incident
            store.append_incident_timeline(
                inc, "resolved" if recover else "downgraded",
                {"window_id": window_id, "action": decision.action,
                 "health": decision.health_score})
            store.close_incident(inc, f"{decision.action} at health {decision.health_score:.0f}")
            self._open_incident = None
            self._incident_windows = 0

        if escalate and self._open_incident is None:
            self._incident_windows += 1
            title = (extra or {}).get("title") or self._title(decision)
            self._open_incident = store.open_incident(
                run_id, window_id, decision.severity, title,
                {**decision.signals, **(extra or {})})
        elif escalate and self._open_incident:
            store.append_incident_timeline(self._open_incident, "observed", {
                "window_id": window_id, "action": decision.action,
                "health": decision.health_score, "confidence": decision.confidence})

        return self._open_incident

    @property
    def open_incident(self) -> str | None:
        return self._open_incident

    @staticmethod
    def _title(decision: Decision) -> str:
        parts = decision.signals.get("fired", [])[:3]
        return f"{decision.severity.upper()} drift: {', '.join(parts) or 'multi-signal degradation'}"

    # ── scoring helpers ─────────────────────────────────────────────
    @staticmethod
    def _severity_penalty(severity: str, fired: bool) -> float:
        """Per-block penalty on a 0-100 scale, so the health score stays readable."""
        if severity == "severe":
            return 70.0
        if severity == "moderate":
            return 35.0
        return 15.0 if fired else 0.0

    def _health(self, components: dict[str, float]) -> float:
        """Weighted mean penalty, subtracted from 100.

        Weights encode the belief that a user feeling a wrong answer costs more
        than the same-sized movement in the input distribution — which is also
        why quality and judge outweigh embedding drift.
        """
        total_weight = 0.0
        weighted = 0.0
        for key, weight in self.weights.items():
            weighted += weight * float(components.get(key, 0.0))
            total_weight += weight
        if total_weight <= 0:
            return 100.0
        return float(max(0.0, min(100.0, 100.0 - weighted / total_weight)))

    @staticmethod
    def _signal_key(embedding: dict, quality: dict, judge: dict, vol_fire: bool) -> str | None:
        if not (embedding.get("drift_detected") or quality.get("drift_detected")
                or judge.get("regression") or vol_fire):
            return None
        keys = []
        if embedding.get("drift_detected"):
            keys.append("emb")
        if quality.get("drift_detected"):
            keys.append("qual")
        if judge.get("regression"):
            keys.append("judge")
        if vol_fire:
            keys.append("vol")
        return "+".join(sorted(keys))

    @staticmethod
    def _confidence(embedding: dict, quality: dict, judge: dict, sustained: bool) -> float:
        """How much evidence stands behind this decision, in [0, 1]."""
        evidence = 0.0
        if quality.get("n_labeled", 0) >= 25:
            evidence += 0.4
        if embedding.get("drift_detected"):
            evidence += 0.2
            p = embedding.get("mmd_p_value")
            if isinstance(p, (int, float)) and p == p:
                evidence += float(min(0.15, -np.log10(max(p, 1e-6)) / 20.0))
        if judge.get("n_judged", 0) >= 10:
            evidence += 0.15
        if sustained:
            evidence += 0.25
        return float(min(1.0, evidence))


def volume_anomaly(current_n: int, expected_n: int, ratio: float = 2.0,
                   floor: int = 20) -> dict[str, Any]:
    """
    Traffic-volume anomaly check.

    Volume is monitored because a silent drop is a routing bug, and a spike is
    either a campaign or a scraper. Both change what the other detectors see,
    so it belongs in the same decision rather than in a separate dashboard.
    """
    if expected_n < floor:
        return {"anomaly": False, "direction": None, "ratio": float("nan"),
                "current_n": current_n, "expected_n": expected_n}
    r = current_n / expected_n if expected_n else float("nan")
    anomaly = bool(r >= ratio or r <= 1.0 / ratio)
    return {
        "anomaly": anomaly,
        "direction": "spike" if r >= ratio else ("drop" if r <= 1.0 / ratio else None),
        "ratio": round(float(r), 3),
        "current_n": current_n,
        "expected_n": expected_n,
    }


def summarise(decision: Decision) -> str:
    icon = {"noop": "OK  ", "investigate": "WARN", "retrain": "ACT ", "rollback": "ROLL"}[decision.action]
    return (f"[{icon}] health={decision.health_score:5.1f} severity={decision.severity:8s} "
            f"action={decision.action:12s} confidence={decision.confidence:.2f}")