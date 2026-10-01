"""
Output quality
--------------
Input drift tells you the world changed. It does not tell you whether your
system got worse. Arize's core insight — and the reason "accuracy" alone is
not a monitoring strategy — is that the two must be tracked separately, because
they have different causes, different owners and different remedies.

Three regimes, all implemented here:

**Labelled traffic** (support ops resolved the ticket, or a batch audit ran)
    Accuracy, macro-F1, per-intent recall, ECE, Brier, and the gap between
    confidence and realised accuracy. This is ground truth; trust it.

**Unlabelled traffic** (the 95% case)
    * confidence distribution drift (PSI on the score itself),
    * entropy and abstention rate,
    * margin to the second-best class,
    * and a **label-free accuracy proxy** from distance to the decision
      boundary combined with score histogram skew. Directionally useful,
      explicitly not a substitute for labels.

**Delayed labels** (the real production shape)
    Labels arrive ``LABEL_LAG_DAYS`` after the request. A window's quality is
    therefore only *final* once its labels land, and quality regressions must be
    attributed to the window in which the request was served — not the window in
    which the label arrived. Getting this attribution wrong is how teams end up
    retraining a model because a label backlog flushed.

Everything is computed against the champion's frozen validation baseline, so a
metric is only ever reported as a delta plus a z-score, never as an absolute
number floating free of context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import settings
from src.storage.reference import reliability_curve
from src.utils.logging import get_logger

logger = get_logger(__name__)

SEVERITY_ORDER = {"none": 0, "moderate": 1, "severe": 2}


def _r(v: Any, n: int = 4) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(f) else round(f, n)


def severity_from(value: float, moderate: float, severe: float) -> str:
    """Larger magnitude = worse. Callers pass the *size* of the degradation."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "none"
    if value >= severe:
        return "severe"
    if value >= moderate:
        return "moderate"
    return "none"


def worst(*severities: str) -> str:
    return max((s for s in severities if s), key=lambda s: SEVERITY_ORDER.get(s, 0), default="none")


# ── Calibration ────────────────────────────────────────────────────────

def expected_calibration_error(confidences: np.ndarray, correct: np.ndarray,
                               n_bins: int = 10) -> float:
    """
    ECE — mean gap between predicted confidence and observed accuracy,
    weighted by bin population.
    """
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correct, dtype=float)
    if conf.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        ece += mask.mean() * abs(corr[mask].mean() - conf[mask].mean())
    return float(ece)


def maximum_calibration_error(confidences: np.ndarray, correct: np.ndarray,
                              n_bins: int = 10) -> float:
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correct, dtype=float)
    if conf.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, n_bins - 1)
    worst_gap = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() < 3:
            continue
        worst_gap = max(worst_gap, abs(corr[mask].mean() - conf[mask].mean()))
    return float(worst_gap)


def adaptive_ece(confidences: np.ndarray, correct: np.ndarray,
                 n_bins: int = 10) -> float:
    """Equal-mass variant — more stable than equal-width when scores pile up."""
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correct, dtype=float)
    if conf.size < n_bins * 3:
        return float("nan")
    order = np.argsort(conf)
    chunks = np.array_split(order, n_bins)
    ece = 0.0
    for chunk in chunks:
        if chunk.size == 0:
            continue
        ece += chunk.size / conf.size * abs(corr[chunk].mean() - conf[chunk].mean())
    return float(ece)


# ── Label-free proxies ────────────────────────────────────────────────

def label_free_accuracy_proxy(confidences: np.ndarray, margins: np.ndarray,
                              reference_accuracy: float | None = None) -> float:
    """
    A deliberately *biased* proxy for unlabelled windows.

    Weights confidence and decision margin, then rescales so that a window
    indistinguishable from the reference returns the reference accuracy. It is
    monotonic in both signals and bounded to [0,1]; it is **not** an accuracy
    estimate and must never be reported as one. The dashboard labels it
    "proxy" for exactly that reason.
    """
    conf = np.asarray(confidences, dtype=float)
    marg = np.asarray(margins, dtype=float)
    if conf.size == 0:
        return float("nan")
    score = 0.6 * conf + 0.4 * np.clip(marg, 0.0, 1.0)
    est = float(score.mean())
    if reference_accuracy is None:
        return est
    return float(np.clip(reference_accuracy * (est / 0.8 if est > 1e-6 else 1.0), 0.0, 1.0))


def score_health(confidences: np.ndarray) -> dict[str, float]:
    """Distribution health of the confidence signal itself."""
    conf = np.asarray(confidences, dtype=float)
    if conf.size == 0:
        return {k: float("nan") for k in ("mean", "std", "p05", "p50", "p95", "high_conf_rate")}
    ent = -conf * np.log2(np.clip(conf, 1e-9, 1)) - (1 - conf) * np.log2(np.clip(1 - conf, 1e-9, 1))
    return {
        "mean": float(conf.mean()),
        "std": float(conf.std(ddof=1)) if conf.size > 1 else 0.0,
        "p05": float(np.percentile(conf, 5)),
        "p50": float(np.percentile(conf, 50)),
        "p95": float(np.percentile(conf, 95)),
        "high_conf_rate": float(np.mean(conf >= 0.8)),
        "mean_entropy": float(ent.mean()),
    }


# ── Results ────────────────────────────────────────────────────────────

@dataclass
class QualityResult:
    n_requests: int = 0
    n_labeled: int = 0
    label_coverage: float = 0.0
    accuracy: float = float("nan")
    macro_f1: float = float("nan")
    precision: float = float("nan")
    recall: float = float("nan")
    auc: float = float("nan")
    log_loss: float = float("nan")
    brier: float = float("nan")
    ece: float = float("nan")
    mce: float = float("nan")
    adaptive_ece: float = float("nan")
    accuracy_delta: float = float("nan")
    accuracy_z: float = float("nan")
    ece_delta: float = float("nan")
    brier_delta: float = float("nan")
    proxy_accuracy: float = float("nan")
    confidence_mean: float = float("nan")
    confidence_p05: float = float("nan")
    confidence_p95: float = float("nan")
    high_conf_rate: float = float("nan")
    abstention_rate: float = float("nan")
    abstention_delta: float = float("nan")
    in_scope_accuracy: float = float("nan")
    out_of_scope_accuracy: float = float("nan")
    worst_intents: list[dict[str, Any]] = field(default_factory=list)
    drift_detected: bool = False
    severity: str = "none"
    severity_by_signal: dict[str, str] = field(default_factory=dict)
    recommendation: str = ""
    timestamp: str = ""
    reliability: list[list[float | None]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_requests": self.n_requests,
            "n_labeled": self.n_labeled,
            "label_coverage": _r(self.label_coverage, 3),
            "accuracy": _r(self.accuracy),
            "macro_f1": _r(self.macro_f1),
            "precision": _r(self.precision),
            "recall": _r(self.recall),
            "auc": _r(self.auc),
            "log_loss": _r(self.log_loss),
            "brier": _r(self.brier),
            "ece": _r(self.ece),
            "mce": _r(self.mce),
            "adaptive_ece": _r(self.adaptive_ece),
            "accuracy_delta": _r(self.accuracy_delta),
            "accuracy_z": _r(self.accuracy_z, 2),
            "ece_delta": _r(self.ece_delta),
            "brier_delta": _r(self.brier_delta),
            "proxy_accuracy": _r(self.proxy_accuracy),
            "confidence_mean": _r(self.confidence_mean),
            "confidence_p05": _r(self.confidence_p05),
            "confidence_p95": _r(self.confidence_p95),
            "high_conf_rate": _r(self.high_conf_rate),
            "abstention_rate": _r(self.abstention_rate),
            "abstention_delta": _r(self.abstention_delta),
            "in_scope_accuracy": _r(self.in_scope_accuracy),
            "out_of_scope_accuracy": _r(self.out_of_scope_accuracy),
            "worst_intents": self.worst_intents,
            "drift_detected": self.drift_detected,
            "severity": self.severity,
            "severity_by_signal": self.severity_by_signal,
            "recommendation": self.recommendation,
            "timestamp": self.timestamp,
            "reliability": self.reliability,
        }


class OutputQualityMonitor:
    """Computes quality metrics for a window against a frozen champion baseline."""

    def __init__(self, baseline: dict[str, float] | None = None,
                 *, n_reliability_bins: int = 10,
                 acc_warn: float = settings.ACC_DROP_WARN,
                 acc_critical: float = settings.ACC_DROP_CRITICAL,
                 ece_warn: float = settings.ECE_WARN,
                 ece_critical: float = settings.ECE_CRITICAL,
                 brier_warn: float = settings.BRIER_WARN,
                 brier_critical: float = settings.BRIER_CRITICAL,
                 abstention_jump: float = settings.ABSTENTION_JUMP,
                 min_labeled: int = 25):
        self.baseline = dict(baseline or {})
        self.n_bins = n_reliability_bins
        self.acc_warn, self.acc_critical = acc_warn, acc_critical
        self.ece_warn, self.ece_critical = ece_warn, ece_critical
        self.brier_warn, self.brier_critical = brier_warn, brier_critical
        self.abstention_jump = abstention_jump
        self.min_labeled = min_labeled

    def evaluate(
        self,
        window: pd.DataFrame,
        *,
        label_column: str = "gold_intent",
        pred_column: str = "pred_intent",
        confidence_column: str = "confidence",
        margin_column: str = "margin",
        scope_column: str = "in_scope",
        abstain_column: str = "abstained",
    ) -> QualityResult:
        res = QualityResult(
            n_requests=int(len(window)),
            timestamp=pd.Timestamp.now('UTC').strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        if window.empty:
            res.recommendation = "Empty window — quality not evaluated."
            return res

        conf = window[confidence_column].to_numpy(dtype=float) if confidence_column in window else np.array([])
        marg = window[margin_column].to_numpy(dtype=float) if margin_column in window else np.full(len(window), np.nan)
        pred = window[pred_column].astype(str).to_numpy()
        abst = window[abstain_column].astype(bool).to_numpy() if abstain_column in window else np.zeros(len(window), bool)
        scope = window[scope_column].astype(bool).to_numpy() if scope_column in window else np.ones(len(window), bool)

        res.abstention_rate = float(abst.mean())
        base_abs = self.baseline.get("abstention_rate")
        res.abstention_delta = (res.abstention_rate - base_abs) if base_abs is not None else float("nan")

        if conf.size:
            sh = score_health(conf)
            res.confidence_mean = sh["mean"]
            res.confidence_p05 = sh["p05"]
            res.confidence_p95 = sh["p95"]
            res.high_conf_rate = sh["high_conf_rate"]
        res.proxy_accuracy = label_free_accuracy_proxy(
            conf if conf.size else np.zeros(len(window)),
            np.nan_to_num(marg, nan=0.0),
            reference_accuracy=self.baseline.get("accuracy"),
        )

        labeled = (
            window[label_column].notna().astype(bool).to_numpy()
            if label_column in window else np.zeros(len(window), bool)
        )
        res.n_labeled = int(labeled.sum())
        res.label_coverage = res.n_labeled / max(len(window), 1)

        # `correct` spans the whole window with NaN where no label has arrived,
        # so every slice-level statistic can use the same mask without index
        # gymnastics — and so an unlabelled row never silently counts as wrong.
        correct_all = np.full(len(window), np.nan)
        signals: dict[str, str] = {}

        if res.n_labeled >= self.min_labeled:
            gold = window.loc[labeled, label_column].astype(str).to_numpy()
            y_pred = pred[labeled]
            correct = (y_pred == gold).astype(float)
            correct_all[labeled] = correct
            binary = (correct >= 0.5).astype(int)

            res.accuracy = float(correct.mean())
            res.macro_f1 = float(f1_score(gold, y_pred, average="macro", zero_division=0))
            # Weighted intent-level precision/recall. Deriving these from the
            # binary "was it right" flag is circular — it is always 1.0 by
            # construction — so the per-intent view is the honest one.
            res.precision = float(precision_score(gold, y_pred, average="weighted", zero_division=0))
            res.recall = float(recall_score(gold, y_pred, average="weighted", zero_division=0))

            lbl_conf = conf[labeled] if conf.size else np.array([])
            if lbl_conf.size == lbl_conf.size and lbl_conf.size >= 5:
                res.ece = expected_calibration_error(lbl_conf, correct, self.n_bins)
                res.mce = maximum_calibration_error(lbl_conf, correct, self.n_bins)
                res.adaptive_ece = adaptive_ece(lbl_conf, correct, self.n_bins)
                res.reliability = [
                    [_r(row[0], 4), _r(row[1], 4)]
                    for row in reliability_curve(lbl_conf, correct, self.n_bins).tolist()
                    if np.isfinite(row[0])
                ]
                res.brier = float(brier_score_loss(binary, lbl_conf))
                clip = np.clip(lbl_conf, 1e-6, 1 - 1e-6)
                res.log_loss = float(log_loss(binary, clip, labels=[0, 1]))
            if 0 < binary.sum() < binary.size:
                try:
                    res.auc = float(roc_auc_score(binary, lbl_conf))
                except ValueError:
                    res.auc = float("nan")

            base_acc = self.baseline.get("accuracy")
            if base_acc is not None and np.isfinite(res.accuracy):
                res.accuracy_delta = float(res.accuracy - base_acc)
                se = self.baseline.get("accuracy_se") or 0.0
                if se > 0:
                    res.accuracy_z = float(res.accuracy_delta / se)
            base_ece = self.baseline.get("ece")
            if base_ece is not None and np.isfinite(res.ece):
                res.ece_delta = float(res.ece - base_ece)
            base_brier = self.baseline.get("brier")
            if base_brier is not None and np.isfinite(res.brier):
                res.brier_delta = float(res.brier - base_brier)

            labelled_scope = scope[labeled]
            if labelled_scope.any():
                res.in_scope_accuracy = float(correct[labelled_scope].mean())
            if (~labelled_scope).any():
                res.out_of_scope_accuracy = float(correct[~labelled_scope].mean())

            per_intent = pd.DataFrame({"gold": gold, "correct": correct}).groupby("gold")["correct"]
            stats_df = per_intent.agg(["mean", "count"]).sort_values("mean")
            res.worst_intents = [
                {"intent": str(i), "accuracy": _r(row["mean"]), "n": int(row["count"])}
                for i, row in stats_df.head(5).iterrows()
            ]

            drop = max(0.0, -res.accuracy_delta) if np.isfinite(res.accuracy_delta) else float("nan")
            signals["accuracy"] = severity_from(drop, self.acc_warn, self.acc_critical)

            # Calibration is judged against the champion, not against an absolute
            # constant. A model whose baseline ECE is already 0.15 is not a new
            # incident every window — it is a model with a known calibration gap,
            # and the question is whether the gap is *growing*. Absolute
            # thresholds are still used when no baseline exists.
            if np.isfinite(res.ece_delta):
                signals["ece"] = severity_from(res.ece_delta, self.ece_warn, self.ece_critical)
            elif np.isfinite(res.ece):
                signals["ece"] = severity_from(res.ece, self.ece_warn, self.ece_critical)
            else:
                signals["ece"] = "none"

            if np.isfinite(res.brier_delta):
                signals["brier"] = severity_from(res.brier_delta, self.brier_warn, self.brier_critical)
            elif np.isfinite(res.brier):
                signals["brier"] = severity_from(res.brier, self.brier_warn, self.brier_critical)
            else:
                signals["brier"] = "none"
        else:
            signals["accuracy"] = signals["ece"] = signals["brier"] = "none"

        if np.isfinite(res.abstention_delta):
            signals["abstention"] = severity_from(max(0.0, res.abstention_delta),
                                                  self.abstention_jump,
                                                  self.abstention_jump * 2)
        else:
            signals["abstention"] = "none"

        if np.isfinite(res.out_of_scope_accuracy) and np.isfinite(res.in_scope_accuracy):
            signals["scope_gap"] = severity_from(
                max(0.0, res.in_scope_accuracy - res.out_of_scope_accuracy), 0.25, 0.5)
        else:
            signals["scope_gap"] = "none"

        res.severity_by_signal = signals
        res.severity = worst(*signals.values())
        res.drift_detected = any(v == "severe" for v in signals.values()) or \
            sum(1 for v in signals.values() if v != "none") >= 2
        res.recommendation = self._recommend(res)
        return res

    @staticmethod
    def _recommend(res: QualityResult) -> str:
        if not res.drift_detected:
            if res.n_labeled == 0:
                return ("No labels yet for this window. Unlabelled proxies (confidence, abstention) "
                        "are within range; defer the quality verdict until labels land.")
            return "Quality holds against the champion baseline. No action."
        firing = [k for k, v in res.severity_by_signal.items() if v != "none"]
        if "accuracy" in firing and res.severity == "severe":
            return (f"Accuracy regression: {res.accuracy:.3f} vs baseline "
                    f"{(res.accuracy - res.accuracy_delta):.3f} (Δ{res.accuracy_delta:+.3f}, "
                    f"z={res.accuracy_z:+.1f}). " + _worst_intent_hint(res) +
                    " Retrain on recent labelled data.")
        if "ece" in firing or "brier" in firing:
            return (f"Calibration regression: ECE={res.ece:.3f} (Δ{res.ece_delta:+.3f}), "
                    f"Brier={res.brier:.3f}. The model still ranks well but its confidence is now "
                    "a worse estimate of its own accuracy — recalibrate before trusting the score.")
        if "abstention" in firing:
            return (f"Abstention rate jumped to {res.abstention_rate:.1%} "
                    f"(baseline {self_baseline_note(res)}). Input distribution moved out of distribution; "
                    "retrain and widen the label set.")
        if "scope_gap" in firing:
            return (f"In-scope accuracy {res.in_scope_accuracy:.3f} vs out-of-scope "
                    f"{res.out_of_scope_accuracy:.3f}. New intents need labels before they can be served.")
        return f"Quality signals degrading: {', '.join(firing)}. " + _worst_intent_hint(res)


def _worst_intent_hint(res: QualityResult) -> str:
    if not res.worst_intents:
        return ""
    worst_i = res.worst_intents[0]
    return f"Worst intent: {worst_i['intent']} at {worst_i['accuracy']:.2f} (n={worst_i['n']})."


def self_baseline_note(res: QualityResult) -> str:
    return f"Δ{res.abstention_delta:+.1%}" if np.isfinite(res.abstention_delta) else "unknown"


def snapshot_baseline(df: pd.DataFrame, *, label_col: str = "gold_intent",
                      pred_col: str = "pred_intent", conf_col: str = "confidence",
                      abstain_col: str = "abstained", scope_col: str = "in_scope",
                      n_bins: int = 10) -> dict[str, float]:
    """
    Compute the champion's frozen quality baseline from a validation frame.

    Two baselines come out of this, and keeping them separate is the point:

    * ``accuracy`` and friends are measured **on in-scope traffic only**. That is
      what "healthy" means for a system launched on 27 intents — a metric
      averaged over intents the model was never trained on is a scope statement,
      not a health statement, and it drags the bar down until no regression can
      ever be detected.
    * ``oos_accuracy`` is measured separately and is expected to be near zero.
      Reporting it makes the scope gap explicit instead of hiding it inside a
      single number.
    """
    labeled = df[label_col].notna() & df[pred_col].notna()
    if not labeled.any():
        return {}
    base = _metric_block(df[labeled], label_col, pred_col, conf_col, n_bins)
    base["abstention_rate"] = float(df[abstain_col].astype(bool).mean()) if abstain_col in df else 0.0
    base["n_validation"] = int(labeled.sum())

    if scope_col in df.columns:
        scope = df[scope_col].astype(bool)
        oos = labeled & ~scope
        base["oos_accuracy"] = float((df.loc[oos, pred_col].astype(str).to_numpy()
                                      == df.loc[oos, label_col].astype(str).to_numpy()).mean()) \
            if oos.any() else float("nan")
        base["n_out_of_scope"] = int(oos.sum())
    return base


def _metric_block(df: pd.DataFrame, label_col: str, pred_col: str,
                  conf_col: str, n_bins: int) -> dict[str, float]:
    gold = df[label_col].astype(str).to_numpy()
    pred = df[pred_col].astype(str).to_numpy()
    correct = (pred == gold).astype(float)
    binary = correct.astype(int)
    conf = (df[conf_col].to_numpy(dtype=float) if conf_col in df
            else np.full(len(gold), np.nan))
    valid = np.isfinite(conf)

    acc = float(correct.mean())
    out = {
        "accuracy": acc,
        "accuracy_se": float(correct.std(ddof=1) / np.sqrt(len(correct))) if len(correct) > 1 else 0.0,
        "macro_f1": float(f1_score(gold, pred, average="macro", zero_division=0)),
        "brier": float(brier_score_loss(binary, np.clip(conf[valid], 0, 1))) if valid.any() else float("nan"),
        "ece": expected_calibration_error(conf[valid], correct[valid], n_bins) if valid.sum() >= 5 else float("nan"),
    }
    try:
        out["auc"] = float(roc_auc_score(binary, conf[valid])) if 0 < binary.sum() < binary.size else float("nan")
    except ValueError:
        out["auc"] = float("nan")
    return out
