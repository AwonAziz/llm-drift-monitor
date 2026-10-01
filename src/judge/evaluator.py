"""
LLM-as-judge
------------
Scores a produced response against an anchored rubric, and — the part that
separates a monitoring system from a vibe check — *tracks whether the judge
itself is trustworthy* over time.

Four capabilities, each answering a different question:

1. **Absolute rubric scoring.** Does this answer meet our bar, right now?
   One judge call returns every dimension at once, which keeps cost linear in
   samples instead of dimensions.

2. **Pairwise comparison.** Is the candidate better or worse than the baseline
   response to the *same* input? Pairwise judgements are far more reliable
   than absolute scores in the literature, so where a baseline exists we
   prefer win-rate over absolute score.

3. **Regression tracking.** A window's mean judge score is not enough. We keep
   the EWMA control chart and its standard error, and only raise a regression
   when the drop exceeds ``JUDGE_REGRESSION_Z`` standard errors. This is what
   stops a 200-sample window from paging someone about a 0.4-point wobble.

4. **Judge validation.** Cohen's kappa against ground truth, self-consistency
   under repeats, and a hard validity gate. If the judge stops agreeing with
   human labels, every judge-sourced alert is suppressed and the judge becomes
   the incident. Teams routinely miss this; it is the difference between
   "LLM-as-judge" and "LLM-shaped astrology".

Cost control matters as much as statistical control: judging every request with
an 8B model is not affordable, so evaluation runs on a stratified sample per
window and the sample size is recorded alongside the score.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from config import settings
from src.judge.rubric import (
    CORRECTNESS_KEY,
    RUBRIC,
    RUBRIC_VERSION,
    RubricDimension,
    normalise_score,
    render_rubric,
    rubric_fingerprint,
    weighted_score,
)
from src.llm.client import BaseLLM, extract_json, resolve_auto
from src.utils.logging import get_logger
from src.utils.parallel import ordered_map
from src.utils.stats import mean_ci, stable_hash

logger = get_logger(__name__)

JUDGE_SYSTEM = (
    "You are a strict but fair evaluator of customer-support replies for a retail bank. "
    "You score only what is written in the reply, against the supplied playbook. "
    "You never assume facts that are not present. "
    "You always answer with a single JSON object and nothing else."
)

ABSOLUTE_PROMPT = """Evaluate the customer-support reply against the rubric.

RUBRIC
{rubric}

SCALE
5 = {s5}
4 = {s4}
3 = {s3}
2 = {s2}
1 = {s1}

CONTEXT
Detected intent: {intent}
Classifier confidence: {confidence:.2f}
Intent in scope when the model was launched: {in_scope}

PLAYBOOK
{playbook}

CUSTOMER MESSAGE
{message}

ASSISTANT REPLY
{reply}

Return ONLY this JSON object, with no prose before or after:
{{"scores": {{"{key1}": <1-5>, "{key2}": <1-5>, "{key3}": <1-5>, "{key4}": <1-5>, "{key5}": <1-5>}},
  "correct": true|false,
  "rationale": "<one sentence citing the specific playbook fact or violation>"}}"""

PAIRWISE_PROMPT = """Compare two candidate replies to the same customer message.

RUBRIC
{rubric}

CONTEXT
Detected intent: {intent}
PLAYBOOK
{playbook}

CUSTOMER MESSAGE
{message}

REPLY A (baseline)
{a}

REPLY B (candidate)
{b}

Judge which reply better satisfies the rubric. Prefer the baseline unless the
candidate is clearly better or clearly worse — do not reward stylistic novelty.

Return ONLY this JSON object:
{{"winner": "A" | "B" | "tie",
  "margin": 1-5,
  "rationale": "<one sentence>"}}"""


@dataclass
class Judgement:
    """One judged sample."""

    sample_hash: int
    snippet: str
    dimension_scores: dict[str, float]
    weighted_score: float
    correct: bool | None
    rationale: str
    judge_model: str
    rubric_version: str
    latency_ms: float
    parse_ok: bool = True
    verdict: str | None = None
    pairwise: dict[str, Any] | None = None
    in_scope: bool = True
    agreement_with_gold: float | None = None

    def flat(self) -> list[dict[str, Any]]:
        rows = [{
            "sample_hash": self.sample_hash, "sample_snippet": self.snippet,
            "dimension": dimension, "score": score, "rationale": self.rationale,
            "judge_model": self.judge_model, "rubric_version": self.rubric_version,
            "in_scope": int(self.in_scope), "agreement_with_gold": self.agreement_with_gold,
            "verdict": self.verdict,
            "extra": json.dumps({"weighted": self.weighted_score, "parse_ok": self.parse_ok}),
        } for dimension, score in self.dimension_scores.items()]
        return rows


@dataclass
class JudgeWindowResult:
    window_index: int = 0
    n_sampled: int = 0
    n_judged: int = 0
    n_unparseable: int = 0
    mean_score: float = float("nan")
    score_se: float = float("nan")
    ci_low: float = float("nan")
    ci_high: float = float("nan")
    dimension_means: dict[str, float] = field(default_factory=dict)
    dimension_deltas: dict[str, float] = field(default_factory=dict)
    correctness_rate: float = float("nan")
    correctness_delta: float = float("nan")
    pairwise_win_rate: float = float("nan")
    baseline_score: float = float("nan")
    score_delta: float = float("nan")
    score_z: float = float("nan")
    regression: bool = False
    severity: str = "none"
    veto_rate: float = float("nan")
    judge_cost_ms: float = 0.0
    judge_model: str = ""
    rubric_version: str = RUBRIC_VERSION
    recommendation: str = ""
    samples: list[Judgement] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_sampled": self.n_sampled, "n_judged": self.n_judged,
            "n_unparseable": self.n_unparseable,
            "mean_score": _r(self.mean_score), "score_se": _r(self.score_se),
            "ci_low": _r(self.ci_low), "ci_high": _r(self.ci_high),
            "dimension_means": {k: _r(v) for k, v in self.dimension_means.items()},
            "dimension_deltas": {k: _r(v) for k, v in self.dimension_deltas.items()},
            "correctness_rate": _r(self.correctness_rate),
            "correctness_delta": _r(self.correctness_delta),
            "pairwise_win_rate": _r(self.pairwise_win_rate),
            "baseline_score": _r(self.baseline_score),
            "score_delta": _r(self.score_delta),
            "score_z": _r(self.score_z, 2),
            "regression": self.regression,
            "severity": self.severity,
            "veto_rate": _r(self.veto_rate),
            "judge_cost_ms": _r(self.judge_cost_ms, 1),
            "judge_model": self.judge_model,
            "rubric_version": self.rubric_version,
            "recommendation": self.recommendation,
        }


def _r(v: Any, n: int = 4) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(f) else round(f, n)


class LLMBasedJudge:
    """Rubric judge with regression tracking and self-validation."""

    def __init__(self, llm: BaseLLM | None = None, *,
                 dimensions: Sequence[RubricDimension] = RUBRIC,
                 baseline_scores: dict[str, float] | None = None,
                 baseline_score: float | None = None,
                 baseline_score_se: float = 0.0,
                 score_warn: float = settings.JUDGE_SCORE_DROP_WARN,
                 score_critical: float = settings.JUDGE_SCORE_DROP_CRITICAL,
                 regression_z: float = settings.JUDGE_REGRESSION_Z,
                 sample_size: int = settings.JUDGE_SAMPLE_SIZE,
                 repeats: int = settings.JUDGE_REPEATS_FOR_SELF_CONSISTENCY,
                 max_workers: int = settings.JUDGE_MAX_WORKERS,
                 pair_with_baseline: bool = True):
        self.llm = llm or resolve_auto(settings.JUDGE_PROVIDER)
        self.dimensions = tuple(dimensions)
        self.dimension_keys = [d.key for d in self.dimensions]
        self.baseline_dimension_scores = dict(baseline_scores or {})
        self.baseline_score = baseline_score
        self.baseline_score_se = baseline_score_se
        self.score_warn = score_warn
        self.score_critical = score_critical
        self.regression_z = regression_z
        self.sample_size = sample_size
        self.repeats = max(1, repeats)
        self.pair_with_baseline = pair_with_baseline
        self.max_workers = max(1, max_workers)
        self.rubric_fingerprint = rubric_fingerprint(self.dimensions)

    # ── signature (comparability guard) ─────────────────────────────
    @property
    def signature(self) -> str:
        return f"{self.llm.provider}/{self.llm.model}@t{self.llm.temperature}#{self.rubric_fingerprint}"

    def is_comparable_to(self, other_signature: str) -> bool:
        return other_signature == self.signature

    # ── sampling ────────────────────────────────────────────────────
    def sample(self, frame: pd.DataFrame, n: int | None = None,
               seed: int = 0) -> pd.DataFrame:
        """
        Stratified sample of the window.

        Stratifying on ``in_scope`` guarantees that the out-of-scope slice is
        represented in the judged sample. Uniform sampling would quietly
        evaluate only traffic the model can plausibly handle, and the regression
        you most need to catch is the one hiding in the slice you never judged.
        """
        n = n or self.sample_size
        if frame.empty:
            return frame
        if len(frame) <= n:
            return frame
        rng = np.random.default_rng(seed)
        if "in_scope" in frame.columns:
            scope = frame["in_scope"].astype(bool)
            groups = [frame[scope], frame[~scope]]
        else:
            groups = [frame]
        weights = np.array([max(len(g), 0) for g in groups], dtype=float)
        weights = weights / weights.sum() if weights.sum() else np.ones(len(groups)) / len(groups)
        picks = []
        for group, w in zip(groups, weights):
            take = int(round(n * w))
            if take <= 0 or group.empty:
                continue
            take = min(take, len(group))
            picks.append(group.iloc[rng.choice(len(group), size=take, replace=False)])
        out = pd.concat(picks) if picks else frame
        return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # ── single judgement ────────────────────────────────────────────
    def judge_one(self, row: pd.Series, playbook: dict[str, Any] | None = None,
                  baseline_response: str | None = None) -> Judgement:
        intent = str(row.get("pred_intent", row.get("intent", "unknown")))
        confidence = float(row.get("confidence", 0.0) or 0.0)
        message = str(row.get("text", ""))
        reply = str(row.get("response", ""))
        in_scope = bool(row.get("in_scope", True))

        pb = _render_playbook(playbook)
        prompt = ABSOLUTE_PROMPT.format(
            rubric=render_rubric(self.dimensions),
            s1="Vetoed or unusable", s2="Fails the dimension", s3="Partial",
            s4="Minor weakness", s5="Fully satisfies",
            intent=intent, confidence=confidence,
            in_scope="yes" if in_scope else "NO — outside the scope the model was launched for",
            playbook=pb, message=message, reply=reply,
            key1=self.dimension_keys[0], key2=self.dimension_keys[1], key3=self.dimension_keys[2],
            key4=self.dimension_keys[3], key5=self.dimension_keys[4],
        )

        scores: dict[str, float] = {}
        correct: bool | None = None
        rationale = ""
        parse_ok = False
        latency = 0.0
        verdict = None
        pairwise = None

        t0 = time.perf_counter()
        try:
            raw = self.llm.generate(prompt, system=JUDGE_SYSTEM)
            latency = (time.perf_counter() - t0) * 1000
            payload = extract_json(raw.text)
            if payload:
                parse_ok = True
                raw_scores = payload.get("scores") or {}
                for key in self.dimension_keys:
                    scores[key] = normalise_score(raw_scores.get(key))
                scores = {k: v for k, v in scores.items() if v is not None}
                rationale = str(payload.get("rationale", ""))[:600]
                if isinstance(payload.get("correct"), bool):
                    correct = payload["correct"]
                else:
                    cs = scores.get(CORRECTNESS_KEY)
                    correct = None if cs is None else bool(cs >= 4)
                verdict = "pass" if correct else "fail"
        except Exception as exc:  # noqa: BLE001 - a judge outage must not stop monitoring
            latency = (time.perf_counter() - t0) * 1000
            logger.warning("Judge call failed: %s", exc)
            rationale = f"judge_error: {exc}"

        if self.pair_with_baseline and baseline_response and parse_ok:
            pairwise, pair_latency = self._pairwise(message, intent, pb, baseline_response, reply)
            latency += pair_latency

        return Judgement(
            sample_hash=stable_hash(message) % (10 ** 12),
            snippet=message[:200], dimension_scores=scores,
            weighted_score=weighted_score(scores, self.dimensions),
            correct=correct, rationale=rationale,
            judge_model=self.llm.model, rubric_version=RUBRIC_VERSION,
            latency_ms=latency, parse_ok=parse_ok, verdict=verdict,
            pairwise=pairwise, in_scope=in_scope,
        )

    def _pairwise(self, message: str, intent: str, playbook: str,
                  a: str, b: str) -> tuple[dict[str, Any] | None, float]:
        prompt = PAIRWISE_PROMPT.format(rubric=render_rubric(self.dimensions), intent=intent,
                                        playbook=playbook, message=message, a=a, b=b)
        t0 = time.perf_counter()
        try:
            raw = self.llm.generate(prompt, system=JUDGE_SYSTEM)
            payload = extract_json(raw.text)
            if payload:
                return {
                    "winner": str(payload.get("winner", "tie")).strip().upper(),
                    "margin": normalise_score(payload.get("margin")),
                    "rationale": str(payload.get("rationale", ""))[:400],
                }, (time.perf_counter() - t0) * 1000
        except Exception as exc:  # noqa: BLE001
            logger.debug("Pairwise comparison failed: %s", exc)
        return None, (time.perf_counter() - t0) * 1000

    # ── window evaluation ───────────────────────────────────────────
    def evaluate_window(self, frame: pd.DataFrame, window_index: int = 0,
                        n: int | None = None, seed: int = 0,
                        playbooks: dict[str, dict[str, Any]] | None = None,
                        baseline_responses: dict[int, str] | None = None) -> JudgeWindowResult:
        sample = self.sample(frame, n=n, seed=seed)
        res = JudgeWindowResult(window_index=window_index, n_sampled=len(sample),
                                judge_model=self.llm.model, rubric_version=RUBRIC_VERSION)
        if sample.empty:
            res.recommendation = "Window empty — judge not run."
            return res

        judgements: list[Judgement] = [
            j for j in
            ordered_map(
                lambda pair: self.judge_one(pair[1], playbook=(playbooks or {}).get(str(pair[1].get("pred_intent", "unknown"))),
                                            baseline_response=(baseline_responses or {}).get(pair[0])),
                list(sample.iterrows()),
                max_workers=self.max_workers,
                label="judged",
            )
            if j is not None
        ]

        res.samples = judgements
        res.n_judged = len(judgements)
        res.judge_cost_ms = sum(j.latency_ms for j in judgements)
        self._aggregate(res, judgements)
        return res

    def _aggregate(self, res: JudgeWindowResult, judgements: list[Judgement]) -> None:
        finite = [j.weighted_score for j in judgements if np.isfinite(j.weighted_score)]
        res.n_unparseable = sum(1 for j in judgements if not j.parse_ok)
        if finite:
            mean, half, se = mean_ci(finite)
            res.mean_score, res.score_se = mean, se
            res.ci_low, res.ci_high = mean - half, mean + half
            for key in self.dimension_keys:
                vals = [j.dimension_scores.get(key) for j in judgements]
                vals = [v for v in vals if v is not None]
                if vals:
                    res.dimension_means[key] = float(np.mean(vals))
                    base = self.baseline_dimension_scores.get(key)
                    res.dimension_deltas[key] = float(np.mean(vals) - base) if base is not None else float("nan")

        correctness = [1.0 if j.correct else 0.0 for j in judgements if j.correct is not None]
        if correctness:
            res.correctness_rate = float(np.mean(correctness))

        winners = [j.pairwise.get("winner") for j in judgements if j.pairwise]
        if winners:
            wins = sum(1 for w in winners if w == "B")
            res.pairwise_win_rate = (wins + 0.5 * sum(1 for w in winners if w == "tie")) / len(winners)

        veto_keys = tuple(d.key for d in self.dimensions if d.vetoes)
        vetoes = [j for j in judgements
                  if any((j.dimension_scores.get(k) or 5) <= 1 for k in veto_keys)]
        res.veto_rate = len(vetoes) / max(len(judgements), 1)

        # ── regression test against the frozen baseline ────────────
        if self.baseline_score is not None and np.isfinite(res.mean_score):
            res.baseline_score = float(self.baseline_score)
            res.score_delta = float(res.mean_score - self.baseline_score)
            se = max(self.baseline_score_se, res.score_se, 1e-3)
            res.score_z = float(res.score_delta / se)
            drop = -res.score_delta
            res.severity = _severity(drop, self.score_warn, self.score_critical)
            res.regression = bool(res.severity != "none" and abs(res.score_z) >= self.regression_z) \
                or res.severity == "severe"
        if not res.regression and res.severity == "none":
            res.recommendation = f"Judge score {res.mean_score:.3f} vs baseline {self.baseline_score if self.baseline_score is not None else float('nan'):.3f}. Stable." \
                if self.baseline_score is not None else f"Judge mean {res.mean_score:.3f}."
        else:
            res.recommendation = self._regression_text(res)

    def _regression_text(self, res: JudgeWindowResult) -> str:
        worst_dims = sorted(
            ((k, v) for k, v in res.dimension_deltas.items() if np.isfinite(v)),
            key=lambda kv: kv[1],
        )[:2]
        dims = ", ".join(f"{k} {v:+.3f}" for k, v in worst_dims) or "no dimension breakdown"
        return (
            f"Judge regression: mean {res.mean_score:.3f} vs baseline {res.baseline_score:.3f} "
            f"(Δ{res.score_delta:+.3f}, z={res.score_z:+.1f}, n={res.n_judged}). Biggest movers: {dims}. "
            f"Pairwise win-rate vs baseline: {res.pairwise_win_rate:.2f}. "
            + ("Re-run the offline judge regression suite before touching the model."
               if res.severity == "severe" else "Review the largest-moved dimension in the dashboard.")
        )

    # ── self-validation ────────────────────────────────────────────
    def self_consistency(self, frame: pd.DataFrame, n: int = 20, seed: int = 5,
                         temperature: float = 0.7) -> dict[str, float]:
        """
        Re-judge the same samples at non-zero temperature and measure agreement.

        A judge whose own score moves 0.4 points when you ask it twice is not a
        regression detector; it is a random number generator with good manners.
        """
        sample = self.sample(frame, n=n, seed=seed)
        if sample.empty:
            return {"n": 0, "mean_abs_diff": float("nan"), "exact_agreement": float("nan"),
                    "dimension_std": float("nan")}
        original_temp = self.llm.temperature
        deltas: list[float] = []
        exact = 0
        dim_stds: list[float] = []
        try:
            self.llm.temperature = original_temp
            first = [self.judge_one(row) for _, row in sample.iterrows()]
            self.llm.temperature = temperature
            second = [self.judge_one(row) for _, row in sample.iterrows()]
        finally:
            self.llm.temperature = original_temp

        for a, b in zip(first, second):
            if np.isfinite(a.weighted_score) and np.isfinite(b.weighted_score):
                deltas.append(abs(a.weighted_score - b.weighted_score))
            if a.dimension_scores and b.dimension_scores:
                shared = set(a.dimension_scores) & set(b.dimension_scores)
                if shared:
                    exact += sum(1 for k in shared
                                 if abs(a.dimension_scores[k] - b.dimension_scores[k]) < 1e-9)
                    dim_stds.extend(abs(a.dimension_scores[k] - b.dimension_scores[k]) for k in shared)
        n_comp = sum(len(set(a.dimension_scores) & set(b.dimension_scores))
                     for a, b in zip(first, second))
        return {
            "n": float(len(sample)),
            "mean_abs_diff": float(np.mean(deltas)) if deltas else float("nan"),
            "exact_agreement": (exact / n_comp) if n_comp else float("nan"),
            "dimension_std": float(np.std(dim_stds, ddof=1)) if len(dim_stds) > 1 else float("nan"),
        }

    def validate_against_ground_truth(self, frame: pd.DataFrame) -> dict[str, float]:
        """
        Cohen's kappa between the judge's binary verdict and the real answer.

        The gold label here is *model correctness*, which we know for labelled
        traffic. If the judge cannot recognise a correct answer from an incorrect
        one, none of its absolute scores mean anything — and we say so loudly
        instead of quietly alerting on noise.
        """
        labeled = frame[frame.get("gold_intent").notna()] if "gold_intent" in frame.columns else frame.iloc[0:0]
        if labeled.empty:
            return {"kappa": float("nan"), "n": 0, "agreement": float("nan")}
        subset = labeled.head(max(settings.JUDGE_SAMPLE_SIZE, 40))
        judgements = [self.judge_one(row) for _, row in subset.iterrows()]
        usable = [j for j in judgements if j.correct is not None and not j.pairwise]
        if len(usable) < 8:
            return {"kappa": float("nan"), "n": float(len(usable)), "agreement": float("nan")}

        from sklearn.metrics import cohen_kappa_score

        gold = (subset.iloc[: len(usable)]["pred_intent"].astype(str).to_numpy()
                == subset.iloc[: len(usable)]["gold_intent"].astype(str).to_numpy()).astype(int)
        pred = np.array([1 if j.correct else 0 for j in usable])
        kappa = float(cohen_kappa_score(gold, pred))
        agreement = float(np.mean(gold == pred))
        logger.info("Judge validation: kappa=%.3f agreement=%.3f on n=%d", kappa, agreement, len(usable))
        return {"kappa": kappa, "n": float(len(usable)), "agreement": agreement,
                "judge_positive_rate": float(np.mean(pred)), "gold_positive_rate": float(np.mean(gold))}


def _render_playbook(playbook: dict[str, Any] | None) -> str:
    if not playbook:
        return "No playbook available (out of scope). The reply must not assert specifics."
    steps = "\n".join(f"- {s}" for s in playbook.get("steps", []))
    return (f"Summary: {playbook.get('summary', '')}\n"
            f"Expected resolution: {playbook.get('eta', '')}\nSteps:\n{steps}")


def _severity(drop: float, moderate: float, severe: float) -> str:
    if drop >= severe:
        return "severe"
    if drop >= moderate:
        return "moderate"
    return "none"