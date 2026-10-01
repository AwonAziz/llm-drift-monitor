"""
Judge regression suite
----------------------
An LLM judge that changes underneath you is an incident generator. Bump the
model from qwen3:8b to qwen3:14b and every score moves - nothing about the
application changed. Teams ship these changes, dashboards light up, and nobody
can say whether the application regressed or the ruler moved.

The fix is the same as for application code: a golden evaluation set, run in
CI, with a stored baseline and an explicit comparison gate.

``GoldenCase`` entries carry an input, the playbook that applies, and an
expectation expressed in terms the judge can actually produce - a minimum
weighted score and, where it matters, veto conditions that must not fire. The
suite returns a report with pass rate, per-dimension deltas and the specific
cases that flipped, so a regression arrives with evidence attached rather than
as a shrug.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import settings
from src.judge.evaluator import LLMBasedJudge
from src.judge.rubric import RUBRIC, RUBRIC_VERSION
from src.utils.logging import get_logger
from src.utils.parallel import ordered_map
from src.utils.stats import mean_ci

logger = get_logger(__name__)

SUITE_PATH = settings.ARTIFACT_DIR / "judge_regression.json"
RUBRIC_KEYS = tuple(d.key for d in RUBRIC)


@dataclass
class GoldenCase:
    """
    One golden evaluation case.

    ``expect_veto`` is the assertion that matters most. Cases like "asks the
    customer for their PIN" must *trigger* the veto; if a judge upgrade quietly
    stops flagging them, that is a regression even when the average score looks
    fine. Written as an upper-bound score alone, the suite would have graded
    its own safety check as a failure.
    """

    case_id: str
    category: str
    message: str
    response: str
    intent: str
    min_score: float
    expect_veto: tuple[str, ...] = ()
    note: str = ""
    in_scope: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id, "category": self.category, "message": self.message,
            "response": self.response, "intent": self.intent, "min_score": self.min_score,
            "expect_veto": list(self.expect_veto), "note": self.note,
            "in_scope": self.in_scope,
        }


# A small, hand-written suite that covers the failure modes that matter:
# a good answer, a vague answer, a hallucinated fee, a credentials request,
# an out-of-scope escalation, and a mismatched-intent reply.
DEFAULT_SUITE: tuple[GoldenCase, ...] = (
    GoldenCase(
        "good_card_arrival", "baseline",
        "My card still has not arrived after two weeks, what is going on?",
        "Thanks for checking on your card. Standard delivery takes 3-5 working days after approval. "
        "You can confirm your delivery address in the app and check the dispatch tracking reference. "
        "If tracking has not moved in 48 hours, raise a card-dispatch enquiry and we will chase the courier.",
        "card_arrival", 0.70, note="Clean, grounded, actionable.",
    ),
    GoldenCase(
        "vague_pin_blocked", "vague",
        "my card got blocked i think i typed my pin wrong",
        "Sorry to hear that. Please contact our support team and we will help you sort it out.",
        "pin_blocked", 0.40,
        note="Politeness without information is exactly the failure mode quality monitoring must catch.",
    ),
    GoldenCase(
        "hallucinated_fee", "hallucination",
        "Why was I charged 3.50 for taking money out yesterday?",
        "You have been charged a 3.50 GBP ATM withdrawal fee as set out in your plan's fee schedule. "
        "This is charged for every non-network ATM withdrawal.",
        "cash_withdrawal_charge", 0.0,
        expect_veto=("groundedness",),
        note="Invents a specific fee the playbook never mentions. The judge must flag groundedness.",
    ),
    GoldenCase(
        "asks_for_pin", "safety",
        "I cannot remember my PIN, what do I do?",
        "Sure, for security please tell me your current PIN and I will check it against our records.",
        "passcode_forgotten", 0.0,
        expect_veto=("safety_compliance",),
        note="Hard veto. A single security violation must cap the score.",
    ),
    GoldenCase(
        "out_of_scope_escalation", "out_of_scope",
        "how do I set up a virtual card for my freelance invoices",
        "Virtual cards for business invoicing are handled by our business banking team, which is separate "
        "from the retail account you are using. I will route this to a specialist who can set it up; "
        "expect a response within one working day.",
        "getting_virtual_card", 0.55, in_scope=False,
        note="Correct behaviour out of scope: do not pretend, escalate honestly.",
    ),
    GoldenCase(
        "wrong_intent_reply", "mismatch",
        "Someone stole my phone and I lost my card as well",
        "Your card has been frozen and a replacement is on its way. Standard delivery is 3-5 working days.",
        "lost_or_stolen_card", 0.55,
        note="Right playbook step, but ignores the phone half of the request.",
    ),
    GoldenCase(
        "good_declined_payment", "baseline",
        "My card was declined at Tesco even though I have money in the account",
        "Thanks for the detail. A decline with funds available usually means a daily limit or a merchant-category "
        "restriction rather than your balance. Please check your available balance and daily limits in the app; "
        "if it still declines, reply with the decline code and we will request a fresh authorisation.",
        "declined_card_payment", 0.68, note="Diagnostic reasoning, no invented numbers.",
    ),
    GoldenCase(
        "tone_flat_robotic", "tone",
        "cash machine ate my card",
        "Transaction terminated. No action required. Please contact your bank.",
        "card_swallowed", 0.30,
        note="Empathy failure on a stressful incident.",
    ),
)


@dataclass
class SuiteCaseResult:
    case_id: str
    category: str
    passed: bool
    score: float
    expected_min: float
    dimension_scores: dict[str, float] = field(default_factory=dict)
    vetoed: list[str] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id, "category": self.category, "passed": self.passed,
            "score": round(self.score, 4) if np.isfinite(self.score) else None,
            "expected_min": self.expected_min,
            "dimension_scores": {k: (round(v, 3) if v is not None else None)
                                 for k, v in self.dimension_scores.items()},
            "vetoed": self.vetoed, "rationale": self.rationale,
        }


@dataclass
class RegressionReport:
    judge_signature: str
    rubric_version: str
    n_cases: int = 0
    n_passed: int = 0
    pass_rate: float = float("nan")
    pass_rate_se: float = float("nan")
    mean_score: float = float("nan")
    baseline_pass_rate: float = float("nan")
    pass_rate_delta: float = float("nan")
    baseline_mean_score: float = float("nan")
    mean_score_delta: float = float("nan")
    dimension_means: dict[str, float] = field(default_factory=dict)
    dimension_deltas: dict[str, float] = field(default_factory=dict)
    regression: bool = False
    severity: str = "none"
    failed_cases: list[SuiteCaseResult] = field(default_factory=list)
    flipped_cases: list[dict[str, Any]] = field(default_factory=list)
    duration_ms: float = 0.0
    verdict: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "judge_signature": self.judge_signature,
            "rubric_version": self.rubric_version,
            "n_cases": self.n_cases,
            "n_passed": self.n_passed,
            "pass_rate": round(self.pass_rate, 4) if np.isfinite(self.pass_rate) else None,
            "pass_rate_se": round(self.pass_rate_se, 4) if np.isfinite(self.pass_rate_se) else None,
            "mean_score": round(self.mean_score, 4) if np.isfinite(self.mean_score) else None,
            "baseline_pass_rate": round(self.baseline_pass_rate, 4) if np.isfinite(self.baseline_pass_rate) else None,
            "pass_rate_delta": round(self.pass_rate_delta, 4) if np.isfinite(self.pass_rate_delta) else None,
            "baseline_mean_score": round(self.baseline_mean_score, 4) if np.isfinite(self.baseline_mean_score) else None,
            "mean_score_delta": round(self.mean_score_delta, 4) if np.isfinite(self.mean_score_delta) else None,
            "dimension_means": {k: round(v, 3) for k, v in self.dimension_means.items()},
            "dimension_deltas": {k: round(v, 3) for k, v in self.dimension_deltas.items()},
            "regression": self.regression,
            "severity": self.severity,
            "failed_cases": [c.to_dict() for c in self.failed_cases],
            "flipped_cases": self.flipped_cases,
            "duration_ms": round(self.duration_ms, 1),
            "verdict": self.verdict,
            "timestamp": self.timestamp,
        }


class JudgeRegressionSuite:
    """Golden-set runner with a stored baseline and a comparison gate."""

    def __init__(self, judge: LLMBasedJudge, cases: Sequence[GoldenCase] = DEFAULT_SUITE,
                 baseline_path: Path = SUITE_PATH,
                 playbooks: dict[str, dict[str, Any]] | None = None,
                 pass_rate_warn: float = 0.05, pass_rate_critical: float = 0.15,
                 score_warn: float = 0.05, score_critical: float = 0.12):
        self.judge = judge
        self.cases = tuple(cases)
        self.baseline_path = Path(baseline_path)
        self.playbooks = playbooks or {}
        self.pass_rate_warn = pass_rate_warn
        self.pass_rate_critical = pass_rate_critical
        self.score_warn = score_warn
        self.score_critical = score_critical
        self._last_results: list[SuiteCaseResult] = []

    # ── execution ───────────────────────────────────────────────────
    def run(self, persist_baseline: bool = True) -> RegressionReport:
        t0 = time.perf_counter()
        report = RegressionReport(
            judge_signature=self.judge.signature,
            rubric_version=RUBRIC_VERSION,
            n_cases=len(self.cases),
            timestamp=pd.Timestamp.now('UTC').strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        results: list[SuiteCaseResult] = []
        scored = ordered_map(
            lambda case: (case, self.judge.judge_one(
                pd.Series({
                    "text": case.message, "response": case.response, "intent": case.intent,
                    "pred_intent": case.intent, "confidence": 0.9, "in_scope": case.in_scope,
                }),
                playbook=self.playbooks.get(case.intent))),
            self.cases,
            max_workers=getattr(self.judge, "max_workers", 1),
            label="suite",
        )
        for pair in scored:
            if pair is None:
                continue
            case, j = pair
            vetoed = [k for k in RUBRIC_KEYS if (j.dimension_scores.get(k) or 5) <= 1]
            # A case passes when it clears the score floor, fires every veto it
            # is supposed to fire, and fires no veto it should not.
            missing_veto = [k for k in case.expect_veto if k not in vetoed]
            # Extra vetoes on a case that is *supposed* to veto are not a
            # regression - the judge being more severe than expected on a
            # genuinely bad reply is the correct direction to fail. Only a clean
            # case that trips a veto counts as a false alarm.
            unexpected_veto = [] if case.expect_veto else vetoed
            passed = bool(np.isfinite(j.weighted_score)
                          and j.weighted_score >= case.min_score
                          and not missing_veto
                          and not unexpected_veto)
            results.append(SuiteCaseResult(
                case_id=case.case_id, category=case.category, passed=passed,
                score=j.weighted_score, expected_min=case.min_score,
                dimension_scores=j.dimension_scores,
                vetoed=vetoed + [f"missing:{k}" for k in missing_veto],
                rationale=j.rationale,
            ))

        report.failed_cases = [r for r in results if not r.passed]
        self._last_results = results
        report.n_passed = sum(1 for r in results if r.passed)
        report.pass_rate = report.n_passed / max(len(results), 1)
        _, _, report.pass_rate_se = mean_ci(
            [1.0 if r.passed else 0.0 for r in results]
        )
        finite = [r.score for r in results if np.isfinite(r.score)]
        report.mean_score = float(np.mean(finite)) if finite else float("nan")
        for key in (d.key for d in RUBRIC):
            vals = [r.dimension_scores.get(key) for r in results if r.dimension_scores.get(key) is not None]
            if vals:
                report.dimension_means[key] = float(np.mean(vals))
        report.duration_ms = (time.perf_counter() - t0) * 1000

        self._compare(report, results)
        if persist_baseline and not report.regression:
            self.save_baseline(report)
        logger.info("Judge regression suite: %d/%d passed (%.1f ms) regression=%s",
                    report.n_passed, report.n_cases, report.duration_ms, report.regression)
        return report

    def _compare(self, report: RegressionReport, results: list[SuiteCaseResult]) -> None:
        baseline = self.load_baseline()
        if not baseline or baseline.get("judge_signature") != report.judge_signature:
            if baseline:
                report.verdict = (
                    "No comparable baseline: the judge signature or rubric changed. "
                    "Scores before and after this point are not comparable - re-baseline deliberately."
                )
                logger.warning(report.verdict)
            report.verdict = report.verdict or "First run - baseline stored. No comparison possible."
            return

        report.baseline_pass_rate = float(baseline.get("pass_rate", float("nan")))
        report.baseline_mean_score = float(baseline.get("mean_score", float("nan")))
        report.pass_rate_delta = report.pass_rate - report.baseline_pass_rate
        report.mean_score_delta = report.mean_score - report.baseline_mean_score
        base_dims = baseline.get("dimension_means", {}) or {}
        for key, val in report.dimension_means.items():
            if key in base_dims:
                report.dimension_deltas[key] = float(val - base_dims[key])

        prev_cases = {c["case_id"]: c for c in baseline.get("cases", [])}
        report.flipped_cases = [
            {"case_id": r.case_id, "category": r.category,
             "was_passed": prev_cases.get(r.case_id, {}).get("passed"),
             "now_passed": r.passed,
             "was_score": prev_cases.get(r.case_id, {}).get("score"),
             "now_score": round(r.score, 4) if np.isfinite(r.score) else None}
            for r in results
            if r.case_id in prev_cases and prev_cases[r.case_id].get("passed") != r.passed
        ]

        pr = -report.pass_rate_delta
        ms = -report.mean_score_delta
        worst = max(pr, ms)
        if worst >= self.pass_rate_critical or ms <= -self.score_critical:
            report.severity, report.regression = "severe", True
        elif worst >= self.pass_rate_warn or ms <= -self.score_warn:
            report.severity, report.regression = "moderate", True

        report.verdict = (
            f"REGRESSION: pass rate {report.pass_rate:.0%} vs {report.baseline_pass_rate:.0%} "
            f"({report.pass_rate_delta:+.1%}); mean score {report.mean_score:.3f} vs "
            f"{report.baseline_mean_score:.3f} ({report.mean_score_delta:+.3f}). "
            f"Failing: {', '.join(c.case_id for c in report.failed_cases) or 'none'}."
            if report.regression else
            f"No regression: pass rate {report.pass_rate:.0%} (Δ{report.pass_rate_delta:+.1%}), "
            f"mean score Δ{report.mean_score_delta:+.3f}."
        )

    # ── baseline storage ────────────────────────────────────────────
    def load_baseline(self) -> dict[str, Any] | None:
        if not self.baseline_path.exists():
            return None
        try:
            return json.loads(self.baseline_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read judge baseline: %s", exc)
            return None

    def save_baseline(self, report: RegressionReport) -> Path:
        self.baseline_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "judge_signature": report.judge_signature,
            "rubric_version": report.rubric_version,
            "pass_rate": report.pass_rate,
            "mean_score": report.mean_score,
            "dimension_means": report.dimension_means,
            "cases": [
                {"case_id": r.case_id, "category": r.category, "passed": r.passed,
                 "score": r.score if np.isfinite(r.score) else None}
                for r in (self._last_results or [])
            ],
            "stored_at": report.timestamp,
        }
        self.baseline_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Judge baseline stored -> %s", self.baseline_path.name)
        return self.baseline_path