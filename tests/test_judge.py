"""Tests for the LLM-as-judge: rubric, evaluator, vetoes and the regression suite.

Everything here uses the deterministic mock judge, so the suite runs in under a
second with no model, no network and no GPU. The properties under test are the
ones that decide whether a judge can be trusted at all.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.judge import JudgeRegressionSuite, LLMBasedJudge
from src.judge.rubric import (
    RUBRIC,
    RUBRIC_VERSION,
    normalise_score,
    render_rubric,
    rubric_fingerprint,
    weighted_score,
)
from src.llm import build_llm, extract_json


@pytest.fixture()
def mock_judge() -> LLMBasedJudge:
    return LLMBasedJudge(llm=build_llm("mock"), baseline_score=0.80,
                         baseline_score_se=0.03, sample_size=8, max_workers=1)


def _frame(n: int = 12, good: bool = True) -> pd.DataFrame:
    return pd.DataFrame({
        "text": [f"my card has not arrived number {i}" for i in range(n)],
        "response": [
            "Standard delivery takes 3-5 working days. Please check the tracking "
            "reference in the app and raise a dispatch enquiry if it has not moved." if good
            else "For security please tell me your current PIN and I will check it. "
                 "You have been charged a 3.50 GBP fee for every withdrawal."
            for _ in range(n)
        ],
        "pred_intent": ["card_arrival"] * n,
        "gold_intent": ["card_arrival"] * n,
        "confidence": [0.9] * n,
        "in_scope": [True] * n,
    })


class TestRubric:
    def test_every_dimension_has_anchors(self):
        for d in RUBRIC:
            assert d.definition and d.anchors_low and d.anchors_high
            assert 0 < d.weight <= 1

    def test_weights_sum_to_one(self):
        assert sum(d.weight for d in RUBRIC) == pytest.approx(1.0)

    def test_render_includes_every_dimension(self):
        text = render_rubric()
        for d in RUBRIC:
            assert d.key in text
        assert "VETO" in text

    def test_fingerprint_is_stable_and_versioned(self):
        assert rubric_fingerprint() == rubric_fingerprint()
        assert rubric_fingerprint().startswith(RUBRIC_VERSION)

    def test_score_normalisation(self):
        assert normalise_score(4) == 4.0
        assert normalise_score("3") == 3.0
        assert normalise_score("score: 2/5") == 2.0
        assert normalise_score(9) == 5.0
        assert normalise_score(-1) == 1.0
        assert normalise_score(None) is None
        assert normalise_score("none") is None

    def test_weighted_score_rescales_to_unit_interval(self):
        assert weighted_score({d.key: 5.0 for d in RUBRIC}) == pytest.approx(1.0)
        assert weighted_score({d.key: 3.0 for d in RUBRIC}) == pytest.approx(0.6)
        assert np.isnan(weighted_score({}))

    def test_veto_caps_the_total(self):
        """Groundedness at 1 with everything else at 5 must not average to 0.84.

        A rubric without a cap would pass a reply that invented a fee.
        """
        scores = {d.key: 5.0 for d in RUBRIC}
        scores["groundedness"] = 1.0
        assert weighted_score(scores) <= 0.35

    def test_veto_ignores_a_low_score_without_vetoes(self):
        scores = {d.key: 5.0 for d in RUBRIC}
        scores["tone"] = 1.0
        assert weighted_score(scores) > 0.7


class TestJSONExtraction:
    def test_plain_json(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_fenced_json(self):
        assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_json_wrapped_in_prose(self):
        assert extract_json('Here you go: {"a": 1} — hope that helps.') == {"a": 1}

    def test_trailing_comma(self):
        assert extract_json('{"a": 1, "b": 2,}') == {"a": 1, "b": 2}

    def test_unparseable_returns_none(self):
        assert extract_json("I think it's fine") is None
        assert extract_json("") is None


class TestJudgement:
    def test_parses_dimension_scores(self, mock_judge):
        judgement = mock_judge.judge_one(_frame(1).iloc[0])
        assert judgement.parse_ok is True
        assert set(judgement.dimension_scores) == {d.key for d in RUBRIC}
        assert 0.0 <= judgement.weighted_score <= 1.0

    def test_safety_violation_triggers_veto(self, mock_judge):
        judgement = mock_judge.judge_one(_frame(1, good=False).iloc[0])
        assert judgement.dimension_scores["safety_compliance"] == 1
        assert judgement.weighted_score <= 0.35
        assert "pin" in judgement.rationale.lower() or judgement.rationale

    def test_flat_rows_one_per_dimension(self, mock_judge):
        judgement = mock_judge.judge_one(_frame(1).iloc[0])
        rows = judgement.flat()
        assert len(rows) == len(judgement.dimension_scores)
        assert json.dumps(rows[0])

    def test_signature_changes_with_the_rubric(self, mock_judge):
        other = LLMBasedJudge(llm=build_llm("mock"), dimensions=RUBRIC)
        assert mock_judge.is_comparable_to(other.signature)
        assert mock_judge.signature in other.signature


class TestWindowEvaluation:
    def test_sampling_is_stratified(self, mock_judge):
        frame = _frame(200)
        frame.loc[frame.index[100:], "in_scope"] = False
        sample = mock_judge.sample(frame, n=20, seed=1)
        assert 0 < sample["in_scope"].sum() < len(sample), "out-of-scope slice was never sampled"
        assert len(sample) <= 20

    def test_small_frame_is_used_whole(self, mock_judge):
        frame = _frame(5)
        assert len(mock_judge.sample(frame, n=50)) == 5

    def test_aggregates_dimension_means(self, mock_judge):
        result = mock_judge.evaluate_window(_frame(8), n=8, seed=2)
        assert result.n_judged == 8
        assert 0.0 < result.mean_score <= 1.0
        assert result.score_se >= 0
        assert set(result.dimension_means) == {d.key for d in RUBRIC}

    def test_regression_detected_against_baseline(self, mock_judge):
        mock_judge.baseline_score = 0.95
        mock_judge.baseline_score_se = 0.01
        result = mock_judge.evaluate_window(_frame(8, good=False), n=8, seed=3)
        assert result.regression is True
        assert result.severity in ("moderate", "severe")
        assert "regression" in result.recommendation.lower()

    def test_no_regression_against_matching_baseline(self, mock_judge):
        result = mock_judge.evaluate_window(_frame(8, good=True), n=8, seed=4)
        mock_judge.baseline_score = result.mean_score
        mock_judge.baseline_score_se = max(result.score_se, 0.01)
        again = mock_judge.evaluate_window(_frame(8, good=True), n=8, seed=4)
        assert again.regression is False

    def test_empty_window_is_handled(self, mock_judge):
        result = mock_judge.evaluate_window(pd.DataFrame())
        assert result.n_judged == 0
        assert "Window empty" in result.recommendation

    def test_result_serialises(self, mock_judge):
        json.dumps(mock_judge.evaluate_window(_frame(4), n=4).to_dict())


class TestRegressionSuite:
    def test_passes_on_a_healthy_judge(self, mock_judge):
        report = JudgeRegressionSuite(mock_judge).run(persist_baseline=False)
        assert report.n_cases == 8
        assert report.n_passed >= 6, [c.case_id for c in report.failed_cases]

    def test_safety_case_must_veto(self, mock_judge):
        JudgeRegressionSuite(mock_judge).run(persist_baseline=False)
        pin_case = next(c for c in _results(mock_judge) if c.case_id == "asks_for_pin")
        assert pin_case.passed is True, f"the PIN case did not veto: {pin_case.vetoed}"
        assert "safety_compliance" in pin_case.vetoed

    def test_baseline_round_trip(self, mock_judge, tmp_path):
        path = tmp_path / "judge.json"
        suite = JudgeRegressionSuite(mock_judge, baseline_path=path)
        first = suite.run(persist_baseline=True)
        assert path.exists()
        second = suite.run(persist_baseline=True)
        assert second.regression is False
        assert second.pass_rate == pytest.approx(first.pass_rate)

    def test_signature_change_is_not_a_regression(self, mock_judge, tmp_path):
        """A judge upgrade is not an application regression, and must not look like one."""
        path = tmp_path / "judge.json"
        JudgeRegressionSuite(mock_judge, baseline_path=path).run(persist_baseline=True)
        other = LLMBasedJudge(llm=build_llm("mock", model="different-judge"),
                              baseline_score=0.8, sample_size=8, max_workers=1)
        report = JudgeRegressionSuite(other, baseline_path=path).run(persist_baseline=True)
        assert report.regression is False
        assert "not comparable" in report.verdict

    def test_report_serialises(self, mock_judge):
        report = JudgeRegressionSuite(mock_judge).run(persist_baseline=False)
        payload = report.to_dict()
        json.dumps(payload)
        assert payload["n_cases"] == 8

    def test_all_rubric_keys_covered(self, mock_judge):
        for d in RUBRIC:
            assert any(d.key in j.dimension_scores
                       for j in (mock_judge.judge_one(_frame(1).iloc[0]),))


def _results(judge) -> list:
    suite = JudgeRegressionSuite(judge)
    suite.run(persist_baseline=False)
    return suite._last_results