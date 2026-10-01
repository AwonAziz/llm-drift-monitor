"""Tests for tabular drift, output quality and calibration metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.drift.tabular import TabularDriftDetector
from src.quality import (
    OutputQualityMonitor,
    adaptive_ece,
    expected_calibration_error,
    maximum_calibration_error,
    score_health,
    severity_from,
    snapshot_baseline,
)


class TestPSI:
    def test_psi_is_non_negative(self, tabular_detector, tabular_frame):
        result = tabular_detector.detect(tabular_frame.sample(200, random_state=1))
        for col in result.columns:
            assert not np.isnan(col.psi) or col.kind == "categorical"

    def test_same_distribution_does_not_fire(self, tabular_detector, tabular_frame):
        result = tabular_detector.detect(tabular_frame.sample(300, random_state=2))
        assert result.drift_detected is False, result.drifted_columns

    def test_mean_shift_is_detected(self, tabular_detector, tabular_frame):
        shifted = tabular_frame.sample(300, random_state=3).copy()
        shifted["text_length"] += 40
        result = tabular_detector.detect(shifted)
        assert "text_length" in result.drifted_columns
        assert result.severity in ("moderate", "severe")

    def test_scale_change_is_detected(self, tabular_detector, tabular_frame):
        stretched = tabular_frame.sample(300, random_state=4).copy()
        stretched["digit_ratio"] = np.clip(stretched["digit_ratio"] * 12, 0, 1)
        assert "digit_ratio" in tabular_detector.detect(stretched).drifted_columns

    def test_categorical_drift_is_detected(self, tabular_detector, tabular_frame):
        skewed = tabular_frame.copy()
        skewed["emoji_flag"] = np.where(skewed.index % 4 == 0, 1.0, 0.0)
        result = tabular_detector.detect(skewed)
        assert "emoji_flag" in result.drifted_columns

    def test_low_variance_columns_are_marked_not_flagged(self, rng):
        frame = pd.DataFrame({
            "constant": np.ones(400),
            "normal": rng.normal(size=400),
        })
        detector = TabularDriftDetector(frame, min_rows=40)
        result = detector.detect(frame.sample(200, random_state=1))
        constant = next(c for c in result.columns if c.column == "constant")
        assert constant.low_variance is True
        assert constant.drift_detected is False

    def test_empty_window_is_handled(self, tabular_detector):
        result = tabular_detector.detect(pd.DataFrame())
        assert result.drift_detected is False
        assert "No current window" in result.recommendation

    def test_report_serialises(self, tabular_detector, tabular_frame):
        import json

        json.dumps(tabular_detector.detect(tabular_frame).to_dict())


class TestCalibrationMetrics:
    def test_ece_zero_when_perfectly_calibrated(self):
        """Confidence drawn uniformly with accuracy == confidence => ECE ~ 0.

        Built bin by bin rather than from a step function: equal-width binning of a
        two-point distribution puts every 0.9 in the top bin, whose *mean*
        confidence is still 0.9, so a step function reports a spurious gap.
        """
        rng = np.random.default_rng(0)
        conf = rng.uniform(0.02, 0.98, 4000)
        correct = (rng.random(4000) < conf).astype(float)
        assert expected_calibration_error(conf, correct, n_bins=10) < 0.03

    def test_ece_penalises_a_systematic_gap(self):
        conf = np.array([0.1] * 100 + [0.9] * 100)
        correct = np.array([0.0] * 100 + [1.0] * 100)
        assert expected_calibration_error(conf, correct, n_bins=10) > 0.09

    def test_ece_large_when_maximally_overconfident(self):
        conf = np.full(200, 0.99)
        correct = np.concatenate([np.ones(100), np.zeros(100)])
        assert expected_calibration_error(conf, correct) > 0.45

    def test_mce_at_least_ece_bound(self):
        conf = np.full(300, 0.95)
        correct = np.concatenate([np.ones(100), np.zeros(200)])
        assert maximum_calibration_error(conf, correct) >= expected_calibration_error(conf, correct)

    def test_adaptive_ece_handles_piled_up_scores(self):
        conf = np.concatenate([np.full(150, 0.95), np.linspace(0, 1, 150)])
        correct = (np.random.default_rng(0).random(300) < conf).astype(float)
        value = adaptive_ece(conf, correct, n_bins=10)
        assert 0.0 <= value <= 1.0

    def test_empty_inputs_are_nan(self):
        assert np.isnan(expected_calibration_error([], []))
        assert np.isnan(maximum_calibration_error([], []))

    def test_score_health_shape(self):
        health = score_health(np.array([0.2, 0.5, 0.9]))
        assert health["mean"] == pytest.approx(0.5333, abs=1e-3)
        assert 0.0 <= health["mean_entropy"] <= 1.0


class TestSeverityHelper:
    def test_larger_magnitude_is_worse(self):
        assert severity_from(0.0, 0.1, 0.2) == "none"
        assert severity_from(0.15, 0.1, 0.2) == "moderate"
        assert severity_from(0.3, 0.1, 0.2) == "severe"

    def test_nan_is_none(self):
        assert severity_from(float("nan"), 0.1, 0.2) == "none"


def _window(n: int = 200, accuracy: float = 0.9, seed: int = 0,
            in_scope_rate: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    gold = rng.choice(["a", "b", "c", "d"], n)
    correct = rng.random(n) < accuracy
    pred = np.where(correct, gold, rng.choice(["a", "b", "c", "d"], n))
    return pd.DataFrame({
        "text": [f"t{i}" for i in range(n)],
        "gold_intent": gold,
        "pred_intent": pred,
        "confidence": np.clip(rng.beta(6, 2, n), 0.05, 0.99),
        "margin": rng.uniform(0, 1, n),
        "abstained": rng.random(n) < 0.05,
        "in_scope": rng.random(n) < in_scope_rate,
    })


class TestOutputQualityMonitor:
    def test_accuracy_is_the_fraction_correct(self):
        """Regression test: a circular binary encoding made accuracy always 1.0."""
        frame = _window(n=400, accuracy=0.7, seed=2)
        monitor = OutputQualityMonitor(snapshot_baseline(frame))
        result = monitor.evaluate(frame)
        observed = float((frame["pred_intent"] == frame["gold_intent"]).mean())
        assert result.accuracy == pytest.approx(observed, abs=1e-6)
        assert 0.5 < result.accuracy < 0.9

    def test_no_labels_means_no_verdict(self):
        frame = _window(n=200)
        frame["gold_intent"] = None
        result = OutputQualityMonitor().evaluate(frame)
        assert result.n_labeled == 0
        assert result.severity == "none"
        assert "No labels yet" in result.recommendation

    def test_too_few_labels_defers(self):
        frame = _window(n=200)
        frame.loc[frame.index[100:], "gold_intent"] = None
        result = OutputQualityMonitor(min_labeled=50).evaluate(frame)
        assert result.n_labeled == 100
        assert result.accuracy == pytest.approx(
            float((frame.loc[frame.index[:100], "pred_intent"]
                   == frame.loc[frame.index[:100], "gold_intent"]).mean()))

    def test_regression_is_flagged_against_baseline(self):
        baseline = snapshot_baseline(_window(n=600, accuracy=0.95, seed=1))
        monitor = OutputQualityMonitor(baseline)
        result = monitor.evaluate(_window(n=400, accuracy=0.5, seed=9))
        assert result.severity in ("moderate", "severe")
        assert result.drift_detected is True
        assert result.accuracy_delta < -0.2
        assert "Accuracy regression" in result.recommendation or "Quality signals" in result.recommendation

    def test_improvement_is_not_a_regression(self):
        baseline = snapshot_baseline(_window(n=600, accuracy=0.7, seed=1))
        result = OutputQualityMonitor(baseline).evaluate(_window(n=400, accuracy=0.95, seed=3))
        assert result.accuracy_delta > 0
        assert result.severity_by_signal["accuracy"] == "none"

    def test_scope_gap_is_measured(self):
        frame = _window(n=400, accuracy=0.9, seed=4, in_scope_rate=0.5)
        # Out-of-scope rows are wrong, in-scope rows mostly right.
        frame.loc[~frame["in_scope"], "pred_intent"] = "zzz"
        result = OutputQualityMonitor().evaluate(frame)
        assert result.out_of_scope_accuracy == pytest.approx(0.0)
        assert result.in_scope_accuracy is not None
        assert result.severity_by_signal["scope_gap"] in ("moderate", "severe")

    def test_worst_intents_ranked(self):
        frame = _window(n=400, seed=6)
        frame.loc[frame["gold_intent"] == "d", "pred_intent"] = "a"
        result = OutputQualityMonitor().evaluate(frame)
        assert result.worst_intents
        assert result.worst_intents[-1]["accuracy"] >= result.worst_intents[0]["accuracy"]

    def test_result_serialises(self):
        import json

        frame = _window(n=200)
        json.dumps(OutputQualityMonitor().evaluate(frame).to_dict())


class TestBaselineSnapshot:
    def test_separates_in_scope_and_out_of_scope(self):
        """The baseline is built on in-scope traffic, as build_artifacts does.

        A single number averaged over intents the model was never trained on is
        a scope statement, not a health statement — so the caller filters, and
        the out-of-scope rate is reported alongside it.
        """
        frame = _window(n=400, accuracy=0.9, in_scope_rate=0.7, seed=8)
        frame.loc[~frame["in_scope"], "pred_intent"] = "zzz"
        baseline = snapshot_baseline(frame[frame["in_scope"]])
        assert baseline["accuracy"] > 0.85
        oos = frame[~frame["in_scope"]]
        full = snapshot_baseline(frame)
        assert full["oos_accuracy"] == pytest.approx(0.0)
        assert full["n_out_of_scope"] == len(oos)

    def test_includes_standard_error(self):
        baseline = snapshot_baseline(_window(n=300, seed=10))
        assert 0.0 < baseline["accuracy_se"] < 0.1

    def test_unlabelled_returns_empty(self):
        frame = _window(n=100)
        frame["gold_intent"] = None
        assert snapshot_baseline(frame) == {}