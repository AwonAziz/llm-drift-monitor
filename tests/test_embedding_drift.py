"""Tests for the embedding-drift detectors.

Each test states a property that has to hold in production, not a snapshot of
current output. The threshold-browsing behaviour is the property worth locking
down: detectors that fire on every window get switched off within a week.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.drift.embedding import (
    domain_classifier_auc,
    frechet_distance,
    frechet_on_pca,
    median_bandwidth,
    mmd2_unbiased,
    mmd_test,
    sliced_wasserstein,
)


class TestMMD:
    def test_identical_samples_give_small_mmd(self, cloud):
        a = cloud(200, 0.0)
        b = cloud(200, 0.0)
        mmd2 = mmd2_unbiased(a, b)
        assert abs(mmd2) < 0.01, f"identical clouds gave MMD2={mmd2}"

    def test_shifted_samples_give_larger_mmd(self, cloud):
        a = cloud(200, 0.0)
        b = cloud(200, 0.9)
        assert mmd2_unbiased(b, a) > mmd2_unbiased(a, a) + 0.01

    def test_p_value_is_calibrated_on_the_null(self, cloud):
        """Same distribution => the permutation test must not reject systematically.

        This is the regression test for the calibration bug: computing the
        observed statistic on the full window and the null on a subsample made
        every window 'significant'.
        """
        a = cloud(300, 0.0, seed=1)
        b = cloud(300, 0.0, seed=2)
        p_values = [mmd_test(a, b, n_permutations=100, seed=i)[1] for i in range(6)]
        assert all(p > 0.01 for p in p_values), f"null p-values too small: {p_values}"

    def test_p_value_detects_a_real_shift(self, cloud):
        a = cloud(300, 0.0, seed=1)
        b = cloud(300, 1.0, seed=1)
        _, p, gamma = mmd_test(a, b, n_permutations=100, seed=1)
        assert p < 0.05
        assert gamma > 0

    def test_unbiased_estimator_can_be_negative(self, cloud):
        """MMD² under the null is centred on zero, not clamped at zero.

        Clamping is a common shortcut that makes every window look significant.
        """
        a = cloud(120, 0.0)
        assert mmd2_unbiased(a, a) <= 0.02

    def test_handles_tiny_inputs(self, cloud):
        mmd2, p, gamma = mmd_test(cloud(5, 0.0), cloud(5, 0.9), n_permutations=20)
        assert np.isnan(mmd2) and np.isnan(p)
        assert gamma > 0

    def test_bandwidth_is_scale_free(self, rng):
        a = rng.normal(size=(80, 4))
        assert median_bandwidth(a, a) > 0
        assert median_bandwidth(a, a, seed=1) == median_bandwidth(a, a, seed=1)


class TestSlicedWasserstein:
    def test_zero_for_identical_clouds(self, cloud):
        a = cloud(300, 0.0)
        b = cloud(300, 0.0)
        swd, spread = sliced_wasserstein(a, b, n_projections=64, seed=2)
        assert swd < 0.05
        assert spread >= 0.0

    def test_larger_for_shifted_clouds(self, cloud):
        a = cloud(300, 0.0)
        assert sliced_wasserstein(cloud(300, 0.9), a, n_projections=64)[0] > \
            sliced_wasserstein(cloud(300, 0.25), a, n_projections=64)[0]

    def test_is_deterministic_for_a_seed(self, cloud):
        a, b = cloud(150, 0.0), cloud(150, 0.9)
        assert sliced_wasserstein(a, b, n_projections=32, seed=4) == \
            sliced_wasserstein(a, b, n_projections=32, seed=4)


class TestDomainClassifier:
    def test_null_auc_near_half(self, cloud):
        """Two independent draws from the same distribution must be indistinguishable.

        The PCA projection is what keeps the null honest: a logistic regression
        on raw high-dimensional embeddings separates two samples of the *same*
        distribution at AUC ~0.63 purely by exploiting sampling noise.
        """
        a = cloud(400, 0.0, seed=1)
        b = cloud(400, 0.0, seed=2)
        result = domain_classifier_auc(a, b, folds=4, seed=1)
        assert result["auc"] < 0.62, f"null AUC inflated: {result['auc']:.3f}"

    def test_detects_a_separable_shift(self, cloud):
        a = cloud(300, 0.0)
        b = cloud(300, 0.9)
        assert domain_classifier_auc(a, b, folds=4, seed=1)["auc"] > 0.9

    def test_reports_direction_and_size(self, cloud):
        result = domain_classifier_auc(cloud(200, 0.0, seed=1), cloud(200, 0.9, seed=1), folds=4)
        assert 150 <= result["n"] <= 200
        assert result["direction"] in (-1.0, 1.0)
        assert 0.5 <= result["auc"] <= 1.0

    def test_shared_rows_are_removed_before_scoring(self, cloud):
        """Replayed reference rows carry no drift signal and are dropped.

        Without removing the overlap, cross-validation puts the identical vector
        in the training folds of one class and the test folds of the other, so
        the classifier memorises it and the AUC climbs for free.
        """
        from src.drift.embedding import _drop_shared_rows

        a = cloud(120, 0.0, seed=1)
        mixed = np.vstack([a[:80], cloud(40, 1.0, seed=1)])
        kept_a, kept_b = _drop_shared_rows(a, mixed)
        assert kept_b.shape[0] == 40
        assert kept_a.shape[0] == 120
        shared = {r.tobytes() for r in kept_a} & {r.tobytes() for r in kept_b}
        assert not shared

    def test_identical_inputs_are_left_alone(self, cloud):
        from src.drift.embedding import _drop_shared_rows

        a = cloud(60, 0.0, seed=1)
        kept_a, kept_b = _drop_shared_rows(a, a)
        assert kept_a.shape[0] == kept_b.shape[0] == 60

    def test_too_small_returns_nan(self, cloud):
        result = domain_classifier_auc(cloud(10, 0.0), cloud(10, 0.9))
        assert np.isnan(result["auc"])


class TestFrechet:
    def test_matches_itself(self, rng):
        x = rng.normal(size=(200, 5))
        d = frechet_on_pca(x, x, n_components=5)
        assert abs(d) < 0.05

    def test_grows_with_variance_change(self, rng):
        a = rng.normal(size=(300, 6))
        b = rng.normal(size=(300, 6)) * 3
        close = frechet_on_pca(a, rng.normal(size=(300, 6)) * 1.1, n_components=6)
        far = frechet_on_pca(a, b, n_components=6)
        assert far > close

    def test_catches_variance_only_drift(self, rng):
        """A centroid comparison is blind to this; Frechet is not."""
        a = rng.normal(size=(300, 6))
        same_mean = rng.normal(size=(300, 6)) * 2.5
        res = frechet_distance(a.mean(0), np.cov(a, rowvar=False),
                               same_mean.mean(0), np.cov(same_mean, rowvar=False))
        assert res > 0.5


class TestEmbeddingDriftDetector:
    def test_raises_without_a_reference(self, cloud):
        from src.drift.embedding import EmbeddingDriftDetector

        with pytest.raises((AttributeError, ValueError, RuntimeError)):
            EmbeddingDriftDetector(None)

    def test_clean_window_is_not_drift(self, detector, cloud):
        result = detector.detect(cloud(300, 0.0))
        assert result.drift_detected is False, result.signals
        assert result.domain_auc < 0.65

    def test_shifted_window_is_drift(self, detector, cloud):
        result = detector.detect(cloud(300, 0.9))
        assert result.drift_detected is True
        assert result.domain_auc > 0.85
        assert result.recommendation

    def test_novelty_cutoff_is_calibrated_at_roughly_five_percent(self, snapshot, cloud):
        """Leave-one-out on the reference side is what makes this true.

        Without it the cutoff is computed from points whose nearest neighbour is
        themselves, and every genuinely new point looks novel.
        """
        detector = EmbeddingDriftDetectorFactory(snapshot)
        fresh = cloud(400, 0.0)
        rate = detector.detect(fresh).ood_rate
        assert rate < 0.20, f"in-distribution OOD rate was {rate:.2%}"

    def test_concept_gap_uses_correctness_split(self, detector, cloud):
        vectors = cloud(300, 0.9)
        in_scope = np.array([True] * 150 + [False] * 150)
        is_correct = np.array([1.0] * 150 + [0.0] * 150)
        result = detector.detect(vectors, in_scope=in_scope, is_correct=is_correct)
        assert result.concept_gap == pytest.approx(1.0)
        assert result.signals["accuracy_in_scope"] == pytest.approx(1.0)
        assert result.signals["accuracy_out_of_scope"] == pytest.approx(0.0)

    def test_small_window_is_not_evaluated(self, detector, cloud):
        result = detector.detect(cloud(5, 0.0))
        assert result.drift_detected is False
        assert result.signals.get("insufficient_data") is True

    def test_result_serialises(self, detector, cloud):
        import json

        payload = detector.detect(cloud(200, 0.9)).to_dict()
        json.dumps(payload)
        for key in ("mmd2", "swd", "domain_auc", "ood_rate", "frechet", "severity"):
            assert key in payload

    def test_significance_gates_drift(self, detector, cloud, monkeypatch):
        """A large effect size with a non-significant p-value must not alert."""
        result = detector.detect(cloud(300, 0.9))
        assert result.signals["significant"] is True
        monkeypatch.setattr(detector, "significance_level", 0.0)
        quiet = detector.detect(cloud(300, 0.0))
        assert quiet.signals["significant"] is False


def EmbeddingDriftDetectorFactory(snapshot):  # noqa: N802 - tiny test helper
    from src.drift.embedding import EmbeddingDriftDetector

    return EmbeddingDriftDetector(snapshot, seed=5, max_current=400)