"""
Embedding drift
---------------
Text drift cannot be reduced to per-column PSI: the interesting failures live
in *relationships* between words, in topics you never enumerated, and in the
geometry of the representation itself. These detectors measure the
distribution of sentence embeddings directly.

Five complementary views, each with a different failure mode:

``MMD`` (Maximum Mean Discrepancy, RBF kernel, permutation-calibrated)
    The workhorse. A kernel two-sample test: it fires when the *shape* of the
    embedding cloud changes even when every marginal looks plausible. Cost is
    quadratic in the sample size, hence the importance of permuting a
    permutation test for a p-value rather than trusting a raw number.

``Sliced Wasserstein`` (average over random 1-D projections)
    Wasserstein-1 between the reference and current clouds, projected onto
    random directions and averaged. Cheap, robust, and — unlike MMD — it has
    an interpretable scale ("the average gap along this axis is 0.03 cosine
    units"), which makes thresholds arguable in a code review.

``Domain classifier AUC``
    Train a regularised logistic regression to tell reference from current.
    AUC of 0.5 means the two are indistinguishable; 0.75+ means a model could
    exploit the difference, which is the most decision-relevant framing of
    drift you can give a stakeholder. This is Arize/Evidently's "drift as a
    prediction problem" and it is deliberately the headline signal.

``Fréchet distance`` (Frechet Inception Distance, on PCA-reduced embeddings)
    Penalises both a mean shift and a covariance change, i.e. it also catches
    "same average but different spread" — variance drift — which centroid
    cosine misses entirely.

``Novelty / OOD rate``
    Fraction of traffic that sits further from its k nearest reference points
    than 95% of the reference does from itself. Converts an abstract distance
    into "38% of today's traffic is unlike anything this model was trained on".

Plus a first-class **concept drift** signal: performance on in-scope vs
out-of-scope traffic, which separates "the inputs changed" from "our answer
became wrong for a reason that is not visible in the inputs".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from config import settings
from src.storage.reference import ReferenceSnapshot
from src.utils.logging import get_logger
from src.utils.stats import l2_normalize

logger = get_logger(__name__)

SEVERITY_ORDER = {"none": 0, "moderate": 1, "severe": 2}


def _severity(value: float, moderate: float, severe: float, higher_is_worse: bool = True) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "none"
    if higher_is_worse:
        if value >= severe:
            return "severe"
        if value >= moderate:
            return "moderate"
        return "none"
    if value <= severe:
        return "severe"
    if value <= moderate:
        return "moderate"
    return "none"


def worst(*severities: str) -> str:
    return max((s for s in severities if s), key=lambda s: SEVERITY_ORDER.get(s, 0), default="none")


# ── Kernel utilities ───────────────────────────────────────────────────

def _pairwise_sqdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a2 = (a ** 2).sum(axis=1)[:, None]
    b2 = (b ** 2).sum(axis=1)[None, :]
    d2 = a2 + b2 - 2.0 * (a @ b.T)
    np.maximum(d2, 0.0, out=d2)
    return d2


def median_bandwidth(a: np.ndarray, b: np.ndarray, max_samples: int = 600,
                     seed: int = 0) -> float:
    """Median heuristic for the RBF bandwidth — scale-free and robust."""
    rng = np.random.default_rng(seed)
    pool = np.vstack([a, b])
    if pool.shape[0] > max_samples:
        pool = pool[rng.choice(pool.shape[0], size=max_samples, replace=False)]
    d2 = _pairwise_sqdist(pool, pool)
    iu = np.triu_indices(d2.shape[0], k=1)
    med = float(np.median(d2[iu]))
    return float(np.sqrt(max(med, 1e-8) / 2.0))


def _rbf(a: np.ndarray, b: np.ndarray, gamma: float) -> np.ndarray:
    return np.exp(-gamma * _pairwise_sqdist(a, b))


def _gram(x: np.ndarray, gamma: float) -> np.ndarray:
    """Self-kernel matrix, cached because permutations reuse it constantly."""
    return _rbf(x, x, gamma)


def mmd2_unbiased(a: np.ndarray, b: np.ndarray, gamma: float | None = None) -> float:
    """Unbiased MMD² estimate under the null H0: P=A, Q=B."""
    if gamma is None:
        gamma = 1.0 / (2.0 * median_bandwidth(a, b) ** 2)
    kaa = _rbf(a, a, gamma)
    kbb = _rbf(b, b, gamma)
    kab = _rbf(a, b, gamma)
    n, m = a.shape[0], b.shape[0]
    if n < 2 or m < 2:
        return float("nan")
    term_a = (kaa.sum() - np.trace(kaa)) / (n * (n - 1))
    term_b = (kbb.sum() - np.trace(kbb)) / (m * (m - 1))
    term_ab = kab.mean()
    return float(term_a + term_b - 2.0 * term_ab)


def _mmd2_from_grams(kxx: np.ndarray, kyy: np.ndarray, kxy: np.ndarray) -> float:
    n, m = kxx.shape[0], kyy.shape[0]
    term_a = (kxx.sum() - np.trace(kxx)) / (n * (n - 1))
    term_b = (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
    return float(term_a + term_b - 2.0 * kxy.mean())


def mmd_test(a: np.ndarray, b: np.ndarray, n_permutations: int | None = None,
             seed: int = 17, max_pool: int = 420) -> tuple[float, float, float]:
    """
    Permutation-calibrated MMD.

    Returns ``(mmd2, p_value, gamma)``. The p-value is what makes this usable
    in CI: the MMD² scale depends on the bandwidth and the sample size, so a
    raw threshold is meaningless across windows with different traffic volumes.

    Both sides are subsampled to the same bound *before* the statistic is
    computed, and the permutation null is built from that same pooled matrix.
    Skipping this calibration step — computing the observed value on the full
    window and the null on a subsample — makes every window significant,
    because the unbiased MMD² estimator has a downward bias whose magnitude
    falls with n. Getting the two on the same footing is the difference
    between a usable p-value and a detector that cries wolf 100% of the time.
    """
    n_permutations = n_permutations or settings.MMD_PERMUTATIONS
    rng = np.random.default_rng(seed)
    gamma = 1.0 / (2.0 * median_bandwidth(a, b, seed=seed) ** 2)

    n_side = int(min(max_pool, a.shape[0], b.shape[0]))
    if n_side < 10:
        logger.info("MMD skipped: only %d rows per side (need 10).", n_side)
        return float("nan"), float("nan"), gamma

    A = a if a.shape[0] == n_side else a[rng.choice(a.shape[0], n_side, replace=False)]
    B = b if b.shape[0] == n_side else b[rng.choice(b.shape[0], n_side, replace=False)]

    pooled = np.vstack([A, B])
    kpool = _rbf(pooled, pooled, gamma)          # one kernel pass, reused below
    n = A.shape[0]

    observed = _mmd2_from_grams(kpool[:n, :n], kpool[n:, n:], kpool[:n, n:])
    if not np.isfinite(observed):
        return float("nan"), float("nan"), gamma

    null = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        perm = rng.permutation(pooled.shape[0])
        iu, ju = perm[:n], perm[n:]
        null[i] = _mmd2_from_grams(kpool[np.ix_(iu, iu)], kpool[np.ix_(ju, ju)],
                                   kpool[np.ix_(iu, ju)])

    p = float((np.sum(null >= observed) + 1) / (n_permutations + 1))
    return observed, p, gamma


# ── Sliced Wasserstein ─────────────────────────────────────────────────

def _sorted_wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    """
    1-D Wasserstein-1 between equal-size samples.

    For equal sample sizes the optimal transport plan is the sorted pairing, so
    this is O(n log n) with no SciPy call in the inner loop — which matters when
    it runs a few hundred times per window.
    """
    n = min(a.size, b.size)
    if n == 0:
        return float("nan")
    if a.size != n:
        a = np.quantile(a, np.linspace(0, 1, n))
    if b.size != n:
        b = np.quantile(b, np.linspace(0, 1, n))
    return float(np.abs(np.sort(a) - np.sort(b)).mean())


def sliced_wasserstein(a: np.ndarray, b: np.ndarray, n_projections: int | None = None,
                       seed: int = 23) -> tuple[float, float]:
    """Average 1-D Wasserstein-1 distance over random unit projections.

    Returns ``(swd, swd_std_over_projections)``. The spread across projections
    tells you whether the shift is broad or concentrated in a few directions.
    """
    n_projections = n_projections or settings.WASSERSTEIN_PROJECTIONS
    rng = np.random.default_rng(seed)
    d = a.shape[1]
    n_proj = min(n_projections, max(32, d))
    directions = rng.standard_normal((d, n_proj)).astype(np.float64)
    directions /= np.linalg.norm(directions, axis=0, keepdims=True) + 1e-12

    proj_a = a @ directions
    proj_b = b @ directions
    dists = np.array([_sorted_wasserstein_1d(proj_a[:, j], proj_b[:, j])
                      for j in range(n_proj)], dtype=np.float64)
    return float(dists.mean()), float(dists.std())


# ── Fréchet distance ───────────────────────────────────────────────────

def frechet_distance(mu1: np.ndarray, sigma1: np.ndarray,
                     mu2: np.ndarray, sigma2: np.ndarray, eps: float = 1e-6) -> float:
    """Raw Frechet distance between two multivariate Gaussians (FID formulation)."""
    from scipy import linalg

    diff = mu1 - mu2
    # scipy >= 1.16 dropped the `disp` argument; handle both without a version pin.
    try:
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    except TypeError:
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        try:
            covmean, _ = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=False)
        except TypeError:
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))


def frechet_on_pca(ref: np.ndarray, cur: np.ndarray, n_components: int = 50,
                   seed: int = 11) -> float:
    """
    Normalised Frechet distance between two clouds.

    The PCA basis is fitted on the **reference only** and both clouds are
    projected into it — fitting on the pooled data would let the basis absorb
    part of the shift and understate the distance.

    The raw Frechet number scales with embedding magnitude, so a threshold
    tuned for one encoder is meaningless on another. Dividing by the average
    covariance trace makes it dimensionless: ~0 means "same Gaussian shape",
    and it grows with the relative size of the mean and covariance change.
    """
    from sklearn.decomposition import PCA

    pooled = np.vstack([ref, cur])
    k = int(min(n_components, ref.shape[0] - 1, cur.shape[0] - 1, pooled.shape[1] - 1))
    if k < 2:
        return float("nan")
    pca = PCA(n_components=k, random_state=seed).fit(ref)
    r = pca.transform(ref)
    c = pca.transform(cur)
    s1 = np.cov(r, rowvar=False) + np.eye(k) * 1e-8
    s2 = np.cov(c, rowvar=False) + np.eye(k) * 1e-8
    raw = frechet_distance(r.mean(axis=0), s1, c.mean(axis=0), s2)
    scale = 0.5 * (float(np.trace(s1)) + float(np.trace(s2)))
    return float(raw / scale) if scale > 1e-12 else float("nan")


# ── Domain classifier ──────────────────────────────────────────────────

def _drop_shared_rows(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove vectors that appear in both samples.

    A row present in the reference *and* in the current window carries no
    information about drift, but cross-validation puts the identical vector in
    the training folds of one class and the test folds of the other, so the
    classifier memorises it and the AUC inflates. Reference traffic and current
    traffic are normally disjoint, but a replayed window, a resampled test set
    or a duplicated scrape would all trigger this, and the failure is silent.
    """
    keys_a = {row.tobytes() for row in np.ascontiguousarray(a, dtype=np.float32)}
    keep = [i for i, row in enumerate(np.ascontiguousarray(b, dtype=np.float32))
            if row.tobytes() not in keys_a]
    if not keep:
        return a, b
    return a, b[np.array(keep)]


def domain_classifier_auc(ref: np.ndarray, cur: np.ndarray, folds: int | None = None,
                          seed: int = 31, n_components: int = 48) -> dict[str, float]:
    """
    Can a linear model tell reference from current traffic?

    AUC is reported as ``max(auc, 1-auc)`` so the statistic always answers
    "how separable are these?", never "can it tell them apart, and in which
    direction". A p-value comes from a permutation of the labels.

    The projection matters more than it looks. A logistic regression on 384
    raw sentence-embedding dimensions separates two samples of the *same*
    distribution at AUC ≈ 0.63 purely by exploiting sampling noise — high
    dimensionality is enough. Fitting PCA on the reference and projecting both
    clouds into it removes that inflation, so the null sits near 0.50 and the
    configured thresholds mean what they say. Without this step the detector
    cries wolf on every window, which is the fastest way to get a drift
    dashboard switched off.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    folds = folds or settings.DOMAIN_CV_FOLDS
    n = min(ref.shape[0], cur.shape[0])
    if n < 20:
        return {"auc": float("nan"), "p_value": float("nan"), "n": float(n), "direction": float("nan")}

    rng = np.random.default_rng(seed)
    idx_r = rng.choice(ref.shape[0], size=n, replace=False)
    idx_c = rng.choice(cur.shape[0], size=n, replace=False)
    R, C = _drop_shared_rows(ref[idx_r], cur[idx_c])

    if R.shape[0] < 20 or C.shape[0] < 20:
        return {"auc": float("nan"), "p_value": float("nan"),
                "n": float(min(R.shape[0], C.shape[0])), "direction": float("nan")}

    k = int(min(n_components, R.shape[0] - 1, R.shape[1] - 1))
    if k < 2:
        return {"auc": float("nan"), "p_value": float("nan"), "n": float(n), "direction": float("nan")}
    pca = PCA(n_components=k, random_state=seed).fit(R)
    X = np.vstack([pca.transform(R), pca.transform(C)])
    y = np.concatenate([np.zeros(R.shape[0], dtype=int), np.ones(C.shape[0], dtype=int)])
    n = int(min(R.shape[0], C.shape[0]))

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
    cv = StratifiedKFold(n_splits=min(folds, max(2, n // 10)), shuffle=True, random_state=seed)
    oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, oof))
    strength = float(max(auc, 1.0 - auc))

    # Permutation p-value on the out-of-fold ranking.
    observed = auc
    null = np.empty(100, dtype=np.float64)
    for i in range(null.size):
        null[i] = roc_auc_score(rng.permutation(y), oof)
    p = float((np.sum(null >= observed) + 1) / (null.size + 1))

    return {
        "auc": strength,
        "raw_auc": auc,
        "direction": float(np.sign(auc - 0.5)),
        "p_value": p,
        "n": float(n),
        "pca_components": float(k),
    }


# ── Results container ──────────────────────────────────────────────────

@dataclass
class EmbeddingDriftResult:
    drift_detected: bool = False
    severity: str = "none"
    n_reference: int = 0
    n_current: int = 0
    dim: int = 0
    mmd2: float = float("nan")
    mmd_p_value: float = float("nan")
    mmd_gamma: float = float("nan")
    swd: float = float("nan")
    swd_std: float = float("nan")
    centroid_cosine_shift: float = float("nan")
    frechet: float = float("nan")
    domain_auc: float = float("nan")
    domain_p_value: float = float("nan")
    ood_rate: float = float("nan")
    ood_threshold: float = float("nan")
    concept_gap: float = float("nan")
    signals: dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "drift_detected": self.drift_detected,
            "severity": self.severity,
            "n_reference": self.n_reference,
            "n_current": self.n_current,
            "dim": self.dim,
            "mmd2": _r(self.mmd2, 6),
            "mmd_p_value": _r(self.mmd_p_value, 5),
            "mmd_gamma": _r(self.mmd_gamma, 5),
            "swd": _r(self.swd, 5),
            "swd_std": _r(self.swd_std, 5),
            "centroid_cosine_shift": _r(self.centroid_cosine_shift, 5),
            "frechet": _r(self.frechet, 4),
            "domain_auc": _r(self.domain_auc, 4),
            "domain_p_value": _r(self.domain_p_value, 5),
            "ood_rate": _r(self.ood_rate, 4),
            "ood_threshold": _r(self.ood_threshold, 4),
            "concept_gap": _r(self.concept_gap, 4),
            "signals": self.signals,
            "recommendation": self.recommendation,
            "timestamp": self.timestamp,
        }


def _r(v: float, n: int) -> float | None:
    if v is None:
        return None
    v = float(v)
    return None if not np.isfinite(v) else round(v, n)


class EmbeddingDriftDetector:
    """
    Stateful detector bound to one reference snapshot. One instance per
    monitored system; call :meth:`detect` once per evaluation window.
    """

    def __init__(
        self,
        snapshot: ReferenceSnapshot,
        *,
        mmd_moderate: float = settings.MMD_MODERATE,
        mmd_severe: float = settings.MMD_SEVERE,
        swd_moderate: float = settings.SWD_MODERATE,
        swd_severe: float = settings.SWD_SEVERE,
        centroid_moderate: float = settings.CENTROID_COS_MODERATE,
        centroid_severe: float = settings.CENTROID_COS_SEVERE,
        domain_moderate: float = settings.DOMAIN_AUC_MODERATE,
        domain_severe: float = settings.DOMAIN_AUC_SEVERE,
        frechet_moderate: float = settings.FRECHET_MODERATE,
        frechet_severe: float = settings.FRECHET_SEVERE,
        novelty_moderate: float = settings.NOVELTY_MODERATE,
        novelty_severe: float = settings.NOVELTY_SEVERE,
        required_votes: int = settings.EMBEDDING_DRIFT_VOTES,
        significance_level: float = 0.05,
        max_current: int = 900,
        seed: int = 17,
    ):
        self.snap = snapshot
        self.mmd_moderate, self.mmd_severe = mmd_moderate, mmd_severe
        self.swd_moderate, self.swd_severe = swd_moderate, swd_severe
        self.centroid_moderate, self.centroid_severe = centroid_moderate, centroid_severe
        self.domain_moderate, self.domain_severe = domain_moderate, domain_severe
        self.frechet_moderate, self.frechet_severe = frechet_moderate, frechet_severe
        self.novelty_moderate, self.novelty_severe = novelty_moderate, novelty_severe
        self.required_votes = required_votes
        self.significance_level = significance_level
        self.max_current = max_current
        self.seed = seed

        # Reference-side statistics computed once, with leave-one-out k-NN so the
        # novelty cutoff is calibrated on genuinely unseen-to-themselves points.
        ref = snapshot.embeddings
        k_ref = min(5, max(1, ref.shape[0] - 1))
        d_ref = snapshot.kth_neighbour_distance(ref, k=k_ref, exclude_self=True)
        self._novelty_cutoff = float(np.quantile(d_ref, 0.95))
        self._ref_knn_mean = float(d_ref.mean())

    # ── main entry point ───────────────────────────────────────────
    def detect(
        self,
        current_embeddings: np.ndarray,
        *,
        in_scope: np.ndarray | None = None,
        is_correct: np.ndarray | None = None,
    ) -> EmbeddingDriftResult:
        cur = l2_normalize(np.asarray(current_embeddings, dtype=np.float32))
        res = EmbeddingDriftResult(
            n_reference=self.snap.n, n_current=int(cur.shape[0]), dim=int(cur.shape[1]),
            timestamp=pd.Timestamp.now('UTC').strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        if cur.shape[0] < settings.MIN_WINDOW or self.snap.n < 10:
            res.recommendation = (
                f"Insufficient traffic ({cur.shape[0]} rows, need {settings.MIN_WINDOW}) — "
                "drift not evaluated for this window."
            )
            res.signals["insufficient_data"] = True
            return res

        if cur.shape[0] > self.max_current:
            rng = np.random.default_rng(self.seed)
            cur = cur[rng.choice(cur.shape[0], size=self.max_current, replace=False)]

        # Reference subsample keeps MMD cost bounded on long-lived systems.
        ref = self.snap.embeddings
        if ref.shape[0] > self.max_current:
            ref = ref[np.random.default_rng(self.seed).choice(ref.shape[0], self.max_current, replace=False)]

        # 1. MMD with a permutation-calibrated p-value
        res.mmd2, res.mmd_p_value, res.mmd_gamma = mmd_test(ref, cur, seed=self.seed)
        mmd_sev = _severity(max(0.0, res.mmd2), self.mmd_moderate, self.mmd_severe)

        # 2. Sliced Wasserstein
        res.swd, res.swd_std = sliced_wasserstein(ref, cur, seed=self.seed)
        swd_sev = _severity(res.swd, self.swd_moderate, self.swd_severe)

        # 3. Centroid cosine shift
        cos = float(np.dot(l2_normalize(cur.mean(axis=0)), l2_normalize(ref.mean(axis=0))))
        res.centroid_cosine_shift = 1.0 - cos
        centroid_sev = _severity(res.centroid_cosine_shift, self.centroid_moderate, self.centroid_severe)

        # 4. Fréchet (variance + mean)
        res.frechet = frechet_on_pca(ref, cur, seed=self.seed)
        frechet_sev = _severity(res.frechet, self.frechet_moderate, self.frechet_severe)

        # 5. Domain classifier
        dom = domain_classifier_auc(ref, cur, seed=self.seed)
        res.domain_auc = dom["auc"]
        res.domain_p_value = dom["p_value"]
        domain_sev = _severity(res.domain_auc, self.domain_moderate, self.domain_severe)

        # 6. OOD / novelty rate
        knn = self.snap.kth_neighbour_distance(cur, k=min(5, self.snap.n - 1))
        res.ood_rate = float(np.mean(knn > self._novelty_cutoff))
        res.ood_threshold = self._novelty_cutoff
        ood_sev = _severity(res.ood_rate, self.novelty_moderate, self.novelty_severe)

        # 7. Concept drift: does accuracy collapse on the unfamiliar slice?
        if in_scope is not None and is_correct is not None:
            in_scope = np.asarray(in_scope, dtype=bool)
            is_correct = np.asarray(is_correct, dtype=float)
            ins = is_correct[in_scope].mean() if in_scope.any() else np.nan
            oos = is_correct[~in_scope].mean() if (~in_scope).any() else np.nan
            if np.isfinite(ins) and np.isfinite(oos) and in_scope.sum() > 5 and (~in_scope).sum() > 5:
                res.concept_gap = float(ins - oos)
            res.signals["accuracy_in_scope"] = _r(ins, 4)
            res.signals["accuracy_out_of_scope"] = _r(oos, 4)
            res.signals["out_of_scope_share"] = _r(float((~in_scope).mean()), 4)
            concept_sev = _severity(max(0.0, -res.concept_gap), 0.20, 0.40) \
                if np.isfinite(res.concept_gap) else "none"
        else:
            concept_sev = "none"

        votes = {
            "mmd": mmd_sev,
            "sliced_wasserstein": swd_sev,
            "centroid_shift": centroid_sev,
            "frechet": frechet_sev,
            "domain_classifier": domain_sev,
            "novelty": ood_sev,
            "concept": concept_sev,
        }
        res.signals.update(votes)

        # A permutation test is a *confirmation*, not an escalation. Firing on a
        # p-value alone means every window of a busy hour looks significant, and
        # the permutation null is exactly what stops that. So: severity comes
        # from effect size (does the shift matter?), and significance gates
        # whether the window is allowed to raise drift at all.
        significant = any(
            p is not None and np.isfinite(p) and p < self.significance_level
            for p in (res.mmd_p_value, res.domain_p_value)
        )
        res.signals["significant"] = bool(significant)
        res.signals["significance_level"] = self.significance_level
        res.signals["nn_distance_ratio"] = _r(float(np.mean(knn) / (self._ref_knn_mean or 1.0)), 4)

        firing = [s for s in votes.values() if s != "none"]
        res.severity = worst(*votes.values())
        res.drift_detected = len(firing) >= self.required_votes and significant
        if res.severity == "severe" and len(firing) >= 1 and significant:
            res.drift_detected = True
        res.recommendation = self._recommend(res, votes)
        return res

    @staticmethod
    def _recommend(res: EmbeddingDriftResult, votes: dict[str, str]) -> str:
        if not res.drift_detected:
            return "Embedding distribution matches the reference window. No action."
        firing = [k for k, v in votes.items() if v == "severe"]
        firing = firing or [k for k, v in votes.items() if v != "none"]
        listed = ", ".join(sorted(set(firing)))
        if res.severity == "severe":
            return (
                f"SEVERE embedding drift ({listed}). The traffic no longer resembles the "
                f"reference cloud — domain classifier AUC {res.domain_auc:.2f}, OOD rate "
                f"{res.ood_rate:.0%}. Retrain on recent data and re-scope intents before trusting output."
            )
        if "concept" in votes and votes["concept"] != "none":
            return (
                f"Concept drift on out-of-scope traffic (gap {res.concept_gap:.2f}) with input drift "
                f"({listed}). The inputs changed and our answers got worse for the new traffic — "
                "retrain, and add the new intents to the label set."
            )
        return (
            f"Moderate embedding drift ({listed}); domain classifier AUC {res.domain_auc:.2f}. "
            "Investigate the top-decaying features and hold retraining until quality confirms it."
        )
