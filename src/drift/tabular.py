"""
Tabular drift
-------------
The classic first line of defence. For numeric *and* categorical columns:

* **PSI** (Population Stability Index) — the finance standard, cheap, stable,
  and interpretable in a dashboard because everyone has seen it before.
* **KS test** — p-value based, so it adapts to sample size, which PSI does not.
  A large window makes PSI fire on trivial differences; the KS p-value does not.
* **Jensen–Shannon divergence** — bounded and symmetric, which makes it safe
  to average across columns of very different scales.
* **Chi-square** for categoricals, with the same "is it significant at this n?"
  property as KS.

Two operational details that are usually missing from textbook implementations
and that matter in production:

1. **Reference bin edges are frozen** at training time. Recomputing quantiles
   per window makes every window look identical by construction — the single
   most common way a drift dashboard ends up lying to you.
2. **Minimum sample counts** are enforced per column, and columns that are
   effectively constant in the reference are marked ``low_variance`` instead
   of being allowed to generate noise-driven alerts.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

from config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

EPS = 1e-6
SEVERITY_ORDER = {"none": 0, "moderate": 1, "severe": 2}

CATEGORICAL_HINT = ("_is", "_cat", "_flag", "channel", "tier", "region", "segment",
                    "device", "locale", "source")


def _severity(value: float, moderate: float, severe: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "none"
    if value >= severe:
        return "severe"
    if value >= moderate:
        return "moderate"
    return "none"


@dataclass
class ColumnDrift:
    column: str
    kind: str                     # "numeric" | "categorical"
    psi: float = float("nan")
    ks_statistic: float = float("nan")
    ks_p_value: float = float("nan")
    chi2_p_value: float = float("nan")
    js_divergence: float = float("nan")
    chi2_statistic: float = float("nan")
    reference_mean: float | None = None
    current_mean: float | None = None
    reference_mean_shift_sd: float | None = None
    current_missing: float = 0.0
    reference_missing: float = 0.0
    low_variance: bool = False
    drift_detected: bool = False
    severity: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "kind": self.kind,
            "psi": _r(self.psi, 4),
            "ks_statistic": _r(self.ks_statistic, 4),
            "ks_p_value": _r(self.ks_p_value, 5),
            "chi2_p_value": _r(self.chi2_p_value, 5),
            "js_divergence": _r(self.js_divergence, 4),
            "reference_mean": _r(self.reference_mean, 4),
            "current_mean": _r(self.current_mean, 4),
            "reference_mean_shift_sd": _r(self.reference_mean_shift_sd, 3),
            "current_missing": _r(self.current_missing, 4),
            "reference_missing": _r(self.reference_missing, 4),
            "low_variance": self.low_variance,
            "drift_detected": self.drift_detected,
            "severity": self.severity,
        }


def _r(v: Any, n: int) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(f) else round(f, n)


@dataclass
class TabularDriftResult:
    drift_detected: bool = False
    severity: str = "none"
    columns: list[ColumnDrift] = field(default_factory=list)
    drifted_columns: list[str] = field(default_factory=list)
    psi_mean: float = 0.0
    psi_max: float = 0.0
    psi_max_column: str = ""
    n_drifted_fraction: float = 0.0
    recommendation: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "drift_detected": self.drift_detected,
            "severity": self.severity,
            "drifted_columns": self.drifted_columns,
            "psi_mean": _r(self.psi_mean, 4),
            "psi_max": _r(self.psi_max, 4),
            "psi_max_column": self.psi_max_column,
            "n_drifted_fraction": _r(self.n_drifted_fraction, 4),
            "recommendation": self.recommendation,
            "timestamp": self.timestamp,
            "columns": [c.to_dict() for c in self.columns],
        }


class TabularDriftDetector:
    """Frozen-bin tabular drift detector. One instance per monitored dataset."""

    def __init__(
        self,
        reference: pd.DataFrame,
        columns: Sequence[str] | None = None,
        *,
        n_bins: int = settings.DRIFT_N_BINS,
        psi_moderate: float = settings.PSI_MODERATE,
        psi_severe: float = settings.PSI_SEVERE,
        p_threshold: float = settings.KS_P_THRESHOLD,
        min_rows: int = settings.MIN_WINDOW,
        max_categorical_levels: int = 40,
    ):
        self.reference = reference.copy()
        self.columns = list(columns) if columns else [
            c for c in reference.columns if c not in {"target", "gold_intent", "intent_id", "source"}
        ]
        self.n_bins = n_bins
        self.psi_moderate = psi_moderate
        self.psi_severe = psi_severe
        self.p_threshold = p_threshold
        self.min_rows = min_rows
        self._frozen: dict[str, dict[str, Any]] = {}

        for col in self.columns:
            if col not in reference.columns:
                continue
            ref = reference[col]
            if self._is_categorical(col, ref):
                levels = ref.value_counts(normalize=True)
                self._frozen[col] = {
                    "kind": "categorical",
                    "levels": levels.index.astype(str).tolist(),
                    "ref_probs": levels.values.astype(float),
                    "missing": float(ref.isna().mean()),
                }
            else:
                vals = ref.dropna().to_numpy(dtype=float)
                if vals.size < 2:
                    continue
                lo, hi = float(vals.min()), float(vals.max())
                pad = (hi - lo) * 0.01 + 1e-9
                edges = np.unique(np.quantile(vals, np.linspace(0, 1, self.n_bins + 1)))
                if edges.size < 3:
                    edges = np.linspace(lo - pad, hi + pad, 3)
                edges = np.concatenate([[edges[0] - pad], edges[1:-1], [edges[-1] + pad]])
                counts, _ = np.histogram(vals, bins=edges)
                probs = counts / max(counts.sum(), 1)
                self._frozen[col] = {
                    "kind": "numeric",
                    "edges": edges,
                    "ref_probs": np.maximum(probs, EPS),
                    "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                    "missing": float(ref.isna().mean()),
                    "low_variance": bool(np.std(vals) < 1e-8 or np.unique(vals).size <= 2),
                }
        logger.info("Tabular detector frozen on %d columns", len(self._frozen))

    @staticmethod
    def _is_categorical(col: str, series: pd.Series) -> bool:
        if series.dtype == object or series.dtype.name in {"category", "bool"}:
            return True
        return any(hint in col.lower() for hint in CATEGORICAL_HINT)

    # ── metrics ─────────────────────────────────────────────────────
    def _psi(self, ref_probs: np.ndarray, cur_counts: np.ndarray, edges: np.ndarray) -> float:
        cur_counts, _ = np.histogram(cur_counts, bins=edges)
        cur_probs = np.maximum(cur_counts / max(cur_counts.sum(), 1), EPS)
        ref_probs = np.maximum(ref_probs, EPS)
        return float(np.sum((cur_probs - ref_probs) * np.log(cur_probs / ref_probs)))

    def _cat_psi(self, levels: list[str], ref_probs: np.ndarray, cur: pd.Series) -> float:
        counts = cur.astype(str).value_counts()
        cur_probs = np.array([counts.get(lv, 0) for lv in levels], dtype=float)
        cur_probs = np.maximum(cur_probs / max(cur_probs.sum(), 1), EPS)
        return float(np.sum((cur_probs - ref_probs) * np.log(cur_probs / ref_probs)))

    def _chi2(self, levels: list[str], ref_probs: np.ndarray, cur: pd.Series) -> tuple[float, float]:
        counts = cur.astype(str).value_counts()
        observed = np.array([counts.get(lv, 0) for lv in levels], dtype=float)
        expected = ref_probs * observed.sum()
        mask = expected > 0
        if mask.sum() < 2 or observed.sum() < settings.MIN_WINDOW:
            return float("nan"), float("nan")
        stat, p = stats.chisquare(observed[mask], expected[mask])
        return float(stat), float(p)

    def _js(self, ref_vals: np.ndarray, cur_vals: np.ndarray, edges: np.ndarray) -> float:
        if ref_vals.size < 3 or cur_vals.size < 3:
            return float("nan")
        rh, _ = np.histogram(ref_vals, bins=edges)
        ch, _ = np.histogram(cur_vals, bins=edges)
        rh = rh.astype(float) + EPS
        ch = ch.astype(float) + EPS
        return float(jensenshannon(rh / rh.sum(), ch / ch.sum()))

    # ── main entry point ────────────────────────────────────────────
    def detect(self, current: pd.DataFrame) -> TabularDriftResult:
        res = TabularDriftResult(timestamp=pd.Timestamp.now('UTC').strftime("%Y-%m-%dT%H:%M:%SZ"))
        if current.empty:
            res.recommendation = "No current window data — drift not evaluated."
            return res

        for col, frozen in self._frozen.items():
            if col not in current.columns:
                continue
            res.columns.append(self._check_column(col, frozen, current[col]))

        psis = [c.psi for c in res.columns if np.isfinite(c.psi) and not c.low_variance]
        res.psi_mean = float(np.mean(psis)) if psis else 0.0
        if psis:
            best = max(res.columns, key=lambda c: c.psi if np.isfinite(c.psi) else -1)
            res.psi_max, res.psi_max_column = float(best.psi), best.column

        usable = [c for c in res.columns if not c.low_variance]
        drifted = [c for c in usable if c.drift_detected]
        res.drifted_columns = [c.column for c in drifted]
        res.n_drifted_fraction = len(drifted) / max(len(usable), 1)
        res.severity = max((c.severity for c in usable), key=lambda s: SEVERITY_ORDER.get(s, 0), default="none")
        res.drift_detected = (
            len(drifted) >= 2 or res.n_drifted_fraction >= 0.2 or res.severity == "severe"
        )
        res.recommendation = self._recommend(res)
        return res

    def _check_column(self, col: str, frozen: dict[str, Any], cur: pd.Series) -> ColumnDrift:
        out = ColumnDrift(column=col, kind=frozen["kind"],
                          reference_missing=frozen["missing"], current_missing=float(cur.isna().mean()))
        if frozen["kind"] == "numeric":
            ref_vals = self.reference[col].dropna().to_numpy(dtype=float)
            cur_vals = cur.dropna().to_numpy(dtype=float)
            if cur_vals.size < max(10, self.min_rows // 4):
                out.psi = out.ks_statistic = out.ks_p_value = float("nan")
                return out
            out.low_variance = bool(frozen.get("low_variance", False))
            edges = frozen["edges"]
            out.psi = self._psi(frozen["ref_probs"], cur_vals, edges)
            if ref_vals.size > 1 and cur_vals.size > 1:
                ks = stats.ks_2samp(ref_vals, cur_vals)
                out.ks_statistic, out.ks_p_value = float(ks.statistic), float(ks.pvalue)
            out.js_divergence = self._js(ref_vals, cur_vals, edges)
            out.reference_mean = float(frozen["mean"])
            out.current_mean = float(cur_vals.mean())
            sd = frozen["std"]
            out.reference_mean_shift_sd = float(abs(out.current_mean - out.reference_mean) / (sd if sd > 1e-9 else 1.0))
            out.drift_detected = (
                out.psi >= self.psi_moderate
                or (np.isfinite(out.ks_p_value) and out.ks_p_value < self.p_threshold)
                or (np.isfinite(out.reference_mean_shift_sd) and out.reference_mean_shift_sd > 0.5)
            ) and not out.low_variance
            out.severity = _severity(out.psi, self.psi_moderate, self.psi_severe) if not out.low_variance else "none"
            if out.severity == "none" and out.drift_detected:
                out.severity = "moderate"
        else:
            levels, probs = frozen["levels"], frozen["ref_probs"]
            out.psi = self._cat_psi(levels, probs, cur)
            out.chi2_statistic, out.chi2_p_value = self._chi2(levels, probs, cur)
            ref_js = np.maximum(probs, EPS)
            counts = cur.astype(str).value_counts()
            tot = max(counts.sum(), 1)
            cur_js = np.array([max(counts.get(lv, 0) / tot, EPS) for lv in levels])
            out.js_divergence = float(jensenshannon(ref_js, cur_js))
            out.drift_detected = out.psi >= self.psi_moderate or (
                np.isfinite(out.chi2_p_value) and out.chi2_p_value < self.p_threshold
            )
            out.severity = _severity(out.psi, self.psi_moderate, self.psi_severe)
            if out.severity == "none" and out.drift_detected:
                out.severity = "moderate"
        return out

    @staticmethod
    def _recommend(res: TabularDriftResult) -> str:
        if not res.drift_detected:
            return "Tabular features stable. No action."
        names = ", ".join(res.drifted_columns[:5])
        if res.severity == "severe":
            return (f"Severe tabular drift in {len(res.drifted_columns)} columns ({names}). "
                    "Largest PSI: "
                    f"{res.psi_max_column}={res.psi_max:.3f}. Retrain and re-check feature scaling.")
        return (f"{len(res.drifted_columns)} columns drifting ({names}). "
                f"Max PSI {res.psi_max_column}={res.psi_max:.3f}. Investigate the top feature before retraining.")
