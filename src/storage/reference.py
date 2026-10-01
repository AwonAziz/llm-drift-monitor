"""
Reference snapshot
------------------
A single, versioned artifact describing the distribution the model was
*healthy* at: the reference embeddings, per-intent centroids, calibration
metadata and the quality bars the champion cleared on its validation set.

Every detector loads this rather than recomputing a reference from whatever
data happens to be lying around. Without a frozen reference you cannot answer
the only question that matters during an incident — "is this worse than
normal?" — because "normal" keeps moving.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import settings
from src.utils.logging import get_logger
from src.utils.stats import json_safe, l2_normalize

logger = get_logger(__name__)

NPZ_KEYS = ("embeddings", "centroids", "intent_ids", "mu", "sigma", "sample_index")


@dataclass
class ReferenceSnapshot:
    """Frozen healthy-state description of the monitored system."""

    embeddings: np.ndarray                 # (n_ref, dim) L2-normalised
    centroids: np.ndarray                  # (n_intents, dim) per-intent means
    intent_ids: list[str]
    mu: np.ndarray                         # global embedding mean
    sigma: np.ndarray                      # global embedding std (per dim)
    intent_counts: dict[str, int] = field(default_factory=dict)
    calib_bins: np.ndarray | None = None   # (n_bins, 2) -> (confidence, accuracy)
    quality: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    # ── derived statistics used by the detectors ────────────────────
    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])

    @property
    def n(self) -> int:
        return int(self.embeddings.shape[0])

    def max_pairwise_sqdist(self, idx: np.ndarray) -> np.ndarray:
        """Squared distances between a batch and every reference vector.

        Returned as (len(idx), n_ref); used for the k-NN novelty statistic and
        for nearest-intent-centroid assignment.
        """
        a = self.embeddings[idx]
        # ||a||^2 = 1 after L2 normalisation, so this stays numerically tight.
        return 2.0 - 2.0 * (a @ self.embeddings.T)

    def kth_neighbour_distance(self, vectors: np.ndarray, k: int = 5,
                               chunk: int = 256, exclude_self: bool = False) -> np.ndarray:
        """
        Mean distance to the k nearest reference points, per vector.

        ``exclude_self`` drops the zero-distance match when the query points are
        themselves reference points. Skipping it silently deflates the reference
        distribution (a point's nearest neighbour is itself at distance 0), which
        makes every *new* point look novel and drives the out-of-scope rate to
        100%. This is the single most common bug in k-NN novelty scoring.
        """
        v = l2_normalize(vectors)
        out = np.zeros(v.shape[0], dtype=np.float64)
        for start in range(0, v.shape[0], chunk):
            block = v[start : start + chunk]
            d2 = 2.0 - 2.0 * (block @ self.embeddings.T)
            np.maximum(d2, 0.0, out=d2)
            if exclude_self:
                rows = np.arange(start, start + block.shape[0])
                d2[np.arange(block.shape[0]), np.clip(rows, 0, d2.shape[1] - 1)] = np.inf
            kk = min(k, d2.shape[1] - 1) if d2.shape[1] > 1 else 1
            part = np.partition(d2, kk - 1, axis=1)[:, :kk]
            out[start : start + block.shape[0]] = np.sqrt(part).mean(axis=1)
        return out

    def assign_intents(self, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Nearest-centroid intent + cosine margin, the retrieval-style baseline."""
        v = l2_normalize(vectors)
        sims = v @ l2_normalize(self.centroids).T
        best = np.argmax(sims, axis=1)
        if sims.shape[1] > 1:
            part = np.partition(sims, -2, axis=1)
            margin = part[:, -1] - part[:, -2]
        else:
            margin = np.zeros(v.shape[0])
        return best, margin

    # ── persistence ─────────────────────────────────────────────────
    def save(self, path: Path = settings.REFERENCE_SNAPSHOT) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            embeddings=self.embeddings.astype(np.float32),
            centroids=self.centroids.astype(np.float32),
            intent_ids=np.array(self.intent_ids, dtype=object),
            mu=self.mu.astype(np.float32),
            sigma=self.sigma.astype(np.float32),
            calib_bins=self.calib_bins if self.calib_bins is not None else np.zeros((0, 2)),
        )
        sidecar = path.with_suffix(".json")
        sidecar.write_text(
            json.dumps({
                "intent_counts": self.intent_counts,
                "quality": json_safe(self.quality),
                "meta": json_safe(self.meta),
            }, indent=2),
            encoding="utf-8",
        )
        logger.info("Reference snapshot saved -> %s (%d vectors, %d dims)", path.name, self.n, self.dim)
        return path

    @classmethod
    def load(cls, path: Path = settings.REFERENCE_SNAPSHOT) -> ReferenceSnapshot:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"No reference snapshot at {path}. Run scripts/bootstrap.py first."
            )
        blob = np.load(path, allow_pickle=True)
        sidecar = path.with_suffix(".json")
        payload = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
        calib = blob["calib_bins"]
        return cls(
            embeddings=blob["embeddings"].astype(np.float32),
            centroids=blob["centroids"].astype(np.float32),
            intent_ids=[str(x) for x in blob["intent_ids"].tolist()],
            mu=blob["mu"].astype(np.float64),
            sigma=blob["sigma"].astype(np.float64),
            calib_bins=calib if calib.size else None,
            intent_counts=payload.get("intent_counts", {}),
            quality=payload.get("quality", {}),
            meta=payload.get("meta", {}),
        )

    @classmethod
    def exists(cls, path: Path = settings.REFERENCE_SNAPSHOT) -> bool:
        return Path(path).exists()

    # ── construction ────────────────────────────────────────────────
    @classmethod
    def build(
        cls,
        df: pd.DataFrame,
        embeddings: np.ndarray,
        *,
        intent_col: str = "gold_intent",
        confidences: np.ndarray | None = None,
        correctness: np.ndarray | None = None,
        quality: dict[str, float] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> ReferenceSnapshot:
        """Build a snapshot from a reference dataframe and its embeddings."""
        emb = l2_normalize(np.asarray(embeddings, dtype=np.float32))
        intents = df[intent_col].astype(str).tolist()
        uniq = sorted(set(intents))
        index = {name: i for i, name in enumerate(uniq)}

        centroids = np.zeros((len(uniq), emb.shape[1]), dtype=np.float64)
        counts: dict[str, int] = {}
        for name in uniq:
            mask = np.array([t == name for t in intents])
            counts[name] = int(mask.sum())
            centroids[index[name]] = emb[mask].mean(axis=0)
        centroids = l2_normalize(centroids)

        calib_bins = None
        if confidences is not None and correctness is not None:
            calib_bins = reliability_curve(
                np.asarray(confidences, dtype=float),
                np.asarray(correctness, dtype=float),
                n_bins=10,
            )

        return cls(
            embeddings=emb.astype(np.float32),
            centroids=centroids.astype(np.float32),
            intent_ids=uniq,
            mu=emb.mean(axis=0).astype(np.float64),
            sigma=np.maximum(emb.std(axis=0), 1e-6).astype(np.float64),
            intent_counts=counts,
            calib_bins=calib_bins,
            quality=quality or {},
            meta=meta or {},
        )


def reliability_curve(confidences: np.ndarray, correctness: np.ndarray,
                      n_bins: int = 10) -> np.ndarray:
    """Classic equal-width reliability curve → (n_bins, 2) array of (conf, acc)."""
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correctness, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, n_bins - 1)
    out = np.full((n_bins, 2), np.nan)
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() >= 3:
            out[b, 0] = conf[mask].mean()
            out[b, 1] = corr[mask].mean()
    return out
