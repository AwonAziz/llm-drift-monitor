"""Shared numeric utilities.

These are the primitives every detector is built from, kept in one place so
that definitions stay consistent across embedding drift, output drift and the
judge evaluators.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np


def json_safe(obj: Any) -> Any:
    """Recursively convert numpy / dataclass-ish values into JSON-safe types."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        obj = obj.item()
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if hasattr(obj, "to_dict") and not isinstance(obj, type):
        return json_safe(obj.to_dict())
    if hasattr(obj, "__dataclass_fields__"):
        from dataclasses import asdict

        return json_safe(asdict(obj))
    return obj


def stable_hash(text: str) -> int:
    """Deterministic 64-bit hash (Python's hash() is salted per process)."""
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = x.std()
    return (x - x.mean()) / (sd if sd > 1e-12 else 1.0)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        n = np.linalg.norm(x)
        return x / (n if n > eps else 1.0)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, eps, None)


def mean_ci(values: Sequence[float], z: float = 1.96) -> tuple[float, float, float]:
    """Return (mean, half-width, std-error) of a sample."""
    arr = np.asarray([v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))],
                     dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(arr.mean())
    if arr.size < 2:
        return mean, 0.0, 0.0
    se = float(arr.std(ddof=1) / math.sqrt(arr.size))
    return mean, z * se, se


def ewma(values: Sequence[float], alpha: float = 0.3) -> list[float]:
    """Exponentially weighted moving average (inclusive of the first point)."""
    out: list[float] = []
    cur: float | None = None
    for v in values:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            out.append(cur if cur is not None else float("nan"))
            continue
        cur = float(v) if cur is None else alpha * float(v) + (1 - alpha) * cur
        out.append(cur)
    return out


def rolling_median(values: Sequence[float], window: int = 5) -> list[float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return []
    out: list[float] = []
    for i in range(arr.size):
        lo = max(0, i - window + 1)
        chunk = arr[lo : i + 1]
        chunk = chunk[~np.isnan(chunk)]
        out.append(float(np.median(chunk)) if chunk.size else float("nan"))
    return out


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b not in (0, 0.0) else default


def flatten(seq: Iterable[Any]) -> list[Any]:
    return list(seq)
