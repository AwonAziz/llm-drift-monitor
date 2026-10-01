"""
Dataset ingestion
-----------------
Primary source: **Banking77** (Casanueva et al., 2020) — 13,083 real customer
service queries annotated into 77 fine-grained banking intents. Distributed
under CC BY 4.0 and mirrored in the `task-specific-datasets` repository.

Why this dataset for an LLM monitoring demo:

* it is *real* production-shaped text, not synthetic Gaussian blobs;
* 77 intents is enough to build a convincing out-of-scope story — you can train
  a "launched" assistant on ~30 intents and let production traffic escape them;
* every sample carries a gold label, so LLM-as-judge scores can be validated
  against ground truth instead of being taken on faith.

The loader is defensive by design: it downloads once, caches the raw CSV under
``data/raw``, verifies a row count, and falls back to a small bundled corpus
(``data/bundled/banking77_mini.jsonl``) when the machine is offline. The
fallback is explicitly flagged in the run manifest so nobody mistakes it for
the real thing.
"""

from __future__ import annotations

import csv
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

MIN_VALID_ROWS = 5_000
BUNDLED_PATH = settings.DATA_DIR / "bundled" / "banking77_mini.jsonl"
LABEL_COL = "category"
TEXT_COL = "text"


@dataclass
class DatasetInfo:
    name: str
    rows: int
    intents: int
    source: str          # "remote" | "bundled"
    license: str
    path: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "rows": self.rows,
            "intents": self.intents,
            "source": self.source,
            "license": self.license,
            "path": self.path,
        }


def _cache_path(url: str) -> Path:
    return settings.RAW_DIR / url.rsplit("/", 1)[-1]


def _download(url: str, timeout: float = 60.0) -> pd.DataFrame | None:
    import requests

    dest = _cache_path(url)
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("Using cached dataset %s", dest.name)
        return pd.read_csv(dest)

    try:
        logger.info("Downloading %s", url)
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        logger.info("Saved %s (%.1f KB)", dest.name, len(resp.content) / 1024)
        return pd.read_csv(io.BytesIO(resp.content))
    except Exception as exc:  # noqa: BLE001 - network failure is expected offline
        logger.warning("Download failed (%s). Falling back to bundled corpus.", exc)
        return None


def _load_bundled() -> pd.DataFrame | None:
    if not BUNDLED_PATH.exists():
        logger.warning("No bundled corpus at %s", BUNDLED_PATH)
        return None
    rows = [json.loads(line) for line in BUNDLED_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    logger.warning(
        "Using BUNDLED offline corpus (%d rows, %d intents) — results are "
        "indicative, not the full Banking77 benchmark.",
        len(df), df[LABEL_COL].nunique(),
    )
    return df


def _load_cached() -> tuple[pd.DataFrame | None, str | None]:
    """Read the prepared parquet cache. This is the fully-offline happy path."""
    for path in sorted(settings.CACHE_DIR.glob("*_banking77.parquet")):
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:  # noqa: BLE001 - pyarrow missing or a corrupt file
            logger.warning("Ignoring unreadable cache %s (%s)", path.name, exc)
            continue
        if len(frame) >= MIN_VALID_ROWS:
            return frame, path.name.split("_", 1)[0]
    return None, None


def load_banking77(allow_download: bool = True) -> tuple[pd.DataFrame, DatasetInfo]:
    """Load Banking77 with a bundled-corpus fallback. Returns (df, info)."""
    df: pd.DataFrame | None = None
    source = "bundled"

    cached, cached_source = _load_cached()
    if cached is not None:
        logger.info("Using prepared dataset cache (%d rows)", len(cached))
        df, source = cached, cached_source
    else:
        df = _download(settings.DATASET_URL)
        if df is not None and len(df) < MIN_VALID_ROWS:
            logger.warning("Remote dataset only had %d rows; rejecting.", len(df))
            df = None
        if df is not None:
            source = "remote"
        if df is None:
            df = _download(settings.DATASET_URL_TEST, timeout=20.0)
            if df is not None and len(df) < MIN_VALID_ROWS:
                logger.warning("Remote test split only had %d rows; rejecting.", len(df))
                df = None
            if df is not None:
                source = "remote"

    if df is None:
        df = _load_bundled()
    if df is None:
        raise RuntimeError(
            "Banking77 unavailable. Either allow network access or place a corpus at "
            f"{BUNDLED_PATH}."
        )

    df = df[[TEXT_COL, LABEL_COL]].dropna().reset_index(drop=True)
    df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
    df = df[df[TEXT_COL].str.len() > 0].reset_index(drop=True)
    df = df.drop_duplicates(subset=[TEXT_COL, LABEL_COL]).reset_index(drop=True)

    parquet = settings.CACHE_DIR / f"{source}_banking77.parquet"
    raw = _cache_path(settings.DATASET_URL)
    if source == "remote" and not parquet.exists():
        path = str(raw if raw.exists() else BUNDLED_PATH)
    elif parquet.exists():
        path = str(parquet)
    else:
        path = str(BUNDLED_PATH)

    info = DatasetInfo(
        name=settings.DATASET_NAME,
        rows=len(df),
        intents=int(df[LABEL_COL].nunique()),
        source=source,
        license="CC BY 4.0 (Casanueva et al., 2020)",
        path=path,
    )
    logger.info("Dataset ready: %s rows / %d intents (%s)", f"{info.rows:,}", info.intents, info.source)
    return df, info


def write_dataset(df: pd.DataFrame, info: DatasetInfo) -> None:
    """Persist the loaded dataset so later runs are fully offline and reproducible."""
    path = settings.CACHE_DIR / f"{info.source}_banking77.parquet"
    try:
        df.to_parquet(path, index=False)
    except Exception as exc:  # noqa: BLE001 - parquet is optional (pyarrow missing)
        logger.debug("Parquet cache unavailable (%s); skipping.", exc)
        return
    info.path = str(path)
    write_manifest(info)


# ── Reference / traffic construction ────────────────────────────────────

def intent_catalog(df: pd.DataFrame) -> list[str]:
    return sorted(df[LABEL_COL].unique().tolist())


def build_reference(df: pd.DataFrame,
                    n: int = settings.REFERENCE_SIZE,
                    seed: int = 20240917) -> pd.DataFrame:
    """
    The reference window: what the assistant was trained on and validated against.

    Only *launch intents* are eligible. Sampling is proportional per intent so
    the reference mirrors a balanced production rollout rather than the long
    tail of the raw dataset.
    """
    launch = [i for i in settings.LAUNCH_INTENTS if i in set(df[LABEL_COL])]
    if not launch:
        raise RuntimeError("No launch intents present in the dataset.")

    rng = np.random.default_rng(seed)
    sub = df[df[LABEL_COL].isin(launch)]
    per_intent = max(6, n // len(launch))

    frames = []
    for intent in launch:
        pool = sub[sub[LABEL_COL] == intent][TEXT_COL].tolist()
        if len(pool) < 3:
            continue
        take = min(per_intent, len(pool))
        picks = rng.choice(len(pool), size=take, replace=take > len(pool))
        frames.append(pd.DataFrame({TEXT_COL: [pool[i] for i in picks], LABEL_COL: intent}))

    out = pd.concat(frames, ignore_index=True)
    if len(out) > n:
        out = out.sample(n=n, random_state=seed).reset_index(drop=True)
    out = out.rename(columns={LABEL_COL: "gold_intent"})
    out["in_scope"] = True
    out["source"] = "reference"
    logger.info("Reference window: %d rows across %d launch intents", len(out), out["gold_intent"].nunique())
    return out[[TEXT_COL, "gold_intent", "in_scope", "source"]]


def balanced_holdout(df: pd.DataFrame,
                     launch_intents: Sequence[str] | None = None,
                     per_intent: int = 8,
                     seed: int = 7) -> pd.DataFrame:
    """Held-out human-labelled set used to validate the LLM judge itself."""
    launch = list(launch_intents or settings.LAUNCH_INTENTS)
    rng = np.random.default_rng(seed)
    rows = []
    for intent in launch:
        pool = df.loc[df[LABEL_COL] == intent, TEXT_COL].tolist()
        if len(pool) < 2:
            continue
        idx = rng.choice(len(pool), size=min(per_intent, len(pool)), replace=False)
        rows += [{"text": pool[i], "gold_intent": intent} for i in idx]
    return pd.DataFrame(rows)


def write_manifest(info: DatasetInfo, extra: dict | None = None) -> Path:
    """Persist a run manifest so results are always attributable to a data source."""
    payload = info.to_dict()
    payload["launch_intents"] = list(settings.LAUNCH_INTENTS)
    payload["out_of_scope_intents"] = list(settings.OUT_OF_SCOPE_INTENTS)
    if extra:
        payload.update(extra)
    path = settings.ARTIFACT_DIR / "run_manifest.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def ensure_bundled_corpus(rows: Iterable[dict]) -> Path:
    """Helper used by scripts/bundled_corpus.py to (re)generate the offline sample."""
    BUNDLED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BUNDLED_PATH.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps({TEXT_COL: row[TEXT_COL], LABEL_COL: row[LABEL_COL]}) + "\n")
    return BUNDLED_PATH
