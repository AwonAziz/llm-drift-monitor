"""Shared pytest fixtures. The whole suite runs offline in well under a minute."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

from src.drift import EmbeddingDriftDetector, TabularDriftDetector  # noqa: E402
from src.embeddings import HashingEncoder  # noqa: E402
from src.storage.reference import ReferenceSnapshot  # noqa: E402
from src.storage.telemetry import TelemetryStore  # noqa: E402


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(7)


@pytest.fixture(scope="session")
def encoder() -> HashingEncoder:
    """The offline hashing encoder — deterministic and dependency-free."""
    corpus = [f"synthetic reference sentence number {i} about cards and payments"
              for i in range(400)]
    return HashingEncoder(dim=48, n_features=1 << 12).fit_reference(corpus)


@pytest.fixture(scope="session")
def cloud(encoder):
    """
    Text clouds with a controllable vocabulary shift.

    ``offset`` is the fraction of samples drawn from a disjoint vocabulary, so
    0.0 is the reference distribution, 1.0 is a fully different world, and 0.25
    is a quarter-new traffic mix. Working in vocabulary rather than in raw
    numeric coordinates matters: after L2 normalisation an offset in one
    coordinate is largely cancelled out, and a "shifted" fixture that the
    embedding cannot see is a fixture that tests nothing.
    """
    ref_words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
    new_words = ["zulu", "yankee", "xray", "whiskey", "victor", "uniform", "tango", "sierra"]

    def make(n: int, offset: float, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(1234 + int(round(offset * 1000)) + 7919 * seed)
        texts = []
        for _ in range(n):
            vocab = new_words if rng.random() < offset else ref_words
            texts.append(" ".join(str(w) for w in rng.choice(vocab, size=4)))
        return encoder.encode(texts)

    return make


@pytest.fixture()
def snapshot(cloud) -> ReferenceSnapshot:
    emb = cloud(500, 0.0)
    labels = ["a"] * 250 + ["b"] * 250
    frame = pd.DataFrame({"text": [str(i) for i in range(len(labels))],
                          "gold_intent": labels, "in_scope": True})
    return ReferenceSnapshot.build(frame, emb, intent_col="gold_intent")

@pytest.fixture()
def detector(snapshot) -> EmbeddingDriftDetector:
    return EmbeddingDriftDetector(snapshot, seed=5, max_current=400)


@pytest.fixture()
def tabular_frame() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    return pd.DataFrame({
        "text_length": rng.normal(120, 20, 600),
        "word_count": rng.normal(20, 4, 600),
        "unique_word_ratio": rng.beta(5, 2, 600),
        "digit_ratio": rng.uniform(0, 0.2, 600),
        "upper_ratio": rng.beta(2, 3, 600),
        "punct_ratio": rng.uniform(0, 0.1, 600),
        "is_question": rng.integers(0, 2, 600).astype(float),
        "emoji_flag": rng.integers(0, 2, 600).astype(float),
        "has_currency": rng.integers(0, 2, 600).astype(float),
    })


@pytest.fixture()
def tabular_detector(tabular_frame) -> TabularDriftDetector:
    return TabularDriftDetector(tabular_frame, min_rows=40)


@pytest.fixture()
def store(tmp_path) -> TelemetryStore:
    return TelemetryStore(tmp_path / "telemetry.db")


@pytest.fixture()
def tiny_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    intents = [f"intent_{i}" for i in range(6)]
    rows = []
    for intent in intents:
        for k in range(40):
            rows.append({
                "text": f"{intent.replace('_', ' ')} message {k} about {' '.join(['word'] * (k % 5 + 1))}",
                "gold_intent": intent,
                "in_scope": intent.startswith("intent_0"),
            })
    frame = pd.DataFrame(rows).sample(frac=1.0, random_state=1).reset_index(drop=True)
    frame.loc[rng.random(len(frame)) < 0.3, "gold_intent"] = None
    return frame