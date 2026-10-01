"""
Embedding backends
------------------
A drift detector is only as trustworthy as the representation it measures drift
*in*. This module exposes three interchangeable encoders behind one interface:

``SentenceTransformerEncoder``
    ``all-MiniLM-L6-v2`` (or any ST model). What you would run in production.

``HashingEncoder``
    Character/word n-gram features projected into a fixed dense space with a
    signed hashing trick, then reduced with a truncated SVD fit on the
    reference corpus. Zero downloads, microseconds per text, fully
    deterministic. It is the fallback that keeps this repo runnable on a plane,
    and it is honest about being a lexical approximation of semantic space.

``OllamaEncoder``
    Embeddings from a locally served model. Opt-in only (``LDM_EMBEDDING_BACKEND=ollama``):
    most small instruct models expose no embedding head, and a per-request HTTP
    round trip is far slower than a local ST model for no benefit.

Every encoder exposes ``name`` and ``version``; those strings are stamped onto
each drift measurement. If the encoder changes, previously recorded drift
numbers are not comparable - the telemetry store uses this to refuse to draw a
trend line across an encoder change, which is a real operational footgun.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from config import settings
from src.utils.logging import get_logger
from src.utils.stats import l2_normalize

logger = get_logger(__name__)


class BaseEncoder(ABC):
    """Common interface: encode raw text into a fixed-width float matrix."""

    name: str = "base"
    version: str = "0"
    dim: int = 0

    @abstractmethod
    def encode(self, texts: list[str], *, batch_size: int = 64) -> np.ndarray:
        """Encode a list of strings into an (n, dim) float32 array."""

    @property
    def signature(self) -> str:
        return f"{self.name}@{self.version}:{self.dim}d"

    def encode_frame(self, df: pd.DataFrame, text_col: str = "text") -> np.ndarray:
        return self.encode(df[text_col].astype(str).tolist())


class HashingEncoder(BaseEncoder):
    """
    Signed hashing-trick n-gram encoder + SVD.

    Character 3–5 grams give robustness to typos, spelling variation and
    non-native phrasing - exactly the properties you want when the question is
    "did the *shape* of the traffic change?" Word 1–2 grams add topical signal.
    """

    name = "hashing-ngram-svd"

    def __init__(
        self,
        dim: int = settings.EMBEDDING_DIM,
        n_features: int = settings.HASHING_FEATURES,
        word_ngram: tuple[int, int] = (1, 2),
        char_ngram: tuple[int, int] = (3, 5),
        seed: int = 13,
    ):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import HashingVectorizer
        from sklearn.pipeline import FeatureUnion

        self.dim = dim
        self.n_features = n_features
        self.version = f"n{self.n_features}-d{dim}-s{seed}"
        self.seed = seed

        self._char = HashingVectorizer(
            analyzer="char_wb", ngram_range=char_ngram, n_features=n_features,
            alternate_sign=True, norm="l2", lowercase=True,
        )
        self._word = HashingVectorizer(
            analyzer="word", ngram_range=word_ngram, n_features=n_features,
            alternate_sign=True, norm="l2", lowercase=True,
            token_pattern=r"(?u)\b\w+\b",
        )
        self._union = FeatureUnion([("char", self._char), ("word", self._word)])
        self._svd: TruncatedSVD | None = None
        self._rng = np.random.default_rng(seed)

    # -- fitting -------------------------------------------------------
    def fit_reference(self, texts: list[str], max_rows: int = 4000) -> HashingEncoder:
        """Fit the SVD basis on reference text so the projection is anchored."""
        from sklearn.decomposition import TruncatedSVD

        corpus = texts[:max_rows]
        if len(corpus) < 50:
            logger.warning("Reference corpus too small (%d) for a stable SVD basis.", len(corpus))
        X = self._union.transform(corpus)
        # A truncated SVD cannot have more components than the data supports, and
        # asking for more than n/3 makes the fit minutes long for no extra signal.
        n_components = int(min(self.dim, X.shape[1] - 1, max(2, len(corpus) // 3)))
        self._svd = TruncatedSVD(n_components=n_components, random_state=self.seed,
                                 algorithm="randomized", n_iter=4)
        self._svd.fit(X)
        self.dim = int(n_components)
        self.version = f"n{self.n_features}-d{self.dim}-s{self.seed}"
        logger.info("HashingEncoder SVD fitted: %d docs -> %d dims", len(corpus), self.dim)
        return self

    def save(self, path: Path) -> None:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"svd": self._svd, "dim": self.dim, "version": self.version,
                     "n_features": self.n_features}, path)

    @classmethod
    def load(cls, path: Path, **kwargs) -> HashingEncoder:
        import joblib

        blob = joblib.load(path)
        enc = cls(dim=blob["dim"], n_features=blob["n_features"], **kwargs)
        enc._svd = blob["svd"]
        enc.version = blob["version"]
        return enc

    # -- inference -----------------------------------------------------
    def encode(self, texts: list[str], *, batch_size: int = 512) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        X = self._union.transform(texts)
        if self._svd is not None:
            Z = self._svd.transform(X)
        else:
            # Unfitted: deterministic Gaussian projection seeded by the corpus.
            proj = self._rng.standard_normal((X.shape[1], self.dim)).astype(np.float32)
            Z = X @ proj
        return l2_normalize(Z).astype(np.float32)


class SentenceTransformerEncoder(BaseEncoder):
    """Wraps a sentence-transformers model; used when weights are available."""

    name = "sentence-transformers"

    def __init__(self, model_name: str = settings.EMBEDDING_MODEL, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        # 6.x renamed this accessor; support both so the encoder works on 2.x-5.x too.
        dim_fn = getattr(self._model, "get_embedding_dimension", None) or \
            self._model.get_sentence_embedding_dimension
        self.dim = int(dim_fn())
        self.version = self._resolve_version(model_name)

    def _resolve_version(self, model_name: str) -> str:
        """
        A stable identifier for the *weights* in use.

        Model cards expose a ``version`` key that is usually an environment dump
        rather than a revision, so we read the pinned commit when the hub gives
        us one and fall back to the library version. Either way the value is
        stable across processes, which is what the telemetry store needs to
        decide whether two drift numbers may be compared.
        """
        try:
            from sentence_transformers import __version__ as st_version

            lib = f"st{st_version}"
        except Exception:  # noqa: BLE001
            lib = "st?"
        revision = getattr(self._model, "model_card_data", None)
        rev = getattr(revision, "revision", None) if revision is not None else None
        return f"{model_name}@{rev[:8] if rev else lib}"

    def encode(self, texts: list[str], *, batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self._model.encode(
            texts, batch_size=batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)


class OllamaEncoder(BaseEncoder):
    """Embeddings from a local Ollama server."""

    name = "ollama"

    def __init__(self, model: str = settings.OLLAMA_EMBED_MODEL, host: str = settings.OLLAMA_HOST):
        import requests

        self.model = model
        self.host = host.rstrip("/")
        self._session = requests.Session()
        probe = self._session.get(f"{self.host}/api/tags", timeout=5)
        probe.raise_for_status()
        self.dim = 256
        self.version = model

    def encode(self, texts: list[str], *, batch_size: int = 1) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = []
        for text in texts:
            resp = self._session.post(
                f"{self.host}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=60,
            )
            resp.raise_for_status()
            out.append(resp.json()["embedding"])
        mat = np.asarray(out, dtype=np.float32)
        self.dim = int(mat.shape[1])
        return l2_normalize(mat)


# ── Resolution ───────────────────────────────────────────────────────────

def _try_sentence_transformer() -> BaseEncoder | None:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        logger.info("sentence-transformers not installed - using hashing encoder.")
        return None
    try:
        enc = SentenceTransformerEncoder()
        logger.info("Loaded sentence-transformers encoder: %s (%dd)", enc.model_name, enc.dim)
        return enc
    except Exception as exc:  # noqa: BLE001 - weights may be unavailable offline
        logger.warning("sentence-transformers unavailable (%s) - using hashing encoder.", exc)
        return None


def _try_ollama() -> BaseEncoder | None:
    try:
        enc = OllamaEncoder()
        logger.info("Loaded Ollama encoder: %s", enc.model_name if hasattr(enc, "model_name") else enc.model)
        return enc
    except Exception as exc:  # noqa: BLE001 - server may not be running
        logger.info("Ollama embeddings unavailable (%s).", exc)
        return None


def build_encoder(backend: str = settings.EMBEDDING_BACKEND,
                  reference_texts: list[str] | None = None) -> BaseEncoder:
    """
    Resolve the configured encoder. ``auto`` prefers semantic embeddings and
    degrades to the deterministic hashing encoder rather than failing - a
    monitoring system that will not boot offline is not a monitoring system.
    """
    backend = (backend or "auto").lower()

    if backend == "sentence-transformers":
        enc = _try_sentence_transformer()
        if enc is None:
            raise RuntimeError("sentence-transformers backend requested but unavailable.")
        return enc
    if backend == "ollama":
        enc = _try_ollama()
        if enc is None:
            raise RuntimeError("ollama embedding backend requested but server unreachable.")
        return enc
    if backend == "hashing":
        enc = HashingEncoder()
        return enc.fit_reference(reference_texts) if reference_texts else enc

    for factory in (_try_sentence_transformer,):
        enc = factory()
        if enc is not None:
            return enc

    logger.warning("Falling back to the hashing encoder (lexical, offline).")
    enc = HashingEncoder()
    return enc.fit_reference(reference_texts) if reference_texts else enc


def load_or_build_encoder(reference_texts: list[str] | None = None,
                          fingerprint: str | None = None) -> BaseEncoder:
    """
    Load a previously fitted hashing encoder when the reference fingerprint
    still matches, otherwise fit a fresh one. Keeps monitoring runs fast and
    - more importantly - keeps the projection stable across windows, without
    which every window looks like drift.
    """
    state_path = settings.ARTIFACT_DIR / "encoder.json"
    if state_path.exists() and reference_texts is not None:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if fingerprint and state.get("fingerprint") == fingerprint and state.get("backend") == "hashing":
            try:
                enc = HashingEncoder.load(settings.ARTIFACT_DIR / "hashing_encoder.joblib")
                logger.info("Reusing fitted hashing encoder (%s)", enc.version)
                return enc
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not reuse encoder (%s); refitting.", exc)

    enc = build_encoder(reference_texts=reference_texts)
    if isinstance(enc, HashingEncoder):
        enc.save(settings.ARTIFACT_DIR / "hashing_encoder.joblib")
        state_path.write_text(
            json.dumps({"backend": "hashing", "version": enc.version, "fingerprint": fingerprint}, indent=2),
            encoding="utf-8",
        )
    else:
        state_path.write_text(
            json.dumps({"backend": enc.name, "version": enc.version, "fingerprint": fingerprint}, indent=2),
            encoding="utf-8",
        )
    return enc
