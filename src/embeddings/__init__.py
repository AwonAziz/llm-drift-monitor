"""Embedding backends for drift measurement."""

from .encoder import (  # noqa: F401
    BaseEncoder,
    HashingEncoder,
    OllamaEncoder,
    SentenceTransformerEncoder,
    build_encoder,
    load_or_build_encoder,
)
