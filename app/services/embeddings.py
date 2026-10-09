"""Lazy sentence-transformers wrapper producing normalised embeddings."""

from functools import lru_cache
from typing import TYPE_CHECKING

from app.config import get_settings

if TYPE_CHECKING:  # heavy import (torch) kept out of module import time
    from sentence_transformers import SentenceTransformer

_BATCH_SIZE = 64


@lru_cache
def get_model() -> "SentenceTransformer":
    """Load (once) and return the configured embedding model."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(get_settings().embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed ``texts`` into unit-length vectors (one list of floats per text).

    Other modules must call this as ``embeddings.embed_texts(...)`` (module
    attribute access) so tests can substitute a lightweight fake.
    """
    if not texts:
        return []
    vectors = get_model().encode(
        texts,
        batch_size=_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.tolist()
