"""Persistent ChromaDB client and collection accessors."""

from functools import lru_cache

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from app.config import get_settings


@lru_cache
def get_chroma_client() -> ClientAPI:
    """Return a process-wide persistent Chroma client (data kept on disk)."""
    settings = get_settings()
    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(settings.chroma_path))


def get_collection() -> Collection:
    """Return the documents collection, creating it on first use.

    The collection uses *cosine* distance so that ``distance_threshold`` has a
    consistent meaning. We always supply our own embeddings, so no Chroma
    embedding function is attached.
    """
    settings = get_settings()
    return get_chroma_client().get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,
    )
