"""Per-user semantic retrieval with a relevance (distance) cutoff."""

import logging
from dataclasses import dataclass

from app.config import get_settings
from app.core.vectorstore import get_collection
from app.services import embeddings


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned from the vector store, with its provenance."""

    text: str
    source: str
    page: int
    distance: float


def retrieve(
    query: str,
    user_id: int,
    top_k: int | None = None,
    distance_threshold: float | None = None,
) -> list[RetrievedChunk]:
    """Return the most relevant chunks *owned by* ``user_id``.

    Isolation is enforced in the vector store query itself via
    ``where={"user_id": user_id}`` (and re-checked on the way out as defence in
    depth). Chunks whose cosine distance exceeds ``distance_threshold`` are
    dropped, so an empty list means "nothing relevant in this user's documents".

    Results are ordered best-first (smallest distance).
    """
    settings = get_settings()
    k = top_k or settings.top_k
    threshold = settings.distance_threshold if distance_threshold is None else distance_threshold

    query_vector = embeddings.embed_texts([query])[0]
    result = get_collection().query(
        query_embeddings=[query_vector],
        n_results=k,
        where={"user_id": user_id},
        include=["documents", "metadatas", "distances"],
    )

    docs = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    logger.info(
        "retrieval user=%s candidates=%d nearest=%s threshold=%.2f",
        user_id,
        len(distances),
        f"{min(distances):.3f}" if distances else "-",
        threshold,
    )

    chunks: list[RetrievedChunk] = []
    for text, meta, distance in zip(docs, metas, distances):
        if meta.get("user_id") != user_id or distance > threshold:
            continue
        chunks.append(
            RetrievedChunk(
                text=text,
                source=str(meta["source"]),
                page=int(meta["page"]),
                distance=float(distance),
            )
        )
    return chunks
