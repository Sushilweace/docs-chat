"""PDF ingestion: parse per page, chunk, embed, and store in Chroma."""

import logging
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from app.config import get_settings
from app.core.database import SessionLocal
from app.core.vectorstore import get_collection
from app.models.db import Document, DocumentStatus
from app.services import embeddings

logger = logging.getLogger(__name__)

_CHROMA_ADD_BATCH = 256


@dataclass(frozen=True)
class PageText:
    """Extracted text of a single PDF page (``page`` is 1-based)."""

    page: int
    text: str


def normalise_text(text: str) -> str:
    """Collapse all whitespace runs (and strip NULs) into single spaces."""
    return " ".join(text.replace("\x00", " ").split())


def extract_pages(path: Path) -> tuple[list[PageText], int]:
    """Parse a PDF and return ``(non-empty pages, total page count)``.

    Raises:
        ValueError: if the PDF is encrypted with a non-empty password.
    """
    reader = PdfReader(str(path))
    if reader.is_encrypted and not reader.decrypt(""):
        raise ValueError("PDF is password-protected")

    pages: list[PageText] = []
    for number, page in enumerate(reader.pages, start=1):
        text = normalise_text(page.extract_text() or "")
        if text:
            pages.append(PageText(page=number, text=text))
    return pages, len(reader.pages)


def chunk_text(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    """Split ``text`` into overlapping character windows.

    Windows are ``size`` characters long and consecutive windows share roughly
    ``overlap`` characters. Boundaries are nudged to whitespace so words are not
    cut in half. Text is assumed to be already whitespace-normalised.

    Raises:
        ValueError: if ``overlap`` is not smaller than half of ``size``.
    """
    if size <= 0 or not 0 <= overlap < size // 2:
        raise ValueError("require size > 0 and 0 <= overlap < size/2")

    text = text.strip()
    if len(text) <= size:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # Prefer to end on a space, but never shrink below half a window.
            space = text.rfind(" ", start + size // 2, end)
            if space != -1:
                end = space
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
        # Don't begin the next chunk in the middle of a word.
        if not text[start - 1].isspace():
            next_space = text.find(" ", start, end)
            if next_space != -1:
                start = next_space + 1
    return chunks


def _mark_failed(db_session, document: Document, error: str) -> None:
    document.status = DocumentStatus.FAILED
    document.error = error[:1000]
    db_session.commit()


def ingest_document(document_id: int) -> None:
    """Background task: ingest one uploaded document into the vector store.

    Opens its *own* DB session (the request's session is closed by the time a
    background task runs). On success the document becomes ``ready``; on any
    failure it becomes ``failed`` and partial vectors are removed.
    """
    settings = get_settings()
    collection = get_collection()

    with SessionLocal() as db:
        document = db.get(Document, document_id)
        if document is None:
            logger.warning("Ingestion skipped: document %s no longer exists", document_id)
            return

        try:
            pages, page_count = extract_pages(Path(document.stored_path))

            ids: list[str] = []
            texts: list[str] = []
            metadatas: list[dict] = []
            for page in pages:
                pieces = chunk_text(page.text, settings.chunk_size, settings.chunk_overlap)
                for index, piece in enumerate(pieces):
                    ids.append(f"{document.id}:{page.page}:{index}")
                    texts.append(piece)
                    metadatas.append(
                        {
                            "user_id": document.user_id,
                            "source": document.filename,
                            "page": page.page,
                            "document_id": document.id,
                            "chunk_index": index,
                        }
                    )

            if not texts:
                document.page_count = page_count
                _mark_failed(db, document, "No extractable text found (scanned or image-only PDF?)")
                return

            vectors = embeddings.embed_texts(texts)
            for i in range(0, len(ids), _CHROMA_ADD_BATCH):
                s = slice(i, i + _CHROMA_ADD_BATCH)
                collection.add(
                    ids=ids[s], documents=texts[s], embeddings=vectors[s], metadatas=metadatas[s]
                )

            document.page_count = page_count
            document.chunk_count = len(texts)
            document.status = DocumentStatus.READY
            document.error = None
            db.commit()
            logger.info("Ingested document %s: %d chunks", document.id, len(texts))
        except Exception as exc:  # noqa: BLE001 - background task must never raise
            logger.exception("Ingestion failed for document %s", document_id)
            db.rollback()
            try:
                collection.delete(where={"document_id": document_id})  # drop partial vectors
            except Exception:  # noqa: BLE001
                logger.exception("Cleanup failed for document %s", document_id)
            _mark_failed(db, document, f"{type(exc).__name__}: {exc}")
