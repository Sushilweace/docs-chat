"""Document upload and listing endpoints."""

import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, status
from sqlalchemy import select

from app.api.deps import CurrentUser, DbSession
from app.config import get_settings
from app.models.db import Document, DocumentStatus
from app.models.schemas import DocumentOut
from app.services.ingestion import ingest_document

router = APIRouter(prefix="/documents", tags=["documents"])

_PDF_MAGIC = b"%PDF-"
_READ_CHUNK = 1024 * 1024


def _save_upload(file: UploadFile, destination: Path, max_bytes: int) -> None:
    """Stream ``file`` to ``destination`` enforcing a size cap and PDF signature.

    Raises:
        HTTPException: 400 if it is not a PDF, 413 if it exceeds ``max_bytes``.
        The partial file is removed on any failure.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        with destination.open("wb") as out:
            first = True
            while chunk := file.file.read(_READ_CHUNK):
                if first:
                    if not chunk.startswith(_PDF_MAGIC):
                        raise HTTPException(status.HTTP_400_BAD_REQUEST, "File is not a valid PDF")
                    first = False
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        f"File exceeds the {max_bytes // (1024 * 1024)} MB limit",
                    )
                out.write(chunk)
            if written == 0:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, "Uploaded file is empty")
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


@router.post("/upload", response_model=DocumentOut, status_code=status.HTTP_202_ACCEPTED)
def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    user: CurrentUser,
    db: DbSession,
) -> Document:
    """Accept a PDF and ingest it asynchronously.

    Returns ``202`` immediately with ``status="processing"``; poll
    ``GET /documents`` until it becomes ``ready`` (or ``failed``).
    """
    settings = get_settings()
    original_name = Path(file.filename or "").name
    if not original_name.lower().endswith(".pdf"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Only .pdf files are supported")

    stored_path = settings.upload_dir / str(user.id) / f"{uuid.uuid4().hex}.pdf"
    _save_upload(file, stored_path, settings.max_upload_bytes)

    document = Document(
        user_id=user.id,
        filename=original_name[:255],
        stored_path=str(stored_path),
        status=DocumentStatus.PROCESSING,
    )
    db.add(document)
    db.commit()

    background_tasks.add_task(ingest_document, document.id)
    return document


@router.get("", response_model=list[DocumentOut])
def list_documents(user: CurrentUser, db: DbSession) -> list[Document]:
    """List the current user's documents, newest first."""
    stmt = (
        select(Document)
        .where(Document.user_id == user.id)
        .order_by(Document.created_at.desc(), Document.id.desc())
    )
    return list(db.scalars(stmt))
