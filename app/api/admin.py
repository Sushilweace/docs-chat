"""Admin panel: read-only, paginated views of every table and of the vector store.

Access is limited to the emails listed in ``ADMIN_EMAILS`` (everyone else gets 403, and
with the setting empty nobody is an admin). Nothing here writes data, and password
hashes are never selected.

Every list endpoint returns the same envelope::

    {"items": [...], "total": 123, "page": 1, "page_size": 20, "pages": 7}
"""

import math
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Query
from fastapi.responses import FileResponse
from sqlalchemy import Select, func, select

from app.api.deps import AdminUser, DbSession
from app.core.vectorstore import get_collection
from app.models.db import Conversation, Document, Message, User

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

router = APIRouter(prefix="/admin", tags=["admin"])

Page = Annotated[int, Query(ge=1, description="1-based page number")]
PageSize = Annotated[int, Query(ge=1, le=100, description="Rows per page")]
Search = Annotated[str | None, Query(max_length=200, description="Case-insensitive text filter")]


def _envelope(items: list[dict[str, Any]], total: int, page: int, page_size: int) -> dict[str, Any]:
    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": max(1, math.ceil(total / page_size)),
    }


def _paginate(db, stmt: Select, page: int, page_size: int) -> dict[str, Any]:
    """Run ``stmt`` (labelled columns, with its ORDER BY) for one page, plus the total count."""
    total = db.scalar(select(func.count()).select_from(stmt.order_by(None).subquery())) or 0
    rows = db.execute(stmt.limit(page_size).offset((page - 1) * page_size)).mappings().all()
    return _envelope([dict(row) for row in rows], total, page, page_size)


@router.get("", include_in_schema=False)
def admin_page() -> FileResponse:
    """The admin UI shell. It holds no data: every request it makes is checked server-side."""
    return FileResponse(STATIC_DIR / "admin.html", headers={"Cache-Control": "no-cache"})


@router.get("/data/stats")
def stats(_: AdminUser, db: DbSession) -> dict[str, int]:
    """Row counts for every table and the number of chunks in the vector store."""
    count = lambda model: db.scalar(select(func.count()).select_from(model)) or 0  # noqa: E731
    return {
        "users": count(User),
        "documents": count(Document),
        "conversations": count(Conversation),
        "messages": count(Message),
        "chunks": get_collection().count(),
    }


@router.get("/data/users")
def users(_: AdminUser, db: DbSession, page: Page = 1, page_size: PageSize = 20, q: Search = None):
    """Accounts (never including password hashes) with their document / chat counts."""
    documents = select(func.count(Document.id)).where(Document.user_id == User.id).scalar_subquery()
    conversations = (
        select(func.count(Conversation.id)).where(Conversation.user_id == User.id).scalar_subquery()
    )
    stmt = select(
        User.id.label("id"),
        User.email.label("email"),
        User.created_at.label("created_at"),
        documents.label("documents"),
        conversations.label("conversations"),
    ).order_by(User.id)
    if q:
        stmt = stmt.where(User.email.icontains(q, autoescape=True))
    return _paginate(db, stmt, page, page_size)


@router.get("/data/documents")
def documents(
    _: AdminUser,
    db: DbSession,
    page: Page = 1,
    page_size: PageSize = 20,
    q: Search = None,
    user_id: Annotated[int | None, Query(ge=1)] = None,
):
    """Uploaded PDFs and their ingestion status."""
    stmt = (
        select(
            Document.id.label("id"),
            Document.user_id.label("user_id"),
            User.email.label("user_email"),
            Document.filename.label("filename"),
            Document.status.label("status"),
            Document.page_count.label("page_count"),
            Document.chunk_count.label("chunk_count"),
            Document.error.label("error"),
            Document.created_at.label("created_at"),
        )
        .join(User, User.id == Document.user_id)
        .order_by(Document.id.desc())
    )
    if q:
        stmt = stmt.where(Document.filename.icontains(q, autoescape=True))
    if user_id:
        stmt = stmt.where(Document.user_id == user_id)
    result = _paginate(db, stmt, page, page_size)
    for row in result["items"]:  # enum -> plain string for JSON
        row["status"] = getattr(row["status"], "value", row["status"])
    return result


@router.get("/data/conversations")
def conversations(
    _: AdminUser,
    db: DbSession,
    page: Page = 1,
    page_size: PageSize = 20,
    q: Search = None,
    user_id: Annotated[int | None, Query(ge=1)] = None,
):
    """Chats with their message counts (open one with the messages endpoint)."""
    message_count = (
        select(func.count(Message.id)).where(Message.conversation_id == Conversation.id).scalar_subquery()
    )
    stmt = (
        select(
            Conversation.id.label("id"),
            Conversation.user_id.label("user_id"),
            User.email.label("user_email"),
            Conversation.title.label("title"),
            message_count.label("messages"),
            Conversation.created_at.label("created_at"),
        )
        .join(User, User.id == Conversation.user_id)
        .order_by(Conversation.id.desc())
    )
    if q:
        stmt = stmt.where(Conversation.title.icontains(q, autoescape=True))
    if user_id:
        stmt = stmt.where(Conversation.user_id == user_id)
    return _paginate(db, stmt, page, page_size)


@router.get("/data/messages")
def messages(
    _: AdminUser,
    db: DbSession,
    page: Page = 1,
    page_size: PageSize = 20,
    q: Search = None,
    conversation_id: Annotated[int | None, Query(ge=1)] = None,
    user_id: Annotated[int | None, Query(ge=1)] = None,
):
    """Chat messages (newest first; within one conversation, oldest first)."""
    stmt = (
        select(
            Message.id.label("id"),
            Message.conversation_id.label("conversation_id"),
            Conversation.user_id.label("user_id"),
            User.email.label("user_email"),
            Message.role.label("role"),
            Message.content.label("content"),
            Message.sources.label("sources"),
            Message.created_at.label("created_at"),
        )
        .join(Conversation, Conversation.id == Message.conversation_id)
        .join(User, User.id == Conversation.user_id)
    )
    stmt = stmt.order_by(Message.id if conversation_id else Message.id.desc())
    if q:
        stmt = stmt.where(Message.content.icontains(q, autoescape=True))
    if conversation_id:
        stmt = stmt.where(Message.conversation_id == conversation_id)
    if user_id:
        stmt = stmt.where(Conversation.user_id == user_id)
    result = _paginate(db, stmt, page, page_size)
    for row in result["items"]:
        row["role"] = getattr(row["role"], "value", row["role"])
    return result


@router.get("/data/chunks")
def chunks(
    _: AdminUser,
    page: Page = 1,
    page_size: PageSize = 20,
    q: Search = None,
    user_id: Annotated[int | None, Query(ge=1)] = None,
    document_id: Annotated[int | None, Query(ge=1)] = None,
):
    """Text chunks stored in ChromaDB (the vectors themselves are not returned)."""
    clauses = []
    if user_id:
        clauses.append({"user_id": user_id})
    if document_id:
        clauses.append({"document_id": document_id})
    where = None if not clauses else clauses[0] if len(clauses) == 1 else {"$and": clauses}
    where_document = {"$contains": q} if q else None

    collection = get_collection()
    if where is None and where_document is None:
        total = collection.count()
    else:  # Chroma has no filtered count; ids are cheap to list at this scale
        total = len(collection.get(where=where, where_document=where_document, include=[])["ids"])

    got = collection.get(
        where=where,
        where_document=where_document,
        limit=page_size,
        offset=(page - 1) * page_size,
        include=["metadatas", "documents"],
    )
    items = [
        {
            "id": chunk_id,
            "user_id": meta.get("user_id"),
            "document_id": meta.get("document_id"),
            "source": meta.get("source"),
            "page": meta.get("page"),
            "chunk_index": meta.get("chunk_index"),
            "chars": len(text),
            "text": text,
        }
        for chunk_id, meta, text in zip(got["ids"], got["metadatas"], got["documents"])
    ]
    return _envelope(items, total, page, page_size)
