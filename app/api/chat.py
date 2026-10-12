"""Chat endpoint: retrieval-grounded answers streamed over Server-Sent Events.

SSE event types (``data`` is always JSON):

* ``sources`` - ``[{index, source, page, distance, snippet}, ...]`` (sent first)
* ``token``   - ``{"text": "<delta>"}`` (repeated)
* ``done``    - ``{"conversation_id": <int>, "user_message_id": <int>, "assistant_message_id": <int>}``
* ``error``   - ``{"detail": "<message>"}`` (terminates the stream)
"""

import json
import logging
from collections.abc import Iterator

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy import delete, func, select

from app.api.deps import CurrentUser, DbSession
from app.config import get_settings
from app.core.database import SessionLocal
from app.core.rate_limit import limiter
from app.models.db import Conversation, Message, MessageRole
from app.models.schemas import ChatRequest, SourceOut
from app.services import llm, retrieval
from app.services.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

_SNIPPET_CHARS = 240


def _sse(event: str, data: object) -> str:
    """Encode one SSE frame. JSON keeps newlines inside ``data`` safe."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _to_sources(chunks: list[RetrievedChunk]) -> list[SourceOut]:
    """Number chunks 1..n so they match the ``[n]`` markers used in the prompt."""
    return [
        SourceOut(
            index=i,
            source=c.source,
            page=c.page,
            distance=round(c.distance, 4),
            snippet=c.text[:_SNIPPET_CHARS],
        )
        for i, c in enumerate(chunks, start=1)
    ]


def _load_history(
    db,
    conversation_id: int | None,
    user_id: int,
    limit: int,
    before_message_id: int | None = None,
) -> list[llm.HistoryTurn]:
    """Return the last ``limit`` messages of a conversation, oldest first.

    When ``before_message_id`` is given (edit mode) only messages *older* than it are
    used, so the model never sees the turns that the edit is about to replace.

    Raises:
        HTTPException: 404 if the conversation (or edited message) does not exist or isn't
            the user's; 400 if the edited message is not one of the user's own questions.
    """
    if conversation_id is None:
        return []
    conversation = db.get(Conversation, conversation_id)
    if conversation is None or conversation.user_id != user_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found")

    if before_message_id is not None:
        target = db.get(Message, before_message_id)
        if target is None or target.conversation_id != conversation_id:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Message not found")
        if target.role != MessageRole.USER:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Only your own questions can be edited")

    stmt = select(Message).where(Message.conversation_id == conversation_id)
    if before_message_id is not None:
        stmt = stmt.where(Message.id < before_message_id)
    stmt = stmt.order_by(Message.id.desc()).limit(limit)
    recent = list(db.scalars(stmt))[::-1]
    return [llm.HistoryTurn(role=m.role.value, content=m.content) for m in recent]

def _persist_exchange(
    user_id: int,
    conversation_id: int | None,
    question: str,
    answer: str,
    sources: list[SourceOut],
    edit_message_id: int | None = None,
) -> tuple[int, int, int]:
    """Save the user question and assistant answer; create the conversation if new.

    Returns ``(conversation_id, user_message_id, assistant_message_id)``.

    Everything happens in one transaction, so history never contains an unanswered
    question. In edit mode the replaced question and every later message are deleted in
    that same transaction, i.e. only *after* the new answer succeeded, so a failed edit
    leaves the original conversation untouched. Uses its own session because it runs while
    the response is streaming, after the request-scoped session may have been closed.
    """
    with SessionLocal() as db:
        if conversation_id is None:
            conversation = Conversation(user_id=user_id, title=question[:80])
            db.add(conversation)
            db.flush()
        else:
            conversation = db.get(Conversation, conversation_id)
            assert conversation is not None and conversation.user_id == user_id

            if edit_message_id is not None:
                older = db.scalar(
                    select(func.count())
                    .select_from(Message)
                    .where(Message.conversation_id == conversation.id, Message.id < edit_message_id)
                )
                db.execute(
                    delete(Message).where(
                        Message.conversation_id == conversation.id, Message.id >= edit_message_id
                    )
                )
                if not older:  # the very first question was edited: retitle the chat
                    conversation.title = question[:80]

        user_msg = Message(conversation_id=conversation.id, role=MessageRole.USER, content=question)
        assistant_msg = Message(
            conversation_id=conversation.id,
            role=MessageRole.ASSISTANT,
            content=answer,
            # Citations only (file/page); excerpt text isn't duplicated into history.
            sources=[s.model_dump(exclude={"snippet"}) for s in sources],
        )
        db.add_all([user_msg, assistant_msg])
        db.commit()
        return conversation.id, user_msg.id, assistant_msg.id


@router.post(
    "",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}, "description": "SSE stream"}},
)
@limiter.limit(get_settings().chat_rate_limit)
def chat(request: Request, payload: ChatRequest, user: CurrentUser, db: DbSession) -> StreamingResponse:
    """Answer a question from the user's own documents, streamed as SSE.

    Flow: load history -> retrieve (filtered to this user) -> if nothing is
    relevant, reply "not in your documents" *without calling the LLM* -> else
    stream the grounded answer -> persist the exchange. Plain greetings
    ("hi", "thanks") get a canned reply and skip retrieval and the LLM entirely.
    """
    settings = get_settings()
    history = _load_history(
        db, payload.conversation_id, user.id, settings.history_messages, payload.edit_message_id
    )
    canned = llm.GREETING_MESSAGE if llm.is_greeting(payload.question) else None
    chunks = [] if canned else retrieval.retrieve(payload.question, user_id=user.id)
    logger.info(
        "chat user=%s conversation=%s%s history=%d relevant_chunks=%d%s",
        user.id,
        payload.conversation_id,
        f" edit_of={payload.edit_message_id}" if payload.edit_message_id else "",
        len(history),
        len(chunks),
        " -> greeting, replying without retrieval or the LLM"
        if canned
        else ("" if chunks else " -> answering 'not in documents' without calling the LLM"),
    )
    sources = _to_sources(chunks)
    user_id, conversation_id, question = user.id, payload.conversation_id, payload.question
    edit_message_id = payload.edit_message_id

    def event_stream() -> Iterator[str]:
        yield _sse("sources", [s.model_dump() for s in sources])

        parts: list[str] = []
        try:
            if canned or not chunks:
                reply = canned or llm.NOT_FOUND_MESSAGE
                parts.append(reply)
                yield _sse("token", {"text": reply})
            else:
                for delta in llm.stream_answer(question, chunks, history):
                    parts.append(delta)
                    yield _sse("token", {"text": delta})
        except llm.LLMNotConfiguredError as exc:
            logger.error("LLM not configured: %s (provider=%s)", exc, settings.llm_provider)
            yield _sse("error", {"detail": "The language model is not configured on the server."})
            return
        except llm.LLMAuthError:
            logger.error(
                "LLM provider %r rejected the API key: check it in .env", settings.llm_provider
            )
            yield _sse("error", {"detail": "The language model is not configured on the server."})
            return
        except llm.LLMRateLimitError:
            logger.warning("LLM provider rate limit hit (provider=%s)", settings.llm_provider)
            yield _sse("error", {"detail": "The language model is busy. Please retry in a moment."})
            return
        except Exception:  # noqa: BLE001 - surface a safe message, log the details
            logger.exception("LLM streaming failed")
            yield _sse("error", {"detail": "The language model request failed. Please retry."})
            return

        if not "".join(parts).strip():
            # e.g. safety-blocked or truncated before any text; don't save a blank answer.
            logger.error("LLM returned an empty response (provider=%s)", settings.llm_provider)
            yield _sse("error", {"detail": "The model returned an empty response. Please retry."})
            return

        saved_id, user_message_id, assistant_message_id = _persist_exchange(
            user_id, conversation_id, question, "".join(parts), sources, edit_message_id
        )
        yield _sse(
            "done",
            {
                "conversation_id": saved_id,
                "user_message_id": user_message_id,
                "assistant_message_id": assistant_message_id,
            },
        )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
