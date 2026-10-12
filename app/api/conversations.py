"""Saved chat history: list, reopen and delete conversations.

Conversations and messages are written by ``POST /chat``; these endpoints only read them
back (or delete them). Every lookup is scoped to the authenticated user.
"""

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import delete, select

from app.api.deps import CurrentUser, DbSession
from app.models.db import Conversation, Message
from app.models.schemas import ConversationDetail, ConversationOut

router = APIRouter(prefix="/conversations", tags=["conversations"])


def _get_owned(db, conversation_id: int, user_id: int) -> Conversation:
    """Fetch a conversation, hiding other users' conversations behind a 404."""
    conversation = db.get(Conversation, conversation_id)
    if conversation is None or conversation.user_id != user_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found")
    return conversation


@router.get("", response_model=list[ConversationOut])
def list_conversations(user: CurrentUser, db: DbSession) -> list[Conversation]:
    """The caller's saved chats, newest first."""
    stmt = (
        select(Conversation)
        .where(Conversation.user_id == user.id)
        .order_by(Conversation.id.desc())
    )
    return list(db.scalars(stmt))


@router.get("/{conversation_id}", response_model=ConversationDetail)
def get_conversation(conversation_id: int, user: CurrentUser, db: DbSession) -> Conversation:
    """One saved chat with all its messages (oldest first), so the UI can reopen it."""
    return _get_owned(db, conversation_id, user.id)


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(conversation_id: int, user: CurrentUser, db: DbSession) -> None:
    """Delete a saved chat and all of its messages."""
    conversation = _get_owned(db, conversation_id, user.id)
    db.execute(delete(Message).where(Message.conversation_id == conversation.id))
    db.delete(conversation)
    db.commit()
