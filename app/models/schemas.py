"""Pydantic request/response schemas (the public API contract)."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator, model_validator

from app.models.db import DocumentStatus, MessageRole


# --- Auth ---
class SignupRequest(BaseModel):
    """Payload for ``POST /auth/signup``."""

    email: EmailStr
    password: str = Field(min_length=8, max_length=128)

    @field_validator("email")
    @classmethod
    def _normalise_email(cls, value: str) -> str:
        return value.strip().lower()


class UserOut(BaseModel):
    """Public view of a user (never includes the password hash)."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    email: EmailStr
    created_at: datetime


class MeOut(UserOut):
    """``GET /auth/me``: the current user and whether they are an admin."""

    is_admin: bool


class TokenResponse(BaseModel):
    """OAuth2-style bearer token response."""

    access_token: str
    token_type: str = "bearer"


# --- Documents ---
class DocumentOut(BaseModel):
    """Public view of an uploaded document."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    status: DocumentStatus
    page_count: int
    chunk_count: int
    error: str | None = None
    created_at: datetime


# --- Chat ---
class ChatRequest(BaseModel):
    """Payload for ``POST /chat``."""

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"question": "What does the document say about refunds?"}]}
    )

    question: str = Field(min_length=1, max_length=2000)
    conversation_id: int | None = Field(
        default=None,
        gt=0,
        description="Continue an existing conversation; omit (or null) to start a new one.",
    )
    edit_message_id: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Edit mode: id of one of your earlier questions in this conversation. That question "
            "and everything after it are replaced by this new question and its answer."
        ),
    )

    @field_validator("question")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value

    @model_validator(mode="after")
    def _edit_needs_conversation(self) -> "ChatRequest":
        if self.edit_message_id is not None and self.conversation_id is None:
            raise ValueError("edit_message_id requires conversation_id")
        return self


class SourceOut(BaseModel):
    """A retrieved chunk surfaced to the client as a citation."""

    index: int = Field(description="1-based number matching [n] markers in the answer")
    source: str
    page: int
    distance: float
    snippet: str


# --- Saved conversations ---
class ConversationOut(BaseModel):
    """One row of the chat-history list."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    created_at: datetime


class MessageOut(BaseModel):
    """A saved chat message. ``sources`` are citations only (file + page), no excerpt text."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    role: MessageRole
    content: str
    sources: list[dict] | None = None
    created_at: datetime


class ConversationDetail(ConversationOut):
    """A conversation with all of its messages, oldest first."""

    messages: list[MessageOut]
