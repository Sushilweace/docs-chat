"""Pydantic request/response schemas (the public API contract)."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from app.models.db import DocumentStatus


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

    @field_validator("question")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class SourceOut(BaseModel):
    """A retrieved chunk surfaced to the client as a citation."""

    index: int = Field(description="1-based number matching [n] markers in the answer")
    source: str
    page: int
    distance: float
    snippet: str
