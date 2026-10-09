"""Application settings, loaded from environment variables / a ``.env`` file."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed, validated configuration for the whole application.

    Values are read (in priority order) from real environment variables and then
    from a ``.env`` file in the working directory. Names are case-insensitive, so
    ``jwt_secret`` is populated from ``JWT_SECRET``.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- App ---
    app_name: str = "Chat with Your Documents"
    log_level: str = "INFO"
    preload_embeddings: bool = True  # load the embedding model at startup

    # --- Auth ---
    jwt_secret: str = Field(min_length=32)  # required: no insecure default
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # --- Storage ---
    database_url: str = "sqlite:///./data/app.db"
    chroma_path: Path = Path("./data/chroma")
    chroma_collection: str = "documents"
    upload_dir: Path = Path("./data/uploads")
    max_upload_mb: int = 25

    # --- Embeddings / ingestion ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 800
    chunk_overlap: int = 150

    # --- Retrieval ---
    top_k: int = 5
    # Cosine distance (0 = identical, 2 = opposite). Chunks farther than this are
    # discarded; if nothing survives, the LLM is never called. Short questions
    # embed less sharply (valid ones often land at 0.7-0.8), so this is a loose
    # pre-filter: the prompt still makes the LLM refuse when excerpts don't answer.
    distance_threshold: float = 0.8
    history_messages: int = 5

    # --- LLM ---
    llm_provider: Literal["anthropic", "gemini"] = "anthropic"
    llm_max_tokens: int = 1024
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5-5"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"

    # --- Rate limiting ---
    chat_rate_limit: str = "10/minute"

    @property
    def llm_api_key(self) -> str:
        """API key for the currently selected LLM provider (may be empty)."""
        return self.gemini_api_key if self.llm_provider == "gemini" else self.anthropic_api_key

    @property
    def llm_model(self) -> str:
        """Model name for the currently selected LLM provider."""
        return self.gemini_model if self.llm_provider == "gemini" else self.anthropic_model

    @property
    def max_upload_bytes(self) -> int:
        """Maximum accepted upload size in bytes."""
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings` singleton."""
    return Settings()  # type: ignore[call-arg]  # values come from the environment
