"""Shared fixtures. Environment is configured *before* the app is imported."""

import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="rag-tests-"))
os.environ.update(
    JWT_SECRET="test-secret-test-secret-test-secret-123",
    DATABASE_URL=f"sqlite:///{(_TMP / 'test.db').as_posix()}",
    CHROMA_PATH=str(_TMP / "chroma"),
    UPLOAD_DIR=str(_TMP / "uploads"),
    PRELOAD_EMBEDDINGS="false",
    ANTHROPIC_API_KEY="test-key",
    DISTANCE_THRESHOLD="0.7",
    ADMIN_EMAILS="Admin@Example.com, other-admin@example.com",
    TOP_K="5",
)

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.core.database import Base, engine  # noqa: E402
from app.core.rate_limit import limiter  # noqa: E402
from app.core.vectorstore import get_chroma_client  # noqa: E402
from app.main import app  # noqa: E402
from app.services import embeddings, llm  # noqa: E402
from tests.helpers import fake_embed, make_pdf  # noqa: E402


class FakeLLM:
    """Records calls to ``llm.stream_answer`` and streams a canned answer."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, question, chunks, history):  # noqa: ANN001
        self.calls.append({"question": question, "chunks": list(chunks), "history": list(history)})
        yield from ["The answer ", "is here [1]."]


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """A TestClient on a clean database + vector store with fake embeddings."""
    import app.core.database as database  # noqa: F401  (ensures models registered)
    from app.models import db as _models  # noqa: F401

    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    try:
        get_chroma_client().delete_collection("documents")
    except Exception:  # noqa: BLE001 - collection may not exist yet
        pass

    monkeypatch.setattr(embeddings, "embed_texts", fake_embed)
    limiter.enabled = False
    with TestClient(app) as test_client:
        yield test_client
    limiter.enabled = True


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    """Replace the real Anthropic call with a recording fake."""
    fake = FakeLLM()
    monkeypatch.setattr(llm, "stream_answer", fake)
    return fake


@pytest.fixture
def auth_headers(client: TestClient):
    """Factory: sign up (if needed) + log in, returning an Authorization header."""

    def _make(email: str = "alice@example.com", password: str = "correct-horse-1") -> dict[str, str]:
        client.post("/auth/signup", json={"email": email, "password": password})
        resp = client.post("/auth/login", data={"username": email, "password": password})
        assert resp.status_code == 200, resp.text
        return {"Authorization": f"Bearer {resp.json()['access_token']}"}

    return _make


@pytest.fixture
def upload_pdf(client: TestClient):
    """Factory: upload a generated PDF as the given user and return the response."""

    def _upload(headers: dict[str, str], pages: list[str], name: str = "doc.pdf"):
        return client.post(
            "/documents/upload",
            headers=headers,
            files={"file": (name, make_pdf(pages), "application/pdf")},
        )

    return _upload


@pytest.fixture(autouse=True)
def _no_llm_retry_delays(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never wait on LLM retry backoff (retry tests opt back in explicitly)."""
    monkeypatch.setattr(llm, "_RETRY_DELAYS", ())
