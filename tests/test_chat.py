import pytest
from fastapi.testclient import TestClient

from app.core.rate_limit import limiter
from app.services import llm
from app.services.retrieval import RetrievedChunk
from tests.helpers import parse_sse


def _ask(client: TestClient, headers: dict, question: str, conversation_id: int | None = None):
    payload: dict = {"question": question}
    if conversation_id is not None:
        payload["conversation_id"] = conversation_id
    resp = client.post("/chat", json=payload, headers=headers)
    return resp, (parse_sse(resp.text) if resp.status_code == 200 else [])


def test_streams_grounded_answer_with_citations(client, auth_headers, upload_pdf, fake_llm) -> None:
    headers = auth_headers()
    upload_pdf(headers, ["Cooking tips for pasta.", "Paris is the capital of France."], name="geo.pdf")

    resp, events = _ask(client, headers, "What is the capital of France?")

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    kinds = [e for e, _ in events]
    assert kinds[0] == "sources" and kinds[-1] == "done" and "token" in kinds

    sources = dict(events)["sources"]
    assert (sources[0]["source"], sources[0]["page"]) == ("geo.pdf", 2)  # best hit, correct page
    assert "".join(d["text"] for e, d in events if e == "token") == "The answer is here [1]."
    assert len(fake_llm.calls) == 1


def test_no_relevant_chunks_skips_llm(client, auth_headers, upload_pdf, fake_llm) -> None:
    headers = auth_headers()
    upload_pdf(headers, ["Paris is the capital of France."])

    _, events = _ask(client, headers, "quantum chromodynamics lagrangian")

    assert fake_llm.calls == []  # the LLM must NOT be called
    assert dict(events)["sources"] == []
    text = "".join(d["text"] for e, d in events if e == "token")
    assert text == llm.NOT_FOUND_MESSAGE


@pytest.mark.parametrize("text", ["hi", "Hello!", "hey there", "Thanks", "thank you so much.", "Good morning"])
def test_greetings_get_a_friendly_reply_without_retrieval_or_llm(
    client, auth_headers, upload_pdf, fake_llm, monkeypatch, text
) -> None:
    monkeypatch.setattr(
        "app.api.chat.retrieval.retrieve", lambda *a, **k: pytest.fail("retrieval must be skipped")
    )
    headers = auth_headers()
    upload_pdf(headers, ["Paris is the capital of France."])

    _, events = _ask(client, headers, text)
    assert fake_llm.calls == []
    assert "".join(d["text"] for e, d in events if e == "token") == llm.GREETING_MESSAGE
    assert events[-1][0] == "done"


@pytest.mark.parametrize(
    "text", ["hi, what is the capital of France?", "hello world program", "What is the salary?", "thanks for nothing, explain returns"]
)
def test_real_questions_are_not_mistaken_for_greetings(text: str) -> None:
    assert llm.is_greeting(text) is False


def test_retrieval_is_isolated_per_user(client, auth_headers, upload_pdf, fake_llm) -> None:
    alice, bob = auth_headers("alice@example.com"), auth_headers("bob@example.com")
    upload_pdf(alice, ["The vault password is swordfish."], name="secret.pdf")

    _, bob_events = _ask(client, bob, "What is the vault password?")
    assert fake_llm.calls == []
    assert dict(bob_events)["sources"] == []

    _, alice_events = _ask(client, alice, "What is the vault password?")
    assert dict(alice_events)["sources"][0]["source"] == "secret.pdf"


def test_history_is_persisted_and_limited_to_last_five(client, auth_headers, upload_pdf, fake_llm) -> None:
    headers = auth_headers()
    upload_pdf(headers, ["Paris is the capital of France."])

    _, events = _ask(client, headers, "What is the capital of France?")
    conversation_id = dict(events)["done"]["conversation_id"]
    assert len(fake_llm.calls[0]["history"]) == 0

    for _ in range(3):
        _ask(client, headers, "What is the capital of France?", conversation_id)

    # 3 prior exchanges = 6 stored messages; only the latest 5 may be sent.
    history = fake_llm.calls[-1]["history"]
    assert len(history) == 5
    assert [t.role for t in history] == ["assistant", "user", "assistant", "user", "assistant"]


def test_saved_history_keeps_citations_but_not_excerpt_text(client, auth_headers, upload_pdf, fake_llm) -> None:
    from sqlalchemy import select

    from app.core.database import SessionLocal
    from app.models.db import Message, MessageRole

    headers = auth_headers()
    upload_pdf(headers, ["Paris is the capital of France."], name="geo.pdf")
    _ask(client, headers, "What is the capital of France?")

    with SessionLocal() as db:
        saved = db.scalar(select(Message).where(Message.role == MessageRole.ASSISTANT))
    assert saved.sources and saved.sources[0]["source"] == "geo.pdf" and saved.sources[0]["page"] == 1
    assert all("snippet" not in s for s in saved.sources)


def test_conversations_are_private(client, auth_headers, upload_pdf, fake_llm) -> None:
    alice, bob = auth_headers("alice@example.com"), auth_headers("bob@example.com")
    upload_pdf(alice, ["Paris is the capital of France."])
    _, events = _ask(client, alice, "What is the capital of France?")
    conversation_id = dict(events)["done"]["conversation_id"]

    resp, _ = _ask(client, bob, "hello there", conversation_id)
    assert resp.status_code == 404


def test_unknown_conversation_is_404_and_logged(client, auth_headers, caplog) -> None:
    headers = auth_headers()
    with caplog.at_level("WARNING", logger="app.main"):
        resp, _ = _ask(client, headers, "hello", conversation_id=999)
    assert resp.status_code == 404
    assert any("POST /chat -> 404: Conversation not found" in r.message for r in caplog.records)


def test_conversation_id_zero_is_rejected_with_422(client, auth_headers, caplog) -> None:
    headers = auth_headers()
    with caplog.at_level("WARNING", logger="app.main"):
        resp = client.post("/chat", json={"question": "hi", "conversation_id": 0}, headers=headers)
    assert resp.status_code == 422
    assert any("conversation_id" in r.message for r in caplog.records)


def test_llm_failure_emits_error_event_and_saves_nothing(client, auth_headers, upload_pdf, monkeypatch) -> None:
    def boom(*_args, **_kwargs):
        raise RuntimeError("upstream down")
        yield  # pragma: no cover

    monkeypatch.setattr(llm, "stream_answer", boom)
    headers = auth_headers()
    upload_pdf(headers, ["Paris is the capital of France."])

    _, events = _ask(client, headers, "What is the capital of France?")
    assert events[-1][0] == "error"
    assert "done" not in [e for e, _ in events]


@pytest.mark.parametrize(
    ("error", "expected_text"),
    [
        (llm.LLMAuthError("bad key"), "not configured"),
        (llm.LLMNotConfiguredError("no key"), "not configured"),
        (llm.LLMRateLimitError("slow down"), "busy"),
    ],
)
def test_provider_errors_become_friendly_error_events(
    client, auth_headers, upload_pdf, monkeypatch, error, expected_text
) -> None:
    def failing(*_args, **_kwargs):
        raise error
        yield  # pragma: no cover

    monkeypatch.setattr(llm, "stream_answer", failing)
    headers = auth_headers()
    upload_pdf(headers, ["Paris is the capital of France."])

    _, events = _ask(client, headers, "What is the capital of France?")
    assert events[-1][0] == "error"
    assert expected_text in events[-1][1]["detail"]
    assert "done" not in [e for e, _ in events]


def test_empty_model_response_is_an_error_and_not_saved(
    client, auth_headers, upload_pdf, monkeypatch
) -> None:
    monkeypatch.setattr(llm, "stream_answer", lambda *a, **k: iter(["", "  "]))
    headers = auth_headers()
    upload_pdf(headers, ["Paris is the capital of France."])

    _, events = _ask(client, headers, "What is the capital of France?")
    assert events[-1][0] == "error" and "empty" in events[-1][1]["detail"]
    assert "done" not in [e for e, _ in events]


def test_chat_is_rate_limited(client, auth_headers, fake_llm) -> None:
    headers = auth_headers()
    limiter.enabled = True
    limiter.reset()
    statuses = [client.post("/chat", json={"question": "anything"}, headers=headers).status_code for _ in range(11)]
    assert statuses[:10] == [200] * 10
    assert statuses[10] == 429


def test_prompt_construction() -> None:
    chunks = [RetrievedChunk(text="Paris is in France.", source="geo.pdf", page=3, distance=0.1)]
    history = [
        llm.HistoryTurn("assistant", "stale answer"),  # leading assistant turn must be dropped
        llm.HistoryTurn("user", "earlier question"),
        llm.HistoryTurn("assistant", "earlier answer"),
    ]
    messages = llm.build_messages("Where is Paris?", chunks, history)

    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    last = messages[-1]["content"]
    assert "[1] (source: geo.pdf, page 3)" in last
    assert "<context>" in last and last.rstrip().endswith("Question: Where is Paris?")
