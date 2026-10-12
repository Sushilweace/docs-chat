"""Saved chat history (list / reopen / delete) and editing an earlier question."""

from fastapi.testclient import TestClient

from tests.helpers import parse_sse


def _ask(client: TestClient, headers: dict, question: str, conversation_id=None, edit_message_id=None):
    payload: dict = {"question": question}
    if conversation_id is not None:
        payload["conversation_id"] = conversation_id
    if edit_message_id is not None:
        payload["edit_message_id"] = edit_message_id
    resp = client.post("/chat", json=payload, headers=headers)
    return resp, (parse_sse(resp.text) if resp.status_code == 200 else [])


def _setup(client, auth_headers, upload_pdf, email="alice@example.com"):
    headers = auth_headers(email)
    upload_pdf(headers, ["Paris is the capital of France.", "Berlin is the capital of Germany."], name="geo.pdf")
    return headers


# --- saving & reading back ---------------------------------------------------------------
def test_chat_is_saved_and_can_be_reopened(client, auth_headers, upload_pdf, fake_llm) -> None:
    headers = _setup(client, auth_headers, upload_pdf)
    _, events = _ask(client, headers, "What is the capital of France?")
    done = dict(events)["done"]

    listing = client.get("/conversations", headers=headers).json()
    assert [(c["id"], c["title"]) for c in listing] == [(done["conversation_id"], "What is the capital of France?")]

    detail = client.get(f"/conversations/{done['conversation_id']}", headers=headers).json()
    assert [(m["role"], m["content"]) for m in detail["messages"]] == [
        ("user", "What is the capital of France?"),
        ("assistant", "The answer is here [1]."),
    ]
    assert [m["id"] for m in detail["messages"]] == [done["user_message_id"], done["assistant_message_id"]]
    assert detail["messages"][1]["sources"][0]["source"] == "geo.pdf"
    assert "snippet" not in detail["messages"][1]["sources"][0]


def test_conversation_list_is_newest_first_and_per_user(client, auth_headers, upload_pdf, fake_llm) -> None:
    alice = _setup(client, auth_headers, upload_pdf)
    bob = auth_headers("bob@example.com")
    _ask(client, alice, "first question about France?")
    _ask(client, alice, "second question about Germany?")

    titles = [c["title"] for c in client.get("/conversations", headers=alice).json()]
    assert titles == ["second question about Germany?", "first question about France?"]
    assert client.get("/conversations", headers=bob).json() == []
    assert client.get("/conversations").status_code == 401


def test_conversations_cannot_be_read_or_deleted_by_others(client, auth_headers, upload_pdf, fake_llm) -> None:
    alice = _setup(client, auth_headers, upload_pdf)
    bob = auth_headers("bob@example.com")
    _, events = _ask(client, alice, "What is the capital of France?")
    cid = dict(events)["done"]["conversation_id"]

    assert client.get(f"/conversations/{cid}", headers=bob).status_code == 404
    assert client.delete(f"/conversations/{cid}", headers=bob).status_code == 404
    assert client.get(f"/conversations/{cid}", headers=alice).status_code == 200


def test_delete_conversation_removes_its_messages(client, auth_headers, upload_pdf, fake_llm) -> None:
    from sqlalchemy import func, select

    from app.core.database import SessionLocal
    from app.models.db import Message

    headers = _setup(client, auth_headers, upload_pdf)
    _, events = _ask(client, headers, "What is the capital of France?")
    cid = dict(events)["done"]["conversation_id"]

    assert client.delete(f"/conversations/{cid}", headers=headers).status_code == 204

    assert client.get("/conversations", headers=headers).json() == []
    assert client.get(f"/conversations/{cid}", headers=headers).status_code == 404
    with SessionLocal() as db:
        assert db.scalar(select(func.count()).select_from(Message)) == 0
    # chatting into the deleted conversation is a clean 404, not a crash
    assert _ask(client, headers, "What is the capital of France?", cid)[0].status_code == 404


# --- editing a question ------------------------------------------------------------------
def _two_turns(client, headers):
    _, e1 = _ask(client, headers, "What is the capital of France?")
    first = dict(e1)["done"]
    _, e2 = _ask(client, headers, "What is the capital of Germany?", first["conversation_id"])
    return first, dict(e2)["done"]


def test_edit_replaces_that_question_and_everything_after_it(client, auth_headers, upload_pdf, fake_llm) -> None:
    headers = _setup(client, auth_headers, upload_pdf)
    first, second = _two_turns(client, headers)
    cid = first["conversation_id"]

    resp, events = _ask(client, headers, "What is the capital of Spain?", cid, second["user_message_id"])
    assert resp.status_code == 200 and events[-1][0] == "done"

    messages = client.get(f"/conversations/{cid}", headers=headers).json()["messages"]
    assert [m["content"] for m in messages if m["role"] == "user"] == [
        "What is the capital of France?",  # earlier turn untouched
        "What is the capital of Spain?",  # replaced the German question
    ]
    assert len(messages) == 4  # old German question + its answer are gone
    new_ids = dict(events)["done"]
    assert [m["id"] for m in messages[-2:]] == [new_ids["user_message_id"], new_ids["assistant_message_id"]]


def test_editing_the_first_question_drops_the_whole_tail_and_retitles(client, auth_headers, upload_pdf, fake_llm) -> None:
    headers = _setup(client, auth_headers, upload_pdf)
    first, _ = _two_turns(client, headers)
    cid = first["conversation_id"]

    _ask(client, headers, "Tell me about Berlin?", cid, first["user_message_id"])

    detail = client.get(f"/conversations/{cid}", headers=headers).json()
    assert [m["content"] for m in detail["messages"] if m["role"] == "user"] == ["Tell me about Berlin?"]
    assert detail["title"] == "Tell me about Berlin?"
    assert len(client.get("/conversations", headers=headers).json()) == 1  # same conversation, not a new one


def test_edit_only_uses_older_messages_as_history(client, auth_headers, upload_pdf, fake_llm) -> None:
    headers = _setup(client, auth_headers, upload_pdf)
    first, second = _two_turns(client, headers)

    _ask(client, headers, "What is the capital of Germany?", first["conversation_id"], second["user_message_id"])

    history = fake_llm.calls[-1]["history"]
    assert [t.content for t in history if t.role == "user"] == ["What is the capital of France?"]
    assert len(history) == 2  # just the first exchange; the replaced turn is excluded


def test_failed_edit_leaves_the_conversation_untouched(client, auth_headers, upload_pdf, fake_llm, monkeypatch) -> None:
    from app.services import llm

    headers = _setup(client, auth_headers, upload_pdf)
    first, second = _two_turns(client, headers)
    cid = first["conversation_id"]

    def boom(*args, **kwargs):
        raise llm.LLMRateLimitError("busy")
        yield  # pragma: no cover - makes this a generator

    monkeypatch.setattr(llm, "stream_answer", boom)
    _, events = _ask(client, headers, "What is the capital of Spain?", cid, second["user_message_id"])

    assert events[-1][0] == "error"
    messages = client.get(f"/conversations/{cid}", headers=headers).json()["messages"]
    assert [m["content"] for m in messages if m["role"] == "user"] == [
        "What is the capital of France?",
        "What is the capital of Germany?",
    ]


def test_edit_validation(client, auth_headers, upload_pdf, fake_llm) -> None:
    alice = _setup(client, auth_headers, upload_pdf)
    bob = auth_headers("bob@example.com")
    first, _ = _two_turns(client, alice)
    cid = first["conversation_id"]

    # needs a conversation
    assert _ask(client, alice, "x?", None, first["user_message_id"])[0].status_code == 422
    # only user questions are editable, not assistant answers
    assert _ask(client, alice, "x?", cid, first["assistant_message_id"])[0].status_code == 400
    # unknown message
    assert _ask(client, alice, "x?", cid, 99999)[0].status_code == 404
    # someone else's conversation
    assert _ask(client, bob, "x?", cid, first["user_message_id"])[0].status_code == 404
    # message from a different conversation of the same user
    _, other = _ask(client, alice, "What is the capital of Germany?")
    other_msg = dict(other)["done"]["user_message_id"]
    assert _ask(client, alice, "x?", cid, other_msg)[0].status_code == 404
