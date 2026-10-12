"""Admin panel API: access control, pagination, filters and data exposure."""

import pytest
from fastapi.testclient import TestClient

from tests.helpers import parse_sse

ADMIN = "admin@example.com"  # configured in conftest as "Admin@Example.com" (case-insensitive)
DATA_ENDPOINTS = ["stats", "users", "documents", "conversations", "messages", "chunks"]


def _ask(client: TestClient, headers: dict, question: str, conversation_id=None):
    payload = {"question": question}
    if conversation_id:
        payload["conversation_id"] = conversation_id
    resp = client.post("/chat", json=payload, headers=headers)
    return dict(parse_sse(resp.text))["done"]["conversation_id"]


@pytest.fixture
def populated(client, auth_headers, upload_pdf, fake_llm):
    """An admin, two regular users with documents and chats."""
    admin = auth_headers(ADMIN)
    alice, bob = auth_headers("alice@example.com"), auth_headers("bob@example.com")
    upload_pdf(alice, ["Paris is the capital of France.", "The Seine flows through Paris."], name="geo.pdf")
    upload_pdf(alice, ["Boil pasta in salted water."], name="cooking.pdf")
    upload_pdf(bob, ["Berlin is the capital of Germany."], name="bob.pdf")
    c1 = _ask(client, alice, "What is the capital of France?")
    _ask(client, alice, "What river flows through Paris?", c1)
    _ask(client, bob, "What is the capital of Germany?")
    return SimpleAdmin(admin, alice, bob, c1)


class SimpleAdmin:
    def __init__(self, admin, alice, bob, conversation_id):
        self.admin, self.alice, self.bob, self.conversation_id = admin, alice, bob, conversation_id


# --- access control -------------------------------------------------------------------------
@pytest.mark.parametrize("endpoint", DATA_ENDPOINTS)
def test_admin_data_requires_login_and_admin_rights(client, auth_headers, endpoint) -> None:
    alice = auth_headers("alice@example.com")
    url = f"/admin/data/{endpoint}"
    assert client.get(url).status_code == 401
    assert client.get(url, headers=alice).status_code == 403
    assert client.get(url, headers=auth_headers(ADMIN)).status_code == 200


def test_admin_is_disabled_when_no_admin_emails_are_configured(client, auth_headers, monkeypatch) -> None:
    from app.api import deps
    from app.config import get_settings

    headers = auth_headers(ADMIN)
    monkeypatch.setattr(deps, "get_settings", lambda: get_settings().model_copy(update={"admin_emails": ""}))
    assert client.get("/admin/data/stats", headers=headers).status_code == 403


def test_me_reports_admin_flag(client, auth_headers) -> None:
    assert client.get("/auth/me", headers=auth_headers(ADMIN)).json()["is_admin"] is True
    me = client.get("/auth/me", headers=auth_headers("alice@example.com")).json()
    assert me["is_admin"] is False and me["email"] == "alice@example.com"
    assert "hashed_password" not in me
    assert client.get("/auth/me").status_code == 401


def test_admin_page_shell_is_served(client) -> None:
    resp = client.get("/admin")
    assert resp.status_code == 200 and "text/html" in resp.headers["content-type"]


# --- data -----------------------------------------------------------------------------------
def test_stats(client, populated) -> None:
    stats = client.get("/admin/data/stats", headers=populated.admin).json()
    assert stats == {"users": 3, "documents": 3, "conversations": 2, "messages": 6, "chunks": 4}


def test_users_never_expose_password_hashes(client, populated) -> None:
    body = client.get("/admin/data/users", headers=populated.admin)
    assert "hashed_password" not in body.text and "pbkdf2" not in body.text
    rows = {r["email"]: r for r in body.json()["items"]}
    assert set(rows) == {ADMIN, "alice@example.com", "bob@example.com"}
    assert (rows["alice@example.com"]["documents"], rows["alice@example.com"]["conversations"]) == (2, 1)
    assert (rows[ADMIN]["documents"], rows[ADMIN]["conversations"]) == (0, 0)


def test_admin_sees_other_users_documents_conversations_and_messages(client, populated) -> None:
    get = lambda path: client.get(f"/admin/data/{path}", headers=populated.admin).json()  # noqa: E731

    docs = get("documents")
    assert {(d["filename"], d["user_email"], d["status"]) for d in docs["items"]} == {
        ("geo.pdf", "alice@example.com", "ready"),
        ("cooking.pdf", "alice@example.com", "ready"),
        ("bob.pdf", "bob@example.com", "ready"),
    }
    convs = get("conversations")
    assert convs["total"] == 2 and {c["messages"] for c in convs["items"]} == {4, 2}

    msgs = get("messages")
    assert msgs["total"] == 6
    assert {m["role"] for m in msgs["items"]} == {"user", "assistant"}
    assert msgs["items"][0]["id"] > msgs["items"][-1]["id"]  # newest first overall


def test_messages_of_one_conversation_are_in_chat_order(client, populated) -> None:
    items = client.get(
        f"/admin/data/messages?conversation_id={populated.conversation_id}", headers=populated.admin
    ).json()["items"]
    assert [m["role"] for m in items] == ["user", "assistant", "user", "assistant"]
    assert items[0]["content"] == "What is the capital of France?"


def test_chunks_come_from_the_vector_store(client, populated) -> None:
    body = client.get("/admin/data/chunks", headers=populated.admin).json()
    assert body["total"] == 4
    chunk = next(c for c in body["items"] if c["source"] == "bob.pdf")
    assert (chunk["page"], chunk["chunk_index"]) == (1, 0)
    assert "Berlin" in chunk["text"] and chunk["chars"] == len(chunk["text"])
    assert "embedding" not in chunk


# --- filters --------------------------------------------------------------------------------
def test_filters(client, populated) -> None:
    get = lambda path: client.get(f"/admin/data/{path}", headers=populated.admin).json()  # noqa: E731
    alice_id = next(u["id"] for u in get("users")["items"] if u["email"] == "alice@example.com")

    assert [u["email"] for u in get("users?q=BOB")["items"]] == ["bob@example.com"]  # case-insensitive
    assert {d["filename"] for d in get(f"documents?user_id={alice_id}")["items"]} == {"geo.pdf", "cooking.pdf"}
    assert [d["filename"] for d in get("documents?q=cook")["items"]] == ["cooking.pdf"]
    assert get("conversations?q=river")["total"] == 0  # titles come from the first question only
    assert get(f"conversations?user_id={alice_id}")["total"] == 1
    assert get("messages?q=seine")["total"] == 0 and get("messages?q=river")["total"] >= 1
    assert get(f"chunks?user_id={alice_id}")["total"] == 3
    assert get(f"chunks?user_id={alice_id}&q=pasta")["total"] == 1
    doc_id = next(d["id"] for d in get("documents?q=bob")["items"])
    assert [c["source"] for c in get(f"chunks?document_id={doc_id}")["items"]] == ["bob.pdf"]


def test_search_treats_percent_and_underscore_literally(client, populated) -> None:
    # '%' / '_' are LIKE wildcards; they must not match everything.
    assert client.get("/admin/data/users?q=%25", headers=populated.admin).json()["total"] == 0
    assert client.get("/admin/data/users?q=_", headers=populated.admin).json()["total"] == 0


# --- pagination -----------------------------------------------------------------------------
@pytest.mark.parametrize("table", ["users", "documents", "messages", "chunks"])
def test_pagination_covers_every_row_exactly_once(client, populated, table) -> None:
    total = client.get(f"/admin/data/{table}?page_size=100", headers=populated.admin).json()["total"]
    assert total >= 3
    seen, page = [], 1
    while True:
        body = client.get(f"/admin/data/{table}?page={page}&page_size=2", headers=populated.admin).json()
        assert body["page"] == page and body["page_size"] == 2 and body["total"] == total
        assert len(body["items"]) <= 2
        seen += [row["id"] for row in body["items"]]
        if page >= body["pages"]:
            break
        page += 1
    assert body["pages"] == -(-total // 2)  # ceil
    assert len(seen) == total and len(set(seen)) == total


def test_page_past_the_end_is_empty_and_bad_params_are_rejected(client, populated) -> None:
    beyond = client.get("/admin/data/users?page=99", headers=populated.admin).json()
    assert beyond["items"] == [] and beyond["total"] == 3 and beyond["pages"] == 1
    empty = client.get("/admin/data/documents?q=nothing-matches", headers=populated.admin).json()
    assert empty["total"] == 0 and empty["pages"] == 1 and empty["items"] == []
    for bad in ("page=0", "page_size=0", "page_size=101", "page=abc"):
        assert client.get(f"/admin/data/users?{bad}", headers=populated.admin).status_code == 422
