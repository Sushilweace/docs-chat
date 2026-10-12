from fastapi.testclient import TestClient

from app.core.vectorstore import get_collection


def test_upload_ingests_in_background(client: TestClient, auth_headers, upload_pdf) -> None:
    headers = auth_headers()
    resp = upload_pdf(headers, ["Paris is the capital of France.", "The Seine flows through Paris."])
    assert resp.status_code == 202
    assert resp.json()["status"] == "processing"

    docs = client.get("/documents", headers=headers).json()
    assert len(docs) == 1
    assert docs[0]["status"] == "ready"  # TestClient runs background tasks before returning
    assert docs[0]["page_count"] == 2
    assert docs[0]["chunk_count"] == 2

    stored = get_collection().get(include=["metadatas"])["metadatas"]
    assert {(m["source"], m["page"]) for m in stored} == {("doc.pdf", 1), ("doc.pdf", 2)}
    assert all(set(m) >= {"user_id", "source", "page"} for m in stored)


def test_rejects_non_pdf(client: TestClient, auth_headers) -> None:
    headers = auth_headers()
    wrong_ext = client.post("/documents/upload", headers=headers, files={"file": ("a.txt", b"hello", "text/plain")})
    assert wrong_ext.status_code == 400
    fake_pdf = client.post("/documents/upload", headers=headers, files={"file": ("a.pdf", b"not a pdf", "application/pdf")})
    assert fake_pdf.status_code == 400
    empty = client.post("/documents/upload", headers=headers, files={"file": ("a.pdf", b"", "application/pdf")})
    assert empty.status_code == 400
    assert client.get("/documents", headers=headers).json() == []


def test_pdf_without_text_is_marked_failed(client: TestClient, auth_headers, upload_pdf) -> None:
    headers = auth_headers()
    upload_pdf(headers, [""])
    doc = client.get("/documents", headers=headers).json()[0]
    assert doc["status"] == "failed"
    assert "No extractable text" in doc["error"]


def test_document_listing_is_per_user(client: TestClient, auth_headers, upload_pdf) -> None:
    alice, bob = auth_headers("alice@example.com"), auth_headers("bob@example.com")
    upload_pdf(alice, ["alice secret plans"], name="alice.pdf")
    assert [d["filename"] for d in client.get("/documents", headers=alice).json()] == ["alice.pdf"]
    assert client.get("/documents", headers=bob).json() == []


def _ask_for_sources(client, headers, question):
    from tests.helpers import parse_sse

    resp = client.post("/chat", json={"question": question}, headers=headers)
    return dict(parse_sse(resp.text))["sources"]


def test_delete_document_removes_vectors_file_and_row(client, auth_headers, upload_pdf, fake_llm) -> None:
    from pathlib import Path

    from app.core.database import SessionLocal
    from app.models.db import Document

    headers = auth_headers()
    upload_pdf(headers, ["Paris is the capital of France."], name="geo.pdf")
    doc = client.get("/documents", headers=headers).json()[0]
    with SessionLocal() as db:
        stored = Path(db.get(Document, doc["id"]).stored_path)
    assert stored.exists()
    assert _ask_for_sources(client, headers, "What is the capital of France?")  # found before

    resp = client.delete(f"/documents/{doc['id']}", headers=headers)

    assert resp.status_code == 204
    assert client.get("/documents", headers=headers).json() == []
    assert not stored.exists()
    assert get_collection().count() == 0
    fake_llm.calls.clear()
    assert _ask_for_sources(client, headers, "What is the capital of France?") == []  # chat no longer sees it
    assert fake_llm.calls == []


def test_delete_document_only_removes_that_document(client, auth_headers, upload_pdf) -> None:
    headers = auth_headers()
    upload_pdf(headers, ["Paris is the capital of France."], name="a.pdf")
    upload_pdf(headers, ["Berlin is the capital of Germany."], name="b.pdf")
    docs = {d["filename"]: d["id"] for d in client.get("/documents", headers=headers).json()}

    assert client.delete(f"/documents/{docs['a.pdf']}", headers=headers).status_code == 204

    assert [d["filename"] for d in client.get("/documents", headers=headers).json()] == ["b.pdf"]
    assert {m["source"] for m in get_collection().get(include=["metadatas"])["metadatas"]} == {"b.pdf"}


def test_cannot_delete_someone_elses_or_missing_document(client, auth_headers, upload_pdf) -> None:
    alice, bob = auth_headers("alice@example.com"), auth_headers("bob@example.com")
    upload_pdf(alice, ["alice secret plans"], name="alice.pdf")
    doc_id = client.get("/documents", headers=alice).json()[0]["id"]

    assert client.delete(f"/documents/{doc_id}", headers=bob).status_code == 404
    assert client.delete("/documents/9999", headers=alice).status_code == 404
    assert client.delete(f"/documents/{doc_id}").status_code == 401
    assert len(client.get("/documents", headers=alice).json()) == 1  # untouched
    assert get_collection().count() == 1


def test_cannot_delete_a_document_that_is_still_processing(client, auth_headers) -> None:
    from app.core.database import SessionLocal
    from sqlalchemy import select

    from app.models.db import Document, DocumentStatus, User

    headers = auth_headers("alice@example.com")
    with SessionLocal() as db:
        user_id = db.scalar(select(User.id).where(User.email == "alice@example.com"))
        doc = Document(user_id=user_id, filename="x.pdf", stored_path="x.pdf", status=DocumentStatus.PROCESSING)
        db.add(doc)
        db.commit()
        doc_id = doc.id

    assert client.delete(f"/documents/{doc_id}", headers=headers).status_code == 409
