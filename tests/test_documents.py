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
