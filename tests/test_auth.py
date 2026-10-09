from fastapi.testclient import TestClient


def test_web_ui_is_served(client: TestClient) -> None:
    page = client.get("/")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    assert "/static/app.js" in page.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/styles.css").status_code == 200


def test_signup_and_login(client: TestClient) -> None:
    resp = client.post("/auth/signup", json={"email": "Bob@Example.com", "password": "supersecret1"})
    assert resp.status_code == 201
    assert resp.json()["email"] == "bob@example.com"
    assert "password" not in resp.text and "hashed" not in resp.text

    login = client.post("/auth/login", data={"username": "bob@example.com", "password": "supersecret1"})
    assert login.status_code == 200
    assert login.json()["token_type"] == "bearer"
    assert login.json()["access_token"]


def test_duplicate_signup_conflicts(client: TestClient) -> None:
    body = {"email": "dup@example.com", "password": "supersecret1"}
    assert client.post("/auth/signup", json=body).status_code == 201
    assert client.post("/auth/signup", json=body).status_code == 409


def test_wrong_password_rejected(client: TestClient) -> None:
    client.post("/auth/signup", json={"email": "c@example.com", "password": "supersecret1"})
    resp = client.post("/auth/login", data={"username": "c@example.com", "password": "nope-nope-1"})
    assert resp.status_code == 401


def test_weak_password_and_bad_email_rejected(client: TestClient) -> None:
    assert client.post("/auth/signup", json={"email": "d@example.com", "password": "short"}).status_code == 422
    assert client.post("/auth/signup", json={"email": "not-an-email", "password": "supersecret1"}).status_code == 422


def test_protected_routes_require_valid_token(client: TestClient) -> None:
    assert client.get("/documents").status_code == 401
    assert client.get("/documents", headers={"Authorization": "Bearer garbage"}).status_code == 401
    assert client.post("/chat", json={"question": "hi"}).status_code == 401
