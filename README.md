# Chat with Your Documents (RAG)

A FastAPI service where users sign up, upload PDFs, and chat with them. Answers are
**grounded in the user's own documents**, **streamed** over Server-Sent Events, and
carry **file + page citations**. If nothing relevant is found, the service says so
*without calling the LLM*.

> **Deep dive:** see [`docs/PROJECT_GUIDE.md`](docs/PROJECT_GUIDE.md) for the full explanation
> (tech choices, internals with diagrams, limitations) plus an interview Q&A bank.

| Concern        | Choice                                                    |
| -------------- | --------------------------------------------------------- |
| API            | FastAPI (Python 3.11+)                                    |
| Vector store   | ChromaDB (persistent, cosine distance)                    |
| Embeddings     | `sentence-transformers/all-MiniLM-L6-v2` (local, CPU OK)  |
| Generation     | Anthropic SDK (default), Google Gemini or Groq, via `LLM_PROVIDER` |
| PDF parsing    | `pypdf` (per page)                                        |
| Relational DB  | SQLite + SQLAlchemy 2.0 (users, documents, chat history)  |
| Auth           | JWT (`python-jose`) + `passlib` password hashing          |
| Rate limiting  | `slowapi` on `/chat` (per user, falls back to IP)         |

## Setup

```powershell
# 1. Virtual environment (Python 3.11+)
python -m venv venv
.\venv\Scripts\Activate.ps1          # macOS/Linux: source venv/bin/activate

# 2. Dependencies (first run downloads torch + the embedding model, ~1 GB)
pip install -r requirements.txt

# 3. Configuration
copy .env.example .env               # macOS/Linux: cp .env.example .env
#   - set JWT_SECRET:  python -c "import secrets; print(secrets.token_urlsafe(48))"
#   - pick an LLM: set ANTHROPIC_API_KEY, or LLM_PROVIDER=gemini + GEMINI_API_KEY,
#     or LLM_PROVIDER=groq + GROQ_API_KEY

# 4. Run
uvicorn app.main:app --reload
```

- **Web UI:** <http://127.0.0.1:8000/> (sign up, upload PDFs, chat)
- **Interactive API docs:** <http://127.0.0.1:8000/docs> (use **Authorize** after logging in)

> Tip: `uvicorn app.main:app --reload --reload-dir app` restarts only when app code changes
> (each restart reloads the embedding model, which takes a few seconds).

## Web UI

A dependency-free single page app (`app/static/`: HTML + CSS + vanilla JS) is served by FastAPI
at `/`, so there is no build step and no CORS to configure.

- **Log in / Sign up:** tabs on one card; server validation errors are shown inline; the JWT is
  kept in `localStorage`, and an expired token returns you to the login screen.
- **Documents sidebar:** click or drag & drop PDFs. Each document shows `processing` (with a
  spinner), `ready` (pages/chunks) or `failed` (with the reason). The list refreshes itself
  while anything is still being ingested. The **✕** button removes a document (its vectors,
  PDF file and database row), so chat stops using it immediately; it is disabled while the
  document is still being indexed.
- **Chat history:** every chat is saved in SQLite and listed in the sidebar. Click one to
  reopen it (and continue it), **✕** deletes it, and **New chat** starts a fresh one.
- **Edit a question:** the **Edit** link under any of your messages lets you change it and
  resend. That question and everything after it are replaced by the new question and its new
  answer (older messages are kept). If the new answer fails, the original chat is left intact.
- **Chat:** answers stream token by token (the UI reads the SSE stream with `fetch`, since
  `EventSource` cannot POST or send an `Authorization` header). `[n]` markers become citation
  chips whose tooltip shows the file and page (hover a chip). There is no separate sources list,
  and excerpt text is neither displayed nor stored in chat history. Follow-up questions continue the same conversation; **New chat**
  starts a fresh one. Enter sends, Shift+Enter inserts a newline.
- All user-controlled text (filenames, answers) is rendered with `textContent`, never `innerHTML`.

> Want to pin Python 3.11 exactly? With [uv](https://docs.astral.sh/uv/):
> `uv venv --python 3.11 venv && uv pip install -r requirements.txt`

### Try it with curl

```bash
# Sign up, then log in (login is an OAuth2 *form*; username = email)
curl -X POST localhost:8000/auth/signup -H "Content-Type: application/json" \
     -d '{"email":"me@example.com","password":"a-strong-password"}'
TOKEN=$(curl -s -X POST localhost:8000/auth/login \
     -d "username=me@example.com&password=a-strong-password" | python -c "import sys,json;print(json.load(sys.stdin)['access_token'])")

# Upload a PDF (returns 202 immediately; ingestion runs in the background)
curl -X POST localhost:8000/documents/upload -H "Authorization: Bearer $TOKEN" -F "file=@report.pdf"
curl localhost:8000/documents -H "Authorization: Bearer $TOKEN"     # wait for status "ready"

# Chat (-N disables curl buffering so you see the stream)
curl -N -X POST localhost:8000/chat -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" -d '{"question":"What does the report conclude?"}'
```

## API

| Method & path           | Auth | Description                                                         |
| ----------------------- | ---- | ------------------------------------------------------------------- |
| `POST /auth/signup`     | no   | Create an account (`201`; `409` if the email exists)                |
| `POST /auth/login`      | no   | OAuth2 password form -> `{access_token, token_type}`                |
| `POST /documents/upload`| yes  | Upload a PDF; `202` with `status: processing`                       |
| `GET /documents`        | yes  | List the caller's documents (`processing` / `ready` / `failed`)     |
| `DELETE /documents/{id}`| yes  | Remove a document and its vectors (`204`; `409` while processing)   |
| `POST /chat`            | yes  | Ask a question; SSE stream. Rate limited (default `10/minute`)      |
| `GET /conversations`    | yes  | The caller's saved chats, newest first                              |
| `GET /conversations/{id}` | yes | One chat with all its messages (oldest first)                      |
| `DELETE /conversations/{id}` | yes | Delete a chat and its messages (`204`)                          |
| `GET /auth/me`          | yes  | The caller's profile, including `is_admin`                          |
| `GET /admin`            | no*  | Admin panel page (*its data calls are admin-only, see below)        |
| `GET /admin/data/{stats,users,documents,conversations,messages,chunks}` | admin | Paginated read-only views of all data (`?page=&page_size=&q=`) |
| `GET /health`           | no   | Liveness probe                                                      |

### `/chat` stream format

Request: `{"question": "...", "conversation_id": 12}` (`conversation_id` optional; omit to start a new thread).
To **edit** an earlier question, also send `"edit_message_id": <id of your old question>`
(requires `conversation_id`): that message and all later ones are replaced by this exchange,
but only after the new answer succeeded.

Every event's `data` is JSON:

```
event: sources
data: [{"index":1,"source":"report.pdf","page":4,"distance":0.31,"snippet":"..."}]

event: token
data: {"text":"The report concludes "}

event: token
data: {"text":"that revenue grew [1]."}

event: done
data: {"conversation_id": 12, "user_message_id": 31, "assistant_message_id": 32}
```

`[n]` markers in the answer refer to `sources[].index`. An `error` event
(`{"detail": "..."}`) ends the stream if generation fails. When no chunk passes the
relevance threshold, `sources` is `[]` and a fixed "not in your documents" message is
streamed; the LLM is never invoked.

## Architecture

```mermaid
flowchart TD
    Client([Client])

    subgraph API["FastAPI app"]
        Auth["/auth<br/>signup, login"]
        Docs["/documents<br/>upload, list"]
        Chat["/chat<br/>SSE stream"]
        RL{{"slowapi<br/>rate limit"}}
    end

    subgraph Services
        Ing["ingestion<br/>parse, chunk, embed"]
        Ret["retrieval<br/>top-k + threshold"]
        LLM["llm<br/>prompt + stream"]
        Emb["embeddings<br/>MiniLM-L6-v2"]
    end

    SQL[("SQLite<br/>users, documents,<br/>conversations, messages")]
    Chroma[("ChromaDB<br/>vectors + metadata<br/>user_id, source, page")]
    Files[/"uploads/&lt;user_id&gt;/*.pdf"/]
    Claude[["Anthropic API"]]

    Client -->|JWT| Auth --> SQL
    Client -->|PDF| Docs --> Files
    Docs -.->|BackgroundTask| Ing
    Docs --> SQL
    Ing --> Emb --> Chroma
    Ing -->|status ready/failed| SQL

    Client -->|question| RL --> Chat
    Chat -->|last 5 messages| SQL
    Chat --> Ret
    Ret --> Emb
    Ret -->|"where user_id = me"| Chroma
    Ret -->|relevant chunks| Chat
    Chat -->|"no chunks: canned reply"| Client
    Chat -->|"chunks + history"| LLM --> Claude
    Claude -->|token stream| LLM -->|SSE| Client
    Chat -->|save exchange| SQL
```

### Request lifecycles

**Ingestion** (`POST /documents/upload`)
1. Validate extension, `%PDF-` magic bytes and size cap while streaming to `uploads/<user_id>/<uuid>.pdf`.
2. Insert a `Document` row (`processing`) and return `202`.
3. A `BackgroundTask` (own DB session) extracts text **per page**, chunks each page
   (800 chars, 150 overlap, snapped to word boundaries), embeds the chunks, and stores
   them in Chroma with metadata `{user_id, source, page, document_id, chunk_index}`.
4. The row becomes `ready` (with page/chunk counts) or `failed` (with an error message
   and any partial vectors removed). Scanned/image-only PDFs fail with a clear message.

**Chat** (`POST /chat`)
1. Rate limit check (keyed by user id from the JWT).
2. Load the last 5 messages of the conversation (404 if it belongs to someone else).
3. Embed the question and query Chroma with `where={"user_id": <caller>}`; drop chunks whose
   cosine distance exceeds `DISTANCE_THRESHOLD`.
4. **No chunks left?** Stream the "not in your documents" reply; skip the LLM.
5. Otherwise build the prompt (numbered excerpts + history), stream tokens from Claude.
6. Persist the user question and assistant answer (with sources) in one transaction.

### Design notes

- **Per-user isolation** is enforced in the vector query itself (`where` filter), the user
  id always comes from the verified JWT (never the request body), and results are re-checked
  after retrieval. Conversations are also owner-checked.
- **Prompt-injection hygiene:** retrieved text is wrapped in `<context>` tags and the system
  prompt declares it untrusted data, to be cited, not obeyed.
- **Consistent history:** exchanges are saved only after a complete answer, so a failed stream
  never leaves an unanswered question that would corrupt later turns.
- **Cosine distance** (`hnsw:space=cosine`) makes `DISTANCE_THRESHOLD` meaningful: 0 is
  identical, higher is less related. With MiniLM, specific questions score ~0.3-0.55, short
  or generic ones (e.g. "What is the job title?") land around 0.7-0.8, and small talk such as
  "hi" scores ~0.8+. The default cutoff of 0.8 is therefore a loose pre-filter; the prompt makes
  the LLM refuse when the excerpts don't actually answer. Each request logs the nearest
  distance (`retrieval ... nearest=...`) so you can tune `DISTANCE_THRESHOLD` with the eval script.
- **Password hashing** uses PBKDF2-SHA256 via passlib, because passlib 1.7.4's bcrypt backend
  breaks with `bcrypt>=5`.

## Project layout

```
app/
  main.py              app factory, lifespan, router + rate-limit wiring
  config.py            pydantic-settings (.env)
  api/                 deps.py, auth.py, documents.py, chat.py, conversations.py, admin.py
  services/            embeddings.py, ingestion.py, retrieval.py, llm.py
  models/              db.py (SQLAlchemy), schemas.py (pydantic)
  core/                security.py, database.py, vectorstore.py, rate_limit.py
  static/              index.html, styles.css, app.js (web UI); admin.html, admin.css, admin.js
scripts/               inspect_chroma.py (read-only vector DB helper)
tests/                 pytest suite (fake embedder + fake LLM; no network/API key)
eval/                  run_eval.py, questions.json
```

## Tests

```powershell
pytest -q
```

The suite uses a deterministic fake embedder and a recording fake LLM, so it needs no model
download or API key. It covers chunking, auth, upload + background ingestion, per-user
isolation, the no-answer path (LLM not called), history windowing, SSE format, error events,
rate limiting, deleting documents, saved conversations, editing a question, and automatic
retry of temporary LLM errors, and the admin panel (access control, pagination, filters).

## Evaluating retrieval

1. Sign up and upload your PDFs, wait until they are `ready`.
2. Edit `eval/questions.json` (a list of objects):

   ```json
   [
     {"question": "What is the refund window?", "expected_source": "policy.pdf", "expected_page": 1},
     {"question": "How long is international shipping?", "expected_source": "policy.pdf", "contains": "14 days"},
     {"question": "Unrelated question", "expected_source": null}
   ]
   ```

   `expected_page` and `contains` are optional. `expected_source: null` marks a *negative*
   question that should be refused.
3. Run it (uses the same `.env` and the real embedding model; no LLM calls):

   ```powershell
   python eval/run_eval.py --user-email me@example.com
   python eval/run_eval.py --user-email me@example.com --top-k 3 --no-threshold
   ```

It prints per-question results plus **hit rate** (a matching chunk in the top-k), **MRR**,
and the **correct-refusal rate** for negative questions. Use `--no-threshold` to judge pure
ranking quality, then tune `DISTANCE_THRESHOLD` using the refusal rate.

## Admin panel

A built-in, read-only admin panel shows **all** data with pagination: users, documents,
conversations, messages and the ChromaDB chunks (plus an overview with counts).

1. Add your account's email to `.env` (comma-separated for several admins; empty means nobody is admin):

   ```env
   ADMIN_EMAILS=you@example.com
   ```

2. Restart the server, log in, and click **Admin** in the sidebar, or open <http://127.0.0.1:8000/admin>.

Each tab has search, a page-size selector (10/20/50/100), first/prev/next/last and jump-to-page.
Counts link to the related rows (for example a conversation's message count opens its messages).
Password hashes are never exposed. The page shell is public, but every `/admin/data/*` call is
checked on the server (`401` without a token, `403` for non-admins).

## Inspecting the databases

- **SQLite** (`data/app.db`: users, documents, chats): open it with any SQLite viewer, e.g.
  DB Browser for SQLite, or in the browser with `sqlite-web` (`pip install sqlite-web`).
- **ChromaDB** (`data/chroma/`: chunks + embeddings) has no viewer, so use the read-only helper:

  ```powershell
  python scripts/inspect_chroma.py                              # chunks per user/document
  python scripts/inspect_chroma.py --show 3                     # + first 3 chunks of each document
  python scripts/inspect_chroma.py --query "tech skills" --user 2   # nearest chunks + distances
  ```

  `--query` shows each hit's cosine distance and whether it passes `DISTANCE_THRESHOLD`, which is the
  quickest way to see why a question was (or wasn't) answered.

## Configuration

### Choosing the LLM (Anthropic, Gemini or Groq)

Only the generation step is provider-specific (embeddings are always local MiniLM). Pick one in `.env`:

```ini
# Anthropic (default)
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-5-5

# ...or Gemini (key from https://aistudio.google.com/apikey)
LLM_PROVIDER=gemini
GEMINI_API_KEY=AIza...
GEMINI_MODEL=gemini-3.8-flash

# ...or Groq (key from https://console.groq.com/keys; free tier available)
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...
GROQ_MODEL=qwen/qwen3.8-27b   # model access differs per account, see below
```

Restart the server after changing it; the startup log prints `LLM provider: ... (model: ...)`.

Groq (and Gemini) model access differs per account. If the chat says the model is not available, list
the models your key can use and put one into `GROQ_MODEL`:

```powershell
python -c "import groq; from app.config import get_settings as g; print(*sorted(m.id for m in groq.Groq(api_key=g().groq_api_key).models.list().data), sep='\n')"
```
Each adapter in `services/llm.py` turns its SDK's errors into `LLMAuthError` /
`LLMRateLimitError`, so the API layer and UI behave identically for both. Adding another
provider means adding one `_stream_<name>` function.

### All settings

All settings live in `.env` (see `.env.example`): `JWT_SECRET` (required, >= 32 chars),
`LLM_PROVIDER`, `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` (default `claude-sonnet-5-5`),
`GEMINI_API_KEY`, `GEMINI_MODEL` (default `gemini-3.8-flash`), `GROQ_API_KEY`,
`GROQ_MODEL` (default `qwen/qwen3.8-27b`), `CHUNK_SIZE`,
`CHUNK_OVERLAP`, `TOP_K`, `DISTANCE_THRESHOLD`, `HISTORY_MESSAGES`, `CHAT_RATE_LIMIT`,
`MAX_UPLOAD_MB`, `ADMIN_EMAILS`, and storage paths.

## Production notes / next steps

- Replace `create_all` with Alembic migrations; move SQLite to Postgres if you scale writes.
- Rate-limit storage is in-memory (per process). Use `Limiter(storage_uri="redis://...")` for
  multiple workers. Background ingestion is in-process; use a real queue (Celery/RQ/arq) for
  heavy loads.
- Add CORS configuration for browser clients, and query rewriting for follow-up questions
  (retrieval currently uses only the latest question).
- Temporary LLM errors (HTTP 429/5xx, e.g. Gemini's `503 high demand`) are retried up to twice
  with a short backoff, but only before any text has been streamed; if they persist, the UI
  shows "The language model is busy". See `_RETRY_DELAYS` in `services/llm.py`.
- Serve behind a reverse proxy with TLS; set `X-Accel-Buffering: no` (already sent) so SSE isn't buffered.
