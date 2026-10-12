"""Provider selection, Gemini message conversion and error translation (no network)."""

from types import SimpleNamespace

import pytest
from google.genai import errors as genai_errors

from app.config import get_settings
from app.services import llm
from app.services.retrieval import RetrievedChunk

CHUNKS = [RetrievedChunk(text="Paris is in France.", source="geo.pdf", page=1, distance=0.1)]


def _use(monkeypatch: pytest.MonkeyPatch, **overrides) -> None:
    """Point llm at a modified copy of the settings."""
    monkeypatch.setattr(llm, "get_settings", lambda: get_settings().model_copy(update=overrides))


def test_gemini_contents_use_model_role_and_keep_context() -> None:
    history = [llm.HistoryTurn("user", "q1"), llm.HistoryTurn("assistant", "a1")]
    contents = llm.build_gemini_contents(llm.build_messages("q2", CHUNKS, history))

    assert [c.role for c in contents] == ["user", "model", "user"]
    assert contents[1].parts[0].text == "a1"
    assert "[1] (source: geo.pdf, page 1)" in contents[2].parts[0].text


def test_dispatches_on_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "_stream_gemini", lambda m: iter(["gem"]))
    monkeypatch.setattr(llm, "_stream_anthropic", lambda m: iter(["ant"]))

    _use(monkeypatch, llm_provider="gemini")
    assert "".join(llm.stream_answer("q", CHUNKS, [])) == "gem"
    _use(monkeypatch, llm_provider="anthropic")
    assert "".join(llm.stream_answer("q", CHUNKS, [])) == "ant"


def test_settings_expose_active_provider_key_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    base = get_settings().model_copy(
        update={"gemini_api_key": "g-key", "anthropic_api_key": "a-key", "gemini_model": "gm"}
    )
    gemini = base.model_copy(update={"llm_provider": "gemini"})
    anthropic_ = base.model_copy(update={"llm_provider": "anthropic"})
    assert (gemini.llm_api_key, gemini.llm_model) == ("g-key", "gm")
    assert (anthropic_.llm_api_key, anthropic_.llm_model) == ("a-key", anthropic_.anthropic_model)


def test_gemini_without_key_is_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    llm._get_gemini_client.cache_clear()
    _use(monkeypatch, llm_provider="gemini", gemini_api_key="")
    with pytest.raises(llm.LLMNotConfiguredError):
        list(llm.stream_answer("q", CHUNKS, []))


def _fake_gemini(monkeypatch: pytest.MonkeyPatch, stream) -> None:
    client = SimpleNamespace(models=SimpleNamespace(generate_content_stream=stream))
    monkeypatch.setattr(llm, "_get_gemini_client", lambda: client)
    _use(monkeypatch, llm_provider="gemini", gemini_api_key="k")


def test_gemini_streams_text_and_skips_empty_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    def stream(**kwargs):
        assert kwargs["config"].system_instruction == llm.SYSTEM_PROMPT
        yield SimpleNamespace(text="Hello ")
        yield SimpleNamespace(text=None)  # metadata-only chunk
        yield SimpleNamespace(text="world [1].")

    _fake_gemini(monkeypatch, stream)
    assert "".join(llm.stream_answer("q", CHUNKS, [])) == "Hello world [1]."


@pytest.mark.parametrize(
    ("code", "message", "expected"),
    [
        (400, "API key not valid. Please pass a valid API key.", llm.LLMAuthError),
        (403, "Permission denied", llm.LLMAuthError),
        (429, "Quota exceeded", llm.LLMRateLimitError),
        (503, "This model is currently experiencing high demand.", llm.LLMRateLimitError),
        (504, "Deadline exceeded", llm.LLMRateLimitError),
        (404, "Model not found", llm.LLMNotConfiguredError),  # retired / unknown model name
    ],
)
def test_gemini_errors_are_translated(monkeypatch, code, message, expected) -> None:
    def stream(**_kwargs):
        raise genai_errors.APIError(code, {"error": {"message": message, "status": "X"}})
        yield  # pragma: no cover

    _fake_gemini(monkeypatch, stream)
    with pytest.raises(expected):
        list(llm.stream_answer("q", CHUNKS, []))


def test_gemini_unrelated_400_is_not_reported_as_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    def stream(**_kwargs):
        raise genai_errors.APIError(400, {"error": {"message": "Bad request body", "status": "X"}})
        yield  # pragma: no cover

    _fake_gemini(monkeypatch, stream)
    with pytest.raises(genai_errors.APIError):
        list(llm.stream_answer("q", CHUNKS, []))

# --- automatic retry of transient errors ----------------------------------------------------
@pytest.fixture
def retry_delays(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Make retries instant but record the waits that would have happened."""
    waited: list[float] = []
    monkeypatch.setattr(llm, "_RETRY_DELAYS", (1.0, 3.0))
    monkeypatch.setattr(llm.time, "sleep", waited.append)
    return waited


def _overloaded() -> genai_errors.APIError:
    return genai_errors.APIError(503, {"error": {"message": "high demand", "status": "UNAVAILABLE"}})


def test_transient_error_is_retried_and_then_succeeds(monkeypatch, retry_delays) -> None:
    attempts = {"n": 0}

    def stream(**_kwargs):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise _overloaded()
        yield SimpleNamespace(text="finally [1].")

    _fake_gemini(monkeypatch, stream)
    assert "".join(llm.stream_answer("q", CHUNKS, [])) == "finally [1]."
    assert attempts["n"] == 3
    assert retry_delays == [1.0, 3.0]


def test_gives_up_after_the_last_retry(monkeypatch, retry_delays) -> None:
    attempts = {"n": 0}

    def stream(**_kwargs):
        attempts["n"] += 1
        raise _overloaded()
        yield  # pragma: no cover

    _fake_gemini(monkeypatch, stream)
    with pytest.raises(llm.LLMRateLimitError):
        list(llm.stream_answer("q", CHUNKS, []))
    assert attempts["n"] == 3  # first try + 2 retries


def test_no_retry_once_text_has_been_streamed(monkeypatch, retry_delays) -> None:
    attempts = {"n": 0}

    def stream(**_kwargs):
        attempts["n"] += 1
        yield SimpleNamespace(text="partial answer ")
        raise _overloaded()

    _fake_gemini(monkeypatch, stream)
    received: list[str] = []
    with pytest.raises(llm.LLMRateLimitError):
        for piece in llm.stream_answer("q", CHUNKS, []):
            received.append(piece)
    assert received == ["partial answer "]  # nothing repeated or restarted
    assert attempts["n"] == 1 and retry_delays == []


def test_auth_errors_are_never_retried(monkeypatch, retry_delays) -> None:
    attempts = {"n": 0}

    def stream(**_kwargs):
        attempts["n"] += 1
        raise genai_errors.APIError(403, {"error": {"message": "denied", "status": "X"}})
        yield  # pragma: no cover

    _fake_gemini(monkeypatch, stream)
    with pytest.raises(llm.LLMAuthError):
        list(llm.stream_answer("q", CHUNKS, []))
    assert attempts["n"] == 1

# --- Groq ------------------------------------------------------------------------------------
def _groq_error(cls, status: int, message: str = "boom"):
    """Build a real groq SDK exception (they need an httpx response)."""
    import httpx

    response = httpx.Response(status, request=httpx.Request("POST", "https://api.groq.com/x"))
    return cls(message, response=response, body=None)


def _fake_groq(monkeypatch: pytest.MonkeyPatch, create) -> None:
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(llm, "_get_groq_client", lambda: client)
    _use(monkeypatch, llm_provider="groq", groq_api_key="k", groq_model="test-model")


def _groq_chunk(text):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


def test_settings_expose_groq_key_and_model() -> None:
    settings = get_settings().model_copy(
        update={"llm_provider": "groq", "groq_api_key": "gr-key", "groq_model": "llama-x"}
    )
    assert (settings.llm_api_key, settings.llm_model) == ("gr-key", "llama-x")


def test_dispatches_to_groq(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "_stream_groq", lambda m: iter(["groq says hi"]))
    _use(monkeypatch, llm_provider="groq")
    assert "".join(llm.stream_answer("q", CHUNKS, [])) == "groq says hi"


def test_groq_without_key_is_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    llm._get_groq_client.cache_clear()
    _use(monkeypatch, llm_provider="groq", groq_api_key="")
    with pytest.raises(llm.LLMNotConfiguredError):
        list(llm.stream_answer("q", CHUNKS, []))


def test_groq_streams_text_sends_system_prompt_and_skips_empty_chunks(monkeypatch) -> None:
    seen: dict = {}

    def create(**kwargs):
        seen.update(kwargs)
        yield _groq_chunk("Hello ")
        yield _groq_chunk(None)  # role/finish chunks carry no text
        yield SimpleNamespace(choices=[])  # usage-only chunk
        yield _groq_chunk("world [1].")

    _fake_groq(monkeypatch, create)
    history = [llm.HistoryTurn("user", "q1"), llm.HistoryTurn("assistant", "a1")]

    assert "".join(llm.stream_answer("q2", CHUNKS, history)) == "Hello world [1]."
    assert seen["stream"] is True and seen["model"] == "test-model"
    assert [m["role"] for m in seen["messages"]] == ["system", "user", "assistant", "user"]
    assert seen["messages"][0]["content"] == llm.SYSTEM_PROMPT


def test_groq_errors_are_translated(monkeypatch, retry_delays) -> None:
    import groq

    cases = [
        (groq.AuthenticationError, 401, llm.LLMAuthError),
        (groq.PermissionDeniedError, 403, llm.LLMAuthError),
        (groq.RateLimitError, 429, llm.LLMRateLimitError),
        (groq.InternalServerError, 503, llm.LLMRateLimitError),  # "over capacity"
        (groq.NotFoundError, 404, llm.LLMNotConfiguredError),  # model_not_found -> clear message
    ]
    for sdk_error, status, expected in cases:

        def create(**_kwargs):
            raise _groq_error(sdk_error, status)
            yield  # pragma: no cover

        _fake_groq(monkeypatch, create)
        with pytest.raises(expected):
            list(llm.stream_answer("q", CHUNKS, []))

def test_unknown_model_error_names_the_setting_to_fix(monkeypatch) -> None:
    import groq

    def create(**_kwargs):
        raise _groq_error(groq.NotFoundError, 404, "The model `nope` does not exist")
        yield  # pragma: no cover

    _fake_groq(monkeypatch, create)
    with pytest.raises(llm.LLMNotConfiguredError) as info:
        list(llm.stream_answer("q", CHUNKS, []))
    assert "GROQ_MODEL" in str(info.value) and "test-model" in str(info.value)
