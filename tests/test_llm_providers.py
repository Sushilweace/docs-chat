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
        (500, "Internal error", genai_errors.APIError),
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
