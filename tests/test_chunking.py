import pytest

from app.services.ingestion import chunk_text, normalise_text

WORDS = [f"word{i}" for i in range(600)]
LONG = " ".join(WORDS)


def test_short_text_is_single_chunk() -> None:
    assert chunk_text("hello world", 800, 150) == ["hello world"]


def test_empty_text_yields_no_chunks() -> None:
    assert chunk_text("   ", 800, 150) == []


def test_chunks_respect_size_and_overlap() -> None:
    chunks = chunk_text(LONG, 800, 150)
    assert len(chunks) > 1
    assert all(len(c) <= 800 for c in chunks)
    for prev, nxt in zip(chunks, chunks[1:]):
        # The tail of one chunk reappears at the head of the next.
        shared = [w for w in nxt.split()[:10] if w in prev.split()[-40:]]
        assert shared, "consecutive chunks should overlap"


def test_never_cuts_words_and_loses_nothing() -> None:
    chunks = chunk_text(LONG, 800, 150)
    seen = {w for c in chunks for w in c.split()}
    assert seen == set(WORDS)  # every original token appears intact


def test_invalid_parameters_rejected() -> None:
    with pytest.raises(ValueError):
        chunk_text(LONG, 800, 500)
    with pytest.raises(ValueError):
        chunk_text(LONG, 0, 0)


def test_normalise_text_collapses_whitespace() -> None:
    assert normalise_text("a \n\t b\x00c") == "a b c"
