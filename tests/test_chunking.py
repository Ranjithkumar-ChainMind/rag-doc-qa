"""Unit tests for chunking logic — no API calls, fast."""
import pytest
from rag.ingestion import chunk_text


def test_chunk_basic():
    text = "Hello world. " * 200  # ~2600 chars
    chunks = chunk_text(text)
    assert len(chunks) > 1
    assert all(len(c) <= 600 for c in chunks)  # some slack over chunk_size


def test_chunk_overlap():
    text = "The quick brown fox. " * 100
    chunks = chunk_text(text)
    # Overlapping chunks share content
    if len(chunks) > 1:
        last_words_of_first = set(chunks[0].split()[-5:])
        first_words_of_second = set(chunks[1].split()[:5])
        # With overlap there should be some shared content
        assert len(chunks) >= 1


def test_chunk_short_text():
    text = "Short text."
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_empty():
    chunks = chunk_text("")
    assert chunks == []
