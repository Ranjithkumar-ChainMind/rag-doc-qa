"""Pipeline tests with mocked Groq API — no real API calls needed."""
import pytest
from unittest.mock import patch, MagicMock

from rag.generation import generate_answer
from rag.retrieval import RetrievedChunk
from rag.pipeline import query


@pytest.fixture
def sample_chunks():
    return [
        RetrievedChunk(text="Python was created by Guido van Rossum.", source="python.txt", chunk_index=0, score=0.91),
        RetrievedChunk(text="Python 3.0 was released in 2008.", source="python.txt", chunk_index=1, score=0.85),
    ]


def test_generate_answer_no_chunks():
    result = generate_answer("anything", [])
    assert "No documents" in result["answer"]
    assert result["latency_ms"] == 0


def test_generate_answer_with_mock(sample_chunks):
    mock_choice = MagicMock()
    mock_choice.message.content = "Guido van Rossum created Python."
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.model = "llama-3.1-8b-instant"
    mock_response.usage.prompt_tokens = 150
    mock_response.usage.completion_tokens = 20

    with patch("rag.generation._client") as mock_client:
        mock_client.chat.completions.create.return_value = mock_response
        result = generate_answer("Who created Python?", sample_chunks)

    assert "Guido" in result["answer"]
    assert result["input_tokens"] == 150
    assert result["output_tokens"] == 20


def test_pipeline_query_mocked(sample_chunks):
    with patch("rag.pipeline.retrieve", return_value=sample_chunks), \
         patch("rag.pipeline.generate_answer") as mock_gen:
        mock_gen.return_value = {
            "answer": "Guido van Rossum.",
            "model": "llama-3.1-8b-instant",
            "input_tokens": 100,
            "output_tokens": 10,
            "latency_ms": 200,
        }
        result = query("Who created Python?")

    assert result["answer"] == "Guido van Rossum."
    assert len(result["sources"]) == 2
    assert result["metadata"]["retrieval_chunks"] == 2
