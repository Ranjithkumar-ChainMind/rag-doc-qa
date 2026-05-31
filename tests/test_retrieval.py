"""Integration tests for retrieval — uses an in-memory ChromaDB instance."""
import uuid
import pytest
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from unittest.mock import patch


@pytest.fixture
def in_memory_collection():
    """Ephemeral collection with a unique name so tests don't share state."""
    client = chromadb.EphemeralClient()
    ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    col = client.create_collection(f"test_{uuid.uuid4().hex[:8]}", embedding_function=ef)
    return col


def test_collection_upsert_and_query(in_memory_collection):
    col = in_memory_collection
    col.add(
        documents=["Python is a programming language.", "ChromaDB stores vectors."],
        ids=["doc1", "doc2"],
        metadatas=[{"source": "test.txt", "chunk_index": 0}, {"source": "test.txt", "chunk_index": 1}],
    )
    assert col.count() == 2

    results = col.query(query_texts=["programming"], n_results=1)
    assert "Python" in results["documents"][0][0]


def test_empty_collection_query(in_memory_collection):
    col = in_memory_collection
    assert col.count() == 0


def test_upsert_idempotent(in_memory_collection):
    col = in_memory_collection
    col.upsert(
        documents=["Same document."],
        ids=["same_id"],
        metadatas=[{"source": "x.txt", "chunk_index": 0}],
    )
    col.upsert(
        documents=["Same document."],
        ids=["same_id"],
        metadatas=[{"source": "x.txt", "chunk_index": 0}],
    )
    assert col.count() == 1  # not duplicated
