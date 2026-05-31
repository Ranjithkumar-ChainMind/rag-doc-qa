"""
RAG pipeline: the single entry point that wires retrieval → generation.

This is the function your API and UI both call.
"""
import logging
import time

from rag.config import settings
from rag.retrieval import retrieve
from rag.generation import generate_answer

logger = logging.getLogger(__name__)


def query(question: str, top_k: int | None = None) -> dict:
    """
    Full RAG query:
    1. Embed question and find top-k chunks (semantic search)
    2. Pass chunks + question to Claude
    3. Return answer with full provenance (sources, scores, tokens, latency)
    """
    t0 = time.perf_counter()

    chunks = retrieve(question, top_k=top_k)
    gen = generate_answer(question, chunks)

    total_ms = round((time.perf_counter() - t0) * 1000)

    return {
        "question": question,
        "answer": gen["answer"],
        "sources": [
            {"source": c.source, "chunk_index": c.chunk_index, "score": c.score}
            for c in chunks
        ],
        "metadata": {
            "model": gen["model"],
            "input_tokens": gen["input_tokens"],
            "output_tokens": gen["output_tokens"],
            "retrieval_chunks": len(chunks),
            "generation_latency_ms": gen["latency_ms"],
            "total_latency_ms": total_ms,
        },
    }
