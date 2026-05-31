"""
Generation layer: Groq API with retrieved context.

Groq runs Llama 3.1 on their cloud — free tier, very fast (~200ms).
"""
import time
import logging
from typing import List

from groq import Groq

from rag.config import settings
from rag.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

_client = Groq(api_key=settings.groq_api_key)

SYSTEM_PROMPT = """You are a precise document assistant. Answer questions using ONLY the provided context.
Rules:
- If the context doesn't contain the answer, say "I don't have enough information in the provided documents."
- Cite sources inline like [source.pdf, chunk 3].
- Be concise and factual. No speculation."""


def build_context_block(chunks: List[RetrievedChunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks):
        parts.append(
            f"[Context {i+1} | {chunk.source}, chunk {chunk.chunk_index} | relevance: {chunk.score}]\n{chunk.text}"
        )
    return "\n\n---\n\n".join(parts)


def generate_answer(query: str, chunks: List[RetrievedChunk]) -> dict:
    """
    Call Groq (Llama 3.1) with retrieved context.
    Returns answer + usage metadata.
    """
    if not chunks:
        return {
            "answer": "No documents have been ingested yet. Please upload documents first.",
            "model": settings.llm_model,
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms": 0,
        }

    context = build_context_block(chunks)

    t0 = time.perf_counter()
    response = _client.chat.completions.create(
        model=settings.llm_model,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"<context>\n{context}\n</context>\n\nQuestion: {query}",
            },
        ],
    )
    latency_ms = round((time.perf_counter() - t0) * 1000)

    return {
        "answer": response.choices[0].message.content,
        "model": response.model,
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "latency_ms": latency_ms,
    }
