"""
Retrieval layer: semantic search over ChromaDB.

Returns ranked chunks with scores for transparency.
"""
import time
from dataclasses import dataclass
from typing import List

from rag.config import settings
from rag.ingestion import get_collection


@dataclass
class RetrievedChunk:
    text: str
    source: str
    chunk_index: int
    score: float  # cosine similarity (0–1, higher = more relevant)


def retrieve(query: str, top_k: int | None = None) -> List[RetrievedChunk]:
    """
    Semantic search: embed query → find nearest chunks in vector space.

    ChromaDB returns distance (0=identical, 2=opposite for cosine).
    We convert to similarity = 1 - distance/2 for readability.
    """
    k = top_k or settings.top_k
    collection = get_collection()

    if collection.count() == 0:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append(
            RetrievedChunk(
                text=doc,
                source=meta.get("source", "unknown"),
                chunk_index=meta.get("chunk_index", 0),
                score=round(1 - dist / 2, 4),
            )
        )
    return chunks
