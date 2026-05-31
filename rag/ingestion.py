"""
Document ingestion pipeline.

Flow: file -> load -> chunk -> embed -> store in ChromaDB

Supports PDF, TXT, and Markdown files.
"""
import time
import logging
from pathlib import Path
from typing import List

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from rich.console import Console
from rich.progress import track

from rag.config import settings

logger = logging.getLogger(__name__)
console = Console(highlight=False)


def load_file(path: Path) -> str:
    """Load raw text from PDF, TXT, or MD."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    elif suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")
    raise ValueError(f"Unsupported file type: {suffix}")


def chunk_text(text: str) -> List[str]:
    """Split text into overlapping chunks for better context preservation."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name=settings.embedding_model
    )
    return client.get_or_create_collection(
        name=settings.collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


def ingest_file(path: Path) -> dict:
    """
    Ingest a single file into ChromaDB.
    Returns metadata dict with chunk count and timing.
    """
    t0 = time.perf_counter()
    print(f"Ingesting {path.name}...")

    text = load_file(path)
    chunks = chunk_text(text)
    print(f"  >> {len(chunks)} chunks from {len(text):,} characters")

    collection = get_collection()

    doc_id = path.stem
    ids = [f"{doc_id}__chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": path.name, "chunk_index": i} for i in range(len(chunks))]

    collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)

    elapsed = time.perf_counter() - t0
    result = {
        "file": path.name,
        "chunks": len(chunks),
        "elapsed_s": round(elapsed, 2),
    }
    print(f"  Done in {elapsed:.2f}s")
    return result


def ingest_directory(directory: Path) -> List[dict]:
    """Ingest all supported files in a directory."""
    supported = {".pdf", ".txt", ".md"}
    files = [f for f in directory.iterdir() if f.suffix.lower() in supported]
    if not files:
        print("No supported files found.")
        return []
    return [ingest_file(f) for f in files]
