"""
FastAPI REST API for the RAG pipeline.

Endpoints:
  POST /ingest      - upload a document file
  POST /query       - ask a question
  GET  /health      - liveness probe
  GET  /stats       - collection stats
"""
import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag.ingestion import ingest_file, get_collection
from rag.pipeline import query as rag_query

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
)

app = FastAPI(
    title="RAG Document Q&A",
    description="Ask questions over your documents using ChromaDB + Claude.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
def stats():
    collection = get_collection()
    return {
        "total_chunks": collection.count(),
        "collection": collection.name,
    }


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    allowed = {".pdf", ".txt", ".md"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported file type: {suffix}. Use: {allowed}")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        result = ingest_file(tmp_path)
        result["original_filename"] = file.filename
        return result
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/query")
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty.")
    return rag_query(req.question, top_k=req.top_k)
