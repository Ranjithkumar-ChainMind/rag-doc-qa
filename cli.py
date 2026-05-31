"""
Command-line interface for the RAG pipeline.

Usage:
  python cli.py ingest docs/
  python cli.py ingest paper.pdf
  python cli.py query "What does the paper say about transformers?"
  python cli.py stats
"""
import sys
from pathlib import Path

import typer

from rag.ingestion import ingest_file, ingest_directory, get_collection
from rag.pipeline import query as rag_query

app = typer.Typer(help="RAG Document Q&A -- CLI")


@app.command()
def ingest(path: Path = typer.Argument(..., help="File or directory to ingest")):
    """Ingest documents into the vector store."""
    if path.is_dir():
        results = ingest_directory(path)
        total = sum(r["chunks"] for r in results)
        print(f"\nTotal chunks stored: {total}")
    elif path.is_file():
        result = ingest_file(path)
        print(result)
    else:
        print(f"Path not found: {path}")
        raise typer.Exit(1)


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask"),
    top_k: int = typer.Option(5, help="Number of chunks to retrieve"),
):
    """Ask a question over ingested documents."""
    result = rag_query(question, top_k=top_k)

    print(f"\nQ: {result['question']}")
    print("-" * 60)
    print(f"\nA: {result['answer']}\n")

    print("Sources:")
    for s in result["sources"]:
        print(f"  - {s['source']} (chunk {s['chunk_index']}, score={s['score']})")

    m = result["metadata"]
    print(f"\nLatency: {m['total_latency_ms']}ms | Tokens in: {m['input_tokens']} out: {m['output_tokens']}")


@app.command()
def stats():
    """Show vector store statistics."""
    collection = get_collection()
    print(f"Collection: {collection.name}")
    print(f"Chunks stored: {collection.count()}")


if __name__ == "__main__":
    app()
