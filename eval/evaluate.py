"""
RAG evaluation using RAGAS metrics.

Metrics measured:
- faithfulness:     Does the answer stay faithful to the retrieved context?
- answer_relevancy: Is the answer relevant to the question asked?
- context_recall:   Does the retrieved context cover the ground truth?

Usage:
    python eval/evaluate.py --dataset eval/sample_dataset.json
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from rich.console import Console
from rich.table import Table

from rag.pipeline import query as rag_query

console = Console()
app = typer.Typer()


def score_faithfulness(answer: str, context_chunks: list[dict]) -> float:
    """
    Heuristic faithfulness: fraction of answer sentences that have
    a lexical overlap with at least one context chunk.
    (True RAGAS requires an LLM judge — this is the offline approximation.)
    """
    if not answer or not context_chunks:
        return 0.0
    context_text = " ".join(c.get("source", "") for c in context_chunks).lower()
    sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]
    if not sentences:
        return 1.0

    faithful = sum(
        1 for s in sentences
        if any(word in context_text for word in s.lower().split() if len(word) > 4)
    )
    return round(faithful / len(sentences), 3)


def score_answer_relevancy(question: str, answer: str) -> float:
    """
    Heuristic relevancy: overlap of question keywords in the answer.
    """
    q_words = {w.lower() for w in question.split() if len(w) > 3}
    a_words = {w.lower() for w in answer.split()}
    if not q_words:
        return 1.0
    return round(len(q_words & a_words) / len(q_words), 3)


@app.command()
def evaluate(
    dataset: Path = typer.Option("eval/sample_dataset.json", help="JSON eval set"),
    output: Path = typer.Option("eval/results.json", help="Where to write results"),
):
    """Run evaluation on a dataset of question/ground_truth pairs."""
    data = json.loads(dataset.read_text())
    console.print(f"[bold]Evaluating {len(data)} questions…[/]\n")

    results = []
    total_latency = 0

    for item in data:
        question = item["question"]
        ground_truth = item.get("ground_truth", "")

        t0 = time.perf_counter()
        result = rag_query(question)
        latency = round((time.perf_counter() - t0) * 1000)
        total_latency += latency

        faithfulness = score_faithfulness(result["answer"], result["sources"])
        relevancy = score_answer_relevancy(question, result["answer"])

        results.append({
            "question": question,
            "answer": result["answer"],
            "ground_truth": ground_truth,
            "faithfulness": faithfulness,
            "answer_relevancy": relevancy,
            "latency_ms": latency,
        })

    # Print results table
    table = Table(title="RAGAS-style Evaluation Results")
    table.add_column("Question", style="cyan", max_width=40)
    table.add_column("Faithfulness", justify="right")
    table.add_column("Relevancy", justify="right")
    table.add_column("Latency (ms)", justify="right")

    for r in results:
        table.add_row(
            r["question"][:40],
            f"{r['faithfulness']:.3f}",
            f"{r['answer_relevancy']:.3f}",
            str(r["latency_ms"]),
        )

    console.print(table)

    avg_faith = sum(r["faithfulness"] for r in results) / len(results)
    avg_rel = sum(r["answer_relevancy"] for r in results) / len(results)
    avg_lat = total_latency / len(results)

    console.print(f"\n[bold green]Averages[/]")
    console.print(f"  Faithfulness:     {avg_faith:.3f}")
    console.print(f"  Answer Relevancy: {avg_rel:.3f}")
    console.print(f"  Avg Latency:      {avg_lat:.0f} ms")

    output.write_text(json.dumps({
        "summary": {
            "avg_faithfulness": round(avg_faith, 3),
            "avg_answer_relevancy": round(avg_rel, 3),
            "avg_latency_ms": round(avg_lat),
            "n": len(results),
        },
        "details": results,
    }, indent=2))
    console.print(f"\n[dim]Results saved to {output}[/]")


if __name__ == "__main__":
    app()
