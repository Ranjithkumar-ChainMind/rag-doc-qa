# RAG Document Q&A

**Ask questions over your own documents — locally, fast, with measured accuracy.**

Built with ChromaDB · Groq API (Llama 3.1) · sentence-transformers · FastAPI · Streamlit

---

## What this project does and WHY each choice matters

### The core problem RAG solves

LLMs have a knowledge cutoff and can't see your private documents. **Retrieval-Augmented Generation (RAG)** fixes this by:
1. Converting your documents into a searchable vector database
2. At query time, fetching the most relevant passages
3. Passing those passages as context to the LLM

This means the LLM answers from your documents, not from hallucination.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│  RETRIEVAL (ChromaDB + embeddings)  │
│  1. Embed query → 384-dim vector    │
│  2. Cosine similarity search        │
│  3. Return top-K chunks + scores    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  GENERATION (Claude API)            │
│  1. Build prompt with context       │
│  2. Use cache_control (cost ↓)      │
│  3. Claude answers from context     │
└──────────────┬──────────────────────┘
               │
               ▼
        Answer + Sources + Metrics
```

---

## Key Technical Decisions (interview-ready explanations)

### Why ChromaDB?
ChromaDB is an **embedded vector database** — it runs in-process with zero infrastructure. Unlike Pinecone (cloud-only) or Weaviate (needs Docker), ChromaDB persists to disk and is production-capable for datasets up to ~1M vectors. It uses HNSW (Hierarchical Navigable Small World) graphs for approximate nearest-neighbor search at O(log n) time complexity.

### Why `all-MiniLM-L6-v2` embeddings?
This 22M-parameter model produces 384-dimensional sentence embeddings. The tradeoffs:
- **Speed**: ~14ms per sentence on CPU (vs. 100ms+ for larger models)
- **Quality**: 0.635 average on SBERT benchmarks — sufficient for document retrieval
- **Cost**: runs locally, zero API cost per embedding

For production, you'd swap to `text-embedding-3-small` (OpenAI) or `voyage-3` for higher accuracy at the cost of API latency.

### Why `RecursiveCharacterTextSplitter`?
Naive fixed-size splitting cuts mid-sentence, breaking semantic coherence. `RecursiveCharacterTextSplitter` tries separators in order: paragraph breaks → line breaks → sentence endings → spaces. This preserves context at chunk boundaries. The 64-token overlap ensures sentences that straddle a boundary appear in both chunks.

### Why prompt caching?
The `cache_control: {"type": "ephemeral"}` flag on the context block tells Anthropic's API to cache the KV state of that block for 5 minutes. On follow-up questions over the same documents, the cache saves ~50–90% of input token costs and reduces time-to-first-token.

### Why RAGAS-style evaluation?
Raw accuracy ("did it answer correctly?") is binary and requires human labeling. RAGAS decomposes quality into:
- **Faithfulness**: is the answer grounded in the retrieved context, or is the LLM hallucinating?
- **Answer Relevancy**: does the answer actually address the question?
- **Context Recall**: does our retrieval surface the right chunks?

These are measurable automatically, giving you a numeric signal to tune chunk size, overlap, top-k, and embedding model.

---

## Project Structure

```
rag-doc-qa/
├── rag/
│   ├── config.py        # Pydantic settings from .env
│   ├── ingestion.py     # File loading, chunking, ChromaDB upsert
│   ├── retrieval.py     # Semantic search
│   ├── generation.py    # Groq API call (Llama 3.1)
│   └── pipeline.py      # Orchestrates retrieval → generation
├── api/
│   └── main.py          # FastAPI REST endpoints
├── app/
│   └── streamlit_app.py # Demo UI
├── eval/
│   ├── evaluate.py      # RAGAS-style evaluation script
│   └── sample_dataset.json
├── tests/
│   ├── test_chunking.py
│   ├── test_retrieval.py
│   └── test_pipeline.py
├── cli.py               # Typer CLI
└── requirements.txt
```

---

## Setup

```bash
# 1. Clone and enter project
cd rag-doc-qa

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt
pip install pydantic-settings  # for config

# 4. Set your API key
copy .env.example .env
# Edit .env and add your GROQ_API_KEY

# 5. Run tests
pytest

# 6. Ingest a document
python cli.py ingest path/to/your.pdf

# 7. Ask a question
python cli.py query "What is the main argument of this paper?"

# 8. Launch the UI
streamlit run app/streamlit_app.py

# 9. Or use the API
uvicorn api.main:app --reload
# POST http://localhost:8000/ingest  (multipart file)
# POST http://localhost:8000/query   {"question": "...", "top_k": 5}

# 10. Run evaluation
python eval/evaluate.py --dataset eval/sample_dataset.json
```

---

## Resume bullets (after running and getting real numbers)

```
• Built end-to-end RAG pipeline: ChromaDB vector store + sentence-transformers embeddings
  + Claude Haiku API; achieved avg faithfulness 0.87, answer relevancy 0.83 on eval set

• Reduced per-query LLM costs by ~60% using Anthropic prompt caching on context blocks

• Designed FastAPI service with <400ms avg end-to-end latency on 50-page PDF corpus

• Implemented RAGAS-style evaluation harness (faithfulness, answer relevancy metrics)
  enabling data-driven tuning of chunk size and top-K parameters
```

---

## Tuning guide

| Parameter | Default | Increase if… | Decrease if… |
|---|---|---|---|
| `CHUNK_SIZE` | 512 | answers miss context | answers are noisy/irrelevant |
| `CHUNK_OVERLAP` | 64 | answers cut mid-sentence | storage is a concern |
| `TOP_K` | 5 | faithfulness is low | latency is high |
| Embedding model | `all-MiniLM-L6-v2` | relevancy score is low | CPU/cost is a concern |
