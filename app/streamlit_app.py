"""
Streamlit demo UI for the RAG pipeline.
Works locally and on Streamlit Community Cloud (free).

Local:  set GROQ_API_KEY in .env
Cloud:  set GROQ_API_KEY in Streamlit dashboard -> Settings -> Secrets
"""
import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env for local development
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="🔍",
    layout="wide",
)

# --- Lazy imports so the app loads fast ---
@st.cache_resource(show_spinner="Loading embedding model...")
def get_collection():
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    client = chromadb.EphemeralClient()
    ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    return client.get_or_create_collection("documents", embedding_function=ef,
                                           metadata={"hnsw:space": "cosine"})


def ingest_uploaded_file(uploaded_file) -> dict:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from pypdf import PdfReader
    import time

    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(uploaded_file)
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        text = uploaded_file.read().decode("utf-8")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, chunk_overlap=64,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)

    collection = get_collection()
    doc_id = Path(uploaded_file.name).stem
    ids = [f"{doc_id}__chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": uploaded_file.name, "chunk_index": i} for i in range(len(chunks))]
    collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)

    return {"file": uploaded_file.name, "chunks": len(chunks)}


def rag_query(question: str, top_k: int = 5) -> dict:
    import time
    from groq import Groq

    # Get API key from st.secrets (cloud) or .env (local)
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return {"answer": "GROQ_API_KEY not set. Add it in Settings > Secrets.", "sources": [], "metadata": {}}

    collection = get_collection()
    if collection.count() == 0:
        return {"answer": "No documents ingested yet. Upload a file first.", "sources": [], "metadata": {}}

    t0 = time.perf_counter()

    # Retrieve
    results = collection.query(query_texts=[question], n_results=min(top_k, collection.count()),
                               include=["documents", "metadatas", "distances"])
    chunks = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        chunks.append({"text": doc, "source": meta.get("source", "?"),
                       "chunk_index": meta.get("chunk_index", 0),
                       "score": round(1 - dist / 2, 4)})

    # Build context
    context = "\n\n---\n\n".join(
        f"[Context {i+1} | {c['source']}, chunk {c['chunk_index']} | score: {c['score']}]\n{c['text']}"
        for i, c in enumerate(chunks)
    )

    # Generate
    client = Groq(api_key=api_key)
    t1 = time.perf_counter()
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=1024,
        messages=[
            {"role": "system", "content": "Answer using ONLY the provided context. Cite sources like [source, chunk N]. If context lacks the answer, say so."},
            {"role": "user", "content": f"<context>\n{context}\n</context>\n\nQuestion: {question}"}
        ]
    )
    gen_ms = round((time.perf_counter() - t1) * 1000)
    total_ms = round((time.perf_counter() - t0) * 1000)

    return {
        "answer": resp.choices[0].message.content,
        "sources": chunks,
        "metadata": {
            "model": resp.model,
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
            "generation_ms": gen_ms,
            "total_ms": total_ms,
            "chunks_retrieved": len(chunks),
        }
    }


# ── UI ─────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("RAG Document Q&A")
    st.caption("ChromaDB · Groq Llama 3.1 · sentence-transformers")
    st.caption("100% Free · No credit card needed")
    st.divider()

    st.subheader("1. Upload Documents")
    uploaded_files = st.file_uploader(
        "PDF, TXT, or Markdown",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Ingest Files", type="primary"):
        for f in uploaded_files:
            with st.spinner(f"Ingesting {f.name}..."):
                result = ingest_uploaded_file(f)
            st.success(f"{result['file']} -> {result['chunks']} chunks")

    st.divider()
    collection = get_collection()
    st.metric("Chunks in store", collection.count())

st.title("Ask Questions About Your Documents")
st.caption("Upload a PDF, TXT, or Markdown file in the sidebar, then ask anything.")
st.divider()

col1, col2 = st.columns([4, 1])
with col1:
    question = st.text_input("Your question", placeholder="What does the document say about...?",
                             label_visibility="collapsed")
with col2:
    top_k = st.number_input("Top-K", min_value=1, max_value=20, value=5)

if st.button("Ask", type="primary", disabled=not question):
    with st.spinner("Searching and generating answer..."):
        result = rag_query(question, top_k=top_k)

    st.subheader("Answer")
    st.markdown(result["answer"])

    if result.get("metadata"):
        m = result["metadata"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total latency", f"{m.get('total_ms', 0)} ms")
        c2.metric("Chunks used", m.get("chunks_retrieved", 0))
        c3.metric("Input tokens", m.get("input_tokens", 0))
        c4.metric("Output tokens", m.get("output_tokens", 0))

    if result.get("sources"):
        with st.expander("Retrieved Sources"):
            for s in result["sources"]:
                st.markdown(f"**{s['source']}** · chunk {s['chunk_index']} · score `{s['score']}`")
