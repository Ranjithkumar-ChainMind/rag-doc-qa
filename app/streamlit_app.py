"""
Streamlit demo UI for the RAG pipeline.

Run: streamlit run app/streamlit_app.py --browser.gatherUsageStats false
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tempfile
import streamlit as st

from rag.ingestion import ingest_file, get_collection
from rag.pipeline import query as rag_query

st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="🔍",
    layout="wide",
)

# --- Sidebar: document upload ---
with st.sidebar:
    st.title("RAG Document Q&A")
    st.caption("ChromaDB · Groq (Llama 3.1) · sentence-transformers")
    st.divider()

    st.subheader("1. Upload Documents")
    uploaded = st.file_uploader(
        "PDF, TXT, or Markdown",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded and st.button("Ingest Files", type="primary"):
        for uf in uploaded:
            suffix = Path(uf.name).suffix
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(uf.read())
                tmp_path = Path(tmp.name)
            with st.spinner(f"Ingesting {uf.name}..."):
                result = ingest_file(tmp_path)
                result["original_filename"] = uf.name
            st.success(f"{uf.name} -> {result['chunks']} chunks ({result['elapsed_s']}s)")
            tmp_path.unlink(missing_ok=True)

    st.divider()
    collection = get_collection()
    st.metric("Chunks in store", collection.count())
    st.caption("Embeddings: all-MiniLM-L6-v2\nLLM: Groq llama-3.1-8b-instant")

# --- Main: Q&A ---
st.header("2. Ask a Question")

col1, col2 = st.columns([3, 1])
with col1:
    question = st.text_input(
        "Your question",
        placeholder="What does the document say about...?",
        label_visibility="collapsed",
    )
with col2:
    top_k = st.number_input("Top-K chunks", min_value=1, max_value=20, value=5)

if st.button("Ask", type="primary", disabled=not question):
    with st.spinner("Retrieving and generating..."):
        result = rag_query(question, top_k=top_k)

    st.subheader("Answer")
    st.markdown(result["answer"])

    m = result["metadata"]
    cols = st.columns(4)
    cols[0].metric("Total latency", f"{m['total_latency_ms']} ms")
    cols[1].metric("Chunks used", m["retrieval_chunks"])
    cols[2].metric("Input tokens", m["input_tokens"])
    cols[3].metric("Output tokens", m["output_tokens"])

    with st.expander("Retrieved sources"):
        for s in result["sources"]:
            st.markdown(
                f"**{s['source']}** · chunk {s['chunk_index']} · "
                f"relevance score: `{s['score']}`"
            )
