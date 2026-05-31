"""
Central config — works in two environments:
- Local: reads from .env file
- Streamlit Cloud: reads from st.secrets (set in the Streamlit dashboard)
"""
import os

def _get(key: str, default: str = "") -> str:
    """Try st.secrets first (cloud), fall back to env var (local)."""
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)


class Settings:
    @property
    def groq_api_key(self) -> str:
        return _get("GROQ_API_KEY")

    @property
    def embedding_model(self) -> str:
        return _get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    @property
    def chroma_persist_dir(self) -> str:
        return _get("CHROMA_PERSIST_DIR", "./chroma_db")

    @property
    def chunk_size(self) -> int:
        return int(_get("CHUNK_SIZE", "512"))

    @property
    def chunk_overlap(self) -> int:
        return int(_get("CHUNK_OVERLAP", "64"))

    @property
    def top_k(self) -> int:
        return int(_get("TOP_K", "5"))

    @property
    def llm_model(self) -> str:
        return _get("LLM_MODEL", "llama-3.1-8b-instant")

    @property
    def collection_name(self) -> str:
        return "documents"


settings = Settings()
