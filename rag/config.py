"""Central config loaded from .env — single source of truth for all modules."""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    groq_api_key: str
    embedding_model: str = "all-MiniLM-L6-v2"
    chroma_persist_dir: str = "./chroma_db"
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    llm_model: str = "llama-3.1-8b-instant"
    collection_name: str = "documents"

    model_config = {"env_file": ".env"}


settings = Settings()
