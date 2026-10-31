import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    groq_api_key: str | None
    model: str
    persist_dir: str
    collection: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    min_score: float


def load_settings() -> Settings:
    """Read settings from environment variables (and a local .env file)."""
    return Settings(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        persist_dir=os.getenv("CHROMA_DIR", ".chroma"),
        collection=os.getenv("CHROMA_COLLECTION", "documents"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "100")),
        top_k=int(os.getenv("TOP_K", "4")),
        min_score=float(os.getenv("MIN_SCORE", "0.25")),
    )
