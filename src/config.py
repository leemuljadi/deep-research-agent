"""Configuration loaded from environment (.env)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root regardless of CWD.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    # Postgres / pgvector
    pg_host: str = field(default_factory=lambda: os.getenv("POSTGRES_HOST", "localhost"))
    pg_port: int = field(default_factory=lambda: int(os.getenv("POSTGRES_PORT", "5432")))
    pg_user: str = field(default_factory=lambda: os.getenv("POSTGRES_USER", "dra"))
    pg_password: str = field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD", "dra"))
    pg_db: str = field(default_factory=lambda: os.getenv("POSTGRES_DB", "dra"))

    # LLM routing
    chat_model: str = field(default_factory=lambda: os.getenv("LITELLM_MODEL_CHAT", "ollama/llama3.2"))
    chat_model_fallback: str | None = field(
        default_factory=lambda: os.getenv("LITELLM_MODEL_CHAT_FALLBACK") or None
    )
    embed_model: str = field(default_factory=lambda: os.getenv("LITELLM_MODEL_EMBED", "ollama/nomic-embed-text"))
    embedding_dim: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "768")))

    # Retrieval
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "800")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "100")))
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "5")))

    # Observability
    langfuse_host: str | None = field(default_factory=lambda: os.getenv("LANGFUSE_HOST") or None)
    langfuse_public_key: str | None = field(default_factory=lambda: os.getenv("LANGFUSE_PUBLIC_KEY") or None)
    langfuse_secret_key: str | None = field(default_factory=lambda: os.getenv("LANGFUSE_SECRET_KEY") or None)

    # Azure AI Search (optional)
    azure_search_enabled: bool = field(
        default_factory=lambda: os.getenv("AZURE_SEARCH_ENABLED", "false").lower() == "true"
    )
    azure_search_endpoint: str | None = field(default_factory=lambda: os.getenv("AZURE_SEARCH_ENDPOINT") or None)
    azure_search_key: str | None = field(default_factory=lambda: os.getenv("AZURE_SEARCH_KEY") or None)
    azure_search_index: str = field(default_factory=lambda: os.getenv("AZURE_SEARCH_INDEX", "deep-research"))

    @property
    def pg_dsn(self) -> str:
        return (
            f"postgresql://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_db}"
        )


settings = Settings()
