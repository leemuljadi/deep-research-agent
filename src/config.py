"""Configuration loaded from environment (.env)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root regardless of CWD.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")



def _cost_cap_env(raw: str | None) -> float | None:
    """Parse RUN_COST_CAP_USD; unset/empty/garbage → None (uncapped, AD-15)."""
    if raw is None or not raw.strip():
        return None
    try:
        cap = float(raw)
    except ValueError:
        return None
    if cap != cap or cap in (float("inf"), float("-inf")):  # NaN / ±Inf
        return None
    return cap


def _cost_cap_warning(raw: str | None) -> str | None:
    """Non-fatal warning text when RUN_COST_CAP_USD is set but unparseable —
    surfaced by the worker/api at startup; the run proceeds uncapped."""
    if raw is None or not raw.strip():
        return None
    try:
        cap = float(raw)
    except ValueError:
        return f"ignoring unparseable RUN_COST_CAP_USD={raw!r}; runs are uncapped"
    if cap != cap or cap in (float("inf"), float("-inf")):
        return f"ignoring non-finite RUN_COST_CAP_USD={raw!r}; runs are uncapped"
    return None

def _tool_timeout_env(raw: str | None) -> float:
    """Parse TOOL_TIMEOUT_SECONDS; unset/empty/garbage/non-finite/non-positive
    → 30.0 default (AD-16 per-call tool timeout)."""
    if raw is None or not raw.strip():
        return 30.0
    try:
        val = float(raw)
    except ValueError:
        return 30.0
    if val != val or val in (float("inf"), float("-inf")) or val <= 0:  # NaN / ±Inf
        return 30.0
    return val


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

    # Per-run cost cap in USD (AD-15). Unset/garbage = uncapped — the AD-13
    # zero-key path never trips a cap. Per-run override: job row
    # `cost_cap_usd` (worker picks row value first).
    run_cost_cap_usd: float | None = field(
        default_factory=lambda: _cost_cap_env(os.getenv("RUN_COST_CAP_USD"))
    )
    run_cost_cap_warning: str | None = field(
        default_factory=lambda: _cost_cap_warning(os.getenv("RUN_COST_CAP_USD"))
    )

    # Per-call tool timeout in seconds (AD-16): every tool call terminates
    # within this window with a terminal ToolResult — never an unbounded wait.
    tool_timeout_seconds: float = field(
        default_factory=lambda: _tool_timeout_env(os.getenv("TOOL_TIMEOUT_SECONDS"))
    )

    @property
    def pg_dsn(self) -> str:
        return (
            f"postgresql://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_db}"
        )


settings = Settings()
