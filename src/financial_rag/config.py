"""Configuration management for Financial RAG evaluation system.

This module implements a two-layer configuration approach following MLOps best practices:
1. YAML file (config/app_config.yaml) - Application/business settings (version controlled)
2. Environment variables/.env - Secrets and infrastructure (not version controlled)
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def load_yaml_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load application configuration from YAML file.

    Args:
        config_path: Path to YAML config file. Defaults to config/app_config.yaml.

    Returns:
        Dictionary with configuration values.
    """
    if config_path is None:
        # Default to project_root/config/app_config.yaml
        config_path = Path(__file__).parent.parent.parent / "config" / "app_config.yaml"

    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


class Settings(BaseSettings):
    """Application settings combining YAML config and environment variables.

    Environment variables take precedence over YAML config for secrets.
    YAML config is used for application/business settings.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Project paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)

    # OpenAI (from .env)
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL"
    )

    # Anthropic (from .env)
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")

    # Google Cloud / Vertex AI (from .env)
    google_cloud_project: str = Field(default="", alias="GOOGLE_CLOUD_PROJECT")
    google_cloud_location: str = Field(default="europe-west1", alias="GOOGLE_CLOUD_LOCATION")

    # ChromaDB (from .env)
    chroma_persist_directory: str = Field(
        default="./data/chroma_db", alias="CHROMA_PERSIST_DIRECTORY"
    )
    chroma_collection_name: str = Field(
        default="financial_documents", alias="CHROMA_COLLECTION_NAME"
    )

    # Qdrant (from .env)
    qdrant_host: str = Field(default="localhost", alias="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, alias="QDRANT_PORT")
    qdrant_api_key: str = Field(default="", alias="QDRANT_API_KEY")
    qdrant_collection_name: str = Field(
        default="financial_documents", alias="QDRANT_COLLECTION_NAME"
    )

    @property
    def data_dir(self) -> Path:
        """Path to data directory."""
        return self.project_root / "data"

    @property
    def documents_dir(self) -> Path:
        """Path to financial documents directory."""
        return self.data_dir / "documents"

    @property
    def gold_qn_dir(self) -> Path:
        """Path to gold Q&A directory."""
        return self.data_dir / "gold_qn"

    @property
    def prompts_dir(self) -> Path:
        """Path to prompts directory."""
        return self.project_root / "prompts"

    @property
    def config_dir(self) -> Path:
        """Path to config directory."""
        return self.project_root / "config"


class AppConfig:
    """Application configuration loaded from YAML file.

    This class provides access to application/business settings that are
    appropriate for version control (non-sensitive configuration).
    """

    def __init__(self, config_path: Path | None = None):
        """Initialize application config from YAML.

        Args:
            config_path: Path to YAML config file.
        """
        self._config = load_yaml_config(config_path)

    @property
    def selected_documents(self) -> list[str]:
        """List of selected document filenames to process."""
        return self._config.get("documents", {}).get("selected", [])

    @property
    def question_types(self) -> list[str]:
        """List of question types to include in evaluation."""
        return self._config.get("test_data", {}).get("question_types", [])

    @property
    def chunk_types(self) -> list[str]:
        """List of chunk types to include in evaluation."""
        return self._config.get("test_data", {}).get("chunk_types", [])

    @property
    def top_k(self) -> int:
        """Number of documents to retrieve."""
        return self._config.get("rag", {}).get("top_k", 5)

    @property
    def chunk_size(self) -> int:
        """Maximum chunk size for document splitting."""
        return self._config.get("rag", {}).get("chunk_size", 1000)

    @property
    def chunk_overlap(self) -> int:
        """Overlap between chunks."""
        return self._config.get("rag", {}).get("chunk_overlap", 200)

    @property
    def chunker_tokenizer(self) -> str:
        """Tokenizer model for Docling HybridChunker."""
        return self._config.get("processing", {}).get(
            "chunker_tokenizer", "sentence-transformers/all-MiniLM-L6-v2"
        )

    @property
    def temperature(self) -> float:
        """LLM temperature setting."""
        return self._config.get("llm", {}).get("temperature", 0.1)

    @property
    def max_tokens(self) -> int:
        """Maximum tokens for LLM response."""
        return self._config.get("llm", {}).get("max_tokens", 1024)

    @property
    def max_test_cases(self) -> int | None:
        """Maximum number of test cases to run."""
        return self._config.get("evaluation", {}).get("max_test_cases")


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance (environment/infrastructure config)."""
    return Settings()


@lru_cache
def get_app_config() -> AppConfig:
    """Get cached application config instance (YAML config)."""
    return AppConfig()


# Model pricing per 1M tokens (USD)
# Sources:
# - OpenAI: https://openai.com/api/pricing/
# - Anthropic: https://docs.anthropic.com/en/docs/about-claude/models
# - Google: https://cloud.google.com/vertex-ai/generative-ai/pricing
# Updated: December 2025
MODEL_PRICING = {
    # OpenAI (per 1M tokens)
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    # Anthropic - Current models (per 1M tokens)
    # Claude 3.5 models retired July 2025, use Claude 4.x
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    # Google Gemini 2.5 (per 1M tokens) - current production models
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    "gemini-2.5-flash-lite": {"input": 0.075, "output": 0.30},
    # Embeddings (per 1M tokens)
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
}


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int = 0) -> float:
    """Calculate cost for a model call.

    Args:
        model: Model identifier (e.g., 'gpt-4o', 'claude-3-5-sonnet-20241022')
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens

    Returns:
        Cost in USD
    """
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        # Try to find a partial match
        for key in MODEL_PRICING:
            if key in model or model in key:
                pricing = MODEL_PRICING[key]
                break

    if not pricing:
        return 0.0

    # Pricing is per 1M tokens
    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost
