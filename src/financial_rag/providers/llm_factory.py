"""LLM Factory - Single source of truth for LLM instantiation.

This module eliminates code duplication across providers by centralizing
LLM creation logic. Supports OpenAI, Anthropic, and Google Vertex AI.
"""

from dataclasses import dataclass
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI

from financial_rag import get_logger
from financial_rag.config import Settings, get_settings

logger = get_logger("providers.llm_factory")

# Default models for each provider
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
    "google": "gemini-2.5-flash",
}


@dataclass
class LLMInstance:
    """Container for LLM instance and its metadata."""

    llm: Any
    model_name: str
    provider: str


def create_llm(
    provider: str = "openai",
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 1024,
    settings: Settings | None = None,
) -> LLMInstance:
    """Create an LLM instance based on provider configuration.

    Args:
        provider: LLM provider name ("openai", "anthropic", or "google").
        model: Model name. If None, uses provider's default.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        settings: Settings instance. If None, loads from environment.

    Returns:
        LLMInstance containing the LLM and metadata.

    Raises:
        ValueError: If provider is not supported.
    """
    if settings is None:
        settings = get_settings()

    model_name = model or DEFAULT_MODELS.get(provider, DEFAULT_MODELS["openai"])

    logger.info("Creating LLM: provider=%s, model=%s", provider, model_name)

    if provider == "openai":
        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=settings.openai_api_key,
        )

    elif provider == "anthropic":
        llm = ChatAnthropic(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=settings.anthropic_api_key,
        )

    elif provider == "google":
        llm = ChatVertexAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            project=settings.google_cloud_project,
            location=settings.google_cloud_location,
        )

    else:
        raise ValueError(
            f"Unknown LLM provider: {provider}. "
            f"Supported providers: openai, anthropic, google"
        )

    return LLMInstance(llm=llm, model_name=model_name, provider=provider)
