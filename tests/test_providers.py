"""Tests for provider modules."""

import pytest

from financial_rag.providers import (
    BaseProvider,
    ChromaDBRAGProvider,
    GenerationOnlyProvider,
    ModelOnlyProvider,
    QdrantRAGProvider,
    RetrieverOnlyProvider,
    create_provider_response,
)
from financial_rag.providers.llm_factory import DEFAULT_MODELS, LLMInstance, create_llm


class TestProviderResponse:
    """Tests for create_provider_response function."""

    def test_creates_valid_response(self):
        """Verify response has required fields."""
        response = create_provider_response(
            output="Test output",
            prompt_tokens=100,
            completion_tokens=50,
            model="gpt-4o-mini",
        )
        assert response["output"] == "Test output"
        assert "tokenUsage" in response
        assert response["tokenUsage"]["prompt"] == 100
        assert response["tokenUsage"]["completion"] == 50
        assert response["tokenUsage"]["total"] == 150

    def test_calculates_cost(self):
        """Verify cost is calculated when model is provided."""
        response = create_provider_response(
            output="Test",
            prompt_tokens=1000,
            completion_tokens=500,
            model="gpt-4o-mini",
        )
        assert "cost" in response
        assert response["cost"] > 0

    def test_includes_metadata(self):
        """Verify metadata is included when provided."""
        metadata = {"custom_field": "value"}
        response = create_provider_response(
            output="Test",
            metadata=metadata,
        )
        assert response["metadata"] == metadata


class TestLLMFactory:
    """Tests for LLM factory module."""

    def test_default_models_defined(self):
        """Verify default models are defined for all providers."""
        assert "openai" in DEFAULT_MODELS
        assert "anthropic" in DEFAULT_MODELS
        assert "google" in DEFAULT_MODELS

    def test_llm_instance_dataclass(self):
        """Verify LLMInstance dataclass works correctly."""
        instance = LLMInstance(llm=None, model_name="test-model", provider="test")
        assert instance.model_name == "test-model"
        assert instance.provider == "test"

    def test_create_llm_invalid_provider(self):
        """Verify create_llm raises error for invalid provider."""
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_llm(provider="invalid-provider")


class TestProviderClasses:
    """Tests for provider class instantiation."""

    def test_chromadb_provider_instantiates(self):
        """Verify ChromaDBRAGProvider can be instantiated."""
        provider = ChromaDBRAGProvider(config={})
        assert provider is not None
        assert isinstance(provider, BaseProvider)

    def test_qdrant_provider_instantiates(self):
        """Verify QdrantRAGProvider can be instantiated."""
        provider = QdrantRAGProvider(config={})
        assert provider is not None
        assert isinstance(provider, BaseProvider)

    def test_generation_only_provider_instantiates(self):
        """Verify GenerationOnlyProvider can be instantiated."""
        provider = GenerationOnlyProvider(config={})
        assert provider is not None
        assert isinstance(provider, BaseProvider)

    def test_model_only_provider_instantiates(self):
        """Verify ModelOnlyProvider can be instantiated."""
        provider = ModelOnlyProvider(config={})
        assert provider is not None
        assert isinstance(provider, BaseProvider)

    def test_retriever_only_provider_instantiates(self):
        """Verify RetrieverOnlyProvider can be instantiated."""
        provider = RetrieverOnlyProvider(config={})
        assert provider is not None
        assert isinstance(provider, BaseProvider)


class TestProviderConfig:
    """Tests for provider configuration handling."""

    def test_provider_uses_config_values(self):
        """Verify provider uses configuration values."""
        config = {
            "temperature": 0.5,
            "max_tokens": 2048,
            "top_k": 10,
        }
        provider = ChromaDBRAGProvider(config=config)
        assert provider.temperature == 0.5
        assert provider.max_tokens == 2048
        assert provider.top_k == 10

    def test_provider_uses_defaults(self):
        """Verify provider uses defaults when config not provided."""
        provider = ChromaDBRAGProvider(config={})
        # Should not raise and should have sensible defaults
        assert provider.temperature is not None
        assert provider.max_tokens is not None
        assert provider.top_k is not None
