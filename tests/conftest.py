"""Pytest configuration and fixtures."""

import os

import pytest


@pytest.fixture(autouse=True)
def set_test_env():
    """Set environment variables for testing."""
    # Ensure we have minimal env vars to prevent errors
    # These won't be used for actual API calls in unit tests
    original_env = os.environ.copy()

    # Set dummy values if not already set (for CI environments)
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "test-key-not-for-real-use"

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def sample_provider_config():
    """Sample configuration for provider tests."""
    return {
        "llm_provider": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0.1,
        "max_tokens": 1024,
        "top_k": 5,
    }


@pytest.fixture
def sample_context():
    """Sample context for provider tests."""
    return {
        "vars": {
            "query": "What was Apple's revenue in Q3 2023?",
            "source_docs": "*",
        }
    }
