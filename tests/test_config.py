"""Tests for configuration module."""

from financial_rag.config import calculate_cost, get_app_config, get_settings


class TestSettings:
    """Tests for Settings class."""

    def test_settings_loads(self):
        """Verify settings can be loaded from environment."""
        settings = get_settings()
        assert settings is not None

    def test_settings_has_required_attributes(self):
        """Verify settings has required attributes."""
        settings = get_settings()
        # These should exist even if empty
        assert hasattr(settings, "openai_api_key")
        assert hasattr(settings, "chroma_persist_directory")
        assert hasattr(settings, "qdrant_host")

    def test_settings_is_cached(self):
        """Verify settings are cached (same instance returned)."""
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2


class TestAppConfig:
    """Tests for AppConfig class."""

    def test_app_config_loads(self):
        """Verify app config can be loaded."""
        config = get_app_config()
        assert config is not None

    def test_app_config_has_defaults(self):
        """Verify app config has sensible defaults."""
        config = get_app_config()
        assert config.temperature >= 0
        assert config.temperature <= 1
        assert config.max_tokens > 0
        assert config.top_k > 0


class TestCostCalculation:
    """Tests for cost calculation function."""

    def test_calculate_cost_known_model(self):
        """Verify cost calculation works for known models."""
        cost = calculate_cost("gpt-4o-mini", 1000, 500)
        assert cost > 0
        assert isinstance(cost, float)

    def test_calculate_cost_unknown_model(self):
        """Verify cost calculation returns 0 for unknown models."""
        cost = calculate_cost("unknown-model-xyz", 1000, 500)
        assert cost == 0.0

    def test_calculate_cost_zero_tokens(self):
        """Verify cost is 0 when no tokens used."""
        cost = calculate_cost("gpt-4o-mini", 0, 0)
        assert cost == 0.0

    def test_calculate_cost_increases_with_tokens(self):
        """Verify cost increases with more tokens."""
        cost_small = calculate_cost("gpt-4o-mini", 100, 50)
        cost_large = calculate_cost("gpt-4o-mini", 1000, 500)
        assert cost_large > cost_small
