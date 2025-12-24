"""Base provider functionality for promptfoo custom providers."""

import time
from abc import ABC, abstractmethod
from pathlib import Path

from financial_rag import get_logger
from financial_rag.config import calculate_cost, get_app_config, get_settings

logger = get_logger("providers.base")


class BaseProvider(ABC):
    """Abstract base class for promptfoo custom providers."""

    def __init__(self, config: dict | None = None):
        """Initialize base provider.

        Args:
            config: Provider configuration from promptfoo.
        """
        self._config = config or {}
        self._settings = get_settings()
        self._app_config = get_app_config()

    @property
    def temperature(self) -> float:
        """Get LLM temperature from config or app config."""
        return self._config.get("temperature", self._app_config.temperature)

    @property
    def max_tokens(self) -> int:
        """Get max tokens from config or app config."""
        return self._config.get("max_tokens", self._app_config.max_tokens)

    @property
    def top_k(self) -> int:
        """Get top_k for retrieval from config or app config."""
        return self._config.get("top_k", self._app_config.top_k)

    @abstractmethod
    def generate(self, prompt: str, context: dict) -> dict:
        """Generate a response.

        Args:
            prompt: The prompt/query to process.
            context: Test case context with vars.

        Returns:
            Dictionary with output, tokenUsage, cost, etc.
        """
        pass

    def call_api(self, prompt: str, options: dict, context: dict) -> dict:
        """Promptfoo entry point for the provider.

        Args:
            prompt: The prompt/query from promptfoo.
            options: Provider options including config.
            context: Test case context with vars.

        Returns:
            Dictionary with output and metadata.
        """
        # Merge options config with instance config
        if "config" in options:
            self._config.update(options["config"])

        start_time = time.time()
        logger.debug("Processing request with prompt length: %d", len(prompt))

        try:
            result = self.generate(prompt, context)
            latency_ms = (time.time() - start_time) * 1000
            result["latency_ms"] = latency_ms
            logger.info("Request completed in %.2fms", latency_ms)
            return result
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error("Request failed after %.2fms: %s", latency_ms, str(e))
            return {
                "output": None,
                "error": str(e),
                "latency_ms": latency_ms,
            }

    def _format_context_for_prompt(self, documents: list[dict]) -> str:
        """Format retrieved documents as context for the LLM.

        Args:
            documents: List of document dictionaries with content and metadata.

        Returns:
            Formatted context string.
        """
        context_parts = []

        for i, doc in enumerate(documents, 1):
            source = doc.get("metadata", {}).get("source_document", "Unknown")
            chunk_type = doc.get("metadata", {}).get("chunk_type", "Unknown")
            content = doc.get("content", "")

            context_parts.append(f"[Source {i}: {source} ({chunk_type})]\n{content}")

        return "\n\n---\n\n".join(context_parts)

    def _load_prompt_template(self, template_name: str) -> str:
        """Load a prompt template from the prompts directory.

        Args:
            template_name: Name of the template file (relative to prompts dir).

        Returns:
            Template content string.
        """
        # Check config for custom path first
        if "prompt_template" in self._config:
            template_path = Path(self._config["prompt_template"])
            # If relative path, resolve from prompts directory
            if not template_path.is_absolute():
                template_path = self._settings.prompts_dir / template_path
        else:
            template_path = self._settings.prompts_dir / template_name

        if template_path.exists():
            return template_path.read_text()

        # Return default template if file not found
        return """Answer the following question based on the provided context.
If the information is not in the context, say "I don't have enough information."

Context:
{{context}}

Question: {{query}}

Answer:"""


def create_provider_response(
    output: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    model: str = "",
    cached: bool = False,
    metadata: dict | None = None,
) -> dict:
    """Create a standardized provider response.

    Args:
        output: The generated output text.
        prompt_tokens: Number of prompt tokens used.
        completion_tokens: Number of completion tokens generated.
        model: Model identifier for cost calculation.
        cached: Whether the response was cached.
        metadata: Additional metadata to include.

    Returns:
        Dictionary in promptfoo provider response format.
    """
    total_tokens = prompt_tokens + completion_tokens
    cost = calculate_cost(model, prompt_tokens, completion_tokens) if model else 0.0

    response = {
        "output": output,
        "tokenUsage": {
            "total": total_tokens,
            "prompt": prompt_tokens,
            "completion": completion_tokens,
        },
        "cost": cost,
        "cached": cached,
    }

    if metadata:
        response["metadata"] = metadata

    return response
