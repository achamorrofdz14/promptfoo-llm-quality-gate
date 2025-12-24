"""Model-only provider for promptfoo evaluation.

This provider calls the LLM directly without RAG retrieval.
Useful for testing prompt engineering and red team scenarios
where you want to test the model's behavior without retrieval interference.
"""

from financial_rag import get_logger
from financial_rag.providers.base_provider import BaseProvider, create_provider_response
from financial_rag.providers.llm_factory import create_llm

logger = get_logger("providers.model_only")


class ModelOnlyProvider(BaseProvider):
    """Provider that calls LLM directly without RAG retrieval.

    This is useful for:
    - Testing prompt engineering effectiveness
    - Red team testing without retrieval interference
    - Comparing model behavior with vs without context
    """

    NO_RETRIEVAL_MESSAGE = "[No document retrieval - testing prompt engineering only]"

    def __init__(self, config: dict | None = None):
        """Initialize model-only provider.

        Args:
            config: Provider configuration. Supports:
                - llm_provider: "openai", "anthropic", or "google" (default: "openai")
                - model: Model name (default varies by provider)
                - temperature: LLM temperature
                - max_tokens: Max response tokens
                - prompt_template: Path to custom prompt template
                - mock_context: Optional fake context to inject (for testing)
        """
        super().__init__(config)
        self._llm = None
        self._model_name = ""

    def _get_llm(self):
        """Get or create LLM instance based on configuration."""
        if self._llm is not None:
            return self._llm

        llm_instance = create_llm(
            provider=self._config.get("llm_provider", "openai"),
            model=self._config.get("model"),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        self._llm = llm_instance.llm
        self._model_name = llm_instance.model_name

        return self._llm

    def generate(self, prompt: str, context: dict) -> dict:
        """Generate response by calling LLM directly.

        Args:
            prompt: The query to answer.
            context: Test case context with vars.

        Returns:
            Provider response with output and metadata.
        """
        # Get the query from context vars or use prompt directly
        query = context.get("vars", {}).get("query", prompt)

        # Use mock context if provided, otherwise indicate no retrieval
        mock_context = self._config.get("mock_context", self.NO_RETRIEVAL_MESSAGE)

        # Load and format prompt template
        template = self._load_prompt_template("redteam/strict_analyst.txt")
        full_prompt = template.replace("{{context}}", mock_context).replace("{{query}}", query)

        # Generate response
        llm = self._get_llm()
        response = llm.invoke(full_prompt)

        # Extract token usage from response metadata
        usage = getattr(response, "usage_metadata", {}) or {}
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)

        # Build metadata
        metadata = {
            "model": self._model_name,
            "llm_provider": self._config.get("llm_provider", "openai"),
            "mode": "model_only",
            "has_context": mock_context != self.NO_RETRIEVAL_MESSAGE,
        }

        return create_provider_response(
            output=response.content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=self._model_name,
            metadata=metadata,
        )


# Promptfoo entry point
def call_api(prompt: str, options: dict, context: dict) -> dict:
    """Promptfoo provider entry point.

    Args:
        prompt: The prompt/query from promptfoo.
        options: Provider options including config.
        context: Test case context with vars.

    Returns:
        Provider response dictionary.
    """
    config = options.get("config", {})
    provider = ModelOnlyProvider(config)
    return provider.call_api(prompt, options, context)
