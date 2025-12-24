"""Generation-only provider for evaluating LLM response quality.

This provider only performs generation with pre-provided context, useful for:
- Comparing LLM providers on the same context
- Evaluating prompt templates
- Testing generation quality without retrieval variance
"""

from financial_rag import get_logger
from financial_rag.providers.base_provider import BaseProvider, create_provider_response
from financial_rag.providers.llm_factory import create_llm

logger = get_logger("providers.generation_only")


class GenerationOnlyProvider(BaseProvider):
    """Provider that only performs LLM generation with provided context."""

    def __init__(self, config: dict | None = None):
        """Initialize generation-only provider.

        Args:
            config: Provider configuration. Supports:
                - llm_provider: "openai", "anthropic", or "google" (default: "openai")
                - model: Model name (default varies by provider)
                - temperature: LLM temperature
                - max_tokens: Max response tokens
                - prompt_template: Path to custom prompt template
        """
        super().__init__(config)
        self._llm = None

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
        """Generate response using provided context.

        Args:
            prompt: The prompt with context already embedded, or the query.
            context: Test case context with vars (may include pre-retrieved context).

        Returns:
            Provider response with generated output.
        """
        vars_dict = context.get("vars", {})

        # Check if context is provided in vars
        provided_context = vars_dict.get("context", "")
        query = vars_dict.get("query", prompt)

        if provided_context:
            # Use template with provided context
            template = self._load_prompt_template("rag/generation_prompt.txt")
            full_prompt = template.replace("{{context}}", provided_context).replace(
                "{{query}}", query
            )
        else:
            # Use prompt directly (assumes context is already embedded)
            full_prompt = prompt

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
            "has_provided_context": bool(provided_context),
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
    provider = GenerationOnlyProvider(config)
    return provider.call_api(prompt, options, context)
