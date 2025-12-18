"""ChromaDB RAG provider for promptfoo evaluation.

This provider implements full RAG pipeline:
1. Retrieves relevant documents from ChromaDB
2. Generates response using configured LLM
"""

from langchain_anthropic import ChatAnthropic
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI

from financial_rag.config import get_settings
from financial_rag.providers.base_provider import BaseProvider, create_provider_response
from financial_rag.vectorstores.chroma_store import ChromaStore


class ChromaDBRAGProvider(BaseProvider):
    """Full RAG provider using ChromaDB for retrieval."""

    def __init__(self, config: dict | None = None):
        """Initialize ChromaDB RAG provider.

        Args:
            config: Provider configuration. Supports:
                - llm_provider: "openai", "anthropic", or "google" (default: "openai")
                - model: Model name (default varies by provider)
                - temperature: LLM temperature
                - max_tokens: Max response tokens
                - top_k: Number of documents to retrieve
                - collection_name: ChromaDB collection name
        """
        super().__init__(config)
        self._store = None
        self._llm = None

    def _get_store(self) -> ChromaStore:
        """Get or create ChromaDB store instance."""
        if self._store is None:
            collection_name = self._config.get("collection_name")
            self._store = ChromaStore(collection_name=collection_name)
        return self._store

    def _get_llm(self):
        """Get or create LLM instance based on configuration."""
        if self._llm is not None:
            return self._llm

        settings = get_settings()
        provider = self._config.get("llm_provider", "openai")

        if provider == "openai":
            model = self._config.get("model", "gpt-4o-mini")
            self._llm = ChatOpenAI(
                model=model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=settings.openai_api_key,
            )
            self._model_name = model

        elif provider == "anthropic":
            model = self._config.get("model", "claude-haiku-4-5-20251001")
            self._llm = ChatAnthropic(
                model=model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=settings.anthropic_api_key,
            )
            self._model_name = model

        elif provider == "google":
            model = self._config.get("model", "gemini-2.5-flash")
            self._llm = ChatVertexAI(
                model=model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                project=settings.google_cloud_project,
                location=settings.google_cloud_location,
            )
            self._model_name = model

        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

        return self._llm

    def generate(self, prompt: str, context: dict) -> dict:
        """Generate response using RAG pipeline.

        Args:
            prompt: The query to answer.
            context: Test case context with vars.

        Returns:
            Provider response with output and metadata.
        """
        # Get the query from context vars or use prompt directly
        query = context.get("vars", {}).get("query", prompt)

        # Build filter from context if source document is specified
        filter_metadata = None
        source_doc = context.get("vars", {}).get("source_docs")
        if source_doc and not source_doc.startswith("*"):
            # Exact document filter
            filter_metadata = {"source_document": source_doc}

        # Retrieve relevant documents
        store = self._get_store()
        retrieval_result = store.similarity_search(
            query=query,
            k=self.top_k,
            filter_metadata=filter_metadata,
        )

        # Format context from retrieved documents
        # Handle empty results gracefully for red team and edge case testing
        if retrieval_result.documents:
            formatted_context = self._format_context_for_prompt(
                [doc.to_dict() for doc in retrieval_result.documents]
            )
        else:
            formatted_context = "[No relevant documents found in the database for this query]"

        # Load and format prompt template
        template = self._load_prompt_template("rag/generation_prompt.txt")
        full_prompt = template.replace("{{context}}", formatted_context).replace(
            "{{query}}", query
        )

        # Generate response
        llm = self._get_llm()
        response = llm.invoke(full_prompt)

        # Extract token usage from response metadata
        usage = getattr(response, "usage_metadata", {}) or {}
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)

        # Build metadata
        metadata = {
            "retrieval_latency_ms": retrieval_result.latency_ms,
            "documents_retrieved": len(retrieval_result.documents),
            "model": self._model_name,
            "llm_provider": self._config.get("llm_provider", "openai"),
            "sources": [
                {
                    "document": doc.metadata.get("source_document"),
                    "chunk_type": doc.metadata.get("chunk_type"),
                    "score": doc.score,
                }
                for doc in retrieval_result.documents
            ],
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
    provider = ChromaDBRAGProvider(config)
    return provider.call_api(prompt, options, context)
