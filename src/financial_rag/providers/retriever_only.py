"""Retriever-only provider for evaluating retrieval quality.

This provider only performs retrieval without generation, useful for:
- Evaluating retrieval relevance
- Comparing vector store performance
- Debugging retrieval issues
"""

import json

from financial_rag.providers.base_provider import BaseProvider, create_provider_response
from financial_rag.vectorstores.chroma_store import ChromaStore
from financial_rag.vectorstores.qdrant_store import QdrantStore


class RetrieverOnlyProvider(BaseProvider):
    """Provider that only performs document retrieval."""

    def __init__(self, config: dict | None = None):
        """Initialize retriever-only provider.

        Args:
            config: Provider configuration. Supports:
                - vector_store: "chromadb" or "qdrant" (default: "chromadb")
                - top_k: Number of documents to retrieve
                - collection_name: Collection name
                - return_format: "json" or "text" (default: "json")
        """
        super().__init__(config)
        self._store = None

    def _get_store(self):
        """Get or create vector store instance."""
        if self._store is not None:
            return self._store

        store_type = self._config.get("vector_store", "chromadb")
        collection_name = self._config.get("collection_name")

        if store_type == "chromadb":
            self._store = ChromaStore(collection_name=collection_name)
        elif store_type == "qdrant":
            self._store = QdrantStore(collection_name=collection_name)
        else:
            raise ValueError(f"Unknown vector store: {store_type}")

        return self._store

    def generate(self, prompt: str, context: dict) -> dict:
        """Retrieve documents for the query.

        Args:
            prompt: The query to search for.
            context: Test case context with vars.

        Returns:
            Provider response with retrieved documents.
        """
        # Get the query from context vars or use prompt directly
        query = context.get("vars", {}).get("query", prompt)

        # Build filter from context if source document is specified
        filter_metadata = None
        source_doc = context.get("vars", {}).get("source_docs")
        if source_doc and not source_doc.startswith("*"):
            filter_metadata = {"source_document": source_doc}

        # Retrieve documents
        store = self._get_store()
        retrieval_result = store.similarity_search(
            query=query,
            k=self.top_k,
            filter_metadata=filter_metadata,
        )

        # Format output based on configuration
        return_format = self._config.get("return_format", "json")

        if return_format == "json":
            output = json.dumps(retrieval_result.to_dict(), indent=2)
        else:
            # Text format - concatenate document contents
            output = self._format_context_for_prompt(
                [doc.to_dict() for doc in retrieval_result.documents]
            )

        # Build metadata
        metadata = {
            "vector_store": self._config.get("vector_store", "chromadb"),
            "retrieval_latency_ms": retrieval_result.latency_ms,
            "documents_retrieved": len(retrieval_result.documents),
            "top_k": self.top_k,
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
            output=output,
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
    provider = RetrieverOnlyProvider(config)
    return provider.call_api(prompt, options, context)
