"""Abstract base class for vector store implementations."""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


def sanitize_metadata(metadata: dict) -> dict:
    """Sanitize metadata to only contain simple types supported by vector stores.

    ChromaDB and Qdrant only support: str, int, float, bool, None.
    Complex types (dict, list) are either serialized to JSON strings or removed.

    Args:
        metadata: Raw metadata dictionary.

    Returns:
        Sanitized metadata with only simple types.
    """
    sanitized = {}

    # Keys to skip entirely (Docling internal metadata)
    skip_keys = {"dl_meta", "doc_items", "headings", "origin"}

    for key, value in metadata.items():
        if key in skip_keys:
            continue

        if value is None:
            sanitized[key] = None
        elif isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        elif isinstance(value, (list, dict)):
            # Serialize complex types to JSON string
            try:
                sanitized[key] = json.dumps(value)
            except (TypeError, ValueError):
                # Skip if can't serialize
                pass
        else:
            # Convert other types to string
            sanitized[key] = str(value)

    return sanitized


@dataclass
class Document:
    """A document with content and metadata."""

    content: str
    metadata: dict
    score: float | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "content": self.content,
            "metadata": self.metadata,
            "score": self.score,
        }


@dataclass
class RetrievalResult:
    """Result from a retrieval operation."""

    documents: list[Document]
    query: str
    latency_ms: float

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "documents": [doc.to_dict() for doc in self.documents],
            "query": self.query,
            "latency_ms": self.latency_ms,
            "count": len(self.documents),
        }


class BaseVectorStore(ABC):
    """Abstract base class for vector store implementations."""

    def __init__(self, collection_name: str):
        """Initialize the vector store.

        Args:
            collection_name: Name of the collection/index to use.
        """
        self.collection_name = collection_name

    @abstractmethod
    def add_documents(
        self,
        documents: list[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Add documents to the vector store.

        Args:
            documents: List of document texts to add.
            metadatas: Optional list of metadata dicts for each document.
            ids: Optional list of IDs for each document.

        Returns:
            List of document IDs that were added.
        """
        pass

    @abstractmethod
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: dict | None = None,
    ) -> RetrievalResult:
        """Search for similar documents.

        Args:
            query: Query text to search for.
            k: Number of results to return.
            filter_metadata: Optional metadata filters.

        Returns:
            RetrievalResult with matching documents.
        """
        pass

    @abstractmethod
    def delete_collection(self) -> None:
        """Delete the entire collection."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Get the number of documents in the collection."""
        pass

    def add_langchain_documents(self, documents: list) -> list[str]:
        """Add pre-processed LangChain documents to the vector store.

        This method accepts documents from Docling or other LangChain document loaders.

        Args:
            documents: List of LangChain Document objects with page_content and metadata.

        Returns:
            List of document IDs that were added.
        """
        if not documents:
            return []

        texts = [doc.page_content for doc in documents]
        # Sanitize metadata to remove complex Docling types
        metadatas = [sanitize_metadata(doc.metadata) for doc in documents]
        ids = [f"doc_{i}" for i in range(len(documents))]

        return self.add_documents(texts, metadatas, ids)
