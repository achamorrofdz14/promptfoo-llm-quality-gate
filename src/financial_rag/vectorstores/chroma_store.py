"""ChromaDB vector store implementation."""

import time
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from financial_rag.config import get_settings
from financial_rag.embeddings.embedder import get_embeddings
from financial_rag.vectorstores.base_store import BaseVectorStore, Document, RetrievalResult


class ChromaStore(BaseVectorStore):
    """ChromaDB vector store implementation."""

    def __init__(self, collection_name: str | None = None, persist_directory: str | None = None):
        """Initialize ChromaDB store.

        Args:
            collection_name: Name of the collection. Defaults to settings.
            persist_directory: Directory to persist data. Defaults to settings.
        """
        settings = get_settings()
        collection_name = collection_name or settings.chroma_collection_name
        super().__init__(collection_name)

        persist_dir = persist_directory or settings.chroma_persist_directory
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._embeddings = get_embeddings()
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self,
        documents: list[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Add documents to ChromaDB.

        Args:
            documents: List of document texts to add.
            metadatas: Optional list of metadata dicts for each document.
            ids: Optional list of IDs for each document.

        Returns:
            List of document IDs that were added.
        """
        if not documents:
            return []

        # Generate IDs if not provided
        if ids is None:
            existing_count = self.count()
            ids = [f"doc_{existing_count + i}" for i in range(len(documents))]

        # Generate embeddings
        embeddings = self._embeddings.embed_documents(documents)

        # Ensure metadatas list has correct length
        if metadatas is None:
            metadatas = [{} for _ in documents]

        # Add to collection
        self._collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: dict | None = None,
    ) -> RetrievalResult:
        """Search for similar documents in ChromaDB.

        Args:
            query: Query text to search for.
            k: Number of results to return.
            filter_metadata: Optional metadata filters (ChromaDB where clause).

        Returns:
            RetrievalResult with matching documents.
        """
        start_time = time.time()

        # Generate query embedding
        query_embedding = self._embeddings.embed_query(query)

        # Build query parameters
        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }

        if filter_metadata:
            query_params["where"] = filter_metadata

        # Execute query
        results = self._collection.query(**query_params)

        # Convert to Document objects
        documents = []
        if results["documents"] and results["documents"][0]:
            for i, doc_content in enumerate(results["documents"][0]):
                # ChromaDB returns distances, convert to similarity score
                distance = results["distances"][0][i] if results["distances"] else 0
                similarity_score = 1 - distance  # Cosine distance to similarity

                metadata = results["metadatas"][0][i] if results["metadatas"] else {}

                documents.append(
                    Document(
                        content=doc_content,
                        metadata=metadata,
                        score=similarity_score,
                    )
                )

        latency_ms = (time.time() - start_time) * 1000

        return RetrievalResult(
            documents=documents,
            query=query,
            latency_ms=latency_ms,
        )

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        """Get the number of documents in the collection."""
        return self._collection.count()
