"""Qdrant vector store implementation."""

import time
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from financial_rag.config import get_settings
from financial_rag.embeddings.embedder import get_embeddings
from financial_rag.vectorstores.base_store import BaseVectorStore, Document, RetrievalResult


class QdrantStore(BaseVectorStore):
    """Qdrant vector store implementation."""

    def __init__(
        self,
        collection_name: str | None = None,
        host: str | None = None,
        port: int | None = None,
    ):
        """Initialize Qdrant store.

        Args:
            collection_name: Name of the collection. Defaults to settings.
            host: Qdrant host. Defaults to settings.
            port: Qdrant port. Defaults to settings.
        """
        settings = get_settings()
        collection_name = collection_name or settings.qdrant_collection_name
        super().__init__(collection_name)

        host = host or settings.qdrant_host
        port = port or settings.qdrant_port
        api_key = settings.qdrant_api_key or None

        # Use URL format for explicit HTTP connection (no SSL for local Docker)
        # If api_key is set, assume cloud deployment with HTTPS
        if api_key:
            self._client = QdrantClient(
                host=host,
                port=port,
                api_key=api_key,
                https=True,
            )
        else:
            # Local Docker - use HTTP
            self._client = QdrantClient(
                url=f"http://{host}:{port}",
            )
        self._embeddings = get_embeddings()
        self._vector_size = 1536  # OpenAI text-embedding-3-small dimension

        # Create collection if it doesn't exist
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Ensure the collection exists."""
        collections = self._client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self._vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def add_documents(
        self,
        documents: list[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Add documents to Qdrant.

        Args:
            documents: List of document texts to add.
            metadatas: Optional list of metadata dicts for each document.
            ids: Optional list of IDs for each document.

        Returns:
            List of document IDs that were added.
        """
        if not documents:
            return []

        # Generate IDs if not provided (Qdrant uses UUIDs or ints)
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]

        # Generate embeddings
        embeddings = self._embeddings.embed_documents(documents)

        # Ensure metadatas list has correct length
        if metadatas is None:
            metadatas = [{} for _ in documents]

        # Create points
        points = []
        for i, (doc_id, doc, embedding, metadata) in enumerate(
            zip(ids, documents, embeddings, metadatas)
        ):
            # Add document content to payload
            payload = {
                "content": doc,
                **metadata,
            }
            points.append(
                PointStruct(
                    id=doc_id if doc_id.isdigit() else i,
                    vector=embedding,
                    payload=payload,
                )
            )

        # Upsert points
        self._client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: dict | None = None,
    ) -> RetrievalResult:
        """Search for similar documents in Qdrant.

        Args:
            query: Query text to search for.
            k: Number of results to return.
            filter_metadata: Optional metadata filters (Qdrant filter format).

        Returns:
            RetrievalResult with matching documents.
        """
        start_time = time.time()

        # Generate query embedding
        query_embedding = self._embeddings.embed_query(query)

        # Build query filter if provided
        query_filter = None
        if filter_metadata:
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            conditions = [
                FieldCondition(key=key, match=MatchValue(value=value))
                for key, value in filter_metadata.items()
            ]
            query_filter = Filter(must=conditions)

        # Execute query using query_points (qdrant-client >= 1.16)
        results = self._client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=k,
            with_payload=True,
            query_filter=query_filter,
        )

        # Convert to Document objects
        documents = []
        for hit in results.points:
            payload = hit.payload or {}
            content = payload.pop("content", "")

            documents.append(
                Document(
                    content=content,
                    metadata=payload,
                    score=hit.score,
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
        self._ensure_collection()

    def count(self) -> int:
        """Get the number of documents in the collection."""
        collection_info = self._client.get_collection(self.collection_name)
        return collection_info.points_count
