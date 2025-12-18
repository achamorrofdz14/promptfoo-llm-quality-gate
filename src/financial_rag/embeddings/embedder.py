"""Embedding model wrapper for document and query embeddings."""

from langchain_openai import OpenAIEmbeddings

from financial_rag.config import get_settings


def get_embeddings() -> OpenAIEmbeddings:
    """Get the configured embedding model.

    Returns:
        OpenAI embeddings instance configured with settings.
    """
    settings = get_settings()
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        openai_api_key=settings.openai_api_key,
    )
