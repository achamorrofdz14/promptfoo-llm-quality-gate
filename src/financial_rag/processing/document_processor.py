"""Document processor using Docling for PDF parsing with table extraction."""

import hashlib
import json
import re
from pathlib import Path

from docling.chunking import HybridChunker
from langchain_core.documents import Document
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from financial_rag.config import get_app_config, get_settings

console = Console()


def extract_document_metadata(filename: str) -> dict:
    """Extract metadata from document filename.

    Expected format: "YYYY Qx TICKER.pdf"
    Example: "2023 Q3 AAPL.pdf" -> {"year": "2023", "quarter": "Q3", "ticker": "AAPL"}

    Args:
        filename: Document filename.

    Returns:
        Dictionary with extracted metadata.
    """
    pattern = r"(\d{4})\s+(Q\d)\s+([A-Z]+)\.pdf"
    match = re.match(pattern, filename)

    if match:
        return {
            "year": match.group(1),
            "quarter": match.group(2),
            "ticker": match.group(3),
            "source_document": filename,
        }

    return {"source_document": filename}


def get_cache_key(file_path: Path, chunk_size: int, chunk_overlap: int) -> str:
    """Generate cache key based on file content and processing parameters.

    Args:
        file_path: Path to the document.
        chunk_size: Chunk size parameter.
        chunk_overlap: Chunk overlap parameter.

    Returns:
        Hash string for cache key.
    """
    file_stat = file_path.stat()
    key_data = f"{file_path.name}:{file_stat.st_size}:{file_stat.st_mtime}:{chunk_size}:{chunk_overlap}"
    return hashlib.md5(key_data.encode()).hexdigest()


class DocumentProcessor:
    """Process financial PDF documents using Docling for table-aware extraction."""

    def __init__(
        self,
        documents_dir: Path | None = None,
        cache_dir: Path | None = None,
        use_cache: bool = True,
    ):
        """Initialize document processor.

        Args:
            documents_dir: Directory containing PDF documents.
            cache_dir: Directory for caching processed documents.
            use_cache: Whether to use caching for processed documents.
        """
        settings = get_settings()
        self._app_config = get_app_config()

        self._documents_dir = documents_dir or settings.documents_dir
        self._cache_dir = cache_dir or settings.data_dir / "processed"
        self._use_cache = use_cache

        # Ensure cache directory exists
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def selected_documents(self) -> list[str]:
        """Get list of selected documents from configuration."""
        return self._app_config.selected_documents

    @property
    def chunk_size(self) -> int:
        """Get chunk size from configuration."""
        return self._app_config.chunk_size

    @property
    def chunk_overlap(self) -> int:
        """Get chunk overlap from configuration."""
        return self._app_config.chunk_overlap

    @property
    def chunker_tokenizer(self) -> str:
        """Get tokenizer model for chunking from configuration."""
        return self._app_config.chunker_tokenizer

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for a document.

        Args:
            cache_key: Cache key hash.

        Returns:
            Path to cache file.
        """
        return self._cache_dir / f"{cache_key}.json"

    def _load_from_cache(self, cache_key: str) -> list[Document] | None:
        """Load processed documents from cache.

        Args:
            cache_key: Cache key hash.

        Returns:
            List of Document objects or None if not cached.
        """
        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            return None

        with open(cache_path) as f:
            cached_data = json.load(f)

        return [
            Document(page_content=doc["page_content"], metadata=doc["metadata"])
            for doc in cached_data
        ]

    def _save_to_cache(self, cache_key: str, documents: list[Document]) -> None:
        """Save processed documents to cache.

        Args:
            cache_key: Cache key hash.
            documents: List of Document objects to cache.
        """
        cache_path = self._get_cache_path(cache_key)

        cache_data = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in documents
        ]

        with open(cache_path, "w") as f:
            json.dump(cache_data, f)

    def process_document(self, file_path: Path) -> list[Document]:
        """Process a single PDF document using Docling.

        Uses HybridChunker for table-aware chunking that preserves table structure.

        Args:
            file_path: Path to the PDF document.

        Returns:
            List of LangChain Document objects with content and metadata.
        """
        cache_key = get_cache_key(file_path, self.chunk_size, self.chunk_overlap)

        # Check cache first
        if self._use_cache:
            cached_docs = self._load_from_cache(cache_key)
            if cached_docs is not None:
                console.print(f"  [dim]Loaded from cache: {file_path.name}[/dim]")
                return cached_docs

        # Extract base metadata from filename
        base_metadata = extract_document_metadata(file_path.name)

        # Use Docling for PDF processing with table-aware chunking
        loader = DoclingLoader(
            file_path=str(file_path),
            export_type=ExportType.DOC_CHUNKS,
            chunker=HybridChunker(
                tokenizer=self.chunker_tokenizer,
                max_tokens=self.chunk_size,
            ),
        )

        # Load and process document
        raw_docs = loader.load()

        # Enrich metadata for each chunk
        processed_docs = []
        for i, doc in enumerate(raw_docs):
            # Merge base metadata with Docling metadata
            enriched_metadata = {
                **base_metadata,
                **doc.metadata,
                "chunk_index": i,
            }

            # Determine chunk type based on content characteristics
            content = doc.page_content
            if self._is_table_content(content):
                enriched_metadata["chunk_type"] = "Table"
            else:
                enriched_metadata["chunk_type"] = "Text"

            processed_docs.append(
                Document(page_content=content, metadata=enriched_metadata)
            )

        # Save to cache
        if self._use_cache:
            self._save_to_cache(cache_key, processed_docs)

        return processed_docs

    def _is_table_content(self, content: str) -> bool:
        """Determine if content is likely table data.

        Uses heuristics to detect table-like content:
        - Multiple columns separated by tabs or multiple spaces
        - Presence of numerical patterns with currency/percentage symbols
        - Row-like structure with consistent delimiters

        Args:
            content: Text content to analyze.

        Returns:
            True if content appears to be table data.
        """
        lines = content.strip().split("\n")

        if len(lines) < 2:
            return False

        # Check for consistent column structure
        tab_lines = sum(1 for line in lines if "\t" in line or "  " in line)

        # Check for numerical data patterns common in financial tables
        number_pattern = r"[\$€£]?\s*[\d,]+\.?\d*\s*[%]?"
        number_lines = sum(
            1 for line in lines if len(re.findall(number_pattern, line)) >= 2
        )

        # Consider table if significant portion has structure
        total_lines = len(lines)
        return (tab_lines / total_lines > 0.3) or (number_lines / total_lines > 0.4)

    def process_selected_documents(self) -> list[Document]:
        """Process all selected documents from configuration.

        Returns:
            Combined list of Document objects from all selected documents.
        """
        all_documents = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Processing documents...", total=len(self.selected_documents)
            )

            for doc_name in self.selected_documents:
                file_path = self._documents_dir / doc_name

                if not file_path.exists():
                    console.print(f"[yellow]Warning: Document not found: {doc_name}[/yellow]")
                    progress.advance(task)
                    continue

                progress.update(task, description=f"Processing {doc_name}...")

                docs = self.process_document(file_path)
                all_documents.extend(docs)

                console.print(f"  [green]✓[/green] {doc_name}: {len(docs)} chunks")
                progress.advance(task)

        console.print(f"\n[bold]Total chunks processed: {len(all_documents)}[/bold]")

        return all_documents

    def get_document_stats(self) -> dict:
        """Get statistics about processed documents.

        Returns:
            Dictionary with document statistics.
        """
        stats = {
            "selected_documents": self.selected_documents,
            "documents_dir": str(self._documents_dir),
            "cache_dir": str(self._cache_dir),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "chunker_tokenizer": self.chunker_tokenizer,
            "use_cache": self._use_cache,
        }

        # Check which documents exist
        existing = []
        missing = []
        for doc_name in self.selected_documents:
            if (self._documents_dir / doc_name).exists():
                existing.append(doc_name)
            else:
                missing.append(doc_name)

        stats["existing_documents"] = existing
        stats["missing_documents"] = missing

        return stats
