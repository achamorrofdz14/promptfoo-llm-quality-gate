#!/usr/bin/env python3
"""Ingest financial documents into vector stores using Docling.

This script:
1. Processes PDF documents using Docling for table-aware extraction
2. Populates both ChromaDB and Qdrant vector stores
3. Shows progress and statistics
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from financial_rag.config import get_app_config, get_settings
from financial_rag.processing import DocumentProcessor
from financial_rag.vectorstores.chroma_store import ChromaStore
from financial_rag.vectorstores.qdrant_store import QdrantStore

console = Console()


def main():
    """Main ingestion workflow."""
    console.print(Panel.fit(
        "[bold blue]Financial RAG Document Ingestion[/bold blue]\n"
        "Processing PDFs with Docling and populating vector stores",
        border_style="blue"
    ))

    # Load configuration
    settings = get_settings()
    app_config = get_app_config()

    # Show configuration
    config_table = Table(title="Configuration", show_header=False)
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="green")
    config_table.add_row("Documents Directory", str(settings.documents_dir))
    config_table.add_row("Selected Documents", str(len(app_config.selected_documents)))
    config_table.add_row("Chunk Size", str(app_config.chunk_size))
    config_table.add_row("Chunk Overlap", str(app_config.chunk_overlap))
    config_table.add_row("Chunker Tokenizer", app_config.chunker_tokenizer)
    console.print(config_table)
    console.print()

    # Initialize document processor
    processor = DocumentProcessor()

    # Show document stats
    stats = processor.get_document_stats()
    if stats["missing_documents"]:
        console.print("[yellow]Warning: Some documents not found:[/yellow]")
        for doc in stats["missing_documents"]:
            console.print(f"  - {doc}")
        console.print()

    # Process documents
    console.print("[bold]Processing documents with Docling...[/bold]")
    documents = processor.process_selected_documents()

    if not documents:
        console.print("[red]No documents were processed. Exiting.[/red]")
        return 1

    # Show chunk statistics
    chunk_stats = Table(title="Chunk Statistics")
    chunk_stats.add_column("Metric", style="cyan")
    chunk_stats.add_column("Value", style="green")

    # Count by chunk type
    table_chunks = sum(1 for d in documents if d.metadata.get("chunk_type") == "Table")
    text_chunks = sum(1 for d in documents if d.metadata.get("chunk_type") == "Text")

    chunk_stats.add_row("Total Chunks", str(len(documents)))
    chunk_stats.add_row("Table Chunks", str(table_chunks))
    chunk_stats.add_row("Text Chunks", str(text_chunks))
    console.print(chunk_stats)
    console.print()

    # Populate ChromaDB
    console.print("[bold]Populating ChromaDB...[/bold]")
    try:
        chroma_store = ChromaStore()
        # Clear existing data
        chroma_store.delete_collection()
        # Add documents
        chroma_ids = chroma_store.add_langchain_documents(documents)
        console.print(f"  [green]✓[/green] Added {len(chroma_ids)} documents to ChromaDB")
        console.print(f"  [dim]Collection: {chroma_store.collection_name}[/dim]")
    except Exception as e:
        console.print(f"  [red]✗[/red] Failed to populate ChromaDB: {e}")

    console.print()

    # Populate Qdrant
    console.print("[bold]Populating Qdrant...[/bold]")
    try:
        qdrant_store = QdrantStore()
        # Clear existing data
        qdrant_store.delete_collection()
        # Add documents
        qdrant_ids = qdrant_store.add_langchain_documents(documents)
        console.print(f"  [green]✓[/green] Added {len(qdrant_ids)} documents to Qdrant")
        console.print(f"  [dim]Collection: {qdrant_store.collection_name}[/dim]")
    except Exception as e:
        console.print(f"  [red]✗[/red] Failed to populate Qdrant: {e}")
        console.print(f"  [dim]Make sure Qdrant is running (docker run -p 6333:6333 qdrant/qdrant)[/dim]")

    console.print()
    console.print(Panel.fit(
        "[bold green]Ingestion Complete![/bold green]\n"
        "You can now run promptfoo evaluations.",
        border_style="green"
    ))

    return 0


if __name__ == "__main__":
    sys.exit(main())
