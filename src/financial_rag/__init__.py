"""Financial RAG - A promptfoo evaluation demo for financial earnings report Q&A."""

import logging

__version__ = "0.1.0"

# Configure logging for the package
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Create package logger
logger = logging.getLogger(__name__)


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a submodule.

    Args:
        name: The submodule name (e.g., 'providers.chromadb_rag')

    Returns:
        A configured logger instance.
    """
    return logging.getLogger(f"{__name__}.{name}")
