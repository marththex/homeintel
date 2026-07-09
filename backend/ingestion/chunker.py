"""
ingestion/chunker.py — Split raw text into overlapping chunks.

Uses LangChain RecursiveCharacterTextSplitter so that splits prefer
paragraph → sentence → word boundaries in that priority order.
chunk_size and chunk_overlap come from settings (CHUNK_SIZE / CHUNK_OVERLAP).
"""

import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import settings

logger = logging.getLogger(__name__)

_splitter: RecursiveCharacterTextSplitter | None = None


def _get_splitter() -> RecursiveCharacterTextSplitter:
    global _splitter
    if _splitter is None:
        _splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
        )
        logger.debug(
            "Text splitter ready — chunk_size=%d overlap=%d",
            settings.chunk_size,
            settings.chunk_overlap,
        )
    return _splitter


def chunk_text(text: str) -> list[str]:
    """Split text into chunks; returns an empty list if text is blank."""
    if not text.strip():
        return []
    chunks = _get_splitter().split_text(text)
    logger.debug("Split into %d chunks", len(chunks))
    return chunks
