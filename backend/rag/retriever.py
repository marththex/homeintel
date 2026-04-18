"""
rag/retriever.py — Thin wrapper around VectorStore.query for the RAG chain.
"""

import logging
from typing import Optional

from langchain_core.documents import Document

from config import settings
from vectorstore.chroma import VectorStore, Modality

logger = logging.getLogger(__name__)

_vs: Optional[VectorStore] = None


def _get_vs() -> VectorStore:
    global _vs
    if _vs is None:
        _vs = VectorStore()
    return _vs


def retrieve(question: str, modality_filter: Optional[str] = None) -> list[Document]:
    """
    Retrieve the top-k most relevant document chunks for a question.

    Args:
        question:        Natural language query.
        modality_filter: Optional modality string ("document", "image", "audio").
                         Invalid values are silently ignored.
    """
    modality: Optional[Modality] = None
    if modality_filter:
        try:
            modality = Modality(modality_filter)
        except ValueError:
            logger.warning("Unknown modality_filter %r — ignoring", modality_filter)

    docs = _get_vs().query(question, top_k=settings.retrieval_top_k, modality=modality)
    logger.debug("Retrieved %d chunks for query: %r", len(docs), question)
    return docs
