"""
rag/retriever.py — Hybrid retrieval + cross-encoder reranking for the RAG chain.

Flow:
  1. Qdrant hybrid search (dense + BM25 RRF) — fetches 3×top_k candidates
  2. bge-reranker-v2-m3 cross-encoder reranks candidates (if RERANKER_ENABLED)
  3. Return top_k docs
"""

import logging
from typing import Optional

from langchain_core.documents import Document

from config import settings
from vectorstore.qdrant import VectorStore, Modality

logger = logging.getLogger(__name__)

_vs: Optional[VectorStore] = None
_reranker = None


def _get_vs() -> VectorStore:
    global _vs
    if _vs is None:
        _vs = VectorStore()
    return _vs


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        logger.info("Loading reranker model: %s", settings.reranker_model)
        _reranker = CrossEncoder(settings.reranker_model, max_length=512)
    return _reranker


def retrieve(question: str, modality_filter: Optional[str] = None) -> list[Document]:
    """
    Retrieve the top-k most relevant document chunks for a question.

    Runs hybrid Qdrant search (dense + BM25 RRF) over 3×top_k candidates,
    then optionally reranks with bge-reranker-v2-m3 before truncating.

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

    # Fetch more candidates for reranker to work with
    pre_k = settings.retrieval_top_k * 3
    docs = _get_vs().query(question, top_k=pre_k, modality=modality)

    if settings.reranker_enabled and len(docs) > 1:
        docs = _rerank(question, docs, settings.retrieval_top_k)
    else:
        docs = docs[:settings.retrieval_top_k]

    logger.debug(
        "Retrieved %d chunks for query: %r (reranker=%s)",
        len(docs),
        question,
        settings.reranker_enabled,
    )
    return docs


def _rerank(question: str, docs: list[Document], top_k: int) -> list[Document]:
    """Score (question, chunk) pairs with a cross-encoder and return top_k."""
    try:
        reranker = _get_reranker()
        pairs = [(question, doc.page_content) for doc in docs]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:top_k]]
    except Exception as exc:
        logger.warning("Reranker failed (%s) — returning unranked top_k", exc)
        return docs[:top_k]
