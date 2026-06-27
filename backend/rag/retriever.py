"""
rag/retriever.py — Hybrid retrieval + cross-encoder reranking for the RAG chain.

Flow:
  1. Qdrant hybrid search (dense + BM25 RRF) — fetches 3×top_k candidates
  2. [Optional] ColPali MaxSim search over page images (COLPALI_ENABLED=true)
  3. Merge and deduplicate by file_path if ColPali results added
  4. bge-reranker-v2-m3 cross-encoder reranks candidates (RERANKER_ENABLED=true)
  5. Return top_k docs
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


# Image queries return more results so the user can browse photos in the
# carousel. Document/audio Q&A stays at RETRIEVAL_TOP_K to keep the LLM
# context (and latency) small.
_IMAGE_DEFAULT_TOP_K = 20


def _resolve_top_k(modality: Optional[Modality], override: Optional[int]) -> int:
    """Adaptive result count: explicit override wins, else 20 for images, else default."""
    if override is not None:
        return max(1, min(override, 20))
    if modality == Modality.IMAGE:
        return _IMAGE_DEFAULT_TOP_K
    return settings.retrieval_top_k


def retrieve(
    question: str,
    modality_filter: Optional[str] = None,
    top_k: Optional[int] = None,
) -> list[Document]:
    """
    Retrieve the top-k most relevant chunks for a question.

    Runs hybrid Qdrant search (dense + BM25 RRF) over 3×top_k candidates.
    If COLPALI_ENABLED=true, also runs ColPali MaxSim search over page images
    and merges the results before reranking.
    Then optionally reranks with bge-reranker-v2-m3 before truncating to top_k.

    Args:
        question:        Natural language query.
        modality_filter: Optional modality string ("document", "image", "audio").
                         Invalid values are silently ignored.
        top_k:           Optional override for result count (1-20). When None,
                         uses 20 for image queries and RETRIEVAL_TOP_K otherwise.
    """
    modality: Optional[Modality] = None
    if modality_filter:
        try:
            modality = Modality(modality_filter)
        except ValueError:
            logger.warning("Unknown modality_filter %r — ignoring", modality_filter)

    final_k = _resolve_top_k(modality, top_k)
    # Image captions can span >1 chunk, so a single photo may appear multiple
    # times. Fetch extra candidates (and dedupe below) so "top N" still yields
    # N distinct photos.
    pre_k = final_k * 4 if modality == Modality.IMAGE else final_k * 3
    docs = _get_vs().query(question, top_k=pre_k, modality=modality)

    # Optional ColPali visual retrieval
    if settings.colpali_enabled and not modality_filter:
        docs = _merge_colpali(question, docs, pre_k)

    # Rerank the full candidate set (cost is the same — it scores all pairs
    # either way) so dedup downstream has enough material to fill final_k.
    if settings.reranker_enabled and len(docs) > 1:
        docs = _rerank(question, docs, len(docs))

    # One result per photo for image queries — collapse duplicate file_paths.
    if modality == Modality.IMAGE:
        docs = _dedupe_by_path(docs)

    docs = docs[:final_k]

    logger.debug(
        "Retrieved %d chunks for query: %r (top_k=%d reranker=%s colpali=%s)",
        len(docs), question, final_k, settings.reranker_enabled, settings.colpali_enabled,
    )
    return docs


def _dedupe_by_path(docs: list[Document]) -> list[Document]:
    """Keep only the first (highest-ranked) chunk per file_path, preserving order."""
    seen: set = set()
    out: list[Document] = []
    for d in docs:
        path = d.metadata.get("file_path")
        if path in seen:
            continue
        seen.add(path)
        out.append(d)
    return out


def _merge_colpali(
    question: str, text_docs: list[Document], top_k: int
) -> list[Document]:
    """
    Run ColPali MaxSim search and merge unique results with text_docs.

    ColPali hits are appended only if their file_path is not already in the
    text results — they surface visually-rich pages that pure text search misses.
    """
    try:
        from ingestion.processors.colpali import embed_query
        query_vecs = embed_query(question)
        colpali_docs = _get_vs().query_colpali(query_vecs, top_k=top_k)

        existing_paths = {d.metadata.get("file_path") for d in text_docs}
        new_docs = [d for d in colpali_docs if d.metadata.get("file_path") not in existing_paths]

        if new_docs:
            logger.debug("ColPali added %d new source(s)", len(new_docs))
        return text_docs + new_docs

    except Exception as exc:
        logger.warning("ColPali retrieval failed (%s) — using text results only", exc)
        return text_docs


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
