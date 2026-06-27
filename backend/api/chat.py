"""
api/chat.py — POST /chat endpoint and shared source shaping.
"""

import logging
from fastapi import APIRouter, HTTPException
from langchain_core.documents import Document

from config import settings
from models.chat import ChatRequest, ChatResponse, SourceDoc
from rag.chain import get_chain
from rag.retriever import get_vectorstore
from vectorstore.qdrant import Modality
from security import redact_secrets

logger = logging.getLogger(__name__)
router = APIRouter()


def _captions_for_images(docs: list[Document]) -> dict[str, str]:
    """Batch-fetch full captions for any image-modality docs (keyed by file_path)."""
    paths = [
        d.metadata.get("file_path", "")
        for d in docs
        if d.metadata.get("modality") == Modality.IMAGE.value
    ]
    paths = [p for p in paths if p]
    if not paths:
        return {}
    return get_vectorstore().captions_for(paths)


def build_sources(docs: list[Document], captions: dict[str, str]) -> list[SourceDoc]:
    """
    Shape retrieved docs into SourceDocs.

    Image sources carry their FULL caption (from `captions`, keyed by file_path,
    falling back to page_content); other modalities get a 200-char excerpt.
    Secrets are redacted when REDACT_SECRETS is enabled.
    """
    def _redact(text: str) -> str:
        return redact_secrets(text) if settings.redact_secrets else text

    sources: list[SourceDoc] = []
    for doc in docs:
        modality = doc.metadata.get("modality", "")
        fp = doc.metadata.get("file_path", "")
        if modality == Modality.IMAGE.value:
            excerpt = _redact(captions.get(fp) or doc.page_content)
        else:
            excerpt = _redact(doc.page_content[:200])
        sources.append(
            SourceDoc(
                file_name=doc.metadata.get("file_name", ""),
                file_path=fp,
                modality=modality,
                excerpt=excerpt,
            )
        )
    return sources


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Ask HomeIntel a question about your files.

    Retrieves the most relevant chunks from Qdrant, sends them to the local LLM
    as context, and returns the answer with source attribution.
    """
    try:
        chain = get_chain()
        result = chain.run(request.question, request.modality_filter, request.top_k)
    except Exception as exc:
        logger.exception("RAG chain failed for question: %r", request.question)
        raise HTTPException(status_code=503, detail=f"LLM unavailable: {exc}") from exc

    docs = result["docs"]
    sources = build_sources(docs, _captions_for_images(docs))

    return ChatResponse(
        answer=result["answer"],
        sources=sources,
        model=settings.ollama_llm_model,
        chunks_used=len(docs),
    )
