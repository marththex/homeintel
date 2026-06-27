"""
api/chat.py — POST /chat endpoint.
"""

import logging
from fastapi import APIRouter, HTTPException

from config import settings
from models.chat import ChatRequest, ChatResponse, SourceDoc
from rag.chain import get_chain
from security import redact_secrets

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Ask HomeIntel a question about your files.

    Retrieves the most relevant chunks from ChromaDB, sends them to the
    local LLM as context, and returns the answer with source attribution.
    """
    try:
        chain = get_chain()
        result = chain.run(request.question, request.modality_filter, request.top_k)
    except Exception as exc:
        logger.exception("RAG chain failed for question: %r", request.question)
        raise HTTPException(status_code=503, detail=f"LLM unavailable: {exc}") from exc

    def _excerpt(text: str) -> str:
        snippet = text[:200]
        return redact_secrets(snippet) if settings.redact_secrets else snippet

    sources = [
        SourceDoc(
            file_name=doc.metadata.get("file_name", ""),
            file_path=doc.metadata.get("file_path", ""),
            modality=doc.metadata.get("modality", ""),
            excerpt=_excerpt(doc.page_content),
        )
        for doc in result["docs"]
    ]

    return ChatResponse(
        answer=result["answer"],
        sources=sources,
        model=settings.ollama_llm_model,
        chunks_used=len(result["docs"]),
    )
