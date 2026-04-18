"""
api/status.py — GET /health and GET /stats endpoints.
"""

import logging
from typing import Optional

import httpx
from fastapi import APIRouter

from config import settings
from vectorstore.chroma import VectorStore

logger = logging.getLogger(__name__)
router = APIRouter()

_vs: Optional[VectorStore] = None


def _get_vs() -> VectorStore:
    global _vs
    if _vs is None:
        _vs = VectorStore()
    return _vs


@router.get("/health")
async def health() -> dict:
    """
    Check Ollama connectivity and ChromaDB accessibility.

    Returns status "ok" only if both services are reachable.
    Returns status "degraded" if either is down.
    """
    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{settings.ollama_base_url_str}/api/tags")
            ollama_ok = resp.status_code == 200
    except Exception as exc:
        logger.debug("Ollama health check failed: %s", exc)

    chroma_ok = False
    try:
        _get_vs().count()
        chroma_ok = True
    except Exception as exc:
        logger.debug("ChromaDB health check failed: %s", exc)

    return {
        "status": "ok" if (ollama_ok and chroma_ok) else "degraded",
        "ollama": ollama_ok,
        "chromadb": chroma_ok,
    }


@router.get("/stats")
async def stats() -> dict:
    """Return per-modality chunk counts from ChromaDB."""
    return _get_vs().stats()
