"""
models/chat.py — Pydantic request/response schemas for the chat API.
"""

from typing import Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question.")
    modality_filter: Optional[str] = Field(
        default=None,
        description="Optional modality to restrict retrieval: 'document', 'image', or 'audio'.",
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Optional override for number of results. Blank uses the adaptive "
                    "default (20 for image queries, RETRIEVAL_TOP_K otherwise).",
    )


class SourceDoc(BaseModel):
    file_name: str
    file_path: str
    modality: str
    excerpt: str = Field(description="First 200 characters of the matched chunk.")


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceDoc]
    model: str
    chunks_used: int
