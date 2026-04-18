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
