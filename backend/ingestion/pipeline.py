"""
ingestion/pipeline.py — Orchestrate document ingestion end-to-end.

Usage:
    from ingestion.pipeline import ingest_file
    chunk_count = ingest_file("/path/to/your/file.pdf")
"""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document

from config import settings
from vectorstore.chroma import VectorStore, Modality
from ingestion.processors.document import parse_document
from ingestion.chunker import chunk_text

logger = logging.getLogger(__name__)

DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".yml", ".yaml", ".json"}

_vs: Optional[VectorStore] = None


def _get_vs() -> VectorStore:
    global _vs
    if _vs is None:
        _vs = VectorStore()
    return _vs


def ingest_file(path: str) -> int:
    """
    Ingest a single file into ChromaDB.

    Parses the file, chunks the text, generates embeddings, and upserts
    into the vector store with full metadata.

    Returns the number of chunks upserted (0 if file is empty or unsupported).
    """
    p = Path(os.path.normpath(path))

    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    ext = p.suffix.lower()

    if ext not in DOCUMENT_EXTENSIONS:
        raise ValueError(
            f"Unsupported extension '{ext}' for document ingestion. "
            f"Supported: {sorted(DOCUMENT_EXTENSIONS)}"
        )

    logger.info("Ingesting %s", p)

    text, title = parse_document(p)
    chunks = chunk_text(text)

    if not chunks:
        logger.warning("No text extracted from %s — skipping", p.name)
        return 0

    now = datetime.now(timezone.utc).isoformat()
    documents: list[Document] = []

    for i, chunk in enumerate(chunks):
        meta: dict = {
            "file_path":   str(p),
            "file_name":   p.name,
            "file_ext":    ext,
            "chunk_index": i,
            "modality":    Modality.DOCUMENT.value,
            "created_at":  now,
        }
        if title:
            meta["title"] = title

        documents.append(Document(page_content=chunk, metadata=meta))

    vs = _get_vs()
    vs.delete_file(str(p))
    vs.upsert(documents)
    logger.info("Ingested %d chunks from %s", len(documents), p.name)
    return len(documents)
