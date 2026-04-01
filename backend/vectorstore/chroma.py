"""
vectorstore/chroma.py — ChromaDB wrapper for HomeIntel.

Single collection, all modalities (documents, images, audio).
Metadata filtering lets callers narrow by file type post-retrieval.

Usage:
    from vectorstore.chroma import VectorStore
    vs = VectorStore()
    vs.upsert(chunks)
    results = vs.query("photos of my dog", top_k=6)
"""

import hashlib
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

from config import settings

logger = logging.getLogger(__name__)


# ── Modality enum ─────────────────────────────────────────────────────────────

class Modality(str, Enum):
    DOCUMENT = "document"   # PDF, DOCX, TXT, MD, YML, JSON, etc.
    IMAGE    = "image"      # PNG, JPG, JPEG
    AUDIO    = "audio"      # MP3, WAV (transcript chunks)


EXTENSION_TO_MODALITY: dict[str, Modality] = {
    # Documents
    ".pdf":  Modality.DOCUMENT,
    ".docx": Modality.DOCUMENT,
    ".txt":  Modality.DOCUMENT,
    ".md":   Modality.DOCUMENT,
    ".yml":  Modality.DOCUMENT,
    ".yaml": Modality.DOCUMENT,
    ".json": Modality.DOCUMENT,
    ".conf": Modality.DOCUMENT,
    ".env":  Modality.DOCUMENT,
    # Images
    ".png":  Modality.IMAGE,
    ".jpg":  Modality.IMAGE,
    ".jpeg": Modality.IMAGE,
    # Audio
    ".mp3":  Modality.AUDIO,
    ".wav":  Modality.AUDIO,
}


def modality_for_path(path: str | Path) -> Modality:
    """Infer modality from file extension."""
    ext = Path(path).suffix.lower()
    return EXTENSION_TO_MODALITY.get(ext, Modality.DOCUMENT)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chunk_id(file_path: str, chunk_index: int) -> str:
    """
    Stable, deterministic ID for a chunk.

    Using a hash of (path + index) means re-indexing the same file
    always produces the same IDs, so ChromaDB upserts update in place
    rather than creating duplicates.
    """
    raw = f"{file_path}::{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


def _file_id(file_path: str) -> str:
    """Stable ID representing an entire file (used for deletion)."""
    return hashlib.md5(file_path.encode()).hexdigest()


# ── VectorStore ───────────────────────────────────────────────────────────────

class VectorStore:
    """
    Thin wrapper around ChromaDB + LangChain for HomeIntel.

    Responsibilities:
      - Manage a single persistent Chroma collection
      - Upsert document chunks with rich metadata
      - Query by semantic similarity, optionally filtered by modality
      - Delete all chunks belonging to a file (for re-indexing)
      - Report basic collection stats
    """

    def __init__(self) -> None:
        self._ensure_chroma_path()
        self._embeddings = self._build_embeddings()
        self._client     = self._build_client()
        self._store      = self._build_store()
        logger.info(
            "VectorStore ready — collection=%s path=%s count=%d",
            settings.chroma_collection_name,
            settings.chroma_path,
            self.count(),
        )

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _ensure_chroma_path(self) -> None:
        """Create the ChromaDB directory on the NAS if it doesn't exist."""
        path = Path(settings.chroma_path)
        path.mkdir(parents=True, exist_ok=True)
        logger.debug("Chroma persistence path: %s", path)

    def _build_embeddings(self) -> OllamaEmbeddings:
        return OllamaEmbeddings(
            model=settings.ollama_embed_model,
            base_url=settings.ollama_base_url_str,
        )

    def _build_client(self) -> chromadb.PersistentClient:
        """
        PersistentClient writes directly to disk (your NAS mount).
        anonymized_telemetry=False keeps Chroma from phoning home.
        """
        return chromadb.PersistentClient(
            path=str(settings.chroma_path),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,           # lets us wipe in dev/testing
            ),
        )

    def _build_store(self) -> Chroma:
        """
        LangChain Chroma wrapper — used for high-level upsert/query.
        We pass our pre-built client so there's only one connection.
        """
        return Chroma(
            client=self._client,
            collection_name=settings.chroma_collection_name,
            embedding_function=self._embeddings,
        )

    # ── Write ─────────────────────────────────────────────────────────────────

    def upsert(self, documents: list[Document]) -> None:
        """
        Upsert a list of LangChain Documents into the collection.

        Each Document must have these metadata keys set by the processor:
            file_path   : str  — absolute path to the source file
            chunk_index : int  — position of this chunk within the file
            modality    : str  — one of Modality enum values
            file_name   : str  — basename of the source file
            file_ext    : str  — lowercase extension e.g. ".pdf"

        Optional metadata (add as available):
            title       : str  — document title or image caption
            created_at  : str  — ISO timestamp
            duration_s  : int  — audio duration in seconds
        """
        if not documents:
            logger.debug("upsert called with empty list, skipping")
            return

        ids = [
            _chunk_id(doc.metadata["file_path"], doc.metadata["chunk_index"])
            for doc in documents
        ]

        # Ensure modality is set; infer from extension if missing
        for doc in documents:
            if "modality" not in doc.metadata:
                doc.metadata["modality"] = modality_for_path(
                    doc.metadata.get("file_path", "")
                ).value

        self._store.add_documents(documents=documents, ids=ids)
        logger.info(
            "Upserted %d chunks from %s",
            len(documents),
            documents[0].metadata.get("file_name", "unknown"),
        )

    # ── Read ──────────────────────────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        modality: Optional[Modality] = None,
        file_ext: Optional[str] = None,
    ) -> list[Document]:
        """
        Semantic similarity search.

        Args:
            query_text : Natural language query.
            top_k      : Number of results (defaults to settings.retrieval_top_k).
            modality   : Optional — restrict to "document", "image", or "audio".
            file_ext   : Optional — restrict to a specific extension e.g. ".pdf".

        Returns:
            List of Documents ordered by relevance (most relevant first).
        """
        k = top_k or settings.retrieval_top_k

        where: Optional[dict[str, Any]] = None
        if modality and file_ext:
            where = {"$and": [
                {"modality": modality.value},
                {"file_ext": file_ext},
            ]}
        elif modality:
            where = {"modality": modality.value}
        elif file_ext:
            where = {"file_ext": file_ext}

        kwargs: dict[str, Any] = {"k": k}
        if where:
            kwargs["filter"] = where

        results = self._store.similarity_search(query_text, **kwargs)
        logger.debug("Query returned %d results for: %r", len(results), query_text)
        return results

    def query_with_scores(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        modality: Optional[Modality] = None,
    ) -> list[tuple[Document, float]]:
        """
        Same as query() but returns (Document, distance) tuples.
        Lower distance = more similar. Useful for debugging retrieval quality.
        """
        k = top_k or settings.retrieval_top_k
        where = {"modality": modality.value} if modality else None

        kwargs: dict[str, Any] = {"k": k}
        if where:
            kwargs["filter"] = where

        return self._store.similarity_search_with_score(query_text, **kwargs)

    # ── Delete ────────────────────────────────────────────────────────────────

    def delete_file(self, file_path: str) -> None:
        """
        Remove all chunks belonging to a file from the collection.

        Called by the watcher when a file is deleted or modified
        (modified = delete old chunks, then upsert new ones).
        """
        collection = self._client.get_collection(settings.chroma_collection_name)
        results = collection.get(where={"file_path": file_path})

        if not results["ids"]:
            logger.debug("No chunks found for %s, nothing to delete", file_path)
            return

        collection.delete(ids=results["ids"])
        logger.info("Deleted %d chunks for %s", len(results["ids"]), file_path)

    def delete_all(self) -> None:
        """
        Wipe the entire collection. 
        Dev/testing only — will prompt for confirmation in CLI scripts.
        """
        self._client.reset()
        # Rebuild store after reset
        self._store = self._build_store()
        logger.warning("Collection wiped — all vectors deleted")

    # ── Stats ─────────────────────────────────────────────────────────────────

    def count(self) -> int:
        """Total number of chunks in the collection."""
        try:
            collection = self._client.get_collection(settings.chroma_collection_name)
            return collection.count()
        except Exception:
            return 0

    def stats(self) -> dict[str, Any]:
        """
        Per-modality breakdown of indexed chunks.
        Used by the /status API endpoint.
        """
        collection = self._client.get_collection(settings.chroma_collection_name)
        total = collection.count()

        counts: dict[str, int] = {}
        for modality in Modality:
            results = collection.get(where={"modality": modality.value})
            counts[modality.value] = len(results["ids"])

        return {
            "total_chunks": total,
            "by_modality": counts,
            "collection": settings.chroma_collection_name,
            "persist_path": str(settings.chroma_path),
        }