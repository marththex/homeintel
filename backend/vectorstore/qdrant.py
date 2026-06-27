"""
vectorstore/qdrant.py — Qdrant wrapper for HomeIntel.

Single collection, named vectors (dense + sparse), all modalities.
Hybrid search (dense + BM25 via fastembed) fused server-side with RRF.
Metadata stored as Qdrant point payload — same schema as the old ChromaDB
wrapper so downstream code (pipeline, retriever, status API) is unchanged.

Usage:
    from vectorstore.qdrant import VectorStore
    vs = VectorStore()
    vs.upsert(chunks)
    results = vs.query("photos of my dog", top_k=6)
"""

import hashlib
import logging
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from fastembed import SparseTextEmbedding
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    Fusion,
    FusionQuery,
    MatchValue,
    MultiVectorComparator,
    MultiVectorConfig,
    Prefetch,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
    PointStruct,
)

from config import settings

logger = logging.getLogger(__name__)

_DENSE_VECTOR   = "dense"
_SPARSE_VECTOR  = "sparse"
_SPARSE_MODEL   = "Qdrant/bm25"
_COLPALI_VECTOR = "colpali"
_COLPALI_DIM    = 128   # patch embedding dim from vidore/colpali-v1.2


# ── Modality enum ─────────────────────────────────────────────────────────────

class Modality(str, Enum):
    DOCUMENT = "document"
    IMAGE    = "image"
    AUDIO    = "audio"


EXTENSION_TO_MODALITY: dict[str, Modality] = {
    ".pdf":  Modality.DOCUMENT,
    ".docx": Modality.DOCUMENT,
    ".txt":  Modality.DOCUMENT,
    ".md":   Modality.DOCUMENT,
    ".yml":  Modality.DOCUMENT,
    ".yaml": Modality.DOCUMENT,
    ".json": Modality.DOCUMENT,
    ".conf": Modality.DOCUMENT,
    ".env":  Modality.DOCUMENT,
    ".png":  Modality.IMAGE,
    ".jpg":  Modality.IMAGE,
    ".jpeg": Modality.IMAGE,
    ".mp3":  Modality.AUDIO,
    ".wav":  Modality.AUDIO,
}


def modality_for_path(path: str | Path) -> Modality:
    ext = Path(path).suffix.lower()
    return EXTENSION_TO_MODALITY.get(ext, Modality.DOCUMENT)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chunk_id(file_path: str, chunk_index: int) -> str:
    """Stable MD5 hex for (path, index) — same logic as old chroma.py."""
    raw = f"{file_path}::{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


def _to_point_id(hex32: str) -> str:
    """Convert 32-char MD5 hex to a UUID string accepted by Qdrant."""
    return str(uuid.UUID(hex32))


# ── VectorStore ───────────────────────────────────────────────────────────────

class VectorStore:
    """
    Qdrant-backed vector store with hybrid dense+sparse retrieval.

    Interface is intentionally identical to the old ChromaDB wrapper so
    ingestion, retriever, and status API code requires only import changes.
    """

    def __init__(self) -> None:
        self._client = self._build_client()
        self._embeddings = self._build_embeddings()
        self._sparse_model = SparseTextEmbedding(model_name=_SPARSE_MODEL)
        self._ensure_collection()
        logger.info(
            "VectorStore ready — collection=%s url=%s count=%d",
            settings.qdrant_collection_name,
            settings.qdrant_url,
            self.count(),
        )

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _build_client(self) -> QdrantClient:
        kwargs: dict[str, Any] = {"url": settings.qdrant_url}
        if settings.qdrant_api_key:
            kwargs["api_key"] = settings.qdrant_api_key
        return QdrantClient(**kwargs)

    def _build_embeddings(self) -> OllamaEmbeddings:
        return OllamaEmbeddings(
            model=settings.ollama_embed_model,
            base_url=settings.ollama_base_url_str,
        )

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if settings.qdrant_collection_name in existing:
            return

        self._client.create_collection(
            collection_name=settings.qdrant_collection_name,
            vectors_config={
                _DENSE_VECTOR: VectorParams(
                    size=settings.embed_dim,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                _SPARSE_VECTOR: SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                )
            },
        )
        logger.info("Created Qdrant collection: %s", settings.qdrant_collection_name)

    # ── Write ─────────────────────────────────────────────────────────────────

    def upsert(self, documents: list[Document]) -> None:
        """
        Upsert LangChain Documents into the collection.

        For each document, computes:
          - Dense vector via Ollama nomic-embed-text
          - Sparse BM25 vector via fastembed

        Metadata keys expected per Document:
            file_path, file_name, file_ext, chunk_index, modality, created_at
        """
        if not documents:
            logger.debug("upsert called with empty list, skipping")
            return

        texts = [doc.page_content for doc in documents]

        # Dense embeddings (Ollama)
        dense_vectors = self._embeddings.embed_documents(texts)

        # Sparse BM25 embeddings (fastembed — local, no Ollama call)
        sparse_results = list(self._sparse_model.embed(texts))

        points: list[PointStruct] = []
        for doc, dense, sparse in zip(documents, dense_vectors, sparse_results):
            if "modality" not in doc.metadata:
                doc.metadata["modality"] = modality_for_path(
                    doc.metadata.get("file_path", "")
                ).value

            hex_id = _chunk_id(
                doc.metadata["file_path"],
                doc.metadata["chunk_index"],
            )
            payload = {"page_content": doc.page_content, **doc.metadata}

            points.append(
                PointStruct(
                    id=_to_point_id(hex_id),
                    vector={
                        _DENSE_VECTOR: dense,
                        _SPARSE_VECTOR: SparseVector(
                            indices=sparse.indices.tolist(),
                            values=sparse.values.tolist(),
                        ),
                    },
                    payload=payload,
                )
            )

        self._client.upsert(
            collection_name=settings.qdrant_collection_name,
            points=points,
        )
        logger.info(
            "Upserted %d chunks from %s",
            len(points),
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
        Hybrid dense+sparse search with RRF fusion, optionally filtered by modality.

        Fetches 3×top_k candidates from each vector branch, fuses with RRF,
        then returns top_k results. The retriever applies cross-encoder reranking
        on top of this if enabled.
        """
        k = top_k or settings.retrieval_top_k
        prefetch_k = k * 3

        dense_query = self._embeddings.embed_query(query_text)
        sparse_query_list = list(self._sparse_model.embed([query_text]))
        sparse_query = sparse_query_list[0]

        query_filter = self._build_filter(modality, file_ext)

        response = self._client.query_points(
            collection_name=settings.qdrant_collection_name,
            prefetch=[
                Prefetch(query=dense_query, using=_DENSE_VECTOR, limit=prefetch_k),
                Prefetch(
                    query=SparseVector(
                        indices=sparse_query.indices.tolist(),
                        values=sparse_query.values.tolist(),
                    ),
                    using=_SPARSE_VECTOR,
                    limit=prefetch_k,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=k,
            query_filter=query_filter,
            with_payload=True,
        )

        docs = [self._point_to_document(p) for p in response.points]
        logger.debug("Hybrid query returned %d results for: %r", len(docs), query_text)
        return docs

    def query_with_scores(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        modality: Optional[Modality] = None,
    ) -> list[tuple[Document, float]]:
        """Same as query() but returns (Document, RRF score) tuples."""
        k = top_k or settings.retrieval_top_k
        prefetch_k = k * 3

        dense_query = self._embeddings.embed_query(query_text)
        sparse_query_list = list(self._sparse_model.embed([query_text]))
        sparse_query = sparse_query_list[0]

        query_filter = self._build_filter(modality, None)

        response = self._client.query_points(
            collection_name=settings.qdrant_collection_name,
            prefetch=[
                Prefetch(query=dense_query, using=_DENSE_VECTOR, limit=prefetch_k),
                Prefetch(
                    query=SparseVector(
                        indices=sparse_query.indices.tolist(),
                        values=sparse_query.values.tolist(),
                    ),
                    using=_SPARSE_VECTOR,
                    limit=prefetch_k,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=k,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=False,
        )

        return [(self._point_to_document(p), p.score) for p in response.points]

    # ── Delete ────────────────────────────────────────────────────────────────

    def delete_file(self, file_path: str) -> None:
        """Remove all chunks belonging to a file (called before re-ingestion)."""
        result = self._client.delete(
            collection_name=settings.qdrant_collection_name,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="file_path", match=MatchValue(value=file_path))]
                )
            ),
        )
        logger.info("Deleted chunks for %s (status=%s)", file_path, result.status)

    def delete_all(self) -> None:
        """Wipe and recreate the collection. Dev/testing only."""
        self._client.delete_collection(settings.qdrant_collection_name)
        self._ensure_collection()
        logger.warning("Collection wiped — all vectors deleted")

    # ── Stats ─────────────────────────────────────────────────────────────────

    def count(self) -> int:
        try:
            return self._client.count(settings.qdrant_collection_name).count
        except Exception:
            return 0

    def indexed_file_paths(self) -> set[str]:
        """
        Return the set of all file_paths currently in the collection.

        Used by reindex.py --skip-existing to resume an interrupted run: one
        scroll pass (file_path payload only) instead of a count query per file.
        """
        paths: set[str] = set()
        next_offset = None
        while True:
            points, next_offset = self._client.scroll(
                collection_name=settings.qdrant_collection_name,
                with_payload=["file_path"],
                with_vectors=False,
                limit=1000,
                offset=next_offset,
            )
            for p in points:
                fp = (p.payload or {}).get("file_path")
                if fp:
                    paths.add(fp)
            if next_offset is None:
                break
        return paths

    def stats(self) -> dict[str, Any]:
        """Per-modality chunk breakdown + CLIP visual photo count."""
        total = self.count()
        counts: dict[str, int] = {}
        for modality in Modality:
            result = self._client.count(
                collection_name=settings.qdrant_collection_name,
                count_filter=Filter(
                    must=[FieldCondition(key="modality", match=MatchValue(value=modality.value))]
                ),
            )
            counts[modality.value] = result.count

        # CLIP visual collection (homeintel_visual) — one vector per photo.
        # Counted via the existing client, no CLIP model load (stats stays cheap).
        visual = 0
        try:
            visual = self._client.count(f"{settings.qdrant_collection_name}_visual").count
        except Exception:
            visual = 0

        return {
            **counts,           # "document": N, "image": N, "audio": N
            "visual": visual,   # CLIP photo vectors
            "total": total,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_filter(
        self,
        modality: Optional[Modality],
        file_ext: Optional[str],
    ) -> Optional[Filter]:
        conditions = []
        if modality:
            conditions.append(
                FieldCondition(key="modality", match=MatchValue(value=modality.value))
            )
        if file_ext:
            conditions.append(
                FieldCondition(key="file_ext", match=MatchValue(value=file_ext))
            )
        return Filter(must=conditions) if conditions else None

    @staticmethod
    def _point_to_document(point) -> Document:
        payload = dict(point.payload)
        page_content = payload.pop("page_content", "")
        return Document(page_content=page_content, metadata=payload)

    # ── ColPali collection ────────────────────────────────────────────────────

    @property
    def _colpali_collection(self) -> str:
        return f"{settings.qdrant_collection_name}_colpali"

    def ensure_colpali_collection(self) -> None:
        """Create the ColPali sibling collection if it doesn't exist."""
        existing = {c.name for c in self._client.get_collections().collections}
        if self._colpali_collection in existing:
            return

        self._client.create_collection(
            collection_name=self._colpali_collection,
            vectors_config={
                _COLPALI_VECTOR: VectorParams(
                    size=_COLPALI_DIM,
                    distance=Distance.COSINE,
                    multivector_config=MultiVectorConfig(
                        comparator=MultiVectorComparator.MAX_SIM,
                    ),
                )
            },
        )
        logger.info("Created ColPali collection: %s", self._colpali_collection)

    def upsert_colpali(self, points: list[PointStruct]) -> None:
        """Upsert ColPali page embeddings into the sibling collection."""
        self.ensure_colpali_collection()
        self._client.upsert(
            collection_name=self._colpali_collection,
            points=points,
        )
        logger.info("Upserted %d ColPali page(s)", len(points))

    def query_colpali(
        self, query_vectors: list[list[float]], top_k: int
    ) -> list[Document]:
        """MaxSim search over ColPali page embeddings."""
        response = self._client.query_points(
            collection_name=self._colpali_collection,
            query=query_vectors,
            using=_COLPALI_VECTOR,
            limit=top_k,
            with_payload=True,
        )
        return [self._point_to_document(p) for p in response.points]

    def delete_file_colpali(self, file_path: str) -> None:
        """Remove ColPali page embeddings for a file."""
        existing = {c.name for c in self._client.get_collections().collections}
        if self._colpali_collection not in existing:
            return
        self._client.delete(
            collection_name=self._colpali_collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="file_path", match=MatchValue(value=file_path))]
                )
            ),
        )
