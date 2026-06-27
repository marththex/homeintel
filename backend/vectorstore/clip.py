"""
vectorstore/clip.py — CLIP-based visual similarity store.

Uses openai/clip-vit-large-patch14 (768-dim) to embed photos as dense
vectors in a sibling Qdrant collection `homeintel_visual`.

Separate from the caption-based `homeintel` collection — encodes raw
image pixels, enabling query-by-photo ("find photos that look like this").
"""

import hashlib
import io
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)
from transformers import CLIPModel, CLIPProcessor

from config import settings

logger = logging.getLogger(__name__)

_CLIP_DIM       = 768
_VECTOR_NAME    = "clip"
_SCORE_THRESHOLD = 0.70
_COLLECTION_SUFFIX = "_visual"


def _collection_name() -> str:
    return f"{settings.qdrant_collection_name}{_COLLECTION_SUFFIX}"


def _point_id(file_path: str) -> str:
    """One point per photo — stable UUID from file path."""
    hex32 = hashlib.md5(file_path.encode()).hexdigest()
    return str(uuid.UUID(hex32))


class CLIPVisualStore:
    """
    Qdrant collection backed by CLIP image embeddings.

    One point per photo (no chunking — images are atomic).
    Metadata: file_path, file_name, created_at.
    """

    def __init__(self) -> None:
        self._client = self._build_client()
        self._model, self._processor = self._load_clip()
        self._ensure_collection()
        count = self._client.count(_collection_name()).count
        logger.info(
            "CLIPVisualStore ready — collection=%s count=%d",
            _collection_name(), count,
        )

    def _build_client(self) -> QdrantClient:
        kwargs: dict[str, Any] = {"url": settings.qdrant_url}
        if settings.qdrant_api_key:
            kwargs["api_key"] = settings.qdrant_api_key
        return QdrantClient(**kwargs)

    def _load_clip(self) -> tuple[CLIPModel, CLIPProcessor]:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = settings.clip_model
        logger.info("Loading CLIP model %s on %s", model_id, device)
        model = CLIPModel.from_pretrained(model_id).to(device)
        processor = CLIPProcessor.from_pretrained(model_id)
        model.eval()
        logger.info("CLIP model loaded")
        return model, processor

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if _collection_name() in existing:
            return
        self._client.create_collection(
            collection_name=_collection_name(),
            vectors_config={
                _VECTOR_NAME: VectorParams(size=_CLIP_DIM, distance=Distance.COSINE)
            },
        )
        logger.info("Created CLIP collection: %s", _collection_name())

    def _embed_image(self, img: Image.Image) -> list[float]:
        return self._embed_images([img])[0]

    def _embed_images(self, imgs: list[Image.Image]) -> list[list[float]]:
        """Embed a batch of images in a single GPU forward pass."""
        device = next(self._model.parameters()).device
        inputs = self._processor(images=imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            out = self._model.get_image_features(**inputs)
            feats = self._project(out)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().tolist()

    def _project(self, out):
        """
        Return the image-embedding tensor across transformers versions.

        transformers <5 returned the projected embeds directly from
        get_image_features. transformers 5.x returns a BaseModelOutputWithPooling
        whose pooler_output is already the projected image embedding
        (projection_dim, 768 for ViT-L/14).
        """
        if torch.is_tensor(out):
            return out
        embeds = getattr(out, "image_embeds", None)
        if embeds is not None:
            return embeds
        return out.pooler_output

    # ── Write ─────────────────────────────────────────────────────────────────

    def upsert(self, file_path: str, img: Image.Image) -> None:
        """Embed one image and upsert into the visual collection."""
        self.upsert_batch([(file_path, img)])

    def upsert_batch(self, items: list[tuple[str, Image.Image]]) -> None:
        """
        Embed and upsert a batch of (file_path, image) pairs in one GPU pass
        and one Qdrant request. Much faster than calling upsert() per image.
        """
        if not items:
            return
        vecs = self._embed_images([img for _, img in items])
        now = datetime.now(timezone.utc).isoformat()
        points = [
            PointStruct(
                id=_point_id(fp),
                vector={_VECTOR_NAME: vec},
                payload={"file_path": fp, "file_name": Path(fp).name, "created_at": now},
            )
            for (fp, _), vec in zip(items, vecs)
        ]
        self._client.upsert(collection_name=_collection_name(), points=points)

    def indexed_file_paths(self) -> set[str]:
        """Return all file_paths already in the visual collection (resume support)."""
        paths: set[str] = set()
        next_offset = None
        while True:
            points, next_offset = self._client.scroll(
                collection_name=_collection_name(),
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

    def delete_file(self, file_path: str) -> None:
        self._client.delete(
            collection_name=_collection_name(),
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="file_path", match=MatchValue(value=file_path))]
                )
            ),
        )

    def delete_all(self) -> None:
        self._client.delete_collection(_collection_name())
        self._ensure_collection()
        logger.warning("Visual collection wiped")

    # ── Read ──────────────────────────────────────────────────────────────────

    def search(self, image_bytes: bytes, top_k: int = 10) -> list[dict]:
        """
        Find the most visually similar photos to the uploaded image.
        Returns results with score >= _SCORE_THRESHOLD, ranked by cosine similarity.
        """
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        vec = self._embed_image(img)

        response = self._client.query_points(
            collection_name=_collection_name(),
            query=vec,
            using=_VECTOR_NAME,
            limit=top_k,
            with_payload=True,
            score_threshold=_SCORE_THRESHOLD,
        )
        return [
            {
                "file_path": p.payload["file_path"],
                "file_name": p.payload["file_name"],
                "score": p.score,
            }
            for p in response.points
        ]

    def count(self) -> int:
        try:
            return self._client.count(_collection_name()).count
        except Exception:
            return 0
