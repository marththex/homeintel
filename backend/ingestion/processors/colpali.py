"""
ingestion/processors/colpali.py — ColPali visual page embedding for PDFs.

ColPali (vidore/colpali-v1.2) embeds each PDF page as a set of patch-level
vectors (multi-vector) that capture visual layout, charts, tables, and text.
These are stored in a separate Qdrant collection ({name}_colpali) and searched
via MaxSim scoring alongside the regular text pipeline.

This module is NOT called inline by the file watcher — it is a heavy GPU
batch job (~5s/page on RTX 5080).  Run it via:

    python scripts/run_colpali.py [--path Z:/]

Set COLPALI_ENABLED=true in .env to activate ColPali retrieval in the API.

Requirements (install separately — not in the main watcher process):
    pip install colpali-engine>=0.3.0 pypdfium2>=4.0.0
"""

import hashlib
import logging
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_model = None
_processor = None


def _get_model_and_processor():
    global _model, _processor
    if _model is None:
        import torch
        from colpali_engine.models import ColPali, ColPaliProcessor

        from config import settings

        logger.info("Loading ColPali model: %s", settings.colpali_model)
        _model = ColPali.from_pretrained(
            settings.colpali_model,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        ).eval()
        _processor = ColPaliProcessor.from_pretrained(settings.colpali_model)
    return _model, _processor


def render_pdf_pages(pdf_path: Path) -> list:
    """Render all pages of a PDF to PIL Images using pypdfium2."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(pdf_path))
    pages = []
    for i in range(len(doc)):
        bitmap = doc[i].render(scale=2.0)   # ~144 DPI — good balance for ColPali
        pages.append(bitmap.to_pil())
    logger.debug("Rendered %d pages from %s", len(pages), pdf_path.name)
    return pages


def embed_pages(images: list) -> list[list[list[float]]]:
    """
    Embed a list of PIL images with ColPali.

    Returns a list of multi-vectors (one per page), where each multi-vector
    is a list of patch-level 128-dim float vectors.
    """
    import torch

    model, processor = _get_model_and_processor()
    results: list[list[list[float]]] = []

    for img in images:
        batch = processor.process_images([img]).to(model.device)
        with torch.no_grad():
            embeddings = model(**batch)   # (1, num_patches, 128)
        patch_vecs = embeddings[0].float().cpu().tolist()
        results.append(patch_vecs)

    return results


def embed_query(query_text: str) -> list[list[float]]:
    """
    Encode a text query with the ColPali text encoder.

    Returns a multi-vector (list of token-level vectors) for MaxSim search.
    """
    import torch

    model, processor = _get_model_and_processor()
    batch = processor.process_queries([query_text]).to(model.device)
    with torch.no_grad():
        embeddings = model(**batch)   # (1, num_tokens, 128)
    return embeddings[0].float().cpu().tolist()


def _page_point_id(file_path: str, page_index: int) -> str:
    raw = f"{file_path}::colpali::{page_index}"
    return str(uuid.UUID(hashlib.md5(raw.encode()).hexdigest()))


def index_pdf(
    pdf_path: Path,
    vs,   # VectorStore — passed in to avoid circular import
    *,
    force: bool = False,
) -> int:
    """
    Index all pages of a PDF with ColPali and upsert into the colpali collection.

    Args:
        pdf_path:  Path to the PDF file.
        vs:        VectorStore instance (provides upsert_colpali / delete_file_colpali).
        force:     Re-index even if pages already exist.

    Returns the number of pages indexed.
    """
    from qdrant_client.models import PointStruct

    if force:
        vs.delete_file_colpali(str(pdf_path))

    images = render_pdf_pages(pdf_path)
    if not images:
        logger.warning("No pages rendered from %s", pdf_path.name)
        return 0

    page_embeddings = embed_pages(images)

    points: list[PointStruct] = []
    for i, patch_vecs in enumerate(page_embeddings):
        points.append(
            PointStruct(
                id=_page_point_id(str(pdf_path), i),
                vector={"colpali": patch_vecs},
                payload={
                    "file_path": str(pdf_path),
                    "file_name": pdf_path.name,
                    "file_ext":  ".pdf",
                    "page":      i,
                    "modality":  "document",
                },
            )
        )

    vs.upsert_colpali(points)
    logger.info("ColPali indexed %d pages from %s", len(points), pdf_path.name)
    return len(points)
