"""
api/files.py — File serving and visual similarity search endpoints.

GET  /file?path=...          Stream a NAS file to the browser (images, etc.)
POST /visual-search          Upload an image, find visually similar photos via CLIP
"""

import logging
import mimetypes
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from fastapi.responses import FileResponse

from config import settings
from vectorstore.clip import get_clip_store

logger = logging.getLogger(__name__)
router = APIRouter()


def _validate_path(path: str) -> Path:
    """Resolve path and ensure it's under NAS_WATCH_PATH."""
    p = Path(path).resolve()
    nas_root = Path(settings.nas_watch_path).resolve()
    try:
        p.relative_to(nas_root)
    except ValueError:
        raise HTTPException(status_code=403, detail="Path outside NAS root")
    if not p.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return p


@router.get("/file")
async def serve_file(path: str = Query(..., description="Absolute NAS file path")):
    """Stream a file from the NAS. Restricted to paths under NAS_WATCH_PATH."""
    p = _validate_path(path)
    media_type, _ = mimetypes.guess_type(str(p))
    return FileResponse(str(p), media_type=media_type or "application/octet-stream")


@router.post("/visual-search")
async def visual_search(
    file: UploadFile = File(...),
    top_k: int = Query(default=10, ge=1, le=50),
):
    """
    Upload an image and find visually similar photos in homeintel_visual.
    Returns top_k results ranked by CLIP cosine similarity, filtered to >= 0.70.
    """
    data = await file.read()
    try:
        results = get_clip_store().search(image_bytes=data, top_k=top_k)
    except Exception as exc:
        logger.exception("Visual search failed")
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "results": [
            {
                "file_path": r["file_path"],
                "file_name": r["file_name"],
                "score": round(r["score"], 4),
            }
            for r in results
        ]
    }
