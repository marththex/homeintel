"""
api/files.py — File serving and visual similarity search endpoints.

GET  /file?path=...          Stream a NAS file to the browser (images, etc.)
GET  /thumb?path=...&w=...   Cached JPEG thumbnail of a NAS image
POST /visual-search          Upload an image, find visually similar photos via CLIP
"""

import hashlib
import logging
import mimetypes
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from fastapi.responses import FileResponse
from PIL import Image, ImageOps

from config import settings
from vectorstore.clip import get_clip_store

logger = logging.getLogger(__name__)
router = APIRouter()

_THUMB_DIR = Path(__file__).parent.parent / ".cache" / "thumbs"
_THUMB_WIDTHS = {320, 480, 768}  # whitelist so the cache can't be blown up
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


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


def _serve_original(p: Path) -> FileResponse:
    media_type, _ = mimetypes.guess_type(str(p))
    return FileResponse(
        str(p),
        media_type=media_type or "application/octet-stream",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@router.get("/file")
async def serve_file(path: str = Query(..., description="Absolute NAS file path")):
    """Stream a file from the NAS. Restricted to paths under NAS_WATCH_PATH."""
    return _serve_original(_validate_path(path))


@router.get("/thumb")
def serve_thumb(
    path: str = Query(..., description="Absolute NAS file path"),
    w: int = Query(default=480, description="Target width; one of 320/480/768"),
):
    """
    Serve a cached JPEG thumbnail of a NAS image. Sync `def` on purpose — the
    Pillow work runs in FastAPI's threadpool instead of blocking the event loop.
    Cache key includes mtime, so edited photos re-thumbnail automatically.
    """
    p = _validate_path(path)
    if p.suffix.lower() not in _IMAGE_EXTS:
        return _serve_original(p)
    if w not in _THUMB_WIDTHS:
        w = 480

    key = hashlib.sha1(f"{p}|{p.stat().st_mtime_ns}|{w}".encode()).hexdigest()
    out = _THUMB_DIR / f"{key}.jpg"
    if not out.exists():
        _THUMB_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with Image.open(p) as im:
                im = ImageOps.exif_transpose(im)  # phone photos carry rotation in EXIF
                im = im.convert("RGB")
                im.thumbnail((w, w * 4))
                im.save(out, "JPEG", quality=82)
        except Exception as exc:
            logger.warning("Thumbnail failed for %s: %s", p, exc)
            return _serve_original(p)
    return FileResponse(
        str(out),
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=604800"},
    )


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
