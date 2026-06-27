"""
ingestion/processors/image.py — Image captioning via Ollama vision model.

Requires OLLAMA_VISION_MODEL to be set in .env (e.g. qwen2.5vl:7b).
If unset, raises ValueError so the watcher silently skips image files.

The caption is treated as plain text and fed into the standard text
embedding pipeline (nomic-embed-text + BM25), storing as modality=image.
"""

import base64
import logging
from pathlib import Path

import httpx

from config import settings

logger = logging.getLogger(__name__)

_CAPTION_PROMPT = (
    "Describe this image in detail. Include: subjects, objects, text visible in the "
    "image, colours, scene, location context, and any other notable elements. "
    "Be thorough — your description will be used to make the image searchable."
)


def parse_image(path: Path) -> tuple[str, str]:
    """
    Caption an image using the configured Ollama vision model.

    Returns (caption_text, title) where title is derived from the filename.
    Raises ValueError if OLLAMA_VISION_MODEL is not configured.
    """
    if not settings.ollama_vision_model:
        raise ValueError(
            "OLLAMA_VISION_MODEL is not set — skipping image ingestion. "
            "Set it to a vision-capable Ollama model (e.g. qwen2.5vl:7b)."
        )

    with open(path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    logger.debug("Captioning %s with %s", path.name, settings.ollama_vision_model)

    try:
        resp = httpx.post(
            f"{settings.ollama_base_url_str}/api/generate",
            json={
                "model": settings.ollama_vision_model,
                "prompt": _CAPTION_PROMPT,
                "images": [image_b64],
                "stream": False,
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        caption = resp.json().get("response", "").strip()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"Ollama vision API returned {exc.response.status_code} for {path.name}: "
            f"{exc.response.text[:200]}"
        ) from exc
    except httpx.RequestError as exc:
        raise RuntimeError(f"Ollama vision API unreachable: {exc}") from exc

    if not caption:
        logger.warning("Empty caption for %s — skipping", path.name)
        return "", ""

    title = path.stem.replace("_", " ").replace("-", " ").title()
    logger.info("Captioned %s (%d chars)", path.name, len(caption))
    return caption, title
