"""
ingestion/processors/document.py — Parse documents into raw text.

Supported extensions:
  .pdf  .docx  → Docling (structured markdown output)
  .txt  .md    → plain file read
  .yml  .yaml  → PyYAML → formatted key-value text
  .json        → json → formatted key-value text

Returns (text, title) where title may be None.

Docling notes:
  - Produces markdown from PDF/DOCX, preserving headings, tables, and lists.
  - Optional VLM picture description is gated by DOCLING_VLM_ENABLED in .env
    (default False); requires VRAM headroom alongside the configured LLM
    (OLLAMA_LLM_MODEL) + nomic-embed-text.
  - No pypdf fallback needed — Docling uses its own PDF pipeline.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


# ── Public entry point ────────────────────────────────────────────────────────

def parse_document(path: Path) -> tuple[str, Optional[str]]:
    """Parse a document file and return (full_text, title_or_None)."""
    ext = path.suffix.lower()
    parsers = {
        ".pdf":  _parse_with_docling,
        ".docx": _parse_with_docling,
        ".txt":  _parse_text,
        ".md":   _parse_text,
        ".yml":  _parse_yaml,
        ".yaml": _parse_yaml,
        ".json": _parse_json,
    }
    parser = parsers.get(ext)
    if parser is None:
        raise ValueError(f"Unsupported document extension: {ext}")
    return parser(path)


# ── Parsers ───────────────────────────────────────────────────────────────────

def _parse_with_docling(path: Path) -> tuple[str, Optional[str]]:
    """Parse PDF or DOCX with Docling, exporting structured markdown."""

    converter = _get_converter()
    result = converter.convert(source=str(path))
    doc = result.document

    markdown = doc.export_to_markdown()
    title = _extract_title_from_markdown(markdown) or path.stem

    logger.debug(
        "Docling parsed %s — %d chars of markdown",
        path.name,
        len(markdown),
    )
    return markdown, title


def _parse_text(path: Path) -> tuple[str, Optional[str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return text, None


def _parse_yaml(path: Path) -> tuple[str, Optional[str]]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        data = yaml.safe_load(raw)
        text = _format_structured(data)
    except yaml.YAMLError as exc:
        logger.warning("YAML parse error in %s (%s), using raw text", path.name, exc)
        text = raw
    return text, None


def _parse_json(path: Path) -> tuple[str, Optional[str]]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        data = json.loads(raw)
        text = _format_structured(data)
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse error in %s (%s), using raw text", path.name, exc)
        text = raw
    return text, None


# ── Docling converter singleton ───────────────────────────────────────────────

_converter = None


def _get_converter():
    """Return a lazily-initialised DocumentConverter (module-level singleton)."""
    global _converter
    if _converter is not None:
        return _converter

    from config import settings

    if settings.docling_vlm_enabled:
        # VLM enrichment: Docling runs picture description via the configured
        # VLM model (default Qwen2.5-VL-7B-Instruct). Requires VRAM headroom
        # alongside Ollama's loaded models. See CLAUDE.md for VRAM notes.
        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import (
                PdfPipelineOptions,
                PictureDescriptionApiOptions,
            )

            pipeline_options = PdfPipelineOptions(
                do_picture_description=True,
                picture_description_options=PictureDescriptionApiOptions(
                    url=f"{settings.ollama_base_url_str}/v1/chat/completions",
                    model_id=settings.docling_vlm_model,
                ),
            )
            _converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            logger.info("Docling VLM enrichment enabled — model=%s", settings.docling_vlm_model)
        except Exception as exc:
            logger.warning(
                "Docling VLM setup failed (%s) — falling back to standard converter", exc
            )
            from docling.document_converter import DocumentConverter
            _converter = DocumentConverter()
    else:
        from docling.document_converter import DocumentConverter
        _converter = DocumentConverter()
        logger.debug("Docling initialised (VLM disabled)")

    return _converter


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_title_from_markdown(md: str) -> Optional[str]:
    """Return text of the first ATX heading (# Heading) if present."""
    for line in md.splitlines():
        stripped = line.lstrip("#").strip()
        if line.startswith("#") and stripped:
            return stripped
    return None


def _format_structured(data, indent: int = 0) -> str:
    """Recursively format a dict/list as readable indented key-value text."""
    pad = "  " * indent
    parts: list[str] = []

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                parts.append(f"{pad}{key}:")
                child = _format_structured(value, indent + 1)
                if child:
                    parts.append(child)
            else:
                parts.append(f"{pad}{key}: {value}")
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                child = _format_structured(item, indent)
                if child:
                    parts.append(child)
            else:
                parts.append(f"{pad}- {item}")
    elif data is not None:
        parts.append(f"{pad}{data}")

    return "\n".join(parts)
