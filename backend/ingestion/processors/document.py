"""
ingestion/processors/document.py — Parse documents into raw text.

Supported extensions:
  .pdf  .docx  → Unstructured (with pypdf fallback for PDF)
  .txt  .md    → plain file read
  .yml  .yaml  → PyYAML → formatted key-value text
  .json        → json → formatted key-value text

Returns (text, title) where title may be None.
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
        ".pdf":  _parse_pdf,
        ".docx": _parse_docx,
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

def _parse_pdf(path: Path) -> tuple[str, Optional[str]]:
    try:
        from unstructured.partition.pdf import partition_pdf
        elements = partition_pdf(filename=str(path), strategy="fast")
        text = "\n\n".join(str(e) for e in elements if str(e).strip())
        title = _first_title(elements)
        logger.debug("PDF parsed via Unstructured: %s (%d elements)", path.name, len(elements))
        return text, title
    except Exception as exc:
        logger.warning("Unstructured PDF failed for %s (%s), falling back to pypdf", path.name, exc)
        return _parse_pdf_fallback(path)


def _parse_pdf_fallback(path: Path) -> tuple[str, Optional[str]]:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(p for p in pages if p.strip())
    return text, None


def _parse_docx(path: Path) -> tuple[str, Optional[str]]:
    from unstructured.partition.docx import partition_docx
    elements = partition_docx(filename=str(path))
    text = "\n\n".join(str(e) for e in elements if str(e).strip())
    title = _first_title(elements)
    logger.debug("DOCX parsed via Unstructured: %s (%d elements)", path.name, len(elements))
    return text, title


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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _first_title(elements) -> Optional[str]:
    """Extract the first Title element from an Unstructured element list."""
    for el in elements:
        if getattr(el, "category", None) == "Title" and str(el).strip():
            return str(el).strip()
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
