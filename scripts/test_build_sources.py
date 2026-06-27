"""
scripts/test_build_sources.py — Unit test for build_sources (no Qdrant/Ollama).

Run from the backend directory:
    cd backend
    python ../scripts/test_build_sources.py
Exit 0 = all pass.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from langchain_core.documents import Document
from api.chat import build_sources


def test_image_uses_full_caption_not_truncated():
    long_caption = "A detailed description. " * 30  # ~720 chars
    doc = Document(
        page_content="short matched chunk",
        metadata={"file_name": "IMG.JPG", "file_path": "/x/IMG.JPG", "modality": "image"},
    )
    sources = build_sources([doc], {"/x/IMG.JPG": long_caption})
    assert len(sources[0].excerpt) > 200, len(sources[0].excerpt)
    assert sources[0].excerpt == long_caption


def test_image_without_caption_falls_back_to_page_content():
    doc = Document(
        page_content="fallback text",
        metadata={"file_name": "IMG.JPG", "file_path": "/x/IMG.JPG", "modality": "image"},
    )
    sources = build_sources([doc], {})
    assert sources[0].excerpt == "fallback text"


def test_document_truncated_to_200():
    doc = Document(
        page_content="x" * 500,
        metadata={"file_name": "a.md", "file_path": "/a.md", "modality": "document"},
    )
    sources = build_sources([doc], {})
    assert len(sources[0].excerpt) == 200, len(sources[0].excerpt)


if __name__ == "__main__":
    test_image_uses_full_caption_not_truncated()
    test_image_without_caption_falls_back_to_page_content()
    test_document_truncated_to_200()
    print("ALL PASS")
