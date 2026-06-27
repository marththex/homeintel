"""
scripts/test_caption_join.py — Unit test for _join_caption_chunks (no Qdrant).

Run from the backend directory:
    cd backend
    python ../scripts/test_caption_join.py
Exit 0 = all pass.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from vectorstore.qdrant import _join_caption_chunks


def test_strips_overlap():
    a = "The quick brown fox jumps over the lazy"
    b = "over the lazy dog and runs away quickly"
    out = _join_caption_chunks([a, b], overlap=64)
    assert out == "The quick brown fox jumps over the lazy dog and runs away quickly", out


def test_single_chunk():
    assert _join_caption_chunks(["only one"], overlap=64) == "only one"


def test_empty():
    assert _join_caption_chunks([], overlap=64) == ""


def test_no_overlap_concatenates():
    assert _join_caption_chunks(["abc", "def"], overlap=64) == "abcdef"


if __name__ == "__main__":
    test_strips_overlap()
    test_single_chunk()
    test_empty()
    test_no_overlap_concatenates()
    print("ALL PASS")
