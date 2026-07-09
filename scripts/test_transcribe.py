"""
scripts/test_transcribe.py — Unit test for _ext_for_content_type (no model load).

Run from the backend directory:
    cd backend
    python ../scripts/test_transcribe.py
Exit 0 = all pass.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from api.transcribe import _ext_for_content_type


def test_ios_mp4():
    assert _ext_for_content_type("audio/mp4") == ".m4a"

def test_chrome_webm_with_codecs():
    assert _ext_for_content_type("audio/webm;codecs=opus") == ".webm"

def test_wav():
    assert _ext_for_content_type("audio/wav") == ".wav"

def test_unknown_falls_back():
    assert _ext_for_content_type("application/octet-stream") == ".bin"

def test_none_falls_back():
    assert _ext_for_content_type(None) == ".bin"


if __name__ == "__main__":
    test_ios_mp4()
    test_chrome_webm_with_codecs()
    test_wav()
    test_unknown_falls_back()
    test_none_falls_back()
    print("ALL PASS")
