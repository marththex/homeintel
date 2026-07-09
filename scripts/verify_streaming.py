"""
scripts/verify_streaming.py — Integration test for POST /chat/stream.

Prereq: test data in Qdrant (cd backend && python ../scripts/test_ingestion.py --keep-chunks)
Run:    cd backend && python ../scripts/verify_streaming.py
Exit 0 = pass. Reuses uvicorn lifecycle helpers from verify_api.py.
"""

import json
import sys
from pathlib import Path

import httpx

SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))
from verify_api import _start_uvicorn, _wait_for_ready, _stop_uvicorn, BASE_URL, STARTUP_TIMEOUT

REQUEST_TIMEOUT = 120


def _read_events(question: str, modality_filter=None):
    """POST /chat/stream and return a list of (event, data) tuples."""
    events = []
    payload = {"question": question, "modality_filter": modality_filter}
    with httpx.stream("POST", f"{BASE_URL}/chat/stream", json=payload, timeout=REQUEST_TIMEOUT) as r:
        r.raise_for_status()
        event = None
        data_lines = []
        for line in r.iter_lines():
            if line == "":  # frame boundary
                if event is not None:
                    raw = "\n".join(data_lines)
                    try:
                        events.append((event, json.loads(raw) if raw else None))
                    except json.JSONDecodeError:
                        events.append((event, None))
                event, data_lines = None, []
                continue
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].strip())
    return events


def check_stream_order() -> bool:
    print("  check_stream_order...", end="", flush=True)
    events = _read_events("What companies has Jane Doe worked for?")
    tags = [e for e, _ in events]
    ok = bool(tags) and tags[0] == "sources" and tags[-1] == "done" and "token" in tags
    answer = "".join(d["delta"] for e, d in events if e == "token" and d)
    ok = ok and bool(answer.strip())
    print(" PASS" if ok else f" FAIL (tags={tags[:3]}..{tags[-2:]})")
    return ok


def check_image_caption_length() -> bool:
    """Image query source caption should exceed 200 chars. SKIP if no image data."""
    print("  check_image_caption_length...", end="", flush=True)
    events = _read_events("people", modality_filter="image")
    src_evt = next((d for e, d in events if e == "sources"), None)
    sources = (src_evt or {}).get("sources", [])
    images = [s for s in sources if s.get("modality") == "image"]
    if not images:
        print(" SKIP (no image data indexed)")
        return True
    longest = max(len(s["excerpt"]) for s in images)
    ok = longest > 200
    print(" PASS" if ok else f" FAIL (longest caption {longest} chars)")
    return ok


def main() -> int:
    print("\n HomeIntel — streaming verification")
    print("-" * 45)
    print("Starting uvicorn...", end="", flush=True)
    proc = _start_uvicorn()
    if not _wait_for_ready(STARTUP_TIMEOUT):
        print(" FAILED")
        _stop_uvicorn(proc)
        return 1
    print(" ready\n")
    try:
        results = [check_stream_order(), check_image_caption_length()]
    finally:
        print("\nShutting down...", end="", flush=True)
        _stop_uvicorn(proc)
        print(" done")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
