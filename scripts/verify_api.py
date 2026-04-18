"""
scripts/verify_api.py — Automated integration tests for the HomeIntel FastAPI backend.

Prerequisites:
    1. Populate ChromaDB with test data:
           cd backend
           python ../scripts/test_ingestion.py --keep-chunks
    2. Run this script from the same directory:
           python ../scripts/verify_api.py

The script starts uvicorn, runs all checks, then shuts it down cleanly.
Exit code 0 = all pass, 1 = one or more failures.
"""

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

BACKEND_DIR = Path(__file__).parent.parent / "backend"
BASE_URL = "http://localhost:8000"
PORT = 8000
STARTUP_TIMEOUT = 90    # seconds to wait for uvicorn to become ready
REQUEST_TIMEOUT = 120   # seconds for LLM inference requests


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class Result:
    name: str
    passed: bool
    detail: str = ""


# ── Uvicorn lifecycle ─────────────────────────────────────────────────────────

def _start_uvicorn() -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", str(PORT)],
        cwd=str(BACKEND_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _wait_for_ready(timeout: int) -> bool:
    """Poll GET /health until we get HTTP 200 or the timeout expires."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = httpx.get(f"{BASE_URL}/health", timeout=3.0)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _stop_uvicorn(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


# ── Individual checks ─────────────────────────────────────────────────────────

def _run(fn) -> Result:
    """Wrap a check function and catch unexpected exceptions."""
    try:
        return fn()
    except Exception as exc:
        return Result(name=fn.__name__, passed=False, detail=f"exception: {exc}")


def check_health() -> Result:
    name = "GET /health → {status:ok, ollama:true, chromadb:true}"
    resp = httpx.get(f"{BASE_URL}/health", timeout=10)
    body = resp.json()
    expected = {"status": "ok", "ollama": True, "chromadb": True}
    if resp.status_code == 200 and body == expected:
        return Result(name, passed=True)
    return Result(name, passed=False, detail=f"got HTTP {resp.status_code}: {body}")


def check_stats() -> Result:
    name = "GET /stats → total_chunks > 0"
    resp = httpx.get(f"{BASE_URL}/stats", timeout=10)
    body = resp.json()
    total = body.get("total_chunks", 0)
    if resp.status_code == 200 and total > 0:
        return Result(name, passed=True, detail=f"total_chunks={total}")
    return Result(name, passed=False, detail=f"total_chunks={total} (run test_ingestion.py --keep-chunks first)")


def check_chat_resume() -> Result:
    name = "POST /chat (Marcus resume) → non-empty answer + resume source"
    payload = {"question": "What companies has Marcus worked for?"}
    resp = httpx.post(f"{BASE_URL}/chat", json=payload, timeout=REQUEST_TIMEOUT)
    body = resp.json()
    answer = body.get("answer", "")
    sources = body.get("sources", [])
    has_resume = any("Marcus_Resume" in s.get("file_name", "") for s in sources)
    if answer and has_resume:
        return Result(name, passed=True)
    return Result(
        name, passed=False,
        detail=f"answer={answer[:80]!r}, has_resume_source={has_resume}, sources={[s['file_name'] for s in sources]}",
    )


def check_chat_docker() -> Result:
    name = "POST /chat (photoprism port) → non-empty answer + compose source"
    payload = {"question": "What port does photoprism run on?"}
    resp = httpx.post(f"{BASE_URL}/chat", json=payload, timeout=REQUEST_TIMEOUT)
    body = resp.json()
    answer = body.get("answer", "")
    sources = body.get("sources", [])
    has_docker = any("docker-compose" in s.get("file_name", "") for s in sources)
    if answer and has_docker:
        return Result(name, passed=True)
    return Result(
        name, passed=False,
        detail=f"answer={answer[:80]!r}, has_compose_source={has_docker}, sources={[s['file_name'] for s in sources]}",
    )


def check_chat_not_in_data() -> Result:
    name = "POST /chat (weather — not in data) → 'don't have information' response"
    payload = {"question": "What is the weather like today?"}
    resp = httpx.post(f"{BASE_URL}/chat", json=payload, timeout=REQUEST_TIMEOUT)
    body = resp.json()
    answer = body.get("answer", "").lower()
    # Accept any phrasing that signals the LLM doesn't have the answer
    indicators = (
        "don't have information",
        "do not have information",
        "not in your files",
        "no information",
        "cannot find",
        "not available",
        "not found",
    )
    if any(phrase in answer for phrase in indicators):
        return Result(name, passed=True)
    return Result(name, passed=False, detail=f"answer={body.get('answer', '')[:150]!r}")


def check_chat_modality_filter() -> Result:
    name = "POST /chat (modality_filter=document) → all sources are documents"
    payload = {"question": "work experience", "modality_filter": "document"}
    resp = httpx.post(f"{BASE_URL}/chat", json=payload, timeout=REQUEST_TIMEOUT)
    body = resp.json()
    sources = body.get("sources", [])
    if not sources:
        return Result(name, passed=False, detail="no sources returned")
    non_doc = [s["file_name"] for s in sources if s.get("modality") != "document"]
    if not non_doc:
        return Result(name, passed=True, detail=f"{len(sources)} sources, all modality=document")
    return Result(name, passed=False, detail=f"non-document sources returned: {non_doc}")


def check_empty_question() -> Result:
    name = "POST /chat (empty question) → HTTP 422"
    resp = httpx.post(f"{BASE_URL}/chat", json={"question": ""}, timeout=10)
    if resp.status_code == 422:
        return Result(name, passed=True)
    return Result(name, passed=False, detail=f"expected 422, got HTTP {resp.status_code}")


def check_no_body() -> Result:
    name = "POST /chat (no body) → HTTP 422"
    resp = httpx.post(
        f"{BASE_URL}/chat",
        headers={"Content-Type": "application/json"},
        timeout=10,
    )
    if resp.status_code == 422:
        return Result(name, passed=True)
    return Result(name, passed=False, detail=f"expected 422, got HTTP {resp.status_code}")


CHECKS = [
    check_health,
    check_stats,
    check_chat_resume,
    check_chat_docker,
    check_chat_not_in_data,
    check_chat_modality_filter,
    check_empty_question,
    check_no_body,
]


# ── Runner ────────────────────────────────────────────────────────────────────

def main() -> int:
    print("\n HomeIntel — API verification suite")
    print("─" * 55)

    # Start server
    print(f"\nStarting uvicorn on port {PORT}...", end="", flush=True)
    proc = _start_uvicorn()

    ready = _wait_for_ready(STARTUP_TIMEOUT)
    if not ready:
        print(" FAILED")
        stdout, _ = proc.communicate(timeout=3)
        print(f"\nServer output:\n{stdout}")
        _stop_uvicorn(proc)
        return 1
    print(" ready\n")

    # Run checks
    results: list[Result] = []
    try:
        for fn in CHECKS:
            print(f"  Running: {fn.__name__}...", end="", flush=True)
            r = _run(fn)
            icon = "PASS" if r.passed else "FAIL"
            print(f" [{icon}]")
            if r.detail:
                print(f"           {r.detail}")
            results.append(r)
    finally:
        print("\nShutting down uvicorn...", end="", flush=True)
        _stop_uvicorn(proc)
        print(" done")

    # Summary
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"\n{'─' * 55}")
    print(f"  PASSED: {passed}/{total}")

    failures = [r for r in results if not r.passed]
    if failures:
        print("  FAILED:")
        for r in failures:
            print(f"    ✗ {r.name}")
            if r.detail:
                print(f"      {r.detail}")

    print()
    print("  NOTE: Degraded-state test (Ollama down) must be run manually:")
    print("        Stop Ollama, then verify GET /health returns {\"status\": \"degraded\"}.")
    print()

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
