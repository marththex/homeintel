"""
scripts/test_ingestion.py — Document ingestion smoke test.

Usage:
    cd backend
    python ../scripts/test_ingestion.py              # ingest and clean up
    python ../scripts/test_ingestion.py --keep-chunks # skip cleanup (for API testing)
"""

import argparse
import logging
import os
import re
import sys
from pathlib import Path

# Add backend/ to sys.path so imports resolve correctly
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(name)s — %(message)s",
)
# Quiet noisy third-party loggers
logging.getLogger("unstructured").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

from config import settings
from ingestion.pipeline import ingest_file
from vectorstore.chroma import VectorStore


_HEX_DIR = re.compile(r"^[0-9a-f]{40,}$")
_SKIP_FRAGMENTS = frozenset({"overlay2", "node_modules", "diff"})


def _find_compose_files(nas_root: Path) -> list[Path]:
    """
    Walk nas_root one level deep and return any docker-compose.yml/.yaml
    found directly inside each service subdirectory.

    Skips directories whose name contains overlay2 / node_modules / diff
    or that are 40+ character hex strings (Docker content-addressable storage).
    """
    found: list[Path] = []
    try:
        entries = sorted(nas_root.iterdir())
    except PermissionError:
        return found

    for entry in entries:
        if not entry.is_dir():
            continue
        name = entry.name
        if any(frag in name for frag in _SKIP_FRAGMENTS):
            continue
        if _HEX_DIR.match(name):
            continue
        for compose_name in ("docker-compose.yml", "docker-compose.yaml"):
            candidate = Path(os.path.normpath(entry / compose_name))
            if candidate.is_file():
                found.append(candidate)
                break  # one compose file per service folder is enough

    return found


def separator(label: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")


def ensure_clean_collection(vs: VectorStore) -> None:
    current = vs.count()
    if current > 0:
        print(f"  ⚠️  Collection unexpectedly contains {current} chunks; wiping stale data")
        vs.delete_all()
        print("  ✓ Stale chunks removed")


def main(keep_chunks: bool = False) -> None:
    ingested_paths: list[str] = []
    vs = VectorStore()
    ensure_clean_collection(vs)

    # ── 1. Ingest resume ──────────────────────────────────────────────────────
    separator("Ingesting Marcus_Resume.pdf")
    resume_path = str(settings.nas_watch_path / "Marcus_Resume.pdf")
    try:
        n = ingest_file(resume_path)
        print(f"  ✓ {resume_path}  →  {n} chunks")
        ingested_paths.append(resume_path)
    except Exception as exc:
        print(f"  ✗ Failed: {exc}")
        raise

    # ── 2. Find and ingest docker-compose files from the configured NAS root ─────
    separator("Ingesting docker-compose files from NAS root (one level deep)")
    nas_root = settings.nas_watch_path
    compose_files = _find_compose_files(nas_root)

    if not compose_files:
        print(f"  ✗ No docker-compose.yml/.yaml files found under {nas_root}")
    else:
        total = 0
        for compose in compose_files:
            try:
                n = ingest_file(str(compose))
                print(f"  ✓ {compose.parent.name}/{compose.name}  →  {n} chunks")
                ingested_paths.append(str(compose))
                total += n
            except Exception as exc:
                print(f"  ✗ {compose.parent.name}/{compose.name}: {exc}")
        print(f"\n  Total docker-compose chunks: {total}")

    # ── 3 & 4. Semantic queries ───────────────────────────────────────────────
    vs = VectorStore()

    queries = [
        "experience with Python",
        "which services use port 443",
    ]
    for query in queries:
        separator(f"Query: '{query}'")
        results = vs.query(query, top_k=3)
        if not results:
            print("  (no results)")
        for i, doc in enumerate(results, 1):
            meta = doc.metadata
            snippet = doc.page_content[:180].replace("\n", " ")
            print(f"  [{i}] {meta.get('file_name')}  chunk {meta.get('chunk_index')}")
            print(f"       {snippet!r}")

    # ── 5. Chunk counts by modality ───────────────────────────────────────────
    separator("Chunk count by modality")
    stats = vs.stats()
    for modality, count in stats["by_modality"].items():
        print(f"  {modality:12s}: {count}")
    print(f"  {'total':12s}: {stats['total_chunks']}")

    # ── 6. Cleanup — remove all chunks ingested during this test run ──────────
    if keep_chunks:
        separator("Cleanup skipped (--keep-chunks)")
        print(f"  Chunks left in collection: {vs.count()}")
        print("  Run scripts/cleanup_collection.py when done testing.")
    else:
        separator("Cleanup — removing test chunks")
        for path in ingested_paths:
            vs.delete_file(path)
            print(f"  deleted chunks for {Path(path).name}")
        after = vs.count()
        print(f"\n  Chunks remaining in collection: {after}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HomeIntel ingestion smoke test")
    parser.add_argument(
        "--keep-chunks",
        action="store_true",
        help="Skip end-of-test cleanup so chunks persist in ChromaDB (useful for API testing).",
    )
    args = parser.parse_args()
    main(keep_chunks=args.keep_chunks)
