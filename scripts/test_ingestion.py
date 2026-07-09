"""
scripts/test_ingestion.py — Document ingestion smoke test.

Ingests the synthetic files bundled in data/sample-docs/ (a fictional resume,
a docker-compose-style file, a plain-text notes file, and a JSON config) so
this test works out of the box for anyone who clones the repo — no NAS or
personal files required.

Usage:
    cd backend
    python ../scripts/test_ingestion.py              # ingest and clean up
    python ../scripts/test_ingestion.py --keep-chunks # skip cleanup (for API testing)
"""

import argparse
import logging
import sys
from pathlib import Path

# Add backend/ to sys.path so imports resolve correctly
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(name)s — %(message)s",
)
# Quiet noisy third-party loggers
logging.getLogger("docling").setLevel(logging.WARNING)
logging.getLogger("qdrant_client").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("fastembed").setLevel(logging.WARNING)

from ingestion.pipeline import ingest_file
from vectorstore.qdrant import VectorStore


SAMPLE_DOCS_DIR = Path(__file__).parent.parent / "data" / "sample-docs"


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

    # ── 1. Ingest every sample doc ────────────────────────────────────────────
    separator(f"Ingesting sample docs from {SAMPLE_DOCS_DIR}")
    if not SAMPLE_DOCS_DIR.is_dir():
        print(f"  ✗ Sample docs directory not found: {SAMPLE_DOCS_DIR}")
        raise FileNotFoundError(f"Sample docs directory not found: {SAMPLE_DOCS_DIR}")

    sample_files = sorted(p for p in SAMPLE_DOCS_DIR.iterdir() if p.is_file())
    if not sample_files:
        print(f"  ✗ No files found in {SAMPLE_DOCS_DIR}")

    total = 0
    for path in sample_files:
        try:
            n = ingest_file(str(path))
            print(f"  ✓ {path.name}  →  {n} chunks")
            ingested_paths.append(str(path))
            total += n
        except Exception as exc:
            print(f"  ✗ {path.name}: {exc}")
            raise
    print(f"\n  Total sample-doc chunks: {total}")

    # ── 2. Semantic queries ────────────────────────────────────────────────────
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

    # ── 3. Chunk counts by modality ────────────────────────────────────────────
    separator("Chunk count by modality")
    stats = vs.stats()
    for modality in ("document", "image", "audio"):
        print(f"  {modality:12s}: {stats.get(modality, 0)}")
    print(f"  {'visual':12s}: {stats.get('visual', 0)}")
    print(f"  {'total':12s}: {stats.get('total', 0)}")

    # ── 4. Cleanup — remove all chunks ingested during this test run ──────────
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
        help="Skip end-of-test cleanup so chunks persist in Qdrant (useful for API testing).",
    )
    args = parser.parse_args()
    main(keep_chunks=args.keep_chunks)
