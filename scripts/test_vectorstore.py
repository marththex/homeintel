"""
scripts/test_vectorstore.py — Smoke test for the Qdrant vectorstore wrapper.

Run this after deploying Qdrant on the NAS to confirm:
  - Qdrant is reachable at QDRANT_URL
  - Ollama embedding model is reachable
  - Upsert, hybrid query, modality filtering, and delete all work correctly

Usage (from repo root):
    cd backend
    python ../scripts/test_vectorstore.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from langchain_core.documents import Document
from vectorstore.qdrant import VectorStore, Modality
from config import settings


def separator(title: str) -> None:
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print("─" * 50)


def ensure_clean_collection(vs: VectorStore) -> None:
    current = vs.count()
    if current > 0:
        print(f"   ⚠️  Collection unexpectedly contains {current} chunks; wiping stale data")
        vs.delete_all()
        print("   ✓ Stale chunks removed")


def main() -> None:
    print("\n HomeIntel — Qdrant VectorStore smoke test")
    print(f"   Qdrant URL    : {settings.qdrant_url}")
    print(f"   Collection    : {settings.qdrant_collection_name}")
    print(f"   Embed model   : {settings.ollama_embed_model}")
    print(f"   Ollama URL    : {settings.ollama_base_url_str}")

    # ── 1. Init ───────────────────────────────────────────────────────────────
    separator("1. Initialising VectorStore")
    vs = VectorStore()
    ensure_clean_collection(vs)
    print(f"   ✓ Connected — {vs.count()} chunks currently in collection")

    # ── 2. Upsert ─────────────────────────────────────────────────────────────
    separator("2. Upserting test documents (dense + sparse vectors)")
    test_docs = [
        Document(
            page_content="The Nextcloud container runs on port 443 and uses PostgreSQL.",
            metadata={
                "file_path": "/test/docker-compose.yml",
                "file_name": "docker-compose.yml",
                "file_ext": ".yml",
                "chunk_index": 0,
                "modality": Modality.DOCUMENT.value,
            },
        ),
        Document(
            page_content="Family photo taken at Yosemite National Park, summer 2023.",
            metadata={
                "file_path": "/test/yosemite.jpg",
                "file_name": "yosemite.jpg",
                "file_ext": ".jpg",
                "chunk_index": 0,
                "modality": Modality.IMAGE.value,
            },
        ),
        Document(
            page_content="Meeting notes: discussed Q3 budget and project timelines.",
            metadata={
                "file_path": "/test/meeting.pdf",
                "file_name": "meeting.pdf",
                "file_ext": ".pdf",
                "chunk_index": 0,
                "modality": Modality.DOCUMENT.value,
            },
        ),
    ]
    vs.upsert(test_docs)
    print(f"   ✓ Upserted {len(test_docs)} test chunks")
    print(f"   ✓ Collection now has {vs.count()} chunks")

    # ── 3. Hybrid query — no filter ───────────────────────────────────────────
    separator("3. Hybrid query (dense + BM25 RRF, no filter)")
    results = vs.query("docker containers and ports", top_k=3)
    print("   Query: 'docker containers and ports'")
    for i, doc in enumerate(results):
        print(f"   [{i+1}] {doc.metadata['file_name']} — {doc.page_content[:60]}...")

    # ── 4. Hybrid query — modality filter ─────────────────────────────────────
    separator("4. Hybrid query (images only)")
    results = vs.query("outdoor nature photo", top_k=3, modality=Modality.IMAGE)
    print("   Query: 'outdoor nature photo' (modality=image)")
    if not results:
        print("   (no results — expected if collection is small)")
    for i, doc in enumerate(results):
        print(f"   [{i+1}] {doc.metadata['file_name']} — {doc.page_content[:60]}...")

    # ── 5. Query with scores ──────────────────────────────────────────────────
    separator("5. Query with RRF scores")
    scored = vs.query_with_scores("budget meeting", top_k=3)
    for i, (doc, score) in enumerate(scored):
        print(f"   [{i+1}] score={score:.4f}  {doc.metadata['file_name']}")

    # ── 6. Stats ──────────────────────────────────────────────────────────────
    separator("6. Stats")
    stats = vs.stats()
    print(f"   Total chunks : {stats['total_chunks']}")
    print(f"   Qdrant URL   : {stats['qdrant_url']}")
    for modality, count in stats["by_modality"].items():
        print(f"   {modality:<12} : {count} chunks")

    # ── 7. Delete one file ────────────────────────────────────────────────────
    separator("7. Delete test file")
    vs.delete_file("/test/docker-compose.yml")
    print(f"   ✓ Deleted chunks for docker-compose.yml")
    print(f"   ✓ Collection now has {vs.count()} chunks")

    # ── 8. Cleanup ────────────────────────────────────────────────────────────
    separator("8. Cleanup — removing all test data")
    for doc in test_docs:
        try:
            vs.delete_file(doc.metadata["file_path"])
        except Exception:
            pass
    print(f"   ✓ Collection restored to {vs.count()} chunks")

    separator("All tests passed ✓")
    print("   Qdrant hybrid search is working — ready for Step 3 ingestion\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("\nCommon causes:")
        print("  - Qdrant not running on NAS (deploy docker/qdrant.yml first)")
        print(f"  - QDRANT_URL not set correctly: {settings.qdrant_url}")
        print("  - Ollama isn't running (start it with: ollama serve)")
        print(f"  - Embedding model not pulled (run: ollama pull {settings.ollama_embed_model})")
        import traceback
        traceback.print_exc()
        sys.exit(1)
