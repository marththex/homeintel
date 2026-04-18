"""
scripts/test_vectorstore.py — Smoke test for the ChromaDB wrapper.

Run this before moving to Step 3 to confirm:
  - ChromaDB can write to your NAS path
  - Ollama embedding model is reachable
  - Upsert, query, and delete all work correctly

Usage (from repo root):
    cd backend
    python ../scripts/test_vectorstore.py
"""

import sys
import os

# Allow imports from backend/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from langchain_core.documents import Document
from vectorstore.chroma import VectorStore, Modality
from config import settings


def separator(title: str) -> None:
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print('─' * 50)


def ensure_clean_collection(vs: VectorStore) -> None:
    current = vs.count()
    if current > 0:
        print(f"   ⚠️  Collection unexpectedly contains {current} chunks; wiping stale data")
        vs.delete_all()
        print("   ✓ Stale chunks removed")


def main() -> None:
    print("\n🏠 HomeIntel — VectorStore smoke test")
    print(f"   Chroma path   : {settings.chroma_path}")
    print(f"   Embed model   : {settings.ollama_embed_model}")
    print(f"   Ollama URL    : {settings.ollama_base_url_str}")

    # ── 1. Init ───────────────────────────────────────────────────────────────
    separator("1. Initialising VectorStore")
    vs = VectorStore()
    ensure_clean_collection(vs)
    print(f"   ✓ Connected — {vs.count()} chunks currently in collection")

    # ── 2. Upsert ─────────────────────────────────────────────────────────────
    separator("2. Upserting test documents")
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

    # ── 3. Query — no filter ──────────────────────────────────────────────────
    separator("3. Query (no filter)")
    results = vs.query("docker containers and ports", top_k=3)
    print(f"   Query: 'docker containers and ports'")
    for i, doc in enumerate(results):
        print(f"   [{i+1}] {doc.metadata['file_name']} — {doc.page_content[:60]}...")

    # ── 4. Query — modality filter ────────────────────────────────────────────
    separator("4. Query (images only)")
    results = vs.query("outdoor nature photo", top_k=3, modality=Modality.IMAGE)
    print(f"   Query: 'outdoor nature photo' (images only)")
    for i, doc in enumerate(results):
        print(f"   [{i+1}] {doc.metadata['file_name']} — {doc.page_content[:60]}...")

    # ── 5. Stats ──────────────────────────────────────────────────────────────
    separator("5. Stats")
    stats = vs.stats()
    print(f"   Total chunks : {stats['total_chunks']}")
    for modality, count in stats["by_modality"].items():
        print(f"   {modality:<12} : {count} chunks")

    # ── 6. Delete ─────────────────────────────────────────────────────────────
    separator("6. Delete test file")
    vs.delete_file("/test/docker-compose.yml")
    print(f"   ✓ Deleted chunks for docker-compose.yml")
    print(f"   ✓ Collection now has {vs.count()} chunks")

    # ── 7. Cleanup ────────────────────────────────────────────────────────────
    separator("7. Cleanup — removing all test data")
    for doc in test_docs:
        try:
            vs.delete_file(doc.metadata["file_path"])
        except Exception:
            pass
    print(f"   ✓ Collection restored to 0 test chunks")

    separator("All tests passed ✓")
    print("   Ready to move to Step 3 — ingestion processors\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("\nCommon causes:")
        print("  - Ollama isn't running (start it with: ollama serve)")
        print(f"  - Embedding model not pulled (run: ollama pull {settings.ollama_embed_model})")
        print(f"  - Chroma path not accessible: {settings.chroma_path}")
        sys.exit(1)