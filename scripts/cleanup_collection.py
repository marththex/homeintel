"""
scripts/cleanup_collection.py — One-time safe collection cleanup.

Run from the backend/ directory:
    cd backend
    python ../scripts/cleanup_collection.py
"""

import os
import sys
from pathlib import Path

# Allow imports from backend/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from config import settings
from vectorstore.chroma import VectorStore


def confirm(prompt: str) -> bool:
    answer = input(prompt).strip()
    return answer.upper() == "DELETE"


def main() -> None:
    vs = VectorStore()
    total = vs.count()

    print("HomeIntel collection cleanup")
    print(f"Collection name: {settings.chroma_collection_name}")
    print(f"Chunk count   : {total}")

    if total == 0:
        print("Nothing to delete. Collection is already empty.")
        return

    print("\nThis will permanently delete all chunks from the configured ChromaDB collection.")
    print("Type DELETE to confirm, or press Enter to abort.")
    if not confirm("Confirm delete: "):
        print("Aborted. No data was deleted.")
        return

    vs.delete_all()
    print(f"Deleted all chunks from collection '{settings.chroma_collection_name}'.")


if __name__ == "__main__":
    main()
