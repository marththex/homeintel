"""
scripts/run_colpali.py — Batch ColPali visual indexer.

Scans a directory for PDFs and indexes each page with ColPali multi-vectors
into the {collection}_colpali Qdrant collection.

Usage (from repo root, homeintel conda env):
    cd backend
    python ../scripts/run_colpali.py                         # index all PDFs under NAS_WATCH_PATH
    python ../scripts/run_colpali.py --path /path/to/pdfs
    python ../scripts/run_colpali.py --path /path/to/document.pdf
    python ../scripts/run_colpali.py --force                 # re-index already-indexed files

Requirements:
    pip install colpali-engine>=0.3.0 pypdfium2>=4.0.0

Notes:
  - First run downloads ~8 GB of model weights (vidore/colpali-v1.2) to
    ~/.cache/huggingface/hub/
  - ~5s per page on a modern NVIDIA GPU; a 10-page document takes ~50s
  - GPU VRAM: ~8 GB for colpali-v1.2 in bfloat16 — do not run alongside
    qwen3.5:9b inference; stop Ollama first or schedule during off-hours
  - Storage: ~1 MB per page in Qdrant (128-dim × num_patches vectors)
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# ── Path setup ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from config import settings
from vectorstore.qdrant import VectorStore
from ingestion.processors.colpali import index_pdf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logging.getLogger("qdrant_client").setLevel(logging.WARNING)
logger = logging.getLogger("run_colpali")


def find_pdfs(root: Path, exclude: list[str]) -> list[Path]:
    pdfs: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune excluded dirs in-place so os.walk skips them.
        dp = Path(dirpath)
        dirnames[:] = [
            d for d in dirnames
            if not any(
                os.path.normpath(dp / d).lower().startswith(e)
                for e in [os.path.normpath(ex).lower() for ex in exclude]
            )
        ]
        for fn in filenames:
            if fn.lower().endswith(".pdf"):
                pdfs.append(dp / fn)
    return pdfs


def main() -> None:
    parser = argparse.ArgumentParser(description="ColPali batch PDF indexer")
    parser.add_argument(
        "--path",
        default=str(settings.nas_watch_path),
        help="File or directory to index (default: NAS_WATCH_PATH)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-index files even if they already have ColPali embeddings",
    )
    args = parser.parse_args()

    target = Path(os.path.normpath(args.path))
    vs = VectorStore()
    vs.ensure_colpali_collection()

    if target.is_file():
        if not target.suffix.lower() == ".pdf":
            logger.error("ColPali only supports PDF files, got: %s", target)
            sys.exit(1)
        pdfs = [target]
    else:
        exclude = settings.watcher_exclude_paths_list
        pdfs = find_pdfs(target, exclude)

    if not pdfs:
        logger.info("No PDFs found under %s", target)
        return

    logger.info("Found %d PDF(s) to index", len(pdfs))
    total_pages = 0
    errors = 0

    for i, pdf_path in enumerate(pdfs, 1):
        logger.info("[%d/%d] %s", i, len(pdfs), pdf_path)
        try:
            n = index_pdf(pdf_path, vs, force=args.force)
            total_pages += n
        except Exception:
            logger.exception("Failed to index %s", pdf_path)
            errors += 1

    logger.info(
        "Done — %d pages indexed across %d PDF(s), %d error(s)",
        total_pages, len(pdfs) - errors, errors,
    )


if __name__ == "__main__":
    main()
