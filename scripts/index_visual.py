"""
scripts/index_visual.py — Batch CLIP visual indexer for photos.

Walks a directory, embeds each image with CLIP (openai/clip-vit-large-patch14),
and upserts into the `homeintel_visual` Qdrant collection.

Usage:
    conda activate homeintel
    cd backend
    python ../scripts/index_visual.py
    python ../scripts/index_visual.py --path Z:/marcus_photoprism/originals
    python ../scripts/index_visual.py --path Z:/marcus_photoprism/originals --clear

Options:
    --path PATH     Directory to index (default: NAS_WATCH_PATH)
    --clear         Wipe the visual collection before indexing
    --ext EXT...    File extensions to include (default: .jpg .jpeg .png)

Notes:
    - One Qdrant point per photo (no chunking).
    - CLIP runs on GPU if available (~500 MB VRAM, safe alongside Ollama).
    - ~3000 photos takes 20-40 minutes on RTX 5080.
    - Idempotent: delete + upsert on each run.
    - Before running, temporarily remove marcus_photoprism from
      WATCHER_EXCLUDE_PATHS in .env (or use --path to target originals directly
      without the watcher restriction — the watcher is separate from this script).
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CLIP visual indexer")
    p.add_argument("--path", default=None, help="Root directory to index")
    p.add_argument("--clear", action="store_true", help="Wipe collection before indexing")
    p.add_argument(
        "--ext", nargs="+", default=[".jpg", ".jpeg", ".png"],
        help="File extensions to include",
    )
    return p.parse_args()


def collect_files(root: Path, extensions: list[str]) -> list[Path]:
    exts = {e.lower() for e in extensions}
    files = [f for f in root.rglob("*") if f.suffix.lower() in exts and f.is_file()]
    return sorted(files)


def main() -> None:
    args = parse_args()

    from config import settings
    from vectorstore.clip import CLIPVisualStore

    root = Path(args.path) if args.path else Path(settings.nas_watch_path)
    logger.info("Indexing photos under: %s", root)
    logger.info("Extensions: %s", args.ext)

    store = CLIPVisualStore()

    if args.clear:
        logger.warning("--clear: wiping homeintel_visual collection")
        store.delete_all()

    files = collect_files(root, args.ext)
    logger.info("Found %d image(s)", len(files))

    indexed = 0
    errors = 0
    start = time.monotonic()

    for i, fpath in enumerate(files, 1):
        try:
            img = Image.open(fpath).convert("RGB")
            store.upsert(str(fpath), img)
            indexed += 1

            if i % 50 == 0:
                elapsed = time.monotonic() - start
                rate = i / elapsed
                eta = (len(files) - i) / rate if rate > 0 else 0
                logger.info(
                    "Progress: %d/%d  ETA: %dm %ds",
                    i, len(files), int(eta // 60), int(eta % 60),
                )
        except PermissionError:
            logger.warning("SKIP (permission denied): %s", fpath)
        except Exception as exc:
            logger.error("ERROR %s: %s", fpath, exc)
            errors += 1

    elapsed = time.monotonic() - start
    logger.info(
        "Done in %dm %ds — Indexed: %d photo(s)  Errors: %d",
        int(elapsed // 60), int(elapsed % 60), indexed, errors,
    )
    logger.info("Collection now has %d CLIP vectors", store.count())


if __name__ == "__main__":
    main()
