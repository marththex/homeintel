"""
scripts/index_visual.py — Batch CLIP visual indexer for photos.

Walks a directory, embeds each image with CLIP (openai/clip-vit-large-patch14)
in batches, and upserts into the `homeintel_visual` Qdrant collection.

Usage:
    conda activate homeintel
    cd backend
    python ../scripts/index_visual.py
    python ../scripts/index_visual.py --path /path/to/photos
    python ../scripts/index_visual.py --path /path/to/photos --clear
    python ../scripts/index_visual.py --skip-existing          # resume
    python ../scripts/index_visual.py --batch-size 32          # bigger GPU batches

Options:
    --path PATH        Directory to index (default: NAS_WATCH_PATH)
    --clear            Wipe the visual collection before indexing
    --ext EXT...       File extensions to include (default: .jpg .jpeg .png)
    --batch-size N     Images per GPU forward pass (default 16). Raise if VRAM allows.
    --read-workers N   Parallel SMB image reads (default 4) — overlaps I/O with GPU.
    --skip-existing    Skip photos already in homeintel_visual (resume an interrupted run)

Notes:
    - One Qdrant point per photo (no chunking).
    - Batching the CLIP forward pass is ~5-8x faster than one image at a time.
    - Parallel reads hide SMB latency while the GPU embeds the previous batch.
    - CLIP runs on GPU if available (~500 MB VRAM, safe alongside Ollama).
    - Idempotent: re-running overwrites by stable point id.
    - Before running, temporarily remove that folder from
      WATCHER_EXCLUDE_PATHS in .env (or use --path to target the photos directly).
"""

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
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
    p.add_argument("--batch-size", type=int, default=16, help="Images per GPU batch")
    p.add_argument("--read-workers", type=int, default=4, help="Parallel image reads")
    p.add_argument("--skip-existing", action="store_true", help="Resume — skip indexed photos")
    return p.parse_args()


def collect_files(root: Path, extensions: list[str]) -> list[Path]:
    exts = {e.lower() for e in extensions}
    files = [f for f in root.rglob("*") if f.suffix.lower() in exts and f.is_file()]
    return sorted(files)


def load_image(fp: Path):
    """Read + decode one image. Returns (fp, Image | None | Exception)."""
    try:
        return fp, Image.open(fp).convert("RGB")
    except PermissionError:
        return fp, None
    except Exception as exc:  # noqa: BLE001
        return fp, exc


def main() -> None:
    args = parse_args()

    from config import settings
    from vectorstore.clip import CLIPVisualStore

    root = Path(args.path) if args.path else Path(settings.nas_watch_path)
    logger.info("Indexing photos under: %s", root)
    logger.info("Extensions: %s  batch-size: %d  read-workers: %d",
                args.ext, args.batch_size, args.read_workers)

    store = CLIPVisualStore()

    if args.clear:
        logger.warning("--clear: wiping homeintel_visual collection")
        store.delete_all()

    files = collect_files(root, args.ext)
    logger.info("Found %d image(s)", len(files))

    if args.skip_existing:
        logger.info("Loading already-indexed photo list...")
        existing = store.indexed_file_paths()
        before = len(files)
        files = [f for f in files if str(f) not in existing]
        logger.info("Skipping %d already-indexed; %d remaining", before - len(files), len(files))

    if not files:
        logger.info("Nothing to index.")
        return

    indexed = 0
    skipped = 0
    errors = 0
    start = time.monotonic()
    total = len(files)

    batch: list[tuple[str, Image.Image]] = []

    def flush() -> None:
        nonlocal indexed
        if not batch:
            return
        store.upsert_batch(batch)
        indexed += len(batch)
        batch.clear()

    # Parallel reads overlap SMB I/O with GPU embedding of the previous batch.
    with ThreadPoolExecutor(max_workers=args.read_workers) as ex:
        for n, (fpath, result) in enumerate(ex.map(load_image, files), 1):
            if isinstance(result, Image.Image):
                batch.append((str(fpath), result))
                if len(batch) >= args.batch_size:
                    flush()
            elif result is None:
                skipped += 1
                logger.warning("SKIP (permission denied): %s", fpath)
            else:
                errors += 1
                logger.error("ERROR %s: %s", fpath, result)

            if n % 100 == 0:
                elapsed = time.monotonic() - start
                rate = n / elapsed
                eta = (total - n) / rate if rate > 0 else 0
                logger.info("Progress: %d/%d  (%.1f img/s)  ETA: %dm %ds",
                            n, total, rate, int(eta // 60), int(eta % 60))

        flush()

    elapsed = time.monotonic() - start
    logger.info(
        "Done in %dm %ds — Indexed: %d photo(s)  Skipped: %d  Errors: %d",
        int(elapsed // 60), int(elapsed % 60), indexed, skipped, errors,
    )
    logger.info("Collection now has %d CLIP vectors", store.count())


if __name__ == "__main__":
    main()
