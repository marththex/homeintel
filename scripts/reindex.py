"""
scripts/reindex.py — Full NAS reindex.

Walks NAS_WATCH_PATH, respects WATCHER_EXCLUDE_PATHS and SUPPORTED_EXTENSIONS,
and ingests every file into Qdrant.  Safe to re-run: each file is deleted from
Qdrant before re-ingesting, so no duplicates accumulate.

Usage (from repo root, homeintel conda env):
    cd backend
    python ../scripts/reindex.py                          # full NAS index
    python ../scripts/reindex.py --path /path/to/folder
    python ../scripts/reindex.py --ext .pdf .md           # only specific extensions
    python ../scripts/reindex.py --dry-run                # list files without ingesting
    python ../scripts/reindex.py --workers 3              # parallel ingestion (faster)
    python ../scripts/reindex.py --skip-existing          # resume — skip already-indexed files

Speed:
    --workers N issues N files concurrently. The dominant per-image cost is the
    Ollama vision caption (network-bound), so threads keep the GPU busy. Pair
    with OLLAMA_NUM_PARALLEL=N on the Ollama service. Start with 3; drop to 2 if
    the vision model OOMs on a 16 GB GPU.

Resume:
    --skip-existing fetches every file_path already in Qdrant (one scroll pass)
    and skips them. Cancel an interrupted run with Ctrl+C and restart with
    --skip-existing to pick up where you left off.

Progress is printed live. On error the file is logged and skipped — the rest
of the index continues uninterrupted.
"""

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from config import settings
from ingestion.pipeline import ingest_file, get_vectorstore

logging.basicConfig(
    level=logging.WARNING,          # suppress library noise
    format="%(levelname)-8s %(name)s — %(message)s",
)
logging.getLogger("ingestion.pipeline").setLevel(logging.INFO)
logging.getLogger("vectorstore.qdrant").setLevel(logging.INFO)
logger = logging.getLogger("reindex")


def _is_excluded(path: Path, exclude_roots: list[str]) -> bool:
    norm = os.path.normpath(str(path)).lower()
    return any(norm.startswith(ex) for ex in exclude_roots)


def collect_files(
    root: Path,
    allowed_exts: set[str],
    exclude_roots: list[str],
) -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dp = Path(dirpath)
        # Prune excluded dirs in-place so os.walk doesn't descend into them.
        dirnames[:] = [
            d for d in dirnames
            if not _is_excluded(dp / d, exclude_roots)
        ]
        for fn in filenames:
            fp = dp / fn
            if fp.suffix.lower() in allowed_exts and not _is_excluded(fp, exclude_roots):
                files.append(fp)
    return files


def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"


def process_file(fp: Path, skip_set: set[str]) -> tuple[str, Path, int, float, str]:
    """
    Ingest one file. Returns (status, path, chunk_count, elapsed_seconds, message).
    status is one of: "ok", "skip", "error". Never raises — safe in a thread pool.
    """
    if str(fp) in skip_set:
        return ("skip", fp, 0, 0.0, "already indexed")
    try:
        t0 = time.monotonic()
        n = ingest_file(str(fp))
        elapsed = time.monotonic() - t0
        if n == 0:
            return ("skip", fp, 0, elapsed, "no text extracted")
        return ("ok", fp, n, elapsed, "")
    except ValueError as exc:
        return ("skip", fp, 0, 0.0, str(exc))
    except FileNotFoundError:
        return ("skip", fp, 0, 0.0, "file vanished")
    except PermissionError:
        return ("skip", fp, 0, 0.0, "permission denied")
    except Exception as exc:
        logger.exception("Ingestion failed for %s", fp)
        return ("error", fp, 0, 0.0, str(exc))


def main() -> None:
    parser = argparse.ArgumentParser(description="Full NAS reindex into Qdrant")
    parser.add_argument(
        "--path",
        default=str(settings.nas_watch_path),
        help="Root path to index (default: NAS_WATCH_PATH from .env)",
    )
    parser.add_argument(
        "--ext",
        nargs="+",
        metavar="EXT",
        help="Override extensions to index (e.g. --ext .pdf .md)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be indexed without ingesting them",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of files to ingest concurrently (default 1). "
             "Pair with OLLAMA_NUM_PARALLEL. Try 3.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files already present in Qdrant (resume an interrupted run)",
    )
    args = parser.parse_args()

    root = Path(os.path.normpath(args.path))
    if not root.exists():
        print(f"ERROR: path does not exist: {root}")
        sys.exit(1)

    workers = max(1, args.workers)
    allowed_exts = set(args.ext) if args.ext else set(settings.supported_extensions_list)
    exclude_roots = [
        os.path.normpath(p).lower() for p in settings.watcher_exclude_paths_list
    ]

    print(f"\n  HomeIntel — Full Reindex")
    print(f"  {'─' * 50}")
    print(f"  Root        : {root}")
    print(f"  Extensions  : {', '.join(sorted(allowed_exts))}")
    print(f"  Excluded    : {len(exclude_roots)} path(s)")
    print(f"  Workers     : {workers}")
    print(f"  Skip existing: {args.skip_existing}")
    print(f"  Dry run     : {args.dry_run}")
    print(f"  {'─' * 50}\n")

    print("  Scanning...", end="", flush=True)
    files = collect_files(root, allowed_exts, exclude_roots)
    print(f" {len(files)} file(s) found\n")

    if not files:
        print("  Nothing to index.")
        return

    if args.dry_run:
        for f in files:
            print(f"  {f}")
        print(f"\n  {len(files)} file(s) would be indexed.")
        return

    # Warm the VectorStore singleton before spawning threads (avoids lazy-init race).
    vs = get_vectorstore()

    skip_set: set[str] = set()
    if args.skip_existing:
        print("  Loading already-indexed file list...", end="", flush=True)
        skip_set = vs.indexed_file_paths()
        print(f" {len(skip_set)} indexed\n")

    total = len(files)
    done = 0
    skipped = 0
    errors = 0
    total_chunks = 0
    t_start = time.monotonic()

    def report(i: int, result: tuple[str, Path, int, float, str]) -> None:
        nonlocal done, skipped, errors, total_chunks
        status, fp, n, elapsed, msg = result
        rel = fp.relative_to(root) if fp.is_relative_to(root) else fp
        prefix = f"  [{i:>{len(str(total))}}/{total}]"

        if status == "ok":
            total_chunks += n
            done += 1
            print(f"{prefix} OK    {rel}  → {n} chunk(s)  ({elapsed:.1f}s)")
        elif status == "skip":
            skipped += 1
            print(f"{prefix} SKIP  {rel}  ({msg})")
        else:
            errors += 1
            print(f"{prefix} ERROR {rel}  — {msg}")

        if i % 10 == 0:
            elapsed_total = time.monotonic() - t_start
            rate = i / elapsed_total
            remaining = (total - i) / rate if rate > 0 else 0
            print(f"\n  Progress: {i}/{total}  ETA: {fmt_duration(remaining)}\n")

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(process_file, fp, skip_set): fp for fp in files}
            for i, fut in enumerate(as_completed(futures), 1):
                report(i, fut.result())
    else:
        for i, fp in enumerate(files, 1):
            report(i, process_file(fp, skip_set))

    wall = time.monotonic() - t_start
    print(f"\n  {'─' * 50}")
    print(f"  Done in {fmt_duration(wall)}")
    print(f"  Indexed  : {done} file(s) → {total_chunks} chunks")
    print(f"  Skipped  : {skipped} file(s)")
    print(f"  Errors   : {errors} file(s)")
    print()

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
