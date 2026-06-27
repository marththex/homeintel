"""
scripts/reindex.py — Full NAS reindex.

Walks NAS_WATCH_PATH, respects WATCHER_EXCLUDE_PATHS and SUPPORTED_EXTENSIONS,
and ingests every file into Qdrant.  Safe to re-run: each file is deleted from
Qdrant before re-ingesting, so no duplicates accumulate.

Usage (from repo root, homeintel conda env):
    cd backend
    python ../scripts/reindex.py                      # full NAS index
    python ../scripts/reindex.py --path Z:/homeassistant
    python ../scripts/reindex.py --ext .pdf .md       # only specific extensions
    python ../scripts/reindex.py --dry-run            # list files without ingesting

Progress is printed live. On error the file is logged and skipped — the rest
of the index continues uninterrupted.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from config import settings
from ingestion.pipeline import ingest_file

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
    args = parser.parse_args()

    root = Path(os.path.normpath(args.path))
    if not root.exists():
        print(f"ERROR: path does not exist: {root}")
        sys.exit(1)

    allowed_exts = set(args.ext) if args.ext else set(settings.supported_extensions_list)
    exclude_roots = [
        os.path.normpath(p).lower() for p in settings.watcher_exclude_paths_list
    ]

    print(f"\n  HomeIntel — Full Reindex")
    print(f"  {'─' * 50}")
    print(f"  Root        : {root}")
    print(f"  Extensions  : {', '.join(sorted(allowed_exts))}")
    print(f"  Excluded    : {len(exclude_roots)} path(s)")
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

    total = len(files)
    done = 0
    skipped = 0
    errors = 0
    total_chunks = 0
    t_start = time.monotonic()

    for i, fp in enumerate(files, 1):
        rel = fp.relative_to(root) if fp.is_relative_to(root) else fp
        prefix = f"  [{i:>{len(str(total))}}/{total}]"

        try:
            t0 = time.monotonic()
            n = ingest_file(str(fp))
            elapsed = time.monotonic() - t0

            if n == 0:
                skipped += 1
                print(f"{prefix} SKIP  {rel}  (no text extracted)")
            else:
                total_chunks += n
                done += 1
                print(f"{prefix} OK    {rel}  → {n} chunk(s)  ({elapsed:.1f}s)")

        except ValueError as exc:
            # Unsupported extension or vision model not configured — expected skip
            skipped += 1
            print(f"{prefix} SKIP  {rel}  ({exc})")
        except FileNotFoundError:
            skipped += 1
            print(f"{prefix} SKIP  {rel}  (file vanished)")
        except Exception as exc:
            errors += 1
            print(f"{prefix} ERROR {rel}  — {exc}")
            logger.exception("Ingestion failed for %s", fp)

        # Rolling ETA every 10 files
        if i % 10 == 0:
            elapsed_total = time.monotonic() - t_start
            rate = i / elapsed_total
            remaining = (total - i) / rate if rate > 0 else 0
            print(f"\n  Progress: {i}/{total}  ETA: {fmt_duration(remaining)}\n")

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
