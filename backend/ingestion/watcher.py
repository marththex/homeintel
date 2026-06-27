"""
ingestion/watcher.py — Watchdog-based file watcher for the NAS mount.

Watches NAS_WATCH_PATH recursively and keeps Qdrant in sync:
  - File created/modified  → delete old chunks + re-ingest
  - File deleted           → delete chunks from Qdrant

Excluded paths (WATCHER_EXCLUDE_PATHS) and unsupported extensions are
silently ignored.  SMB disconnections are handled with exponential backoff
restart of the observer.

Usage (standalone):
    cd backend
    python -m ingestion.watcher

Usage (from FastAPI lifespan):
    from ingestion.watcher import start_watcher, stop_watcher
    thread = start_watcher()
    ...
    stop_watcher(thread)
"""

import logging
import os
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from config import settings
from ingestion.pipeline import ingest_file
from vectorstore.qdrant import VectorStore

logger = logging.getLogger(__name__)

# Seconds to wait between observer restart attempts after SMB failures.
_RETRY_DELAYS = [5, 10, 30, 60, 120]


def _normalise(path: str) -> str:
    """Return a normalised, lower-cased absolute path string for comparisons."""
    return os.path.normpath(path).lower()


class _NASEventHandler(FileSystemEventHandler):
    """Translate watchdog events into Qdrant upsert / delete calls."""

    def __init__(self, vs: VectorStore, watch_root: Path,
                 exclude_roots: list[str], allowed_exts: set[str]) -> None:
        super().__init__()
        self._vs = vs
        self._watch_root = watch_root
        # Pre-normalise exclusions for fast prefix matching.
        self._exclude_roots = [_normalise(p) for p in exclude_roots]
        self._allowed_exts = {e.lower() for e in allowed_exts}

    # ── helpers ──────────────────────────────────────────────────────────────

    def _is_excluded(self, path: str) -> bool:
        norm = _normalise(path)
        return any(norm.startswith(ex) for ex in self._exclude_roots)

    def _is_supported(self, path: str) -> bool:
        return Path(path).suffix.lower() in self._allowed_exts

    def _should_handle(self, path: str) -> bool:
        return self._is_supported(path) and not self._is_excluded(path)

    # ── event handlers ───────────────────────────────────────────────────────

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory or not self._should_handle(event.src_path):
            return
        logger.info("Created: %s", event.src_path)
        self._ingest(event.src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory or not self._should_handle(event.src_path):
            return
        logger.info("Modified: %s", event.src_path)
        self._ingest(event.src_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        if event.is_directory or not self._should_handle(event.src_path):
            return
        logger.info("Deleted: %s", event.src_path)
        try:
            self._vs.delete_file(event.src_path)
        except Exception:
            logger.exception("Failed to delete chunks for %s", event.src_path)

    def on_moved(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        # Treat a move as delete-old + create-new.
        if self._should_handle(event.src_path):
            logger.info("Moved (src deleted): %s", event.src_path)
            try:
                self._vs.delete_file(event.src_path)
            except Exception:
                logger.exception("Failed to delete chunks for %s", event.src_path)
        if self._should_handle(event.dest_path):
            logger.info("Moved (dst created): %s", event.dest_path)
            self._ingest(event.dest_path)

    # ── ingestion ────────────────────────────────────────────────────────────

    def _ingest(self, path: str) -> None:
        try:
            n = ingest_file(path)
            logger.info("Ingested %d chunks ← %s", n, path)
        except ValueError:
            # Extension not handled by ingestion pipeline (image/audio — future steps).
            logger.debug("Skipped (no pipeline): %s", path)
        except FileNotFoundError:
            logger.warning("File vanished before ingestion: %s", path)
        except Exception:
            logger.exception("Ingestion failed for %s", path)


# ── public API ───────────────────────────────────────────────────────────────

def run_watcher() -> None:
    """
    Block indefinitely, restarting the watchdog Observer after any failure.

    Call this in a daemon thread (see start_watcher) so it doesn't prevent
    clean process exit.
    """
    watch_root = Path(os.path.normpath(settings.nas_watch_path))
    exclude_paths = settings.watcher_exclude_paths_list
    allowed_exts = set(settings.supported_extensions_list)

    logger.info(
        "Watcher starting — root=%s  excluded=%d paths  extensions=%s",
        watch_root, len(exclude_paths), sorted(allowed_exts),
    )

    vs = VectorStore()
    retry_index = 0

    while True:
        handler = _NASEventHandler(vs, watch_root, exclude_paths, allowed_exts)
        observer = Observer()
        observer.schedule(handler, str(watch_root), recursive=True)

        try:
            observer.start()
            logger.info("Watchdog observer started on %s", watch_root)
            retry_index = 0  # reset backoff on successful start

            while observer.is_alive():
                observer.join(timeout=5)

        except FileNotFoundError:
            logger.error(
                "Watch root not found: %s — is the NAS mount up?", watch_root
            )
        except PermissionError:
            logger.error("Permission denied accessing %s", watch_root)
        except OSError as exc:
            logger.error("Observer OSError (SMB drop?): %s", exc)
        except Exception:
            logger.exception("Unexpected observer error")
        finally:
            try:
                observer.stop()
                observer.join()
            except Exception:
                pass

        delay = _RETRY_DELAYS[min(retry_index, len(_RETRY_DELAYS) - 1)]
        retry_index += 1
        logger.warning("Observer stopped — restarting in %ds (attempt %d)", delay, retry_index)
        time.sleep(delay)


def start_watcher() -> threading.Thread:
    """Start run_watcher() in a background daemon thread and return it."""
    t = threading.Thread(target=run_watcher, name="nas-watcher", daemon=True)
    t.start()
    logger.info("NAS watcher thread started")
    return t


def stop_watcher(thread: threading.Thread) -> None:
    """
    Signal the watcher to stop.

    Because run_watcher() loops indefinitely, the cleanest shutdown is just
    letting the daemon thread die when the process exits.  This function is a
    no-op today but exists so callers can add graceful-stop logic later.
    """
    logger.info("NAS watcher stop requested (will exit with process)")


# ── standalone entry-point ───────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    )
    run_watcher()
