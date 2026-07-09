#!/usr/bin/env bash
# HomeIntel run script (Linux / macOS / Git-Bash on Windows).
#
# Starts the backend (uvicorn), frontend (vite), or both using the .venv
# created by ./setup.sh.
#
# Usage:
#   ./run.sh            # start both (default) — backend in background,
#                        # frontend in foreground; Ctrl+C stops both
#   ./run.sh backend     # start only the FastAPI backend (port 8000)
#   ./run.sh frontend     # start only the Vite dev server (port 5173)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-all}"
case "$MODE" in
  backend|frontend|all) ;;
  -h|--help)
    echo "Usage: $0 [backend|frontend|all]"
    exit 0
    ;;
  *)
    echo "Usage: $0 [backend|frontend|all]" >&2
    exit 1
    ;;
esac

VENV_DIR="$SCRIPT_DIR/.venv"
if [ -f "$VENV_DIR/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
elif [ -f "$VENV_DIR/Scripts/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/Scripts/activate"
else
  echo "ERROR: .venv not found. Run ./setup.sh first." >&2
  exit 1
fi

start_backend() {
  echo "==> Starting backend (uvicorn, http://0.0.0.0:8000)"
  # No parenthesized subshell here: this function is invoked as `start_backend &`
  # in `all` mode. Bash already forks a job subshell for that `&`; wrapping the
  # body in `(...)` on top of that adds a SECOND subshell layer that the `exec`
  # tail-call collapse does not reach through a function-call frame, so `$!`
  # would capture an inert bash wrapper instead of uvicorn. With no extra
  # subshell, `exec` replaces the job's own process image, so `$!` is uvicorn's
  # real PID and `kill "$BACKEND_PID"` in the trap actually stops the server
  # instead of orphaning it on port 8000.
  cd "$SCRIPT_DIR/backend" && exec uvicorn main:app --host 0.0.0.0 --port 8000 --reload
}

start_frontend() {
  echo "==> Starting frontend (vite, http://localhost:5173)"
  # Deliberately NOT exec'd (unlike start_backend): in `all` mode this runs in
  # the foreground AFTER the EXIT/INT/TERM trap is set. Exec-ing npm here would
  # replace this bash process's image, so when npm exits there would be no
  # bash process left to run the trap, and the backend would be orphaned
  # again. Running it as a plain command lets the script return, reach its
  # natural end, and fire the trap, which stops the backend.
  cd "$SCRIPT_DIR/frontend" && npm run dev
}

case "$MODE" in
  backend)
    start_backend
    ;;
  frontend)
    start_frontend
    ;;
  all)
    # Keep it simple: run the backend as a background job and the frontend in
    # the foreground. Ctrl+C (or the frontend exiting) triggers the trap below,
    # which stops the backend too so nothing is left running.
    start_backend &
    BACKEND_PID=$!
    trap 'echo "==> Stopping backend (pid $BACKEND_PID)"; kill "$BACKEND_PID" 2>/dev/null || true' EXIT INT TERM
    # Give the backend a moment to boot before the frontend starts probing it.
    sleep 2
    start_frontend
    ;;
esac
