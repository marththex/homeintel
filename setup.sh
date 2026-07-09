#!/usr/bin/env bash
# HomeIntel setup script (Linux / macOS / Git-Bash on Windows).
#
# Creates a Python virtualenv at .venv, installs PyTorch (CPU by default,
# CUDA cu128 with --gpu), installs backend Python deps, installs frontend
# npm deps, and creates .env from .env.example if missing.
#
# Usage:
#   ./setup.sh          # CPU-only PyTorch (works everywhere)
#   ./setup.sh --gpu    # CUDA cu128 PyTorch (NVIDIA GPU on Windows/Linux only)
#
# Safe to re-run: existing .venv / .env are left alone, npm install is
# naturally idempotent.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GPU=0
for arg in "$@"; do
  case "$arg" in
    --gpu)
      GPU=1
      ;;
    -h|--help)
      echo "Usage: $0 [--gpu]"
      echo "  --gpu   Install the CUDA (cu128) build of PyTorch instead of the CPU build."
      echo "          Only meaningful on Windows/Linux with an NVIDIA GPU + drivers."
      echo "          macOS has no CUDA support and always uses the CPU/MPS build."
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      echo "Usage: $0 [--gpu]" >&2
      exit 1
      ;;
  esac
done

step() { printf '\n==> %s\n' "$1"; }

# ── 1. Locate Python 3.11+ ────────────────────────────────────────────────────
step "Checking for Python 3.11+"

PYTHON_BIN=""
for candidate in python3.11 python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then
    ver="$("$candidate" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)"
    if [ -n "$ver" ]; then
      major="${ver%%.*}"
      minor="${ver##*.}"
      if [ "$major" -eq 3 ] && [ "$minor" -ge 11 ]; then
        PYTHON_BIN="$candidate"
        break
      fi
    fi
  fi
done

if [ -z "$PYTHON_BIN" ]; then
  echo "ERROR: Python 3.11+ not found on PATH (tried python3.11, python3, python)." >&2
  echo "Install Python 3.11 or newer: https://www.python.org/downloads/" >&2
  exit 1
fi

echo "Using $("$PYTHON_BIN" --version) at $(command -v "$PYTHON_BIN")"

# ── 2. Create virtualenv ──────────────────────────────────────────────────────
VENV_DIR="$SCRIPT_DIR/.venv"
if [ -d "$VENV_DIR" ]; then
  step "Virtualenv already exists at .venv (skipping creation)"
else
  step "Creating virtualenv at .venv"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

if [ -f "$VENV_DIR/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
elif [ -f "$VENV_DIR/Scripts/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/Scripts/activate"
else
  echo "ERROR: could not find the venv activate script under $VENV_DIR" >&2
  exit 1
fi

# ── 3. Upgrade pip ────────────────────────────────────────────────────────────
step "Upgrading pip"
python -m pip install --upgrade pip

# ── 4. Install PyTorch (CPU by default, CUDA cu128 with --gpu) ────────────────
OS_NAME="$(uname -s 2>/dev/null || echo unknown)"

if [ "$GPU" -eq 1 ]; then
  if [ "$OS_NAME" = "Darwin" ]; then
    echo "NOTE: --gpu was passed but macOS has no CUDA support. Installing the CPU/MPS build instead."
    step "Installing PyTorch (CPU/MPS build — macOS)"
    python -m pip install torch torchvision
  else
    step "Installing PyTorch (CUDA cu128 build — NVIDIA GPU)"
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  fi
else
  step "Installing PyTorch (CPU build)"
  echo "NOTE: macOS always uses this CPU/MPS build — there is no CUDA support for macOS."
  echo "NOTE: On Windows/Linux with an NVIDIA GPU, re-run as './setup.sh --gpu' for the CUDA build."
  python -m pip install torch torchvision
fi

# ── 5. Backend dependencies ───────────────────────────────────────────────────
step "Installing backend dependencies (backend/requirements.txt)"
python -m pip install -r "$SCRIPT_DIR/backend/requirements.txt"

if [ -f "$SCRIPT_DIR/backend/requirements-dev.txt" ]; then
  echo "NOTE: backend/requirements-dev.txt is available for contributors (pytest, lint, etc.)."
  echo "      Install it with: pip install -r backend/requirements-dev.txt"
fi

# ── 6. Frontend dependencies ──────────────────────────────────────────────────
step "Checking for Node.js / npm"
if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
  echo "ERROR: node and/or npm not found on PATH." >&2
  echo "Install Node.js (includes npm): https://nodejs.org/" >&2
  exit 1
fi
echo "Using node $(node --version), npm $(npm --version)"

step "Installing frontend dependencies (npm install)"
(cd "$SCRIPT_DIR/frontend" && npm install)

# ── 7. .env file ──────────────────────────────────────────────────────────────
step "Checking for .env"
if [ -f "$SCRIPT_DIR/.env" ]; then
  echo ".env already exists — leaving it untouched."
else
  cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
  echo "Created .env from .env.example — review and edit it before running the app."
fi

step "Setup complete"
echo "Next steps:"
echo "  1. Review/edit .env (NAS_WATCH_PATH, QDRANT_URL, OLLAMA_* as needed)."
echo "  2. Make sure Ollama and Qdrant are reachable."
echo "  3. Start the app: ./run.sh"
