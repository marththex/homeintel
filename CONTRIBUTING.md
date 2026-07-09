# Contributing to HomeIntel

Thanks for your interest in contributing. This is a personal/open-source project
run in spare time, so please be patient with review turnaround — but PRs, issues,
and questions are all welcome.

---

## Getting set up

1. Fork and clone the repo:

   ```bash
   git clone https://github.com/<your-fork>/homeintel.git
   cd homeintel
   ```

2. Run the setup script for your OS. This creates a `.venv`, installs backend +
   frontend dependencies, and bootstraps `.env` from `.env.example`:

   ```bash
   ./setup.sh          # macOS/Linux (add --gpu for a CUDA PyTorch build)
   .\setup.ps1         # Windows        (add -Gpu for a CUDA PyTorch build)
   ```

3. Install the dev/test dependencies (not installed by `setup.*`, since they're
   only needed for contributing, not running the app):

   ```bash
   pip install -r backend/requirements-dev.txt
   ```

4. Start the app to make sure everything works before you start changing things:

   ```bash
   ./run.sh            # or .\run.ps1 on Windows
   ```

   The bundled `data/sample-docs/` means you don't need a NAS, real documents, or
   even Ollama/Qdrant running just to poke at the frontend — though full
   end-to-end testing does need both (see below).

---

## Running the tests

**Unit tests** — fast, pure-Python, no Qdrant/Ollama required. This is what CI runs
and what you should run before opening a PR:

```bash
python -m pytest backend/tests
```

**Lint:**

```bash
ruff check backend
```

**Integration/manual smoke tests** — these need a live Qdrant and Ollama instance
and exercise real ingestion/retrieval against the bundled sample docs. Not
required for most PRs, but useful if you're touching ingestion, retrieval, or the
API surface:

```bash
cd backend
python ../scripts/test_vectorstore.py
python ../scripts/test_ingestion.py
python ../scripts/verify_api.py
python ../scripts/verify_streaming.py
```

**Frontend:**

```bash
cd frontend
npm run build   # type-checks (tsc) + builds — should stay green on every commit
```

---

## Branching & PR conventions

- Branch off `main`; give your branch a short descriptive name
  (e.g. `fix/watcher-permission-error`, `feat/audio-language-filter`).
- Commit messages follow roughly: `feat(scope): ...`, `fix(scope): ...`,
  `perf(scope): ...`, `docs: ...`, `chore: ...` — scope is usually `ui`, `api`,
  `ingestion`, `rag`, etc. Not strictly enforced, but keep messages focused on
  *why* a change was made, not just *what* changed.
- Keep PRs focused — one logical change per PR is much easier to review than a
  large mixed diff.
- Update relevant docs (`README.md`, `CLAUDE.md`, or `.env.example`) in the same
  PR if you change configuration, setup, or behavior a user would notice.
- Make sure `python -m pytest backend/tests`, `ruff check backend`, and
  `npm run build` (frontend) all pass before requesting review.

---

## Secrets and `.env`

- **Never commit `.env`** — it's gitignored for a reason. Only `.env.example`
  (with placeholder/default values, no real credentials or private
  hostnames/IPs) should ever be committed.
- If you add a new setting, add it to **both** `backend/config.py` (with a
  sensible default and a description) **and** `.env.example` (with a comment
  explaining it) — `.env.example` is the documented source of truth for
  configuration.
- Don't commit real API keys, tokens, IP addresses, or personal file paths in
  code, tests, or example scripts. Use placeholders (`<QDRANT_HOST>`, generic
  paths, etc.) the way the rest of the codebase does.
- If you're touching anything security-relevant (the redaction logic in
  `backend/security/redact.py`, path validation in `backend/api/files.py`,
  etc.), please read [`SECURITY.md`](SECURITY.md) first.

---

## Code style

- Python: follow the existing style in the file you're editing; run
  `ruff check backend` before committing.
- TypeScript/React: no additional state-management library — component state and
  a handful of hooks are intentional (see `docs/ARCHITECTURE.md` for the current
  shape); mobile-first CSS with additive `min-width` breakpoints (see `App.css`).
- Prefer small, focused functions/components over large ones, matching the
  existing structure of `backend/rag/`, `backend/ingestion/`, and
  `frontend/src/components/`.

---

## Questions

Open a GitHub issue for bugs, feature requests, or questions about the codebase.
