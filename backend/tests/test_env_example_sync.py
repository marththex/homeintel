"""
tests/test_env_example_sync.py — Keeps .env.example and backend/config.py in sync.

.env.example claims ("Every setting below maps 1:1 to a field in
backend/config.py") that every Settings field has a corresponding
.env.example entry, and vice versa. Nothing enforced that claim, so it could
silently drift — a new Settings field added without a matching .env.example
line (or an .env.example line for a field that no longer exists) would go
unnoticed. This test parses both sides and diffs them.

Pure-unit module: only imports `config` (pydantic-settings — a light
dependency, no torch/docling/transformers/faster-whisper/qdrant-client/
langchain). Safe for CI, consistent with the other tests in this directory.
"""

import re
from pathlib import Path

from config import Settings

# .env.example lives at the repo root: backend/tests/test_env_example_sync.py
# -> tests -> backend -> <repo root>. Mirrors the env_file anchoring in
# config.py (Path(__file__).parent.parent / ".env").
ENV_EXAMPLE_PATH = Path(__file__).parent.parent.parent / ".env.example"

# Matches KEY=... lines, ignoring comments (#...) and blank lines.
_KEY_LINE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def _env_example_keys() -> set[str]:
    """Parse the KEY names out of .env.example (case-normalized to lower)."""
    keys = set()
    for line in ENV_EXAMPLE_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _KEY_LINE_RE.match(stripped)
        if match:
            key = stripped.split("=", 1)[0].strip()
            keys.add(key.lower())
    return keys


def _settings_field_names() -> set[str]:
    """Settings field names (case-normalized to lower)."""
    return {name.lower() for name in Settings.model_fields.keys()}


def test_env_example_file_exists():
    assert ENV_EXAMPLE_PATH.is_file(), f".env.example not found at {ENV_EXAMPLE_PATH}"


def test_every_settings_field_is_documented_in_env_example():
    env_keys = _env_example_keys()
    settings_fields = _settings_field_names()
    missing = sorted(settings_fields - env_keys)
    assert not missing, (
        "Settings field(s) in backend/config.py missing from .env.example: "
        f"{missing}"
    )


def test_every_env_example_key_is_a_settings_field():
    env_keys = _env_example_keys()
    settings_fields = _settings_field_names()
    unknown = sorted(env_keys - settings_fields)
    assert not unknown, (
        ".env.example key(s) do not correspond to any Settings field in "
        f"backend/config.py: {unknown}"
    )
