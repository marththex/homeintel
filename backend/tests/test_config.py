"""
tests/test_config.py — Unit tests for backend/config.py list-property parsing.

pydantic-settings is a light dependency (no torch/docling/transformers/
faster-whisper/qdrant-client/langchain) — safe for CI.

Settings() also reads a repo-root .env file if one is present (real dev
machines have one). Every test explicitly monkeypatches the env vars it
cares about — env vars take precedence over .env file values in
pydantic-settings, so tests are deterministic regardless of the local .env
contents. get_settings() is an lru_cache singleton, so cache_clear() is
called before and after each test to force a fresh Settings() read.
"""

import pytest

from config import get_settings


@pytest.fixture(autouse=True)
def _fresh_settings_cache():
    """Force get_settings() to rebuild Settings() from the current env."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ── supported_extensions_list ────────────────────────────────────────────────

def test_supported_extensions_list_parses_csv(monkeypatch):
    monkeypatch.setenv("SUPPORTED_EXTENSIONS", ".pdf,.docx,.txt")
    settings = get_settings()
    assert settings.supported_extensions_list == [".pdf", ".docx", ".txt"]


def test_supported_extensions_list_strips_whitespace_and_drops_empties(monkeypatch):
    monkeypatch.setenv("SUPPORTED_EXTENSIONS", " .pdf , .docx ,, .md ")
    settings = get_settings()
    assert settings.supported_extensions_list == [".pdf", ".docx", ".md"]


def test_supported_extensions_list_single_value(monkeypatch):
    monkeypatch.setenv("SUPPORTED_EXTENSIONS", ".txt")
    settings = get_settings()
    assert settings.supported_extensions_list == [".txt"]


def test_supported_extensions_list_empty_string_yields_empty_list(monkeypatch):
    monkeypatch.setenv("SUPPORTED_EXTENSIONS", "")
    settings = get_settings()
    assert settings.supported_extensions_list == []


def test_supported_extensions_env_var_is_case_insensitive(monkeypatch):
    """config.py sets case_sensitive=False, so lowercase env var names work too."""
    monkeypatch.setenv("supported_extensions", ".png,.jpg")
    settings = get_settings()
    assert settings.supported_extensions_list == [".png", ".jpg"]


# ── watcher_exclude_paths_list ───────────────────────────────────────────────

def test_watcher_exclude_paths_list_parses_csv(monkeypatch):
    monkeypatch.setenv("WATCHER_EXCLUDE_PATHS", "/data/cache,/data/logs")
    settings = get_settings()
    assert settings.watcher_exclude_paths_list == ["/data/cache", "/data/logs"]


def test_watcher_exclude_paths_list_strips_whitespace_and_drops_empties(monkeypatch):
    monkeypatch.setenv("WATCHER_EXCLUDE_PATHS", " /a/b , /c/d ,, ")
    settings = get_settings()
    assert settings.watcher_exclude_paths_list == ["/a/b", "/c/d"]


def test_watcher_exclude_paths_list_empty_by_default(monkeypatch):
    monkeypatch.setenv("WATCHER_EXCLUDE_PATHS", "")
    settings = get_settings()
    assert settings.watcher_exclude_paths_list == []


# ── cors_allow_origins_list (bonus coverage — same parsing pattern) ──────────

def test_cors_allow_origins_list_parses_csv(monkeypatch):
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "http://localhost:5173,http://localhost:3000")
    settings = get_settings()
    assert settings.cors_allow_origins_list == [
        "http://localhost:5173",
        "http://localhost:3000",
    ]


# ── Modality extension helpers ───────────────────────────────────────────────

def test_document_extensions_filters_to_known_document_types(monkeypatch):
    monkeypatch.setenv("SUPPORTED_EXTENSIONS", ".pdf,.docx,.txt,.md,.yml,.yaml,.json,.jpg,.mp3,.exe")
    settings = get_settings()
    assert settings.document_extensions == [
        ".pdf", ".docx", ".txt", ".md", ".yml", ".yaml", ".json",
    ]


def test_image_extensions_filters_to_known_image_types(monkeypatch):
    monkeypatch.setenv("SUPPORTED_EXTENSIONS", ".png,.jpg,.jpeg,.gif,.webp,.pdf,.mp3")
    settings = get_settings()
    assert settings.image_extensions == [".png", ".jpg", ".jpeg", ".gif", ".webp"]


def test_audio_extensions_filters_to_known_audio_types(monkeypatch):
    monkeypatch.setenv("SUPPORTED_EXTENSIONS", ".mp3,.wav,.m4a,.flac,.ogg,.pdf,.jpg")
    settings = get_settings()
    assert settings.audio_extensions == [".mp3", ".wav", ".m4a", ".flac", ".ogg"]


def test_video_extensions_filters_to_known_video_types(monkeypatch):
    monkeypatch.setenv("SUPPORTED_EXTENSIONS", ".mp4,.mov,.avi,.mkv,.pdf")
    settings = get_settings()
    assert settings.video_extensions == [".mp4", ".mov", ".avi", ".mkv"]


def test_modality_extensions_are_empty_when_not_in_supported_list(monkeypatch):
    """An extension not present in SUPPORTED_EXTENSIONS is excluded even
    though it belongs to a known modality set."""
    monkeypatch.setenv("SUPPORTED_EXTENSIONS", ".pdf")
    settings = get_settings()
    assert settings.image_extensions == []
    assert settings.audio_extensions == []
    assert settings.video_extensions == []
    assert settings.document_extensions == [".pdf"]


def test_unknown_extension_excluded_from_every_modality_helper(monkeypatch):
    monkeypatch.setenv("SUPPORTED_EXTENSIONS", ".pdf,.xyz")
    settings = get_settings()
    assert ".xyz" not in settings.document_extensions
    assert ".xyz" not in settings.image_extensions
    assert ".xyz" not in settings.audio_extensions
    assert ".xyz" not in settings.video_extensions
