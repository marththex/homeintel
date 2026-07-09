"""
tests/test_redact.py — Unit tests for backend/security/redact.py.

Pure regex module (`import re` only) — no external services, no torch/
docling/transformers/faster-whisper/qdrant-client/langchain. Safe to run
anywhere, including CI.
"""

import pytest

from security.redact import redact_secrets, contains_secret, REDACTED


# ── PEM private key blocks ───────────────────────────────────────────────────

def test_redacts_rsa_private_key_block():
    text = (
        "before\n"
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEpAIBAAKCAQEA1234567890abcdefghijklmnopqrstuvwxyz\n"
        "-----END RSA PRIVATE KEY-----\n"
        "after"
    )
    out = redact_secrets(text)
    assert "MIIEpAIBAAKCAQEA" not in out
    assert REDACTED in out
    assert out.startswith("before\n")
    assert out.endswith("\nafter")


def test_redacts_generic_private_key_block_no_algo_prefix():
    text = (
        "-----BEGIN PRIVATE KEY-----\n"
        "abc123def456\n"
        "-----END PRIVATE KEY-----"
    )
    out = redact_secrets(text)
    assert out == REDACTED
    assert contains_secret(text) is True


def test_redacts_openssh_private_key_block():
    text = (
        "-----BEGIN OPENSSH PRIVATE KEY-----\n"
        "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQ==\n"
        "-----END OPENSSH PRIVATE KEY-----"
    )
    out = redact_secrets(text)
    assert "b3BlbnNzaC1rZXktdjEA" not in out
    assert REDACTED in out


def test_does_not_redact_public_key_block():
    """PUBLIC keys are not secrets — must be left alone."""
    text = (
        "-----BEGIN PUBLIC KEY-----\n"
        "MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAE\n"
        "-----END PUBLIC KEY-----"
    )
    out = redact_secrets(text)
    assert out == text
    assert contains_secret(text) is False


# ── High-signal token prefixes ───────────────────────────────────────────────

@pytest.mark.parametrize(
    "token",
    [
        "sk-" + "a" * 20,
        "ghp_" + "A" * 24,
        "gho_" + "B" * 24,
        "ghs_" + "C" * 24,
        "xoxb-" + "1234567890",
        "AKIA" + "A" * 16,
        "AIza" + "A" * 35,
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        ".eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIn0"
        ".SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
    ],
)
def test_redacts_known_token_prefixes(token):
    text = f"here is a token: {token} — do not share"
    out = redact_secrets(text)
    assert token not in out
    assert REDACTED in out
    assert contains_secret(text) is True


def test_does_not_redact_too_short_akia_like_string():
    """AKIA followed by fewer than 16 chars is not a real AWS key id shape."""
    text = "reference code AKIAABCDEFGHIJKL end"  # only 12 chars after AKIA
    out = redact_secrets(text)
    assert out == text
    assert contains_secret(text) is False


def test_does_not_redact_lowercase_akia_like_string():
    text = "AKIAiosfodnn7example"  # AWS key ids are uppercase only
    out = redact_secrets(text)
    assert out == text


def test_does_not_redact_two_segment_jwt_lookalike():
    """Only two dot-separated segments — not a full JWT."""
    text = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0"
    out = redact_secrets(text)
    assert out == text
    assert contains_secret(text) is False


# ── key/value secret pairs ───────────────────────────────────────────────────

@pytest.mark.parametrize(
    "text,expected_key_kept",
    [
        ("POSTGRES_PASSWORD=hunter2irl", "POSTGRES_PASSWORD="),
        ('"password": "hunter2irl"', '"password":'),
        ("password: hunter2irl", "password:"),
        ("api_key=abcdef123456", "api_key="),
        ('"api-key": "abcdef123456"', '"api-key":'),
        ("AUTH_TOKEN=abcdef123456", "AUTH_TOKEN="),
        ("client_secret=abcdef123456", "client_secret="),
        ("aws_access_key_id=abcdef123456", "aws_access_key_id="),
        ("otp_secret=abcdef123456", "otp_secret="),
        ("recovery_code=abcdef123456", "recovery_code="),
        ("seed_phrase=abcdef123456", "seed_phrase="),
        ("mnemonic=abcdef123456", "mnemonic="),
        ("totp=123456789", "totp="),
    ],
)
def test_redacts_secret_key_value_pairs(text, expected_key_kept):
    out = redact_secrets(text)
    assert REDACTED in out
    # Key name / structure stays, so the line is still searchable.
    assert expected_key_kept in out
    assert contains_secret(text) is True


def test_redacts_value_but_keeps_surrounding_json_structure():
    text = '{"username": "jane", "password": "hunter2irl", "role": "admin"}'
    out = redact_secrets(text)
    assert '"username": "jane"' in out
    assert '"role": "admin"' in out
    assert '"password": "<REDACTED>"' in out
    assert "hunter2irl" not in out


def test_does_not_redact_password_word_without_value():
    """A sentence merely mentioning 'password' with no key=value shape."""
    text = "Please enter your password to continue."
    out = redact_secrets(text)
    assert out == text
    assert contains_secret(text) is False


def test_does_not_redact_secret_word_without_value():
    text = "There is no secret ingredient in this recipe."
    out = redact_secrets(text)
    assert out == text
    assert contains_secret(text) is False


def test_does_not_redact_value_shorter_than_three_chars():
    text = "pwd=ab"
    out = redact_secrets(text)
    assert out == text
    assert contains_secret(text) is False


def test_does_not_redact_unrelated_key_value_pair():
    text = "query=value&page=2"
    out = redact_secrets(text)
    assert out == text
    assert contains_secret(text) is False


# ── Connection-string credentials ────────────────────────────────────────────

def test_redacts_connection_string_password():
    text = "postgres://dbuser:s3cr3tPass@db.internal:5432/mydb"
    out = redact_secrets(text)
    assert out == "postgres://dbuser:<REDACTED>@db.internal:5432/mydb"
    assert contains_secret(text) is True


def test_redacts_mongodb_style_connection_string():
    text = "mongodb://admin:hunter2@mongo-host:27017/app"
    out = redact_secrets(text)
    assert "hunter2" not in out
    assert "mongodb://admin:<REDACTED>@mongo-host:27017/app" == out


def test_does_not_redact_url_without_credentials():
    text = "https://example.com/path?query=value"
    out = redact_secrets(text)
    assert out == text
    assert contains_secret(text) is False


# ── Multiple secrets / mixed content ─────────────────────────────────────────

def test_redacts_multiple_distinct_secrets_in_one_text():
    text = (
        "DB_PASSWORD=hunter2irl\n"
        "postgres://dbuser:s3cr3tPass@db.internal:5432/mydb\n"
        "token: " + "sk-" + "b" * 20
    )
    out = redact_secrets(text)
    assert "hunter2irl" not in out
    assert "s3cr3tPass" not in out
    assert "sk-" + "b" * 20 not in out
    assert out.count(REDACTED) == 3


def test_plain_text_with_no_secrets_is_unchanged():
    text = (
        "This is a perfectly ordinary paragraph about home network notes.\n"
        "Trash pickup is every Tuesday morning; recycling is every other week."
    )
    out = redact_secrets(text)
    assert out == text
    assert contains_secret(text) is False


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_empty_string_returns_empty_string():
    assert redact_secrets("") == ""
    assert contains_secret("") is False


def test_none_like_falsy_input_is_returned_as_is():
    # redact_secrets guards on `if not text: return text`
    assert redact_secrets(None) is None
    assert contains_secret(None) is False
