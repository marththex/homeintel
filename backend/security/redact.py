"""
security/redact.py — Detect and redact secrets from text.

Used to keep passwords, API keys, private keys, and tokens out of:
  1. The vector store (scrubbed at ingestion time)
  2. The LLM context (scrubbed before the prompt is built)
  3. The source excerpts shown in the UI

This is best-effort, regex-based secret scanning (in the spirit of gitleaks /
trufflehog). It errs toward over-redaction — for a personal NAS, hiding a
non-secret occasionally is far better than leaking a real credential.

The actual secret value is replaced with the literal "<REDACTED>". The
surrounding key name / structure is kept so the text stays searchable
("what is my postgres password" still finds the line, just shows <REDACTED>).
"""

import re

REDACTED = "<REDACTED>"

# 1. PEM private key blocks (RSA / EC / OPENSSH / generic) — redact the whole block.
_PRIVATE_KEY = re.compile(
    r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----.*?-----END (?:[A-Z0-9 ]+ )?PRIVATE KEY-----",
    re.DOTALL,
)

# 2. High-signal standalone token formats (known prefixes + JWT).
_TOKEN_PREFIX = re.compile(
    r"\b(?:"
    r"sk-[A-Za-z0-9]{16,}"                 # OpenAI-style
    r"|ghp_[A-Za-z0-9]{20,}|gho_[A-Za-z0-9]{20,}|ghs_[A-Za-z0-9]{20,}"  # GitHub
    r"|xox[baprs]-[A-Za-z0-9-]{10,}"       # Slack
    r"|AKIA[0-9A-Z]{16}"                   # AWS access key id
    r"|AIza[0-9A-Za-z_\-]{30,}"            # Google API key
    r"|eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"  # JWT
    r")\b"
)

# 3. key/value pairs where the key name implies a secret. Handles env-style
#    (POSTGRES_PASSWORD=...), YAML/JSON (password: ..., "password":"..."),
#    and prefixed/suffixed key names. Redacts the VALUE only.
_SECRET_TERM = (
    r"(?:password|passwd|pwd|passphrase|secret(?:[_-]?key)?|api[_-]?key|apikey"
    r"|access[_-]?key|secret[_-]?access[_-]?key|auth[_-]?token|token"
    r"|client[_-]?secret|private[_-]?key|encryption[_-]?key|credentials?|bearer"
    r"|totp|otp[_-]?secret|recovery[_-]?code|seed[_-]?phrase|mnemonic)"
)
_SECRET_KV = re.compile(
    r"(?P<key>(?:[A-Za-z0-9]+[_-])*" + _SECRET_TERM + r"(?:[_-][A-Za-z0-9]+)?)"
    r"(?P<sep>['\"]?\s*[:=]\s*)"   # tolerates a closing key-quote, e.g. JSON "password":
    r"(?P<q>['\"]?)"
    r"(?P<val>[^\s'\"|,}{]{3,})"
    r"(?P=q)",
    re.IGNORECASE,
)

# 4. Credentials embedded in a connection string: scheme://user:PASSWORD@host
_CONN_STRING = re.compile(r"(?P<pre>://[^:\s/@]+:)(?P<val>[^@\s/]+)(?P<post>@)")


def redact_secrets(text: str) -> str:
    """Return text with detected secrets replaced by <REDACTED>."""
    if not text:
        return text
    text = _PRIVATE_KEY.sub(REDACTED, text)
    text = _TOKEN_PREFIX.sub(REDACTED, text)
    text = _SECRET_KV.sub(
        lambda m: f"{m.group('key')}{m.group('sep')}{m.group('q')}{REDACTED}{m.group('q')}",
        text,
    )
    text = _CONN_STRING.sub(lambda m: f"{m.group('pre')}{REDACTED}{m.group('post')}", text)
    return text


def contains_secret(text: str) -> bool:
    """True if any secret pattern is present (cheap pre-check)."""
    if not text:
        return False
    return bool(
        _PRIVATE_KEY.search(text)
        or _TOKEN_PREFIX.search(text)
        or _SECRET_KV.search(text)
        or _CONN_STRING.search(text)
    )
