# Security Policy

HomeIntel is a self-hosted, fully local application — there is no hosted/cloud
service to secure on our end, but the codebase still handles potentially
sensitive local data (your documents, photos, audio, and anything ingested from
config files), so vulnerability reports are taken seriously.

## Supported Versions

This project does not yet maintain multiple release branches — security fixes
are made against the `main` branch. Please always run the latest commit on
`main` and keep dependencies (`backend/requirements.txt`,
`frontend/package.json`) up to date.

## Reporting a Vulnerability

If you find a security vulnerability, **please do not open a public GitHub
issue.** Instead:

- Preferred: use **[GitHub's private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing/privately-reporting-a-security-vulnerability)**
  feature on this repository (Security tab → "Report a vulnerability"), if
  enabled.
- Alternative: contact the maintainer directly at **`<CONTACT>`**.

Please include:
- A description of the vulnerability and its potential impact
- Steps to reproduce (a minimal repro is very helpful)
- Any relevant logs, versions, or configuration

We'll aim to acknowledge reports promptly and follow up with a fix timeline once
the issue is understood. Please give us a reasonable amount of time to address
the issue before any public disclosure.

## Scope Notes

Things that are **in scope** for a report:
- Path traversal or unauthorized file access via the API (`GET /file`,
  `GET /thumb`, ingestion path handling)
- Ways secrets could leak despite the redaction feature (see below)
- Authentication/authorization gaps if you're running HomeIntel exposed beyond
  `localhost` (e.g. via Tailscale or a reverse proxy)
- Dependency vulnerabilities with a demonstrated exploit path in this project's
  usage

Things that are generally **out of scope**:
- The application is designed to run on a trusted local network with no
  built-in user authentication — exposing it directly to the public internet
  without your own auth/reverse-proxy layer in front is a deployment choice, not
  a HomeIntel vulnerability. (See `docs/tailscale-https-setup.md` for a
  private-network-only way to reach it from a phone.)
- Vulnerabilities that require local admin/root access to exploit are generally
  out of scope, since local code execution already implies broad access.

## Built-in Secret Redaction

HomeIntel includes a secret-redaction feature (`REDACT_SECRETS=true` by default,
implemented in `backend/security/redact.py`) that detects and replaces
passwords, API keys, tokens, and private-key blocks with `<REDACTED>` at three
points: before storing chunks in Qdrant, before they reach the LLM's context,
and before source excerpts are returned to the UI. It errs toward
over-redaction rather than under-redaction, but it is a **best-effort regex-based
detector, not a guarantee** — do not rely on it as your only safeguard for
highly sensitive material. Prefer excluding directories that contain credential
stores (password manager exports, TLS/SSH keys, etc.) entirely via
`WATCHER_EXCLUDE_PATHS` rather than relying on redaction alone. If you find a
class of secret the detector misses, please report it as described above (or
open a normal issue/PR if it's not security-sensitive on its own).
