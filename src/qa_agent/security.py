"""Security helpers: SSRF protection, path traversal prevention, secret handling.

These are deliberately strict by default. Tests/callers can opt-out via
explicit parameters when they really need to (none currently do).
"""

from __future__ import annotations

import ipaddress
import logging
import os
import re
import secrets
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# -- Limits (overridable via env) ------------------------------------------------

MAX_TEXT_BYTES = int(os.environ.get("QA_MAX_TEXT_BYTES", str(64 * 1024)))           # 64 KiB
MAX_IMAGE_BYTES = int(os.environ.get("QA_MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))  # 10 MiB
MAX_HTTP_RESPONSE_BYTES = int(os.environ.get("QA_MAX_HTTP_RESPONSE_BYTES", str(2 * 1024 * 1024)))  # 2 MiB

ALLOWED_URL_SCHEMES = frozenset({"http", "https"})
ALLOWED_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif", ".gif"})
ALLOWED_DOC_SUFFIXES = frozenset({".pdf", ".docx"})
# Template uploads now accept either an image OR a document (PDF / Word).
ALLOWED_TEMPLATE_SUFFIXES = ALLOWED_IMAGE_SUFFIXES | ALLOWED_DOC_SUFFIXES
# Documents (esp. PDF) can be larger than a single screenshot. Cap separately
# from MAX_IMAGE_BYTES so a malicious upload still has a bound, but a real
# multi-page brief isn't rejected at the door.
MAX_DOC_BYTES = int(os.environ.get("QA_MAX_DOC_BYTES", str(20 * 1024 * 1024)))  # 20 MiB

# Optional whitelist of directories template files may be loaded from.
# Defaults to the project's `templates/` directory plus the current working dir.
def _allowed_template_roots() -> list[Path]:
    raw = os.environ.get("QA_TEMPLATE_DIRS", "").strip()
    if raw:
        return [Path(p).expanduser().resolve() for p in raw.split(os.pathsep) if p.strip()]
    return [Path.cwd().resolve(), (Path.cwd() / "templates").resolve()]


# -- SSRF protection -------------------------------------------------------------

class UnsafeURLError(ValueError):
    """Raised when a URL fails SSRF safety checks."""


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
        or (isinstance(ip, ipaddress.IPv4Address) and str(ip).startswith("169.254."))  # cloud metadata
    )


def validate_public_url(url: str) -> str:
    """Reject schemes other than http(s) and hostnames that resolve to private IPs.

    Returns the (normalised) URL on success; raises UnsafeURLError otherwise.
    DNS rebinding is not fully solved here, but Playwright's own networking layer
    plus this hostname check covers the common SSRF surface.
    """
    if not isinstance(url, str) or not url.strip():
        raise UnsafeURLError("URL must be a non-empty string.")
    parsed = urlparse(url.strip())
    if parsed.scheme.lower() not in ALLOWED_URL_SCHEMES:
        raise UnsafeURLError(f"URL scheme '{parsed.scheme}' is not allowed.")
    if not parsed.hostname:
        raise UnsafeURLError("URL must include a hostname.")
    host = parsed.hostname
    # If the host is a bare IP literal, block private/loopback/etc ranges.
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address | None = None
    bare_host = host.strip("[]")  # IPv6 in brackets in URLs
    try:
        ip = ipaddress.ip_address(bare_host)
    except ValueError:
        ip = None
    if ip is not None:
        if _is_blocked_ip(ip):
            raise UnsafeURLError(f"URL host {host} resolves to a blocked range.")
    else:
        # Hostname — shallow checks only (we don't resolve DNS here on purpose).
        if host.lower() in {"localhost", "metadata.google.internal"} or host.lower().endswith(".internal"):
            raise UnsafeURLError(f"URL host {host} is not allowed.")
    return parsed.geturl()


# -- Path traversal protection ---------------------------------------------------

class UnsafePathError(ValueError):
    """Raised when a filesystem path fails safety checks."""


def safe_resolve_template(path_str: str) -> Path:
    """Resolve a template path, refusing traversal and unknown extensions.

    Accepts both image templates (PNG / JPEG / WebP / etc) and document
    templates (PDF / DOCX). Size limits are applied per-class — images are
    capped at MAX_IMAGE_BYTES, documents at MAX_DOC_BYTES. The resolved
    path must live under one of the allowed template roots (project cwd,
    ./templates, or directories listed in QA_TEMPLATE_DIRS).
    """
    if not path_str or not isinstance(path_str, str):
        raise UnsafePathError("Template path must be a non-empty string.")
    candidate = Path(path_str).expanduser().resolve()
    if not candidate.exists() or not candidate.is_file():
        raise UnsafePathError(f"Template file not found: {path_str}")
    suffix = candidate.suffix.lower()
    if suffix not in ALLOWED_TEMPLATE_SUFFIXES:
        raise UnsafePathError(
            f"Template extension '{suffix}' is not in the allowed set "
            f"{sorted(ALLOWED_TEMPLATE_SUFFIXES)}."
        )
    size = candidate.stat().st_size
    cap = MAX_DOC_BYTES if suffix in ALLOWED_DOC_SUFFIXES else MAX_IMAGE_BYTES
    if size > cap:
        raise UnsafePathError(
            f"Template file exceeds {cap} bytes ({size})."
        )
    roots = _allowed_template_roots()
    if not any(_is_within(candidate, r) for r in roots):
        raise UnsafePathError(
            f"Template path is outside allowed roots ({[str(r) for r in roots]})."
        )
    return candidate


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


# -- Secret handling -------------------------------------------------------------

_SECRET_ENV_VAR_NAMES = (
    "OPENROUTER_API_KEY",
    "MCP_AUTH_TOKEN",
    "TESSERACT_CMD",  # not strictly secret but treat as sensitive infra detail
)

_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),                # OpenAI/OpenRouter style
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]+", re.I),
    re.compile(r"[A-Za-z0-9]{32,}"),                     # generic long opaque token
]


def redact(text: str) -> str:
    """Best-effort redaction of secrets in arbitrary log/error text."""
    if not isinstance(text, str):
        return text
    redacted = text
    # Replace exact known secret values first (safer than pattern matching).
    for name in _SECRET_ENV_VAR_NAMES:
        val = os.environ.get(name, "").strip()
        if val and len(val) >= 8 and val in redacted:
            redacted = redacted.replace(val, f"<redacted:{name}>")
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("<redacted>", redacted)
    return redacted


def require_env(name: str) -> str:
    """Fetch a required env var without ever logging its value."""
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(
            f"Required environment variable '{name}' is not set. "
            f"Copy .env.example to .env and fill it in."
        )
    return val


def constant_time_equals(a: str, b: str) -> bool:
    """Compare two strings in constant time (auth-token check)."""
    return secrets.compare_digest(a or "", b or "")


# -- Tool-input sanitisation -----------------------------------------------------

def truncate_text(text: str | None, limit: int = MAX_TEXT_BYTES) -> str:
    if not text:
        return ""
    encoded = text.encode("utf-8", errors="ignore")
    if len(encoded) <= limit:
        return text
    return encoded[:limit].decode("utf-8", errors="ignore")
