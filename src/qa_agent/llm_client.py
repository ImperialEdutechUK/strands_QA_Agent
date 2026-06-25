"""Hardened OpenRouter client used by tools that need structured JSON output."""

from __future__ import annotations

import json
import logging
import os
import re
import time

import httpx

from .provider_policy import openrouter_provider_block
from .security import MAX_HTTP_RESPONSE_BYTES, redact, require_env, truncate_text

logger = logging.getLogger(__name__)

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)
_LIMITS = httpx.Limits(max_keepalive_connections=5, max_connections=10)

# HTTP statuses that are worth retrying — rate limits and transient upstream
# failures. OpenRouter's free tiers return 429 readily, which used to fail a
# tool outright (and, for the template tool, silently drop the whole checklist).
_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
_MAX_ATTEMPTS = int(os.environ.get("LLM_MAX_ATTEMPTS", "4"))
_BACKOFF_BASE_SECONDS = float(os.environ.get("LLM_BACKOFF_SECONDS", "2.0"))
_BACKOFF_CAP_SECONDS = 30.0


def _client() -> httpx.Client:
    # verify=True is the default; pin it explicitly for reviewer comfort.
    return httpx.Client(timeout=_TIMEOUT, limits=_LIMITS, verify=True, follow_redirects=False)


def _retry_wait(resp: httpx.Response | None, attempt: int) -> float:
    """Seconds to wait before the next attempt — honour Retry-After if present."""
    if resp is not None:
        ra = resp.headers.get("retry-after")
        if ra:
            try:
                return min(float(ra), _BACKOFF_CAP_SECONDS)
            except ValueError:
                pass  # HTTP-date form — fall through to exponential backoff
    return min(_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)), _BACKOFF_CAP_SECONDS)


def call_llm(
    prompt: str,
    system: str | None = None,
    json_mode: bool = False,
    temperature: float = 0.2,
) -> str:
    api_key = require_env("OPENROUTER_API_KEY")
    if not ENDPOINT.startswith("https://"):  # paranoia
        raise RuntimeError("LLM endpoint must be HTTPS.")

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": truncate_text(system, limit=8 * 1024)})
    messages.append({"role": "user", "content": truncate_text(prompt)})

    body: dict = {
        "model": os.environ.get("MODEL", "deepseek/deepseek-v4-pro"),
        "messages": messages,
        "temperature": temperature,
        # Restrict OpenRouter to GDPR-jurisdiction, no-data-collection providers
        # and block Chinese-jurisdiction ones. See provider_policy.py.
        "provider": openrouter_provider_block(),
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost",
        "X-Title": "Strands QA Agent",
    }

    resp: httpx.Response | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            with _client() as client:
                resp = client.post(ENDPOINT, json=body, headers=headers)
        except httpx.HTTPError as exc:
            # Connection/timeout error — transient. Retry unless out of attempts.
            if attempt < _MAX_ATTEMPTS:
                wait = _retry_wait(None, attempt)
                logger.warning(
                    "OpenRouter connection error (attempt %d/%d): %s; retrying in %.1fs",
                    attempt, _MAX_ATTEMPTS, redact(str(exc)), wait,
                )
                time.sleep(wait)
                continue
            raise RuntimeError(f"OpenRouter call failed: {redact(str(exc))}") from None

        if resp.status_code in _RETRYABLE_STATUS and attempt < _MAX_ATTEMPTS:
            wait = _retry_wait(resp, attempt)
            logger.warning(
                "OpenRouter HTTP %s (attempt %d/%d); retrying in %.1fs",
                resp.status_code, attempt, _MAX_ATTEMPTS, wait,
            )
            time.sleep(wait)
            continue

        # Either a success, a non-retryable error, or the final attempt.
        try:
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"OpenRouter call failed: {redact(str(exc))}") from None
        break

    assert resp is not None  # the loop always assigns or raises
    if resp.headers.get("content-length"):
        try:
            if int(resp.headers["content-length"]) > MAX_HTTP_RESPONSE_BYTES:
                raise RuntimeError("OpenRouter response exceeds size cap.")
        except ValueError:
            pass

    data = resp.json()
    return data["choices"][0]["message"]["content"]


def call_llm_json(prompt: str, system: str | None = None, temperature: float = 0.2) -> dict:
    raw = call_llm(prompt, system=system, json_mode=True, temperature=temperature)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            return json.loads(match.group(0))
        raise RuntimeError(f"LLM did not return valid JSON:\n{redact(raw)}")
