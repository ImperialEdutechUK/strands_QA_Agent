"""Hardened OpenRouter client used by tools that need structured JSON output."""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import httpx

from .provider_policy import openrouter_provider_block
from .security import (
    MAX_HTTP_RESPONSE_BYTES,
    MAX_TEXT_BYTES,
    redact,
    require_env,
    truncate_text,
)

logger = logging.getLogger(__name__)

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)
_LIMITS = httpx.Limits(max_keepalive_connections=5, max_connections=10)

# HTTP statuses that are worth retrying — rate limits and transient upstream
# failures. OpenRouter returns 429 whenever the upstream provider is at
# capacity; because our provider policy pins routing to a small allow-list with
# fallbacks disabled (see provider_policy.py), there's nowhere to spill over, so
# brief provider saturation surfaces as a 429. Left unretried it used to fail a
# tool outright (and, for the template tool, silently drop the whole checklist).
_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
_MAX_ATTEMPTS = int(os.environ.get("LLM_MAX_ATTEMPTS", "5"))
_BACKOFF_BASE_SECONDS = float(os.environ.get("LLM_BACKOFF_SECONDS", "2.0"))
_BACKOFF_CAP_SECONDS = 30.0
# A server-specified Retry-After / rate-limit reset is authoritative, so we
# honour it up to this ceiling — long enough to let a saturated provider's
# per-minute window clear, but never so long that one call blocks for minutes.
_RETRY_AFTER_CAP_SECONDS = float(os.environ.get("LLM_RETRY_AFTER_CAP_SECONDS", "30.0"))
# Total wall-clock budget for ALL retries of a SINGLE call. Once spent we stop
# retrying and fail fast, so sustained provider rate-limiting can't turn one
# call into a multi-minute grind (which made a whole run take 25+ minutes).
_RETRY_BUDGET_SECONDS = float(os.environ.get("LLM_RETRY_BUDGET_SECONDS", "75.0"))


class RateLimitError(RuntimeError):
    """OpenRouter kept returning 429 after every retry was spent.

    A subclass of RuntimeError so existing `except RuntimeError` handlers (and
    the tools that degrade gracefully on LLM failure) keep working, while
    callers that care can distinguish a rate-limit exhaustion from other faults.
    """


def _client(timeout: httpx.Timeout | None = None) -> httpx.Client:
    # verify=True is the default; pin it explicitly for reviewer comfort.
    return httpx.Client(timeout=timeout or _TIMEOUT, limits=_LIMITS, verify=True,
                        follow_redirects=False)


def _parse_retry_after(value: str) -> float | None:
    """Interpret a Retry-After header (delta-seconds OR an HTTP-date)."""
    value = value.strip()
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


def _server_requested_wait(resp: httpx.Response) -> float | None:
    """Authoritative wait the provider asked for, if any.

    Prefers the standard Retry-After header, falling back to OpenRouter's
    `X-RateLimit-Reset` (a unix timestamp in milliseconds).
    """
    ra = resp.headers.get("retry-after")
    if ra:
        secs = _parse_retry_after(ra)
        if secs is not None:
            return secs
    reset = resp.headers.get("x-ratelimit-reset")
    if reset:
        try:
            delta = float(reset) / 1000.0 - time.time()
        except ValueError:
            return None
        if delta > 0:
            return delta
    return None


def _retry_wait(resp: httpx.Response | None, attempt: int) -> float:
    """Seconds to wait before the next attempt.

    A server-specified Retry-After / rate-limit reset is authoritative and is
    honoured up to `_RETRY_AFTER_CAP_SECONDS`. Otherwise we back off
    exponentially with jitter so concurrent callers don't retry in lockstep and
    immediately re-trip the same rate limit.
    """
    if resp is not None:
        requested = _server_requested_wait(resp)
        if requested is not None:
            return min(requested, _RETRY_AFTER_CAP_SECONDS)
    base = min(_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)), _BACKOFF_CAP_SECONDS)
    return base * random.uniform(0.75, 1.25)  # full-ish jitter (±25%)


def call_llm(
    prompt: str,
    system: str | None = None,
    json_mode: bool = False,
    # Default to greedy decoding: the QA tools extract and judge facts, so we
    # want deterministic, reproducible output, not creative variation. A higher
    # temperature was making the model guess (e.g. confusing a "Level 5 Diploma"
    # with a "Level 5 Extended Diploma").
    temperature: float = 0.0,
    # Hard cap on the user-prompt size (bytes). Defaults to the 64 KiB
    # tool-input bound, but analytical calls that must see the WHOLE page
    # (compliance, spell) raise it so the page tail isn't silently chopped —
    # those calls are single, server-side, and well within the model's context.
    max_prompt_bytes: int = MAX_TEXT_BYTES,
    # Upper bound on the model's OUTPUT tokens. Left unset the provider applies
    # its own (sometimes small) default, which truncated large JSON rule sets /
    # issue lists mid-array and made them unparseable. Structured callers raise it.
    max_tokens: int | None = None,
    # Per-call time bounds. `read_timeout` overrides the HTTP read timeout (a big
    # non-streaming generation must finish within it, else the attempt fails);
    # `retry_budget` caps the total wall-clock spent retrying. Callers that must
    # not stall the whole run (e.g. `template`, which safely falls back to the
    # baseline checklist) pass tight values so a saturated provider triggers a
    # FAST fallback instead of a multi-minute grind that looks like a hang.
    read_timeout: float | None = None,
    retry_budget: float | None = None,
    # Override the model for this call (e.g. the vision pass needs an
    # image-capable model — the default DeepSeek models are text-only).
    model: str | None = None,
    # Base64 PNG/JPEG images to attach to the user message (multimodal calls).
    # Each entry is either a raw base64 string or a full data: URI.
    images: list[str] | None = None,
    # Turn off hidden chain-of-thought on reasoning models (OpenRouter's
    # unified `reasoning` param; ignored by models that don't support it).
    # Reasoning models can burn the ENTIRE max_tokens budget on thinking and
    # return an empty `content` — the vision pass hit exactly that.
    disable_reasoning: bool = False,
) -> str:
    api_key = require_env("OPENROUTER_API_KEY")
    if not ENDPOINT.startswith("https://"):  # paranoia
        raise RuntimeError("LLM endpoint must be HTTPS.")

    http_timeout = (
        httpx.Timeout(connect=10.0, read=read_timeout, write=30.0, pool=10.0)
        if read_timeout is not None else _TIMEOUT
    )
    budget = retry_budget if retry_budget is not None else _RETRY_BUDGET_SECONDS

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": truncate_text(system, limit=8 * 1024)})
    user_text = truncate_text(prompt, limit=max_prompt_bytes)
    if images:
        # Multimodal message: OpenAI-style content parts (text + image_url).
        parts: list[dict] = [{"type": "text", "text": user_text}]
        for img in images:
            uri = img if img.startswith("data:") else f"data:image/png;base64,{img}"
            parts.append({"type": "image_url", "image_url": {"url": uri}})
        messages.append({"role": "user", "content": parts})
    else:
        messages.append({"role": "user", "content": user_text})

    body: dict = {
        "model": model or os.environ.get("MODEL", "deepseek/deepseek-v4-pro"),
        "messages": messages,
        "temperature": temperature,
        # Restrict OpenRouter to GDPR-jurisdiction, no-data-collection providers
        # and block Chinese-jurisdiction ones. See provider_policy.py.
        "provider": openrouter_provider_block(),
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if max_tokens is not None and max_tokens > 0:
        body["max_tokens"] = max_tokens
    if disable_reasoning:
        body["reasoning"] = {"enabled": False}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost",
        "X-Title": "Strands QA Agent",
    }

    def _can_retry(attempt: int, wait: float) -> bool:
        """Retry only while attempts AND the per-call time budget both remain."""
        if attempt >= _MAX_ATTEMPTS:
            return False
        return (time.monotonic() - _started) + wait <= budget

    resp: httpx.Response | None = None
    _started = time.monotonic()
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            with _client(http_timeout) as client:
                resp = client.post(ENDPOINT, json=body, headers=headers)
        except httpx.HTTPError as exc:
            # Connection/timeout error — transient. Retry unless out of attempts
            # or out of time budget.
            wait = _retry_wait(None, attempt)
            if _can_retry(attempt, wait):
                logger.warning(
                    "OpenRouter connection error (attempt %d/%d): %s; retrying in %.1fs",
                    attempt, _MAX_ATTEMPTS, redact(str(exc)), wait,
                )
                time.sleep(wait)
                continue
            raise RuntimeError(f"OpenRouter call failed: {redact(str(exc))}") from None

        if resp.status_code in _RETRYABLE_STATUS:
            wait = _retry_wait(resp, attempt)
            if _can_retry(attempt, wait):
                logger.warning(
                    "OpenRouter HTTP %s (attempt %d/%d); retrying in %.1fs",
                    resp.status_code, attempt, _MAX_ATTEMPTS, wait,
                )
                time.sleep(wait)
                continue
            # Out of attempts/budget — fall through to raise below.

        # Either a success, a non-retryable error, or the final attempt.
        try:
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            if resp.status_code == 429:
                # Distinct, clean message so the report's tool_failure entry is
                # human-readable rather than a redacted httpx URL blob.
                raise RateLimitError(
                    "OpenRouter rate limit (HTTP 429) persisted after "
                    f"{_MAX_ATTEMPTS} attempts — the allow-listed provider(s) are "
                    "at capacity for this model. Retry later, widen "
                    "OPENROUTER_ONLY_PROVIDERS, or enable fallbacks."
                ) from None
            raise RuntimeError(f"OpenRouter call failed: {redact(str(exc))}") from None

        if resp.headers.get("content-length"):
            try:
                if int(resp.headers["content-length"]) > MAX_HTTP_RESPONSE_BYTES:
                    raise RuntimeError("OpenRouter response exceeds size cap.")
            except ValueError:
                pass

        # Parse the API envelope INSIDE the retry loop: a saturated provider can
        # drop a large response mid-body, leaving a truncated/malformed JSON
        # envelope. That is transient, so retry it rather than let a raw
        # json.JSONDecodeError escape call_llm (it surfaced to the user as an
        # opaque "Template could not be parsed (JSONDecodeError)"). On exhaustion
        # we raise a clean, explanatory RuntimeError instead.
        try:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, ValueError, KeyError, IndexError, TypeError) as exc:
            wait = _retry_wait(None, attempt)
            if _can_retry(attempt, wait):
                logger.warning(
                    "OpenRouter returned an incomplete/unparseable response body "
                    "(attempt %d/%d: %s); retrying in %.1fs",
                    attempt, _MAX_ATTEMPTS, type(exc).__name__, wait,
                )
                time.sleep(wait)
                continue
            raise RuntimeError(
                "OpenRouter returned an incomplete or malformed response body "
                f"after {_MAX_ATTEMPTS} attempts — the allow-listed provider "
                "likely dropped a large response mid-stream. Retry later, or "
                "reduce the requested output size."
            ) from None
        # An EMPTY content on a JSON call is always useless (a reasoning model
        # that spent its whole token budget thinking, or a provider glitch) —
        # retry it here rather than bubbling "" up to a JSON-parse failure.
        if json_mode and not (content or "").strip():
            wait = _retry_wait(None, attempt)
            if _can_retry(attempt, wait):
                logger.warning(
                    "OpenRouter returned empty content on a JSON call "
                    "(attempt %d/%d); retrying in %.1fs",
                    attempt, _MAX_ATTEMPTS, wait,
                )
                time.sleep(wait)
                continue
        return content if content is not None else ""

    assert resp is not None  # the loop always returns or raises
    raise RuntimeError("OpenRouter call failed: no response received.")


# Default OUTPUT-token budget for structured (JSON) calls. Enough for a full QA
# rule set / issue list without cutting it off mid-array, but NOT so large that a
# saturated provider spends minutes generating it (an 8000-token target made the
# `template` call overrun the read timeout and grind through retries — it looked
# like the run hung at tool 2). Truncation is still repaired downstream.
_JSON_MAX_TOKENS = int(os.environ.get("LLM_JSON_MAX_TOKENS", "4000"))

_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.I)


def _strip_fences(raw: str) -> str:
    """Drop a leading/trailing ```json … ``` markdown fence if present."""
    return _CODE_FENCE_RE.sub("", raw).strip()


def _repair_truncated_json(raw: str) -> str | None:
    """Best-effort recovery of a JSON object that was cut off mid-output.

    A large rule set / issue list often gets truncated when the model hits its
    output-token limit, leaving an unterminated array like
    ``{"rules":[{...},{...},{"id":"R30","rul``. Rather than lose the ENTIRE
    payload (and every rule that DID come through), we walk from the first ``{``
    tracking string state and bracket depth, cut back to the end of the last
    COMPLETE element, and append the closing brackets. Returns a parseable string
    or None if nothing salvageable is found.
    """
    start = raw.find("{")
    if start < 0:
        return None
    s = raw[start:]
    in_str = False
    esc = False
    stack: list[str] = []
    last_safe: int | None = None  # index (exclusive) just past a complete element
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
            if not stack:
                # The root object closed cleanly — nothing to repair.
                candidate = s[: i + 1]
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    break
            # A nested element (a rule/issue object, or a value) just closed;
            # this is a safe point to truncate to if the tail is incomplete.
            last_safe = i + 1
        elif ch == "," and len(stack) >= 1:
            last_safe = i  # end of a complete element, before the comma
    if last_safe is None:
        return None
    head = s[:last_safe]
    # Re-derive the open-bracket stack for the salvaged head and close it.
    depth_stack: list[str] = []
    in_str = False
    esc = False
    for ch in head:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            depth_stack.append(ch)
        elif ch in "}]":
            if depth_stack:
                depth_stack.pop()
    closer = "".join("}" if b == "{" else "]" for b in reversed(depth_stack))
    candidate = head + closer
    try:
        json.loads(candidate)
        return candidate
    except json.JSONDecodeError:
        return None


def _loads_lenient(raw: str) -> dict | None:
    """Parse model output into a dict, tolerating fences, prose and truncation."""
    for text in (raw, _strip_fences(raw)):
        if not text:
            continue
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                obj = json.loads(match.group(0))
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
    repaired = _repair_truncated_json(raw)
    if repaired:
        try:
            obj = json.loads(repaired)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return None


def call_llm_json(prompt: str, system: str | None = None, temperature: float = 0.0,
                  max_prompt_bytes: int = MAX_TEXT_BYTES,
                  max_tokens: int | None = None,
                  read_timeout: float | None = None,
                  retry_budget: float | None = None,
                  model: str | None = None,
                  images: list[str] | None = None,
                  disable_reasoning: bool = False) -> dict:
    token_budget = max_tokens if max_tokens is not None else _JSON_MAX_TOKENS
    raw = call_llm(prompt, system=system, json_mode=True, temperature=temperature,
                   max_prompt_bytes=max_prompt_bytes, max_tokens=token_budget,
                   read_timeout=read_timeout, retry_budget=retry_budget,
                   model=model, images=images, disable_reasoning=disable_reasoning)
    obj = _loads_lenient(raw)
    if obj is not None:
        return obj
    # One clean retry: a single malformed / truncated response shouldn't sink a
    # whole rule set or issue list. The client already retries transport errors;
    # this covers a one-off bad body.
    logger.warning("LLM returned unparseable JSON (%d chars); retrying once.", len(raw or ""))
    raw = call_llm(prompt, system=system, json_mode=True, temperature=temperature,
                   max_prompt_bytes=max_prompt_bytes, max_tokens=token_budget,
                   read_timeout=read_timeout, retry_budget=retry_budget,
                   model=model, images=images, disable_reasoning=disable_reasoning)
    obj = _loads_lenient(raw)
    if obj is not None:
        return obj
    if not (raw or "").strip():
        # An empty body after every retry (a provider glitch, or a reasoning
        # model that spent its whole output budget thinking). Say so plainly —
        # the old message ended in a blank line and read as a mystery.
        raise RuntimeError(
            "LLM returned an EMPTY response after retries — the provider "
            "glitched or spent the output-token budget. Retrying the run "
            "usually clears it."
        )
    raise RuntimeError(f"LLM did not return valid JSON:\n{redact(raw)[:500]}")
