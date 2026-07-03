"""Strands Agent that connects to the QA MCP server and orchestrates a QA run."""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from typing import Any

from mcp.client.streamable_http import streamablehttp_client
from strands import Agent
from strands.hooks import (
    AfterToolCallEvent,
    BeforeInvocationEvent,
    BeforeToolCallEvent,
    HookProvider,
    HookRegistry,
    MessageAddedEvent,
)
from strands.tools.mcp import MCPClient

from .llm import build_model

logger = logging.getLogger(__name__)

# Dedicated logger for the live execution trace. Uses INFO so it shows up under
# the default LOG_LEVEL, but lives on its own logger so a user who wants only
# the trace can filter by name. The standard logging machinery routes it to
# the configured StreamHandler (stderr) — that's what reaches the terminal
# reliably under uvicorn / threads / Windows.
trace = logging.getLogger("qa_agent.trace")

# ----------------------------------------------------------------------------
# Terminal logging: tool-call lifecycle + streamed model reasoning
# ----------------------------------------------------------------------------

# Truncate long blobs in the terminal so screenshots / scrape bodies don't
# flood the console. The full data is still passed through to the LLM / report.
_LOG_VALUE_LIMIT = 240


def _short(value: Any) -> str:
    try:
        s = json.dumps(value, default=str, ensure_ascii=False)
    except Exception:
        s = str(value)
    s = s.replace("\n", " ")
    return s if len(s) <= _LOG_VALUE_LIMIT else f"{s[:_LOG_VALUE_LIMIT]}…(+{len(s) - _LOG_VALUE_LIMIT} chars)"


def _writeln(line: str) -> None:
    """Emit one line of the live trace via the standard logging machinery.

    Going through `logger.info` (rather than `print`) is what makes these
    lines actually show up in the user's terminal when the web server is
    running under uvicorn — uvicorn drives its own log handlers, but stray
    `print(file=sys.stderr)` from worker threads on Windows is unreliable.
    """
    trace.info("%s", line)


class ToolCallLoggerHook(HookProvider):
    """Logs every MCP tool invocation with its inputs, output, and duration.

    Output goes to stderr so it stays visible in the terminal but doesn't
    contaminate the agent's stdout (which the CLI parses as JSON).
    """

    def __init__(self) -> None:
        self._call_n = 0
        self._starts: dict[str, float] = {}

    def register_hooks(self, registry: HookRegistry, **_kwargs: Any) -> None:
        registry.add_callback(BeforeInvocationEvent, self._on_invocation_start)
        registry.add_callback(BeforeToolCallEvent, self._on_tool_start)
        registry.add_callback(AfterToolCallEvent, self._on_tool_end)
        registry.add_callback(MessageAddedEvent, self._on_message)

    def _on_invocation_start(self, _ev: BeforeInvocationEvent) -> None:
        self._call_n = 0
        self._starts.clear()
        _writeln("[AGENT] starting QA run")

    def _on_tool_start(self, ev: BeforeToolCallEvent) -> None:
        self._call_n += 1
        tool_use = ev.tool_use or {}
        name = tool_use.get("name") or (ev.selected_tool and ev.selected_tool.tool_name) or "?"
        tool_id = tool_use.get("toolUseId") or f"call_{self._call_n}"
        self._starts[tool_id] = time.monotonic()
        raw_input = tool_use.get("input")
        if isinstance(raw_input, dict):
            redacted = {k: ("<token>" if k == "auth_token" else v) for k, v in raw_input.items()}
        else:
            redacted = raw_input
        _writeln(f"[TOOL #{self._call_n}] -> {name}  args={_short(redacted)}")

    def _on_tool_end(self, ev: AfterToolCallEvent) -> None:
        tool_use = ev.tool_use or {}
        name = tool_use.get("name") or (ev.selected_tool and ev.selected_tool.tool_name) or "?"
        tool_id = tool_use.get("toolUseId") or ""
        started = self._starts.pop(tool_id, None)
        elapsed = (time.monotonic() - started) if started else 0.0
        if ev.exception is not None:
            _writeln(f"[TOOL #{self._call_n}] x  {name}  FAILED in {elapsed:.2f}s: {ev.exception!r}")
            return
        result = ev.result or {}
        # ToolResult is a TypedDict with `content` (a list of content blocks).
        # Pull the first text block out so the log shows what the agent saw.
        preview: Any = result
        try:
            blocks = result.get("content") or []  # type: ignore[union-attr]
            if blocks:
                first = blocks[0]
                if isinstance(first, dict) and "text" in first:
                    preview = first["text"]
        except Exception:
            pass
        status = (result.get("status") if isinstance(result, dict) else None) or "ok"
        _writeln(f"[TOOL #{self._call_n}] <- {name}  ({status}, {elapsed:.2f}s)  result={_short(preview)}")

    def _on_message(self, ev: MessageAddedEvent) -> None:
        # Surface every assistant text turn so the user can see the agent's
        # planning between tool calls. Skip user/system/tool-result turns
        # because they're already visible from BeforeToolCall logs.
        msg = getattr(ev, "message", None) or {}
        if msg.get("role") != "assistant":
            return
        for block in msg.get("content") or []:
            text = block.get("text") if isinstance(block, dict) else None
            if text and text.strip():
                _writeln(f"[REASON] {text.strip()}")


_stream_buffer: list[str] = []


def _streaming_text_handler(**kwargs: Any) -> None:
    """Buffer streamed model text deltas and flush them as whole lines.

    Strands sends text in fragments. Writing each fragment directly to stderr
    looked broken in mixed-thread output and was sometimes swallowed by
    uvicorn — instead we coalesce fragments and emit a single `[STREAM] …`
    log line per newline (or on completion). The hook's `[REASON]` log
    already captures the final message so this is just for live feedback.
    """
    chunk = kwargs.get("data") or ""
    complete = bool(kwargs.get("complete"))

    if chunk:
        _stream_buffer.append(chunk)
        if "\n" in chunk:
            _flush_stream_buffer()
    if complete:
        _flush_stream_buffer()


def _flush_stream_buffer() -> None:
    if not _stream_buffer:
        return
    text = "".join(_stream_buffer).strip()
    _stream_buffer.clear()
    if text:
        # Multi-line bursts come through as a single fragment sometimes; split
        # them so each terminal line gets its own [STREAM] prefix.
        for line in text.splitlines():
            line = line.strip()
            if line:
                trace.info("[STREAM] %s", line[:500])

# OpenRouter (especially the free DeepSeek tier) drops streamed responses
# occasionally. These substrings are the ones we've seen in real failures.
_TRANSIENT_ERROR_MARKERS = (
    "network connection lost",
    "provider returned error",
    "remote protocol error",
    "connection reset",
    "incomplete response",
    "stream error",
    "idle timeout",
    "upstream",
    "timed out",
    "timeout exceeded",
    "503",
    "502",
    "504",
    "429",
)


def _is_transient(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in _TRANSIENT_ERROR_MARKERS)


def invoke_with_retry(agent: Agent, prompt: str, *, max_attempts: int = 2,
                      backoff_seconds: float = 4.0):
    """Run the agent, retrying on transient OpenRouter / streaming failures.

    Each retry rebuilds the message context — Strands' Agent keeps its own
    history across calls, so a retried `agent(prompt)` re-uses any tool calls
    that already succeeded and only re-issues the steps that didn't.

    Kept deliberately low (2 attempts: the initial try plus one retry after a
    4 s backoff) so a genuinely failing run doesn't re-drive the whole tool
    chain several times and rack up token cost. The per-call LLM client already
    retries transient 429s/5xx internally, so this outer retry only needs to
    cover a single hard streaming drop, not sustained provider trouble.
    """
    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return agent(prompt)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt == max_attempts or not _is_transient(exc):
                raise
            wait = backoff_seconds * (2 ** (attempt - 1))
            logger.warning(
                "agent attempt %d/%d failed (%s: %s); retrying in %.1fs",
                attempt, max_attempts, type(exc).__name__, str(exc)[:160], wait,
            )
            time.sleep(wait)
    if last_exc:
        raise last_exc
    return None

SYSTEM_PROMPT = """You are a course-page QA agent. Be silent and efficient.

Strict rules — these exist because every LLM call costs money:

  * DO NOT narrate your plan, your reasoning, or your progress between tool calls.
  * Call each tool AT MOST ONCE per run. If a tool returns an error, do NOT
    retry it with the same arguments. Record the failure in the JSON and move on.
  * Use exactly this fixed order — skip a step if its inputs are missing:
      1. `extract(url)` — the primary extraction stage. It renders the page in a
         browser and returns a SMALL summary: `extraction_id`, `stats`,
         `general_content`, `price_candidates`, `course_identity` (course_name,
         qualification_number, level, variant, awarding_body — use these in
         step 4), evidence counts, `extraction_warnings`, and a `report_path`
         (the FULL evidence report — banners, every image, OCR text, claims,
         screenshots — written to disk). The full page text and all banner /
         image / FAQ evidence are cached server-side under `extraction_id` and
         read directly by `spell` / `compliance`. CRITICAL: do NOT try to copy
         page text or evidence into later tool calls — just pass the small
         `extraction_id`. Do NOT read `report_path`.
      2. `template(...)`        — only if a template was provided. The
         template may be inline `text`, OR a `document_path` to an image,
         PDF, or Word `.docx` (whichever the user supplied — pass the path
         through verbatim, don't try to read or parse the file yourself). It
         returns a small `{template_id, summary, rule_count, needs_spec}`; the
         full rule list is cached server-side under `template_id`. Keep that id
         for step 5 and do NOT try to reproduce or copy the rules yourself.
      3. `spell(extraction_id)` — pass the `extraction_id` from step 1 (the
         server resolves the page text). Do NOT paste the page text here.
      4. `spec_lookup(course_name, ...)` — ONLY if step 2's result has
         `needs_spec: true`. Take the arguments from step 1's `course_identity`:
         pass `course_name` EXACTLY as given (keep words like "Extended" — a
         "Level 5 Extended Diploma" is a DIFFERENT qualification from a "Level 5
         Diploma"), and ALWAYS pass `qualification_number` (e.g. 610/1675/5),
         `level` and `awarding_body` whenever `course_identity` provides them.
         The qualification number is what pins down the correct spec variant, so
         never omit it when it is present. If the user message includes a "QA
         specification file path", ALSO pass it as `document_path` and pass
         `variant` from `course_identity`: the spec is then read from that
         uploaded sheet (matched strictly to this qualification's number /
         variant — one sheet may list many courses) instead of web search. It
         web-searches for the official Qualification Specification and returns a
         `{found, specification, source_urls}` object. Keep that object to pass
         into the next step. Copy its `source_urls` into the report's
         `specification_source` field so the reviewer can see exactly which
         specification document you checked against. Skip this step entirely if
         no rule needs a spec.
      5. `compliance(template_id, extraction_id, specification)` — only if you
         ran step 2. Pass the `template_id` from step 2, the `extraction_id`
         from step 1, and (if you ran step 4) the whole object it returned as
         `specification`. The server resolves the full rule list from
         `template_id`, plus the page text, headings, price
         candidates and the high-priority banner/image evidence, so claims that
         live in banners or are baked into images (price, discount,
         accreditation, awarding body, guarantee, rating, urgency, etc.) are
         compared against the rules too, not just the body text. needs_spec
         rules are checked against `specification`; if it's missing or
         `found:false`, those items are SILENTLY SKIPPED — do NOT emit a
         "verify manually" / "check against the specification" issue. Verifying
         the page is your job, never the reader's. Do NOT paste any evidence or
         page text here.
      6. `evidence(url, [...])` — collect proof for EVERY issue. Pass ONE list
                                  containing the `excerpt` of every spell AND
                                  compliance issue (so each issue gets its own
                                  screenshot). The tool returns
                                  `{"shots": {excerpt: token}}` where each token
                                  is a SHORT opaque string of the form
                                  `evidence://<hash>`. Copy the token verbatim
                                  into the corresponding issue's `screenshot`
                                  field. An excerpt may be omitted from `shots`
                                  if no matching section was found on the page —
                                  in that case leave that issue's `screenshot`
                                  off. NEVER inline base64 image data — only the
                                  token.
        DO NOT call `screenshot` (the full-page tool) — it is wasteful here.
        DO NOT call `scrape` — `extract` (step 1) supersedes it; the page text,
        headings and price candidates it produces are cached server-side.
      7. `reason(instructions, issues, page_summary)` — ALWAYS call last.
         Pass the original user instructions verbatim, the issues array
         exactly as you will emit it (same order, same fields except you may
         drop `screenshot` to keep the payload small), and a small
         `page_summary` dict containing `title` and `h1` (from step 1's
         `general_content`), `url`, and `template_summary` if any. Put the
         returned object into the report under the top-level `reasoning` field
         — do NOT alter or summarise it.
  * After step 7 (or earlier if extract failed), output ONE JSON object and STOP.
    Do not write any prose before or after the JSON.

JSON schema:
  {
    "course_name": "<page title or 'QA run incomplete'>",
    "url": "<the input url>",
    "template_summary": "<from template tool, or null>",
    "specification_source": "<spec_lookup source_urls (the official spec document you checked), or null>",
    "issues": [
      {
        "type": "Spelling|Grammar|Punctuation|Style|Template",
        "severity": "Critical|Minor|Info",
        "ruleId": "<from compliance, else omit>",
        "excerpt": "<short quote>",
        "description": "<what is wrong>",
        "suggestion": "<how to fix>",
        "screenshot": "<evidence:// token from the evidence tool, or omit>"
      }
    ],
    "reasoning": "<the entire object returned by the reason tool>",
    "tool_failures": ["<tool_name: short reason>", ...]
  }

If MCP tools require an auth token, pass it as the `auth_token` argument.
Use UK English in all descriptions. Do not invent issues."""


def _client_factory(url: str):
    headers: dict[str, str] = {}
    token = os.environ.get("MCP_AUTH_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    def factory():
        # streamable HTTP client signature changed across mcp versions; pass
        # headers when supported, fall back gracefully.
        try:
            return streamablehttp_client(url, headers=headers or None)
        except TypeError:
            return streamablehttp_client(url)

    return factory


@contextmanager
def build_agent(mcp_url: str | None = None):
    url = mcp_url or os.environ.get("MCP_URL", "http://127.0.0.1:3001/mcp")
    client = MCPClient(_client_factory(url))
    with client:
        tools = client.list_tools_sync()
        _writeln(f"[AGENT] connected to MCP at {url}; {len(tools)} tools available: "
                 f"{', '.join(getattr(t, 'tool_name', '?') for t in tools)}")
        agent = Agent(
            model=build_model(),
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
            hooks=[ToolCallLoggerHook()],
            callback_handler=_streaming_text_handler,
        )
        yield agent, client


def build_user_prompt(url: str, template_path: str | None, template_text: str | None,
                      spec_path: str | None = None) -> str:
    parts = [f"Course URL: {url}"]
    if template_path:
        parts.append(f"QA template file path: {template_path}")
    if template_text:
        parts.append(f"QA template text:\n{template_text}")
    if spec_path:
        parts.append(
            f"QA specification file path: {spec_path}\n"
            f"(Pass this to spec_lookup as `document_path` for the needs_spec rules.)"
        )
    token = os.environ.get("MCP_AUTH_TOKEN", "").strip()
    if token:
        parts.append("All tool calls must include `auth_token` set to the configured MCP token.")
    parts.append("Run the full QA flow now and return the JSON report.")
    return "\n\n".join(parts)
