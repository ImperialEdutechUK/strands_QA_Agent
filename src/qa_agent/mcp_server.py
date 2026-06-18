"""MCP server exposing the QA tools over streamable-HTTP transport.

Run with: python -m qa_agent.mcp_server

Security defaults:
  * Binds to 127.0.0.1 (override with MCP_HOST=0.0.0.0 if you really need it).
  * Optional bearer-token auth via MCP_AUTH_TOKEN.
  * URL/path inputs are validated by the underlying tool implementations.
"""

from __future__ import annotations

import asyncio
import logging
import os

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from .logging_config import configure_logging
from .security import constant_time_equals, redact
from .tools.compliance_tool import check_compliance
from .tools.reasoning_tool import review_findings
from .tools.spell_tool import check_spelling
from .tools.template_tool import analyse_template, analyse_template_text
from .tools.web_tools import capture_excerpts, scrape_page, take_screenshot

load_dotenv()
configure_logging()
logger = logging.getLogger(__name__)

_HOST = os.environ.get("MCP_HOST", "127.0.0.1")
_PORT = int(os.environ.get("MCP_PORT", "3001"))
_AUTH_TOKEN = os.environ.get("MCP_AUTH_TOKEN", "").strip()

mcp = FastMCP("QA Tools", host=_HOST, port=_PORT)


def _check_auth(token: str | None) -> None:
    """If MCP_AUTH_TOKEN is configured, every tool call must present it."""
    if not _AUTH_TOKEN:
        return
    if not token or not constant_time_equals(token, _AUTH_TOKEN):
        raise PermissionError("Invalid or missing auth token.")


@mcp.tool()
async def scrape(url: str, auth_token: str | None = None) -> dict:
    """Fetch a web page and return its title, body text, headings, links, and images."""
    _check_auth(auth_token)
    # sync_playwright() refuses to run inside an active asyncio loop (FastMCP/uvicorn),
    # so push the sync work onto a worker thread.
    return await asyncio.to_thread(scrape_page, url)


@mcp.tool()
async def screenshot(url: str, selector: str | None = None, full_page: bool = True,
                     auth_token: str | None = None) -> dict:
    """Capture a base64-encoded PNG of a page (or a specific CSS selector)."""
    _check_auth(auth_token)
    img = await asyncio.to_thread(take_screenshot, url, selector, full_page)
    return {"img": img}


@mcp.tool()
async def evidence(url: str, excerpts: list[str], auth_token: str | None = None) -> dict:
    """Open `url` once and return focused per-excerpt screenshots.

    Returns {"shots": {excerpt: base64_png, ...}}. Excerpts whose element
    cannot be located on the page are silently omitted.
    """
    _check_auth(auth_token)
    shots = await asyncio.to_thread(capture_excerpts, url, excerpts)
    return {"shots": shots}


@mcp.tool()
def spell(text: str, auth_token: str | None = None) -> dict:
    """Run a UK English spelling/grammar check and return structured issues."""
    _check_auth(auth_token)
    return check_spelling(text)


@mcp.tool()
def template(document_path: str | None = None, text: str | None = None,
             image_path: str | None = None,
             auth_token: str | None = None) -> dict:
    """Interpret a QA template into a rule list.

    The template can be supplied as:
      * `text` — inline rules as a string;
      * `document_path` — filesystem path to a PDF, Word `.docx`, or image
        (PNG/JPEG/WebP/etc). PDFs and DOCX files have their text extracted
        and every embedded image OCR'd; images are OCR'd directly.
      * `image_path` — backwards-compatible alias for `document_path`.

    Provide exactly one of {text, document_path/image_path}.
    """
    _check_auth(auth_token)
    if text:
        return analyse_template_text(text)
    path = document_path or image_path
    if not path:
        raise ValueError("Provide either `document_path`, `image_path`, or `text`.")
    return analyse_template(path)


@mcp.tool()
def compliance(page_text: str, headings: list, rules: list,
               price_candidates: list[str] | None = None,
               auth_token: str | None = None) -> dict:
    """Audit page text + headings against a list of QA template rules."""
    _check_auth(auth_token)
    return check_compliance(
        page_text=page_text,
        headings=headings,
        rules=rules,
        price_candidates=price_candidates,
    )


@mcp.tool()
def reason(instructions: str, issues: list | None = None,
           page_summary: dict | None = None,
           auth_token: str | None = None) -> dict:
    """Self-review: did the agent satisfy the user's instructions?

    Pass the original instructions string, the list of issues collected so far
    (in the same order they will appear in the final report), and a small page
    summary dict ({title, url, headings, template_summary}). Returns a verdict
    (`PASS`/`PARTIAL`/`FAIL`), a summary, what was done well, gaps, and a
    per-issue judgement (supported / weak / spurious).
    """
    _check_auth(auth_token)
    return review_findings(
        instructions=instructions,
        issues=issues or [],
        page_summary=page_summary or {},
    )


if __name__ == "__main__":
    auth_state = "ENABLED" if _AUTH_TOKEN else "DISABLED (set MCP_AUTH_TOKEN to enable)"
    logger.info("MCP Server starting on http://%s:%s/mcp (auth: %s)", _HOST, _PORT, auth_state)
    try:
        mcp.run(transport="streamable-http")
    except Exception as exc:
        logger.error("MCP server crashed: %s", redact(str(exc)))
        raise
