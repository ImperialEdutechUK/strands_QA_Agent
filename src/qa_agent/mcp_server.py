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

from .extraction import extract_page_summary, get_cached_extraction
from .logging_config import configure_logging
from .security import constant_time_equals, redact
from .tools.compliance_tool import check_compliance
from .tools.reasoning_tool import review_findings
from .tools.spec_tool import lookup_specification
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
async def extract(url: str, use_wordpress: bool = True, capture_screenshots: bool = True,
                  run_ocr: bool = True, auth_token: str | None = None) -> dict:
    """Layered QA-evidence extraction for a course page.

    Collects detailed, evidence-based website content for later comparison
    against QA template rules:

      * WordPress REST API metadata (title, slug, content, featured media,
        alt text, captions, modified date) where the site exposes it;
      * rendered-DOM content (JS-rendered text, lazy images, sliders, banners,
        accordions, tabs, page-builder sections);
      * images from <img>, <picture>/srcset, lazy data-* attributes, CSS/inline
        background images and page-builder backgrounds;
      * all carousel slides (not just the first), banners (hero, promotional,
        carousel, sidebar, popup, footer, trust/accreditation);
      * element screenshots as evidence and OCR of text baked into images;
      * per-element claim detection (price, discount, duration, certification,
        accreditation, awarding body, eligibility, guarantee, rating, urgency,
        learner numbers) and QA-priority scoring.

    The FULL structured report is written to disk (path returned as
    `report_path`); this tool returns a compact summary — counts, the
    high-priority banners/images, and any extraction warnings — to keep the
    agent's context small. Read `report_path` for the complete evidence.
    """
    _check_auth(auth_token)
    return await asyncio.to_thread(
        extract_page_summary, url,
        use_wordpress=use_wordpress,
        capture_screenshots=capture_screenshots,
        run_ocr=run_ocr,
    )


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
def spell(text: str | None = None, extraction_id: str | None = None,
          auth_token: str | None = None) -> dict:
    """Run a UK English spelling/grammar check and return structured issues.

    Pass `extraction_id` (from the `extract` tool) and the page text is resolved
    server-side — no need to copy the page text into this call. `text` is still
    accepted for direct use / backwards compatibility.
    """
    _check_auth(auth_token)
    if not text and extraction_id:
        cached = get_cached_extraction(extraction_id)
        if cached is None:
            raise ValueError(
                f"Unknown extraction_id '{extraction_id}'. Re-run `extract` first."
            )
        text = cached.get("page_text", "")
    return check_spelling(text or "")


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
async def spec_lookup(course_name: str, qualification_number: str = "",
                      level: str = "", awarding_body: str = "",
                      auth_token: str | None = None) -> dict:
    """Find the official Qualification Specification for a course via web search.

    Pass the course name plus any of `qualification_number`, `level`,
    `awarding_body` taken from the page. Searches the web (keyless), fetches the
    most relevant awarding-body / Ofqual pages, and returns structured spec
    parameters: `{found, specification:{level, qualification_number,
    accreditation_status, credit_equivalency, glh, tqt, awarding_body, ...},
    source_urls}`. Pass the whole returned object to `compliance` as
    `specification` so the needs_spec rules are checked against it. On failure
    it returns `found: false` — then the needs_spec rules are flagged for manual
    verification rather than guessed.
    """
    _check_auth(auth_token)
    return await asyncio.to_thread(
        lookup_specification, course_name,
        qualification_number=qualification_number,
        level=level,
        awarding_body=awarding_body,
    )


@mcp.tool()
def compliance(rules: list, extraction_id: str | None = None,
               page_text: str | None = None, headings: list | None = None,
               price_candidates: list[str] | None = None,
               banner_evidence: list | None = None,
               image_evidence: list | None = None,
               specification: dict | None = None,
               auth_token: str | None = None) -> dict:
    """Audit page text, headings, banners and images against QA template rules.

    Preferred usage: pass `rules` (from the `template` tool) and `extraction_id`
    (from the `extract` tool). The page text, headings, price candidates and the
    high-priority banner/image evidence are then resolved server-side, so claims
    that live in banners or are baked into images (via OCR) are compared against
    the rules too — without the agent having to copy any large blobs into this
    call. The explicit parameters remain available for direct use.
    """
    _check_auth(auth_token)
    if extraction_id:
        cached = get_cached_extraction(extraction_id)
        if cached is None:
            raise ValueError(
                f"Unknown extraction_id '{extraction_id}'. Re-run `extract` first."
            )
        page_text = page_text if page_text is not None else cached.get("page_text", "")
        headings = headings if headings is not None else cached.get("headings", [])
        price_candidates = price_candidates if price_candidates is not None else cached.get("price_candidates", [])
        banner_evidence = banner_evidence if banner_evidence is not None else cached.get("banner_evidence", [])
        image_evidence = image_evidence if image_evidence is not None else cached.get("image_evidence", [])
    return check_compliance(
        page_text=page_text or "",
        headings=headings or [],
        rules=rules,
        price_candidates=price_candidates,
        banner_evidence=banner_evidence,
        image_evidence=image_evidence,
        specification=specification,
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
