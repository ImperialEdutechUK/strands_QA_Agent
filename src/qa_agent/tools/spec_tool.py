"""Web-search lookup of an official Qualification Specification.

Several checklist items can only be verified by comparing the course page
against the awarding body's official specification (course name, level,
qualification number, credit equivalency, GLH/TQT, accreditation status,
awarding body, entry requirements, access duration). The agent isn't given
that document, so this tool finds it: it searches the web for the
qualification (by course name + qualification number + any other hints),
fetches the most relevant result pages, and uses the LLM to distil the
structured spec parameters that compliance then checks against.

Search is keyless on purpose — it scrapes DuckDuckGo's HTML endpoint via the
httpx dependency we already have, so no extra API key/library is required. If
search or fetching fails, the tool degrades gracefully (``found: False``) and
the agent flags the spec-dependent items for manual verification instead.
"""

from __future__ import annotations

import logging
import re
from html import unescape
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from ..llm_client import call_llm_json
from ..security import (
    MAX_HTTP_RESPONSE_BYTES,
    UnsafeURLError,
    validate_public_url,
)
from .web_tools import USER_AGENT

logger = logging.getLogger(__name__)

_DDG_HTML_ENDPOINT = "https://html.duckduckgo.com/html/"
_SEARCH_TIMEOUT = httpx.Timeout(connect=10.0, read=20.0, write=15.0, pool=10.0)

# Awarding-body / regulator / endorsing-body domains we trust most for a
# specification. Results from these are ranked first, but other results are
# still used as a fallback. Endorsing bodies (Quality Licence Scheme, CPD) are
# included because many course pages are endorsed rather than Ofqual-regulated —
# for those there is no Ofqual specification, so the QLS/CPD listing is the
# closest thing to an official spec.
_PREFERRED_DOMAINS = (
    "ofqual.gov.uk", "register.ofqual.gov.uk", "gov.uk",
    "cityandguilds.com", "pearson.com", "qualifi.net", "tquk.org",
    "highfield.co.uk", "othm.org.uk", "abma.uk.com", "managers.org.uk",
    "i-l-m.com", "ncfe.org.uk", "open.ac.uk",
    # Endorsing bodies for non-regulated provision.
    "qualitylicencescheme.co.uk", "qualitylicencescheme.com", "cpduk.co.uk",
    "cpdgroup.co.uk", "thecpdgroup.co.uk",
)

_RESULT_HREF_RE = re.compile(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"', re.I)
_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[\s\S]*?</\1>", re.I)


def _decode_ddg_href(href: str) -> str:
    """DuckDuckGo HTML wraps results as `//duckduckgo.com/l/?uddg=<encoded>`."""
    href = unescape(href)
    if href.startswith("//"):
        href = "https:" + href
    try:
        q = parse_qs(urlparse(href).query)
    except ValueError:
        return href
    if "uddg" in q and q["uddg"]:
        return unquote(q["uddg"][0])
    return href


def _domain_rank(url: str) -> int:
    host = urlparse(url).netloc.lower()
    for i, dom in enumerate(_PREFERRED_DOMAINS):
        if host.endswith(dom):
            return i
    return len(_PREFERRED_DOMAINS) + 1


def _ddg_search(query: str, max_results: int = 6) -> list[str]:
    """Return de-duplicated result URLs for `query`, preferred domains first."""
    try:
        with httpx.Client(timeout=_SEARCH_TIMEOUT, follow_redirects=True,
                          headers={"User-Agent": USER_AGENT}) as client:
            resp = client.post(_DDG_HTML_ENDPOINT, data={"q": query})
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        logger.info("spec search failed for %r: %s", query[:80], exc)
        return []

    urls: list[str] = []
    seen: set[str] = set()
    for raw in _RESULT_HREF_RE.findall(resp.text):
        url = _decode_ddg_href(raw)
        try:
            url = validate_public_url(url)
        except UnsafeURLError:
            continue
        if url in seen:
            continue
        seen.add(url)
        urls.append(url)
    urls.sort(key=_domain_rank)
    return urls[:max_results]


def _pdf_to_text(data: bytes, cap: int) -> str:
    """Extract text from a PDF spec (awarding-body specifications are often PDFs)."""
    try:
        import fitz  # PyMuPDF, already a project dependency
    except Exception:  # noqa: BLE001
        return ""
    chunks: list[str] = []
    try:
        with fitz.open(stream=data, filetype="pdf") as doc:
            for page in doc:
                chunks.append(page.get_text("text") or "")
                if sum(len(c) for c in chunks) >= cap:
                    break
    except Exception as exc:  # noqa: BLE001
        logger.debug("spec PDF parse failed: %s", exc)
        return ""
    return re.sub(r"\s+", " ", " ".join(chunks)).strip()[:cap]


def _fetch_page_text(url: str, cap: int = 12_000) -> str:
    """Fetch a page (HTML or PDF) and return cleaned text, capped."""
    try:
        safe = validate_public_url(url)
    except UnsafeURLError:
        return ""
    try:
        with httpx.Client(timeout=_SEARCH_TIMEOUT, follow_redirects=True,
                          headers={"User-Agent": USER_AGENT}) as client:
            resp = client.get(safe)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        logger.debug("spec page fetch failed for %s: %s", safe, exc)
        return ""
    if len(resp.content) > MAX_HTTP_RESPONSE_BYTES:
        return ""
    content_type = resp.headers.get("content-type", "").lower()
    if "pdf" in content_type or safe.lower().endswith(".pdf"):
        return _pdf_to_text(resp.content, cap)
    if "html" not in content_type:
        return ""
    text = _SCRIPT_STYLE_RE.sub(" ", resp.text)
    text = _TAG_RE.sub(" ", text)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:cap]


SYSTEM = (
    "You extract the official specification of a regulated UK qualification "
    "from search-result page text. Report ONLY values clearly supported by the "
    "provided text — never guess. If a field is not stated, use an empty string. "
    "VARIANT MATCH IS CRITICAL: the requested qualification has a specific level "
    "and a specific variant (e.g. 'Extended Diploma' is a DIFFERENT, larger "
    "qualification than a plain 'Diploma' — typically double the credits, GLH, "
    "TQT and duration). Only treat the text as the requested spec when the title "
    "matches the SAME level AND the SAME variant word ('Extended', 'Award', "
    "'Certificate', 'Diploma', 'Extended Diploma'), and ideally the same "
    "qualification number. If the text describes a different variant or level, "
    "set found:false — returning a near-miss spec is worse than returning none."
)

SCHEMA_INSTRUCTION = """Return a JSON object of exactly this shape:
{
  "found": true | false,
  "specification": {
    "course_name": "<official title, or empty>",
    "level": "<e.g. Level 4, or empty>",
    "qualification_number": "<e.g. 603/1234/5, or empty>",
    "accreditation_status": "<e.g. Regulated by Ofqual, or empty>",
    "credit_equivalency": "<e.g. 120 credits, or empty>",
    "glh": "<Guided Learning Hours, or empty>",
    "tqt": "<Total Qualification Time, or empty>",
    "awarding_body": "<e.g. Qualifi, or empty>",
    "access_duration": "<if stated, or empty>",
    "notes": "<anything else relevant, or empty>"
  }
}
Set "found" to false if the text does not describe the requested qualification,
OR if it describes a different level/variant (e.g. a plain 'Diploma' when an
'Extended Diploma' was requested). Output ONLY the JSON object."""


def lookup_specification(
    course_name: str,
    qualification_number: str = "",
    level: str = "",
    awarding_body: str = "",
) -> dict:
    """Search the web for a qualification spec and return structured parameters.

    Always returns a dict; on any failure the result has ``found: False`` and an
    empty specification so the caller can fall back to manual verification.
    """
    course_name = (course_name or "").strip()
    if not course_name and not qualification_number:
        return {"found": False, "specification": {}, "source_urls": [],
                "summary": "No course name or qualification number provided."}

    # Lead with the qualification number when we have one — it uniquely
    # identifies the exact qualification VARIANT (an Extended Diploma and a plain
    # Diploma share a name but have different numbers), so it avoids fetching the
    # wrong-sized spec. Fall back to the name-based query if that finds nothing.
    urls: list[str] = []
    if qualification_number:
        urls = _ddg_search(f'"{qualification_number}" qualification specification')
        if not urls:
            urls = _ddg_search(f"{qualification_number} Ofqual qualification")
    if not urls:
        terms = [t for t in (course_name, qualification_number, awarding_body) if t]
        urls = _ddg_search(" ".join(terms) + " qualification specification")
    # Endorsed (non-Ofqual) courses — e.g. a Quality Licence Scheme "QLS-03616"
    # code — won't be in the Ofqual register, so a plain code search finds the
    # endorsing body's listing instead.
    if not urls and qualification_number:
        urls = _ddg_search(f'"{qualification_number}" course')
    if not urls and course_name:
        urls = _ddg_search(f"{course_name} endorsed course specification")

    pages: list[str] = []
    used: list[str] = []
    for url in urls:
        text = _fetch_page_text(url)
        if len(text) >= 200:
            pages.append(f"[source: {url}]\n{text}")
            used.append(url)
        if len(pages) >= 3:
            break

    if not pages:
        return {
            "found": False, "specification": {}, "source_urls": used,
            "summary": "Could not locate a usable specification page via web search.",
        }

    prompt = (
        f"{SCHEMA_INSTRUCTION}\n\n"
        f"REQUESTED QUALIFICATION:\n"
        f"  course_name: {course_name or '(unknown)'}\n"
        f"  qualification_number: {qualification_number or '(unknown)'}\n"
        f"  level: {level or '(unknown)'}\n"
        f"  awarding_body: {awarding_body or '(unknown)'}\n\n"
        f"SEARCH RESULT PAGE TEXT:\n" + "\n\n".join(pages)[:14_000]
    )
    result = call_llm_json(prompt, system=SYSTEM)
    spec = result.get("specification") or {}
    found = bool(result.get("found")) and any(str(v).strip() for v in spec.values())
    return {
        "found": found,
        "specification": spec if isinstance(spec, dict) else {},
        "source_urls": used,
        "summary": (
            "Specification details located via web search."
            if found else
            "Search pages were fetched but no matching specification was confirmed."
        ),
    }
