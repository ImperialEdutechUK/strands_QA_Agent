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
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from ..llm_client import call_llm_json
from ..security import (
    ALLOWED_IMAGE_SUFFIXES,
    MAX_HTTP_RESPONSE_BYTES,
    UnsafeURLError,
    safe_resolve_template,
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


# ---------------------------------------------------------------------------
# Uploaded specification document (preferred over web search when provided)
# ---------------------------------------------------------------------------

def _extract_spec_text(path: Path) -> str:
    """Pull plain text out of an uploaded specification (PDF / DOCX / image).

    PDFs are read text-only (a regulated spec stores its numbers as selectable
    text), so a 60+ page sheet doesn't spend minutes OCR'ing every page image.
    """
    suffix = path.suffix.lower()
    if suffix in ALLOWED_IMAGE_SUFFIXES:
        from .template_tool import _extract_image
        return _extract_image(path)
    if suffix == ".docx":
        from .template_tool import _extract_docx
        return _extract_docx(path)
    if suffix == ".pdf":
        import fitz  # PyMuPDF, already a project dependency
        chunks: list[str] = []
        with fitz.open(path) as doc:
            for page in doc:
                t = (page.get_text("text") or "").strip()
                if t:
                    chunks.append(t)
                if sum(len(c) for c in chunks) > 200_000:
                    break
        return "\n".join(chunks).strip()
    return ""


def _title_anchor(course_name: str) -> str:
    """A loose regex matching the course title in spec prose (sep/case tolerant)."""
    words = [w for w in re.findall(r"[A-Za-z0-9]+", course_name or "")
             if w.lower() not in {"qualifi", "rqf"}]
    if len(words) < 2:
        return ""
    return r"\s*[-\s]?\s*".join(re.escape(w) for w in words)


def _focus_spec_text(text: str, qualification_number: str, course_name: str,
                     cap: int = 14_000) -> str:
    """Return only the slice(s) of the sheet that belong to THIS qualification.

    One specification sheet often documents many qualifications (the QUALIFI IT
    sheet covers 12). We anchor on the qualification number (unique per variant)
    AND the exact variant title, then hand the LLM only those windows — so a
    sibling course's credits / GLH / TQT can never be mistaken for this one's.
    Returns "" when neither anchor is found (we never guess from the wrong rows).
    """
    if not text:
        return ""
    anchors = [re.escape(qualification_number)] if qualification_number else []
    title = _title_anchor(course_name)
    if title:
        anchors.append(title)
    windows: list[tuple[int, int]] = []
    for pat in anchors:
        try:
            for m in re.finditer(pat, text, re.I):
                windows.append((max(0, m.start() - 1500), min(len(text), m.end() + 2800)))
        except re.error:
            continue
    if not windows:
        return ""
    windows.sort()
    merged: list[list[int]] = []
    for s, e in windows:
        if merged and s <= merged[-1][1] + 200:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    parts: list[str] = []
    total = 0
    for s, e in merged:
        seg = text[s:e]
        if total + len(seg) > cap:
            seg = seg[: max(0, cap - total)]
        if seg:
            parts.append(seg)
            total += len(seg)
        if total >= cap:
            break
    return "\n...\n".join(parts).strip()


def _spec_from_document(document_path: str, course_name: str,
                        qualification_number: str, level: str,
                        awarding_body: str, variant: str) -> dict:
    """Read the official spec for THIS qualification from an uploaded document."""
    try:
        safe = safe_resolve_template(document_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("spec document rejected: %s", exc)
        return {"found": False, "specification": {}, "source_urls": [],
                "summary": "Specification document could not be read (rejected)."}
    try:
        full = _extract_spec_text(safe)
    except Exception as exc:  # noqa: BLE001
        logger.warning("spec document text extraction failed: %s", exc)
        full = ""
    if not full:
        return {"found": False, "specification": {}, "source_urls": [safe.name],
                "summary": "No text could be extracted from the specification document."}

    focused = _focus_spec_text(full, qualification_number, course_name)
    if not focused:
        return {
            "found": False, "specification": {}, "source_urls": [safe.name],
            "summary": ("The specification document does not contain this exact "
                        "qualification (no matching qualification number or "
                        "title found) — needs_spec rules are skipped."),
        }

    prompt = (
        f"{SCHEMA_INSTRUCTION}\n\n"
        f"REQUESTED QUALIFICATION — extract ONLY this one:\n"
        f"  course_name: {course_name or '(unknown)'}\n"
        f"  qualification_number: {qualification_number or '(unknown)'}\n"
        f"  level: {level or '(unknown)'}\n"
        f"  variant: {variant or '(unknown)'}\n"
        f"  awarding_body: {awarding_body or '(unknown)'}\n\n"
        f"SPECIFICATION DOCUMENT EXTRACT — this sheet may document SEVERAL "
        f"qualifications. Use ONLY the rows/section for the requested "
        f"qualification number and exact variant; if the only data you can see "
        f"belongs to a different level or variant (e.g. a plain 'Diploma' when an "
        f"'Extended Diploma' was requested), set found:false rather than "
        f"returning the wrong course's values:\n{focused}"
    )
    result = call_llm_json(prompt, system=SYSTEM)
    spec = result.get("specification") or {}
    found = bool(result.get("found")) and any(str(v).strip() for v in spec.values())
    return {
        "found": found,
        "specification": spec if isinstance(spec, dict) else {},
        "source_urls": [safe.name],
        "summary": (
            "Specification details read from the uploaded document."
            if found else
            "The uploaded document did not yield a confident match for this exact "
            "qualification variant — needs_spec rules are skipped."
        ),
    }


def lookup_specification(
    course_name: str,
    qualification_number: str = "",
    level: str = "",
    awarding_body: str = "",
    variant: str = "",
    document_path: str | None = None,
) -> dict:
    """Resolve a qualification spec and return structured parameters.

    If ``document_path`` is given (an uploaded specification sheet), the spec is
    read straight from that document — matched strictly to this qualification's
    number / variant — instead of web search. Otherwise it searches the web.
    Always returns a dict; on any failure the result has ``found: False`` and an
    empty specification so the caller silently skips the needs_spec rules.
    """
    course_name = (course_name or "").strip()
    if document_path:
        return _spec_from_document(
            document_path, course_name, qualification_number, level,
            awarding_body, variant,
        )
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
    try:
        result = call_llm_json(prompt, system=SYSTEM)
    except RuntimeError as exc:
        # Honour this tool's contract: any failure (notably an OpenRouter 429
        # rate limit) degrades to found:false so the agent silently skips the
        # needs_spec rules instead of aborting the whole run. We never fabricate
        # a spec from a failed call.
        logger.warning(
            "spec extraction LLM call failed (%s: %s) — returning found:false",
            type(exc).__name__, str(exc)[:160],
        )
        return {
            "found": False, "specification": {}, "source_urls": used,
            "summary": "Specification pages were fetched but the extraction step "
                       "could not complete (LLM unavailable); needs_spec rules "
                       "are skipped.",
        }
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
