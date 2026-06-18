from __future__ import annotations

import base64
import hashlib
import logging
import re
import tempfile
from pathlib import Path
from typing import Callable, TypeVar

from io import BytesIO

from PIL import Image as PILImage
from playwright.sync_api import Page, TimeoutError as PWTimeoutError, sync_playwright

from ..security import validate_public_url

logger = logging.getLogger(__name__)

T = TypeVar("T")

NAV_TIMEOUT_MS = 45_000
LOAD_BEST_EFFORT_MS = 8_000
VIEWPORT = {"width": 1440, "height": 900}

# The full page text is fetched and stored locally, but the slice we hand back
# to the agent is capped — every byte we return ends up in the LLM context
# twice (once as a tool result, once when re-passed to spell/compliance), which
# makes long pages tip OpenRouter into "Network connection lost" mid-stream.
SCRAPE_TEXT_LLM_CAP = 8_000
SCRAPE_TEXT_HEAD_CHUNK = 6000
SCRAPE_TEXT_TAIL_CHUNK = 2000

PRICE_SELECTOR_CANDIDATES = (
    "[class*=price]",
    "[id*=price]",
    "[class*=cost]",
    "[id*=cost]",
    "[class*=fee]",
    "[id*=fee]",
    "[class*=sidebar]",
    "[id*=sidebar]",
    "aside",
)

_PRICE_RE = re.compile(
    r'''
    (?:£|GBP|EUR|€|USD|\$)\s?\d[\d,]*(?:\.\d{1,2})?
    |
    \d[\d,]*\s?(?:GBP|USD|EUR|pounds|pound|£|€|usd|eur|gbp)
    ''',
    re.I | re.X,
)
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36 QA-Agent"
)


def _new_browser_page(p):
    browser = p.chromium.launch()
    ctx = browser.new_context(
        viewport=VIEWPORT,
        ignore_https_errors=False,
        java_script_enabled=True,
        user_agent=USER_AGENT,
    )
    page = ctx.new_page()
    page.set_default_navigation_timeout(NAV_TIMEOUT_MS)
    return browser, page


# Hide fixed/sticky overlays (cookie banners, popups, chat widgets) so element
# screenshots show the actual content rather than a modal backdrop. We accept
# losing sticky headers in evidence shots — that's a fair trade for clarity.
_HIDE_OVERLAYS_JS = r"""
() => {
    // Targeted class/id selectors — these are the common overlay patterns
    // (cookie banners, modals, newsletter popups, chat widgets).
    const css = `
        [class*="popup" i]:not(body):not(html),
        [class*="modal" i][class*="open" i],
        [class*="overlay" i]:not(body):not(html),
        [class*="cookie" i]:not(body):not(html),
        [class*="consent" i]:not(body):not(html),
        [class*="newsletter" i]:not(body):not(html),
        [class*="livechat" i],
        [class*="chat-widget" i],
        [id*="cookie" i]:not(body):not(html),
        [id*="popup" i]:not(body):not(html),
        [aria-modal="true"],
        dialog[open] {
            display: none !important;
        }
        /* Many popup scripts lock scroll by setting overflow:hidden on body. */
        html, body {
            overflow: visible !important;
            position: static !important;
        }
    `;
    const style = document.createElement('style');
    style.textContent = css;
    document.head.appendChild(style);

    // Hide small fixed/sticky elements (sticky CTAs, mobile bars, chat bubbles).
    // Skip body/html (WordPress sets position:fixed on body to lock scroll for
    // popups — hiding body would nuke all content), and skip elements with
    // substantial text content (those are real article-level wrappers).
    document.querySelectorAll('*').forEach(el => {
        if (el === document.body || el === document.documentElement) return;
        const cs = getComputedStyle(el);
        if (cs.position !== 'fixed' && cs.position !== 'sticky') return;
        const textLen = (el.innerText || '').trim().length;
        if (textLen >= 800) return;  // probably real content, not an overlay
        el.style.setProperty('display', 'none', 'important');
    });
}
"""


def _navigate(page: Page, url: str) -> None:
    page.goto(url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
    try:
        page.wait_for_load_state("load", timeout=LOAD_BEST_EFFORT_MS)
    except PWTimeoutError:
        logger.info("load event did not fire within %sms — continuing", LOAD_BEST_EFFORT_MS)
    try:
        page.evaluate(_HIDE_OVERLAYS_JS)
    except Exception as exc:  # noqa: BLE001
        logger.debug("overlay hide failed (continuing): %s", exc)


def _cap_text(text: str) -> str:
    if len(text) <= SCRAPE_TEXT_LLM_CAP:
        return text
    return (
        text[:SCRAPE_TEXT_HEAD_CHUNK]
        + "\n\n... [content truncated; tail preserved] ...\n\n"
        + text[-SCRAPE_TEXT_TAIL_CHUNK:]
    )


def _extract_price_snippets(page: Page) -> list[str]:
    snippets: list[str] = []
    seen: set[str] = set()

    def push(text: str) -> None:
        normalized = text.strip()
        if len(normalized) >= 4 and normalized not in seen:
            seen.add(normalized)
            snippets.append(normalized)

    for selector in PRICE_SELECTOR_CANDIDATES:
        try:
            elements = page.query_selector_all(selector)
        except Exception:
            continue
        for el in elements:
            try:
                text = el.inner_text().strip()
            except Exception:
                continue
            if not text:
                continue
            for match in _PRICE_RE.findall(text):
                push(text)
                break

    if not snippets:
        # Fallback: grep the whole page text for obvious pricing strings.
        try:
            text = page.inner_text("body") or ""
        except Exception:
            text = ""
        for match in _PRICE_RE.findall(text):
            sample = text[max(0, text.index(match) - 40): text.index(match) + len(match) + 40]
            push(sample)
            if len(snippets) >= 6:
                break

    return snippets


def _with_page(url: str, fn: Callable[[Page], T]) -> T:
    safe_url = validate_public_url(url)
    with sync_playwright() as p:
        browser, page = _new_browser_page(p)
        try:
            _navigate(page, safe_url)
            return fn(page)
        finally:
            browser.close()


def scrape_page(url: str) -> dict:
    def _scrape(page: Page) -> dict:
        full_text = page.inner_text("body") or ""
        capped_text = _cap_text(full_text)
        return {
            "url": url,
            "title": page.title(),
            "text": capped_text,
            "text_truncated": len(full_text) > SCRAPE_TEXT_LLM_CAP,
            "text_total_chars": len(full_text),
            "headings": page.eval_on_selector_all(
                "h1, h2, h3",
                "els => els.map(e => ({tag: e.tagName.toLowerCase(), text: e.innerText.trim()}))",
            ),
            "links": page.eval_on_selector_all(
                "a[href]",
                "els => els.slice(0, 80).map(e => ({text: e.innerText.trim(), href: e.getAttribute('href')}))",
            ),
            "images": page.eval_on_selector_all(
                "img",
                "els => els.slice(0, 80).map(e => ({alt: e.getAttribute('alt') || '', src: e.getAttribute('src')}))",
            ),
            "price_candidates": _extract_price_snippets(page),
        }

    return _with_page(url, _scrape)


def take_screenshot(url: str, selector: str | None = None, full_page: bool = True) -> str:
    """Full-page or selector-based screenshot.

    Kept available as an MCP tool but the agent prompt steers towards
    `capture_excerpts`, which yields focused per-issue crops.
    """
    def _shot(page: Page) -> str:
        if selector:
            el = page.query_selector(selector)
            if not el:
                raise RuntimeError(f"Selector not found: {selector}")
            return base64.b64encode(el.screenshot()).decode()
        return base64.b64encode(page.screenshot(full_page=full_page)).decode()

    return _with_page(url, _shot)


# ---------------------------------------------------------------------------
# Per-issue evidence: locate the smallest block element containing each
# excerpt, scroll it into view, and clip a focused screenshot with padding.
# ---------------------------------------------------------------------------

# Walk up from the matched node until the bounding rect is a sensible block
# (avoids screenshotting a 1×1 inline span). Returns the parent ElementHandle
# so we can let Playwright do the scrolling + clipping natively.
_WALK_UP_JS = r"""
(el) => {
    let target = el;
    for (let i = 0; i < 5; i++) {
        const r = target.getBoundingClientRect();
        if (r.height >= 28 && r.width >= 100) break;
        if (!target.parentElement) break;
        target = target.parentElement;
    }
    return target;
}
"""


# LLMs sometimes annotate excerpts with the element type — "H1: ...", "Heading: ...",
# "Title: ..." — which doesn't appear verbatim on the page. Strip those.
_PREFIX_RE = re.compile(r"^(?:h[1-6]|heading|title|caption|alt|label)\s*[:\-]\s*", re.I)


def _normalise_excerpt(excerpt: str) -> str:
    text = (excerpt or "").strip().strip("\"'“”‘’")
    text = _PREFIX_RE.sub("", text).strip()
    return text.rstrip(".,;:!?…").strip()


def _candidate_snippets(text: str) -> list[str]:
    """Try the full text first, then progressively shorter prefixes / word slices."""
    out: list[str] = []
    seen: set[str] = set()

    def push(s: str) -> None:
        s = s.strip()
        if len(s) >= 4 and s not in seen:
            seen.add(s)
            out.append(s)

    push(text)
    if len(text) > 60:
        push(text[:60])
    if len(text) > 30:
        push(text[:30])
    # Longest run of words (often more distinctive than a prefix)
    words = text.split()
    if len(words) >= 4:
        push(" ".join(words[:4]))
    return out


def _first_visible_match(page: Page, snippet: str, max_candidates: int = 20):
    """Return the first VISIBLE locator matching `snippet` (skips hidden nav/footer)."""
    loc = page.get_by_text(snippet, exact=False)
    try:
        count = loc.count()
    except Exception:
        return None
    if count == 0:
        return None
    for i in range(min(count, max_candidates)):
        cand = loc.nth(i)
        try:
            if cand.is_visible(timeout=200):
                return cand
        except Exception:
            continue
    return None


def _is_blank_png(buf: bytes, min_unique_colours: int = 4) -> bool:
    """Reject screenshots that are essentially a single colour (modal backdrops,
    hidden elements, white-on-white text, etc)."""
    try:
        img = PILImage.open(BytesIO(buf)).convert("RGB")
        # Sample at low res — fast and good enough to detect "all one colour".
        small = img.resize((32, 32))
        unique = len(set(small.get_flattened_data()))
    except Exception:
        return False  # if we can't read it, let the caller decide
    return unique < min_unique_colours


def _capture_excerpt(page: Page, excerpt: str) -> str | None:
    text = _normalise_excerpt(excerpt)
    if not text:
        return None

    handle = None
    matched_with: str | None = None
    for snippet in _candidate_snippets(text):
        try:
            cand = _first_visible_match(page, snippet)
            if cand is None:
                continue
            handle = cand.element_handle(timeout=2500)
            if handle is not None:
                matched_with = snippet
                break
        except Exception as exc:
            logger.debug("evidence: snippet %r missed (%s)", snippet[:40], type(exc).__name__)
            continue
    if handle is None:
        logger.info("evidence: no element found for %r", text[:60])
        return None
    logger.debug("evidence: matched %r via %r", text[:60], matched_with)

    # Walk up to a sensible block-level ancestor so the screenshot has context.
    try:
        js_handle = handle.evaluate_handle(_WALK_UP_JS)
        block = js_handle.as_element()
    except Exception as exc:  # noqa: BLE001
        logger.info("evidence walk-up failed for %r: %s", text[:40], exc)
        return None
    target = block or handle

    # Let Playwright handle scrolling — it knows about every weird scroll
    # container, smooth-scroll CSS, sticky headers, etc. Then take an element
    # screenshot, which clips natively without manual viewport math.
    try:
        target.scroll_into_view_if_needed(timeout=3000)
    except Exception as exc:  # noqa: BLE001
        logger.debug("evidence: scroll_into_view best-effort failed: %s", exc)

    try:
        buf = target.screenshot(timeout=5000)
    except Exception as exc:  # noqa: BLE001
        logger.info("evidence screenshot failed for %r: %s", text[:40], exc)
        return None
    if _is_blank_png(buf):
        logger.info("evidence: rejected blank screenshot for %r", text[:60])
        return None
    return base64.b64encode(buf).decode()


EVIDENCE_TOKEN_PREFIX = "evidence://"


def _evidence_dir() -> Path:
    d = Path(tempfile.gettempdir()) / "qa_agent_evidence"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _store_evidence_png(b64: str) -> str:
    """Persist a base64 PNG to the evidence cache and return an opaque token.

    Tokens are content-addressed (sha256) so the same screenshot is stored once.
    Callers exchange tokens for the actual PNG via `read_evidence_png`.
    """
    raw = base64.b64decode(b64)
    digest = hashlib.sha256(raw).hexdigest()
    path = _evidence_dir() / f"{digest}.png"
    if not path.exists():
        path.write_bytes(raw)
    return f"{EVIDENCE_TOKEN_PREFIX}{digest}"


def read_evidence_png(token: str) -> str | None:
    """Resolve an `evidence://<sha256>` token back to a base64 PNG, or None."""
    if not isinstance(token, str) or not token.startswith(EVIDENCE_TOKEN_PREFIX):
        return None
    digest = token[len(EVIDENCE_TOKEN_PREFIX):].strip()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        return None
    path = _evidence_dir() / f"{digest}.png"
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode()


def capture_excerpts(url: str, excerpts: list[str]) -> dict[str, str]:
    """Open `url` once and return {excerpt: evidence_token} for each unique non-empty excerpt.

    Tokens are tiny opaque strings (`evidence://<sha256>`); the underlying PNGs are
    cached on disk and resolved by `read_evidence_png`. Returning tokens (not
    base64) keeps the LLM's tool-result context small enough for the model to
    produce a final JSON without truncating or erroring on context size.

    Excerpts that can't be located are silently omitted from the returned dict.
    """
    safe_url = validate_public_url(url)
    deduped: list[str] = []
    seen: set[str] = set()
    for e in excerpts or []:
        if not e:
            continue
        if e in seen:
            continue
        seen.add(e)
        deduped.append(e)

    out: dict[str, str] = {}
    if not deduped:
        return out

    with sync_playwright() as p:
        browser, page = _new_browser_page(p)
        try:
            _navigate(page, safe_url)
            for excerpt in deduped:
                shot_b64 = _capture_excerpt(page, excerpt)
                if shot_b64:
                    out[excerpt] = _store_evidence_png(shot_b64)
        finally:
            browser.close()
    return out
