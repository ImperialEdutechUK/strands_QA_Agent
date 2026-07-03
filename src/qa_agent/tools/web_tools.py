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
    //
    // IMPORTANT: a sticky *price block* (price + "Buy Now"/"Enquire Now" +
    // money-back guarantee + Trustpilot) is real QA content, not an overlay —
    // but it's short and fixed, so the naive rule would hide it and every
    // price-related evidence screenshot would come back blank. Protect any
    // sticky element whose class/id/text marks it as pricing or a purchase CTA.
    const PROTECT_STICKY = /price|pricing|cart|checkout|buy[\s-]?now|enquir|enrol|enroll|purchase|add[\s-]?to[\s-]?cart|guarantee/i;
    document.querySelectorAll('*').forEach(el => {
        if (el === document.body || el === document.documentElement) return;
        const cs = getComputedStyle(el);
        if (cs.position !== 'fixed' && cs.position !== 'sticky') return;
        const textLen = (el.innerText || '').trim().length;
        if (textLen >= 800) return;  // probably real content, not an overlay
        const sig = ((el.className && el.className.toString) ? el.className.toString() : '')
            + ' ' + (el.id || '') + ' ' + (el.innerText || '');
        if (PROTECT_STICKY.test(sig)) return;  // keep price / purchase CTAs visible
        el.style.setProperty('display', 'none', 'important');
    });
}
"""


# Expand collapsed FAQ accordions / toggles / native <details> so their answer
# text is rendered (and therefore captured by innerText and screenshottable).
# Many FAQ sections keep each answer hidden (display:none / collapsed panel)
# until its heading is clicked; without this pass the agent never sees the
# answer content and so cannot check that it aligns with its FAQ heading or the
# course details. We also click generic "View more" / "Show more" / "Load more"
# reveal buttons (step 4) — course-curriculum / module lists are routinely
# truncated behind one of these, with the remaining modules hidden until clicked,
# so without this the agent only ever QAs the surface modules against the
# qualification specification, not the full curriculum.
# We only *open* sections — we never close, navigate or alter the text. Real
# navigation links are skipped so a toggle that is actually an <a href> can't
# take us off-page.
_EXPAND_SECTIONS_JS = r"""
() => {
    let opened = 0;

    // 1. Native <details>/<summary>: force open without a click.
    document.querySelectorAll('details:not([open])').forEach((d) => {
        try { d.open = true; opened++; } catch (e) {}
    });

    const safeToClick = (el) => {
        if (!el) return false;
        // Don't follow real navigation; in-page (#...) / JS toggles are fine.
        if (el.tagName === 'A') {
            const href = el.getAttribute('href') || '';
            if (href && !href.startsWith('#') && !href.startsWith('javascript:')) return false;
        }
        // Skip anything that already reports itself as expanded/open.
        if (el.getAttribute('aria-expanded') === 'true') return false;
        return true;
    };

    // 2. ARIA accordions: click controls explicitly marked collapsed.
    document.querySelectorAll('[aria-expanded="false"]').forEach((el) => {
        try { if (safeToClick(el)) { el.click(); opened++; } } catch (e) {}
    });

    // 3. Common FAQ / accordion / tab toggles that don't use aria-expanded.
    const toggleSel = [
        '.accordion-toggle', '.accordion-header', '.accordion-button.collapsed',
        '.accordion-title', '.accordion__title', '.accordion-trigger',
        '[class*="accordion" i] [class*="head" i]',
        '[class*="accordion" i] [class*="title" i]',
        '[class*="faq" i] [class*="question" i]',
        '[class*="faq" i] [class*="title" i]',
        '[class*="faq" i] [class*="toggle" i]',
        '.elementor-tab-title:not(.elementor-active)',
        '.et_pb_toggle_title',
        '[data-toggle="collapse"]', '[data-bs-toggle="collapse"]'
    ].join(',');
    document.querySelectorAll(toggleSel).forEach((el) => {
        try { if (safeToClick(el)) { el.click(); opened++; } } catch (e) {}
    });

    // 4. Generic "View more" / "Show more" / "Read more" / "Load more" reveal
    //    controls. A course curriculum / module list is commonly truncated to
    //    the first few items with the rest hidden behind one of these buttons.
    //    They are usually NOT aria-accordions and don't match the FAQ/accordion
    //    selectors above, so without this pass the hidden modules never reach
    //    innerText and the curriculum can't be QA'd in full against the spec.
    //    Match on the control's own short label so we don't click large blocks
    //    that merely contain the words, and never click a "show less / collapse"
    //    control (which would re-hide content we just revealed).
    const MORE_RE = /^\s*(?:\+\s*)?(?:view|show|read|see|load|display|reveal)\s+(?:more|all|full|the\s+full|complete)\b|\bview\s+full\b|\bfull\s+curriculum\b|\bexpand(?:\s+all)?\b|\b(?:all|more)\s+modules?\b|\bsee\s+(?:the\s+)?(?:full|complete|entire)\b/i;
    const LESS_RE = /\b(?:less|collapse|hide|fewer)\b/i;
    const moreSel = [
        'button', '[role="button"]',
        '[class*="view-more" i]', '[class*="viewmore" i]',
        '[class*="show-more" i]', '[class*="showmore" i]',
        '[class*="read-more" i]', '[class*="readmore" i]',
        '[class*="load-more" i]', '[class*="loadmore" i]',
        '[class*="see-more" i]', '[class*="seemore" i]',
        '[class*="more-link" i]', '[class*="moretoggle" i]',
        '[class*="expand" i]', 'a'
    ].join(',');
    const moreSeen = new Set();
    let moreClicks = 0;
    document.querySelectorAll(moreSel).forEach((el) => {
        if (moreClicks >= 40 || moreSeen.has(el)) return;
        moreSeen.add(el);
        let label = '';
        try {
            label = ('' + (el.innerText || el.textContent || el.getAttribute('aria-label') || '')).trim();
        } catch (e) { return; }
        // Reveal-button labels are short; a long string means we matched a
        // wrapper, not the actual control.
        if (!label || label.length > 60) return;
        if (LESS_RE.test(label) || !MORE_RE.test(label)) return;
        if (!safeToClick(el)) return;
        try { el.click(); opened++; moreClicks++; } catch (e) {}
    });

    return opened;
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
    try:
        page.evaluate(_EXPAND_SECTIONS_JS)
        page.wait_for_timeout(600)  # let expanded FAQ panels render before reading
    except Exception as exc:  # noqa: BLE001
        logger.debug("section expand failed (continuing): %s", exc)


def _truncate_head_at_word(text: str, limit: int) -> str:
    """``text[:limit]`` backtracked to a whitespace boundary so the slice never
    ends mid-word.

    A naive ``text[:limit]`` can land inside a word — e.g. slicing
    "...recognised qualifications designed..." at the wrong offset leaves
    "...recognised qualifications d", which the spell/grammar checker then
    reports as a real "cut off mid-word / incomplete sentence" error. Backtracking
    to the last space (or newline) means the slice always ends on a whole word.
    """
    if len(text) <= limit:
        return text
    # Only trim if the cut actually lands inside a word (both sides non-space).
    if text[limit - 1].strip() and text[limit].strip():
        boundary = max(text.rfind(" ", 0, limit), text.rfind("\n", 0, limit))
        if boundary > 0:
            return text[:boundary].rstrip()
    return text[:limit].rstrip()


def _truncate_tail_at_word(text: str, limit: int) -> str:
    """Last ``limit`` chars, advanced past any leading partial word, so a tail
    slice never *begins* mid-word (the mirror of `_truncate_head_at_word`)."""
    if len(text) <= limit:
        return text
    start = len(text) - limit
    if text[start].strip() and text[start - 1].strip():
        nxt = text.find(" ", start)
        if nxt != -1 and nxt - start < 200:
            return text[nxt:].lstrip()
    return text[start:].lstrip()


def _cap_text(text: str) -> str:
    if len(text) <= SCRAPE_TEXT_LLM_CAP:
        return text
    return (
        _truncate_head_at_word(text, SCRAPE_TEXT_HEAD_CHUNK)
        + "\n\n... [content truncated; tail preserved] ...\n\n"
        + _truncate_tail_at_word(text, SCRAPE_TEXT_TAIL_CHUNK)
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

# Walk up from the matched node to the smallest *block-level* element that
# contains the offending text — a paragraph, list item, heading, table cell or
# the like — so the evidence screenshot shows just the part with the issue, not
# the whole surrounding section. We deliberately do NOT climb to section / card
# / row containers: the reviewer asked for the specific line, not the page.
# A hard height cap keeps a stray full-height wrapper from swallowing the page.
_WALK_UP_JS = r"""
(el) => {
    const vh = window.innerHeight || 900;
    // Keep the crop TIGHT around the offending text — reviewers asked for just the
    // part with the issue, not a whole screenful (which was pulling in the nav bar,
    // hero banner and unrelated copy). ~1/3 of the viewport is enough context.
    const maxH = vh * 0.34;
    const BLOCK = new Set(['p','li','h1','h2','h3','h4','h5','h6','td','th',
        'dd','dt','figcaption','blockquote','label','a','button','span','div']);
    const isBlock = (n) => {
        if (!n || !n.tagName) return false;
        const disp = getComputedStyle(n).display;
        return disp === 'block' || disp === 'list-item' || disp === 'table-cell'
            || disp === 'flex' || disp === 'grid'
            || BLOCK.has(n.tagName.toLowerCase());
    };
    // Climb out of tiny inline wrappers until we reach a block that is big
    // enough to read but still no taller than the cap. Stop at the FIRST such
    // block — don't keep climbing into the enclosing section.
    let target = el;
    for (let i = 0; i < 6 && target.parentElement; i++) {
        const r = target.getBoundingClientRect();
        if (isBlock(target) && r.height >= 18 && r.width >= 80 && r.height <= maxH) {
            break;
        }
        // If this node already overflows the cap, the previous (smaller) one was
        // the best focused choice — but we only reach here when it wasn't a
        // usable block yet, so step up once more and re-test.
        target = target.parentElement;
    }
    // Safety net: if we still ended up with something taller than the cap,
    // prefer the originally matched (smaller) element over the oversized block.
    if (target.getBoundingClientRect().height > maxH) {
        return el;
    }
    return target;
}
"""

# Stop-words stripped when deriving section keywords from an excerpt/description.
_STOPWORDS = frozenset(
    "the a an of to and or for is are be on in at with this that section page "
    "must should not no missing present above below text course your you will".split()
)


def _section_keywords(text: str) -> list[str]:
    """Pull the few most distinctive words from an excerpt/description.

    Used to locate a section by its heading when the excerpt itself isn't
    present verbatim on the page (e.g. a compliance issue that says a section
    is missing or mis-titled). Returns [] when nothing distinctive remains.
    """
    words = re.findall(r"[A-Za-z][A-Za-z'&-]{2,}", text or "")
    keep: list[str] = []
    seen: set[str] = set()
    for w in words:
        low = w.lower()
        if low in _STOPWORDS or low in seen:
            continue
        seen.add(low)
        keep.append(w)
        if len(keep) >= 6:
            break
    return keep


def _capture_section_by_keywords(page: Page, text: str) -> str | None:
    """Fallback: screenshot the page section whose heading matches `text`.

    Looks for an h1–h4 containing the excerpt's distinctive keywords, then
    screenshots that heading's enclosing section. Returns base64 PNG or None.
    """
    keywords = _section_keywords(text)
    if not keywords:
        return None
    try:
        headings = page.query_selector_all("h1, h2, h3, h4")
    except Exception:
        return None
    best = None
    best_score = 0
    for h in headings:
        try:
            if not h.is_visible(timeout=150):
                continue
            htext = (h.inner_text() or "").lower()
        except Exception:
            continue
        score = sum(1 for kw in keywords if kw.lower() in htext)
        if score > best_score:
            best_score = score
            best = h
    if best is None or best_score == 0:
        return None
    try:
        js_handle = best.evaluate_handle(_WALK_UP_JS)
        block = js_handle.as_element() or best
        block.scroll_into_view_if_needed(timeout=3000)
        buf = block.screenshot(timeout=5000)
    except Exception as exc:  # noqa: BLE001
        logger.info("evidence: section fallback failed for %r: %s", text[:40], exc)
        return None
    if _is_blank_png(buf):
        return None
    return base64.b64encode(buf).decode()


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
    # The model often COMPOSES an excerpt by joining separate page items with
    # " - " / " | " / newlines (e.g. "Course Highlights - Access Duration: 365
    # Days - Awarded by: ..."). That composite never exists verbatim in the
    # DOM, so also try each segment, longest first — the most distinctive one
    # usually locates the right block.
    segments = [s for s in re.split(r"\s+[-–|•]\s+|\n", text) if len(s.strip()) >= 8]
    if len(segments) > 1:
        for seg in sorted(segments, key=len, reverse=True)[:4]:
            push(seg)
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
        unique = len(set(small.getdata()))
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
        # The excerpt isn't on the page verbatim (e.g. a structural / "missing
        # section" finding). Fall back to screenshotting the section whose
        # heading best matches the excerpt's keywords; skip if none matches.
        section_shot = _capture_section_by_keywords(page, text)
        if section_shot is None:
            logger.info("evidence: no element or section found for %r", text[:60])
        return section_shot
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


def attach_issue_screenshots(report: dict) -> dict:
    """Deterministically attach a cropped screenshot to every issue that has one.

    The agent is supposed to call the `evidence` tool and copy tokens onto each
    issue, but that depends on the model remembering to do it — and when it
    forgets, the report comes back with no screenshots at all (which the sample
    QA docs always include). This runs the capture ourselves, from the issues'
    own excerpts, so a screenshot is attached whenever the excerpt can be located
    on the page. Issues that already carry a screenshot, or whose excerpt can't
    be found, are left as-is. Best-effort: any failure leaves the report intact.

    Returns the same report (mutated in place) for convenience.
    """
    url = report.get("url")
    issues = report.get("issues") or []
    if not url or not issues:
        return report
    wanted = [
        (issue.get("excerpt") or "").strip()
        for issue in issues
        if isinstance(issue, dict) and not issue.get("screenshot")
        and (issue.get("excerpt") or "").strip()
        # Reference-diff findings quote the REFERENCE page — that text is by
        # definition absent from the page under review, so don't hunt for it.
        and not str(issue.get("ruleId", "")).upper().startswith("REF")
    ]
    if not wanted:
        return report
    try:
        shots = capture_excerpts(url, wanted)  # {excerpt: evidence_token}
    except Exception as exc:  # noqa: BLE001 - evidence is best-effort
        logger.info("deterministic screenshot capture failed: %s", exc)
        return report
    for issue in issues:
        if not isinstance(issue, dict) or issue.get("screenshot"):
            continue
        excerpt = (issue.get("excerpt") or "").strip()
        token = shots.get(excerpt)
        if token:
            issue["screenshot"] = token
    return report
