"""Layered web content & image extraction for QA evidence collection.

This module collects *evidence* from a course web page so a downstream QA agent
can compare it against rules extracted from a QA template (PDF / DOCX / image).
It is deliberately **deterministic** — no LLM is involved in the extraction
itself, so the evidence is faithful and reproducible.

Extraction is layered (see the module docstring of `extract_page`):

  1. WordPress REST API  — structured page + media metadata where available.
  2. Rendered DOM (Playwright) — JS-rendered text, lazy images, sliders,
     banners, accordions, tabs and page-builder sections.
  3. CSS / background / srcset / lazy / page-builder image discovery.
  4. Carousel slides extracted straight from the DOM (all slides, not just the
     first visible one).
  5. Element screenshots as evidence for banners and QA-relevant images.
  6. OCR (Tesseract) on those screenshots to recover text baked into images.
  7. Claim detection + QA-priority scoring.

The public entry points are `extract_page(url, ...)` (returns the full
structured report) and `extract_page_summary(url, ...)` (persists the full
report to disk and returns a compact summary — used by the MCP tool so the
LLM's context stays small).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections import OrderedDict
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import httpx
from PIL import Image as PILImage
from playwright.sync_api import Page, TimeoutError as PWTimeoutError, sync_playwright

from .security import (
    MAX_HTTP_RESPONSE_BYTES,
    UnsafeURLError,
    validate_public_url,
)
from .tools.web_tools import (
    USER_AGENT,
    _EXPAND_SECTIONS_JS,
    _cap_text,
    _new_browser_page,
    _truncate_head_at_word,
)

logger = logging.getLogger(__name__)

# OCR is optional: if Tesseract isn't installed we still return everything else
# and record a warning rather than failing the whole extraction.
try:  # pragma: no cover - import guard
    import pytesseract

    if os.environ.get("TESSERACT_CMD"):
        pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_CMD"]
    _OCR_IMPORTED = True
except Exception:  # noqa: BLE001
    pytesseract = None  # type: ignore
    _OCR_IMPORTED = False

NAV_TIMEOUT_MS = 45_000
LOAD_BEST_EFFORT_MS = 8_000
SCROLL_SETTLE_MS = 900

# Bounds so a hostile / huge page can't blow up memory or OCR cost.
MAX_OCR_IMAGES = int(os.environ.get("QA_EXTRACTION_MAX_OCR", "40"))
MIN_OCR_AREA_PX = 2_000          # skip OCR on tiny icons
MIN_OCR_CONFIDENCE = 45          # below this we keep the text but flag low confidence

# Server-side cache of per-run extraction inputs (page text + headings + price
# candidates + banner/image evidence), keyed by a short `extraction_id`.
#
# WHY: the agent's LLM must not re-emit large blobs between tool calls. If the
# model had to copy the page text + banner/image evidence into the `spell` and
# `compliance` tool arguments, generating that much output on a flaky provider
# stalls past the upstream idle timeout. Instead `extract` stashes everything
# here and returns a tiny id; `spell`/`compliance` resolve it server-side. This
# mirrors the `evidence://` token pattern used for screenshots.
_EXTRACTION_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_CACHE_MAX = int(os.environ.get("QA_EXTRACTION_CACHE_SIZE", "8"))


def _cache_put(extraction_id: str, payload: dict) -> None:
    _EXTRACTION_CACHE[extraction_id] = payload
    _EXTRACTION_CACHE.move_to_end(extraction_id)
    while len(_EXTRACTION_CACHE) > _CACHE_MAX:
        _EXTRACTION_CACHE.popitem(last=False)


def get_cached_extraction(extraction_id: str | None) -> dict | None:
    """Resolve an `extraction_id` to its cached compliance/spell inputs, or None."""
    if not extraction_id:
        return None
    return _EXTRACTION_CACHE.get(extraction_id)


# ---------------------------------------------------------------------------
# Claim detection
# ---------------------------------------------------------------------------
# Each pattern returns the matched *evidence snippet* (we never invent text).

_CLAIM_PATTERNS: dict[str, re.Pattern] = {
    "price": re.compile(
        r"(?:£|GBP|EUR|€|USD|\$)\s?\d[\d,]*(?:\.\d{1,2})?"
        r"|\d[\d,]*\s?(?:GBP|USD|EUR|pounds?|£|€)"
        r"|\bfree\b|\bno cost\b",
        re.I,
    ),
    "discount": re.compile(
        r"\b\d{1,3}\s?%\s?(?:off|discount)\b"
        r"|\b(?:save|discount|offer|sale|deal|reduced|was\s+£?\d)\b"
        r"|\bup to\s+\d{1,3}\s?%",
        re.I,
    ),
    "duration": re.compile(
        r"\b\d+(?:\.\d+)?\s?(?:hours?|hrs?|days?|weeks?|months?|years?)\b"
        r"|\bself[- ]paced\b|\bduration\b",
        re.I,
    ),
    "certification": re.compile(
        r"\b(?:certificate|certified|certification|diploma|cpd|qualification|"
        r"award|credential|nvq|rqf|ofqual)\b",
        re.I,
    ),
    "accreditation": re.compile(
        r"\b(?:accredited|accreditation|approved by|endorsed|recognised|regulated)\b",
        re.I,
    ),
    "awarding_body": re.compile(
        r"\b(?:awarding body|awarded by|iso\s?9001|city\s?&\s?guilds|pearson|"
        r"edexcel|nccq|qualifi|tquk|highfield|othm|abma|cmi|ilm)\b",
        re.I,
    ),
    "eligibility": re.compile(
        r"\b(?:eligib\w+|entry requirement|no prior|prerequisite|open to all|"
        r"anyone can|who is this (?:course )?for)\b",
        re.I,
    ),
    "guarantee": re.compile(
        r"\b(?:guarantee\w*|money[- ]back|refund|risk[- ]free)\b",
        re.I,
    ),
    "ranking_or_rating": re.compile(
        r"\b\d(?:\.\d)?\s?/\s?5\b|\b\d(?:\.\d)?\s?stars?\b"
        r"|\b(?:rated|rating|#1|no\.?\s?1|number one|top[- ]rated|best[- ]selling|"
        r"trustpilot|reviews?)\b",
        re.I,
    ),
    "urgency": re.compile(
        r"\b(?:limited time|limited offer|hurry|last chance|ends (?:soon|today|"
        r"tonight)|today only|don'?t miss|while stocks last|few (?:seats|spots|"
        r"places) left|enrol(?:ment)? closes|act now|book now)\b",
        re.I,
    ),
}

_LEARNER_NUMBERS_RE = re.compile(
    r"\b[\d,]{2,}\+?\s?(?:students?|learners?|graduates?|members?|enrolled|"
    r"professionals?|people)\b",
    re.I,
)


def _clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def detect_claims(text: str) -> dict[str, list[str]]:
    """Return matched evidence snippets per claim category (schema section B/H).

    Snippets are taken verbatim from `text` — nothing is paraphrased.
    """
    out: dict[str, list[str]] = {k: [] for k in _CLAIM_PATTERNS}
    out["other"] = []
    if not text:
        return out
    for category, pattern in _CLAIM_PATTERNS.items():
        seen: set[str] = set()
        for m in pattern.finditer(text):
            snippet = _clean(m.group(0))
            low = snippet.lower()
            if snippet and low not in seen:
                seen.add(low)
                out[category].append(snippet)
            if len(out[category]) >= 6:
                break
    # Learner-numbers folds into "other" so the schema's fixed keys stay stable.
    for m in _LEARNER_NUMBERS_RE.finditer(text):
        snippet = _clean(m.group(0))
        if snippet and snippet not in out["other"]:
            out["other"].append(snippet)
        if len(out["other"]) >= 6:
            break
    return out


def _claim_types(claims: dict[str, list[str]]) -> list[str]:
    return [k for k, v in claims.items() if v]


def _has_claim(claims: dict[str, list[str]]) -> bool:
    return any(claims.values())


# ---------------------------------------------------------------------------
# Image classification + filtering (schema sections C, D, G)
# ---------------------------------------------------------------------------

_AWARDING_BODY_HINT = re.compile(
    r"accredit|awarding|cpd|iso|ofqual|rqf|nvq|city.?&.?guilds|pearson|edexcel|"
    r"qualifi|tquk|highfield|othm|abma|cmi|ilm|endorsed|approved",
    re.I,
)
_TRUST_HINT = re.compile(
    r"trust|secure|guarantee|trustpilot|reviews?|rating|stars?|verified|"
    r"ssl|norton|mcafee|badge", re.I
)
_SOCIAL_HINT = re.compile(
    r"facebook|twitter|x-logo|instagram|linkedin|youtube|tiktok|pinterest|"
    r"whatsapp|social|share", re.I
)
_PAYMENT_HINT = re.compile(
    r"visa|mastercard|maestro|paypal|amex|american-?express|stripe|klarna|"
    r"apple-?pay|google-?pay|payment", re.I
)
_SPACER_HINT = re.compile(r"spacer|blank|transparent|pixel|1x1|placeholder|divider", re.I)
_ICON_HINT = re.compile(r"\bicon\b|sprite|favicon|emoji|chevron|arrow", re.I)
_LOGO_HINT = re.compile(r"logo|brand", re.I)
_HERO_HINT = re.compile(r"hero|jumbotron|masthead|banner", re.I)
_PROMO_HINT = re.compile(r"promo|offer|sale|discount|deal|campaign", re.I)
_THUMB_HINT = re.compile(r"thumb|course|card|product|catalog", re.I)
_DECOR_HINT = re.compile(r"\bbg\b|background|pattern|texture|shape|wave|blob|decor", re.I)


def _img_haystack(img: dict) -> str:
    return " ".join(
        str(img.get(k, ""))
        for k in ("class_name", "resolved_url", "alt_text", "title_attribute",
                  "parent_section_heading", "caption")
    )


def _filter_verdict(img: dict) -> tuple[bool, str]:
    """Section G: should this image be demoted to low priority / decorative?

    Returns (is_low_priority_noise, reason). We never *drop* images — accreditation
    / trust / promotional / hero / thumbnail / text-bearing images always survive.
    """
    hay = _img_haystack(img)
    w, h = img.get("width", 0) or 0, img.get("height", 0) or 0
    if _SOCIAL_HINT.search(hay):
        return True, "social media icon"
    if _PAYMENT_HINT.search(hay):
        return True, "payment icon"
    if _SPACER_HINT.search(hay):
        return True, "spacer / placeholder"
    if (w and w <= 2) or (h and h <= 2):
        return True, "tracking pixel / 1x1"
    if _ICON_HINT.search(hay) and max(w, h) <= 64:
        return True, "decorative icon"
    return False, ""


def _classify_image(img: dict) -> str:
    """Section C: assign an image_type from the allowed enum."""
    hay = _img_haystack(img)
    w, h = img.get("width", 0) or 0, img.get("height", 0) or 0
    source_type = img.get("source_type", "")

    if _AWARDING_BODY_HINT.search(hay):
        # Distinguish awarding-body logos from generic accreditation badges.
        if _LOGO_HINT.search(hay):
            return "awarding_body_logo"
        return "accreditation_logo"
    if _TRUST_HINT.search(hay):
        return "trust_badge"
    if _LOGO_HINT.search(hay):
        return "logo"
    if img.get("in_hero") or (img.get("is_above_the_fold") and max(w, h) >= 600):
        return "hero"
    if _PROMO_HINT.search(hay):
        return "promotional"
    if _THUMB_HINT.search(hay):
        return "course_thumbnail"
    if _ICON_HINT.search(hay) or max(w, h) <= 64:
        return "icon"
    if source_type in ("css_background", "page_builder", "inline_style"):
        return "background"
    if _DECOR_HINT.search(hay) and not img.get("alt_text"):
        return "decorative"
    return "unknown"


_LEADGEN_LINK_RE = re.compile(
    r"enrol|enroll|checkout|cart|buy|purchase|signup|sign-up|register|apply|"
    r"contact|enquir|inquir|lead|book|payment|course", re.I
)

_HIGH_PRIORITY_TYPES = {
    "hero", "promotional", "accreditation_logo", "awarding_body_logo", "trust_badge",
}
_MEDIUM_PRIORITY_TYPES = {"course_thumbnail", "background", "logo"}
_LOW_PRIORITY_TYPES = {"icon", "decorative"}


def _image_priority(img: dict, image_type: str, has_claim: bool,
                    ocr_text: str, noise: bool) -> str:
    """Section H priority rules for images."""
    if noise and not has_claim and not ocr_text:
        return "low"
    linked = img.get("linked_url") or ""
    if (
        has_claim
        or ocr_text                       # text not present in HTML
        or image_type in _HIGH_PRIORITY_TYPES
        or (linked and _LEADGEN_LINK_RE.search(linked))
    ):
        return "high"
    if image_type in _MEDIUM_PRIORITY_TYPES or img.get("alt_text"):
        return "medium"
    if image_type in _LOW_PRIORITY_TYPES:
        return "low"
    # Section J: when unsure, include and mark medium.
    return "medium"


# ---------------------------------------------------------------------------
# Banner classification (schema section B)
# ---------------------------------------------------------------------------

_PRICING_HINT = re.compile(
    r"buy now|enquire now|add to cart|checkout|£\s?\d|\bprice\b|pricing|"
    r"save \d{1,3}\s?%|money[- ]back|get your code",
    re.I,
)


def _classify_banner(banner: dict, combined_text: str) -> str:
    hint = (banner.get("banner_type_hint") or "").lower()
    hay = " ".join([
        str(banner.get(k, "")) for k in ("page_position",)
    ] + banner.get("image_urls", []) + banner.get("background_image_urls", []) + [combined_text])
    # Pricing widgets take precedence: a sticky price block can also carry trust
    # / accreditation badges, but its primary QA role is the price/CTA claims.
    if hint == "pricing" or _PRICING_HINT.search(combined_text):
        return "pricing"
    if _AWARDING_BODY_HINT.search(hay):
        return "accreditation"
    if _TRUST_HINT.search(combined_text):
        return "trust"
    if hint in {"hero", "promotional", "carousel", "popup", "sidebar", "footer"}:
        return hint
    if _PROMO_HINT.search(hay):
        return "promotional"
    if banner.get("is_carousel"):
        return "carousel"
    return "unknown"


def _banner_priority(banner_type: str, claims: dict, ocr_text: str, banner: dict) -> str:
    linked = (banner.get("linked_url") or "") + " " + (banner.get("cta_url") or "")
    if (
        _has_claim(claims)
        or ocr_text
        or banner_type in {"hero", "promotional", "carousel", "course", "trust",
                           "accreditation", "pricing"}
        or _LEADGEN_LINK_RE.search(linked)
    ):
        return "high"
    if banner.get("is_above_the_fold"):
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

def ocr_available() -> bool:
    if not _OCR_IMPORTED:
        return False
    try:
        pytesseract.get_tesseract_version()  # type: ignore[union-attr]
        return True
    except Exception:  # noqa: BLE001
        return False


def _ocr_png(png_bytes: bytes) -> tuple[str, float]:
    """OCR a PNG. Returns (text, mean_confidence 0-100). Empty on any failure."""
    if not png_bytes:
        return "", 0.0
    try:
        with PILImage.open(BytesIO(png_bytes)) as img:
            rgb = img.convert("RGB")
            data = pytesseract.image_to_data(  # type: ignore[union-attr]
                rgb, output_type=pytesseract.Output.DICT  # type: ignore[union-attr]
            )
    except Exception as exc:  # noqa: BLE001
        logger.debug("OCR failed: %s", exc)
        return "", 0.0
    words, confs = [], []
    for word, conf in zip(data.get("text", []), data.get("conf", [])):
        word = (word or "").strip()
        if not word:
            continue
        try:
            c = float(conf)
        except (TypeError, ValueError):
            c = -1.0
        if c >= 0:
            confs.append(c)
        words.append(word)
    text = _clean(" ".join(words))
    mean_conf = (sum(confs) / len(confs)) if confs else 0.0
    return text, round(mean_conf, 1)


# ---------------------------------------------------------------------------
# WordPress REST API layer (schema section: layered method step 1)
# ---------------------------------------------------------------------------

def _wp_http_client() -> httpx.Client:
    return httpx.Client(
        timeout=httpx.Timeout(connect=10.0, read=30.0, write=15.0, pool=10.0),
        verify=True,
        follow_redirects=False,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )


def _strip_html(html: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>", " ", html or "")
    text = re.sub(r"<[^>]+>", " ", text)
    return _clean(text)


def fetch_wordpress(url: str) -> dict | None:
    """Best-effort WordPress REST API extraction for the given page.

    Tries `/wp-json/wp/v2/pages` then `/posts`, matched by slug, with embedded
    media. Returns a structured dict (title, slug, content raw+clean, modified,
    featured + embedded media metadata incl. alt text & captions), or None if
    the site is not WordPress / the API is unavailable. Never raises.
    """
    try:
        safe = validate_public_url(url)
    except UnsafeURLError:
        return None
    parsed = urlparse(safe)
    base = f"{parsed.scheme}://{parsed.netloc}"
    slug = [seg for seg in parsed.path.split("/") if seg]
    slug = slug[-1] if slug else ""

    endpoints = []
    if slug:
        endpoints += [
            f"{base}/wp-json/wp/v2/pages?slug={slug}&_embed=1",
            f"{base}/wp-json/wp/v2/posts?slug={slug}&_embed=1",
        ]
    try:
        with _wp_http_client() as client:
            for endpoint in endpoints:
                try:
                    resp = client.get(endpoint)
                except httpx.HTTPError:
                    continue
                if resp.status_code != 200:
                    continue
                if "json" not in resp.headers.get("content-type", ""):
                    continue
                if len(resp.content) > MAX_HTTP_RESPONSE_BYTES:
                    continue
                try:
                    items = resp.json()
                except ValueError:
                    continue
                if isinstance(items, list) and items:
                    return _shape_wp_item(items[0])
    except Exception as exc:  # noqa: BLE001
        logger.debug("WordPress REST extraction failed: %s", exc)
        return None
    return None


def _shape_wp_item(item: dict) -> dict:
    def rendered(field: str) -> str:
        v = item.get(field)
        if isinstance(v, dict):
            return v.get("rendered", "") or ""
        return v or ""

    content_html = rendered("content")
    media: list[dict] = []
    embedded = item.get("_embedded", {}) or {}
    for group in ("wp:featuredmedia",):
        for m in embedded.get(group, []) or []:
            if not isinstance(m, dict):
                continue
            details = m.get("media_details", {}) or {}
            media.append({
                "id": m.get("id"),
                "source_url": m.get("source_url", ""),
                "alt_text": m.get("alt_text", ""),
                "caption": _strip_html(
                    (m.get("caption", {}) or {}).get("rendered", "")
                    if isinstance(m.get("caption"), dict) else ""
                ),
                "mime_type": m.get("mime_type", ""),
                "width": details.get("width"),
                "height": details.get("height"),
            })
    return {
        "id": item.get("id"),
        "title": _strip_html(rendered("title")),
        "slug": item.get("slug", ""),
        "link": item.get("link", ""),
        "modified": item.get("modified", "") or item.get("modified_gmt", ""),
        "featured_media": item.get("featured_media"),
        "content_clean": _strip_html(content_html),
        "media": media,
    }


def _wp_media_index(wp: dict | None) -> dict[str, dict]:
    """Map basename(source_url) -> media dict, for enriching DOM images."""
    index: dict[str, dict] = {}
    if not wp:
        return index
    for m in wp.get("media", []) or []:
        src = m.get("source_url") or ""
        if src:
            index[os.path.basename(urlparse(src).path)] = m
    return index


# ---------------------------------------------------------------------------
# Browser: navigation + the big DOM extraction script
# ---------------------------------------------------------------------------

# Auto-scroll to trigger lazy-loaded images / sliders, then return to the top.
_AUTOSCROLL_JS = r"""
async () => {
  await new Promise((resolve) => {
    let total = 0;
    const step = 600;
    const timer = setInterval(() => {
      const max = Math.max(
        document.body ? document.body.scrollHeight : 0,
        document.documentElement ? document.documentElement.scrollHeight : 0
      );
      window.scrollBy(0, step);
      total += step;
      if (total >= max + 1200) { clearInterval(timer); resolve(); }
    }, 110);
  });
  window.scrollTo(0, 0);
}
"""

# The DOM extraction script. Tags each captured element with
# `data-qa-extract-id` so Python can screenshot it by selector afterwards.
_EXTRACT_JS = Path(__file__).with_name("_extract_dom.js").read_text(encoding="utf-8") \
    if (Path(__file__).with_name("_extract_dom.js")).exists() else None


# ---------------------------------------------------------------------------
# Per-click FAQ / accordion capture (single-open accordions)
# ---------------------------------------------------------------------------
# Some FAQ / course-content accordions only allow ONE panel open at a time:
# opening the next question auto-closes the previous one. A single end-of-page
# innerText snapshot (what `_EXTRACT_JS` reads) therefore only ever sees the
# last-opened answer. To QA every answer against its own question heading and
# the course details, we click each toggle in turn and read its panel right
# after it opens — capturing the question + answer pair before the next click
# collapses it again. Read-only: we open panels and read text, nothing else.
MAX_FAQ_ITEMS = int(os.environ.get("QA_MAX_FAQ_ITEMS", "60"))
FAQ_CLICK_WAIT_MS = int(os.environ.get("QA_FAQ_CLICK_WAIT_MS", "250"))

_FAQ_TOGGLE_SELECTOR = ", ".join([
    "details > summary",
    "[class*='accordion' i] [class*='title' i]",
    "[class*='accordion' i] [class*='head' i]",
    "[class*='accordion' i] button",
    "[class*='faq' i] [class*='question' i]",
    "[class*='faq' i] [class*='title' i]",
    "[class*='faq' i] [class*='head' i]",
    "[class*='toggle' i] [class*='title' i]",
    ".elementor-tab-title",
    ".et_pb_toggle_title",
    "[data-toggle='collapse']",
    "[data-bs-toggle='collapse']",
])

# Resolve a toggle's answer panel and return its text, with the question text
# (which some containers repeat) trimmed off the front.
_READ_FAQ_PANEL_JS = r"""
(tog) => {
    const clean = (s) => ("" + (s || "")).replace(/\s+/g, " ").trim();
    let panel = null;
    const ac = tog.getAttribute("aria-controls");
    if (ac) { try { panel = document.getElementById(ac); } catch (e) {} }
    if (!panel) {
        const dt = tog.getAttribute("data-bs-target")
            || tog.getAttribute("data-target")
            || (tog.getAttribute("href") || "");
        if (dt && dt.charAt(0) === "#" && dt.length > 1) {
            try { panel = document.querySelector(dt); } catch (e) {}
        }
    }
    if (!panel && tog.tagName === "SUMMARY") panel = tog.parentElement;  // <details>
    if (!panel) {
        panel = tog.closest(
            "[class*='accordion-item' i], .elementor-accordion-item, "
            + ".et_pb_toggle, [class*='faq' i] [class*='item' i], li"
        );
    }
    if (!panel) {
        // Skip <style>/<script>/<noscript> siblings whose text is CSS/JS, not an answer.
        let sib = tog.nextElementSibling;
        while (sib && /^(STYLE|SCRIPT|NOSCRIPT)$/.test(sib.tagName)) sib = sib.nextElementSibling;
        panel = sib;
    }
    if (!panel) return "";
    if (/^(STYLE|SCRIPT|NOSCRIPT)$/.test(panel.tagName)) return "";
    let text = clean(panel.innerText);
    const q = clean(tog.innerText);
    if (q && text.toLowerCase().startsWith(q.toLowerCase())) {
        text = clean(text.slice(q.length));
    }
    return text;
}
"""


# Some panels embed a <style>/<script> block whose CSS/JS text can leak into
# innerText (e.g. a raw `#accordion { ... }` rule). That is not an answer — drop it.
_CODE_NOISE_RE = re.compile(
    r"[#.][\w-]+\s*\{|font-family\s*:|@media\b|!important|function\s*\(|;\s*\}",
    re.I,
)


def _looks_like_code(text: str) -> bool:
    return "{" in text and "}" in text and bool(_CODE_NOISE_RE.search(text))


def _capture_faq_items(page: Page) -> list[dict]:
    """Click each FAQ/accordion toggle one at a time and read its answer.

    Returns a list of ``{"question", "answer"}`` pairs. Best-effort: any toggle
    that can't be clicked or read is skipped. Real navigation links are skipped
    so a toggle that is actually an ``<a href>`` can't take us off-page.
    """
    items: list[dict] = []
    try:
        toggles = page.query_selector_all(_FAQ_TOGGLE_SELECTOR)
    except Exception as exc:  # noqa: BLE001
        logger.debug("FAQ toggle query failed: %s", exc)
        return items

    seen_q: set[str] = set()
    for tog in toggles:
        if len(items) >= MAX_FAQ_ITEMS:
            break
        try:
            question = _clean(tog.inner_text())
        except Exception:  # noqa: BLE001 - stale/detached handle
            continue
        if not question or len(question) > 300:
            continue
        qkey = question.lower()
        if qkey in seen_q:
            continue

        # Never follow a real navigation link.
        try:
            if tog.evaluate("e => e.tagName") == "A":
                href = tog.get_attribute("href") or ""
                if href and not href.startswith("#") and not href.startswith("javascript:"):
                    continue
        except Exception:  # noqa: BLE001
            pass

        # Open it if it isn't already (single-open accordions report state via
        # aria-expanded; if absent we just click and let the panel render).
        try:
            expanded = tog.get_attribute("aria-expanded")
        except Exception:  # noqa: BLE001
            expanded = None
        if expanded != "true":
            try:
                tog.scroll_into_view_if_needed(timeout=1500)
                tog.click(timeout=1500)
                page.wait_for_timeout(FAQ_CLICK_WAIT_MS)
            except Exception as exc:  # noqa: BLE001
                logger.debug("FAQ click failed for %r: %s", question[:40], exc)
                # Fall through — it may already be open; still try to read it.

        try:
            answer = _clean(tog.evaluate(_READ_FAQ_PANEL_JS))
        except Exception as exc:  # noqa: BLE001
            logger.debug("FAQ panel read failed for %r: %s", question[:40], exc)
            continue
        if not answer or answer.lower() == qkey or _looks_like_code(answer):
            continue
        seen_q.add(qkey)
        items.append({"question": question[:300], "answer": answer[:1500]})

    return items


def _format_faq_block(faq_items: list[dict]) -> str:
    """Render captured FAQ Q&A pairs as a labelled text block for QA checks.

    Each answer is kept paired with its own question so the compliance check can
    verify the answer aligns with its FAQ heading and the course content.
    """
    if not faq_items:
        return ""
    lines = ["FAQ SECTION (each answer captured under its own question):"]
    budget = 6000
    for item in faq_items:
        q = (item.get("question") or "").strip()
        a = (item.get("answer") or "").strip()
        if not q or not a:
            continue
        entry = f"Q: {q}\nA: {a}"
        if budget - len(entry) < 0:
            break
        budget -= len(entry)
        lines.append(entry)
    return "\n".join(lines) if len(lines) > 1 else ""


def _navigate(page: Page, url: str) -> None:
    page.goto(url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
    try:
        page.wait_for_load_state("load", timeout=LOAD_BEST_EFFORT_MS)
    except PWTimeoutError:
        logger.info("load event did not fire within %sms — continuing", LOAD_BEST_EFFORT_MS)
    try:
        page.evaluate(_AUTOSCROLL_JS)
    except Exception as exc:  # noqa: BLE001
        logger.debug("autoscroll best-effort failed: %s", exc)
    # Expand collapsed FAQ accordions / toggles / <details> so each answer's
    # text is rendered next to its heading and captured by the DOM extraction
    # below — otherwise hidden answers can't be QA'd against their FAQ heading
    # or the course content.
    try:
        page.evaluate(_EXPAND_SECTIONS_JS)
    except Exception as exc:  # noqa: BLE001
        logger.debug("section expand best-effort failed: %s", exc)
    page.wait_for_timeout(SCROLL_SETTLE_MS)


def _screenshot_dir(url: str) -> Path:
    base = os.environ.get("QA_EXTRACTION_DIR", "").strip()
    root = Path(base) if base else Path("reports") / "extraction"
    host = re.sub(r"[^a-z0-9.-]+", "_", urlparse(url).netloc.lower()) or "page"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = root / f"{host}-{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _screenshot_element(page: Page, qa_id: str, dest: Path) -> bytes | None:
    try:
        el = page.query_selector(f'[data-qa-extract-id="{qa_id}"]')
        if el is None:
            return None
        try:
            el.scroll_into_view_if_needed(timeout=2000)
        except Exception:  # noqa: BLE001
            pass
        buf = el.screenshot(timeout=5000)
    except Exception as exc:  # noqa: BLE001
        logger.debug("element screenshot failed for %s: %s", qa_id, exc)
        return None
    try:
        dest.write_bytes(buf)
    except Exception as exc:  # noqa: BLE001
        logger.debug("could not persist screenshot %s: %s", dest, exc)
    return buf


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_page(
    url: str,
    *,
    use_wordpress: bool = True,
    capture_screenshots: bool = True,
    run_ocr: bool = True,
) -> dict:
    """Run the full layered extraction and return the structured QA-evidence report.

    The shape matches the agreed schema: ``page_url``, ``general_content``,
    ``banners``, ``images`` and ``extraction_warnings`` (plus an extra
    ``wordpress`` block and ``stats`` for convenience).
    """
    safe_url = validate_public_url(url)
    warnings: list[str] = []

    if _EXTRACT_JS is None:
        raise RuntimeError(
            "Missing DOM extraction script (_extract_dom.js next to extraction.py)."
        )

    ocr_ok = run_ocr and ocr_available()
    if run_ocr and not ocr_ok:
        warnings.append(
            "OCR unavailable (Tesseract not installed or not on PATH) — "
            "image-embedded text was not recovered."
        )

    wp = fetch_wordpress(safe_url) if use_wordpress else None
    if use_wordpress and wp is None:
        warnings.append("WordPress REST API not available — used rendered DOM only.")
    wp_media = _wp_media_index(wp)

    with sync_playwright() as p:
        browser, page = _new_browser_page(p)
        try:
            _navigate(page, safe_url)
            try:
                raw = page.evaluate(_EXTRACT_JS)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"DOM extraction failed: {exc}") from exc

            warnings.extend(raw.get("warnings", []) or [])
            shot_dir = _screenshot_dir(safe_url) if capture_screenshots else None

            images = _process_images(
                page, raw.get("images", []), wp_media, shot_dir, ocr_ok, warnings
            )
            banners = _process_banners(
                page, raw.get("banners", []), shot_dir, ocr_ok, warnings
            )
            # Per-click FAQ/accordion capture runs LAST so its clicking can't
            # shift element positions for the image/banner screenshots above.
            faq_items = _capture_faq_items(page)
        finally:
            browser.close()

    general = _shape_general(raw.get("general", {}), wp)

    report = {
        "page_url": safe_url,
        "general_content": general,
        "banners": banners,
        "images": images,
        "faq_items": faq_items,
        "extraction_warnings": _dedupe(warnings),
        "wordpress": wp,
        "stats": {
            "image_count": len(images),
            "banner_count": len(banners),
            "high_priority_images": sum(1 for i in images if i["qa_priority"] == "high"),
            "high_priority_banners": sum(1 for b in banners if b["qa_priority"] == "high"),
            "ocr_enabled": ocr_ok,
            "wordpress_detected": wp is not None,
        },
    }
    return report


def _shape_general(g: dict, wp: dict | None) -> dict:
    return {
        "page_title": g.get("page_title", "") or (wp.get("title") if wp else ""),
        "h1": g.get("h1", ""),
        "headings": g.get("headings", []),
        "main_visible_text": g.get("main_visible_text", ""),
        "raw_text": g.get("raw_text", ""),
        "cleaned_text": g.get("cleaned_text", ""),
        "cta_buttons": g.get("cta_buttons", []),
        "links": g.get("links", []),
        "meta_title": g.get("meta_title", ""),
        "meta_description": g.get("meta_description", ""),
        "canonical_url": g.get("canonical_url", ""),
        "wordpress_modified": wp.get("modified") if wp else None,
    }


def _process_images(page, raw_images, wp_media, shot_dir, ocr_ok, warnings) -> list[dict]:
    out: list[dict] = []
    ocr_budget = MAX_OCR_IMAGES
    for idx, img in enumerate(raw_images or []):
        # Enrich missing alt/caption from WordPress media metadata.
        resolved = img.get("resolved_url", "")
        base = os.path.basename(urlparse(resolved).path)
        wp_meta = wp_media.get(base)
        if wp_meta:
            if not img.get("alt_text"):
                img["alt_text"] = wp_meta.get("alt_text", "")
            if not img.get("caption"):
                img["caption"] = wp_meta.get("caption", "")

        noise, noise_reason = _filter_verdict(img)
        image_type = _classify_image(img)

        notes: list[str] = []
        if noise_reason:
            notes.append(noise_reason)

        ocr_text, cleaned_ocr, screenshot_path = "", "", ""
        area = (img.get("width", 0) or 0) * (img.get("height", 0) or 0)

        # Screenshot + OCR only QA-relevant, visible, non-trivial images.
        wants_evidence = (
            shot_dir is not None
            and img.get("is_visible")
            and not noise
            and area >= MIN_OCR_AREA_PX
            and image_type not in _LOW_PRIORITY_TYPES
        )
        png = None
        if wants_evidence and img.get("qa_id"):
            dest = shot_dir / f"image-{idx:03d}.png"
            png = _screenshot_element(page, img["qa_id"], dest)
            if png:
                screenshot_path = str(dest)
        if png and ocr_ok and ocr_budget > 0:
            ocr_budget -= 1
            ocr_text, conf = _ocr_png(png)
            cleaned_ocr = _clean(ocr_text)
            if ocr_text and conf and conf < MIN_OCR_CONFIDENCE:
                notes.append(f"OCR confidence low ({conf}%)")

        combined_for_claims = " ".join(filter(None, [
            img.get("alt_text", ""), img.get("title_attribute", ""),
            img.get("caption", ""), img.get("nearby_text", ""), cleaned_ocr,
        ]))
        claims = detect_claims(combined_for_claims)
        claim_types = _claim_types(claims)
        has_claim = bool(claim_types)
        priority = _image_priority(img, image_type, has_claim, cleaned_ocr, noise)

        if not img.get("is_visible"):
            notes.append("not visible in rendered viewport")
        if img.get("source_type") in ("css_background", "page_builder", "lazy_loaded", "inline_style"):
            notes.append(f"discovered via {img['source_type'].replace('_', ' ')}")

        out.append({
            "image_id": f"image-{idx:03d}",
            "image_type": image_type,
            "source_type": img.get("source_type", "unknown"),
            "src_url": img.get("src_url", ""),
            "resolved_url": resolved,
            "file_name": base,
            "alt_text": img.get("alt_text", ""),
            "title_attribute": img.get("title_attribute", ""),
            "caption": img.get("caption", ""),
            "nearby_text": img.get("nearby_text", ""),
            "parent_section_heading": img.get("parent_section_heading", ""),
            "linked_url": img.get("linked_url", ""),
            "width": img.get("width", 0),
            "height": img.get("height", 0),
            "is_visible": bool(img.get("is_visible")),
            "is_above_the_fold": bool(img.get("is_above_the_fold")),
            "ocr_text": ocr_text,
            "cleaned_ocr_text": cleaned_ocr,
            "contains_claim": has_claim,
            "claim_types": claim_types,
            "screenshot_path": screenshot_path,
            "qa_priority": priority,
            "notes": "; ".join(notes),
        })
    return out


def _process_banners(page, raw_banners, shot_dir, ocr_ok, warnings) -> list[dict]:
    out: list[dict] = []
    idx = 0
    for banner in raw_banners or []:
        # Drop contentless wrappers (e.g. an empty popup container around a
        # bare link) — they carry no QA evidence and only add noise.
        if not any([
            _clean(banner.get("visible_text_html", "")),
            banner.get("image_urls"),
            banner.get("background_image_urls"),
            _clean(banner.get("cta_text", "")),
        ]):
            continue
        notes: list[str] = []
        ocr_text, screenshot_path = "", ""

        png = None
        if shot_dir is not None and banner.get("qa_id") and banner.get("is_visible"):
            dest = shot_dir / f"banner-{idx:03d}.png"
            png = _screenshot_element(page, banner["qa_id"], dest)
            if png:
                screenshot_path = str(dest)
        if png and ocr_ok:
            ocr_text, conf = _ocr_png(png)
            if ocr_text and conf and conf < MIN_OCR_CONFIDENCE:
                notes.append(f"OCR confidence low ({conf}%)")

        html_text = banner.get("visible_text_html", "")
        cleaned_combined = _clean(" ".join(filter(None, [html_text, ocr_text])))
        claims = detect_claims(cleaned_combined)
        banner_type = _classify_banner(banner, cleaned_combined)
        priority = _banner_priority(banner_type, claims, _clean(ocr_text), banner)

        if banner.get("is_carousel"):
            notes.append("carousel slide" + (
                f" #{banner.get('slide_index')}" if banner.get("slide_index") is not None else ""
            ))
        if not banner.get("is_visible"):
            notes.append("hidden / dynamic (not in current viewport)")

        out.append({
            "banner_id": f"banner-{idx:03d}",
            "banner_type": banner_type,
            "page_position": banner.get("page_position", ""),
            "is_above_the_fold": bool(banner.get("is_above_the_fold")),
            "visible_text_html": html_text,
            "visible_text_ocr": ocr_text,
            "cleaned_combined_text": cleaned_combined,
            "cta_text": banner.get("cta_text", ""),
            "cta_url": banner.get("cta_url", ""),
            "image_urls": banner.get("image_urls", []),
            "background_image_urls": banner.get("background_image_urls", []),
            "linked_url": banner.get("linked_url", ""),
            "screenshot_path": screenshot_path,
            "claims_detected": claims,
            "qa_priority": priority,
            "notes": "; ".join(notes),
        })
        idx += 1
    return out


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it and it not in seen:
            seen.add(it)
            out.append(it)
    return out


def extract_page_summary(
    url: str,
    *,
    use_wordpress: bool = True,
    capture_screenshots: bool = True,
    run_ocr: bool = True,
) -> dict:
    """Run extraction, persist the full report to disk, return a compact summary.

    The full (potentially large) evidence — page text, headings, price
    candidates and the high-priority banner/image evidence — is cached
    server-side under a short ``extraction_id``; ``spell`` and ``compliance``
    resolve it by id so the agent's LLM never has to re-emit those blobs as tool
    arguments (which was stalling the provider past its idle timeout). The
    summary returned here is intentionally small.
    """
    report = extract_page(
        url,
        use_wordpress=use_wordpress,
        capture_screenshots=capture_screenshots,
        run_ocr=run_ocr,
    )
    report_path = _persist_report(url, report)
    extraction_id = hashlib.sha256(report_path.encode("utf-8")).hexdigest()[:16]

    def slim_banner(b: dict) -> dict:
        return {
            "banner_id": b["banner_id"], "banner_type": b["banner_type"],
            "page_position": b["page_position"], "qa_priority": b["qa_priority"],
            "cleaned_combined_text": b["cleaned_combined_text"][:300],
            "cta_text": b["cta_text"], "cta_url": b["cta_url"],
            "claim_types": [k for k, v in b["claims_detected"].items() if v],
        }

    def slim_image(i: dict) -> dict:
        return {
            "image_id": i["image_id"], "image_type": i["image_type"],
            "qa_priority": i["qa_priority"], "resolved_url": i["resolved_url"],
            "alt_text": i["alt_text"], "claim_types": i["claim_types"],
            "ocr_excerpt": i["cleaned_ocr_text"][:200],
        }

    gc = report["general_content"]
    full_text = gc.get("raw_text", "") or gc.get("cleaned_text", "")
    capped_text = _cap_text(full_text)
    price_candidates = _collect_price_candidates(report)
    banner_evidence = [slim_banner(b) for b in report["banners"] if b["qa_priority"] == "high"]
    image_evidence = [slim_image(i) for i in report["images"] if i["qa_priority"] == "high"]

    # Fold per-click-captured FAQ answers into the page text the spell/compliance
    # checks read. Single-open accordions only ever leave their LAST answer in
    # the snapshot text, so each Q&A is appended here as a labelled block. We keep
    # body + FAQ within the compliance read window (~8000 chars) so the FAQ — the
    # content we want aligned against its heading and the course details — is
    # never sliced off the end downstream.
    faq_items = report.get("faq_items") or []
    faq_block = _format_faq_block(faq_items)
    if faq_block:
        head_room = max(1500, 8000 - len(faq_block) - 20)
        page_text_for_checks = f"{_truncate_head_at_word(capped_text, head_room)}\n\n{faq_block}"
    else:
        page_text_for_checks = capped_text

    # Stash the heavy inputs server-side; the model only ever sees `extraction_id`.
    _cache_put(extraction_id, {
        "page_text": page_text_for_checks,
        "headings": gc.get("headings", []),
        "price_candidates": price_candidates,
        "banner_evidence": banner_evidence,
        "image_evidence": image_evidence,
    })

    return {
        "page_url": report["page_url"],
        "extraction_id": extraction_id,
        "report_path": report_path,
        "stats": report["stats"],
        "text_total_chars": len(full_text),
        "text_truncated": len(full_text) > len(capped_text),
        "price_candidates": price_candidates,
        "general_content": {
            "page_title": gc["page_title"],
            "h1": gc["h1"],
            "meta_title": gc["meta_title"],
            "meta_description": gc["meta_description"],
            "canonical_url": gc["canonical_url"],
            "heading_count": len(gc["headings"]),
            "link_count": len(gc["links"]),
            "cta_count": len(gc["cta_buttons"]),
        },
        # Trimmed previews only (full evidence is cached + on disk). The agent
        # does NOT need to copy these anywhere — compliance reads them by id.
        "high_priority_banners": banner_evidence[:8],
        "high_priority_images": image_evidence[:12],
        # FAQ Q&A captured by clicking each (single-open) accordion in turn. The
        # full text is already folded into the cached page_text the checks read;
        # this preview lets the agent/UI see what was captured.
        "faq_count": len(faq_items),
        "faq_items": faq_items[:30],
        "extraction_warnings": report["extraction_warnings"],
    }


def _collect_price_candidates(report: dict) -> list[str]:
    """Gather pricing evidence from page text + banner/image claims (for compliance)."""
    cands: list[str] = []
    seen: set[str] = set()

    def push(s: str) -> None:
        s = _clean(s)
        if s and s.lower() not in seen:
            seen.add(s.lower())
            cands.append(s)

    for snippet in detect_claims(report["general_content"].get("cleaned_text", ""))["price"]:
        push(snippet)
    for b in report["banners"]:
        if b["claims_detected"].get("price"):
            push(b["cleaned_combined_text"][:140] or "; ".join(b["claims_detected"]["price"]))
    for i in report["images"]:
        if "price" in i["claim_types"]:
            push(i.get("cleaned_ocr_text") or i.get("alt_text") or "")
    return cands[:8]


def _persist_report(url: str, report: dict) -> str:
    base = os.environ.get("QA_EXTRACTION_DIR", "").strip()
    root = Path(base) if base else Path("reports") / "extraction"
    root.mkdir(parents=True, exist_ok=True)
    host = re.sub(r"[^a-z0-9.-]+", "_", urlparse(url).netloc.lower()) or "page"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = root / f"extraction-{host}-{stamp}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# Standalone CLI:  python -m qa_agent.extraction --url https://...
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse

    from dotenv import load_dotenv

    from .logging_config import configure_logging

    load_dotenv()
    configure_logging()

    ap = argparse.ArgumentParser(description="Layered QA evidence extraction.")
    ap.add_argument("--url", "-u", required=True, help="Page URL to extract.")
    ap.add_argument("--no-wordpress", action="store_true", help="Skip the WordPress REST API layer.")
    ap.add_argument("--no-screenshots", action="store_true", help="Skip element screenshots.")
    ap.add_argument("--no-ocr", action="store_true", help="Skip OCR of image-embedded text.")
    ap.add_argument("--out", "-o", default=None, help="Write the full report to this JSON path.")
    args = ap.parse_args()

    report = extract_page(
        args.url,
        use_wordpress=not args.no_wordpress,
        capture_screenshots=not args.no_screenshots,
        run_ocr=not args.no_ocr,
    )
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        out_path = args.out
    else:
        out_path = _persist_report(args.url, report)

    s = report["stats"]
    print("\nExtraction complete.")
    print(f"  Report:  {out_path}")
    print(f"  Images:  {s['image_count']} ({s['high_priority_images']} high priority)")
    print(f"  Banners: {s['banner_count']} ({s['high_priority_banners']} high priority)")
    print(f"  WordPress detected: {s['wordpress_detected']}; OCR: {s['ocr_enabled']}")
    if report["extraction_warnings"]:
        print("  Warnings:")
        for w in report["extraction_warnings"]:
            print(f"    - {w}")


if __name__ == "__main__":
    _main()
