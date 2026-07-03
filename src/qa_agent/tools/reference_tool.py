"""Structural cross-check against a reference (known-good) course page.

A human QA reviewer opens a published, approved course page side by side with
the page under review and spots what's absent: a missing section, a missing
'Buy Now' button, no pricing banner. This tool does the same DETERMINISTICALLY
— it extracts the reference page and diffs page STRUCTURE (headings, CTA
buttons, high-value banner types) against the target's evidence. No LLM is
involved, so nothing can be invented; every finding names the concrete
reference element that has no counterpart on the page under review.

Course-specific wording is neutralised before comparison (both pages' course
names / levels / subjects are stripped), so 'What is the Level 3 Diploma in
Adult Care?' on the reference matches 'What is the Level 5 Diploma in
Accounting?' on the target — we compare the page SKELETON, not the copy.
"""

from __future__ import annotations

import base64
import logging
import re

from playwright.sync_api import sync_playwright

from ..extraction import extract_page
from ..security import validate_public_url
from .web_tools import (
    _capture_excerpt,
    _first_visible_match,
    _is_blank_png,
    _navigate,
    _new_browser_page,
    _store_evidence_png,
)

logger = logging.getLogger(__name__)

# Words too generic to identify a section on their own.
_STOP = frozenset(
    "the a an of to and or for is are be on in at with this that your you will "
    "our what who why how course courses page here more now".split()
)

# Headings that are review titles, promo strips, modal/CTA chrome or global
# widgets — they differ between pages legitimately and must never be diffed.
# (A previous run flagged 'Really Glad' — a REVIEW title — and 'Selected
# country' — a footer widget — as missing sections.)
_NOISE_HEADING_RE = re.compile(
    r"download|brochure|subscribe|newsletter|share|follow us|related|"
    r"you may also|students? also|selected country|sign up|log ?in|"
    r"enquire|enrol|apply now|contact|whatsapp|call us|find the right|"
    r"make a difference|affordable|glad|thank",
    re.I,
)

# Zones (tagged by the DOM extraction) whose headings are page furniture or
# per-course content, not comparable structure.
_SKIP_ZONES = {"review", "footer", "header", "popup", "sidebar", "accordion"}

_MAX_FINDINGS = 10

# Banner types that carry QA-critical claims; their absence vs the reference is
# worth flagging. Decorative/footer/unknown banners are not compared.
_KEY_BANNER_TYPES = {"pricing", "accreditation", "trust"}

_BANNER_TYPE_LABEL = {
    "pricing": "pricing / purchase block (price + Buy Now / Enquire Now)",
    "accreditation": "accreditation / awarding-body banner",
    "trust": "trust banner (guarantee / reviews / rating)",
}


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _course_tokens(identity: dict | None, h1: str) -> set[str]:
    """Tokens specific to ONE course (name, subject, level, awarding body).

    Stripped from both pages' headings before comparison so the diff is about
    structure, not about the two courses being different subjects.
    """
    toks: set[str] = set()
    for v in (identity or {}).values():
        toks.update(_words(str(v)))
    toks.update(_words(h1))
    toks.update({"level", "diploma", "certificate", "award", "extended", "rqf", "qls"})
    toks.update(str(n) for n in range(1, 9))
    return toks


def _sig(text: str, course_toks: set[str]) -> frozenset[str]:
    """A heading's structural signature: distinctive words minus course words."""
    return frozenset(
        w for w in _words(text) if w not in _STOP and w not in course_toks
    )


def _heading_findings(ref_headings: list, ref_toks: set[str],
                      target_headings: list, target_toks: set[str],
                      target_text_words: set[str], reference_url: str) -> list[dict]:
    target_sigs = [
        _sig(h.get("text", ""), target_toks)
        for h in target_headings if isinstance(h, dict)
    ]
    target_sigs = [s for s in target_sigs if s]

    findings: list[dict] = []
    seen: set[frozenset[str]] = set()
    for h in ref_headings or []:
        if not isinstance(h, dict) or h.get("tag") not in ("h2", "h3"):
            continue
        # Only MAIN-content headings are comparable structure. Review titles,
        # FAQ questions, curriculum unit names (accordion zone), footer/nav and
        # popup headings differ between courses legitimately. Headings from an
        # older cached extraction without zone info default to comparable.
        if h.get("zone", "main") in _SKIP_ZONES:
            continue
        text = (h.get("text") or "").strip()
        if _NOISE_HEADING_RE.search(text):
            continue
        sig = _sig(text, ref_toks)
        # Need at least 2 distinctive words to identify a section reliably —
        # anything less would flag on wording noise, not structure.
        if len(sig) < 2 or sig in seen:
            continue
        seen.add(sig)
        # Matched if any target heading shares most of the signature, or the
        # words all appear in the target page text (some sections are labelled
        # by a table cell rather than a real heading element).
        best = max((len(sig & ts) / len(sig) for ts in target_sigs), default=0.0)
        if best >= 0.6 or sig <= target_text_words:
            continue
        findings.append({
            "ruleId": "REF-SECTION",
            "type": "Template",
            "severity": "Minor",
            "description": (
                f"The reference course page has a section headed '{text}' "
                f"but no equivalent section/heading was found on this page."
            ),
            "suggestion": "Add this section (see the reference course page for its placement).",
            "excerpt": text[:120],
        })
    return findings


# Only lead-gen / purchase CTAs are QA-meaningful structure. Site chrome
# ('Sign In', 'Submit', 'Start search', 'Selected country' — often hidden
# widgets that never render in the target's visible text) produced pure noise.
_KEY_CTA_RE = re.compile(
    r"buy\s*now|enquir|enrol|apply\s*now|add\s*to\s*cart|checkout|"
    r"get\s*your\s*code|download\s*(?:free\s*)?brochure|whatsapp|"
    r"contact\s*advisor|book\s*now|pay\b",
    re.I,
)


def _cta_findings(ref_ctas: list, target_text: str, reference_url: str) -> list[dict]:
    target_low = (target_text or "").lower()
    findings: list[dict] = []
    seen: set[str] = set()
    for cta in ref_ctas or []:
        label = (cta.get("text") or "").strip() if isinstance(cta, dict) else ""
        # Real button labels are short; ignore long matched wrappers and
        # navigation ("Home", course titles in menus, etc.).
        if not (3 <= len(label) <= 30):
            continue
        if not _KEY_CTA_RE.search(label):
            continue
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        if key in target_low:
            continue
        findings.append({
            "ruleId": "REF-CTA",
            "type": "Template",
            "severity": "Minor",
            "description": (
                f"The reference course page has a '{label}' button/CTA; "
                f"no matching button text was found on this page."
            ),
            "suggestion": f"Add the '{label}' button (match the reference course page).",
            "excerpt": label,
        })
    return findings


def _banner_findings(ref_banners: list, target_banners: list,
                     reference_url: str) -> list[dict]:
    ref_types = {
        b.get("banner_type") for b in ref_banners or []
        if isinstance(b, dict) and b.get("qa_priority") == "high"
    }
    target_types = {
        b.get("banner_type") for b in target_banners or [] if isinstance(b, dict)
    }
    findings: list[dict] = []
    for btype in sorted((ref_types & _KEY_BANNER_TYPES) - target_types):
        label = _BANNER_TYPE_LABEL.get(btype, f"{btype} banner")
        findings.append({
            "ruleId": "REF-BANNER",
            "type": "Template",
            "severity": "Minor",
            "description": (
                f"The reference course page has a {label}; "
                f"no banner of that kind was detected on this page."
            ),
            "suggestion": f"Add the {label} (match the reference course page).",
            "excerpt": "",
            "banner_type": btype,
        })
    return findings


# ---------------------------------------------------------------------------
# Evidence screenshots FROM THE REFERENCE PAGE
# ---------------------------------------------------------------------------
# A "'Buy Now' button is missing" finding is far more actionable when the report
# SHOWS the reviewer what that button looks like on the reference page — and,
# when it sits inside a pricing banner, shows the whole banner so they can see
# exactly what block to reproduce. These crops come from the REFERENCE page
# (clearly captioned as such), since the element is by definition absent from
# the page under review.

# Walk up from a matched CTA to the banner/pricing block that CONTAINS it: climb
# to the largest ancestor still comfortably smaller than a screenful, stopping
# early at a container whose class/id marks it as a banner / pricing / CTA
# widget. This gives the surrounding block, not just the bare button.
_CTA_BLOCK_JS = r"""
(el) => {
    const vh = window.innerHeight || 900;
    const maxH = vh * 0.6;
    let best = el;
    let node = el.parentElement;
    for (let i = 0; i < 8 && node && node !== document.body; i++) {
        const r = node.getBoundingClientRect();
        if (r.height > maxH || r.height < 1 || r.width < 120) break;
        best = node;
        const sig = ((node.className && node.className.toString)
            ? node.className.toString() : '') + ' ' + (node.id || '');
        if (r.height >= 80
            && /price|pricing|banner|hero|sidebar|purchase|buy|cta|card|widget|box|promo/i.test(sig)) {
            return node;
        }
        node = node.parentElement;
    }
    return best;
}
"""

# Best-effort selectors to find a banner of the missing type on the reference
# page (REF-BANNER findings have no excerpt text to search for).
_BANNER_SELECTOR_HINTS = {
    "pricing": "[class*='price' i], [class*='pricing' i], [class*='purchase' i], "
               "[class*='buy' i], [id*='price' i]",
    "accreditation": "[class*='accredit' i], [class*='awarding' i], "
                     "[class*='ofqual' i], [id*='accredit' i]",
    "trust": "[class*='trust' i], [class*='guarantee' i], [class*='review' i], "
             "[class*='rating' i]",
}


def _capture_cta_block(page, label: str) -> str | None:
    """Screenshot the reference CTA button together with its enclosing block."""
    try:
        cand = _first_visible_match(page, label)
        if cand is None:
            return None
        handle = cand.element_handle(timeout=2500)
        if handle is None:
            return None
        block = handle.evaluate_handle(_CTA_BLOCK_JS).as_element() or handle
        block.scroll_into_view_if_needed(timeout=3000)
        buf = block.screenshot(timeout=5000)
    except Exception as exc:  # noqa: BLE001
        logger.info("reference CTA capture failed for %r: %s", label[:40], exc)
        return None
    if _is_blank_png(buf):
        return None
    return base64.b64encode(buf).decode()


def _capture_banner_by_type(page, banner_type: str) -> str | None:
    """Screenshot the first plausible banner of `banner_type` on the reference page."""
    selector = _BANNER_SELECTOR_HINTS.get(banner_type)
    if not selector:
        return None
    try:
        for el in page.query_selector_all(selector)[:20]:
            try:
                if not el.is_visible():
                    continue
                box = el.bounding_box()
                if not box or box["height"] < 60 or box["height"] > 900 or box["width"] < 200:
                    continue
                el.scroll_into_view_if_needed(timeout=2000)
                buf = el.screenshot(timeout=5000)
            except Exception:  # noqa: BLE001 — try the next candidate
                continue
            if not _is_blank_png(buf):
                return base64.b64encode(buf).decode()
    except Exception as exc:  # noqa: BLE001
        logger.info("reference banner capture failed for %r: %s", banner_type, exc)
    return None


def _attach_reference_screenshots(safe_ref: str, findings: list[dict]) -> None:
    """Open the reference page once and attach an evidence crop to each finding.

    Each attached issue also gets a `screenshot_caption` making clear the image
    shows the REFERENCE page (what the target is missing), not the page under
    review. Entirely best-effort — findings without a locatable element are
    left without a screenshot.
    """
    if not findings:
        return
    try:
        with sync_playwright() as p:
            browser, page = _new_browser_page(p)
            try:
                _navigate(page, safe_ref)
                for f in findings:
                    shot: str | None = None
                    caption = ""
                    if f["ruleId"] == "REF-CTA":
                        label = f.get("excerpt") or ""
                        shot = _capture_cta_block(page, label)
                        caption = (f"Reference page — the '{label}' button and its "
                                   f"surrounding block (add the equivalent to this page).")
                    elif f["ruleId"] == "REF-SECTION":
                        heading = f.get("excerpt") or ""
                        shot = _capture_excerpt(page, heading)
                        caption = (f"Reference page — the '{heading}' section "
                                   f"missing from this page.")
                    elif f["ruleId"] == "REF-BANNER":
                        btype = f.get("banner_type") or ""
                        shot = _capture_banner_by_type(page, btype)
                        caption = ("Reference page — the banner of this kind "
                                   "missing from this page.")
                    if shot:
                        f["screenshot"] = _store_evidence_png(shot)
                        f["screenshot_caption"] = caption
            finally:
                browser.close()
    except Exception as exc:  # noqa: BLE001 — evidence is additive, never fatal
        logger.info("reference evidence capture failed: %s", exc)


def extract_reference(reference_url: str) -> dict:
    """Extract the reference page (structure only — no screenshots, OCR or FAQ
    clicking; none of those feed the diff). Raises on failure so the caller can
    record an honest skip. Split out of `compare_with_reference` so the pipeline
    can run this heavy browser step in parallel with the main extraction."""
    safe_ref = validate_public_url(reference_url)
    ref = extract_page(safe_ref, capture_screenshots=False, run_ocr=False,
                       capture_faq=False)
    return {"safe_ref": safe_ref, "ref": ref}


def compare_with_reference(reference_url: str, target: dict,
                           ref_extraction: dict | None = None) -> dict:
    """Diff the target page's structure against a reference course page.

    ``target`` is the cached extraction payload (page_text, headings,
    banner_evidence, course_identity). ``ref_extraction`` is an optional
    pre-computed `extract_reference` result (the pipeline extracts the
    reference in parallel); when omitted the reference is extracted here.
    Returns {"issues": [...], "reference_url": ...}. Best-effort: any
    extraction failure returns an empty list with a "skipped" reason — this
    check is additive evidence.
    """
    try:
        if ref_extraction is None:
            ref_extraction = extract_reference(reference_url)
        safe_ref = ref_extraction["safe_ref"]
        ref = ref_extraction["ref"]
    except Exception as exc:  # noqa: BLE001
        logger.warning("reference extraction failed for %s: %s", reference_url, exc)
        return {"issues": [], "reference_url": reference_url,
                "skipped": f"reference page could not be extracted: {type(exc).__name__}"}

    ref_gc = ref.get("general_content") or {}
    ref_identity_h1 = ref_gc.get("h1") or ref_gc.get("page_title") or ""
    ref_toks = _course_tokens(None, ref_identity_h1)

    target_identity = target.get("course_identity") or {}
    target_toks = _course_tokens(target_identity, target_identity.get("course_name", ""))
    target_text = target.get("page_text", "")
    target_text_words = set(_words(target_text))

    findings = (
        _heading_findings(ref_gc.get("headings", []), ref_toks,
                          target.get("headings", []), target_toks,
                          target_text_words, safe_ref)
        + _cta_findings(ref_gc.get("cta_buttons", []), target_text, safe_ref)
        + _banner_findings(ref.get("banners", []),
                           target.get("banner_evidence", []), safe_ref)
    )
    if len(findings) > _MAX_FINDINGS:
        logger.info("reference diff produced %d findings; capping at %d",
                    len(findings), _MAX_FINDINGS)
        findings = findings[:_MAX_FINDINGS]
    # Show, don't just tell: crop the missing element (CTA + its surrounding
    # banner, section, key banner) FROM THE REFERENCE PAGE onto each finding.
    _attach_reference_screenshots(safe_ref, findings)
    for f in findings:
        f.pop("banner_type", None)  # internal capture hint, not report schema
    return {"issues": findings, "reference_url": safe_ref}
