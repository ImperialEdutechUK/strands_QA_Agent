from __future__ import annotations

import os
import re
from datetime import date, datetime, timezone

from ..llm_client import call_llm_json
from .web_tools import _truncate_head_at_word

# How much page text the compliance call may read — the WHOLE page (a very high
# ceiling). Must be >= the body+FAQ size extraction caches so we never re-clip the
# page MIDDLE/tail back off here (re-clipping was dropping the assessment /
# curriculum / careers sections and producing false "not stated" findings).
# Override via QA_COMPLIANCE_READ_CHARS.
_PAGE_TEXT_READ_CHARS = int(os.environ.get("QA_COMPLIANCE_READ_CHARS", "1000000"))
# Prompt-size budget for the single compliance call: large enough that the full
# page text + rules + evidence are never truncated by the LLM client's default
# 64 KiB tool-input cap. Override via QA_COMPLIANCE_PROMPT_BYTES.
_COMPLIANCE_PROMPT_BYTES = int(os.environ.get("QA_COMPLIANCE_PROMPT_BYTES", str(2 * 1024 * 1024)))
# Time bounds for the compliance call. It is the core analysis (and re-raises on
# failure, so we give it more room than the fail-fast template), but still bounded
# so a saturated provider can't turn it into an open-ended stall — it completes or
# fails honestly. Override via env.
_COMPLIANCE_READ_TIMEOUT = float(os.environ.get("QA_COMPLIANCE_READ_TIMEOUT", "180"))
_COMPLIANCE_RETRY_BUDGET = float(os.environ.get("QA_COMPLIANCE_RETRY_BUDGET", "60"))

SYSTEM = (
    "You audit a course web page against a list of QA template rules. "
    "Be strict but factual: ONLY emit an issue for a rule the page clearly "
    "VIOLATES. Rules that the page satisfies, or where no action is needed, "
    "MUST NOT appear in the output at all — never emit 'compliant', 'passes', "
    "'looks fine', 'no action needed' or confirmation entries. "
    "Claims that appear in banners, hero/promotional graphics, or as text baked "
    "into images (provided as OCR evidence) — such as price, discount, duration, "
    "certification, accreditation, awarding body, guarantee, rating or urgency "
    "claims — count as page content and must be checked against the rules just "
    "like body text. "
    "Some rules are marked needs_spec: check those against the OFFICIAL "
    "SPECIFICATION block (found via web search). If — and only if — the "
    "specification IS available and the page clearly contradicts it (wrong level, "
    "wrong qualification number, wrong awarding body, etc.), emit the issue. If "
    "the specification is absent or does not cover that rule, DO NOT emit "
    "anything for it and DO NOT tell the reader to verify it manually — silently "
    "skip it. Verifying the page is your job, not the reader's; never defer work "
    "back to a human. "
    "SPEC VARIANT CHECK: before flagging any spec mismatch, confirm the OFFICIAL "
    "SPECIFICATION block is for the SAME qualification variant as the page — same "
    "level and the same 'Extended'/non-'Extended' form, and ideally the same "
    "qualification number. If the spec's course_name does not match the page's "
    "variant (e.g. the spec is a plain 'Diploma' but the page is an 'Extended "
    "Diploma'), the wrong spec was retrieved — DO NOT flag credit/GLH/TQT/level "
    "mismatches against it; skip those needs_spec rules silently. "
    "GROUNDING — this is critical: the PAGE TEXT you are given is the COMPLETE "
    "rendered page body (accordions and 'view more' sections were expanded "
    "before capture, and FAQ answers captured per-question are appended at the "
    "end in a labelled FAQ SECTION block). Base every finding on text you can "
    "actually QUOTE from this evidence. The PAGE HEADINGS list is also complete: "
    "if a matching heading is present, the section EXISTS — never flag it as "
    "missing. Only claim something is 'missing' or 'not present' when it is "
    "genuinely absent from BOTH the headings and the page text. "
    "Be conservative: if you cannot clearly see a violation in the evidence "
    "provided, emit NOTHING for that rule — do not guess, and do not invent a "
    "generic list of missing sections. A correct review of a complete page "
    "usually finds only a handful of real issues, not dozens. "
    "DURATION SEMANTICS — these are THREE DIFFERENT concepts; never compare one "
    "against another's value: "
    "(1) ACCESS DURATION = how long the learner can access the course platform "
    "(standard 1 year; Extended Diplomas 2 years). "
    "(2) AVERAGE COMPLETION TIMEFRAME = how long a typical learner takes to "
    "finish (e.g. 'learners typically complete in 6-9 months'). A completion "
    "timeframe SHORTER than the access duration is normal and correct: a page "
    "saying learners complete in 6-9 months does NOT contradict a 1-year access "
    "duration, and a specification duration of 1 year does NOT mean the "
    "completion timeframe must be 1 year — never flag one against the other. "
    "(3) GLH / TQT = hours of study effort, not calendar time; never compare "
    "GLH/TQT hours against months or years. "
    "Only flag a duration value against the SAME kind of duration. "
    "PRICING (rule about pricing being added): if ANY price/currency value "
    "(e.g. '£1,099.00', '£499') appears anywhere in the PRICE CANDIDATES, the "
    "banner/image evidence, or the page text, then pricing HAS been added — do "
    "NOT flag it as missing. The pricing block is the one carrying the price plus "
    "the 'Buy Now' and 'Enquire Now' buttons; do NOT confuse it with a separate "
    "call-to-action block that shows 'Enquire Now' and 'Apply Now' (that block is "
    "unrelated to whether a price is present). Only flag pricing as missing when "
    "there is genuinely no price anywhere on the page. "
    "MISSING SECTIONS — before reporting a section ('Who Is This Course For?', "
    "'Learning Outcomes', 'Career Progression', 'Entry Requirements', "
    "'Assessment Overview', FAQs, etc.) as missing, check BOTH the PAGE HEADINGS "
    "list and the page text: if a matching heading or the section's content is "
    "there, the section EXISTS — never flag it missing, and never hedge with "
    "'not present in the provided headings or text'. "
    "LAYOUT LABELS — course pages use a two-column layout where the LEFT cell "
    "is the section label ('Method of Assessment', 'Certification', 'Career "
    "Progression', …) and the section's own heading sits at the top of the "
    "right-hand content. A rule requiring a heading (e.g. 'Assessment "
    "Overview' above the assessment details) is SATISFIED when that heading "
    "appears with the section content — the left-hand label is not the "
    "heading; never flag it as a wrong heading. "
    "DATE FINDINGS — only emit a date issue when you can quote a date that is "
    "genuinely impossible (e.g. 31 February) or genuinely AFTER today's date. "
    "If your check concludes the dates are fine, output NOTHING for that rule "
    "— NEVER emit an entry whose description says the dates are logical, real, "
    "or in the past. "
    "VISUAL FORMATTING — you receive TEXT plus a verbatim BOLD TEXT RUNS list "
    "(every piece of text the rendered page shows in bold). Judge bold rules "
    "ONLY from that list: text that appears in the list IS bold; only flag a "
    "bold rule when the required text is genuinely absent from the list. You "
    "still CANNOT see font family/size, italics, colour, alignment, bullet "
    "styling or whether a logo is blurred — NEVER emit an issue asserting "
    "those; a separate screenshot-based pass checks them. "
    "SPEC ABSENCE — if the OFFICIAL SPECIFICATION block does not contain a value "
    "(GLH, TQT, credits, access duration, qualification number, etc.), SILENTLY "
    "SKIP that rule. Never say the spec 'does not provide' a value, never tell the "
    "reader to 'verify and update' it. An ACCESS DURATION of 2 years / 24 months "
    "for an Extended Diploma is CORRECT — never flag it. "
    "BAND CHECKS — a value that sits WITHIN a stated band is fine; only flag a "
    "value that falls OUTSIDE the band (e.g. '9-12 months' is inside a 9-12 month "
    "band, so it is NOT an issue)."
)

SCHEMA_INSTRUCTION = """Return a JSON object:
{
  "issues": [
    {
      "ruleId": "<rule id from the template>",
      "type": "Template",
      "severity": "Critical" | "Minor" | "Info",
      "description": "<what is wrong on the page>",
      "suggestion": "<how to fix it>",
      "excerpt": "<short quote from the page that shows the problem, or empty>"
    }
  ]
}
Phrase every `suggestion` as a direct, actionable edit instruction in the QA
team's house style — e.g. "Course pricing has to be added.", "Update this
section.", "Remove this section.", "Add these FAQs.", "Capitalise the unit
names." — not as an observation. Keep `description` to the specific problem on
the page. Output ONLY the JSON object. If the page passes every rule, return
{"issues": []}."""


# Phrases that mark an entry as a non-issue (a pass / confirmation) or as work
# punted back to a human. Either way it must not reach the report — the page is
# the agent's to check, and only genuine violations belong in the issue list.
_NON_ISSUE_MARKERS = re.compile(
    r"is\s+correct|no\s+(?:change|action|issue)s?\s+(?:needed|required)|"
    r"(?:rule|requirement|page|item|section|this)\s+(?:is\s+)?(?:met|compl(?:y|ies|iant)|passe[sd])|"
    r"compliant\b|looks?\s+fine|appears?\s+correct|consistency\s+check|"
    # Confirmations that the content is fine / already there — these describe a
    # PASS, not a violation, and must never be emitted as an issue (a "correctly
    # states …" entry was showing up, even as Critical). Kept unambiguous and
    # negation-safe: a real defect is phrased "does NOT state / is missing", not
    # "correctly states" / "has been added".
    r"correctly\s+(?:state|display|show|list|mention|note|present)s?\b|"
    r"already\s+(?:present|added|included|stated|shown|listed|displayed|there)|"
    r"(?:section|content|statement|information)\s+(?:is|are)\s+present\b|"
    r"(?:has|have)\s+been\s+(?:added|included|provided)|"
    r"\bno\s+(?:issue|violation)\b|"
    # Self-negating date findings: the model emits a 'future date' entry whose
    # own description concludes the dates are fine ("All dates appear logical
    # and real. No future dates are present.", "which is in the past"). A
    # finding that argues itself out of existence is a pass, not an issue.
    r"no\s+future\s+dates?\b|"
    r"all\s+(?:review\s+)?dates?\s+(?:appear|are|look|seem)s?\s+(?:to\s+be\s+)?"
    r"(?:logical|real|valid|correct|plausible|in\s+the\s+past)|"
    r"which\s+(?:is|are)\s+(?:also\s+)?in\s+the\s+past|"
    r"(?:is|are)\s+also\s+in\s+the\s+past|"
    r"dates?\s+(?:is|are)\s+(?:real|logical|valid)\b|"
    r"just\s+confirming|verify\s+manually|manual(?:ly)?\s+(?:verif|check|review)|"
    r"needs?\s+(?:manual|human)|unable\s+to\s+(?:verify|confirm|determine)|"
    r"cannot\s+(?:verify|confirm|be\s+verified)|could\s+not\s+(?:be\s+)?(?:verif|confirm)|"
    r"check\s+against\s+the\s+specification|"
    # Spec-absence deferrals: the spec_lookup didn't surface a value, so the model
    # punts it back to the reader. That is never an issue — silently skip it.
    r"do(?:es)?\s+not\s+provide|not\s+provided\s+(?:in|for)|"
    r"verify\s+and\s+(?:update|confirm)|cannot\s+be\s+(?:determined|established)|"
    r"not\s+(?:available|provided)\s+for\s+verification|"
    r"the\s+spec(?:ification)?\s+(?:does\s+not|doesn't)",
    re.I,
)


def _is_real_issue(issue: dict) -> bool:
    """Drop pass/confirmation entries and items deferred to manual verification."""
    if not isinstance(issue, dict):
        return False
    blob = f"{issue.get('description', '')} {issue.get('suggestion', '')}"
    if _NON_ISSUE_MARKERS.search(blob):
        return False
    # An "issue" with no description is noise.
    return bool(str(issue.get("description", "")).strip())


def _dedupe_issues(issues: list[dict]) -> list[dict]:
    """Drop near-duplicate findings.

    Baseline checklist rules and template-derived rules often overlap (e.g.
    baseline 'S04-2' and a template 'R16' both flag the Learning Outcomes), so
    the same problem can be reported twice under different rule IDs. We collapse
    issues whose subject is the same, keying on the normalised description and
    the normalised excerpt (rule ID intentionally excluded from the key). A
    same-type finding whose excerpt CONTAINS (or is contained by) an earlier
    finding's excerpt is the same spot flagged twice with different wording
    (e.g. 'one-on-one mentoring' vs 'one-on-one mentoring for up to 12 months')
    and is dropped too.
    """
    seen: set[str] = set()
    seen_excerpts: list[tuple[str, str]] = []  # (excerpt, issue type)
    out: list[dict] = []
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        desc = re.sub(r"\s+", " ", str(issue.get("description", ""))).strip().lower()
        excerpt = re.sub(r"\s+", " ", str(issue.get("excerpt", ""))).strip().lower()
        itype = str(issue.get("type", "")).strip().lower()
        # Two findings pointing at the same non-empty page excerpt are almost
        # always the same problem reported under two overlapping rules.
        if excerpt and any(excerpt == prev for prev, _ in seen_excerpts):
            continue
        if excerpt and len(excerpt) >= 10 and any(
            ptype == itype and (excerpt in prev or prev in excerpt)
            for prev, ptype in seen_excerpts if len(prev) >= 10
        ):
            continue
        # Otherwise key on the first chunk of the description plus the excerpt —
        # catches the same finding worded slightly differently across two rules.
        key = f"{desc[:80]}|{excerpt[:60]}"
        if key in seen:
            continue
        seen.add(key)
        if excerpt:
            seen_excerpts.append((excerpt, itype))
        out.append(issue)
    return out


# A currency value anywhere on the page means pricing IS present. £1,099.00 etc.
_PRICE_VALUE_RE = re.compile(
    r"(?:£|GBP|EUR|€|USD|\$)\s?\d[\d,]*(?:\.\d{1,2})?|\d[\d,]*\s?(?:GBP|USD|EUR|pounds?)",
    re.I,
)
# Wording that claims pricing is absent / needs adding.
_PRICE_MENTION_RE = re.compile(r"\bpric(?:e|es|ing)\b", re.I)
_MISSING_CLAIM_RE = re.compile(
    r"\b(?:not\s+(?:added|present|shown|displayed|included|available)|missing|absent|"
    r"no\s+pric|needs?\s+to\s+be\s+added|should\s+be\s+added|has\s+to\s+be\s+added|"
    r"add\s+(?:the\s+|a\s+|course\s+)?pric)\b",
    re.I,
)


def _has_price_evidence(price_candidates, banner_evidence, image_evidence,
                        page_text: str) -> bool:
    """True if a price/currency value is present anywhere we can see it."""
    if price_candidates:
        return True
    for coll in (banner_evidence or [], image_evidence or []):
        for ev in coll:
            if not isinstance(ev, dict):
                continue
            if "price" in (ev.get("claim_types") or []):
                return True
            text = f"{ev.get('cleaned_combined_text', '')} {ev.get('ocr_excerpt', '')}"
            if _PRICE_VALUE_RE.search(text):
                return True
    return bool(_PRICE_VALUE_RE.search(page_text or ""))


# Salary/earnings context around a currency value — 'Career Progression'
# sections are full of these and none of them is the course fee.
_SALARY_CTX_RE = re.compile(
    r"\b(?:salary|salaries|earn(?:ing)?s?|wage|income|per\s+(?:year|annum|month|hour)|"
    r"p\.?a\.?|annually|annual|pension)\b",
    re.I,
)


def _page_has_nonsalary_price(page_text: str) -> bool:
    """A currency value on the page that is NOT a salary figure.

    Stricter than `_has_price_evidence` (which is deliberately broad because it
    only SUPPRESSES false 'pricing missing' findings): this one decides whether
    to ASSERT that pricing is missing, so career-progression salaries must not
    count as the course fee.
    """
    for m in _PRICE_VALUE_RE.finditer(page_text or ""):
        window = page_text[max(0, m.start() - 45): m.end() + 25]
        if not _SALARY_CTX_RE.search(window):
            return True
    return False


def _is_false_pricing_issue(issue: dict, has_price: bool) -> bool:
    """Suppress a 'pricing not added' finding when a price is actually present.

    The pricing block (price + Buy Now + Enquire Now) is easily confused with a
    separate 'Enquire Now' / 'Apply Now' call-to-action block, which led to a
    real price being reported as missing. With a price on the page, any
    'pricing missing / needs adding' claim is wrong.
    """
    if not has_price:
        return False
    rid = str(issue.get("ruleId", "")).upper().strip()
    if rid == "S08-1":
        return True
    blob = f"{issue.get('description', '')} {issue.get('suggestion', '')}"
    return bool(_PRICE_MENTION_RE.search(blob) and _MISSING_CLAIM_RE.search(blob))


def _format_rules(rules: list) -> str:
    """One compact line per rule instead of indented JSON.

    The rule list is the second-largest block in the compliance prompt (after
    the page text); `json.dumps(indent=2)` spent tokens on braces, quotes and
    whitespace that carry no signal. Same information, ~40% fewer tokens.
    """
    lines: list[str] = []
    for r in rules or []:
        if not isinstance(r, dict):
            continue
        flags = str(r.get("severity", "Info"))
        if r.get("needs_spec"):
            flags += ", needs_spec"
        lines.append(f"[{r.get('id', '?')}] ({flags}) {r.get('rule', '')}")
    return "\n".join(lines) or "(none)"


def _format_price_candidates(price_candidates: list[str] | None) -> str:
    if not price_candidates:
        return "(none)"
    return "\n".join(f"- {c}" for c in price_candidates[:8])


def _format_banner_evidence(banners: list | None) -> str:
    """Render high-priority banner evidence (incl. image-embedded OCR text + claims)."""
    if not banners:
        return "(none)"
    lines: list[str] = []
    for b in banners[:20]:
        if not isinstance(b, dict):
            continue
        claims = b.get("claim_types") or [
            k for k, v in (b.get("claims_detected") or {}).items() if v
        ]
        text = b.get("cleaned_combined_text") or b.get("visible_text_html") or ""
        lines.append(
            f"- [{b.get('banner_type', '?')} @ {b.get('page_position', '?')}] "
            f"text=\"{text[:200]}\" cta=\"{b.get('cta_text', '')}\" "
            f"-> {b.get('cta_url', '')} claims={claims}"
        )
    return "\n".join(lines) or "(none)"


def _format_image_evidence(images: list | None) -> str:
    """Render high-priority image evidence (logos, badges, OCR text, claims)."""
    if not images:
        return "(none)"
    lines: list[str] = []
    for i in images[:30]:
        if not isinstance(i, dict):
            continue
        ocr = i.get("ocr_excerpt") or i.get("cleaned_ocr_text") or ""
        lines.append(
            f"- [{i.get('image_type', '?')}] alt=\"{i.get('alt_text', '')}\" "
            f"ocr=\"{ocr[:160]}\" claims={i.get('claim_types') or []} "
            f"src={i.get('resolved_url', '')}"
        )
    return "\n".join(lines) or "(none)"


def _format_specification(spec: dict | None) -> str:
    """Render the web-sourced Qualification Specification for needs_spec rules."""
    if not spec or not isinstance(spec, dict):
        return "(not available — silently skip needs_spec rules, do not raise them)"
    details = spec.get("specification") if isinstance(spec.get("specification"), dict) else spec
    if not spec.get("found", True):
        return "(web search did not confirm a specification — silently skip needs_spec rules, do not raise them)"
    lines = [f"- {k}: {v}" for k, v in (details or {}).items() if str(v).strip()]
    src = spec.get("source_urls") or []
    if src:
        lines.append(f"- sources: {', '.join(src[:3])}")
    return "\n".join(lines) or "(specification empty — silently skip needs_spec rules, do not raise them)"


def _norm(s: str) -> str:
    """Lowercase + collapse to alphanumeric words for tolerant matching."""
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


# A finding that asserts a section / heading / content is absent from the page.
_ABSENCE_RE = re.compile(
    r"\b(?:not\s+(?:present|found|shown|included|listed|visible|provided)|"
    r"missing|absent|is\s+not\s+in\s+the|could\s+not\s+find|no\s+\w+\s+section)\b",
    re.I,
)
# Hedging that reveals an "absent" claim rests only on the (truncated) text we
# supplied — an extraction artefact, never proof the page lacks the section.
_TRUNCATION_HEDGE_RE = re.compile(
    r"provided\s+(?:headings|text)|truncated|"
    r"in\s+the\s+(?:provided|given|supplied)\s+(?:page\s+)?text|"
    r"(?:provided|truncated)\s+page\s+text",
    re.I,
)
# Visual-only baseline rules — even alignment / font style / bullet styling /
# unblurred-logo checks — cannot be judged from extracted text. Asserting them
# from a text-only review is fabrication, so we never emit them here; the
# screenshot-based vision pass (vision_tool.py) checks them instead.
# S02-1 (awarding body bold) is ALSO suppressed from this LLM pass, but not
# because it is vision-only: it is checked DETERMINISTICALLY in the pipeline
# against the DOM's bold_texts list, so an LLM duplicate would only add noise.
_VISUAL_ONLY_RULES = {"S02-1", "S03-2", "S09-1", "S09-3", "S09-4", "S09-5"}

# Rule text that marks a template-derived rule as visual (alignment, fonts,
# bullet styling, logo clarity) — routed to the vision pass too. Bold rules are
# deliberately NOT here: bold is a DOM fact (font-weight), extracted verbatim
# into bold_texts and checked from text — asking a vision model to eyeball font
# weight from downscaled screenshots produced false positives.
_VISUAL_RULE_TEXT_RE = re.compile(
    r"\balign(?:ment|ed)?\b|\bfont\b|\bblur(?:red|ry)?\b|"
    r"\bbullet\b|\blogo\b|\bconsistent\s+styl",
    re.I,
)
# Deterministically checked in the pipeline — never sent to the vision pass.
_DETERMINISTIC_RULES = {"S02-1"}


def select_visual_rules(rules: list) -> list[dict]:
    """The subset of rules that need EYES — checked by the vision pass, not text."""
    out: list[dict] = []
    for r in rules or []:
        if not isinstance(r, dict):
            continue
        rid = str(r.get("id", "")).upper().strip()
        if rid in _DETERMINISTIC_RULES:
            continue
        text = str(r.get("rule", ""))
        if rid in _VISUAL_ONLY_RULES or _VISUAL_RULE_TEXT_RE.search(text):
            out.append(r)
    return out


def _quoted_terms(issue: dict) -> list[str]:
    blob = f"{issue.get('description', '')} {issue.get('suggestion', '')}"
    return [t.strip() for t in re.findall(r"['\"]([^'\"]{3,70})['\"]", blob) if t.strip()]


def _section_present(name: str, hay: str) -> bool:
    words = [w for w in _norm(name).split() if len(w) > 2]
    return bool(words) and all(w in hay for w in words)


def _is_absence_contradicted(issue: dict, evidence_hay: str) -> bool:
    """Drop a 'section/content missing' finding the page evidence contradicts.

    The PAGE TEXT is truncated and accordion panels may still be collapsed, so
    absence from it proves nothing; the complete PAGE HEADINGS list (folded into
    ``evidence_hay``) is the source of truth. If the finding openly hedges on the
    truncated text, or the named section's words appear in the headings/text, the
    'missing' claim is an extraction artefact, not a real defect.
    """
    desc = str(issue.get("description", ""))
    if not _ABSENCE_RE.search(desc):
        return False
    if _TRUNCATION_HEDGE_RE.search(f"{desc} {issue.get('suggestion', '')}"):
        return True
    return any(_section_present(name, evidence_hay) for name in _quoted_terms(issue))


def _is_visual_only_issue(issue: dict) -> bool:
    """True for baseline rules that need visual inspection we cannot do from text."""
    return str(issue.get("ruleId", "")).upper().strip() in _VISUAL_ONLY_RULES


_HEADING_CLAIM_RE = re.compile(r"\bheading\b|\bheader\b|\btitled?\b", re.I)


def _is_false_heading_issue(issue: dict, evidence_hay: str) -> bool:
    """Drop a wrong/missing-heading finding when the required heading IS present.

    Course pages use a two-column layout: the LEFT cell is a section label
    (e.g. 'Method of Assessment') and the section's own heading sits at the top
    of the right-hand content (e.g. 'Assessment Overview'). The model reads the
    label as "the heading" and flags the rule even though the required heading
    is right there. If EVERY heading the finding quotes actually appears in the
    page headings/text, the demanded heading exists — the finding is a layout
    misreading, not a defect. A genuinely missing heading is absent from the
    evidence, so that finding survives.
    """
    blob = f"{issue.get('description', '')} {issue.get('suggestion', '')}"
    if not _HEADING_CLAIM_RE.search(blob):
        return False
    quoted = _quoted_terms(issue)
    if not quoted:
        return False
    return all(_section_present(q, evidence_hay) for q in quoted)


# --- Deterministic review-date check (S07-1) --------------------------------
# The model cannot reliably compare dates (it insisted "June 6, 2026 is after
# July 1, 2026"), so we NEVER trust its future/past arithmetic. We parse the date
# out of the finding and compare it to today in code.
_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12, "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))
# "June 6, 2026" / "Jun 6 2026" / "6 June 2026"
_MDY_RE = re.compile(
    rf"\b({_MONTH_ALT})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.I)
_DMY_RE = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_ALT})\.?,?\s+(\d{{4}})\b", re.I)
# The model marks a date as wrong because it thinks it hasn't happened yet.
_FUTURE_CLAIM_RE = re.compile(
    r"\bfuture\b|\bnot\s+yet\b|\bhas\s+not\s+(?:happened|occurred|passed)\b|"
    r"\byet\s+to\s+(?:happen|occur|come)\b", re.I)


def _parse_dates(text: str) -> list["date"]:
    out: list[date] = []
    for mo, dy, yr in _MDY_RE.findall(text or ""):
        try:
            out.append(date(int(yr), _MONTHS[mo.lower()], int(dy)))
        except (ValueError, KeyError):
            continue
    for dy, mo, yr in _DMY_RE.findall(text or ""):
        try:
            out.append(date(int(yr), _MONTHS[mo.lower()], int(dy)))
        except (ValueError, KeyError):
            continue
    return out


# Month + year with no day ("Jul 2035") — the checklist's own example of an
# accidental future date. Compared at month granularity.
_MY_RE = re.compile(rf"\b({_MONTH_ALT})\.?\s+(\d{{4}})\b", re.I)


def _page_has_future_date(page_text: str, today: date) -> bool:
    """True if ANY date on the page is after today (full dates or month+year).

    Used as a deterministic gate: when the page contains no future date at all,
    every 'date is in the future' finding is definitionally wrong and is
    suppressed in code — the model's date arithmetic is never trusted.
    """
    for d in _parse_dates(page_text or ""):
        if d > today:
            return True
    for mo, yr in _MY_RE.findall(page_text or ""):
        try:
            y, m = int(yr), _MONTHS[mo.lower()]
        except (ValueError, KeyError):
            continue
        if (y, m) > (today.year, today.month):
            return True
    return False


def _is_false_future_date_issue(issue: dict, today: date,
                                page_has_future: bool) -> bool:
    """Drop a 'date is in the future' finding not backed by an actual future date.

    Two layers: (1) if NO date anywhere on the page is after today, any future-
    date claim is wrong — suppress it outright. (2) Otherwise read the dates in
    the finding's own excerpt/description; if every one is on or before today,
    the claim is wrong for THIS finding. A genuinely future date (e.g. 2035)
    survives both checks, so real findings are kept.
    """
    blob = f"{issue.get('description', '')} {issue.get('suggestion', '')}"
    if not _FUTURE_CLAIM_RE.search(blob):
        return False
    if not page_has_future:
        return True
    dates = _parse_dates(f"{issue.get('excerpt', '')} {issue.get('description', '')}")
    if not dates:
        return False
    return all(d <= today for d in dates)


# A finding that cross-compares the ACCESS DURATION (how long the learner can
# use the platform — e.g. 1 year) with the AVERAGE COMPLETION TIMEFRAME (how
# long a typical learner takes — e.g. 6-9 months). These are different concepts:
# completing faster than the access window is normal, so any "mismatch /
# contradiction" between the two is the model confusing them, never a defect.
_ACCESS_DURATION_RE = re.compile(r"\baccess\s+(?:duration|period)\b|\bcourse\s+access\b", re.I)
_COMPLETION_TIME_RE = re.compile(
    r"\bcomplet(?:e|ion|ing)\b.{0,40}\b(?:time(?:frame)?|months?|duration)\b|"
    r"\baverage\s+completion\b|\bcompletion\s+time(?:frame)?\b",
    re.I,
)
_MISMATCH_RE = re.compile(
    r"mismatch|does\s+not\s+match|contradict|inconsisten|conflict|"
    r"differs?\s+from|not\s+(?:the\s+same|aligned)|should\s+(?:be|match|state)",
    re.I,
)


def _is_duration_confusion_issue(issue: dict) -> bool:
    """Drop a finding that flags the completion timeframe against the access
    duration (or vice versa) as a mismatch."""
    blob = f"{issue.get('description', '')} {issue.get('suggestion', '')}"
    return bool(
        _ACCESS_DURATION_RE.search(blob)
        and _COMPLETION_TIME_RE.search(blob)
        and _MISMATCH_RE.search(blob)
    )


def _is_false_extended_duration_issue(issue: dict, course_identity: dict | None,
                                      page_text: str) -> bool:
    """Suppress an access-duration 'mismatch' on an Extended Diploma showing 2 years.

    Per baseline S01-6 an Extended Diploma runs 2 years; when the page already
    shows ~2 years / 24 months that value is correct, and a disagreement only
    arises from a wrong-variant specification (a plain Diploma's shorter
    duration). Never flag the correct value as wrong.
    """
    variant = _norm((course_identity or {}).get("variant", ""))
    if "extended" not in variant:
        return False
    rid = str(issue.get("ruleId", "")).upper().strip()
    blob = _norm(f"{issue.get('description', '')} {issue.get('suggestion', '')}")
    if "duration" not in blob and rid != "S01-6":
        return False
    return bool(re.search(r"\b2\s*years?\b|\b24\s*months?\b", _norm(page_text)))


def _format_bold_texts(bold_texts: list | None) -> str:
    if not bold_texts:
        return "(none captured)"
    return "\n".join(f"- {t}" for t in bold_texts[:120] if str(t).strip())


def check_compliance(page_text: str, headings: list, rules: list,
                     price_candidates: list[str] | None = None,
                     banner_evidence: list | None = None,
                     image_evidence: list | None = None,
                     specification: dict | None = None,
                     course_identity: dict | None = None,
                     bold_texts: list | None = None) -> dict:
    if not rules:
        return {"issues": []}
    compact_headings = "\n".join(f"{h.get('tag', '').upper()}: {h.get('text', '')}" for h in (headings or []))
    today = datetime.now(timezone.utc)
    prompt = (
        f"{SCHEMA_INSTRUCTION}\n\n"
        f"TODAY'S DATE is {today:%A, %d %B %Y}. Use THIS date for every date check: "
        f"a review/publication date is only 'in the future' if it is AFTER today, and "
        f"only 'in the past' if it is BEFORE today. Do NOT guess the current date.\n\n"
        f"TEMPLATE RULES (id, severity, needs_spec flag, rule):\n{_format_rules(rules)}\n\n"
        f"OFFICIAL SPECIFICATION (for needs_spec rules):\n"
        f"{_format_specification(specification)}\n\n"
        f"PAGE HEADINGS:\n{compact_headings or '(none)'}\n\n"
        f"BOLD TEXT RUNS (verbatim text the rendered page shows in bold):\n"
        f"{_format_bold_texts(bold_texts)}\n\n"
        f'PRICE CANDIDATES:\n{_format_price_candidates(price_candidates)}\n\n'
        f"BANNER / HERO / PROMO EVIDENCE (visible + image-embedded OCR text):\n"
        f"{_format_banner_evidence(banner_evidence)}\n\n"
        f"IMAGE EVIDENCE (logos, accreditation/trust badges, thumbnails, OCR text):\n"
        f"{_format_image_evidence(image_evidence)}\n\n"
        f"PAGE TEXT (complete rendered page body; per-question FAQ answers "
        f"appended at the end):\n"
        f'"""{_truncate_head_at_word(page_text or "", _PAGE_TEXT_READ_CHARS)}"""'
    )
    # NOTE: deliberately NOT wrapped in a graceful fallback. Unlike the spec /
    # template tools (which can safely degrade), an empty compliance result is
    # indistinguishable from "the page passed every rule" — swallowing an LLM
    # failure here would hide the whole checklist and produce a falsely-clean
    # report. On failure (e.g. a persistent OpenRouter 429, surfaced as
    # RateLimitError) we let it propagate so the run records an honest
    # tool_failure instead of silently passing the page. The client already
    # retries 429s with Retry-After-aware backoff, so this is a last resort.
    result = call_llm_json(
        prompt, system=SYSTEM, max_prompt_bytes=_COMPLIANCE_PROMPT_BYTES,
        read_timeout=_COMPLIANCE_READ_TIMEOUT, retry_budget=_COMPLIANCE_RETRY_BUDGET,
    )
    issues = result.get("issues") or []
    if not isinstance(issues, list):
        return {"issues": []}
    has_price = _has_price_evidence(price_candidates, banner_evidence, image_evidence, page_text)
    # The headings list is COMPLETE (unlike the truncated page text), so it is the
    # authority for whether a section exists. Fold it in with the visible text so
    # a "section missing" claim can be checked against real evidence.
    evidence_hay = _norm(f"{compact_headings}\n{page_text}")
    today = datetime.now(timezone.utc).date()
    page_has_future = _page_has_future_date(page_text, today)
    kept = [
        i for i in issues
        if _is_real_issue(i)
        and not _is_false_pricing_issue(i, has_price)
        and not _is_absence_contradicted(i, evidence_hay)
        and not _is_visual_only_issue(i)
        and not _is_false_extended_duration_issue(i, course_identity, page_text)
        and not _is_duration_confusion_issue(i)
        and not _is_false_heading_issue(i, evidence_hay)
        and not _is_false_future_date_issue(i, today, page_has_future)
    ]
    # Missing pricing is fully deterministic, so it is asserted in CODE: the
    # model's anti-false-positive instructions made it under-report a genuinely
    # absent price, and the reverse (flagging a present price as missing) is
    # already suppressed above. Asserted only when a pricing rule is in the
    # rule set, the model didn't already flag it, and NO non-salary currency
    # value exists in the candidates, the banner/image OCR evidence, or the
    # page text (career-progression salaries must not count as the fee — note
    # this is stricter than `has_price`, which is deliberately broad because it
    # only suppresses).
    price_definitely_present = (
        _has_price_evidence(price_candidates, banner_evidence, image_evidence, "")
        or _page_has_nonsalary_price(page_text or "")
    )
    if not price_definitely_present and _rules_include_pricing(rules) \
            and not any(_PRICE_MENTION_RE.search(str(i.get("description", ""))) for i in kept):
        kept.append({
            "ruleId": "S08-1",
            "type": "Template",
            "severity": "Critical",
            "description": "No course price appears anywhere on the page — not in "
                           "the body text, banners, or text embedded in images.",
            "suggestion": "Course pricing has to be added.",
            "excerpt": "",
        })
    return {"issues": _dedupe_issues(kept)}


def _rules_include_pricing(rules: list) -> bool:
    for r in rules or []:
        if not isinstance(r, dict):
            continue
        if str(r.get("id", "")).upper().strip() == "S08-1":
            return True
        if re.search(r"\bpricing\b.*\badded\b|\bprice\b.*\badded\b",
                     str(r.get("rule", "")), re.I):
            return True
    return False
