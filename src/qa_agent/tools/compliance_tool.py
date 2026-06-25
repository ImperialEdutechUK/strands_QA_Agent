from __future__ import annotations

import json
import re

from ..llm_client import call_llm_json

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
    "GROUNDING — this is critical: the PAGE TEXT you are given is TRUNCATED "
    "(the middle of long pages is dropped), so a section being absent from the "
    "PAGE TEXT does NOT mean it is missing from the page. Decide whether a "
    "section / heading exists by looking at the full PAGE HEADINGS list, which "
    "is complete. If a matching heading is present, the section EXISTS — never "
    "flag it as missing. Only claim something is 'missing' or 'not present' when "
    "it is genuinely absent from BOTH the headings and the visible evidence. "
    "Be conservative: if you cannot clearly see a violation in the evidence "
    "provided, emit NOTHING for that rule — do not guess, and do not invent a "
    "generic list of missing sections. A correct review of a complete page "
    "usually finds only a handful of real issues, not dozens. "
    "PRICING (rule about pricing being added): if ANY price/currency value "
    "(e.g. '£1,099.00', '£499') appears anywhere in the PRICE CANDIDATES, the "
    "banner/image evidence, or the page text, then pricing HAS been added — do "
    "NOT flag it as missing. The pricing block is the one carrying the price plus "
    "the 'Buy Now' and 'Enquire Now' buttons; do NOT confuse it with a separate "
    "call-to-action block that shows 'Enquire Now' and 'Apply Now' (that block is "
    "unrelated to whether a price is present). Only flag pricing as missing when "
    "there is genuinely no price anywhere on the page."
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
    r"just\s+confirming|verify\s+manually|manual(?:ly)?\s+(?:verif|check|review)|"
    r"needs?\s+(?:manual|human)|unable\s+to\s+(?:verify|confirm|determine)|"
    r"cannot\s+(?:verify|confirm|be\s+verified)|could\s+not\s+(?:be\s+)?(?:verif|confirm)|"
    r"check\s+against\s+the\s+specification",
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
    the normalised excerpt (rule ID intentionally excluded from the key).
    """
    seen: set[str] = set()
    seen_excerpts: set[str] = set()
    out: list[dict] = []
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        desc = re.sub(r"\s+", " ", str(issue.get("description", ""))).strip().lower()
        excerpt = re.sub(r"\s+", " ", str(issue.get("excerpt", ""))).strip().lower()
        # Two findings pointing at the same non-empty page excerpt are almost
        # always the same problem reported under two overlapping rules.
        if excerpt and excerpt in seen_excerpts:
            continue
        # Otherwise key on the first chunk of the description plus the excerpt —
        # catches the same finding worded slightly differently across two rules.
        key = f"{desc[:80]}|{excerpt[:60]}"
        if key in seen:
            continue
        seen.add(key)
        if excerpt:
            seen_excerpts.add(excerpt)
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


def check_compliance(page_text: str, headings: list, rules: list,
                     price_candidates: list[str] | None = None,
                     banner_evidence: list | None = None,
                     image_evidence: list | None = None,
                     specification: dict | None = None) -> dict:
    if not rules:
        return {"issues": []}
    compact_headings = "\n".join(f"{h.get('tag', '').upper()}: {h.get('text', '')}" for h in (headings or []))
    prompt = (
        f"{SCHEMA_INSTRUCTION}\n\n"
        f"TEMPLATE RULES:\n{json.dumps(rules, indent=2)}\n\n"
        f"OFFICIAL SPECIFICATION (for needs_spec rules):\n"
        f"{_format_specification(specification)}\n\n"
        f"PAGE HEADINGS:\n{compact_headings or '(none)'}\n\n"
        f'PRICE CANDIDATES:\n{_format_price_candidates(price_candidates)}\n\n'
        f"BANNER / HERO / PROMO EVIDENCE (visible + image-embedded OCR text):\n"
        f"{_format_banner_evidence(banner_evidence)}\n\n"
        f"IMAGE EVIDENCE (logos, accreditation/trust badges, thumbnails, OCR text):\n"
        f"{_format_image_evidence(image_evidence)}\n\n"
        f'PAGE TEXT (truncated):\n"""{(page_text or "")[:8000]}"""'
    )
    result = call_llm_json(prompt, system=SYSTEM)
    issues = result.get("issues") or []
    if not isinstance(issues, list):
        return {"issues": []}
    has_price = _has_price_evidence(price_candidates, banner_evidence, image_evidence, page_text)
    kept = [
        i for i in issues
        if _is_real_issue(i) and not _is_false_pricing_issue(i, has_price)
    ]
    return {"issues": _dedupe_issues(kept)}
