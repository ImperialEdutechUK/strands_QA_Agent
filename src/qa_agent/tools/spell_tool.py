import re

from ..llm_client import call_llm_json

SYSTEM = (
    "You are a meticulous UK English copy editor reviewing course web page content. "
    "Use UK English spelling (colour, organisation, analyse, behaviour, programme). "
    "Flag every spelling mistake, grammar error, punctuation issue, inconsistent "
    "tense, or awkward phrasing. Ignore navigation labels, cookie notices, and "
    "footer boilerplate.\n\n"
    "UK ENGLISH IS CORRECT — do not get this backwards:\n"
    "  * UK spellings such as colour, behaviour, organisation, organise, "
    "    recognise, specialise, analyse, programme, centre, licence, defence, "
    "    catalogue, fulfil, enrolment, practise (verb), travelling, learnt, "
    "    skilful are ALL CORRECT. NEVER flag a UK spelling as an error or claim "
    "    it is a US spelling. Only flag a word when it is the actual US form "
    "    (color, behavior, organization, analyze, center, license as a noun, "
    "    catalog, fulfillment, enrollment) and suggest the UK form.\n"
    "  * Before emitting a Spelling issue, re-read the cited word and be certain "
    "    it is genuinely misspelt. If the word is correctly spelt in UK English, "
    "    DO NOT emit anything for it.\n\n"
    "STRICT FILTER — only emit an issue if there is a real, fixable problem:\n"
    "  * NEVER emit an issue whose description says 'is correct', 'no change "
    "    needed', 'just confirming', 'consistency check', 'looks fine', or similar.\n"
    "  * NEVER emit issues where excerpt and suggestion are the same string.\n"
    "  * The text was extracted from a rendered web page, so adjacent layout "
    "    elements (table cells, badges, header bars, buttons, menu items, "
    "    multi-column lists) can appear glued together. NEVER flag a 'missing "
    "    space', 'words run together', or spacing/punctuation problem that is "
    "    explained by two separate UI labels being concatenated — e.g. "
    "    'INCLUDEDCOURSE', 'NowEnquire', a run-together of two ALL-CAPS labels, "
    "    or a join across what are clearly two distinct on-screen items. These "
    "    are extraction artifacts, not authored mistakes.\n"
    "  * Only flag spelling/grammar/punctuation inside genuine running prose "
    "    (real sentences in body copy). Do NOT flag isolated ALL-CAPS labels, "
    "    headings, badge text, course codes, or short UI strings.\n"
    "  * When in doubt about whether something is a real authored error or a "
    "    layout artifact, DO NOT emit it.\n"
    "  * If the text is already correct UK English, return {\"issues\": []}."
)

SCHEMA_INSTRUCTION = """Return a JSON object with this exact shape:
{
  "issues": [
    {
      "type": "Spelling" | "Grammar" | "Punctuation" | "Style",
      "severity": "Critical" | "Minor" | "Info",
      "excerpt": "<the offending text, kept short>",
      "description": "<what is wrong>",
      "suggestion": "<the corrected text>"
    }
  ]
}
If there are no issues, return {"issues": []}. Output ONLY the JSON object."""


# The model keeps emitting "explanations" of why something is FINE — e.g.
# "'programme' is correct, no change needed", "the word is not present, no issue
# to flag". Those are not issues and must never reach the report. This is a hard
# code-level filter so a chatty model can't leak them past the prompt rules.
_NON_ISSUE_RE = re.compile(
    r"\bno\s+(?:issue|change|error|problem|stylistic|action|need)\b|"
    r"\bis\s+correct\b|\bare\s+correct\b|\bcorrect\s+spelling\b|"
    r"correctly\s+(?:spel|written|used|punctuat|capitalis)|"
    r"\bnot\s+present\b|\bno\s+change\s+needed\b|\balready\s+correct\b|"
    r"\bnothing\s+to\s+(?:fix|change|flag)\b|\bgrammatically\s+correct\b|"
    r"\bspelt\s+correctly\b|\bspelled\s+correctly\b|\bno\s+issue\s+to\s+flag\b",
    re.I,
)


def _is_real_spell_issue(issue: dict) -> bool:
    """Keep only genuine, fixable spelling/grammar/punctuation problems."""
    if not isinstance(issue, dict):
        return False
    blob = f"{issue.get('description', '')} {issue.get('suggestion', '')}"
    if _NON_ISSUE_RE.search(blob):
        return False
    excerpt = (issue.get("excerpt") or "").strip()
    suggestion = (issue.get("suggestion") or "").strip()
    # A "fix" identical to the original text is a non-issue.
    if excerpt and suggestion and excerpt.lower() == suggestion.lower():
        return False
    return bool(str(issue.get("description", "")).strip())


def check_spelling(text: str) -> dict:
    trimmed = (text or "")[:12000]
    prompt = f'{SCHEMA_INSTRUCTION}\n\nTEXT TO REVIEW:\n"""{trimmed}"""'
    result = call_llm_json(prompt, system=SYSTEM)
    issues = result.get("issues") or []
    if not isinstance(issues, list):
        return {"issues": []}
    return {"issues": [i for i in issues if _is_real_spell_issue(i)]}
