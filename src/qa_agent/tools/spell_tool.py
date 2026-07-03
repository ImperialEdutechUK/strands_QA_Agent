import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor

from ..llm_client import call_llm_json
from .web_tools import _truncate_head_at_word

logger = logging.getLogger(__name__)

# Spellcheck the WHOLE page, not just the first screenful — a typo in the
# mid-page assessment/curriculum sections was never seen under the old 12k head
# cap. The full text is cached server-side and reviewed in one call. Override via
# QA_SPELL_READ_CHARS / QA_SPELL_PROMPT_BYTES.
_SPELL_READ_CHARS = int(os.environ.get("QA_SPELL_READ_CHARS", "1000000"))
_SPELL_PROMPT_BYTES = int(os.environ.get("QA_SPELL_PROMPT_BYTES", str(2 * 1024 * 1024)))
# Pages longer than this are reviewed in several parallel LLM calls instead of
# one giant one. One huge prompt made the provider occasionally return an EMPTY
# body (surfacing as "LLM did not return valid JSON"), and a single failure lost
# the whole spell pass. Spelling/grammar issues are local to their sentence, so
# chunking at paragraph boundaries loses nothing; every chunk still gets the
# full system prompt. Chunks run concurrently, so this is also faster.
_SPELL_CHUNK_CHARS = int(os.environ.get("QA_SPELL_CHUNK_CHARS", "28000"))
_SPELL_MAX_WORKERS = max(1, int(os.environ.get("QA_SPELL_MAX_WORKERS", "3")))

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
    "  * The text may be TRUNCATED for length — you may see a marker like "
    "    '... [content truncated; tail preserved] ...', or the text may simply "
    "    stop. A word or sentence cut off at the very start or end of the text, "
    "    or right next to a truncation marker (e.g. a trailing 'qualifications d' "
    "    where the next word was clipped), is an extraction artifact — NEVER flag "
    "    it as an incomplete sentence, a cut-off / mid-word error, or a typo.\n"
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


# A "missing space / words run together" complaint. When the offending excerpt is
# predominantly UPPER-CASE it is almost always two separate on-screen labels from a
# badge / highlights bar that the DOM extraction glued together (e.g. "OFQUAL
# REGULATED QUALIFICATION" + "CENTRE NUMBER" -> "QUALIFICATIONCENTRE"), NOT an
# authored typo. Real prose spacing errors are in normal-case body copy, so this
# only suppresses the layout artefact.
_SPACING_ISSUE_RE = re.compile(
    r"missing\s+space|space\s+(?:between|is\s+missing|needed)|run[\s-]?together|"
    r"words?\s+(?:are\s+)?(?:run|joined|glued|merged|concatenat)|no\s+space|"
    r"should\s+be\s+(?:two\s+words|separated)|lack(?:s|ing)?\s+a?\s*space",
    re.I,
)


def _is_layout_concat_spacing(issue: dict) -> bool:
    desc = f"{issue.get('description', '')} {issue.get('suggestion', '')}"
    if not _SPACING_ISSUE_RE.search(desc):
        return False
    letters = [c for c in str(issue.get("excerpt", "")) if c.isalpha()]
    if not letters:
        return False
    upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    return upper_ratio >= 0.6


def _is_real_spell_issue(issue: dict) -> bool:
    """Keep only genuine, fixable spelling/grammar/punctuation problems."""
    if not isinstance(issue, dict):
        return False
    blob = f"{issue.get('description', '')} {issue.get('suggestion', '')}"
    if _NON_ISSUE_RE.search(blob):
        return False
    # Two glued-together UI labels wrongly reported as a missing-space typo.
    if _is_layout_concat_spacing(issue):
        return False
    excerpt = (issue.get("excerpt") or "").strip()
    suggestion = (issue.get("suggestion") or "").strip()
    # A "fix" identical to the original text is a non-issue.
    if excerpt and suggestion and excerpt.lower() == suggestion.lower():
        return False
    return bool(str(issue.get("description", "")).strip())


def _split_chunks(text: str, size: int) -> list[str]:
    """Split at word boundaries (preferring paragraph breaks) into <=size chunks."""
    chunks: list[str] = []
    rest = text
    while len(rest) > size:
        head = _truncate_head_at_word(rest, size)
        # Prefer to end the chunk at a paragraph break so no sentence is split
        # across two calls (a mid-sentence cut would look like a typo).
        cut = head.rfind("\n\n")
        if cut > size // 2:
            head = head[:cut]
        chunks.append(head)
        rest = rest[len(head):].lstrip()
    if rest.strip():
        chunks.append(rest)
    return chunks


def _check_chunk(chunk: str) -> list[dict]:
    prompt = f'{SCHEMA_INSTRUCTION}\n\nTEXT TO REVIEW:\n"""{chunk}"""'
    result = call_llm_json(prompt, system=SYSTEM, max_prompt_bytes=_SPELL_PROMPT_BYTES)
    issues = result.get("issues") or []
    return [i for i in issues if isinstance(i, dict)] if isinstance(issues, list) else []


def check_spelling(text: str) -> dict:
    """Full-page UK-English review.

    Long pages are reviewed in parallel paragraph-boundary chunks; issues are
    concatenated in page order. If SOME chunks fail we still return the issues
    from the ones that succeeded, with `failed_chunks`/`chunks` counts so the
    caller can surface an honest partial-failure note. Only a total failure
    (every chunk) raises.
    """
    trimmed = _truncate_head_at_word(text or "", _SPELL_READ_CHARS)
    chunks = _split_chunks(trimmed, _SPELL_CHUNK_CHARS) or [""]

    if len(chunks) == 1:
        issues = _check_chunk(chunks[0])
        return {"issues": [i for i in issues if _is_real_spell_issue(i)]}

    results: list[list[dict] | None] = [None] * len(chunks)
    errors: list[Exception] = []
    with ThreadPoolExecutor(max_workers=_SPELL_MAX_WORKERS,
                            thread_name_prefix="spell") as pool:
        futures = {pool.submit(_check_chunk, c): i for i, c in enumerate(chunks)}
        for fut, idx in futures.items():
            try:
                results[idx] = fut.result()
            except Exception as exc:  # noqa: BLE001 — partial results still count
                logger.warning("spell chunk %d/%d failed: %s: %s",
                               idx + 1, len(chunks), type(exc).__name__, str(exc)[:160])
                errors.append(exc)
    if errors and all(r is None for r in results):
        raise errors[0]
    issues = [i for r in results if r for i in r]
    out = {"issues": [i for i in issues if _is_real_spell_issue(i)]}
    if errors:
        out["failed_chunks"] = len(errors)
        out["chunks"] = len(chunks)
    return out
