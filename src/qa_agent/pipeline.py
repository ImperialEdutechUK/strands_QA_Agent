"""Deterministic QA pipeline — the same tools the Strands agent drives, run
directly in code with NO orchestrator LLM.

WHY: the agent's system prompt already fixes the tool order (extract ->
template -> spell -> spec_lookup -> compliance -> reason) and forbids any
deviation, so the orchestration model added nothing but cost and failure
modes:

  * every orchestration turn re-billed the system prompt plus the whole
    message history (7+ turns per run);
  * the model could mis-copy an `extraction_id`/`template_id`, skip a step,
    or retry a tool it was told not to;
  * worst of all, it RE-TYPED the entire final report as JSON — the single
    biggest hallucination point, where issues got reworded, dropped or
    invented after the tools had already produced the correct list.

Here the only LLM calls left are the analytical ones inside the tools
(template parse, spell check, spec extraction, compliance audit) — the
irreducible cost of the run. The report is assembled in code from the tools'
verbatim output, so nothing can be paraphrased or invented between the tool
result and the report.

The Strands agent (`agent.py` + `mcp_server.py`) remains available for
interactive / MCP use; set QA_USE_PIPELINE=0 to route the web UI / CLI back
through it.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone

from .extraction import extract_page_summary, get_cached_extraction
from .security import redact
from .tools.compliance_tool import _dedupe_issues, check_compliance, select_visual_rules
from .tools.reasoning_tool import review_findings
from .tools.reference_tool import compare_with_reference, extract_reference
from .tools.spec_tool import lookup_specification
from .tools.spell_tool import check_spelling
from .tools.template_tool import (
    analyse_template,
    analyse_template_text,
    get_cached_template_rules,
)
from .tools.vision_tool import check_visual_rules, vision_available

logger = logging.getLogger(__name__)
trace = logging.getLogger("qa_agent.trace")

# The self-review (`reason` tool) is logs-only — web.py/main.py strip it from
# the artefact before the reviewer sees it — so by default we skip its LLM call
# entirely. Set QA_SELF_REVIEW=1 to run it and log its verdict.
_SELF_REVIEW = os.environ.get("QA_SELF_REVIEW", "0").strip() == "1"
# The template rules are the priority check; the spell pass is secondary and
# costs a full-page LLM call. On by default, QA_SKIP_SPELL=1 to save it.
_SKIP_SPELL = os.environ.get("QA_SKIP_SPELL", "0").strip() == "1"
# Run independent stages concurrently (extract ∥ template ∥ reference-extract,
# then spell ∥ spec_lookup ∥ vision). Every stage receives EXACTLY the same
# inputs as the serial order did, so the findings are identical — only the
# wall-clock changes. Set QA_PARALLEL=0 to serialize (e.g. when debugging, or
# if concurrent LLM calls trip the provider's rate limit too often).
_PARALLEL = os.environ.get("QA_PARALLEL", "1").strip() != "0"


def _timed(step: str, fn, /, *args, **kwargs):
    """Run `fn`, tracing how long the stage took (visibility into slow runs)."""
    started = time.perf_counter()
    try:
        return fn(*args, **kwargs)
    finally:
        trace.info("[PIPELINE] %s took %.1fs", step, time.perf_counter() - started)


def _template_step(template_path: str | None, template_text: str | None) -> dict:
    if template_path:
        return analyse_template(template_path)
    # Empty text -> baseline SLC checklist only (never raises for that).
    return analyse_template_text(template_text or "")


def _fail(failures: list[str], step: str, exc: Exception) -> None:
    msg = f"{step}: {type(exc).__name__}: {redact(str(exc))[:200]}"
    failures.append(msg)
    logger.warning("pipeline step failed — %s", msg)


def _check_awarding_body_bold(identity: dict, bold_texts: list) -> dict | None:
    """Deterministic S02-1: is the awarding body name shown in bold anywhere?

    Bold is a DOM fact — `bold_texts` holds every visible bold run verbatim, so
    this needs no model at all (the vision pass was false-flagging genuinely
    bold text). Returns an issue dict, or None when the rule passes or cannot
    be judged (no awarding body detected / no bold runs captured at all —
    an empty capture is indistinguishable from an extraction gap, so we skip
    rather than guess).
    """
    body = (identity.get("awarding_body") or "").strip()
    if not body or not bold_texts:
        return None
    hay = " ".join(str(t) for t in bold_texts).lower()
    if body.lower() in hay:
        return None
    return {
        "ruleId": "S02-1",
        "type": "Template",
        "severity": "Minor",
        "description": (
            f"The awarding body name '{body}' does not appear in bold anywhere "
            f"on the page (checked against the rendered page's bold text)."
        ),
        "suggestion": f"Bold the awarding body name '{body}' in the course overview.",
        "excerpt": body,
    }


def run_qa_pipeline(url: str, template_path: str | None = None,
                    template_text: str | None = None,
                    spec_path: str | None = None,
                    reference_url: str | None = None) -> dict:
    """Run the full QA flow deterministically and return the report dict.

    The report has the same shape the agent was instructed to emit
    (course_name, url, template_summary, specification_source, issues,
    tool_failures [+ reasoning when QA_SELF_REVIEW=1]), so every downstream
    consumer (web UI, PDF, JSON artefact) works unchanged.

    ``reference_url``: a known-good published course page. When given, the
    target's structure (sections, CTA buttons, key banners) is deterministically
    diffed against it — the code-level equivalent of a human reviewer keeping an
    approved page open in the next tab.
    """
    failures: list[str] = []
    report: dict = {
        "course_name": "QA run incomplete",
        "url": url,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "template_summary": None,
        "specification_source": None,
        "issues": [],
        "tool_failures": failures,
    }

    run_started = time.perf_counter()
    # Stages fan out onto worker threads; every stage still receives exactly
    # the inputs the old serial order gave it, so findings are unchanged.
    # QA_PARALLEL=0 -> one worker, which executes the same submissions serially.
    with ThreadPoolExecutor(max_workers=4 if _PARALLEL else 1,
                            thread_name_prefix="qa-pipe") as pool:
        # ---- fan-out A: the three stages that need nothing but the request —
        #      main extraction ∥ template parse ∥ reference-page extraction.
        trace.info("[PIPELINE] extract %s ∥ template (%s)%s", url,
                   "file" if template_path else "text" if template_text else "baseline",
                   f" ∥ reference extract {reference_url}" if reference_url else "")
        extract_f = pool.submit(_timed, "extract", extract_page_summary, url)
        template_f = pool.submit(_timed, "template", _template_step,
                                 template_path, template_text)
        ref_f: Future | None = None
        if reference_url:
            ref_f = pool.submit(_timed, "reference extract",
                                extract_reference, reference_url)

        # ---- template — parse the QA checklist into rules (baseline checklist
        #      always applies, even with no uploaded template).
        template_result: dict | None = None
        try:
            template_result = template_f.result()
        except Exception as exc:  # noqa: BLE001
            _fail(failures, "template", exc)
        rules: list | None = None
        if template_result:
            report["template_summary"] = template_result.get("summary") or None
            rules = get_cached_template_rules(template_result.get("template_id"))

        # ---- extract — nothing downstream can run without it.
        try:
            summary = extract_f.result()
        except Exception as exc:  # noqa: BLE001
            _fail(failures, "extract", exc)
            return report
        cached = get_cached_extraction(summary.get("extraction_id")) or {}
        identity = summary.get("course_identity") or {}
        gc = summary.get("general_content") or {}
        report["course_name"] = (
            identity.get("course_name") or gc.get("page_title") or "QA run incomplete"
        )
        if summary.get("extraction_warnings"):
            report["extraction_warnings"] = summary["extraction_warnings"]
        page_text = cached.get("page_text", "")

        # ---- fan-out B: spell ∥ spec_lookup ∥ vision — they need only the
        #      extraction + rules, never each other's output.
        spell_f: Future | None = None
        if _SKIP_SPELL:
            trace.info("[PIPELINE] spell skipped (QA_SKIP_SPELL=1)")
        else:
            spell_f = pool.submit(_timed, "spell", check_spelling, page_text)

        spec_f: Future | None = None
        if template_result and template_result.get("needs_spec"):
            trace.info("[PIPELINE] spec_lookup %r qn=%r",
                       identity.get("course_name", "")[:60],
                       identity.get("qualification_number", ""))
            spec_f = pool.submit(
                _timed, "spec_lookup", lookup_specification,
                identity.get("course_name") or report["course_name"],
                qualification_number=identity.get("qualification_number", ""),
                level=identity.get("level", ""),
                awarding_body=identity.get("awarding_body", ""),
                variant=identity.get("variant", ""),
                document_path=spec_path,
            )
        else:
            trace.info("[PIPELINE] spec_lookup skipped (no needs_spec rules)")

        visual_rules = select_visual_rules(rules or [])
        vision_f: Future | None = None
        if visual_rules and vision_available():
            trace.info("[PIPELINE] vision (%d visual rules)", len(visual_rules))
            vision_f = pool.submit(_timed, "vision", check_visual_rules,
                                   url, visual_rules)
        else:
            trace.info("[PIPELINE] vision skipped (%s)",
                       "no visual rules" if not visual_rules else "QA_VISION_MODEL not set")

        # ---- spec_lookup result — compliance needs it, so join it first.
        specification: dict | None = None
        if spec_f is not None:
            try:
                specification = spec_f.result()
            except Exception as exc:  # noqa: BLE001 — spec tool degrades internally,
                # but guard anyway: needs_spec rules are silently skipped without it.
                _fail(failures, "spec_lookup", exc)
            if specification and specification.get("source_urls"):
                report["specification_source"] = specification["source_urls"]

        # ---- compliance — the core audit; runs on this thread while
        #      spell / vision / reference still work in the background.
        comp_issues: list[dict] = []
        if rules:
            trace.info("[PIPELINE] compliance (%d rules)", len(rules))
            try:
                comp = _timed(
                    "compliance", check_compliance,
                    page_text=page_text,
                    headings=cached.get("headings", []),
                    rules=rules,
                    price_candidates=cached.get("price_candidates", []),
                    banner_evidence=cached.get("banner_evidence", []),
                    image_evidence=cached.get("image_evidence", []),
                    specification=specification,
                    course_identity=cached.get("course_identity"),
                    bold_texts=cached.get("bold_texts", []),
                )
                comp_issues = [i for i in (comp.get("issues") or []) if isinstance(i, dict)]
            except Exception as exc:  # noqa: BLE001 — an empty compliance result must
                # surface as an honest failure, never as a falsely-clean report.
                _fail(failures, "compliance", exc)
        else:
            _fail(failures, "compliance", RuntimeError("no rules available (template step failed)"))

        # ---- join spell.
        spell_issues: list[dict] = []
        if spell_f is not None:
            try:
                spell_res = spell_f.result() or {}
                if spell_res.get("failed_chunks"):
                    failures.append(
                        f"spell: {spell_res['failed_chunks']} of "
                        f"{spell_res['chunks']} page chunks failed the LLM call — "
                        "the issues below cover the rest of the page."
                    )
                spell_issues = [i for i in (spell_res.get("issues") or [])
                                if isinstance(i, dict)]
            except Exception as exc:  # noqa: BLE001
                _fail(failures, "spell", exc)

        # ---- join vision.
        vision_issues: list[dict] = []
        if vision_f is not None:
            try:
                vis = vision_f.result()
                if vis.get("skipped"):
                    trace.info("[PIPELINE] vision skipped: %s", vis["skipped"])
                vision_issues = [i for i in (vis.get("issues") or []) if isinstance(i, dict)]
            except Exception as exc:  # noqa: BLE001 — additive evidence, never fatal
                _fail(failures, "vision", exc)

        # ---- reference cross-check — deterministic structural diff against a
        #      known-good published course page (sections / CTAs / key banners).
        #      The heavy extraction already ran in parallel; the diff itself is
        #      cheap, and its evidence capture opens the reference page once.
        ref_issues: list[dict] = []
        if ref_f is not None:
            trace.info("[PIPELINE] reference diff vs %s", reference_url)
            ref_extraction = None
            try:
                ref_extraction = ref_f.result()
            except Exception as exc:  # noqa: BLE001
                _fail(failures, "reference", exc)
            if ref_extraction is not None:
                try:
                    ref = _timed("reference diff", compare_with_reference,
                                 reference_url, cached, ref_extraction)
                    if ref.get("skipped"):
                        _fail(failures, "reference", RuntimeError(ref["skipped"]))
                    ref_issues = [i for i in (ref.get("issues") or []) if isinstance(i, dict)]
                except Exception as exc:  # noqa: BLE001
                    _fail(failures, "reference", exc)

    # ---- deterministic checks — DOM facts need no model.
    bold_issue = _check_awarding_body_bold(identity, cached.get("bold_texts", []))

    # Assemble in the same order the serial pipeline used (spell, compliance,
    # bold check, vision, reference) so reports stay byte-comparable.
    issues: list[dict] = list(spell_issues)
    issues.extend(comp_issues)
    if bold_issue:
        issues.append(bold_issue)
    issues.extend(vision_issues)
    issues.extend(ref_issues)
    trace.info("[PIPELINE] all stages joined in %.1fs", time.perf_counter() - run_started)

    # Cross-source dedupe: the spell pass can flag the same spot twice with
    # different wording, and spell/compliance/vision can overlap. Same-subject
    # findings collapse to one regardless of which pass produced them.
    issues = _dedupe_issues(issues)
    report["issues"] = issues

    # ---- reason — optional self-review (logs-only downstream).
    if _SELF_REVIEW:
        trace.info("[PIPELINE] reason (%d issues)", len(issues))
        try:
            report["reasoning"] = review_findings(
                instructions=f"QA the course page at {url} against the QA checklist.",
                issues=[{k: v for k, v in i.items()
                         if k not in ("screenshot", "screenshot_caption")}
                        for i in issues],
                page_summary={
                    "title": gc.get("page_title", ""),
                    "url": url,
                    "template_summary": report.get("template_summary"),
                },
            )
        except Exception as exc:  # noqa: BLE001
            _fail(failures, "reason", exc)
    else:
        trace.info("[PIPELINE] reason skipped (QA_SELF_REVIEW not set)")

    trace.info("[PIPELINE] done — %d issues, %d tool failures", len(issues), len(failures))
    return report
