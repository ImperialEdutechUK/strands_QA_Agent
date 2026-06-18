"""LLM-driven self-review tool.

Once the agent has run scrape -> template -> spell -> compliance -> evidence,
it calls `reason(...)` with the user's instructions and the findings so far.
This produces a short explanation of whether the QA run actually satisfied the
brief, what evidence backs each finding, and a single PASS / PARTIAL / FAIL
verdict that the UI surfaces alongside the issue list.

This is a *separate* LLM call from the agent's own reasoning loop — keeping it
in a tool makes the self-review explicit and reproducible (it shows up as a
named MCP call in the logs) and means the verdict isn't just whatever the
agent felt like saying in its final JSON.
"""

from __future__ import annotations

import json
from typing import Any

from ..llm_client import call_llm_json

SYSTEM = (
    "You are a senior QA reviewer auditing the work of a junior QA agent. "
    "You did NOT inspect the page yourself; you only see (a) the user's "
    "instructions, (b) a summary of what the page contains, and (c) the list "
    "of issues the agent reported. Judge whether the agent followed the "
    "instructions, whether each issue looks plausible from the excerpt, and "
    "whether anything in the brief was clearly missed. Be specific and brief."
)

SCHEMA = """Return JSON of exactly this shape:
{
  "verdict": "PASS|PARTIAL|FAIL",
  "summary": "<one short paragraph: did the agent meet the brief?>",
  "instructions_followed": ["<bullet — what the agent did right>"],
  "gaps": ["<bullet — what was missed or weak, or [] if none>"],
  "issue_review": [
    {"index": <0-based index into the issues list>,
     "judgement": "supported|weak|spurious",
     "comment": "<one sentence>"}
  ]
}
Rules:
  * `verdict` is FAIL only if the agent ignored the brief or the findings are
    obviously wrong; PARTIAL if the brief was partly met; PASS otherwise.
  * `issue_review` MUST contain one entry per issue, in the same order.
    If issues is empty, return [].
  * Use UK English. No prose outside the JSON object.
"""


def _summarise_page(page_summary: dict[str, Any] | None) -> str:
    if not page_summary:
        return "(page summary not provided)"
    parts: list[str] = []
    if page_summary.get("title"):
        parts.append(f"title: {page_summary['title']}")
    if page_summary.get("url"):
        parts.append(f"url: {page_summary['url']}")
    headings = page_summary.get("headings") or []
    if headings:
        sample = ", ".join(
            f"{h.get('tag','?')}={h.get('text','')[:60]!r}"
            for h in headings[:8]
        )
        parts.append(f"headings({len(headings)}): {sample}")
    template_summary = page_summary.get("template_summary")
    if template_summary:
        parts.append(f"template_summary: {template_summary}")
    return " | ".join(parts) if parts else "(empty page summary)"


def _summarise_issues(issues: list[dict[str, Any]] | None) -> str:
    if not issues:
        return "(no issues reported)"
    lines = []
    for i, iss in enumerate(issues):
        lines.append(
            f"[{i}] type={iss.get('type','?')} severity={iss.get('severity','?')}"
            f" rule={iss.get('ruleId','-')} excerpt={(iss.get('excerpt') or '')[:120]!r}"
            f" desc={(iss.get('description') or '')[:200]!r}"
        )
    return "\n".join(lines)


def review_findings(
    instructions: str,
    issues: list[dict[str, Any]] | None = None,
    page_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the self-review LLM call and return its structured verdict."""
    issues = issues or []
    payload = (
        f"{SCHEMA}\n\n"
        f"USER INSTRUCTIONS:\n\"\"\"{(instructions or '').strip() or '(none provided)'}\"\"\"\n\n"
        f"PAGE SUMMARY:\n{_summarise_page(page_summary)}\n\n"
        f"ISSUES REPORTED BY AGENT ({len(issues)}):\n{_summarise_issues(issues)}"
    )
    result = call_llm_json(payload, system=SYSTEM)

    # Normalise to the schema so downstream code can rely on the shape.
    verdict = str(result.get("verdict", "")).upper().strip() or "PARTIAL"
    if verdict not in {"PASS", "PARTIAL", "FAIL"}:
        verdict = "PARTIAL"
    return {
        "verdict": verdict,
        "summary": result.get("summary", "").strip(),
        "instructions_followed": [str(x) for x in (result.get("instructions_followed") or [])],
        "gaps": [str(x) for x in (result.get("gaps") or [])],
        "issue_review": result.get("issue_review") or [],
    }
