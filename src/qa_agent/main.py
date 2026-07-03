"""CLI entry point.

Runs the **Strands agent** end-to-end: the LLM orchestrates the QA tools
exposed by the MCP server (extract -> template -> spell -> compliance ->
evidence -> reason) and returns a single JSON report which is then rendered
to PDF. `extract` is the layered evidence stage (page text + banners + images
+ OCR + claims); its banner/image claims are compared against template rules
in the compliance step.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import click
from dotenv import load_dotenv

from .logging_config import configure_logging
from .security import redact
from .tools.report_tool import generate_pdf
from .tools.web_tools import (
    EVIDENCE_TOKEN_PREFIX,
    attach_issue_screenshots,
    read_evidence_png,
)

load_dotenv()
configure_logging()
logger = logging.getLogger(__name__)


@click.command()
@click.option("--url", "-u", required=True, help="Course page URL to QA.")
@click.option("--template", "-t", "template_path", default=None,
              help="Path to a QA template file (image, PDF, or Word document).")
@click.option("--template-text", default=None, help="Inline QA template text (alternative to --template).")
@click.option("--spec", "spec_path", default=None,
              help="Path to the official Qualification Specification sheet (PDF/DOCX/image).")
@click.option("--reference", "reference_url", default=None,
              help="URL of a known-good published course page to diff structure against.")
@click.option("--out", "-o", default=None, help="Output PDF path (defaults to reports/qa-report-<ts>.pdf).")
@click.option("--agent", "use_agent", is_flag=True, default=False,
              help="Route through the Strands agent + MCP server instead of the "
                   "deterministic pipeline (needs the MCP server running).")
def main(url: str, template_path: str | None, template_text: str | None,
         spec_path: str | None, reference_url: str | None, out: str | None,
         use_agent: bool) -> None:
    if use_agent:
        report = _run_with_agent(
            url=url,
            template_path=template_path,
            template_text=template_text,
        )
    else:
        report = _run_with_pipeline(
            url=url,
            template_path=template_path,
            template_text=template_text,
            spec_path=spec_path,
            reference_url=reference_url,
        )
    # The agent self-review is kept for our logs only — strip it from the report
    # so it never reaches the PDF / JSON artefact. (The `reason` tool still runs;
    # we just don't surface its verdict to the reviewer.)
    reasoning = report.pop("reasoning", None)
    if reasoning:
        logger.info("agent self-review (logs only): %s", json.dumps(reasoning, ensure_ascii=False))

    report.setdefault("url", url)
    attach_issue_screenshots(report)
    _resolve_evidence_tokens(report)

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = reports_dir / f"qa-report-{stamp}.json"
    pdf_path = Path(out) if out else reports_dir / f"qa-report-{stamp}.pdf"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    generate_pdf(report, str(pdf_path))

    click.echo("\nDone.")
    click.echo(f"  JSON: {json_path}")
    click.echo(f"  PDF:  {pdf_path}")
    click.echo(f"  Issues: {len(report.get('issues', []))}")


def _resolve_evidence_tokens(report: dict) -> None:
    """Replace each issue's `evidence://...` screenshot token with base64 PNG.

    The agent emits opaque tokens to keep the LLM's context small; the PDF
    generator and downstream consumers expect inline base64. Tokens that can't
    be resolved are dropped from the issue so the PDF doesn't try to embed them.
    """
    for issue in report.get("issues", []) or []:
        token = issue.get("screenshot")
        if not isinstance(token, str) or not token.startswith(EVIDENCE_TOKEN_PREFIX):
            continue
        b64 = read_evidence_png(token)
        if b64:
            issue["screenshot"] = b64
        else:
            issue.pop("screenshot", None)


def _run_with_pipeline(url: str, template_path: str | None, template_text: str | None,
                       spec_path: str | None = None,
                       reference_url: str | None = None) -> dict:
    """Deterministic run: tools called directly, report assembled in code."""
    from .pipeline import run_qa_pipeline

    click.echo("Running deterministic QA pipeline...")
    try:
        return run_qa_pipeline(url, template_path, template_text, spec_path,
                               reference_url=reference_url)
    except Exception as exc:  # noqa: BLE001 — still produce a JSON+PDF artefact
        click.echo(f"Pipeline run failed mid-flight: {redact(str(exc))}", err=True)
        return {
            "course_name": "QA run incomplete",
            "url": url,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "template_summary": None,
            "issues": [],
            "tool_failures": [f"pipeline: {type(exc).__name__}: {redact(str(exc))}"],
        }


def _run_with_agent(url: str, template_path: str | None, template_text: str | None) -> dict:
    from .agent import build_agent, build_user_prompt, invoke_with_retry

    with build_agent() as (agent, _client):
        prompt = build_user_prompt(url, template_path, template_text)
        click.echo("Running Strands agent...")
        try:
            result = invoke_with_retry(agent, prompt)
        except Exception as exc:
            # Provider-side hiccups (DeepSeek/OpenRouter occasional 5xx, dropped
            # streams, etc) shouldn't lose the run — emit a stub report with
            # the error so the artifact pipeline still produces a JSON+PDF.
            click.echo(f"Agent run failed mid-flight: {redact(str(exc))}", err=True)
            return {
                "course_name": "QA run incomplete",
                "url": url,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "template_summary": None,
                "issues": [],
                "tool_failures": [f"agent: {type(exc).__name__}: {redact(str(exc))}"],
            }

    text = str(result)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        click.echo(
            "Agent did not return valid JSON. Returning a stub report so the run "
            "still produces an artifact you can inspect.",
            err=True,
        )
        return {
            "course_name": "QA run incomplete",
            "url": url,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "template_summary": None,
            "issues": [],
            "raw_agent_output": text,
        }


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        click.echo(f"\nQA run failed: {redact(str(exc))}", err=True)
        sys.exit(1)
