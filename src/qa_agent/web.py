"""Starlette-based web UI + JSON API for the QA agent.

Run with:  python -m qa_agent.web

Endpoints:
  GET  /                      → single-page UI (static/index.html)
  GET  /static/...            → JS / CSS assets
  GET  /api/health            → liveness + MCP reachability check
  POST /api/qa                → run the agent and return the report + artefact ids
  GET  /api/reports/{name}    → fetch a saved JSON / PDF report by filename
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

from dotenv import load_dotenv
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from .logging_config import configure_logging
from .security import (
    ALLOWED_DOC_SUFFIXES,
    ALLOWED_IMAGE_SUFFIXES,
    ALLOWED_TEMPLATE_SUFFIXES,
    MAX_DOC_BYTES,
    MAX_IMAGE_BYTES,
    redact,
    validate_public_url,
)
from .tools.report_tool import generate_pdf
from .tools.web_tools import (
    EVIDENCE_TOKEN_PREFIX,
    attach_issue_screenshots,
    read_evidence_png,
)

load_dotenv()
configure_logging()
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
REPORTS_DIR = REPO_ROOT / "reports"
TEMPLATE_UPLOADS_DIR = REPO_ROOT / "templates" / "uploads"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATE_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
# Uploaded specification sheets live under the same allowed-template root so the
# spec tool's `safe_resolve_template` accepts them.
SPEC_UPLOADS_DIR = REPO_ROOT / "templates" / "spec_uploads"
SPEC_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

MCP_URL = os.environ.get("MCP_URL", "http://127.0.0.1:3001/mcp")

# Default: run the deterministic pipeline (tools called directly in code — no
# orchestrator LLM, no MCP round-trip). Cheaper (only the analytical LLM calls
# remain) and hallucination-free at the report-assembly stage. Set
# QA_USE_PIPELINE=0 to route through the Strands agent + MCP server instead.
USE_PIPELINE = os.environ.get("QA_USE_PIPELINE", "1").strip() != "0"

# Reasonable upper bounds on text fields so a malicious form post can't OOM the
# server. The agent itself caps things further downstream.
MAX_URL_LEN = 2048
MAX_TEMPLATE_TEXT_LEN = 16 * 1024

# Optional wall-clock budget for one QA run. DISABLED by default (0): a slow run
# that finishes in, say, 16 minutes was being cut off at 900s and replaced with a
# placeholder "timed out" report, throwing away the real (already-completed)
# result. We now let the run finish and return the genuine report. Set
# QA_RUN_TIMEOUT_SECONDS to a positive number to re-enable a hard cap.
AGENT_RUN_TIMEOUT = float(os.environ.get("QA_RUN_TIMEOUT_SECONDS", "0"))


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _resolve_evidence_tokens(report: dict) -> None:
    """In-place: swap each issue's `evidence://...` token for inline base64 PNG."""
    for issue in report.get("issues", []) or []:
        token = issue.get("screenshot")
        if not isinstance(token, str) or not token.startswith(EVIDENCE_TOKEN_PREFIX):
            continue
        b64 = read_evidence_png(token)
        if b64:
            issue["screenshot"] = b64
        else:
            issue.pop("screenshot", None)


async def _save_form_upload(upload, dest_dir: Path, label: str) -> Path | None:
    """Validate + persist a multipart file upload; return its path (or None)."""
    if upload is None or not getattr(upload, "filename", ""):
        return None
    suffix = Path(upload.filename).suffix.lower()
    if suffix not in ALLOWED_TEMPLATE_SUFFIXES:
        raise HTTPException(
            400,
            f"Unsupported {label} file type: {suffix}. "
            f"Allowed: {sorted(ALLOWED_TEMPLATE_SUFFIXES)}",
        )
    data = await upload.read()
    max_size = MAX_DOC_BYTES if suffix in ALLOWED_DOC_SUFFIXES else MAX_IMAGE_BYTES
    if len(data) > max_size:
        raise HTTPException(413, f"{label.capitalize()} file exceeds size limit.")
    dest = dest_dir / f"{uuid4().hex}{suffix}"
    dest.write_bytes(data)
    return dest


def _run_agent_sync(url: str, template_path: str | None, template_text: str | None,
                    spec_path: str | None = None,
                    reference_url: str | None = None) -> dict:
    """Blocking QA invocation. Called via asyncio.to_thread from the route handler.

    Default path is the deterministic pipeline (`pipeline.py`): the tools run
    directly in this process and the report is assembled in code, so the run
    needs no orchestrator LLM and no MCP server. QA_USE_PIPELINE=0 restores the
    Strands-agent-over-MCP path.
    """
    if USE_PIPELINE:
        from .pipeline import run_qa_pipeline

        try:
            report = run_qa_pipeline(url, template_path, template_text, spec_path,
                                     reference_url=reference_url)
        except Exception as exc:  # noqa: BLE001 — never lose the run
            return {
                "course_name": "QA run incomplete",
                "url": url,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "template_summary": None,
                "issues": [],
                "tool_failures": [f"pipeline: {type(exc).__name__}: {redact(str(exc))}"],
            }
        reasoning = report.pop("reasoning", None)
        if reasoning:
            logger.info("pipeline self-review (logs only): %s",
                        json.dumps(reasoning, ensure_ascii=False))
        report.setdefault("url", url)
        attach_issue_screenshots(report)
        return report

    # Imported lazily so the web process boots even if Strands has init issues.
    from .agent import build_agent, build_user_prompt, invoke_with_retry

    with build_agent() as (agent, _client):
        prompt = build_user_prompt(url, template_path, template_text, spec_path)
        try:
            result = invoke_with_retry(agent, prompt)
        except Exception as exc:
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
        report = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                report = json.loads(match.group(0))
            except json.JSONDecodeError:
                report = None
        else:
            report = None

    if report is None:
        return {
            "course_name": "QA run incomplete",
            "url": url,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "template_summary": None,
            "issues": [],
            "raw_agent_output": text,
        }

    # The agent self-review is kept for our logs only — strip it from the report
    # so it never reaches the UI / PDF / JSON artefact. (The `reason` tool still
    # runs; we just don't surface its verdict to the reviewer.)
    reasoning = report.pop("reasoning", None)
    if reasoning:
        logger.info("agent self-review (logs only): %s", json.dumps(reasoning, ensure_ascii=False))

    # Make sure the page URL is set so screenshot capture (below) has a target,
    # then attach a cropped screenshot to every issue from its excerpt. Done in
    # this worker thread because Playwright cannot run in the async event loop.
    report.setdefault("url", url)
    attach_issue_screenshots(report)
    return report


async def index(_request: Request) -> Response:
    return FileResponse(STATIC_DIR / "index.html")


async def health(_request: Request) -> JSONResponse:
    """Quick liveness probe + can-we-talk-to-MCP check.

    We check whether the MCP port is *listening* with a plain TCP connect rather
    than issuing an HTTP GET. A bare GET to the streamable-HTTP `/mcp` endpoint
    makes the server spin up a new transport/session and then reply 406 Not
    Acceptable (it expects an SSE `Accept` header) — so the old probe leaked a
    session and logged a 406 on every 15-second poll. A TCP connect proves the
    server is up without touching MCP's request machinery, keeping the logs clean.
    """
    parsed = urlparse(MCP_URL)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    mcp_ok = False
    mcp_status: int | str = "unreachable"
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=3.0
        )
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:  # noqa: BLE001 - best-effort socket teardown
            pass
        mcp_ok = True
        mcp_status = "listening"
    except Exception as exc:
        mcp_status = redact(str(exc))

    return JSONResponse({
        "ok": True,
        "mcp_url": MCP_URL,
        "mcp_reachable": mcp_ok,
        "mcp_status": mcp_status,
    })


async def run_qa(request: Request) -> JSONResponse:
    content_type = (request.headers.get("content-type") or "").lower()

    url: str | None = None
    template_text: str | None = None
    template_path: str | None = None
    spec_path: str | None = None
    reference_url: str | None = None
    saved_uploads: list[Path] = []

    if "multipart/form-data" in content_type:
        form = await request.form()
        url = (form.get("url") or "").strip() or None
        template_text = (form.get("template_text") or "").strip() or None
        reference_url = (form.get("reference_url") or "").strip() or None
        # Prefer the new `template_document` field; accept the legacy
        # `template_image` name too so older clients keep working.
        tpl = await _save_form_upload(
            form.get("template_document") or form.get("template_image"),
            TEMPLATE_UPLOADS_DIR, "template",
        )
        if tpl is not None:
            saved_uploads.append(tpl)
            template_path = str(tpl)
        # Optional official qualification specification sheet. When supplied the
        # spec tool reads it directly (matched to this qualification) instead of
        # web-searching for it.
        spec = await _save_form_upload(
            form.get("spec_document"), SPEC_UPLOADS_DIR, "specification",
        )
        if spec is not None:
            saved_uploads.append(spec)
            spec_path = str(spec)
    else:
        try:
            payload = await request.json()
        except Exception as exc:
            raise HTTPException(400, f"Invalid JSON body: {exc}")
        url = (payload.get("url") or "").strip() or None
        template_text = (payload.get("template_text") or "").strip() or None
        reference_url = (payload.get("reference_url") or "").strip() or None
        # JSON path doesn't accept uploads — clients should use multipart for that.

    if not url:
        raise HTTPException(400, "Field `url` is required.")
    if len(url) > MAX_URL_LEN:
        raise HTTPException(400, "URL is too long.")
    try:
        url = validate_public_url(url)
    except Exception as exc:
        raise HTTPException(400, f"URL rejected: {exc}")
    if template_text and len(template_text) > MAX_TEMPLATE_TEXT_LEN:
        raise HTTPException(400, "template_text exceeds size limit.")
    if reference_url:
        if len(reference_url) > MAX_URL_LEN:
            raise HTTPException(400, "Reference URL is too long.")
        try:
            reference_url = validate_public_url(reference_url)
        except Exception as exc:
            raise HTTPException(400, f"Reference URL rejected: {exc}")

    try:
        agent_run = asyncio.to_thread(
            _run_agent_sync, url, template_path, template_text, spec_path,
            reference_url,
        )
        if AGENT_RUN_TIMEOUT > 0:
            try:
                report = await asyncio.wait_for(agent_run, timeout=AGENT_RUN_TIMEOUT)
            except asyncio.TimeoutError:
                # The worker thread can't be force-killed; it finishes in the
                # background. We return an honest report so the browser isn't left
                # hanging for tens of minutes.
                logger.error("agent run exceeded %.0fs budget for %s", AGENT_RUN_TIMEOUT, url)
                report = {
                    "course_name": "QA run timed out",
                    "url": url,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "template_summary": None,
                    "issues": [],
                    "tool_failures": [
                        f"run exceeded the {int(AGENT_RUN_TIMEOUT)}s time budget — the LLM "
                        "provider is most likely rate-limiting or slow. Retry, widen "
                        "OPENROUTER_ONLY_PROVIDERS, enable fallbacks, or use a "
                        "less-congested model."
                    ],
                }
        else:
            # No wall-clock cap (default): let the run finish and return the real
            # report rather than discarding a completed run at an arbitrary limit.
            report = await agent_run
    finally:
        for saved in saved_uploads:
            try:
                saved.unlink(missing_ok=True)
            except Exception as exc:
                logger.warning("failed to remove temp upload %s: %s", saved, exc)

    _resolve_evidence_tokens(report)

    stamp = _ts()
    json_name = f"qa-report-{stamp}.json"
    pdf_name = f"qa-report-{stamp}.pdf"
    json_path = REPORTS_DIR / json_name
    pdf_path = REPORTS_DIR / pdf_name

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    try:
        generate_pdf(report, str(pdf_path))
    except Exception as exc:
        logger.exception("PDF generation failed")
        return JSONResponse(
            {
                "report": report,
                "json_url": f"/api/reports/{json_name}",
                "pdf_url": None,
                "pdf_error": redact(str(exc)),
            },
        )

    return JSONResponse({
        "report": report,
        "json_url": f"/api/reports/{json_name}",
        "pdf_url": f"/api/reports/{pdf_name}",
    })


_SAFE_REPORT_NAME = re.compile(r"^qa-report-\d{8}T\d{6}Z\.(json|pdf)$")


async def get_report(request: Request) -> Response:
    name = request.path_params["name"]
    if not _SAFE_REPORT_NAME.fullmatch(name):
        raise HTTPException(404, "Not found")
    path = REPORTS_DIR / name
    if not path.exists():
        raise HTTPException(404, "Not found")
    media = "application/pdf" if name.endswith(".pdf") else "application/json"
    return FileResponse(path, media_type=media, filename=name)


routes = [
    Route("/", index, methods=["GET"]),
    Route("/api/health", health, methods=["GET"]),
    Route("/api/qa", run_qa, methods=["POST"]),
    Route("/api/reports/{name}", get_report, methods=["GET"]),
    Mount("/static", app=StaticFiles(directory=str(STATIC_DIR)), name="static"),
]

# Permissive CORS so the same UI can be hosted elsewhere if desired. Tighten in
# prod by setting WEB_ALLOWED_ORIGINS to a comma-separated list of origins.
_origins_env = os.environ.get("WEB_ALLOWED_ORIGINS", "*").strip()
allow_origins = [o.strip() for o in _origins_env.split(",")] if _origins_env else ["*"]

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
]

app = Starlette(routes=routes, middleware=middleware)


def main() -> None:
    import uvicorn

    host = os.environ.get("WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("WEB_PORT", "8000"))
    logger.info("QA web UI on http://%s:%s (MCP at %s)", host, port, MCP_URL)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
