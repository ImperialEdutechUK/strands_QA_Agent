"""Streamlit UI for the Strands QA Agent.

Runs the same deterministic pipeline as `web.py` (`QA_USE_PIPELINE=1` mode) —
no MCP server needed, just this one process. Suitable for Streamlit
Community Cloud, where only a single container/port is available.

Run locally:  streamlit run streamlit_app.py
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

# --- secrets -> environment --------------------------------------------------
# qa_agent's modules read os.environ at IMPORT time (module-level constants),
# so this must run before any `qa_agent` import below.
#
# Local dev: load .env directly (no need to duplicate secrets into
# .streamlit/secrets.toml). Deployed on Streamlit Cloud: no .env file ships
# (gitignored), so this is a no-op and st.secrets (from the dashboard) below
# is what populates the environment. `st.secrets` raises if no secrets.toml
# exists anywhere, which is expected for a pure-.env local setup — swallow it.
load_dotenv()
try:
    _secrets = dict(st.secrets)
except Exception:
    _secrets = {}
for _key, _value in _secrets.items():
    if isinstance(_value, str):
        os.environ.setdefault(_key, _value)

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

st.set_page_config(page_title="Strands QA Agent", page_icon="\U0001f9ea", layout="wide")


@st.cache_resource(show_spinner="Installing Chromium for Playwright (first run only, ~1-2 min)...")
def _ensure_chromium() -> None:
    subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)


_ensure_chromium()

from qa_agent.logging_config import configure_logging  # noqa: E402
from qa_agent.security import (  # noqa: E402
    ALLOWED_TEMPLATE_SUFFIXES,
    MAX_DOC_BYTES,
    validate_public_url,
)
from qa_agent.tools.report_tool import generate_pdf  # noqa: E402
from qa_agent.tools.web_tools import (  # noqa: E402
    EVIDENCE_TOKEN_PREFIX,
    attach_issue_screenshots,
    read_evidence_png,
)
from qa_agent.pipeline import run_qa_pipeline  # noqa: E402

configure_logging()

REPORTS_DIR = REPO_ROOT / "reports"
UPLOADS_DIR = REPO_ROOT / "templates" / "uploads"
SPEC_DIR = REPO_ROOT / "templates" / "spec_uploads"
for _d in (REPORTS_DIR, UPLOADS_DIR, SPEC_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _resolve_evidence_tokens(report: dict) -> None:
    for issue in report.get("issues", []) or []:
        token = issue.get("screenshot")
        if not isinstance(token, str) or not token.startswith(EVIDENCE_TOKEN_PREFIX):
            continue
        b64 = read_evidence_png(token)
        if b64:
            issue["screenshot"] = b64
        else:
            issue.pop("screenshot", None)


def _save_upload(upload, dest_dir: Path) -> str | None:
    if upload is None:
        return None
    suffix = Path(upload.name).suffix.lower()
    if suffix not in ALLOWED_TEMPLATE_SUFFIXES:
        st.error(f"Unsupported file type: {suffix}")
        st.stop()
    data = upload.getvalue()
    if len(data) > MAX_DOC_BYTES:
        st.error("File exceeds the size limit.")
        st.stop()
    dest = dest_dir / f"{uuid4().hex}{suffix}"
    dest.write_bytes(data)
    return str(dest)


st.title("\U0001f9ea Strands QA Agent")
st.caption(
    "Audits a course page for spelling/grammar and QA-template compliance, "
    "then produces a JSON + PDF report."
)

with st.form("qa_form"):
    url = st.text_input("Course page URL", placeholder="https://example.com/your-course")
    reference_url = st.text_input(
        "Reference page URL (optional)",
        placeholder="https://example.com/known-good-course",
    )
    template_text = st.text_area(
        "QA template text (optional)",
        placeholder="All headings sentence case. Page must include learning outcomes.",
    )
    template_file = st.file_uploader(
        "QA template document (optional)",
        type=["pdf", "docx", "png", "jpg", "jpeg", "webp"],
    )
    spec_file = st.file_uploader(
        "Qualification specification (optional)",
        type=["pdf", "docx", "png", "jpg", "jpeg", "webp"],
    )
    submitted = st.form_submit_button("Run QA")

if submitted:
    if not url:
        st.error("URL is required.")
        st.stop()
    try:
        url = validate_public_url(url)
    except Exception as exc:
        st.error(f"URL rejected: {exc}")
        st.stop()
    if reference_url:
        try:
            reference_url = validate_public_url(reference_url)
        except Exception as exc:
            st.error(f"Reference URL rejected: {exc}")
            st.stop()

    template_path = _save_upload(template_file, UPLOADS_DIR)
    spec_path = _save_upload(spec_file, SPEC_DIR)

    with st.spinner("Running QA — this can take a few minutes (page scrape, OCR, LLM checks)..."):
        try:
            report = run_qa_pipeline(
                url,
                template_path,
                template_text or None,
                spec_path,
                reference_url=reference_url or None,
            )
        finally:
            for p in (template_path, spec_path):
                if p:
                    Path(p).unlink(missing_ok=True)

    report.pop("reasoning", None)
    report.setdefault("url", url)
    attach_issue_screenshots(report)
    _resolve_evidence_tokens(report)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = REPORTS_DIR / f"qa-report-{stamp}.json"
    pdf_path = REPORTS_DIR / f"qa-report-{stamp}.pdf"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    generate_pdf(report, str(pdf_path))

    st.session_state["last_report"] = report
    st.session_state["last_json_path"] = str(json_path)
    st.session_state["last_pdf_path"] = str(pdf_path)

report = st.session_state.get("last_report")
if report:
    st.subheader(report.get("course_name", "QA Report"))
    st.caption(report.get("url", ""))

    issues = report.get("issues", []) or []
    sev_order = {"Critical": 0, "Minor": 1, "Info": 2}
    issues = sorted(issues, key=lambda i: sev_order.get(i.get("severity"), 3))

    counts: dict[str, int] = {"Critical": 0, "Minor": 0, "Info": 0}
    for i in issues:
        counts[i.get("severity", "Info")] = counts.get(i.get("severity", "Info"), 0) + 1
    c1, c2, c3 = st.columns(3)
    c1.metric("Critical", counts.get("Critical", 0))
    c2.metric("Minor", counts.get("Minor", 0))
    c3.metric("Info", counts.get("Info", 0))

    if report.get("tool_failures"):
        st.warning("Some steps failed: " + "; ".join(report["tool_failures"]))

    for issue in issues:
        badge = {"Critical": "\U0001f534", "Minor": "\U0001f7e1", "Info": "\U0001f535"}.get(
            issue.get("severity"), "⚪"
        )
        title = f"{badge} {issue.get('severity', 'Info')} — {issue.get('description', '')[:100]}"
        with st.expander(title):
            st.write(issue.get("description", ""))
            if issue.get("excerpt"):
                st.code(issue["excerpt"])
            if issue.get("suggestion"):
                st.info(f"Suggestion: {issue['suggestion']}")
            shot = issue.get("screenshot")
            if isinstance(shot, str) and shot:
                try:
                    st.image(base64.b64decode(shot))
                except Exception:
                    pass

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download JSON",
            data=json.dumps(report, indent=2),
            file_name=Path(st.session_state["last_json_path"]).name,
            mime="application/json",
        )
    with col2:
        pdf_bytes = Path(st.session_state["last_pdf_path"]).read_bytes()
        st.download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name=Path(st.session_state["last_pdf_path"]).name,
            mime="application/pdf",
        )
