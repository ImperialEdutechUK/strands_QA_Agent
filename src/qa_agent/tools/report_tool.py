from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

# A4 minus margins gives ~17 cm of usable width; cap a touch under that.
MAX_IMAGE_WIDTH = 16 * cm
MAX_IMAGE_HEIGHT = 18 * cm
PX_TO_PT = 72.0 / 96.0  # 96 dpi screen px → 72 dpi PDF pt

SEVERITY_COLOURS = {"Critical": "#c0392b", "Minor": "#d68910", "Info": "#2874a6"}
VERDICT_COLOURS = {"PASS": "#1e8449", "PARTIAL": "#d68910", "FAIL": "#c0392b"}


def _html_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _count_severities(issues: list[dict]) -> dict[str, int]:
    counts = {"Critical": 0, "Minor": 0, "Info": 0}
    for issue in issues:
        sev = issue.get("severity", "Info")
        counts[sev] = counts.get(sev, 0) + 1
    return counts


def _format_spec_sources(spec_source) -> str:
    """Render the qualification-specification source(s) as clickable links.

    Accepts a single URL string or a list of URLs (whatever the agent put in
    `specification_source`). Returns an empty string when nothing usable is
    present so the header line is omitted entirely.
    """
    if not spec_source:
        return ""
    if isinstance(spec_source, str):
        urls = [spec_source]
    elif isinstance(spec_source, (list, tuple)):
        urls = [str(u) for u in spec_source]
    else:
        urls = [str(spec_source)]
    links = []
    for u in urls:
        u = u.strip()
        if not u or u.lower() in {"null", "none"}:
            continue
        safe = _html_escape(u)
        if u.lower().startswith(("http://", "https://")):
            links.append(f'<link href="{safe}"><font color="#2874a6">{safe}</font></link>')
        else:
            links.append(safe)
    return ", ".join(links)


def _instruction_text(issue: dict) -> str:
    """The actionable correction line, in the team's house style.

    QA docs read as direct edit instructions ("Course pricing has to be added.",
    "Update this section.", "Remove this section."). We lead with the suggestion
    when it is phrased as an instruction, otherwise fall back to the description.
    """
    suggestion = (issue.get("suggestion") or "").strip()
    description = (issue.get("description") or "").strip()
    # House style is terse and instruction-first ("No logo added", "Update to
    # 24 months"). Lead with the suggestion (the action); the screenshot + the
    # "Current:" excerpt line below carry the context, so we don't repeat the
    # long description here. Fall back to the description only when there is no
    # suggestion to act on.
    return suggestion or description or "Review this item."


def generate_pdf(report: dict, out_path: str) -> str:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(out_path, pagesize=A4, title=report.get("course_name", "QA Report"))

    # House style: lead with Course Name / Course Link, then a numbered list of
    # actionable correction instructions — matching the team's QA update docs.
    flow: list = [
        Paragraph(f"<b>Course Name:</b> {_html_escape(report.get('course_name', ''))}", styles["Normal"]),
        Paragraph(f"<b>Course Link:</b> {_html_escape(report.get('url', ''))}", styles["Normal"]),
    ]
    if report.get("template_summary"):
        flow.append(Paragraph(f"<b>Qualification / Template:</b> {_html_escape(report['template_summary'])}", styles["Normal"]))
    spec_src = _format_spec_sources(report.get("specification_source"))
    if spec_src:
        flow.append(Paragraph(f"<b>Qualification Specification checked:</b> {spec_src}", styles["Normal"]))
    flow.append(Paragraph(f"<i>Generated: {report.get('generated_at', '')}</i>", styles["Italic"]))

    issues = report.get("issues", []) or []
    counts = _count_severities(issues)
    flow.append(Spacer(1, 0.3 * cm))
    flow.append(Paragraph(
        f"<b>QA updates required:</b> {len(issues)} &nbsp; "
        f"<font color='{SEVERITY_COLOURS['Critical']}'>Critical: {counts['Critical']}</font> &nbsp; "
        f"<font color='{SEVERITY_COLOURS['Minor']}'>Minor: {counts['Minor']}</font> &nbsp; "
        f"<font color='{SEVERITY_COLOURS['Info']}'>Info: {counts['Info']}</font>",
        styles["Normal"],
    ))
    flow.append(Spacer(1, 0.3 * cm))

    if not issues:
        flow.append(Paragraph("No updates required — the page passed every checklist item.", styles["Normal"]))

    for i, issue in enumerate(issues, start=1):
        colour = SEVERITY_COLOURS.get(issue.get("severity", "Info"), "#333333")
        # The numbered instruction line, with a small step/rule + severity tag.
        rule_tag = f" <font color='#888888'>[{_html_escape(issue['ruleId'])}]</font>" if issue.get("ruleId") else ""
        sev = issue.get("severity", "Info")
        flow.append(Spacer(1, 0.25 * cm))
        flow.append(Paragraph(
            f"<b>({i})</b> {_html_escape(_instruction_text(issue))}"
            f"{rule_tag} <font color='{colour}'>({_html_escape(sev)})</font>",
            styles["Normal"],
        ))
        if issue.get("excerpt"):
            flow.append(Paragraph(
                f"&nbsp;&nbsp;&nbsp;&nbsp;<i>Current: &ldquo;{_html_escape(issue['excerpt'])}&rdquo;</i>",
                styles["Normal"],
            ))
        if issue.get("screenshot"):
            try:
                data = base64.b64decode(issue["screenshot"])
                nat_w_px, nat_h_px = ImageReader(BytesIO(data)).getSize()
                # Convert pixel dimensions to points, then scale down (never up)
                # to fit the page while preserving the source aspect ratio.
                nat_w_pt = nat_w_px * PX_TO_PT
                nat_h_pt = nat_h_px * PX_TO_PT
                scale = min(1.0, MAX_IMAGE_WIDTH / nat_w_pt, MAX_IMAGE_HEIGHT / nat_h_pt)
                flow.append(Spacer(1, 0.2 * cm))
                flow.append(Image(BytesIO(data),
                                  width=nat_w_pt * scale,
                                  height=nat_h_pt * scale))
            except Exception:
                flow.append(Paragraph("<i>(screenshot could not be embedded)</i>", styles["Italic"]))

    # Overall sign-off (Step ✔ of the checklist), kept compact at the end.
    reasoning = report.get("reasoning")
    if isinstance(reasoning, dict) and reasoning:
        verdict = str(reasoning.get("verdict", "")).upper() or "PARTIAL"
        v_colour = VERDICT_COLOURS.get(verdict, "#333333")
        flow.append(Spacer(1, 0.5 * cm))
        flow.append(Paragraph(
            f"<b>Overall result:</b> <font color='{v_colour}'>{verdict}</font>",
            styles["Normal"],
        ))
        if reasoning.get("summary"):
            flow.append(Paragraph(_html_escape(reasoning["summary"]), styles["Normal"]))

    doc.build(flow)
    return out_path
