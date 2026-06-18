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
    PageBreak,
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


def generate_pdf(report: dict, out_path: str) -> str:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(out_path, pagesize=A4, title=report.get("course_name", "QA Report"))

    flow: list = [
        Paragraph(report.get("course_name", "QA Report"), styles["Title"]),
        Paragraph(report.get("url", ""), styles["Normal"]),
        Paragraph(f"Generated: {report.get('generated_at', '')}", styles["Italic"]),
        Spacer(1, 0.5 * cm),
    ]

    issues = report.get("issues", []) or []
    counts = _count_severities(issues)
    flow.append(Paragraph(
        f"<b>Total issues:</b> {len(issues)} &nbsp; "
        f"<font color='{SEVERITY_COLOURS['Critical']}'>Critical: {counts['Critical']}</font> &nbsp; "
        f"<font color='{SEVERITY_COLOURS['Minor']}'>Minor: {counts['Minor']}</font> &nbsp; "
        f"<font color='{SEVERITY_COLOURS['Info']}'>Info: {counts['Info']}</font>",
        styles["Normal"],
    ))

    if report.get("template_summary"):
        flow.append(Spacer(1, 0.3 * cm))
        flow.append(Paragraph(f"<b>Template:</b> {report['template_summary']}", styles["Normal"]))

    reasoning = report.get("reasoning")
    if isinstance(reasoning, dict) and reasoning:
        verdict = str(reasoning.get("verdict", "")).upper() or "PARTIAL"
        v_colour = VERDICT_COLOURS.get(verdict, "#333333")
        flow.append(Spacer(1, 0.5 * cm))
        flow.append(Paragraph("Agent self-review", styles["Heading2"]))
        flow.append(Paragraph(
            f"<b>Verdict:</b> <font color='{v_colour}'>{verdict}</font>",
            styles["Normal"],
        ))
        if reasoning.get("summary"):
            flow.append(Paragraph(
                f"<b>Summary:</b> {_html_escape(reasoning['summary'])}",
                styles["Normal"],
            ))
        if reasoning.get("instructions_followed"):
            flow.append(Paragraph("<b>Instructions followed:</b>", styles["Normal"]))
            for item in reasoning["instructions_followed"]:
                flow.append(Paragraph(f"• {_html_escape(item)}", styles["Normal"]))
        if reasoning.get("gaps"):
            flow.append(Paragraph("<b>Gaps:</b>", styles["Normal"]))
            for item in reasoning["gaps"]:
                flow.append(Paragraph(f"• {_html_escape(item)}", styles["Normal"]))

    for i, issue in enumerate(issues, start=1):
        flow.append(PageBreak())
        colour = SEVERITY_COLOURS.get(issue.get("severity", "Info"), "#333333")
        flow.append(Paragraph(
            f'<font color="{colour}">Issue {i} — {issue.get("type", "Issue")} '
            f'({issue.get("severity", "Info")})</font>',
            styles["Heading2"],
        ))
        if issue.get("ruleId"):
            flow.append(Paragraph(f"<b>Rule:</b> {issue['ruleId']}", styles["Normal"]))
        if issue.get("description"):
            flow.append(Paragraph(f"<b>Description:</b> {issue['description']}", styles["Normal"]))
        if issue.get("excerpt"):
            flow.append(Paragraph(f"<b>Excerpt:</b> <i>&ldquo;{issue['excerpt']}&rdquo;</i>", styles["Normal"]))
        if issue.get("suggestion"):
            flow.append(Paragraph(f"<b>Suggestion:</b> {issue['suggestion']}", styles["Normal"]))
        if issue.get("screenshot"):
            try:
                data = base64.b64decode(issue["screenshot"])
                nat_w_px, nat_h_px = ImageReader(BytesIO(data)).getSize()
                # Convert pixel dimensions to points, then scale down (never up)
                # to fit the page while preserving the source aspect ratio.
                nat_w_pt = nat_w_px * PX_TO_PT
                nat_h_pt = nat_h_px * PX_TO_PT
                scale = min(1.0, MAX_IMAGE_WIDTH / nat_w_pt, MAX_IMAGE_HEIGHT / nat_h_pt)
                flow.append(Spacer(1, 0.4 * cm))
                flow.append(Image(BytesIO(data),
                                  width=nat_w_pt * scale,
                                  height=nat_h_pt * scale))
            except Exception:
                flow.append(Paragraph("<i>(screenshot could not be embedded)</i>", styles["Italic"]))

    doc.build(flow)
    return out_path
