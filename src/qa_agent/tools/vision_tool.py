"""Screenshot-based visual QA pass — the agent's eyes on the rendered page.

The text pipeline deliberately refuses to judge visual rules (bold awarding
body, clean alignment, consistent fonts, bullet styling, logo presence /
blurriness) because asserting them from extracted text is fabrication. This
pass checks exactly those rules by actually LOOKING at the page: it takes a
full-page screenshot, slices it into ordered viewport-sized crops, and sends
them to an image-capable model (QA_VISION_MODEL — the DeepSeek text models
cannot see images, so this is a separate, cheap multimodal slug).

Grounding rules mirror the text pass: only clear, visible violations are
emitted; anything uncertain is dropped, and the same non-issue filter strips
pass/confirmation entries.
"""

from __future__ import annotations

import base64
import logging
import os
from io import BytesIO

from PIL import Image

from ..llm_client import call_llm_json
from .compliance_tool import _is_real_issue
from .web_tools import take_screenshot

logger = logging.getLogger(__name__)

VISION_MODEL = os.environ.get("QA_VISION_MODEL", "").strip()

# Slice geometry: page screenshots are 1440px wide and often 15,000+ px tall.
# We downscale to a readable width and cut into overlapping slices so a section
# straddling a boundary is fully visible in at least one crop. JPEG keeps the
# payload small; page text is still perfectly legible at quality 80.
_SLICE_WIDTH = int(os.environ.get("QA_VISION_WIDTH", "1100"))
_SLICE_HEIGHT = int(os.environ.get("QA_VISION_SLICE_HEIGHT", "1500"))
_SLICE_OVERLAP = 120
_MAX_SLICES = int(os.environ.get("QA_VISION_MAX_SLICES", "12"))
_JPEG_QUALITY = 80

SYSTEM = (
    "You are a professional front-end QA reviewer LOOKING AT SCREENSHOTS of a "
    "course web page. The images are ordered top-to-bottom slices of the full "
    "page (with a small overlap between consecutive slices). "
    "Check ONLY the visual rules provided — formatting you can actually SEE: "
    "bolding, alignment, font consistency, bullet styling, logo presence and "
    "sharpness. Be strict but factual: emit an issue ONLY for a rule the page "
    "clearly and visibly VIOLATES in the screenshots. "
    "NEVER emit confirmations ('alignment is clean', 'logos are present') — "
    "rules the page satisfies must not appear in the output at all. "
    "If a rule cannot be judged from the screenshots, or you are unsure, emit "
    "NOTHING for it. Content duplicated across two consecutive slices is the "
    "overlap, not a duplication defect. Widget chrome (accordion +/- icons, "
    "carousel arrows/dots, chat bubbles) is normal UI, not a defect. "
    "In `excerpt` quote the nearby visible text or name the section so a "
    "human can find the spot. Use UK English."
)

SCHEMA_INSTRUCTION = """Return a JSON object:
{
  "issues": [
    {
      "ruleId": "<rule id>",
      "type": "Template",
      "severity": "Critical" | "Minor" | "Info",
      "description": "<the visible problem>",
      "suggestion": "<direct edit instruction, e.g. 'Bold the awarding body name in the overview.'>",
      "excerpt": "<visible text near the problem, or the section name>"
    }
  ]
}
If the page passes every rule, return {"issues": []}. Output ONLY the JSON object."""


def vision_available() -> bool:
    return bool(VISION_MODEL)


def _slice_page_png(png_b64: str) -> list[str]:
    """Downscale + cut a full-page PNG into ordered JPEG slices (base64 data URIs)."""
    img = Image.open(BytesIO(base64.b64decode(png_b64))).convert("RGB")
    if img.width > _SLICE_WIDTH:
        ratio = _SLICE_WIDTH / img.width
        img = img.resize((_SLICE_WIDTH, max(1, int(img.height * ratio))), Image.LANCZOS)
    slices: list[str] = []
    step = _SLICE_HEIGHT - _SLICE_OVERLAP
    top = 0
    while top < img.height and len(slices) < _MAX_SLICES:
        crop = img.crop((0, top, img.width, min(top + _SLICE_HEIGHT, img.height)))
        buf = BytesIO()
        crop.save(buf, "JPEG", quality=_JPEG_QUALITY)
        slices.append(
            "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        )
        top += step
    if top < img.height:
        logger.info("vision: page taller than %d slices — bottom %dpx not reviewed",
                    _MAX_SLICES, img.height - top)
    return slices


def _format_rules(rules: list[dict]) -> str:
    lines = [
        f"[{r.get('id', '?')}] ({r.get('severity', 'Info')}) {r.get('rule', '')}"
        for r in rules if isinstance(r, dict)
    ]
    return "\n".join(lines)


def check_visual_rules(url: str, rules: list[dict]) -> dict:
    """LOOK at the rendered page and audit it against the visual rules.

    Returns {"issues": [...]} (possibly with a "skipped" note). Degrades
    gracefully: no vision model configured, or any capture/LLM failure, returns
    an empty issue list with the reason — visual rules are additive evidence,
    never worth failing the run over.
    """
    if not rules:
        return {"issues": []}
    if not vision_available():
        return {"issues": [], "skipped": "QA_VISION_MODEL not configured"}
    try:
        png_b64 = take_screenshot(url, full_page=True)
        slices = _slice_page_png(png_b64)
    except Exception as exc:  # noqa: BLE001
        logger.warning("vision: page capture failed: %s", exc)
        return {"issues": [], "skipped": f"screenshot failed: {type(exc).__name__}"}
    if not slices:
        return {"issues": [], "skipped": "empty screenshot"}

    prompt = (
        f"{SCHEMA_INSTRUCTION}\n\n"
        f"VISUAL RULES TO CHECK (id, severity, rule):\n{_format_rules(rules)}\n\n"
        f"The {len(slices)} attached images are top-to-bottom slices of the "
        f"full course page at {url}."
    )
    try:
        # disable_reasoning: reasoning models can spend the whole output budget
        # on hidden thinking and return empty content (seen with qwen3.5-flash).
        result = call_llm_json(prompt, system=SYSTEM, model=VISION_MODEL,
                               images=slices, max_tokens=3000,
                               disable_reasoning=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("vision: LLM call failed (%s: %s)", type(exc).__name__,
                       str(exc)[:160])
        return {"issues": [], "skipped": f"vision LLM failed: {type(exc).__name__}"}
    issues = result.get("issues") or []
    if not isinstance(issues, list):
        return {"issues": []}
    kept = []
    for i in issues:
        if not _is_real_issue(i):
            continue
        i.setdefault("type", "Template")
        kept.append(i)
    return {"issues": kept}
