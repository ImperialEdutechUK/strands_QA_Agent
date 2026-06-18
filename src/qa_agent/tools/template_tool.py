"""Template document interpretation.

A QA template can arrive as:
  * an image (PNG / JPEG / WebP / etc) — OCR'd directly;
  * a PDF — every page's text is extracted, and every embedded image on every
    page is OCR'd, so screenshots-of-rules pasted into a brief still surface;
  * a Word .docx — every paragraph + table cell is read, and every embedded
    image is OCR'd.

In every case the extracted text is merged into a single string and handed to
the LLM, which produces the structured rule list (`{summary, rules: [...]}`).
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path

import pytesseract
from PIL import Image

from ..llm_client import call_llm_json
from ..security import (
    ALLOWED_DOC_SUFFIXES,
    ALLOWED_IMAGE_SUFFIXES,
    UnsafePathError,
    safe_resolve_template,
    truncate_text,
)

logger = logging.getLogger(__name__)

if os.environ.get("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_CMD"]

SYSTEM = (
    "You convert QA template documents (often supplied as PDFs, Word files, "
    "or images) into a structured rule set that another QA agent can apply "
    "against a course web page."
)

SCHEMA_INSTRUCTION = """Return a JSON object with this shape:
{
  "summary": "<one-sentence summary of the template>",
  "rules": [
    {
      "id": "R1",
      "category": "Content" | "Structure" | "Style" | "Accessibility" | "Branding" | "Other",
      "rule": "<the rule, phrased as a check>",
      "severity": "Critical" | "Minor" | "Info"
    }
  ]
}
Output ONLY the JSON object."""


# Baseline rules: applied to every QA run regardless of template content.
# IDs use a B-prefix so they don't collide with template-derived R-prefixed IDs.
BASELINE_RULES: list[dict] = [
    {
        "id": "B1",
        "category": "Style",
        "rule": (
            "Body content across the course page must use a single, consistent "
            "font family. Heading fonts may differ from body fonts, but all "
            "headings of the same level should themselves be consistent. Flag "
            "any paragraph, list, or block that mixes fonts within body copy."
        ),
        "severity": "Critical",
    },
    {
        "id": "B2",
        "category": "Structure",
        "rule": (
            'The "What will I learn" section must be balanced: the number of '
            "bullet points on the left column must equal the number on the "
            "right column (off by at most one is acceptable only if the total "
            "is odd). Flag missing section, empty column, or visibly uneven "
            "split."
        ),
        "severity": "Critical",
    },
    {
        "id": "B3",
        "category": "Content",
        "rule": (
            "A pricing section must be present on the course page, identified "
            'by the simultaneous presence of "Buy Now" and "Enquire Now" '
            "buttons / icons. Flag if the section is missing, or if either "
            "button is absent."
        ),
        "severity": "Critical",
    },
]


def _merge_baseline(rules: list[dict]) -> list[dict]:
    """Prepend baseline rules, dropping any LLM-emitted rule that reuses a baseline ID."""
    baseline_ids = {r["id"] for r in BASELINE_RULES}
    extra = [r for r in (rules or []) if r.get("id") not in baseline_ids]
    return [dict(r) for r in BASELINE_RULES] + extra


# ---------------------------------------------------------------------------
# Per-format extraction helpers
# ---------------------------------------------------------------------------

def _ocr_image(img: Image.Image) -> str:
    try:
        return (pytesseract.image_to_string(img) or "").strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("OCR failed on embedded image: %s", exc)
        return ""


def _extract_image(path: Path) -> str:
    with Image.open(path) as img:
        return _ocr_image(img)


def _extract_pdf(path: Path) -> str:
    """Pull text + OCR embedded images out of every page of the PDF."""
    # PyMuPDF imports as `fitz`. Imported lazily so an absent dep only breaks
    # PDF templates, not the whole tool.
    import fitz  # type: ignore

    chunks: list[str] = []
    with fitz.open(path) as doc:
        for page_index, page in enumerate(doc, start=1):
            text = (page.get_text("text") or "").strip()
            if text:
                chunks.append(f"[page {page_index} text]\n{text}")

            # Embedded images — useful when a brief is largely screenshots.
            for img_idx, img_info in enumerate(page.get_images(full=True), start=1):
                xref = img_info[0]
                try:
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n - pix.alpha >= 4:
                        # CMYK / unsupported colourspace — convert to RGB.
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    png_bytes = pix.tobytes("png")
                    pix = None  # release immediately
                except Exception as exc:  # noqa: BLE001
                    logger.debug("PDF image extract failed (page %s, img %s): %s",
                                 page_index, img_idx, exc)
                    continue
                with Image.open(io.BytesIO(png_bytes)) as img:
                    ocr = _ocr_image(img)
                if ocr:
                    chunks.append(f"[page {page_index} image {img_idx} OCR]\n{ocr}")
    return "\n\n".join(chunks).strip()


def _extract_docx(path: Path) -> str:
    """Pull paragraphs, table cells, and embedded image OCR out of a DOCX."""
    from docx import Document  # type: ignore

    document = Document(str(path))
    chunks: list[str] = []

    paragraphs = [p.text.strip() for p in document.paragraphs if p.text and p.text.strip()]
    if paragraphs:
        chunks.append("[paragraphs]\n" + "\n".join(paragraphs))

    table_lines: list[str] = []
    for t_idx, table in enumerate(document.tables, start=1):
        for row in table.rows:
            cells = [(cell.text or "").strip() for cell in row.cells]
            cells = [c for c in cells if c]
            if cells:
                table_lines.append(" | ".join(cells))
    if table_lines:
        chunks.append("[tables]\n" + "\n".join(table_lines))

    # Inline / floating images live in the package's media parts.
    media_count = 0
    for rel in document.part.rels.values():
        if "image" not in rel.reltype:
            continue
        try:
            blob = rel.target_part.blob
        except Exception as exc:  # noqa: BLE001
            logger.debug("DOCX image extract failed: %s", exc)
            continue
        try:
            with Image.open(io.BytesIO(blob)) as img:
                ocr = _ocr_image(img)
        except Exception as exc:  # noqa: BLE001
            logger.debug("DOCX image OCR failed: %s", exc)
            continue
        if ocr:
            media_count += 1
            chunks.append(f"[image {media_count} OCR]\n{ocr}")

    return "\n\n".join(chunks).strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _llm_rules_from_text(label: str, extracted: str) -> dict:
    if not extracted:
        return {
            "summary": f"Empty template ({label}: no text extracted) — baseline rules only.",
            "rules": _merge_baseline([]),
        }
    prompt = (
        f"{SCHEMA_INSTRUCTION}\n\n"
        f"EXTRACTED CONTENT FROM TEMPLATE ({label}):\n"
        f'"""{truncate_text(extracted)}"""'
    )
    result = call_llm_json(prompt, system=SYSTEM)
    return {
        "summary": result.get("summary", ""),
        "rules": _merge_baseline(result.get("rules") or []),
    }


def analyse_template(template_path: str) -> dict:
    """Interpret a template document (image / PDF / DOCX) into a rule list.

    Only filesystem paths under the configured allowed roots are accepted; raw
    base64 / data URIs are rejected on purpose to avoid being a generic OCR
    endpoint that processes attacker-supplied bytes.

    The raw extracted document text is intentionally NOT returned: the agent
    only needs `summary` + `rules` to drive compliance, and echoing the full
    document body back into Strands' tool-result history bloats every
    subsequent LLM turn and was a primary cause of mid-stream "Network
    connection lost" errors on OpenRouter.
    """
    safe_path = safe_resolve_template(template_path)
    suffix = safe_path.suffix.lower()
    if suffix in ALLOWED_IMAGE_SUFFIXES:
        extracted = _extract_image(safe_path)
        label = "image OCR"
    elif suffix == ".pdf":
        extracted = _extract_pdf(safe_path)
        label = "PDF"
    elif suffix == ".docx":
        extracted = _extract_docx(safe_path)
        label = "Word docx"
    else:
        raise UnsafePathError(f"Unsupported template extension: {suffix}")

    out = _llm_rules_from_text(label, extracted.strip())
    out["source_kind"] = label
    return out


def analyse_template_text(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        return {"summary": "Empty template — baseline rules only.", "rules": _merge_baseline([])}
    prompt = f'{SCHEMA_INSTRUCTION}\n\nTEMPLATE TEXT:\n"""{truncate_text(text)}"""'
    result = call_llm_json(prompt, system=SYSTEM)
    return {
        "summary": result.get("summary", ""),
        "rules": _merge_baseline(result.get("rules") or []),
    }
