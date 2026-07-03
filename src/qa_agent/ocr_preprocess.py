"""OpenCV-assisted OCR for text baked into banners, hero/promo graphics and badges.

Course pages carry QA-critical claims (price, "Money Back Guarantee", "TrustScore
4.8", payment methods, "Enquire Now") as text *inside* images and banners — not as
HTML. A single `RGB -> Tesseract` pass reads clean dark-on-light text but routinely
returns **nothing** on the cases that actually matter here:

  * light text on a coloured / gradient button (e.g. light-teal "Enquire Now" on a
    dark-teal pill — Tesseract's default binarisation collapses both to one shade);
  * white text over a hero photo;
  * stylised promo bars with low luminance contrast.

When that text is lost, the downstream compliance check can't see the claim and
falsely reports it "not present" — the exact hallucination this module exists to
prevent. The fix is *computer vision before recognition*: use OpenCV to produce
several deterministic renderings of the crop (grayscale upscaled, Otsu and adaptive
thresholds, per-colour-channel thresholds, both polarities), OCR each, and keep the
single best-scoring result. We never invent text — every variant is a faithful
transform of the original pixels, and we only ever return what Tesseract reads.

Cost control: OCR is slow, so we run the *cheap* default pass first and only
escalate to the multi-variant OpenCV pipeline when that pass comes back empty or
low-confidence (`thorough=True` forces the full pipeline — used for banners, which
are few and always QA-relevant). If OpenCV / NumPy aren't importable the module
degrades to a Pillow-only variant set, and if Tesseract itself is missing every
entry point returns empty rather than raising.
"""

from __future__ import annotations

import logging
import os
import shutil
from io import BytesIO

import pytesseract
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

# OpenCV + NumPy are optional. Their absence only costs us the richer variant set
# (we fall back to Pillow transforms), never the whole OCR path.
try:  # pragma: no cover - import guard
    import cv2
    import numpy as np

    _CV2_AVAILABLE = True
except Exception:  # noqa: BLE001
    cv2 = None  # type: ignore
    np = None  # type: ignore
    _CV2_AVAILABLE = False


# Common Windows / Linux / macOS install locations probed when TESSERACT_CMD is
# not set. WHY: extraction.py used to read TESSERACT_CMD only at import time, so a
# module imported before load_dotenv() ran left pytesseract unconfigured and
# disabled OCR for the ENTIRE run (every banner/image text silently dropped, which
# compliance then mis-reported as missing). Discovering the binary here makes OCR
# robust to import ordering and to a forgotten env var.
_TESSERACT_CANDIDATES = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    "/usr/bin/tesseract",
    "/usr/local/bin/tesseract",
    "/opt/homebrew/bin/tesseract",
)

# OCR quality knobs. A word must beat MIN_WORD_CONF to count toward a variant's
# score; a cheap first pass clearing GOOD_ENOUGH_CONF with real words skips the
# expensive escalation.
MIN_WORD_CONF = float(os.environ.get("QA_OCR_MIN_WORD_CONF", "40"))
GOOD_ENOUGH_CONF = float(os.environ.get("QA_OCR_GOOD_ENOUGH_CONF", "70"))
# Upscale small crops toward Tesseract's preferred ~300 DPI; banner pills are
# often only a couple hundred px wide, which Tesseract reads poorly.
_UPSCALE_TARGET_PX = 1400
_MAX_UPSCALE = 4.0
# Hard ceiling on (variant x PSM) OCR attempts in the heavy pass, so a genuinely
# unreadable low-contrast crop can't run dozens of slow Tesseract calls. Each
# attempt is ~0.2-0.3s; the cap bounds worst-case added latency per image.
_MAX_OCR_ATTEMPTS = int(os.environ.get("QA_OCR_MAX_ATTEMPTS", "18"))


def configure_tesseract() -> str | None:
    """Point pytesseract at a real tesseract binary; return its path or None.

    Honours ``TESSERACT_CMD`` first, then anything already on PATH, then the
    well-known install locations. Safe to call repeatedly (idempotent).
    """
    env_cmd = os.environ.get("TESSERACT_CMD", "").strip()
    if env_cmd and os.path.isfile(env_cmd):
        pytesseract.pytesseract.tesseract_cmd = env_cmd
        return env_cmd

    on_path = shutil.which("tesseract")
    if on_path:
        pytesseract.pytesseract.tesseract_cmd = on_path
        return on_path

    for cand in _TESSERACT_CANDIDATES:
        if os.path.isfile(cand):
            pytesseract.pytesseract.tesseract_cmd = cand
            return cand
    return None


def tesseract_available() -> bool:
    """True if a usable tesseract binary is configured / discoverable."""
    configure_tesseract()
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:  # noqa: BLE001
        return False


def opencv_available() -> bool:
    return _CV2_AVAILABLE


# ---------------------------------------------------------------------------
# Scoring: pick the variant Tesseract read most confidently
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    import re

    return re.sub(r"\s+", " ", text or "").strip()


def _score_ocr(img: Image.Image, psm: int) -> tuple[str, float, float]:
    """OCR one rendering at one PSM. Returns (text, score, mean_confident_conf).

    ``score`` rewards more words read above ``MIN_WORD_CONF`` (recall is what we
    care about — a missed price is the failure mode), then total characters, then
    mean confidence, so the best rendering wins a head-to-head comparison.
    """
    try:
        data = pytesseract.image_to_data(
            img, config=f"--psm {psm}", output_type=pytesseract.Output.DICT
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("tesseract image_to_data failed (psm %s): %s", psm, exc)
        return "", 0.0, 0.0

    words: list[str] = []
    good_confs: list[float] = []
    for word, conf in zip(data.get("text", []), data.get("conf", [])):
        word = (word or "").strip()
        if not word:
            continue
        try:
            c = float(conf)
        except (TypeError, ValueError):
            c = -1.0
        words.append(word)
        if c >= MIN_WORD_CONF:
            good_confs.append(c)

    text = _clean(" ".join(words))
    if not text:
        return "", 0.0, 0.0
    mean_conf = (sum(good_confs) / len(good_confs)) if good_confs else 0.0
    # Recall-first composite score (kept comparable across variants).
    score = len(good_confs) * 1000 + len(text) + mean_conf
    return text, score, round(mean_conf, 1)


# ---------------------------------------------------------------------------
# Variant generation
# ---------------------------------------------------------------------------

def _upscale_factor(width: int, height: int) -> float:
    longest = max(width, height) or 1
    if longest >= _UPSCALE_TARGET_PX:
        return 1.0
    return min(_MAX_UPSCALE, _UPSCALE_TARGET_PX / longest)


def _cv_variants(pil_img: Image.Image):
    """Yield OpenCV-derived binarisations as PIL images, best-effort.

    Covers the failure modes raw Tesseract chokes on: light-on-dark (handled by
    emitting both polarities), coloured low-contrast text (per-channel Otsu — the
    channel with the widest text/background separation often binarises cleanly
    when grayscale does not), and small crops (upscaled first).
    """
    if not _CV2_AVAILABLE:
        return
    try:
        rgb = np.array(pil_img.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        factor = _upscale_factor(w, h)
        if factor > 1.0:
            bgr = cv2.resize(bgr, (int(w * factor), int(h * factor)),
                             interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        planes = [("gray", gray)]
        # Per-channel planes catch coloured text that vanishes in luminance.
        b, g, r = cv2.split(bgr)
        planes += [("b", b), ("g", g), ("r", r)]

        for _name, plane in planes:
            plane = cv2.bilateralFilter(plane, 5, 40, 40)
            # Otsu, both polarities (light-on-dark vs dark-on-light).
            _, otsu = cv2.threshold(plane, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            yield Image.fromarray(otsu)
            yield Image.fromarray(cv2.bitwise_not(otsu))
        # One adaptive pass on grayscale for uneven gradient backgrounds.
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 10,
        )
        yield Image.fromarray(adaptive)
        yield Image.fromarray(cv2.bitwise_not(adaptive))
    except Exception as exc:  # noqa: BLE001
        logger.debug("OpenCV variant generation failed: %s", exc)


def _pil_variants(pil_img: Image.Image):
    """Pillow-only fallback variants (used when OpenCV isn't installed).

    Mirrors the OpenCV set's intent: upscale, per-channel autocontrast, threshold
    at a few cut points, both polarities. Less powerful than Otsu/adaptive but
    still recovers many light-on-coloured cases.
    """
    rgb = pil_img.convert("RGB")
    factor = _upscale_factor(*rgb.size)
    if factor > 1.0:
        rgb = rgb.resize((int(rgb.width * factor), int(rgb.height * factor)),
                         Image.LANCZOS)
    planes = [ImageOps.grayscale(rgb), *rgb.split()]
    for plane in planes:
        ac = ImageOps.autocontrast(plane, cutoff=2)
        for thr in (110, 140, 170):
            binary = ac.point(lambda x, t=thr: 255 if x > t else 0)
            yield binary
            yield ImageOps.invert(binary)


# ---------------------------------------------------------------------------
# Public OCR entry points
# ---------------------------------------------------------------------------

def ocr_image(pil_img: Image.Image, *, thorough: bool = False) -> tuple[str, float]:
    """Best-effort OCR of a crop. Returns (text, mean_confidence 0-100).

    Cheap first: a plain pass on the original image. If that already reads real
    text confidently (and we weren't asked to be ``thorough``) we keep it. Only an
    empty / low-confidence first pass triggers the multi-variant OpenCV pipeline,
    so the heavy work happens exactly on the banners and badges that need it.
    """
    if not tesseract_available():
        return "", 0.0

    best_text, best_score, best_conf = "", 0.0, 0.0
    attempts = 0

    def consider(text: str, score: float, conf: float) -> None:
        nonlocal best_text, best_score, best_conf, attempts
        attempts += 1
        if score > best_score:
            best_text, best_score, best_conf = text, score, conf

    def strong() -> bool:
        # A confident multi-word read — good enough to stop early.
        return best_conf >= GOOD_ENOUGH_CONF and best_score >= 2000

    # Pass 1 — the original image at the default page-segmentation mode.
    consider(*_score_ocr(pil_img.convert("RGB"), psm=3))
    if not thorough and best_text and strong():
        return best_text, best_conf

    # Pass 2 — CV / PIL renderings at PSM 6 (uniform block), 7 (single line —
    # button/pill labels like "Enquire Now" or a bare price) and 11 (sparse
    # scattered badge text), covering the banner layouts we see. Stop early on a
    # strong read, or once the attempt budget is spent so a genuinely unreadable
    # crop can't run dozens of slow Tesseract calls.
    variant_gen = _cv_variants(pil_img) if _CV2_AVAILABLE else _pil_variants(pil_img)
    for variant in variant_gen:
        for psm in (6, 7, 11):
            consider(*_score_ocr(variant, psm))
            if strong() or attempts >= _MAX_OCR_ATTEMPTS:
                return best_text, best_conf

    return best_text, best_conf


def ocr_png(png_bytes: bytes, *, thorough: bool = False) -> tuple[str, float]:
    """OCR raw PNG bytes (e.g. a Playwright element screenshot)."""
    if not png_bytes:
        return "", 0.0
    try:
        with Image.open(BytesIO(png_bytes)) as img:
            return ocr_image(img, thorough=thorough)
    except Exception as exc:  # noqa: BLE001
        logger.debug("ocr_png failed to open image: %s", exc)
        return "", 0.0
