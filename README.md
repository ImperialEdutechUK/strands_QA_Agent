# Strands QA Agent

AI-powered QA agent that audits course web pages for spelling/grammar (UK English),
checks them against an image- or text-based QA template, captures screenshots, and
exports a structured PDF report.

Built on the **real [Strands Agents SDK](https://strandsagents.com)** (Python),
with QA tools exposed via an MCP server (FastMCP, streamable-HTTP transport) and
the LLM provided by **OpenRouter** (DeepSeek by default).

See [Instruction_guide.md](Instruction_guide.md) for the architecture walkthrough.

## Project layout

```
.
├── Instruction_guide.md
├── README.md
├── requirements.txt
├── pyproject.toml
├── .env.example
└── src/qa_agent/
    ├── __init__.py
    ├── llm.py                # Strands OpenAIModel pointed at OpenRouter
    ├── llm_client.py         # Direct OpenRouter client for structured-JSON tools
    ├── mcp_server.py         # FastMCP server exposing the QA tools
    ├── agent.py              # Strands Agent + MCPClient wiring
    ├── main.py               # CLI entry point
    ├── security.py           # SSRF / path / secret / redaction helpers
    ├── logging_config.py     # Logging with secret redaction
    └── tools/
        ├── web_tools.py      # Playwright scrape + screenshot (URL validated)
        ├── spell_tool.py     # UK English spelling/grammar check
        ├── template_tool.py  # Tesseract OCR + rule extraction (path validated)
        ├── compliance_tool.py# Page-vs-rules compliance check
        └── report_tool.py    # ReportLab PDF generator
```

## Setup

1. **Python 3.11+** and a virtual environment:
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate

   pip install -r requirements.txt
   pip install -e .          # installs the qa_agent package in editable mode
   ```

2. **Playwright Chromium**:
   ```bash
   playwright install chromium
   ```

3. **Tesseract OCR** (system binary — needed for image templates):
   - Windows: https://github.com/UB-Mannheim/tesseract/wiki then add to PATH
     (or set `TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe` in `.env`)
   - macOS: `brew install tesseract`
   - Debian/Ubuntu: `sudo apt-get install tesseract-ocr`

4. **Environment variables**:
   ```bash
   cp .env.example .env
   ```
   Set `OPENROUTER_API_KEY` (get one at https://openrouter.ai). Other vars are optional.

## Run

Open two terminals.

**Terminal 1 — MCP server:**
```bash
python -m qa_agent.mcp_server
```
You should see `MCP Server running on http://localhost:3001/mcp`.

**Terminal 2 — agent CLI:**
```bash
# URL only — the Strands agent picks the tool sequence
python -m qa_agent.main --url https://example.com/your-course

# With a text template
python -m qa_agent.main --url https://example.com/your-course \
  --template-text "All headings sentence case. Page must include learning outcomes."

# With a template file (image, PDF, or Word docx; images need Tesseract installed)
python -m qa_agent.main --url https://example.com/your-course --template ./qa-template.pdf
```

> **Cost note:** the agent orchestrates each MCP tool call itself, so a run
> uses several LLM calls (one per tool step plus orchestration). The system
> prompt restricts each tool to one call per run to keep cost bounded.

Outputs land in `reports/qa-report-<timestamp>.json` and `.pdf`.

## How it works

1. The **MCP server** (`mcp_server.py`) exposes the QA tools — `scrape`,
   `extract`, `screenshot`, `evidence`, `spell`, `template`, `compliance`,
   `reason` — over MCP streamable-HTTP.
2. The **Strands Agent** (`agent.py`) connects via `MCPClient`, lists those tools,
   and is steered by a system prompt that prescribes the QA flow.
3. The **OpenRouter** model provider (`llm.py`) wraps Strands' `OpenAIModel` with
   the OpenRouter base URL — so any OpenAI-compatible model on OpenRouter works.
4. The **CLI** (`main.py`) writes both a JSON report and a PDF (`report_tool.py`,
   ReportLab) into `reports/`.

## Layered evidence extraction

For QA comparisons that need detailed, image-aware evidence (banners, hero
sections, promotional graphics, accreditation/trust badges, course thumbnails,
CTA blocks, and image-based claims), use the dedicated extraction stage in
[`extraction.py`](src/qa_agent/extraction.py). It is **deterministic** (no LLM),
so the evidence is faithful and reproducible, and it is exposed both as an MCP
tool (`extract`) and a standalone CLI:

```bash
python -m qa_agent.extraction --url https://example.com/your-course
# options: --no-wordpress  --no-screenshots  --no-ocr  --out report.json
```

It runs a layered pipeline:

1. **WordPress REST API** (`/wp-json/wp/v2/pages|posts?slug=…&_embed=1`) for
   structured page + media metadata (title, slug, content, featured media, alt
   text, captions, modified date) when the site exposes it.
2. **Rendered DOM** via Playwright with lazy-load auto-scroll, so JS-rendered
   text, sliders, banners, accordions, tabs and page-builder sections are seen.
3. **Image discovery** beyond `<img>` — `<picture>`/`<source srcset>`, `srcset`,
   lazy `data-*` attributes, computed CSS `background-image`, inline styles, and
   Elementor/Divi/WPBakery/Fusion/Bricks page-builder backgrounds.
4. **Carousel slides** read straight from the DOM (every slide, not just the
   first visible one), marked `is_carousel` + `slide_index`.
5. **Element screenshots** as evidence for banners and QA-relevant images.
6. **OCR** (Tesseract) of text baked into banner/promo images, with a confidence
   flag (gracefully skipped if Tesseract isn't installed).
7. **Claim detection** (price, discount, duration, certification, accreditation,
   awarding body, eligibility, guarantee, rating, urgency, learner numbers) and
   **QA-priority** scoring (high/medium/low) per the agreed rules.

The full structured report (`page_url`, `general_content`, `banners`, `images`,
`extraction_warnings`, plus `wordpress`/`stats`) is written under
`reports/extraction/`; element screenshots land alongside it. The MCP `extract`
tool persists the same full report to disk and returns only a compact summary
(counts + high-priority items + the report path) to keep the agent's LLM context
small. Set `QA_EXTRACTION_DIR` to change the output location and
`QA_EXTRACTION_MAX_OCR` to bound OCR cost.

## Execution model

The CLI runs a single execution path: the **Strands agent** orchestrates the
MCP tools (`extract`, `template`, `spell`, `compliance`, `evidence`, `reason`)
under a strict system prompt that calls each tool at most once and emits one
JSON object at the end. `extract` is the layered evidence stage and runs first
automatically; its high-priority banner/image claims (including OCR text) are
passed into `compliance`, so claims baked into banners or images are checked
against the template rules alongside the body text. The CLI parses the JSON,
persists it, and renders the PDF.

## Security

See [SECURITY.md](SECURITY.md). Highlights:

- `.env` for secrets (gitignored); startup validation; log redaction.
- SSRF protection on every URL handed to Playwright.
- Path-traversal protection + extension/size allowlist on template images.
- MCP server binds to `127.0.0.1` by default and supports optional bearer-token
  auth via `MCP_AUTH_TOKEN` (constant-time comparison).
- Hardened HTTP client (TLS verify, no auto-redirects, timeouts, pool limits).
- Configurable input/output size caps.

Generate an MCP token with `python -c "import secrets; print(secrets.token_urlsafe(32))"`.
