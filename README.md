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

1. The **MCP server** (`mcp_server.py`) exposes five tools — `scrape`, `screenshot`,
   `spell`, `template`, `compliance` — over MCP streamable-HTTP.
2. The **Strands Agent** (`agent.py`) connects via `MCPClient`, lists those tools,
   and is steered by a system prompt that prescribes the QA flow.
3. The **OpenRouter** model provider (`llm.py`) wraps Strands' `OpenAIModel` with
   the OpenRouter base URL — so any OpenAI-compatible model on OpenRouter works.
4. The **CLI** (`main.py`) writes both a JSON report and a PDF (`report_tool.py`,
   ReportLab) into `reports/`.

## Execution model

The CLI runs a single execution path: the **Strands agent** orchestrates the
MCP tools (`scrape`, `template`, `spell`, `compliance`, `evidence`) under a
strict system prompt that calls each tool at most once and emits one JSON
object at the end. The CLI parses that JSON, persists it, and renders the PDF.

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
