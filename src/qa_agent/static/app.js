"use strict";

const $ = (sel) => document.querySelector(sel);
const form = $("#qa-form");
const runBtn = $("#run");
const resetBtn = $("#reset");
const progressEl = $("#progress");
const errorEl = $("#error");
const errorBody = $("#error-body");
const reportEl = $("#report");
const statusEl = $("#status");
const statusLabel = statusEl.querySelector(".label");
const elapsedEl = $("#progress-elapsed");

const SEVERITY_CLASS = { Critical: "critical", Minor: "minor", Info: "info" };

let elapsedTimer = null;

function show(el) { el.classList.remove("hidden"); }
function hide(el) { el.classList.add("hidden"); }

function setStatus(state, text) {
  statusEl.classList.remove("status--ok", "status--bad", "status--unknown");
  statusEl.classList.add(`status--${state}`);
  statusLabel.textContent = text;
}

async function refreshHealth() {
  try {
    const r = await fetch("/api/health");
    const j = await r.json();
    $("#be-url").textContent = location.origin;
    $("#mcp-url").textContent = j.mcp_url || "(unset)";
    if (j.mcp_reachable) {
      setStatus("ok", `MCP reachable (${j.mcp_status})`);
    } else {
      setStatus("bad", `MCP unreachable: ${j.mcp_status}`);
    }
  } catch (err) {
    setStatus("bad", `health failed: ${err}`);
  }
}

function startTimer() {
  const t0 = Date.now();
  elapsedEl.textContent = "elapsed: 0s";
  elapsedTimer = setInterval(() => {
    const s = Math.floor((Date.now() - t0) / 1000);
    elapsedEl.textContent = `elapsed: ${s}s`;
  }, 1000);
}
function stopTimer() {
  if (elapsedTimer) clearInterval(elapsedTimer);
  elapsedTimer = null;
}

function severityCounts(issues) {
  const c = { Critical: 0, Minor: 0, Info: 0, Other: 0 };
  for (const i of issues) {
    const s = i.severity || "Other";
    if (s in c) c[s]++; else c.Other++;
  }
  return c;
}

function renderIssue(issue, index) {
  const el = document.createElement("article");
  el.className = `issue ${SEVERITY_CLASS[issue.severity] || "info"}`;

  const head = document.createElement("div");
  head.className = "issue__head";
  head.innerHTML = `
    <span class="issue__type">${index + 1}. ${escapeHtml(issue.type || "Issue")}</span>
    <span class="tag ${SEVERITY_CLASS[issue.severity] || "info"}">${escapeHtml(issue.severity || "Info")}</span>
    ${issue.ruleId ? `<span class="tag">rule ${escapeHtml(issue.ruleId)}</span>` : ""}
  `;
  el.appendChild(head);

  if (issue.description) {
    const f = document.createElement("p");
    f.className = "issue__field";
    f.innerHTML = `<b>Description:</b> ${escapeHtml(issue.description)}`;
    el.appendChild(f);
  }
  if (issue.excerpt) {
    const f = document.createElement("p");
    f.className = "issue__field issue__excerpt";
    f.textContent = `“${issue.excerpt}”`;
    el.appendChild(f);
  }
  if (issue.suggestion) {
    const f = document.createElement("p");
    f.className = "issue__field";
    f.innerHTML = `<b>Suggestion:</b> ${escapeHtml(issue.suggestion)}`;
    el.appendChild(f);
  }
  if (issue.screenshot && /^[A-Za-z0-9+/=]+$/.test(issue.screenshot.slice(0, 40))) {
    const img = document.createElement("img");
    img.className = "issue__shot";
    img.alt = `Evidence for: ${issue.excerpt || issue.description || ""}`;
    img.loading = "lazy";
    img.src = `data:image/png;base64,${issue.screenshot}`;
    el.appendChild(img);
  }
  return el;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function renderReport(payload) {
  const { report, json_url, pdf_url, pdf_error } = payload;
  $("#report-title").textContent = report.course_name || "QA Report";
  const meta = [];
  if (report.url) meta.push(report.url);
  if (report.generated_at) meta.push(`generated ${report.generated_at}`);
  if (report.template_summary) meta.push(`template: ${report.template_summary}`);
  $("#report-meta").textContent = meta.join(" • ");

  const issues = report.issues || [];
  const counts = severityCounts(issues);
  const summary = $("#report-summary");
  summary.innerHTML = `
    <span class="pill"><strong>${issues.length}</strong> total</span>
    <span class="pill critical"><strong>${counts.Critical}</strong> critical</span>
    <span class="pill minor"><strong>${counts.Minor}</strong> minor</span>
    <span class="pill info"><strong>${counts.Info}</strong> info</span>
  `;

  const failures = report.tool_failures || [];
  const failuresEl = $("#report-failures");
  if (failures.length) {
    failuresEl.innerHTML = `
      <h3>Tool failures (${failures.length})</h3>
      <ul>${failures.map(f => `<li>${escapeHtml(f)}</li>`).join("")}</ul>
    `;
    show(failuresEl);
  } else {
    hide(failuresEl);
  }

  const reasoningEl = $("#report-reasoning");
  const reasoning = report.reasoning;
  if (reasoning && typeof reasoning === "object") {
    const verdict = String(reasoning.verdict || "").toUpperCase();
    const verdictEl = $("#reasoning-verdict");
    verdictEl.textContent = verdict || "—";
    verdictEl.className = `verdict verdict--${(verdict || "partial").toLowerCase()}`;
    $("#reasoning-summary").textContent = reasoning.summary || "(no summary)";
    const followed = $("#reasoning-followed");
    const gaps = $("#reasoning-gaps");
    followed.innerHTML = (reasoning.instructions_followed || [])
      .map(s => `<li>${escapeHtml(s)}</li>`).join("");
    gaps.innerHTML = (reasoning.gaps || [])
      .map(s => `<li>${escapeHtml(s)}</li>`).join("");
    show(reasoningEl);
  } else {
    hide(reasoningEl);
  }

  const issuesEl = $("#report-issues");
  issuesEl.innerHTML = "";
  if (!issues.length) {
    issuesEl.innerHTML = `<p class="muted">No issues reported.</p>`;
  } else {
    issues.forEach((iss, i) => issuesEl.appendChild(renderIssue(iss, i)));
  }

  const dlJson = $("#dl-json");
  const dlPdf = $("#dl-pdf");
  if (json_url) { dlJson.href = json_url; dlJson.classList.remove("hidden"); }
  else { dlJson.classList.add("hidden"); }
  if (pdf_url) { dlPdf.href = pdf_url; dlPdf.classList.remove("hidden"); dlPdf.textContent = "Download PDF"; }
  else if (pdf_error) { dlPdf.classList.add("hidden"); }
  else { dlPdf.classList.add("hidden"); }

  show(reportEl);
}

async function submitForm(ev) {
  ev.preventDefault();
  hide(errorEl);
  hide(reportEl);
  show(progressEl);
  runBtn.disabled = true;
  startTimer();

  const fd = new FormData(form);
  // Drop the file entry if no file was actually selected (else server tries to parse empty upload).
  const fileInput = $("#template_document");
  if (!fileInput.files || fileInput.files.length === 0) {
    fd.delete("template_document");
  }

  try {
    const r = await fetch("/api/qa", { method: "POST", body: fd });
    const text = await r.text();
    let json;
    try { json = JSON.parse(text); } catch { json = null; }
    if (!r.ok) {
      throw new Error((json && (json.detail || json.error)) || `${r.status} ${r.statusText}\n${text}`);
    }
    renderReport(json);
  } catch (err) {
    errorBody.textContent = err && err.message ? err.message : String(err);
    show(errorEl);
  } finally {
    hide(progressEl);
    runBtn.disabled = false;
    stopTimer();
  }
}

function resetForm() {
  form.reset();
  hide(reportEl);
  hide(errorEl);
}

form.addEventListener("submit", submitForm);
resetBtn.addEventListener("click", resetForm);

refreshHealth();
setInterval(refreshHealth, 15000);
