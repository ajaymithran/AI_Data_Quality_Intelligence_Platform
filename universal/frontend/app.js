/* AI Data Quality Intelligence Platform — dashboard app (no build step).
   Deterministic engine = source of truth. Gemini = advisory. No fake data:
   every number shown comes from the API or the user's own actions. */
"use strict";

/* ---------------------------------------------------------------- helpers */
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (v) => v == null ? "" : String(v).replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (n) => n == null ? "—" : Number(n).toLocaleString();
const one = (n) => (n == null ? "—" : Number(n).toFixed(1));
const API = "api/universal";

const ICONS = {
  home: '<path d="M3 10.5 12 3l9 7.5"/><path d="M5 9.5V21h14V9.5"/>',
  upload: '<path d="M12 16V4"/><path d="m6 9 6-6 6 6"/><path d="M4 21h16"/>',
  alert: '<path d="M12 4 2.5 20h19L12 4z"/><path d="M12 10v4.5"/><path d="M12 17.6h.01"/>',
  pulse: '<path d="M3 12h4l2.5-7 5 14 2.5-7h4"/>',
  compare: '<circle cx="6" cy="12" r="3"/><circle cx="18" cy="6" r="3"/><circle cx="18" cy="18" r="3"/><path d="M8.7 10.7 15.3 7.3"/><path d="M8.7 13.3l6.6 3.4"/>',
  wrench: '<path d="M14.7 6.3a4 4 0 0 0-5.4 5.4L3 18l3 3 6.3-6.3a4 4 0 0 0 5.4-5.4L15 12l-3-3 2.7-2.7z"/>',
  report: '<path d="M6 2h9l5 5v15H6z"/><path d="M14 2v5h6"/><path d="M9 13h6M9 17h6"/>',
  settings: '<circle cx="12" cy="12" r="3"/><path d="M19 12a7 7 0 0 0-.1-1.2l2-1.6-2-3.4-2.4 1a7 7 0 0 0-2-1.2L14 3h-4l-.5 2.6a7 7 0 0 0-2 1.2l-2.4-1-2 3.4 2 1.6a7 7 0 0 0 0 2.4l-2 1.6 2 3.4 2.4-1a7 7 0 0 0 2 1.2L10 21h4l.5-2.6a7 7 0 0 0 2-1.2l2.4 1 2-3.4-2-1.6c.07-.4.1-.8.1-1.2z"/>',
  chevleft: '<path d="m15 6-6 6 6 6"/>',
  menu: '<path d="M4 7h16M4 12h16M4 17h16"/>',
  bell: '<path d="M6 9a6 6 0 0 1 12 0c0 7 3 7.5 3 7.5H3S6 16 6 9z"/><path d="M10 20a2 2 0 0 0 4 0"/>',
  x: '<path d="M6 6l12 12M18 6 6 18"/>',
  search: '<circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3"/>',
  database: '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
  download: '<path d="M12 4v11"/><path d="m7 11 5 5 5-5"/><path d="M4 21h16"/>',
  check: '<path d="m4 12.5 5 5L20 6.5"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  play: '<path d="M7 4.5v15l12-7.5-12-7.5z"/>',
  trash: '<path d="M4 7h16"/><path d="M9 7V4h6v3"/><path d="M6.5 7 8 20h8l1.5-13"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v5"/><path d="M12 8h.01"/>',
  shield: '<path d="M12 3l7 3.5V12c0 4.4-3 7.6-7 9-4-1.4-7-4.6-7-9V6.5L12 3z"/>',
  file: '<path d="M6 2h9l5 5v15H6z"/><path d="M14 2v5h6"/>',
};
const icon = (name, size = 18) =>
  `<svg viewBox="0 0 24 24" width="${size}" height="${size}" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[name] || ""}</svg>`;

/* hydrate static icon placeholders in shell */
$$("[data-ico]").forEach((el) => { el.innerHTML = icon(el.dataset.ico); });

/* ------------------------------------------------------------------ state */
const SEV_RANK = { CRITICAL: 0, WARNING: 1, INFO: 2 };
const DIM_LABEL = {
  completeness: "Completeness", uniqueness: "Uniqueness", validity: "Validity",
  consistency: "Consistency", plausibility: "Plausibility",
};
const PREF_KEYS = ["row_evidence_cap", "outlier_iqr_k", "future_date_days",
  "critical_missing_rate"];

function loadPrefs() {
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem("adqp.prefs") || "{}"); } catch { saved = {}; }
  return Object.assign({
    row_evidence_cap: 25, outlier_iqr_k: 3.0, future_date_days: 1,
    critical_missing_rate: 0.30, anomaly_enabled: true, ai_enabled: true,
  }, saved);
}
const prefs = loadPrefs();

const S = {
  route: "overview",
  scans: [],                  // metadata for dataset selector / recent scans
  current: null,
  reports: {},
  rem: {},                    // scan_id -> remediation response
  remStatus: {},              // scan_id -> { suggestion_id: accepted|rejected }
  anomalies: {},
  ai: {},
  compare: null,
  lastCompareKey: null,
  settings: null,
  findings: { q: "", sev: new Set(), dim: "all", col: "all", det: "all", sort: "severity" },
  pendingFinding: null,
  events: [],
};

/* ------------------------------------------------------------------ toasts */
function toast(message, kind = "info", ms = 4200) {
  const box = document.createElement("div");
  box.className = `toast ${kind}`;
  const ico = kind === "success" ? "check" : kind === "error" ? "alert" : "info";
  box.innerHTML = `<span class="t-ico">${icon(ico, 16)}</span><span>${esc(message)}</span>
    <button class="t-close" aria-label="Dismiss">${icon("x", 14)}</button>`;
  box.querySelector(".t-close").onclick = () => box.remove();
  $("#toasts").appendChild(box);
  setTimeout(() => box.remove(), ms);
}

function notify(message) {
  const t = new Date().toLocaleTimeString();
  S.events.push({ t, message });
  $("#notifDot").hidden = false;
  renderNotifs();
}
function renderNotifs() {
  const list = $("#notifList");
  if (!S.events.length) { list.innerHTML = '<p class="muted small">No actions yet in this session.</p>'; return; }
  list.innerHTML = S.events.slice(-12).reverse().map((e) =>
    `<div class="notif-item"><span class="sev-dot INFO"></span><span>${esc(e.message)}</span><small>${e.t}</small></div>`).join("");
}

/* -------------------------------------------------------------------- api */
async function api(path, opts = {}) {
  const res = await fetch(`${API}/${path}`, opts);
  let data = null;
  try { data = await res.json(); } catch { /* empty */ }
  if (!res.ok) {
    const msg = data && data.detail ? data.detail : `${res.status} ${res.statusText}`;
    throw new Error(Array.isArray(msg) ? JSON.stringify(msg) : msg);
  }
  return data;
}

/* ------------------------------------------------------------- shell bits */
function updateShell() {
  const meta = S.scans.find((s) => s.scan_id === S.current) || null;
  $("#wsCurrent").textContent = meta ? meta.filename : "No dataset";
  $("#wsMeta").textContent = meta
    ? `${fmt(meta.rows)} rows · ${meta.release_state}` : "Upload a file to begin";
  const dot = $(".ws-dot");
  dot.className = "ws-dot" + (meta ? (meta.release_state === "BLOCKED" ? " warn" : " ok") : "");

  const chip = $("#scanStatus");
  if (meta) {
    const short = meta.release_state === "PASS WITH WARNINGS" ? "WARNINGS" : meta.release_state;
    chip.hidden = false;
    chip.className = `scan-status ${meta.release_state === "PASS" ? "PASS" : meta.release_state === "BLOCKED" ? "BLOCKED" : "WARNING"}`;
    chip.innerHTML = `${icon("shield", 13)} ${esc(short)} · ${one(meta.overall_score)}/100`;
  } else chip.hidden = true;

  const crit = meta ? meta.severity_counts.CRITICAL : 0;
  const badge = $("#navCrit");
  badge.hidden = !crit;
  badge.textContent = crit;

  const sel = $("#datasetSelect");
  {
    // options are rebuilt on refresh, so always (re)select the current id
    if (S.current) sel.value = S.current;
    else sel.value = "";
  }
}

async function refreshScans() {
  S.scans = await api("scans");
  const remembered = localStorage.getItem("adqp.dataset");
  if (!S.current || !S.scans.some((s) => s.scan_id === S.current)) {
    S.current = S.scans.some((s) => s.scan_id === remembered) ? remembered
      : (S.scans[0] ? S.scans[0].scan_id : null);
  }
  $("#datasetSelect").innerHTML =
    `<option value="">No dataset selected</option>` +
    S.scans.map((s) =>
      `<option value="${s.scan_id}">${esc(s.filename)} — ${one(s.overall_score)}/100</option>`).join("");
  $("#datasetSelect").dataset.value = S.current || "";
  updateShell();
}

async function getReport(id, { force = false } = {}) {
  if (!id) return null;
  if (!force && S.reports[id]) return S.reports[id];
  const d = await api(`scans/${id}`);
  S.reports[id] = d;
  return d;
}

/* ------------------------------------------------------------- components */
const badge = (sev) => `<span class="badge ${esc(sev)}">${esc(sev)}</span>`;
const typeBadge = (t) => `<span class="badge type">${esc(t)}</span>`;
const stateBadge = (state) =>
  `<span class="badge ${state === "PASS" ? "PASS" : state === "BLOCKED" ? "BLOCKED" : "PASS-W"}">${esc(state)}</span>`;
const confBadge = (c) => c === "certain"
  ? `<span class="badge certain">certain</span>`
  : `<span class="badge review">manual review</span>`;

function skeletonRows(colgroup, n = 6) {
  return `<table><colgroup>${colgroup}</colgroup><tbody>` +
    `<tr><td class="sk-td"><div class="skeleton sk-row"></div></td></tr>`.repeat(n) +
    `</tbody></table>`;
}
const skCard = (h = 90) => `<div class="skeleton" style="height:${h}px"></div>`;

function ringSVG(score) {
  const r = 52, C = 2 * Math.PI * r, off = C * (1 - (score / 100));
  return `<svg viewBox="0 0 120 120" width="128" height="128" role="img" aria-label="Quality score ${score} of 100">
    <defs><linearGradient id="rg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#38bdf8"/><stop offset="1" stop-color="#a78bfa"/></linearGradient></defs>
    <circle cx="60" cy="60" r="${r}" stroke="rgba(148,163,184,.16)" stroke-width="10" fill="none"/>
    <circle cx="60" cy="60" r="${r}" stroke="url(#rg)" stroke-width="10" fill="none" stroke-linecap="round"
      stroke-dasharray="${C.toFixed(1)}" stroke-dashoffset="${off.toFixed(1)}" transform="rotate(-90 60 60)"/>
    <text x="60" y="58" text-anchor="middle" fill="#e6eaf3" font-size="24" font-weight="700">${one(score)}</text>
    <text x="60" y="76" text-anchor="middle" fill="#7d879e" font-size="10">of 100</text>
  </svg>`;
}

function histSVG(hist) {
  const W = 640, H = 190, M = { l: 34, r: 8, t: 14, b: 26 };
  const max = Math.max(...hist.map((b) => b.count), 1);
  const bw = (W - M.l - M.r) / hist.length;
  const bars = hist.map((b, i) => {
    const h = (H - M.t - M.b) * (b.count / max);
    const x = M.l + i * bw;
    const yTop = H - M.b - h;
    return b.count
      ? `<rect x="${(x + 1).toFixed(1)}" y="${yTop.toFixed(1)}" width="${(bw - 2).toFixed(1)}" height="${h.toFixed(1)}" rx="2.5" fill="#38bdf8" opacity="0.85"><title>Anomaly score ${b.lo.toFixed(3)}–${b.hi.toFixed(3)}: ${b.count} row(s)</title></rect>`
      : "";
  }).join("");
  const grid = [0.25, 0.5, 0.75, 1].map((f) => {
    const y = H - M.b - (H - M.t - M.b) * f;
    return `<line x1="${M.l}" y1="${y}" x2="${W - M.r}" y2="${y}" stroke="rgba(148,163,184,.1)"/>
      <text x="${M.l - 6}" y="${y + 3}" text-anchor="end" font-size="9" fill="#7d879e">${Math.round(max * f)}</text>`;
  }).join("");
  return `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto" role="img" aria-label="Anomaly score distribution">
    ${grid}${bars}
    <line x1="${M.l}" y1="${H - M.b}" x2="${W - M.r}" y2="${H - M.b}" stroke="rgba(148,163,184,.3)"/>
    <text x="${M.l}" y="${H - 8}" font-size="9.5" fill="#7d879e">score ${hist[0] ? hist[0].lo : 0}</text>
    <text x="${W / 2}" y="${H - 8}" text-anchor="middle" font-size="9.5" fill="#7d879e">anomaly score (higher = more unusual)</text>
    <text x="${W - M.r}" y="${H - 8}" text-anchor="end" font-size="9.5" fill="#7d879e">${hist.length ? hist[hist.length - 1].hi : 1}</text>
  </svg>`;
}

function missingDeltaBar(row) {
  const width = Math.min(100, Math.max(row.baseline_pct, row.current_pct) * 3);
  const color = row.delta_pp > 0 ? "var(--crit)" : "var(--ok)";
  return `
    <div class="bar"><span class="warn-fill" style="width:${Math.min(100, row.baseline_pct * 3)}%;background:rgba(148,163,184,.55)"></span></div>
    <div class="bar"><span style="width:${Math.min(100, row.current_pct * 3)}%;background:${color}"></span></div>`;
}

/* ------------------------------------------------------------------ drawer */
function openDrawer(title, html) {
  $("#drawerTitle").textContent = title;
  $("#drawerBody").innerHTML = html;
  $("#drawer").classList.add("open");
  $("#drawer").setAttribute("aria-hidden", "false");
  let ov = $("#drawerOverlay");
  if (!ov) {
    ov = document.createElement("div");
    ov.id = "drawerOverlay";
    ov.className = "drawer-overlay";
    ov.onclick = closeDrawer;
    document.body.appendChild(ov);
  }
  ov.classList.add("on");
  $("#drawerClose").focus();
}
function closeDrawer() {
  $("#drawer").classList.remove("open");
  $("#drawer").setAttribute("aria-hidden", "true");
  const ov = $("#drawerOverlay"); if (ov) ov.classList.remove("on");
}
document.addEventListener("keydown", (e) => { if (e.key === "Escape") { closeDrawer(); hidePops(); } });

function kv(rows) {
  return `<dl class="kv">${rows.map(([k, v]) =>
    `<div><dt>${esc(k)}</dt><dd>${v}</dd></div>`).join("")}</dl>`;
}

/* ------------------------------------------------------------------ router */
const PAGES = {
  "overview": { title: "Overview", sub: "Data-quality command center", render: pageOverview },
  "upload": { title: "Upload & Profile", sub: "Ingest a dataset and inspect every column", render: pageUpload },
  "findings": { title: "Quality Findings", sub: "Detected issues with row-level evidence", render: pageFindings },
  "anomalies": { title: "Anomaly Detection", sub: "Optional statistical module — separate from rule findings", render: pageAnomalies },
  "compare": { title: "Dataset Comparison", sub: "Current dataset vs saved baseline", render: pageCompare },
  "remediation": { title: "Remediation Review", sub: "Reviewable suggestions, nothing applied without approval", render: pageRemediation },
  "reports": { title: "Reports & Audit", sub: "Exports and audit trail", render: pageReports },
  "settings": { title: "Settings", sub: "Environment, privacy and scan preferences", render: pageSettings },
};

function route() {
  let hash = location.hash.replace(/^#\/?/, "");
  if (hash.includes("/")) hash = hash.split("/")[0];
  if (!PAGES[hash]) hash = "overview";
  S.route = hash;
  $$(".nav-item").forEach((a) => a.classList.toggle("active", a.dataset.route === hash));
  $("#pageTitle").textContent = PAGES[hash].title;
  $("#pageSub").textContent = PAGES[hash].sub;
  closeDrawer(); hidePops();
  const content = $("#content");
  content.innerHTML = `<div class="page">${skCard(120)}<div class="section-gap">${skCard(220)}</div></div>`;
  PAGES[hash].render(content).catch((err) => {
    content.innerHTML = emptyBlock("error", "Something went wrong", esc(err.message), "Try again", "location.reload()");
  });
  if (S.pendingFinding !== null && hash === "findings") {
    // drawer opened after table render
    setTimeout(() => { openFindingById(S.pendingFinding); S.pendingFinding = null; }, 60);
  }
  $("#content").focus({ preventScroll: false });
}
window.addEventListener("hashchange", route);

function goDetailFinder(findingId) {
  if (S.current && S.reports[S.current]) { openFindingById(findingId); return; }
  S.pendingFinding = findingId;
  if (S.route === "findings") route();
  else location.hash = "#/findings";
}

const emptyBlock = (icoName, title, text, btnLabel, onclick) => `
  <div class="empty card">
    <div class="empty-ico">${icon(icoName, 24)}</div>
    <b>${esc(title)}</b>
    <p class="small">${text}</p>
    ${btnLabel ? `<button class="btn primary" onclick="${onclick}">${icon("plus", 15)} ${esc(btnLabel)}</button>` : ""}
  </div>`;

const noDataset = () => emptyBlock("upload",
  "No dataset selected yet",
  "Upload a CSV or Excel file first — scanning is automatic and read-only. All analysis pages work on your most recent dataset.",
  "Go to Upload", "nav('upload')");

/* navigation helper used by inline handlers */
function nav(page) { location.hash = `#/${page}`; }
window.nav = nav;

function goDetailFinder(findingId) {
  if (S.current && S.reports[S.current]) { openFindingById(findingId); return; }
  S.pendingFinding = findingId;
  if (S.route === "findings") route();
  else location.hash = "#/findings";
}

/* ================================================================= OVERVIEW */
async function pageOverview(el) {
  await refreshScans();
  if (!S.current) {
    el.innerHTML = `<div class="page">${noDataset()}
      ${recentScansCard()}</div>`;
    return;
  }
  const report = await getReport(S.current, { force: true });
  const meta = S.scans.find((s) => s.scan_id === S.current) || report;
  const c = report.severity_counts;
  const dims = report.dimensions;
  const affectedRows = new Set(report.findings.filter((f) => f.row).map((f) => f.row)).size;
  const top = report.findings
    .filter((f) => f.severity === "CRITICAL")
    .sort((a, b) => (SEV_RANK[a.severity] - SEV_RANK[b.severity]) || (a.rule_id < b.rule_id ? -1 : 1))
    .slice(0, 6);

  el.innerHTML = `<div class="page">
    <div class="grid cols-3">
      <div class="card hero span-1">
        <div class="score-ring-wrap">
          ${ringSVG(meta.overall_score)}
          <div>
            <div class="flex" style="gap:8px">${stateBadge(meta.release_state || report.release_state)}</div>
            <div class="score-sub">${esc(meta.release_rationale || report.release_rationale || "")}</div>
            <div class="score-sub muted">Overall score = equal-weight mean of the five dimension scores. Computed by deterministic rules.</div>
          </div>
        </div>
      </div>
      <div class="grid cols-3 span-2" style="gap:14px">
        ${metricCard("crit", "Critical", c.CRITICAL, "click to review", "nav('findings')")}
        ${metricCard("warn", "Warning", c.WARNING, "should be reviewed", "nav('findings')")}
        ${metricCard("info", "Info", c.INFO, "for awareness", "nav('findings')")}
        ${metricCard("cyan", "Rows scanned", fmt(report.rows), "source unchanged")}
        ${metricCard("violet", "Columns", fmt(report.columns.length), "all auto-profiled")}
        ${metricCard("ok", "Rows flagged", fmt(affectedRows), `${one(report.rows ? affectedRows / report.rows * 100 : 0)}% of dataset`)}
      </div>
    </div>

    <div class="grid cols-2 section-gap">
      <div class="card">
        <h2>Five quality dimensions</h2>
        ${Object.entries(dims).map(([k, v]) => `
          <div class="dim-row"><small>${esc(DIM_LABEL[k] || k)}</small>
            <div class="bar"><span class="${v < 70 ? "warn-fill" : ""}" style="width:${v}%" title="${esc(k)}: ${v}%"></span></div>
            <span class="dim-val">${one(v)}%</span></div>`).join("")}
        <p class="notice cyan section-gap">${icon("info", 15)} <span>Completeness includes empty, blank and placeholder cells. Plausibility counts unusual values, not proven errors.</span></p>
      </div>
      <div class="card">
        <div class="card-head"><h2>Priority findings</h2>
          <button class="btn sm spacer" onclick="nav('findings')">View all ${fmt(report.finding_total)}</button></div>
        ${top.length ? `<div class="table-wrap">${findingsTable(top, { compact: true })}</div>`
          : `<p class="notice ok">${icon("check", 15)} <span>No CRITICAL findings. Review WARNING-level findings before release.</span></p>`}
      </div>
    </div>

    <div class="section-gap">${recentScansCard(true)}</div>
  </div>`;
}

function metricCard(color, label, value, hint, onclick) {
  return `<button class="card metric ${color}-c" ${onclick ? `onclick="${onclick}" style="cursor:pointer"` : "style=\"text-align:left\""}>
    <span class="metric-glow"></span>
    <div class="metric-label">${esc(label)}</div>
    <div class="metric-value">${value}</div>
    <div class="metric-hint">${esc(hint || "")}</div>
  </button>`;
}

function recentScansCard(withOpenLabel = false) {
  const meta = $("#wsMeta"); // ensure shell updated
  return `<div class="card">
    <h2>Recent scans <span class="muted small" style="text-transform:none;letter-spacing:0">— stored in memory; re-scan after a server restart</span></h2>
    ${S.scans.length ? `<div class="table-wrap"><table>
      <thead><tr><th>File</th><th class="num">Rows</th><th class="num">Cols</th><th class="num">Findings</th>
      <th class="num">Score</th><th>Status</th><th>Scanned at</th><th></th></tr></thead>
      <tbody>${S.scans.map((s) => `<tr class="clickable" data-scan="${s.scan_id}">
        <td><b>${esc(s.filename)}</b></td>
        <td class="num">${fmt(s.rows)}</td><td class="num">${fmt(s.column_count)}</td>
        <td class="num">${fmt(s.finding_total)}</td>
        <td class="num">${one(s.overall_score)}</td>
        <td>${stateBadge(s.release_state)}</td>
        <td class="muted small">${esc(s.generated_at)}</td>
        <td>${withOpenLabel ? `<button class="btn sm" data-scan-open="${s.scan_id}">Open</button>` : ""}</td>
      </tr>`).join("")}</tbody></table></div>`
      : `<p class="muted small">No scans yet.</p>`}
  </div>`;
}
document.addEventListener("click", (e) => {
  if (e.target.closest("a, button, select, input, label")) return;
  const frow = e.target.closest("tr[data-fid]");
  if (frow) {
    if (S.route === "overview") {
      S.current = frow.closest("[data-scan]")?.dataset.scan || S.current;
    }
    goDetailFinder(frow.dataset.fid);
    return;
  }
  const scanRow = e.target.closest("[data-scan-open], [data-scan]");
  if (scanRow && S.route === "overview" && !e.target.closest("tr[data-fid]")) {
    S.current = scanRow.dataset.scanOpen || scanRow.dataset.scan;
    localStorage.setItem("adqp.dataset", S.current);
    refreshScans().then(() => { nav("findings"); });
  }
});

/* ================================================================== UPLOAD */
async function pageUpload(el) {
  el.innerHTML = `<div class="page">
    <div class="card">
      <div class="dropzone" id="dropzone" tabindex="0" role="button" aria-label="Upload dataset">
        <div class="dz-icon">${icon("upload", 24)}</div>
        <b>Drop a CSV or Excel file here, or click to browse</b>
        <small class="muted">.csv · .xlsx — every column inspected automatically (types inferred from values, not names)</small>
        <small class="muted">missing values · blank strings · duplicates · inconsistent types · invalid formats · suspicious values</small>
        <input type="file" id="fileInput" accept=".csv,.xlsx,.xls,text/csv" hidden>
      </div>
      <div class="flex section-gap">
        <button class="btn" id="sampleBtn">${icon("file", 15)} Load sample dataset</button>
        <span class="muted small">Sample = bundled synthetic health data (demo data, clearly labelled). Scanning is read-only: <b>source files are never modified</b>.</span>
        <span id="scanBusy" class="muted small" hidden>Scanning…</span>
      </div>
    </div>
    <div id="uploadResult" class="section-gap"></div>
  </div>`;
  if (S.current) {
    try {
      const rep = await getReport(S.current);
      if (rep) renderProfile(rep);
    } catch { /* report will appear after the next scan */ }
  }

  const dz = $("#dropzone"), fi = $("#fileInput");
  dz.onclick = () => fi.click();
  dz.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fi.click(); } };
  fi.onchange = () => { if (fi.files[0]) scanFile(fi.files[0]); };
  dz.ondragover = (e) => { e.preventDefault(); dz.classList.add("over"); };
  dz.ondragleave = () => dz.classList.remove("over");
  dz.ondrop = (e) => {
    e.preventDefault(); dz.classList.remove("over");
    if (e.dataTransfer.files.length) scanFile(e.dataTransfer.files[0]);
  };
  $("#sampleBtn").onclick = async () => {
    try {
      $("#sampleBtn").disabled = true;
      const res = await fetch(`${API}/sample-data`);
      const blob = await res.blob();
      await scanFile(new File([blob], "sample_universal_health.csv", { type: "text/csv" }), true);
    } catch (err) { toast("Could not load sample: " + err.message, "error"); }
    finally { $("#sampleBtn").disabled = false; }
  };
}

async function scanFile(file, isSample = false) {
  const busy = $("#scanBusy"); busy.hidden = false; busy.textContent = "Scanning " + file.name + "…";
  try {
    const fd = new FormData();
    fd.append("file", file);
    const cfg = {};
    for (const k of PREF_KEYS) cfg[k] = prefs[k];
    fd.append("config", JSON.stringify(cfg));
    const data = await api("scan", { method: "POST", body: fd });
    S.current = data.scan_id;
    S.reports[data.scan_id] = data;
    localStorage.setItem("adqp.dataset", data.scan_id);
    await refreshScans();
    updateShell();
    toast(`Scan complete: ${fmt(data.finding_total)} finding(s) in ${esc(data.filename)}`, "success");
    notify(`Scanned ${data.filename} — ${data.finding_total} finding(s), status ${data.release_state}`);
    if (isSample) toast("Sample dataset loaded — synthetic demo data, clearly labelled", "info");
    renderProfile(data, { scrolled: true });
  } catch (err) {
    toast("Scan failed: " + err.message, "error");
  } finally { busy.hidden = true; }
}

function renderProfile(report, { scrolled = false } = {}) {
  const box = $("#uploadResult"); if (!box) return;
  const c = report.severity_counts;
  const rules = new Set(report.findings.map((f) => f.rule_id));
  box.innerHTML = `
    <div class="grid cols-4">
      ${metricCard("ok", "Rows", fmt(report.rows), esc(report.filename))}
      ${metricCard("cyan", "Columns", fmt(report.columns.length), "auto-profiled")}
      ${metricCard("crit", "Critical", fmt(c.CRITICAL), "findings")}
      ${metricCard("warn", "Warning", fmt(c.WARNING), `${fmt(c.INFO)} info also`)}
    </div>
    <div class="grid cols-2 section-gap">
      <div class="card">
        <h2>Release status (deterministic)</h2>
        <div class="flex" style="gap:10px">${stateBadge(report.release_state || "PASS")}</div>
        <p class="small muted" style="margin:.5rem 0 0">${esc(report.release_rationale || "")}</p>
        <p class="small muted" style="margin:.75rem 0 0">Detectors that ran: ${esc([...rules].join(", ") || "none")}</p>
        ${report.notes && report.notes.length ? `<p class="small muted">Notes: ${esc(report.notes.join("; "))}</p>` : ""}
      </div>
      <div class="card">
        <h2>File preview — first rows</h2>
        ${report.preview && report.preview.rows.length ? `
        <div class="table-wrap"><table class="mono" style="font-size:11.5px">
          <thead><tr>${report.preview.columns.map((h) => `<th>${esc(h)}</th>`).join("")}</tr></thead>
          <tbody>${report.preview.rows.map((r) => `<tr>${r.map((v) => `<td>${esc(v)}</td>`).join("")}</tr>`).join("")}</tbody>
        </table></div>
        <p class="muted small" style="margin-top:8px">Read-only preview. The uploaded file itself is untouched.</p>`
          : `<p class="muted small">Preview unavailable for this file type.</p>`}
      </div>
    </div>
    <div class="card section-gap">
      <h2>Column profiles — every column inspected automatically</h2>
      <div class="table-wrap scalar-scroll"><table>
        <thead><tr><th>Column</th><th>Type</th><th class="num">Non-null</th><th class="num">Missing</th>
        <th class="num">Blank</th><th class="num">Placeholder</th><th class="num">Unique</th><th class="num">Unique %</th>
        <th>Date format</th><th>Top values</th></tr></thead>
        <tbody>${report.columns.map((p) => `<tr>
          <td><b>${esc(p.name)}</b>${p.id_like ? ' <span class="badge violet">id-like</span>' : ""}</td>
          <td>${typeBadge(p.inferred_type)}${p.mixed_types && Object.keys(p.mixed_types).length > 1 ? ' <span class="badge WARNING">mixed</span>' : ""}</td>
          <td class="num">${fmt(p.non_null)}</td>
          <td class="num" ${p.missing ? 'style="color:var(--crit);font-weight:600"' : ""}>${fmt(p.missing)}</td>
          <td class="num" ${p.blank ? 'style="color:var(--warn);font-weight:600"' : ""}>${fmt(p.blank)}</td>
          <td class="num" ${p.placeholder ? 'style="color:var(--warn);font-weight:600"' : ""}>${fmt(p.placeholder)}</td>
          <td class="num">${fmt(p.unique)}</td>
          <td class="num">${one(p.unique_pct)}%</td>
          <td class="mono small">${esc(p.date_format || "—")}</td>
          <td class="mono small">${esc((p.top_values || []).map(([v]) => v).slice(0, 3).join(" · "))}</td>
        </tr>`).join("")}</tbody>
      </table></div>
      <div class="flex section-gap">
        <button class="btn primary" onclick="nav('findings')">${icon("alert", 15)} Review findings</button>
        <button class="btn" onclick="nav('anomalies')">${icon("pulse", 15)} Anomaly detection</button>
        <button class="btn" onclick="nav('reports')">${icon("report", 15)} Reports</button>
      </div>
    </div>`;
  if (scrolled) box.scrollIntoView({ behavior: "smooth", block: "start" });
}

/* ================================================================ FINDINGS */
async function pageFindings(el) {
  await refreshScans();
  if (!S.current) { el.innerHTML = noDataset(); return; }
  const report = await getReport(S.current, { force: true });
  const cols = report.columns.map((p) => p.name);
  const deselectString = '<option value="all">all</option>';
  el.innerHTML = `<div class="page">
    <div class="card">
      <div class="toolbar">
        <div class="search-box">${icon("search", 15)}<input id="fSearch" placeholder="Search message, value, rule, column…" value="${esc(S.findings.q)}"></div>
      </div>
      <div class="toolbar" style="margin-top:0">
        <div class="chips" role="group" aria-label="Severity filter">${["CRITICAL", "WARNING", "INFO"].map((s) =>
    `<button class="chip ${s} ${S.findings.sev.size && !S.findings.sev.has(s) ? "" : "on"}" data-sev="${s}">
            <span class="sev-dot ${s}"></span>${s}</button>`).join("")}</div>
        <div class="filters">
          <select id="fDim" class="input-inline" aria-label="Dimension">${deselectString}${Object.keys(report.dimensions).map((d) =>
    `<option value="${d}" ${S.findings.dim === d ? "selected" : ""}>${esc(DIM_LABEL[d])}</option>`).join("")}</select>
          <select id="fCol" class="input-inline" aria-label="Column">${deselectString}${cols.map((c) =>
    `<option ${S.findings.col === c ? "selected" : ""}>${esc(c)}</option>`).join("")}</select>
          <select id="fDet" class="input-inline" aria-label="Detector">${deselectString}${report.detectors.map((d) =>
    `<option ${S.findings.det === d ? "selected" : ""}>${esc(d)}</option>`).join("")}</select>
          <select id="fSort" class="input-inline" aria-label="Sort">
            <option value="severity" ${S.findings.sort === "severity" ? "selected" : ""}>Sort: severity</option>
            <option value="column" ${S.findings.sort === "column" ? "selected" : ""}>Sort: column</option>
            <option value="row" ${S.findings.sort === "row" ? "selected" : ""}>Sort: row</option>
            <option value="rule" ${S.findings.sort === "rule" ? "selected" : ""}>Sort: rule</option>
          </select>
        </div>
      </div>
      <div id="findingsBox"></div>
    </div>
    <div id="aiBox" class="section-gap">${aiCardHTML()}</div>
  </div>`;
  wireFindings(report);
  wireAI(report);
  if (S.pendingFinding !== null) { openFindingById(S.pendingFinding); S.pendingFinding = null; }
}

function wireFindings(report) {
  const box = $("#findingsBox");
  const apply = () => {
    S.findings.q = $("#fSearch").value.toLowerCase();
    S.findings.dim = $("#fDim").value; S.findings.col = $("#fCol").value;
    S.findings.det = $("#fDet").value; S.findings.sort = $("#fSort").value;
    renderFindingsTable(report, box);
  };
  $("#fSearch").oninput = apply;
  ["fDim", "fCol", "fDet", "fSort"].forEach((id) => { $("#" + id).onchange = apply; });
  $$(".chip[data-sev]").forEach((chip) => {
    chip.onclick = () => {
      const s = chip.dataset.sev;
      if (S.findings.sev.has(s)) S.findings.sev.delete(s); else S.findings.sev.add(s);
      if (S.findings.sev.size === 3) { S.findings.sev.clear(); }
      $$(".chip[data-sev]").forEach((c) =>
        c.classList.toggle("on", S.findings.sev.size === 0 || S.findings.sev.has(c.dataset.sev)));
      renderFindingsTable(report, box);
    };
  });
  renderFindingsTable(report, box);
}

function renderFindingsTable(report, box) {
  const f = S.findings;
  let rows = report.findings;
  if (f.q) rows = rows.filter((r) =>
    [r.message, r.value, r.rule_id, r.column, r.evidence].some((v) => String(v || "").toLowerCase().includes(f.q)));
  if (f.sev.size && f.sev.size < 3) rows = rows.filter((r) => f.sev.has(r.severity));
  if (f.dim !== "all") rows = rows.filter((r) => r.dimension === f.dim);
  if (f.col !== "all") rows = rows.filter((r) => r.column === f.col);
  if (f.det !== "all") rows = rows.filter((r) => r.rule_id === f.det);
  const sorters = {
    severity: (a, b) => SEV_RANK[a.severity] - SEV_RANK[b.severity] || String(a.column).localeCompare(b.column) || (a.row || 0) - (b.row || 0),
    column: (a, b) => String(a.column).localeCompare(b.column) || (a.row || 0) - (b.row || 0),
    row: (a, b) => (a.row || 9e9) - (b.row || 9e9),
    rule: (a, b) => String(a.rule_id).localeCompare(b.rule_id) || (a.row || 0) - (b.row || 0),
  };
  rows = [...rows].sort(sorters[f.sort]);

  const cap = 300;
  const shown = rows.slice(0, cap);
  box.innerHTML = `
    <p class="small muted" style="margin:.4rem 0 .8rem">${fmt(rows.length)} matching finding(s)${rows.length > cap ? ` — showing first ${cap}; download the CSV/JSON report for the full list` : ""} · click a row for evidence</p>
    <div class="table-wrap scalar-scroll">${findingsTable(shown)}</div>`;
}

function findingsTable(list, { compact = false } = {}) {
  return `<table>
    <thead><tr><th>Severity</th><th>Item</th><th>Column</th><th class="num">Row</th>
    <th>Message</th>${compact ? "" : "<th>Evidence</th>"}${compact ? "" : "<th>Suggested action</th>"}</tr></thead>
    <tbody>${list.map((f) => `<tr class="finding-row clickable bad-${f.severity === "CRITICAL" ? "row" : ""}" data-fid="${f.finding_id}">
      <td>${badge(f.severity)}</td>
      <td class="mono">${esc(f.rule_id)}</td>
      <td>${esc(f.column)}</td>
      <td class="num">${f.row ?? "—"}</td>
      <td>${esc(f.message)}</td>
      ${compact ? "" : `<td class="mono small">${esc(truncate(f.evidence, 60))}</td><td class="small muted">${esc(truncate(f.suggested_action, 50))}</td>`}
    </tr>`).join("")}</tbody></table>`;
}
const truncate = (s, n) => (s == null ? "" : String(s).length > n ? String(s).slice(0, n) + "…" : String(s));

function openFindingById(fid, preloaded) {
  const report = preloaded || S.reports[S.current];
  const f = report && report.findings.find((x) => x.finding_id === fid);
  if (!f) { toast("Finding not found in this dataset", "error"); return; }
  openDrawer(`Finding ${f.finding_id}`, `
    ${kv([
    ["Severity", badge(f.severity)],
    ["Rule", `<span class="mono">${esc(f.rule_id)}</span>`],
    ["Column", esc(f.column)],
    ["Row", f.row != null ? esc(String(f.row)) : "column-level"],
    ["Dimension", esc(DIM_LABEL[f.dimension] || f.dimension)],
    ["Value", `<span class="mono">${esc(f.value ?? "—")}</span>`],
    ["Message", esc(f.message)],
    ["Evidence", esc(f.evidence)],
    ["Suggested action", esc(f.suggested_action)],
  ])}
    <div class="notice">${icon("shield", 15)} <span>Evidence comes from the deterministic rule engine only. Remediation review can be started for this dataset.</span></div>
    <div class="flex section-gap">
      <button class="btn" onclick="nav('remediation')">${icon("wrench", 14)} Remediation review</button>
    </div>`);
}

/* ------------------------------------------------------------------ AI card */
function aiCardHTML() {
  if (!prefs.ai_enabled) {
    return `<div class="card"><h2>AI analyst — disabled</h2>
      <p class="muted small">AI analysis is turned off in Settings. Deterministic results are unaffected.</p></div>`;
  }
  return `<div class="card">
    <h2>AI analyst (Google Gemini) — advisory only</h2>
    <p class="muted small">Runs after the deterministic checks. It may explain findings and suggest reviewable steps, but it can never change scores or the release status. Suggestions are never auto-applied. Only structured findings and limited samples are sent — never the full dataset. Works fine without an API key.</p>
    <div class="flex"><button class="btn" id="aiRun">${icon("play", 14)} Run AI analysis</button><span id="aiStatus" class="muted small"></span></div>
    <div id="aiResult" class="section-gap"></div>
  </div>`;
}

function wireAI(report) {
  const btn = $("#aiRun");
  if (!btn) return;
  btn.onclick = async () => {
    btn.disabled = true;
    $("#aiStatus").textContent = "Running AI analysis…";
    try {
      const res = await api(`scan/${report.scan_id}/ai-analysis`, { method: "POST" });
      S.ai[report.scan_id] = res;
      renderAI(res);
      toast("AI analysis finished (" + res.status + ")", res.status === "ok" ? "success" : "info");
    } catch (err) {
      $("#aiStatus").textContent = "";
      toast("AI analysis failed: " + err.message, "error");
    } finally { btn.disabled = false; }
  };
  if (S.ai[report.scan_id]) renderAI(S.ai[report.scan_id]);
}

function renderAI(res) {
  const box = $("#aiResult"); if (!box) return;
  const os = res.overall_summary;
  let html = `<p class="small muted">Status: <b>${esc(res.status)}</b> ${res.model_used ? "· model " + esc(res.model_used) + " " : ""}· ${esc(res.message || "")}</p>`;
  if (os) {
    html += `<div class="ai-block-like">
      ${kv([
      ["Release status", `${stateBadge(os.deterministic_release_status)} <span class="small muted">(computed by rules — AI cannot change it)</span>`],
      ["Rationale", esc(os.deterministic_rationale)],
      ["AI recommendation", esc(os.release_recommendation_text || "—")],
      ["AI explanation", esc(os.ai_explanation_of_status || "—")],
      ["Confidence", os.confidence_overall ?? "—"],
    ])}
      ${(os.most_important_risks || []).length ? `<b class="small">Most important risks</b><ul class="small">${os.most_important_risks.map((r) => `<li>${esc(r)}</li>`).join("")}</ul>` : ""}
      ${(os.limitations || []).length ? `<p class="small muted">Limitations: ${esc(os.limitations.join("; "))}</p>` : ""}
      <p class="small muted">All of the above is AI-generated commentary. The deterministic release status above is rule-based.</p>
    </div>`;
  }
  if ((res.suggestions || []).length) {
    html += `<h3 class="small" style="margin:12px 0 6px">AI suggestions — review required, NOT auto-applied</h3>
      <div class="table-wrap"><table><thead><tr><th>Issue</th><th>Proposed action</th><th>Risk</th><th>Approval</th></tr></thead>
      <tbody>${res.suggestions.map((s) => `<tr><td>${esc(s.issue)}</td><td>${esc(s.proposed_action)}</td><td>${esc(s.risk)}</td><td>${badge("review")}</td></tr>`).join("")}</tbody></table></div>`;
  }
  if ((res.issue_explanations || []).length) {
    html += `<h3 class="small" style="margin:12px 0 6px">AI finding explanations</h3>
      ${res.issue_explanations.map((item) => `<div class="ai-block-like">
        <b class="small">${esc(item.issue_name || item.rule_id || "Finding")}</b>
        <p class="small">${esc(item.plain_english_explanation || item.what_happened || "Review the deterministic finding evidence.")}</p>
        ${(item.repair_steps || []).length ? `<p class="small"><b>Review steps:</b> ${esc(item.repair_steps.join("; "))}</p>` : ""}
      </div>`).join("")}`;
  }
  box.innerHTML = html;
}

/* ================================================================ ANOMALIES */
async function pageAnomalies(el) {
  await refreshScans();
  if (!S.current) { el.innerHTML = noDataset(); return; }
  const report = await getReport(S.current);
  el.innerHTML = `<div class="page"><div id="anomBody"></div></div>`;
  const body = $("#anomBody");
  // No GET probe: a first visit has no stored result yet, and a 404 probe
  // would surface as console noise. The user's own run populates the cache.
  renderAnomaly(S.anomalies[S.current] || null, report, body);
}

function renderAnomaly(anomaly, report, body) {
  if (prefs.anomaly_enabled === false) {
    body.innerHTML = `<div class="card"><h2>Anomaly detection — disabled</h2>
      <p class="muted small">Disabled in Settings. Deterministic rule findings are unaffected.</p>
      <button class="btn" onclick="nav('settings')">Open settings</button></div>`;
    return;
  }
  const runForm = () => `
    <div class="card">
      <h2>Statistical anomaly detection (optional)</h2>
      <p class="muted small">Uses an Isolation Forest on numeric features to surface <b>unusual rows</b>. Anomalies are review candidates — <b>not errors automatically</b> — and are kept separate from the deterministic rule findings. The release status and quality scores are never affected by this module.</p>
      <div class="flex">
        <label class="muted small">Contamination
          <select id="anCont" class="input-inline" style="margin-left:6px">
            <option value="auto">auto</option><option value="0.05">5%</option>
            <option value="0.10">10%</option><option value="0.20">20%</option>
          </select></label>
        <button class="btn primary" id="anRun">${icon("play", 14)} Run anomaly detection</button>
        <span id="anBusy" class="muted small"></span>
      </div>
      <p class="notice section-gap">${icon("info", 15)} <span>Requirements: at least 50 usable rows and at least one numeric column (≥80% parseable values). Sorted with a fixed random seed (42) for reproducibility.</span></p>
    </div>`;
  if (!anomaly) {
    body.innerHTML = runForm();
    $("#anRun").onclick = () => runAnomaly($("#anCont").value);
    return;
  }
  if (anomaly.status !== "ok") {
    body.innerHTML = `<div class="card">
      <h2>Statistical anomaly detection</h2>
      <div class="notice warn">${icon("alert", 15)} <span><b>${esc(anomaly.status)}:</b> ${esc(anomaly.reason || "not run")}</span></div>
      <p class="muted small" style="margin-top:10px">The deterministic findings for ${esc(report.filename)} are unaffected and remain the source of truth.</p>
      <button class="btn" id="anBack">${icon("chevleft", 14)} Back to run form</button>
    </div>`;
    $("#anBack").onclick = () => { delete S.anomalies[S.current]; pageAnomalies($("#content")); };
    return;
  }
  const st = anomaly.score_stats || {};
  body.innerHTML = `
    <div class="grid cols-4">
      ${metricCard("violet", "Rows evaluated", fmt(anomaly.rows_evaluated), esc(anomaly.model))}
      ${metricCard("crit", "Flagged rows", fmt(anomaly.flagged_count), `${one(anomaly.flagged_pct)}% — review candidates, not errors`)}
      ${metricCard("cyan", "Score range", `${st.min}…${st.max}`, "higher = more unusual")}
      ${metricCard("info", "Features used", fmt(anomaly.feature_count), esc(anomaly.features_used.join(", ") || "—"))}
    </div>
    <div class="card section-gap">
      <h2>Anomaly score distribution</h2>
      ${histSVG(anomaly.histogram)}
      <p class="muted small" style="margin-top:6px">Model: Isolation Forest · random seed ${esc(anomaly.random_seed)} · contamination ${esc(anomaly.contamination)} · ${esc(anomaly.duration_ms)} ms. Scores are relative to this dataset only.</p>
    </div>
    <div class="card section-gap">
      <h2>Flagged rows (${fmt(anomaly.flagged_count)}) — most anomalous first</h2>
      ${anomaly.flagged_rows.length ? `
      <div class="table-wrap scalar-scroll"><table>
        <thead><tr><th class="num">Row</th><th class="num">Score</th><th>Why it looks unusual</th><th>Strongest deviations</th></tr></thead>
        <tbody>${anomaly.flagged_rows.map((r) => `<tr class="clickable" data-arow="${r.row}">
          <td class="num"><b>${fmt(r.row)}</b></td><td class="num">${r.score.toFixed(3)}</td>
          <td>${esc(r.summary)}</td>
          <td class="mono small">${esc(r.top_features.map((t) => `${t.feature} (z=${t.z > 0 ? "+" : ""}${t.z})`).join(", ")) || "—"}</td>
        </tr>`).join("")}</tbody></table></div>`
      : `<p class="muted small">No rows flagged with the current setting.</p>`}
      <p class="muted small" style="margin-top:8px">z = (row value − column median) / column stddev after imputation. Click a row for details.</p>
    </div>
    <div class="card section-gap">
      <h2>Model limitations — read before acting</h2>
      ${anomaly.limitations.map((l) => `<p class="notice warn" style="margin-bottom:8px">${icon("alert", 14)} <span>${esc(l)}</span></p>`).join("")}
      <div class="flex"><button class="btn" id="anRerun">${icon("play", 14)} Re-run with new settings</button></div>
    </div>`;
  $("#anRerun").onclick = () => { delete S.anomalies[S.current]; pageAnomalies($("#content")); };
  $$("[data-arow]", body).forEach((tr) => {
    tr.onclick = () => {
      const r = anomaly.flagged_rows.find((x) => String(x.row) === tr.dataset.arow);
      openDrawer(`Row ${r.row} — anomaly detail`, `
        ${kv([
        ["Anomaly score", r.score.toFixed(4)],
        ["Dataset row", fmt(r.row)],
        ["Summary", esc(r.summary)],
      ])}
        <b class="small">Feature deviations</b>
        <div class="table-wrap"><table><thead><tr><th>Feature</th><th class="num">z-score</th></tr></thead>
        <tbody>${r.top_features.map((t) => `<tr><td class="mono">${esc(t.feature)}</td><td class="num">${t.z > 0 ? "+" : ""}${t.z}</td></tr>`).join("")}</tbody></table></div>
        <p class="notice warn section-gap">${icon("alert", 15)} <span>Statistical anomaly — not automatically an error. Verify the row against the source before changing anything.</span></p>`);
    };
  });
}

async function runAnomaly(contamination) {
  const busy = $("#anBusy");
  busy.textContent = "Fitting model…";
  try {
    const form = new FormData();
    const res = await fetch(`${API}/scan/${S.current}/anomalies?contamination=${encodeURIComponent(contamination)}`,
      { method: "POST" });
    if (!res.ok) throw new Error((await res.json()).detail || res.status);
    const anomaly = await res.json();
    S.anomalies[S.current] = anomaly;
    toast(`Anomaly detection finished: ${anomaly.flagged_count} row(s) flagged`, anomaly.status === "ok" ? "success" : "info");
    notify(`Anomaly detection on ${anomaly.filename} — ${anomaly.flagged_count} flagged (statistical, non-blocking)`);
    renderAnomaly(anomaly, S.reports[S.current], $("#anomBody"));
  } catch (err) {
    toast("Anomaly detection failed: " + err.message, "error");
  } finally { busy.textContent = ""; }
}

/* ================================================================== COMPARE */
async function pageCompare(el) {
  await refreshScans();
  if (!S.current) { el.innerHTML = noDataset(); return; }
  const baselines = await api("baselines");
  el.innerHTML = `<div class="page"><div id="cmpShell"></div></div>`;
  const shell = $("#cmpShell");
  if (!baselines.length) {
    shell.innerHTML = `
      ${emptyBlock("compare", "No baseline saved yet",
        `Save a dataset as a baseline first, then upload a newer copy of the same data to see schema, missing-rate and distribution changes. The current dataset is <b>${esc((S.scans.find((s) => s.scan_id === S.current) || {}).filename || "")}</b>.`,
        "Save current dataset as baseline", "saveBaselineFromScan()")}
      <div id="cmpResult"></div>`;
    return;
  }
  shell.innerHTML = `
    <div class="card">
      <div class="toolbar" style="margin-bottom:0">
        <label class="muted small">Baseline
          <select id="baseSel" class="input-inline" style="margin-left:6px;min-width:220px">
            ${baselines.map((b) => `<option value="${b.baseline_id}">${esc(b.filename)} (${fmt(b.rows)} rows, ${esc(b.generated_at)})</option>`).join("")}
          </select></label>
        <button class="btn primary" id="cmpRun">${icon("compare", 15)} Run comparison</button>
        <button class="btn" id="baseSave">${icon("plus", 15)} Save current as baseline</button>
      </div>
      <p class="muted small" style="margin:.6rem 0 0">Current dataset: <b>${esc((S.scans.find((s) => s.scan_id === S.current) || {}).filename || "")}</b>. Baselines live in memory for this session.</p>
    </div>
    <div id="cmpResult" class="section-gap"></div>`;
  $("#cmpRun").onclick = runComparison;
  $("#baseSave").onclick = saveBaselineFromScan;
  if (S.lastCompareKey) {
    const [b, s] = S.lastCompareKey.split("|");
    if (baselines.some((x) => x.baseline_id === b)) { $("#baseSel").value = b; runComparison(); }
    else S.lastCompareKey = null;
  }
}

async function saveBaselineFromScan() {
  try {
    const res = await api(`baselines/from-scan?scan_id=${encodeURIComponent(S.current)}`, { method: "POST", body: "{}" });
    toast(`Baseline saved from ${res.filename}`, "success");
    notify(`Baseline ${res.baseline_id} saved (${res.filename})`);
    route();
  } catch (err) { toast("Could not save baseline: " + err.message, "error"); }
}
window.saveBaselineFromScan = saveBaselineFromScan;

async function runComparison() {
  const baselineId = $("#baseSel").value;
  const out = $("#cmpResult");
  out.innerHTML = skCard(120);
  try {
    const cmp = await api(`compare?scan_id=${encodeURIComponent(S.current)}&baseline_id=${encodeURIComponent(baselineId)}`);
    S.compare = cmp; S.lastCompareKey = `${baselineId}|${S.current}`;
    renderCompare(cmp, out);
    toast("Comparison complete — review signals below", "success");
    notify(`Compared ${cmp.current.filename} vs ${cmp.baseline.filename} — ${cmp.signals.length} signal(s)`);
  } catch (err) {
    out.innerHTML = emptyBlock("error", "Comparison failed", esc(err.message));
  }
}

function renderCompare(cmp, out) {
  const deltaBadge = (v, invert = false) => {
    const cls = (invert ? -v : v) > 0.01 ? "CRITICAL" : (invert ? -v : v) < -0.01 ? "certain" : "neutral";
    return `<span class="badge ${cls}">${v > 0 ? "+" : ""}${one(v)}</span>`;
  };
  out.innerHTML = `
    <div class="grid cols-4">
      ${metricCard("cyan", "Columns added", fmt(cmp.columns_added.length), cmp.columns_added.join(", ") || "—")}
      ${metricCard("crit", "Columns removed", fmt(cmp.columns_removed.length), cmp.columns_removed.join(", ") || "—")}
      ${metricCard("warn", "Type changes", fmt(cmp.type_changes.length), "inferred types differ")}
      ${metricCard("violet", "Review signals", fmt(cmp.signals.length), "investigate, not defects")}
    </div>
    ${cmp.signals.length ? `
    <div class="card section-gap">
      <h2>Review signals (${cmp.signals.length})</h2>
      <div class="table-wrap"><table><thead><tr><th>Severity</th><th>Kind</th><th>Column</th><th>Message</th></tr></thead>
      <tbody>${cmp.signals.map((s) => `<tr><td>${badge(s.severity)}</td><td class="mono small">${esc(s.kind)}</td><td>${esc(s.column)}</td><td>${esc(s.message)}</td></tr>`).join("")}</tbody></table></div>
    </div>` : `<div class="card section-gap"><div class="notice ok">${icon("check", 15)} <span>No drift signals above the thresholds.</span></div></div>`}
    <div class="grid cols-2 section-gap">
      <div class="card">
        <h2>Schema</h2>
        <div class="table-wrap"><table><thead><tr><th>Column</th><th>Baseline type</th><th>Current type</th></tr></thead><tbody>
          ${cmp.columns_added.map((c) => `<tr><td><b>${esc(c)}</b></td><td class="muted">—</td><td>${typeBadge("new")}</td></tr>`).join("")}
          ${cmp.columns_removed.map((c) => `<tr><td><b>${esc(c)}</b></td><td>${typeBadge("removed")}</td><td class="muted">—</td></tr>`).join("")}
          ${cmp.type_changes.map((t) => `<tr><td><b>${esc(t.column)}</b></td><td>${typeBadge(t.baseline)}</td><td>${typeBadge(t.current)} ${badge("WARNING")}</td></tr>`).join("")}
          ${(cmp.columns_added.length || cmp.columns_removed.length || cmp.type_changes.length) ? "" : `<tr><td colspan="3" class="muted small">No schema changes.</td></tr>`}
        </tbody></table></div>
      </div>
      <div class="card">
        <h2>Missing-rate changes (≥1 pp shown)</h2>
        ${cmp.missing_rates.length ? `<div class="table-wrap"><table><thead><tr><th>Column</th><th>Baseline</th><th>Current</th><th>Δ</th><th>Bars</th></tr></thead>
          <tbody>${cmp.missing_rates.map((m) => `<tr><td><b>${esc(m.column)}</b></td><td class="num">${one(m.baseline_pct)}%</td>
          <td class="num">${one(m.current_pct)}%</td><td>${deltaBadge(m.delta_pp, true)}</td><td>${missingDeltaBar(m)}</td></tr>`).join("")}</tbody></table></div>`
      : `<p class="muted small">No missing-rate change above 0.01 pp.</p>`}
      </div>
    </div>
    <div class="card section-gap">
      <h2>Numeric distribution shifts (sorted by KS statistic)</h2>
      ${cmp.numeric_distribution.length ? `<div class="table-wrap"><table>
        <thead><tr><th>Column</th><th>Mean b→c</th><th>Median b→c</th><th>Min b→c</th><th>Max b→c</th><th>KS statistic</th></tr></thead>
        <tbody>${cmp.numeric_distribution.map((n) => `<tr><td><b>${esc(n.column)}</b></td>
          <td class="num">${n.baseline.mean} → ${n.current.mean}</td>
          <td class="num">${n.baseline.median} → ${n.current.median}</td>
          <td class="num">${n.baseline.min} → ${n.current.min}</td>
          <td class="num">${n.baseline.max} → ${n.current.max}</td>
          <td>${n.ks_statistic >= cmp.thresholds.ks_alert ? badge("WARNING") : badge("neutral")} <span class="mono">${n.ks_statistic.toFixed(2)}</span></td></tr>`).join("")}</tbody></table></div>`
      : `<p class="muted small">No shared numeric columns (or not enough valid values).</p>`}
      <p class="muted small" style="margin-top:6px">KS = empirical distribution distance (0 = identical, 1 = disjoint). Threshold ≥ ${cmp.thresholds.ks_alert} is marked for review. Descriptive statistic only — not proof of a defect.</p>
    </div>
    ${(cmp.category_shifts.length || cmp.date_ranges.length) ? `
    <div class="grid cols-2 section-gap">
      <div class="card"><h2>Category shifts</h2>
        ${cmp.category_shifts.map((c) => `<div style="margin-bottom:12px">
          <b>${esc(c.column)}</b>${c.new_labels.length ? ` <span class="badge WARNING">new: ${esc(c.new_labels.join(", "))}</span>` : ""}
          <table class="mono small"><tbody>${c.labels.map((l) =>
    `<tr><td>${esc(l)}</td><td class="num">${c.baseline[l] ?? "—"}% → ${c.current[l] ?? "—"}%</td></tr>`).join("")}</tbody></table>
        </div>`).join("") || '<p class="muted small">—</p>'}
      </div>
      <div class="card"><h2>Date-range changes</h2>
        ${cmp.date_ranges.length ? `<div class="table-wrap"><table><thead><tr><th>Column</th><th>Baseline range</th><th>Current range</th></tr></thead>
          <tbody>${cmp.date_ranges.map((d) => `<tr><td><b>${esc(d.column)}</b></td>
          <td class="mono small">${esc(d.baseline.min)}…${esc(d.baseline.max)}</td>
          <td class="mono small">${esc(d.current.min)}…${esc(d.current.max)}
          ${d.min_changed || d.max_changed ? badge("WARNING") : badge("certain")}</td></tr>`).join("")}</tbody></table></div>`
      : `<p class="muted small">No shared date columns.</p>`}
      </div>
    </div>` : ""}
    <p class="notice small section-gap">${icon("info", 15)} <span>Baseline: ${esc(cmp.baseline.filename)} (${esc(cmp.baseline.generated_at)}, ${fmt(cmp.baseline.rows)} rows) · Current: ${esc(cmp.current.filename)} (${fmt(cmp.current.rows)} rows). All output here is descriptive statistics — signals for investigation, never proof of a data defect.</span></p>`;
}

/* ============================================================== REMEDIATION */
async function pageRemediation(el) {
  await refreshScans();
  if (!S.current) { el.innerHTML = noDataset(); return; }
  el.innerHTML = `<div class="page"><div id="remBody"></div></div>`;
  const body = $("#remBody");
  body.innerHTML = skCard(140);
  try {
    const rem = await api(`scan/${S.current}/remediation`);
    S.rem[S.current] = rem;
    if (!S.remStatus[S.current]) S.remStatus[S.current] = {};
    renderRemediation(rem, body);
  } catch (err) {
    body.innerHTML = emptyBlock("wrench", "Remediation not available", esc(err.message));
  }
}

function renderRemediation(rem, body) {
  const st = S.remStatus[S.current];
  const certain = rem.suggestions.filter((s) => s.confidence === "certain").length;
  const reviews = rem.suggestions.filter((s) => s.confidence === "review").length;
  const acceptedCount = Object.values(st).filter((v) => v === "accepted").length;
  const ruleFilter = body.dataset.ruleFilter || "all";

  const rows = rem.suggestions.filter((s) => ruleFilter === "all" || s.confidence === ruleFilter);
  body.innerHTML = `
    <div class="card">
      <div class="flex">
        <div><b>${fmt(rem.suggestions.length)} suggestion(s)</b>
          <span class="muted small">· ${fmt(certain)} safely inferrable · ${fmt(reviews)} manual review</span></div>
        <div class="spacer" style="margin-left:auto"></div>
        <select id="remFilter" class="input-inline" aria-label="Filter by confidence">
          <option value="all" ${ruleFilter === "all" ? "selected" : ""}>All types</option>
          <option value="certain" ${ruleFilter === "certain" ? "selected" : ""}>Certain only</option>
          <option value="review" ${ruleFilter === "review" ? "selected" : ""}>Manual review only</option>
        </select>
        <button class="btn" id="remCertain">${icon("check", 14)} Accept all certain</button>
        <button class="btn" id="remClear">Clear decisions</button>
        <button class="btn primary" id="remApply" ${acceptedCount ? "" : "disabled"}>${icon("download", 14)} Apply ${acceptedCount ? `(${fmt(acceptedCount)})` : "(0)"}</button>
      </div>
      <p class="notice ok" style="margin-top:12px">${icon("shield", 15)} <span>Nothing touches your original file. Approved changes produce a <b>separate cleaned copy</b> plus a full change log. Manual-review items can never be auto-applied.</span></p>
      <div class="table-wrap scalar-scroll section-gap"><table>
        <thead><tr><th>Decision</th><th>Action</th><th class="num">Row</th><th>Column</th>
        <th>Before → After</th><th>Rule</th><th>Confidence</th><th>Reason</th></tr></thead>
        <tbody>${rows.length ? rows.map((s) => {
    const decision = st[s.suggestion_id] || "pending";
    return `<tr>
          <td><div class="flex" style="gap:4px">
            <button class="btn sm okk" data-dec="accepted" data-sid="${s.suggestion_id}" title="Accept" ${s.action === "review" ? "disabled" : ""}>${icon("check", 12)}</button>
            <button class="btn sm ${decision === "rejected" ? "" : "danger"}" data-dec="rejected" data-sid="${s.suggestion_id}" title="Reject">${icon("x", 12)}</button>
          </div></td>
          <td class="mono small">${esc(s.action)}</td>
          <td class="num">${s.row ?? "—"}</td>
          <td>${esc(s.column)}</td>
          <td class="mono small">${esc(truncate(s.before, 34))} <b style="color:var(--violet)">→</b> ${s.after === undefined || s.after === null ? `<span class="muted">no value change</span>` : `<b>${esc(s.after)}</b>`}</td>
          <td class="mono small">${esc(s.rule_id)}</td>
          <td>${confBadge(s.confidence)}</td>
          <td class="small muted">${esc(truncate(s.reason, 60))}</td>
        </tr>`;
  }).join("") : `<tr><td colspan="8" class="muted small">No suggestions in this filter.</td></tr>`}</tbody>
      </table></div>
    </div>
    <div id="remAudit" class="section-gap"></div>`;

  renderRemAudit(rem);

  body.dataset.ruleFilter = ruleFilter;
  $("#remFilter").onchange = (e) => { body.dataset.ruleFilter = e.target.value; renderRemediation(rem, body); };
  $("#remCertain").onclick = () => {
    rem.suggestions.filter((s) => s.confidence === "certain").forEach((s) => { st[s.suggestion_id] = "accepted"; });
    renderRemediation(rem, body);
  };
  $("#remClear").onclick = () => { S.remStatus[S.current] = {}; renderRemediation(rem, body); };
  $$("[data-dec]", body).forEach((b) => {
    b.onclick = () => {
      const sid = b.dataset.sid;
      if (b.dataset.dec === "rejected" && st[sid] === "rejected") delete st[sid];
      else st[sid] = b.dataset.dec;
      renderRemediation(rem, body);
    };
  });
  $("#remApply").onclick = () => applyRemediation(rem, body);
}

async function applyRemediation(rem, body) {
  const st = S.remStatus[S.current];
  const accepted = Object.entries(st).filter(([, v]) => v === "accepted").map(([k]) => k);
  if (!accepted.length) { toast("Accept at least one suggestion first", "info"); return; }
  const btn = $("#remApply"); btn.disabled = true;
  try {
    const res = await api(`scan/${S.current}/remediation/apply`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ accepted_ids: accepted }),
    });
    S.rem[S.current] = Object.assign(S.rem[S.current], {
      changelog: res.changelog, applied_at: res.applied_at, applied_count: res.applied,
    });
    for (const s of rem.suggestions) {
      const sid = s.suggestion_id;
      if (st[sid] === "accepted") { s.status = s.action === "review" ? "rejected" : "applied"; delete st[sid]; }
    }
    toast(`${res.applied} change(s) applied to the cleaned copy · original file unchanged`, "success");
    notify(`Remediation applied on ${S.current}: ${res.applied} change(s)`);
    renderRemAudit(S.rem[S.current]);
    renderRemediation(S.rem[S.current], body);
  } catch (err) {
    toast("Apply failed: " + err.message, "error");
  } finally { renderRemediation(S.rem[S.current], body); }
}

function renderRemAudit(rem) {
  const box = $("#remAudit"); if (!box) return;
  const applied = (rem.changelog || []).length > 0 && rem.applied_count > 0;
  box.innerHTML = `
    <div class="card">
      <h2>Audit history</h2>
      ${rem.suggestions.some((s) => s.status === "applied") ? `
        <p class="small">Last applied: <b>${esc(rem.applied_at || "")}</b> · ${fmt(rem.applied_count)} change(s). The cleaned copy is held in memory — nothing was written to your source file.</p>
        ${(rem.changelog || []).length ? `<div class="table-wrap"><table>
          <thead><tr><th class="num">Row</th><th>Column</th><th>Action</th><th>Before</th><th>After</th><th>Reason</th></tr></thead>
          <tbody>${rem.changelog.map((e) => `<tr><td class="num">${fmt(e.row)}</td><td>${esc(e.column)}</td>
            <td class="mono small">${esc(e.action)}</td><td class="mono small">${esc(truncate(e.before, 28))}</td>
            <td class="mono small">${e.after == null ? '<span class="muted">row dropped</span>' : esc(truncate(e.after, 28))}</td>
            <td class="small muted">${esc(truncate(e.reason, 60))}</td></tr>`).join("")}</tbody></table></div>` : ""}
        <div class="flex section-gap">
          <a class="btn okk" href="${API}/scan/${S.current}/cleaned.csv" download>${icon("download", 14)} Download cleaned copy (new file)</a>
          <a class="btn" href="${API}/scan/${S.current}/changelog.csv" download>${icon("download", 14)} Changelog CSV</a>
          <a class="btn" href="${API}/scan/${S.current}/changelog.json" download>${icon("download", 14)} Changelog JSON</a>
        </div>` : `
        <p class="muted small">No changes applied yet for this dataset. Accept suggestions above and press Apply to produce a cleaned copy.</p>`}
    </div>`;
}

/* ================================================================== REPORTS */
async function pageReports(el) {
  await refreshScans();
  if (!S.current) { el.innerHTML = noDataset(); return; }
  const report = await getReport(S.current, { force: true });
  const rem = S.rem[S.current] || await api(`scan/${S.current}/remediation`).catch(() => null);
  el.innerHTML = `<div class="page">
    <div class="grid cols-3">
      <div class="card">
        <h2>Deterministic scan report</h2>
        <p class="muted small">Complete evidence: profiles, dimension scores, every finding with row references. Machine-readable.</p>
        <a class="btn" href="${API}/scan/${report.scan_id}/report.json" download>${icon("download", 14)} Download JSON</a>
      </div>
      <div class="card">
        <h2>Findings report</h2>
        <p class="muted small">One row per finding — spreadsheet-friendly for triage and sharing.</p>
        <a class="btn" href="${API}/scan/${report.scan_id}/report.csv" download>${icon("download", 14)} Download CSV</a>
      </div>
      <div class="card">
        <h2>Quality report (HTML)</h2>
        <p class="muted small">Self-contained single file with summary, dimensions, profiles and findings — shareable.</p>
        <a class="btn" href="${API}/scan/${report.scan_id}/report.html" download>${icon("download", 14)} Download HTML</a>
      </div>
    </div>

    <div class="grid cols-2 section-gap">
      <div class="card">
        <h2>Scan metadata</h2>
        ${kv([
    ["Scan ID", `<span class="mono">${esc(report.scan_id)}</span>`],
    ["File", esc(report.filename)],
    ["Generated at", esc(report.generated_at)],
    ["Size", `${fmt(report.rows)} rows × ${fmt(report.columns.length)} columns`],
    ["Release state", `${stateBadge(report.release_state || "PASS")}`],
    ["Rationale", esc(report.release_rationale || "—")],
    ["Detectors", esc(report.detectors.join(", "))],
    ["ML status", `${esc(report.ml.status)} — ${esc(report.ml.note)}`],
    ["Findings", `${fmt(report.finding_total)} (critical ${fmt(report.severity_counts.CRITICAL)}, warning ${fmt(report.severity_counts.WARNING)}, info ${fmt(report.severity_counts.INFO)})`],
    ...(report.notes && report.notes.length ? [["Notes", esc(report.notes.join("; "))]] : []),
  ])}
      </div>
      <div class="card">
        <h2>JSON preview</h2>
        <pre class="mono small" id="jsonPrev" style="background:rgba(7,11,20,.6);border:1px solid var(--border);border-radius:10px;padding:12px;max-height:320px;overflow:auto;color:var(--text-2)">loading…</pre>
      </div>
    </div>

    <div class="card section-gap">
      <h2>Audit trail</h2>
      <div class="table-wrap"><table>
        <thead><tr><th>When</th><th>Event</th><th>Detail</th></tr></thead>
        <tbody>
          <tr><td class="muted small">${esc(report.generated_at)}</td><td><b>Deterministic scan</b></td><td>${fmt(report.finding_total)} finding(s) · status ${esc(report.release_state || "PASS")}</td></tr>
          ${S.anomalies[report.scan_id] ? `<tr><td class="muted small">${esc(S.anomalies[report.scan_id].generated_at)}</td><td><b>Anomaly detection</b> (statistical, advisory)</td><td>${fmt(S.anomalies[report.scan_id].flagged_count)} row(s) flagged · seed ${esc(S.anomalies[report.scan_id].random_seed)} · does not affect release state</td></tr>` : ""}
          ${S.ai[report.scan_id] ? `<tr><td class="muted small">${esc((S.ai[report.scan_id].audit_trail || []).slice(-1).map((a) => a.timestamp)[0] || "")}</td><td><b>AI analysis</b> (advisory)</td><td>status ${esc(S.ai[report.scan_id].status)} · ${fmt((S.ai[report.scan_id].suggestions || []).length)} suggestion(s) shown, none auto-applied</td></tr>` : `<tr><td class="muted small">—</td><td>AI analysis</td><td class="muted small">not run (optional)</td></tr>`}
          ${rem && rem.applied_count ? `<tr><td class="muted small">${esc(rem.applied_at)}</td><td><b>Remediation applied</b></td><td>${fmt(rem.applied_count)} change(s) in separate cleaned copy · original file untouched</td></tr>` : `<tr><td class="muted small">—</td><td>Remediation</td><td class="muted small">nothing applied yet</td></tr>`}
        </tbody></table></div>
      <p class="muted small" style="margin-top:8px">Session events and server-side scan state; in-memory for this demo scale. Scans are not persisted across server restarts.</p>
    </div>
  </div>`;

  fetch(`${API}/scan/${report.scan_id}/report.json`, { headers: { Range: "bytes=0-3500" } })
    .then((r) => r.text())
    .then((t) => { const p = $("#jsonPrev"); if (p) p.textContent = t.slice(0, 3500) + (t.length > 3500 ? "\n…" : ""); })
    .catch(() => {});
}

/* ================================================================= SETTINGS */
async function pageSettings(el) {
  const settings = S.settings || await api("settings").catch(() => null);
  S.settings = settings;
  el.innerHTML = `<div class="page">
    <div class="grid cols-2">
      <div class="card">
        <h2>API configuration status</h2>
        ${settings ? `
        <div class="stat-list">
          <div class="stat-chip"><small>Gemini API key</small>
            <b style="color:${settings.gemini_configured ? "var(--ok)" : "var(--muted)"}">${settings.gemini_configured ? "Configured" : "Not configured"}</b></div>
          <div class="stat-chip"><small>AI model</small><b style="font-size:12.5px">${esc(settings.gemini_model || settings.cursor_model || "gemini-3.8-flash")}</b></div>
          <div class="stat-chip"><small>AI role</small><b style="font-size:12.5px">${esc(settings.gemini_role)}</b></div>
          <div class="stat-chip"><small>Anomaly module</small>
            <b style="color:${settings.anomaly_available ? "var(--ok)" : "var(--warn)"}">${settings.anomaly_available ? "Available" : "Unavailable"}</b>
            <small style="display:block;text-transform:none;letter-spacing:0;color:var(--text-2);margin-top:3px">${esc(settings.anomaly_note)}</small></div>
          <div class="stat-chip"><small>Formats</small><b>${esc(settings.supported_formats.join(", "))}</b></div>
        </div>
        <div class="stat-list section-gap">
          ${Object.entries(settings.versions).map(([k, v]) =>
    `<div class="stat-chip"><small>${esc(k)}</small><b>${esc(v || "not installed")}</b></div>`).join("")}
        </div>` : `<p class="muted small">Status unavailable.</p>`}
      </div>
      <div class="card">
        <h2>Privacy</h2>
        ${(settings ? settings.privacy : []).map((p) => `<p class="notice">${icon("shield", 14)} <span>${esc(p)}</span></p>`).join("")}
        <p class="muted small" style="margin-top:8px">No API key is ever logged, displayed or written into reports. All analysis runs on this machine.</p>
      </div>
    </div>

    <div class="grid cols-2 section-gap">
      <div class="card">
        <h2>Scan preferences (deterministic thresholds)</h2>
        <p class="muted small">Applied to your <b>next scan</b> as explicit detector configuration. Defaults below are the engine defaults — the deterministic engine is always the source of truth.</p>
        <div class="grid cols-2" style="gap:10px">
          ${prefInput("row_evidence_cap", "Row evidence cap", "per rule+column")}
          ${prefInput("outlier_iqr_k", "Outlier IQR k", "e.g. 1.5–3.0")}
          ${prefInput("future_date_days", "Future-date horizon", "days")}
          ${prefInput("critical_missing_rate", "Critical missing rate", "0–1 fraction")}
        </div>
        <div class="flex section-gap">
          <button class="btn primary" id="savePrefs">${icon("check", 14)} Save preferences</button>
          <button class="btn" id="resetPrefs">Reset to defaults</button>
        </div>
      </div>
      <div class="card">
        <h2>Optional modules</h2>
        ${toggle("anomaly_enabled", "Statistical anomaly detection", "off = the Anomaly page shows only rule findings; deterministic results unaffected")}
        ${toggle("ai_enabled", "AI analyst (Google Gemini)", "off = no external AI calls are made; everything deterministic stays available")}
      </div>
      <div class="card span-2">
        <h2>Module slots (architecture)</h2>
        <p class="muted small">The deterministic rule engine (9 detectors, five dimensions, scoring, release state) is authoritative and stable. The platform exposes defined extension points: a future ML detector can plug into the detector registry via <span class="mono">register_detector(MyDetector())</span> without touching the engine, and the anomaly module already communicates only advisory statistics.</p>
      </div>
    </div>
  </div>`;

  $("#savePrefs").onclick = () => {
    for (const k of PREF_KEYS) {
      const raw = $(`input[data-pref="${k}"]`).value.trim();
      if (raw !== "") prefs[k] = Number(raw) === Number(raw) ? Number(raw) : prefs[k];
    }
    localStorage.setItem("adqp.prefs", JSON.stringify(prefs));
    toast("Preferences saved — sent with your next scan", "success");
  };
  $("#resetPrefs").onclick = () => {
    for (const k of PREF_KEYS) prefs[k] = { row_evidence_cap: 25, outlier_iqr_k: 3.0, future_date_days: 1, critical_missing_rate: 0.30 }[k];
    localStorage.setItem("adqp.prefs", JSON.stringify(prefs));
    toast("Preferences reset to engine defaults", "info");
    renderSettingsAgain();
  };
  $$("input[type=checkbox][data-toggle]").forEach((t) => {
    t.checked = prefs[t.dataset.toggle];
    t.onchange = () => {
      prefs[t.dataset.toggle] = t.checked;
      localStorage.setItem("adqp.prefs", JSON.stringify(prefs));
      toast(`${t.dataset.toggle} ${t.checked ? "enabled" : "disabled"}`, "info");
    };
  });
}
function renderSettingsAgain() { pageSettings($("#content")); }
function prefInput(key, label, hint) {
  return `<div>
    <label class="muted small">${esc(label)} <span style="color:var(--muted)">(${esc(hint)})</span></label>
    <input class="input" style="margin-top:4px;padding-left:12px" data-pref="${key}" value="${esc(prefs[key])}">
  </div>`;
}
function toggle(key, label, note) {
  return `<div class="flex" style="justify-content:space-between;border:1px solid var(--border);border-radius:10px;padding:10px 14px;margin-bottom:10px">
    <div><b class="small">${esc(label)}</b><div class="muted small">${esc(note)}</div></div>
    <input type="checkbox" data-toggle="${key}" role="switch" style="width:38px;height:20px;accent-color:#38bdf8;cursor:pointer">
  </div>`;
}

/* ------------------------------------------------------------------ shell */
function hidePops() { $("#notifPop").hidden = true; $("#profilePop").hidden = true; }
$("#notifBtn").onclick = (e) => {
  e.stopPropagation();
  hidePops();
  $("#notifPop").hidden = !$("#notifPop").hidden;
  renderNotifs();
};
$("#profileBtn").onclick = (e) => {
  e.stopPropagation();
  hidePops();
  $("#profilePop").hidden = !$("#profilePop").hidden;
};
document.addEventListener("click", (e) => {
  if (!e.target.closest("#notifPop") && !e.target.closest("#notifBtn")) $("#notifPop").hidden = true;
  if (!e.target.closest("#profilePop") && !e.target.closest("#profileBtn")) $("#profilePop").hidden = true;
});
$("#datasetSelect").onchange = async (e) => {
  S.current = e.target.value || null;
  localStorage.setItem("adqp.dataset", S.current || "");
  $("#datasetSelect").dataset.value = S.current || "";
  updateShell();
  route();
};
$("#collapseBtn").onclick = () => {
  const c = document.body.classList.toggle("app-collapsed") || $("#app").classList.toggle("collapsed");
  localStorage.setItem("adqp.sidebar", $("#app").classList.contains("collapsed") ? "c" : "");
};
$("#navToggle").onclick = () => $("#sidebar").classList.add("open");
$("#sidebarBackdrop").onclick = () => $("#sidebar").classList.remove("open");

/* restore sidebar state */
if (localStorage.getItem("adqp.sidebar") === "c") $("#app").classList.add("collapsed");

/* ------------------------------------------------------------------- boot */
(async function boot() {
  try { await refreshScans(); }
  catch (err) { toast("Server unreachable: " + err.message, "error"); }
  route();
})();
