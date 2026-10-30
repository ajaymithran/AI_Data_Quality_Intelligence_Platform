"""Report builders: JSON, CSV (per-finding), and a self-contained HTML report.

Every builder takes a ScanReport and returns bytes ready to download.
The HTML report is a single file with no external assets.
"""
from __future__ import annotations

import csv
import io
import json

from .schema import ScanReport

SEV_COLORS = {"CRITICAL": "#c0392b", "WARNING": "#d68910", "INFO": "#2471a3"}


def to_json(report: ScanReport) -> bytes:
    return json.dumps(report.to_dict(), indent=2, default=str).encode("utf-8")


def to_csv(report: ScanReport) -> bytes:
    """One row per finding — flat, spreadsheet-friendly."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["finding_id", "severity", "rule_id", "column", "row",
                "value", "message", "evidence", "suggested_action", "dimension"])
    for f in report.findings:
        w.writerow([f["finding_id"], f["severity"], f["rule_id"], f["column"],
                    f["row"] or "", f["value"] if f["value"] is not None else "",
                    f["message"], f["evidence"], f["suggested_action"], f["dimension"]])
    return buf.getvalue().encode("utf-8")


def to_html(report: ScanReport) -> bytes:
    """Single-file HTML report: summary, dimensions, columns, findings."""
    d = report.to_dict()
    sev = d["severity_counts"]

    dim_rows = "".join(
        f"<tr><td>{k.title()}</td><td><div class='bar'><span style='width:{v}%'></span></div></td>"
        f"<td>{v}%</td></tr>" for k, v in d["dimensions"].items())

    col_rows = "".join(
        f"<tr><td>{c['name']}</td><td><span class='tag'>{c['inferred_type']}</span></td>"
        f"<td>{c['non_null']}</td><td class='{'bad' if c['missing'] else ''}'>{c['missing']}</td>"
        f"<td class='{'bad' if c['blank'] else ''}'>{c['blank']}</td>"
        f"<td class='{'warn' if c['placeholder'] else ''}'>{c['placeholder']}</td>"
        f"<td>{c['unique']}</td><td>{c['unique_pct']}%</td>"
        f"<td>{c['date_format'] or ''}</td></tr>"
        for c in d["columns"])

    findings_sorted = sorted(d["findings"],
                             key=lambda f: {"CRITICAL": 0, "WARNING": 1, "INFO": 2}[f["severity"]])
    f_rows = "".join(
        f"<tr><td><span class='sev' style='background:{SEV_COLORS[f['severity']]}'></span>"
        f"{f['severity']}</td><td>{f['rule_id']}</td><td>{f['column']}</td>"
        f"<td>{f['row'] if f['row'] is not None else '—'}</td>"
        f"<td class='mono'>{_esc(f['value'])}</td><td>{_esc(f['message'])}</td>"
        f"<td class='mono small'>{_esc(f['evidence'])}</td></tr>"
        for f in findings_sorted[:500])
    if len(findings_sorted) > 500:
        f_rows += f"<tr><td colspan='7'><em>… {len(findings_sorted) - 500} more findings — " \
                  f"download the CSV/JSON report for the full list</em></td></tr>"

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Data quality report — {_esc(d['filename'])}</title>
<style>
 body{{font:14px/1.5 system-ui,sans-serif;margin:2rem auto;max-width:1100px;color:#222;padding:0 1rem}}
 h1{{font-size:1.4rem}} h2{{margin-top:2rem;font-size:1.1rem}}
 table{{border-collapse:collapse;width:100%}}
 th,td{{border:1px solid #ddd;padding:.4rem .6rem;text-align:left;vertical-align:top}}
 th{{background:#f4f6f8}} .mono{{font-family:Consolas,monospace}} .small{{font-size:12px;color:#555}}
 .sev{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}}
 .tag{{background:#eef2f5;padding:1px 8px;border-radius:10px;font-size:12px}}
 .bar{{background:#eee;height:12px;width:220px;border-radius:6px;overflow:hidden}}
 .bar span{{display:block;background:#2e86c1;height:100%}}
 .bad{{color:#c0392b;font-weight:600}} .warn{{color:#d68910;font-weight:600}}
 .kpi span{{display:inline-block;background:#f4f6f8;border-radius:8px;padding:.4rem .9rem;margin:0 .4rem .4rem 0}}
 .kpi b{{font-size:1.3rem}}
</style></head><body>
<h1>Data quality report — {_esc(d['filename'])}</h1>
<p class="small">Generated {d['generated_at']} · scan {d['scan_id']} ·
{d['rows']} rows × {len(d['columns'])} columns · detectors: {', '.join(d['detectors'])}</p>
<p class="small"><b>Note:</b> {_esc(d['ml']['note'])}. Source data was not modified.</p>

<div class="kpi">
 <span><b style="color:{SEV_COLORS['CRITICAL']}">{sev['CRITICAL']}</b> critical</span>
 <span><b style="color:{SEV_COLORS['WARNING']}">{sev['WARNING']}</b> warnings</span>
 <span><b style="color:{SEV_COLORS['INFO']}">{sev['INFO']}</b> info</span>
 <span><b>{d['finding_total']}</b> findings total</span>
</div>

<h2>Data-quality dimensions</h2>
<table><tr><th>Dimension</th><th>Score</th><th></th></tr>{dim_rows}</table>

<h2>Column profiles</h2>
<table><tr><th>Column</th><th>Type</th><th>Non-null</th><th>Missing</th><th>Blank</th>
<th>Placeholder</th><th>Unique</th><th>Unique %</th><th>Dominant date format</th></tr>{col_rows}</table>

<h2>Findings (row-level evidence)</h2>
<table><tr><th>Severity</th><th>Rule</th><th>Column</th><th>Row</th><th>Value</th>
<th>Message</th><th>Evidence</th></tr>{f_rows}</table>
</body></html>"""
    return html.encode("utf-8")


def _esc(v) -> str:
    if v is None:
        return ""
    return (str(v).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


FORMATS = {"json": to_json, "csv": to_csv, "html": to_html}


def render(report: ScanReport, fmt: str) -> tuple[bytes, str, str]:
    """(payload, media_type, filename) for a given format name."""
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}; choose from {sorted(FORMATS)}")
    stem = (report.filename.rsplit(".", 1)[0] or "report").replace("/", "_")
    media = {"json": "application/json", "csv": "text/csv", "html": "text/html"}[fmt]
    return FORMATS[fmt](report), media, f"{stem}_quality_report.{fmt}"
