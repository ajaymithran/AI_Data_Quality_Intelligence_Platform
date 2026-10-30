"""Simple web interface for the universal scanner.

A self-contained HTML page (no build step, no external assets) served by
a small FastAPI router. Mount it on any FastAPI app:

    from universal.web import router as universal_router
    app.include_router(universal_router)

or run it standalone:  python -m universal.web
"""
from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, Response

from .ai_analyst import analyze_with_gemini
from .anomaly import run_anomaly_detection
from .drift import compare
from .remediation import build_suggestions, changelog_to_csv, apply_changes
from .report import render
from .scanner import read_raw_df, scan_bytes

router = APIRouter(tags=["universal"])

# in-memory stores (demo scale) -------------------------------------------------
_SCANS: dict[str, object] = {}          # scan_id -> ScanReport
_DFS: dict[str, pd.DataFrame] = {}      # scan_id -> raw string DataFrame
_ANOMALIES: dict[str, dict] = {}        # scan_id -> anomaly result
_REMEDIATION: dict[str, dict] = {}      # scan_id -> {suggestions, changelog, ...}
_CLEANED: dict[str, bytes] = {}         # scan_id -> cleaned CSV bytes
_BASELINES: dict[str, dict] = {}        # baseline_id -> {report, df, meta}
_MAX_STORED = 20                        # keep memory bounded; front is an audit note
_BASELINE_SEQ = 0


def _evict() -> None:
    """Drop the oldest entries when the in-memory stores grow too big."""
    for store in (_SCANS, _DFS, _ANOMALIES, _REMEDIATION, _CLEANED):
        keys = list(store.keys())
        for key in keys[: max(0, len(keys) - _MAX_STORED)]:
            store.pop(key, None)


def _release_state(severity_counts: dict) -> tuple[str, str]:
    """Deterministic release recommendation (rules, never AI)."""
    crit = severity_counts.get("CRITICAL", 0)
    warn = severity_counts.get("WARNING", 0)
    info = severity_counts.get("INFO", 0)
    if crit:
        return "BLOCKED", f"{crit} CRITICAL finding(s) require resolution before release."
    if warn:
        return "PASS WITH WARNINGS", f"{warn} WARNING finding(s) should be reviewed; no CRITICAL findings."
    if info:
        return "PASS WITH WARNINGS", f"{info} INFO finding(s) noted for awareness."
    return "PASS", "No issues detected by the configured deterministic rules."


def _scan_meta(report) -> dict:
    """Compact, serialisable scan summary for lists and selectors."""
    dims = report.dimensions or {}
    score = round(sum(dims.values()) / max(len(dims), 1), 1)
    state, rationale = _release_state(report.severity_counts)
    return {
        "scan_id": report.scan_id,
        "filename": report.filename,
        "generated_at": report.generated_at,
        "rows": report.rows,
        "column_count": len(report.columns),
        "finding_total": report.finding_total,
        "severity_counts": report.severity_counts,
        "overall_score": score,
        "release_state": state,
        "release_rationale": rationale,
    }


def _get_scan(scan_id: str):
    if scan_id not in _SCANS:
        raise HTTPException(
            404, "scan not found (scans live in memory; re-scan after restart)")
    return _SCANS[scan_id], _DFS.get(scan_id, pd.DataFrame())


# ---------------------------------------------------------------------------- UI

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"


def index_page() -> str:
    """Load the dashboard HTML (served at / and /universal)."""
    template = FRONTEND_DIR / "index.html"
    return template.read_text(encoding="utf-8")


# --------------------------------------------------------------- legacy + new UI

@router.get("/universal", response_class=HTMLResponse)
def index() -> str:
    """The scanner UI. Served at /universal when mounted on another app."""
    return index_page()


STATIC_FILES = {"styles.css": "text/css", "app.js": "application/javascript"}


@router.get("/static/{filename}")
def static_file(filename: str) -> Response:
    """Serve dashboard assets (no build step)."""
    if filename not in STATIC_FILES:
        raise HTTPException(404, "unknown asset")
    path = FRONTEND_DIR / filename
    return Response(path.read_bytes(), media_type=STATIC_FILES[filename],
                    headers={"Cache-Control": "no-store"})


# ---------------------------------------------------------------------- scanning

@router.post("/api/universal/scan")
async def scan(file: UploadFile = File(...), config: str = Form("")) -> dict:
    """Scan an uploaded CSV/Excel file. The CPU-bound scan runs in a worker
    thread, so a large file cannot block the server or appear to hang."""
    raw = await file.read()
    if not raw.strip():
        raise HTTPException(422, "could not parse CSV: file is empty")
    filename = file.filename or "upload.csv"
    try:
        options = json.loads(config) if config.strip() else {}
    except json.JSONDecodeError as e:
        raise HTTPException(422, f"invalid config JSON: {e}")
    if not isinstance(options, dict):
        raise HTTPException(422, "config must be a JSON object")
    try:
        report = await run_in_threadpool(scan_bytes, raw, filename, options)
    except ValueError as e:
        raise HTTPException(422, str(e))
    _SCANS[report.scan_id] = report
    # Store the raw all-string dataframe for AI analysis / anomalies /
    # comparison (read-only, in memory; never written back to disk).
    try:
        _DFS[report.scan_id] = await run_in_threadpool(read_raw_df, raw, filename)
    except Exception:
        _DFS[report.scan_id] = pd.DataFrame()

    _evict()
    result = report.to_dict()
    stored_df = _DFS.get(report.scan_id, pd.DataFrame())
    result["preview"] = _preview(stored_df)
    result["release_state"], result["release_rationale"] = _release_state(
        report.severity_counts)
    result["overall_score"] = _scan_meta(report)["overall_score"]
    return result


def _preview(df: pd.DataFrame, rows: int = 5, cell: int = 40) -> dict:
    """First rows of the uploaded data, for the file preview card."""
    if df.empty:
        return {"columns": [], "rows": []}
    columns = [str(c) for c in df.columns]
    out_rows = []
    for _, row in df.head(rows).iterrows():
        out_rows.append([str(row[name])[:cell] for name in df.columns])
    return {"columns": columns, "rows": out_rows}


@router.get("/api/universal/sample-data")
def sample_data() -> Response:
    """Serve the bundled synthetic sample CSV (clearly demo data)."""
    path = Path(__file__).resolve().parent.parent / "data" / "sample_universal_health.csv"
    return Response(
        path.read_bytes(), media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=\"sample_universal_health.csv\""})


# ------------------------------------------------------------------ scan history

@router.get("/api/universal/scans")
def list_scans() -> list[dict]:
    """Recent scans, newest first (metadata only)."""
    metas = [_scan_meta(r) for r in _SCANS.values()]
    metas.reverse()
    return metas


@router.get("/api/universal/scans/{scan_id}")
def get_scan(scan_id: str) -> dict:
    """Full report JSON for a stored scan (same shape as the scan response)."""
    report, df = _get_scan(scan_id)
    d = report.to_dict()
    d["preview"] = _preview(df)
    d["release_state"], d["release_rationale"] = _release_state(report.severity_counts)
    d["overall_score"] = _scan_meta(report)["overall_score"]
    return d


@router.delete("/api/universal/scans/{scan_id}")
def delete_scan(scan_id: str) -> dict:
    _get_scan(scan_id)
    for store in (_SCANS, _DFS, _ANOMALIES, _REMEDIATION, _CLEANED):
        store.pop(scan_id, None)
    return {"deleted": scan_id}


# ----------------------------------------------------------------------- anomalies

@router.post("/api/universal/scan/{scan_id}/anomalies")
async def run_anomalies(scan_id: str, contamination: str = "auto") -> dict:
    """Run the optional statistical anomaly module on a stored scan.

    Separate from deterministic findings: anomalies are review candidates,
    not errors, and never change scores or release state.
    """
    report, df = _get_scan(scan_id)
    result = await run_in_threadpool(
        run_anomaly_detection, df,
        contamination=contamination if contamination not in ("", None) else "auto")
    result["scan_id"] = scan_id
    result["filename"] = report.filename
    _ANOMALIES[scan_id] = result
    return result


@router.get("/api/universal/scan/{scan_id}/anomalies")
def get_anomalies(scan_id: str) -> dict:
    _get_scan(scan_id)
    result = _ANOMALIES.get(scan_id)
    if result is None:
        raise HTTPException(404, "no anomaly result yet for this scan — run it first")
    return result


@router.delete("/api/universal/scan/{scan_id}/anomalies")
def delete_anomalies(scan_id: str) -> dict:
    _get_scan(scan_id)
    _ANOMALIES.pop(scan_id, None)
    return {"deleted": scan_id}


# ---------------------------------------------------------------------- baselines

@router.post("/api/universal/baselines")
async def save_baseline(file: UploadFile = File(...)) -> dict:
    """Upload a file and store its scan as a named baseline."""
    raw = await file.read()
    if not raw.strip():
        raise HTTPException(422, "could not parse CSV: file is empty")
    filename = file.filename or "baseline.csv"
    try:
        report = await run_in_threadpool(scan_bytes, raw, filename)
    except ValueError as e:
        raise HTTPException(422, str(e))
    try:
        df = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False,
                         na_filter=False)
    except Exception:
        df = pd.DataFrame()
    global _BASELINE_SEQ
    _BASELINE_SEQ += 1
    baseline_id = f"BASE-{_BASELINE_SEQ:03d}"
    meta = _scan_meta(report)
    _BASELINES[baseline_id] = {
        "baseline_id": baseline_id,
        "report": report,
        "df": df,
        "scan_id": report.scan_id,
        "filename": report.filename,
        "generated_at": report.generated_at,
        "rows": meta["rows"],
        "column_count": meta["column_count"],
        "overall_score": meta["overall_score"],
    }
    return {"baseline_id": baseline_id, "scan_id": report.scan_id,
            "filename": report.filename, "rows": meta["rows"],
            "column_count": meta["column_count"],
            "overall_score": meta["overall_score"],
            "generated_at": report.generated_at}


@router.get("/api/universal/baselines")
def list_baselines() -> list[dict]:
    out = []
    for b in _BASELINES.values():
        out.append({k: b[k] for k in
                    ("baseline_id", "scan_id", "filename", "generated_at",
                     "rows", "column_count", "overall_score")})
    return out


@router.post("/api/universal/baselines/from-scan")
async def save_baseline_from_scan(scan_id: str) -> dict:
    """Promote an already-scanned dataset (in-memory copy) to a baseline."""
    report, df = _get_scan(scan_id)
    global _BASELINE_SEQ
    _BASELINE_SEQ += 1
    baseline_id = f"BASE-{_BASELINE_SEQ:03d}"
    meta = _scan_meta(report)
    _BASELINES[baseline_id] = {
        "baseline_id": baseline_id,
        "report": report,
        "df": df,
        "scan_id": report.scan_id,
        "filename": report.filename,
        "generated_at": report.generated_at,
        "rows": meta["rows"],
        "column_count": meta["column_count"],
        "overall_score": meta["overall_score"],
    }
    return {"baseline_id": baseline_id, "scan_id": report.scan_id,
            "filename": report.filename, "rows": meta["rows"],
            "column_count": meta["column_count"],
            "overall_score": meta["overall_score"],
            "generated_at": report.generated_at}


@router.delete("/api/universal/baselines/{baseline_id}")
def delete_baseline(baseline_id: str) -> dict:
    if baseline_id not in _BASELINES:
        raise HTTPException(404, "baseline not found")
    _BASELINES.pop(baseline_id, None)
    return {"deleted": baseline_id}


# ---------------------------------------------------------------------- comparison

@router.get("/api/universal/compare")
def compare_current_with_baseline(scan_id: str, baseline_id: str) -> dict:
    """Compare a stored current scan against a stored baseline."""
    report, df = _get_scan(scan_id)
    base = _BASELINES.get(baseline_id)
    if base is None:
        raise HTTPException(404, "baseline not found")
    return compare(base["report"], base["df"], report, df)


# ---------------------------------------------------------------------- remediation

@router.get("/api/universal/scan/{scan_id}/remediation")
def get_remediation(scan_id: str) -> dict:
    """Deterministic, reviewable suggestions for a stored scan."""
    report, df = _get_scan(scan_id)
    store = _REMEDIATION.get(scan_id)
    if store is None:
        suggestions = build_suggestions(report, df)
        store = {
            "suggestions": suggestions,
            "changelog": [],
            "applied_at": None,
            "applied_count": 0,
        }
        _REMEDIATION[scan_id] = store
    return {
        "scan_id": scan_id,
        "suggestions": store["suggestions"],
        "changelog": store["changelog"],
        "applied_at": store["applied_at"],
        "applied_count": store["applied_count"],
        "note": "Nothing is applied until you approve. The original file is "
                "never modified; the cleaned copy is a separate download.",
    }


@router.post("/api/universal/scan/{scan_id}/remediation/apply")
async def apply_remediation(scan_id: str, body: dict | None = None) -> dict:
    """Apply accepted suggestions; returns summary + stores cleaned copy."""
    _get_scan(scan_id)
    store = _REMEDIATION.get(scan_id)
    if store is None:
        raise HTTPException(404, "no remediation suggestions yet — open the "
                                 "remediation endpoint first")
    body = body or {}
    accepted_ids = body.get("accepted_ids", [])
    if not isinstance(accepted_ids, list):
        raise HTTPException(422, "accepted_ids must be a list")

    suggestions = store["suggestions"]
    known = {s["suggestion_id"] for s in suggestions}
    unknown = [i for i in accepted_ids if i not in known]
    if unknown:
        raise HTTPException(422, f"unknown suggestion ids: {unknown[:5]}")

    df = _DFS.get(scan_id, pd.DataFrame())
    cleaned, changelog = apply_changes(df, suggestions, accepted_ids)
    _CLEANED[scan_id] = cleaned.to_csv(index=False).encode("utf-8") \
        if not cleaned.empty else b""

    store["changelog"] = changelog
    store["applied_at"] = pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%S")
    store["applied_count"] = len(changelog)

    decisions = {
        s["suggestion_id"]: s["status"] for s in suggestions}
    accepted_n = sum(1 for v in decisions.values() if v == "applied")
    rejected_n = sum(1 for v in decisions.values() if v == "rejected")
    return {
        "scan_id": scan_id,
        "applied": accepted_n,
        "rejected_or_review": rejected_n,
        "changelog": changelog,
        "applied_at": store["applied_at"],
        "cleaned_ready": scan_id in _CLEANED,
        "note": "Original file unchanged. The cleaned copy is a separate "
                "in-memory download, not written to your source file.",
    }


@router.get("/api/universal/scan/{scan_id}/cleaned.csv")
def download_cleaned(scan_id: str) -> Response:
    _get_scan(scan_id)
    payload = _CLEANED.get(scan_id)
    if payload is None:
        raise HTTPException(404, "no cleaned copy yet — apply accepted "
                                 "remediation first")
    name = f"{(_get_scan(scan_id)[0].filename.rsplit('.', 1)[0] or 'report')}_cleaned.csv"
    return Response(payload, media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


@router.get("/api/universal/scan/{scan_id}/changelog.{fmt}")
def download_changelog(scan_id: str, fmt: str) -> Response:
    report, _ = _get_scan(scan_id)
    store = _REMEDIATION.get(scan_id)
    changelog = (store or {}).get("changelog") or []
    if store is None or not store.get("applied_count"):
        raise HTTPException(404, "no applied changelog yet — apply "
                                 "remediation first")
    stem = report.filename.rsplit(".", 1)[0] or "report"
    if fmt == "csv":
        payload = changelog_to_csv(changelog)
        return Response(payload, media_type="text/csv",
                        headers={"Content-Disposition":
                                 f'attachment; filename="{stem}_remediation_changelog.csv"'})
    if fmt == "json":
        payload = json.dumps(
            {"scan_id": scan_id, "applied_at": store["applied_at"],
             "changes": changelog}, indent=2).encode("utf-8")
        return Response(payload, media_type="application/json",
                        headers={"Content-Disposition":
                                 f'attachment; filename="{stem}_remediation_changelog.json"'})
    raise HTTPException(404, f"unknown changelog format {fmt!r}; use csv or json")


# ------------------------------------------------------------------ AI analysis

@router.post("/api/universal/scan/{scan_id}/ai-analysis")
def ai_analysis(scan_id: str) -> dict:
    if scan_id not in _SCANS:
        raise HTTPException(404, "scan not found (scans live in memory; re-scan after restart)")
    report = _SCANS[scan_id]
    df = _DFS.get(scan_id, pd.DataFrame())
    try:
        result = analyze_with_gemini(report, df=df)
        return result.to_dict()
    except Exception as e:
        raise HTTPException(500, f"AI analysis failed: {e}")


# ------------------------------------------------------------------------ reports

@router.get("/api/universal/scan/{scan_id}/report.{fmt}")
def download(scan_id: str, fmt: str) -> Response:
    if scan_id not in _SCANS:
        raise HTTPException(404, "scan not found (scans live in memory; re-scan after restart)")
    try:
        payload, media, name = render(_SCANS[scan_id], fmt)
    except ValueError as e:
        raise HTTPException(404, str(e))
    return Response(payload, media_type=media,
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


# ----------------------------------------------------------------------- settings

@router.get("/api/universal/settings")
def settings() -> dict:
    """Environment/configuration status for the Settings page.
    Never returns key values — only whether one is configured."""
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)
    except Exception:
        pass

    def _has_key() -> bool:
        for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            value = os.environ.get(name)
            if value and str(value).strip() and "here" not in str(value).lower():
                return True
        return False

    try:
        import sklearn
        anomaly_available = True
        sklearn_version = sklearn.__version__
    except Exception:
        anomaly_available = False
        sklearn_version = None

    from .ai_analyst import DEFAULT_MODEL as _ai_model
    return {
        "ai_provider": "Google Gemini",
        "gemini_configured": _has_key(),
        "gemini_role": "advisory only — cannot change scores or release state",
        "gemini_model": os.environ.get("GEMINI_MODEL") or os.environ.get("GOOGLE_MODEL") or _ai_model,
        "cursor_configured": False,
        "cursor_role": "removed — the AI advisor now uses Google Gemini",
        "cursor_model": os.environ.get("GEMINI_MODEL") or os.environ.get("GOOGLE_MODEL") or _ai_model,
        "grok_configured": False,
        "grok_role": "removed — the AI advisor now uses Cursor Agent",
        "grok_model": os.environ.get("CURSOR_MODEL") or _ai_model,
        "anomaly_available": anomaly_available,
        "anomaly_note": "optional, on-demand; runs separately from the "
                        "deterministic engine",
        "versions": {
            "python": f"{__import__('sys').version_info.major}."
                      f"{__import__('sys').version_info.minor}",
            "fastapi": __import__("fastapi").__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn_version,
        },
        "privacy": [
            "Scans run locally; source files are only read, never modified.",
            "AI analysis sends only structured findings and limited sample "
            "values by default — never the full dataset.",
            "No API keys are logged or stored in reports.",
        ],
        "supported_formats": ["csv", "xlsx"],
    }


def build_app():
    """Standalone FastAPI app: UI at / and /universal, API under /api/universal."""
    from fastapi import FastAPI
    app = FastAPI(title="AI Data Quality Intelligence Platform")
    app.include_router(router)

    @app.get("/", include_in_schema=False)
    def root() -> str:
        return index_page()

    return app


def main() -> None:
    import uvicorn
    uvicorn.run(build_app(), host="127.0.0.1", port=8001)


if __name__ == "__main__":
    main()
