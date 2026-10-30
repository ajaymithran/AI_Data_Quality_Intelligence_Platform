"""Universal CSV scanner: the orchestrator.

``scan_csv`` / ``scan_bytes`` accept ANY CSV — no expected columns, no
domain assumptions. Source data is only ever read, never written.

Pipeline:
  raw CSV -> all-string DataFrame -> automatic per-column profiling
          -> registered detectors -> findings -> report + downloads
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd

from .context import DEFAULT_CONFIG, ScanContext
from .detectors import registered_detectors
from .inference import profile_frame
from .schema import DIMENSIONS, SEVERITIES, ScanReport, reset_ids


def _to_bytes(source) -> bytes:
    if isinstance(source, bytes):
        return source
    if isinstance(source, (str, Path)):
        return Path(source).read_bytes()
    data = source.read()
    return data if isinstance(data, bytes) else str(data).encode("utf-8")


def _raw_header(raw: bytes) -> list[str] | None:
    """Header names exactly as they appear in the file (handles quoting)."""
    import csv
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = raw.decode("latin-1")
        except Exception:
            return None
    try:
        reader = csv.reader(io.StringIO(text))
        return next(reader)
    except (StopIteration, csv.Error):
        return None


def _dedupe(names: list[str]) -> tuple[list[str], list[str]]:
    """Return (clean names, originally-duplicated names)."""
    clean: list[str] = []
    seen: dict[str, int] = {}
    dups: list[str] = []
    for raw_name in names:
        name = str(raw_name).strip() or "unnamed"
        count = seen.get(name, 0) + 1
        seen[name] = count
        if count == 1:
            clean.append(name)
        else:
            if count == 2:
                dups.append(name)
            clean.append(f"{name}_{count}")
    return clean, dups


def _read_excel(raw: bytes) -> tuple[pd.DataFrame, list[str]]:
    """Read the first Excel sheet as raw strings (matching _read_raw contract).

    Requires openpyxl (listed in requirements). Values are stringified with
    integral floats rendered without a trailing '.0'.
    """
    try:
        raw_df = pd.read_excel(io.BytesIO(raw), header=0, engine="openpyxl",
                               dtype=object, keep_default_na=False, na_values=[])
    except ImportError as e:
        raise ValueError("Excel support requires openpyxl: pip install openpyxl") from e
    except Exception as e:
        raise ValueError(f"could not parse Excel file: {e}") from e
    if raw_df.shape[1] == 0:
        raise ValueError("Excel sheet has no columns")
    if raw_df.shape[0] == 0:
        raise ValueError("Excel sheet has no data rows")

    names, dups = _dedupe([("" if v is None else str(v)).strip() or "unnamed"
                           for v in raw_df.columns])
    df = raw_df.copy()
    df.columns = names

    def _fmt(v) -> str:
        if v is None or (isinstance(v, float) and pd.isna(v)) or pd.isna(v):
            return ""
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return str(v)

    for c in df.columns:
        df[c] = df[c].map(_fmt)
    df = df.astype(str) if len(df) else df.astype(str)
    warnings = []
    if dups:
        warnings.append(f"duplicate column names renamed: {dups}")
    warnings.append("excel source: first sheet read")
    return df, warnings


def _read_raw(source, *, excel: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """Read CSV (or Excel) as raw strings. Returns (df, warnings). Missing -> ''."""
    warnings: list[str] = []
    raw = _to_bytes(source)
    if not raw.strip():
        raise ValueError("could not parse CSV: file is empty")
    if excel:
        return _read_excel(raw)
    try:
        df = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False, na_filter=False)
    except Exception as e:  # unreadable file
        raise ValueError(f"could not parse CSV: {e}") from e
    if df.shape[1] == 0:
        raise ValueError("CSV has no columns")
    if df.shape[0] == 0:
        raise ValueError("CSV has no data rows")

    header = _raw_header(raw)
    if header is not None and len(header) == df.shape[1]:
        names, dups = _dedupe(header)
        df.columns = names
        if dups:
            warnings.append(f"duplicate column names renamed: {dups}")
    else:
        df.columns = [str(c).strip() or "unnamed" for c in df.columns]
    return df, warnings


def scan_df(df: pd.DataFrame, *, filename: str = "<dataframe>",
            config: dict | None = None) -> ScanReport:
    """Scan an in-memory DataFrame of raw strings."""
    reset_ids()
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    profiles, cache = profile_frame(df)
    ctx = ScanContext(
        df=df, profiles=profiles,
        numeric={c: e["numeric"] for c, e in cache.items() if e["numeric"] is not None},
        dates={c: e["dates"] for c, e in cache.items() if e["dates"] is not None},
        families={c: e["families"] for c, e in cache.items() if e.get("families") is not None},
        stripped={c: e["stripped"] for c, e in cache.items() if e.get("stripped") is not None},
        config=cfg,
    )

    findings: list[dict] = []
    for det in registered_detectors():
        findings.extend(f.to_dict() for f in det.run(ctx))

    severity_counts = {s: 0 for s in SEVERITIES}
    for f in findings:
        severity_counts[f["severity"]] += 1

    return ScanReport(
        scan_id=f"USCAN-{int(time.time() * 1000) % 10_000_000:07d}",
        filename=filename,
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        rows=len(df),
        columns=profiles,
        findings=findings,
        dimensions=_dimensions(df, profiles, findings, cfg),
        severity_counts=severity_counts,
        detectors=[d.rule_id for d in registered_detectors()],
        ml={
            "status": "not_implemented",
            "note": "ML-based detection is a planned extension; this version "
                    "uses deterministic rules only",
        },
        notes=[],
        finding_total=len(findings),
    )


def read_raw_df(raw: bytes, filename: str = "upload.csv") -> pd.DataFrame:
    """Public helper: raw bytes -> all-string DataFrame (CSV or Excel)."""
    df, _ = _read_raw(raw, excel=filename.lower().endswith((".xlsx", ".xls")))
    return df


def scan_csv(source, *, filename: str | None = None,
             config: dict | None = None) -> ScanReport:
    """Scan a CSV from a path, bytes, or file-like object (read-only).

    Excel workbooks (filename ending .xlsx/.xls) are supported the same
    way: values-only, first sheet, source never modified.
    """
    if filename is None:
        filename = source if isinstance(source, (str, Path)) else "<upload>"
        filename = Path(str(filename)).name
    df, warnings = _read_raw(source, excel=filename.lower().endswith((".xlsx", ".xls")))
    report = scan_df(df, filename=filename, config=config)
    report.notes.extend(warnings)
    return report


def scan_bytes(raw: bytes, filename: str = "upload.csv",
               config: dict | None = None) -> ScanReport:
    return scan_csv(raw, filename=filename, config=config)


# --------------------------------------------------------------------------
# data-quality dimensions (0-100, higher is better)
# --------------------------------------------------------------------------

def _dimensions(df: pd.DataFrame, profiles, findings: list[dict], cfg: dict) -> dict:
    rows, cols = df.shape
    cells = max(rows * cols, 1)

    missing_cells = sum(p.missing + p.blank for p in profiles)
    completeness = round((1 - missing_cells / cells) * 100, 1)

    dup_rows = int(df.duplicated(keep="first").sum())
    uniqueness = round((1 - dup_rows / max(rows, 1)) * 100, 1)

    # validity: share of real values that parse in the column's dominant type
    valid = 0
    total_real = 0
    for p in profiles:
        real = p.non_null
        total_real += real
        if p.mixed_types:
            dom = max(p.mixed_types, key=p.mixed_types.get)
            bad = sum(c for f, c in p.mixed_types.items() if f != dom)
        else:
            bad = 0
        valid += real - bad
    validity = round(valid / max(total_real, 1) * 100, 1)

    # consistency: penalise cells flagged by consistency/validity rules
    flagged = {f["row"] for f in findings
               if f["row"] is not None and f["dimension"] in ("consistency", "validity")}
    consistency = round((1 - len(flagged) / max(rows, 1)) * 100, 1)

    # plausibility: penalise suspicious (outlier/future) cells
    susp = {f["row"] for f in findings if f["dimension"] == "plausibility"}
    plausibility = round((1 - len(susp) / max(rows, 1)) * 100, 1)

    dims = {
        "completeness": completeness,
        "uniqueness": uniqueness,
        "validity": validity,
        "consistency": consistency,
        "plausibility": plausibility,
    }
    # guard: all dimensions must be in 0..100
    return {k: max(0.0, min(100.0, v)) for k, v in dims.items()}
