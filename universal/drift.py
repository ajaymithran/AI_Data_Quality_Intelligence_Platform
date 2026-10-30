"""Baseline comparison / drift detection (deterministic statistics).

Compares a current scan against a saved baseline scan using transparent,
descriptive statistics:

- column presence and inferred-type changes,
- missing-rate changes,
- distinct-value-count changes,
- numeric distribution summaries plus a KS statistic,
- categorical frequency shifts,
- date-range changes.

Every change is reported as a SIGNAL for investigation — never as proof
of a data defect. Nothing here modifies source data.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from .inference import parse_dates
from .schema import ColumnProfile, ScanReport

MISSING_RATE_ALERT_PP = 5.0    # percentage points
KS_ALERT = 0.30                # descriptive threshold, not a significance test


def _pct(x: int | float, n: int) -> float:
    return round(x / max(n, 1) * 100, 1)


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Empirical Kolmogorov-Smirnov distance (descriptive, no p-value)."""
    a = np.sort(a[np.isfinite(a)])
    b = np.sort(b[np.isfinite(b)])
    if a.size == 0 or b.size == 0:
        return 0.0
    allv = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, allv, side="right") / a.size
    cdf_b = np.searchsorted(b, allv, side="right") / b.size
    return round(float(np.max(np.abs(cdf_a - cdf_b))), 3)


def _profile_map(report: ScanReport) -> dict[str, ColumnProfile]:
    return {p.name: p for p in report.columns}


def _numeric_values(df: pd.DataFrame, column: str) -> np.ndarray:
    series = df[column].astype(str).str.strip()
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")


def _date_bounds(df: pd.DataFrame, profile: ColumnProfile) -> dict[str, Any] | None:
    """min/max of a date column via its dominant format (None if unparsable)."""
    if not (profile.inferred_type in ("date", "datetime") and profile.date_format):
        return None
    dates = parse_dates(df[profile.name], profile.date_format)
    valid = [d for d in dates if d is not None and d is not pd.NaT]
    if not valid:
        return None
    return {"min": min(valid).isoformat(), "max": max(valid).isoformat(),
            "valid": len(valid)}


def _category_shifts(b_df: pd.DataFrame, c_df: pd.DataFrame,
                     b_prof: ColumnProfile, c_prof: ColumnProfile,
                     top: int = 6) -> dict[str, Any]:
    """Top category frequencies baseline vs current for a low-cardinality text column."""
    def _top(df: pd.DataFrame) -> dict[str, float]:
        series = df[b_prof.name].astype(str).str.strip()
        series = series[series != ""]
        counts = series.value_counts()
        total = int(counts.sum())
        if not total:
            return {}
        return {str(k): round(v / total * 100, 1)
                for k, v in counts.head(top).items()}

    b = _top(b_df)
    c = _top(c_df)
    if not b and not c:
        return {}
    labels = sorted(set(b) | set(c))
    return {
        "column": b_prof.name,
        "baseline": b,
        "current": c,
        "labels": labels,
        "new_labels": [k for k in c if k not in b],
    }


def compare(baseline: ScanReport, baseline_df: pd.DataFrame,
            current: ScanReport, current_df: pd.DataFrame) -> dict[str, Any]:
    """Compare a current scan against a saved baseline scan."""
    b_prof, c_prof = _profile_map(baseline), _profile_map(current)
    b_cols, c_cols = set(b_prof), set(c_prof)

    added = sorted(c_cols - b_cols)
    removed = sorted(b_cols - c_cols)
    type_changes = [
        {"column": col, "baseline": b_prof[col].inferred_type,
         "current": c_prof[col].inferred_type}
        for col in sorted(b_cols & c_cols)
        if b_prof[col].inferred_type != c_prof[col].inferred_type
    ]

    missing_rates: list[dict[str, Any]] = []
    distinct_changes: list[dict[str, Any]] = []
    numeric: list[dict[str, Any]] = []
    date_ranges: list[dict[str, Any]] = []
    categories: list[dict[str, Any]] = []

    for col in sorted(b_cols & c_cols):
        bp, cp = b_prof[col], c_prof[col]
        b_miss = _pct(bp.missing + bp.blank, baseline.rows)
        c_miss = _pct(cp.missing + cp.blank, current.rows)
        if abs(c_miss - b_miss) > 0.01:
            missing_rates.append({
                "column": col,
                "baseline_pct": b_miss, "current_pct": c_miss,
                "delta_pp": round(c_miss - b_miss, 1),
            })
        distinct_changes.append({
            "column": col,
            "baseline": bp.unique, "current": cp.unique,
            "delta": cp.unique - bp.unique,
        })
        if (cp.inferred_type in ("integer", "float")
                and cp.inferred_type == bp.inferred_type):
            a, b = _numeric_values(baseline_df, col), _numeric_values(current_df, col)
            a_ok, b_ok = a[np.isfinite(a)], b[np.isfinite(b)]
            if a_ok.size >= 3 and b_ok.size >= 3:
                numeric.append({
                    "column": col,
                    "baseline": {"mean": round(float(a_ok.mean()), 4),
                                 "median": round(float(np.median(a_ok)), 4),
                                 "min": round(float(a_ok.min()), 4),
                                 "max": round(float(a_ok.max()), 4),
                                 "valid": int(a_ok.size)},
                    "current": {"mean": round(float(b_ok.mean()), 4),
                                "median": round(float(np.median(b_ok)), 4),
                                "min": round(float(b_ok.min()), 4),
                                "max": round(float(b_ok.max()), 4),
                                "valid": int(b_ok.size)},
                    "ks_statistic": _ks_statistic(a_ok, b_ok),
                })
        if cp.inferred_type in ("date", "datetime") and bp.date_format and cp.date_format:
            b_bounds = _date_bounds(baseline_df, bp)
            c_bounds = _date_bounds(current_df, cp)
            if b_bounds and c_bounds:
                date_ranges.append({
                    "column": col,
                    "baseline": b_bounds, "current": c_bounds,
                    "min_changed": b_bounds["min"] != c_bounds["min"],
                    "max_changed": b_bounds["max"] != c_bounds["max"],
                })
        if (bp.inferred_type == c_prof[col].inferred_type == "text"
                and bp.unique_pct < 90 and cp.unique_pct < 90):
            shift = _category_shifts(baseline_df, current_df, bp, cp)
            if shift:
                categories.append(shift)

    missing_rates.sort(key=lambda m: -abs(m["delta_pp"]))
    numeric.sort(key=lambda n: -n["ks_statistic"])

    signals: list[dict[str, Any]] = []
    for col in added:
        signals.append({"severity": "INFO", "column": col,
                        "kind": "column_added",
                        "message": f"column '{col}' is new in the current dataset"})
    for col in removed:
        signals.append({"severity": "WARNING", "column": col,
                        "kind": "column_removed",
                        "message": f"column '{col}' present in the baseline is missing now"})
    for t in type_changes:
        signals.append({"severity": "WARNING", "column": t["column"],
                        "kind": "type_changed",
                        "message": f"inferred type changed: "
                                   f"{t['baseline']} -> {t['current']}"})
    for m in missing_rates:
        if abs(m["delta_pp"]) >= MISSING_RATE_ALERT_PP:
            signals.append({
                "severity": "WARNING", "column": m["column"],
                "kind": "missing_rate_changed",
                "message": f"missing rate moved {m['delta_pp']:+.1f} pp "
                           f"({m['baseline_pct']}% -> {m['current_pct']}%)"})
    for n in numeric:
        if n["ks_statistic"] >= KS_ALERT:
            signals.append({
                "severity": "WARNING", "column": n["column"],
                "kind": "distribution_shift",
                "message": f"numeric distribution differs (KS statistic "
                           f"{n['ks_statistic']:.2f}); investigate, not proof of a defect"})
    for cat in categories:
        for label in cat["new_labels"]:
            share = cat["current"].get(label, 0)
            if share >= 10:
                signals.append({
                    "severity": "INFO", "column": cat["column"],
                    "kind": "category_shift",
                    "message": f"'{label}' accounts for {share}% of '{cat['column']}' "
                               "now but was absent in the baseline"})

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "baseline": {
            "scan_id": baseline.scan_id, "filename": baseline.filename,
            "generated_at": baseline.generated_at, "rows": baseline.rows,
            "columns": len(baseline.columns),
            "dimensions": baseline.dimensions,
        },
        "current": {
            "scan_id": current.scan_id, "filename": current.filename,
            "generated_at": current.generated_at, "rows": current.rows,
            "columns": len(current.columns),
            "dimensions": current.dimensions,
        },
        "columns_added": added,
        "columns_removed": removed,
        "type_changes": type_changes,
        "missing_rates": missing_rates,
        "distinct_changes": distinct_changes,
        "numeric_distribution": numeric,
        "date_ranges": date_ranges,
        "category_shifts": categories,
        "signals": signals,
        "thresholds": {
            "missing_rate_alert_pp": MISSING_RATE_ALERT_PP,
            "ks_alert": KS_ALERT,
        },
        "note": "Changes are descriptive statistics: signals for review, "
                "not proof of a data defect.",
    }
