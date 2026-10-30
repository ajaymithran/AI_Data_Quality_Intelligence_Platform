"""Statistical anomaly detection (Isolation Forest, scikit-learn).

Runs as an OPTIONAL, on-demand module after the deterministic scan.
It never feeds the deterministic quality engine: anomalies are treated
as review candidates (statistically unusual rows), not as errors.

Design rules honoured here:
- suitability is checked first (rows, usable numeric features); when the
  data is not suitable the caller gets an explicit "skipped" status and
  a plain-language reason;
- results are reproducible (fixed random seed);
- every flagged row carries measurable evidence (feature deviations),
  plus model limitations the UI must display;
- nothing here modifies the source data.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

RANDOM_SEED = 42
MIN_ROWS = 50            # too little data -> the model would be noise
MAX_FEATURES = 12        # keep the feature set explainable
MIN_VALID_RATE = 0.80    # fraction of parseable values a feature needs
CONTAMINATION_OPTIONS = ("auto", 0.05, 0.10, 0.20)

LIMITATIONS = [
    "An isolation forest flags statistically unusual rows; an anomaly is "
    "not automatically an error.",
    "Flagged rows are review candidates, not proven defects, and are kept "
    "separate from deterministic rule findings.",
    "Only numeric features are considered; text structure, categories and "
    "dates are not modelled.",
    "Results depend on the contamination setting and are reproducible only "
    "with the fixed random seed.",
    "Missing numeric values are median-imputed, which can affect scoring of "
    "rows with many missing values.",
]


def _to_float(col: pd.Series) -> pd.Series:
    """Numeric view of an all-string column; whitespace stripped."""
    return pd.to_numeric(col.astype(str).str.strip(), errors="coerce")


def pick_features(df: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    """Choose usable numeric features and return an imputed/scaled matrix.

    Returns (feature_names, matrix) where matrix is a float ndarray with
    one column per feature (NaN already handled). Columns are dropped when
    too few values parse, or when the scaled column is constant.
    """
    features: list[str] = []
    cols: list[np.ndarray] = []
    for name in df.columns:
        series = _to_float(df[name])
        valid = series.notna()
        if len(df) == 0 or valid.mean() < MIN_VALID_RATE:
            continue
        values = series.to_numpy(dtype="float64")
        median = np.nanmedian(values)
        if not np.isfinite(median):
            continue
        imputed = np.where(np.isnan(values), median, values)
        std = float(np.nanstd(imputed))
        if std <= 1e-12:
            continue
        scaled = (imputed - median) / std
        scaled[~valid.to_numpy()] = 0.0     # impute at the median (= 0)
        features.append(name)
        cols.append(scaled)
        if len(features) >= MAX_FEATURES:
            break
    if not features:
        return [], np.empty((len(df), 0), dtype="float64")
    return features, np.column_stack(cols)


def run_anomaly_detection(
    df: pd.DataFrame,
    *,
    contamination: str | float = "auto",
    random_seed: int = RANDOM_SEED,
    min_rows: int = MIN_ROWS,
) -> dict[str, Any]:
    """Run Isolation Forest over numeric features of a raw-string DataFrame.

    Returns a fully serialisable dict; never raises for data problems —
    unsuitable data produces status="skipped" with a reason.
    """
    started = time.time()
    rows = len(df)
    result: dict[str, Any] = {
        "status": "skipped",
        "reason": "",
        "model": "IsolationForest",
        "random_seed": int(random_seed),
        "contamination": str(contamination),
        "rows_evaluated": int(rows),
        "features_used": [],
        "feature_count": 0,
        "flagged_count": 0,
        "flagged_pct": 0.0,
        "score_stats": None,
        "histogram": [],
        "flagged_rows": [],
        "limitations": list(LIMITATIONS),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "duration_ms": 0,
    }

    if rows < min_rows:
        result["reason"] = (
            f"dataset has {rows} rows; at least {min_rows} usable rows are "
            "needed before anomaly scoring is meaningful")
        return _finish(result, started)

    try:
        from sklearn.ensemble import IsolationForest
    except Exception as e:  # sklearn not installed
        result["status"] = "unavailable"
        result["reason"] = (
            "scikit-learn is not installed; anomaly detection is optional. "
            f"(import error: {e})")
        return _finish(result, started)

    if contamination != "auto":
        try:
            contamination = float(contamination)
        except (TypeError, ValueError):
            contamination = "auto"
        if isinstance(contamination, float) and not 0.0 < contamination < 0.5:
            contamination = "auto"

    features, matrix = pick_features(df)
    if len(features) < 1:
        result["reason"] = (
            "no usable numeric features: columns must be at least "
            f"{int(MIN_VALID_RATE * 100)}% parseable numbers and vary")
        return _finish(result, started)

    model = IsolationForest(
        n_estimators=100, random_state=random_seed, contamination=contamination)
    try:
        model.fit(matrix)
    except Exception as e:
        result["status"] = "skipped"
        result["reason"] = f"model could not be fitted: {e}"
        return _finish(result, started)

    scores = -model.score_samples(matrix)          # higher = more unusual
    flags = model.predict(matrix) == -1
    flag_positions = np.flatnonzero(flags)

    result["status"] = "ok"
    result["reason"] = ""
    result["features_used"] = list(features)
    result["feature_count"] = len(features)
    result["contamination"] = str(contamination)
    result["flagged_count"] = int(flags.sum())
    result["flagged_pct"] = round(float(flags.mean() * 100), 1) if rows else 0.0
    stats = scores
    result["score_stats"] = {
        "min": round(float(stats.min()), 4),
        "max": round(float(stats.max()), 4),
        "mean": round(float(stats.mean()), 4),
        "median": round(float(np.median(stats)), 4),
    }
    result["histogram"] = _histogram(stats)
    result["flagged_rows"] = _explain_rows(
        flag_positions, scores, features, matrix, cap=200)
    return _finish(result, started)


def _finish(result: dict[str, Any], started: float) -> dict[str, Any]:
    result["duration_ms"] = int((time.time() - started) * 1000)
    return result


def _histogram(scores: np.ndarray, bins: int = 20) -> list[dict[str, Any]]:
    counts, edges = np.histogram(scores, bins=bins)
    return [
        {
            "lo": round(float(edges[i]), 4),
            "hi": round(float(edges[i + 1]), 4),
            "count": int(counts[i]),
        }
        for i in range(len(counts))
    ]


def _explain_rows(positions: np.ndarray, scores: np.ndarray,
                  features: list[str], matrix: np.ndarray,
                  cap: int = 200) -> list[dict[str, Any]]:
    """Flagged rows with the most deviating features as evidence."""
    out: list[dict[str, Any]] = []
    order = positions[np.argsort(-scores[positions])]   # most anomalous first
    for pos in order[:cap]:
        z = matrix[pos]
        devs = [
            {
                "feature": features[i],
                "z": round(float(z[i]), 2),
            }
            for i in np.argsort(-np.abs(z))[:3]
            if abs(z[i]) > 0.05
        ]
        summary = ", ".join(
            f"{d['feature']} {'above' if d['z'] > 0 else 'below'} typical "
            f"(z={d['z']:+.2f})" for d in devs) or "mixed small deviations"
        out.append({
            "row": int(pos) + 1,
            "score": round(float(scores[pos]), 4),
            "summary": summary,
            "top_features": devs,
        })
    return out


def normalize_contamination(value: str | float | None) -> str | float:
    """Coerce a UI contamination choice into a valid value."""
    if value in (None, "", "auto"):
        return "auto"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "auto"
    return f if 0.0 < f < 0.5 else "auto"
