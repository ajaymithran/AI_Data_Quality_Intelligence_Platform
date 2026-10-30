"""Tests for the optional statistical anomaly module (Isolation Forest).

The module must: check suitability first, be reproducible, explain flags
with measurable feature deviations, and stay separate from deterministic
findings (never changing scores or release state).
"""
import numpy as np
import pandas as pd
import pytest

from universal import scan_bytes, scan_df
from universal.anomaly import (
    MAX_FEATURES, run_anomaly_detection, pick_features, RANDOM_SEED)


def _normal_df(n: int = 60, outliers: int = 3) -> pd.DataFrame:
    """Small synthetic numeric dataset: n normal rows + a few extreme ones."""
    rng = np.random.default_rng(0)
    values = list(rng.normal(50, 3, n - outliers)) + [500.0] * outliers
    return pd.DataFrame({
        "reading": [f"{v:.2f}" for v in values],
        "label": [f"sample-{i}" for i in range(n)],
    })


def test_too_few_rows_is_skipped_with_reason():
    result = run_anomaly_detection(_normal_df(10))
    assert result["status"] == "skipped"
    assert "rows" in result["reason"]
    assert result["flagged_rows"] == []


def test_no_numeric_features_is_skipped():
    df = pd.DataFrame({"a": [f"txt-{i}" for i in range(60)]})
    result = run_anomaly_detection(df)
    assert result["status"] == "skipped"
    assert "numeric" in result["reason"]


def test_constant_feature_is_dropped():
    df = pd.DataFrame({
        "const": ["5"] * 61,
        "vary": [str(i % 5) for i in range(61)],
    })
    features, matrix = pick_features(pd.DataFrame(df))
    assert "const" not in features
    assert len(features) <= MAX_FEATURES


def test_detection_flags_extreme_rows_and_explains_them():
    df = _normal_df()
    result = run_anomaly_detection(df)
    assert result["status"] == "ok"
    assert result["model"] == "IsolationForest"
    assert result["random_seed"] == RANDOM_SEED
    assert "reading" in result["features_used"]
    assert result["flagged_count"] >= 1
    # extreme rows (last 3) should be among the flagged, highest first
    rows = {r["row"] for r in result["flagged_rows"]}
    assert rows & {58, 59, 60}
    top = result["flagged_rows"][0]
    assert any(f["feature"] == "reading" and abs(f["z"]) > 1
               for f in top["top_features"])
    assert top["summary"]
    # distribution data is complete
    assert result["histogram"] and sum(b["count"] for b in result["histogram"]) == df.shape[0]
    assert result["score_stats"]["max"] >= result["score_stats"]["min"]
    assert result["limitations"]


def test_reproducible_with_fixed_seed():
    df = _normal_df()
    a = run_anomaly_detection(df)
    b = run_anomaly_detection(df)
    assert [r["row"] for r in a["flagged_rows"]] == \
        [r["row"] for r in b["flagged_rows"]]
    assert a["flagged_count"] == b["flagged_count"]


def test_contamination_0_1_flags_about_10_percent():
    df = _normal_df(80, 2)
    result = run_anomaly_detection(df, contamination=0.1)
    expected = round(df.shape[0] * 0.1)
    assert abs(result["flagged_count"] - expected) <= 2


def test_anomalies_never_touch_deterministic_findings():
    """Scanning works the same regardless of the anomaly module."""
    df = _df = pd.DataFrame({
        _n: [str(i % 7) for i in range(60)]
        for _n in ("num", "flag")
    })
    report = scan_df(df)
    before = report.finding_total
    run_anomaly_detection(df)
    assert report.finding_total == before
    assert report.ml["status"] == "not_implemented"


def test_scan_report_stays_ml_free():
    rep = scan_bytes(b"num,flag\n1,yes\n2,no\n", filename="mini.csv")
    assert rep.ml["status"] == "not_implemented"
    assert "deterministic rules only" in rep.ml["note"]
