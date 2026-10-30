"""Tests for baseline comparison / drift detection (deterministic stats)."""
import io

import pytest

from universal.scanner import scan_bytes
from universal.drift import compare, _ks_statistic


def _csv(rows):
    buf = io.StringIO()
    buf.write(",".join(rows[0]) + "\n")
    for r in rows[1:]:
        buf.write(",".join(str(v) for v in r) + "\n")
    return buf.getvalue().encode("utf-8")


def _make(data):
    """(report, df) pair via the normal scan path."""
    blob = data.encode() if isinstance(data, str) else data
    report = scan_bytes(blob, filename="ds.csv")
    from universal.scanner import read_raw_df
    return report, read_raw_df(blob, "ds.csv")


def test_ks_identical_and_disjoint():
    import numpy as np
    a = np.array([1.0, 2.0, 3.0])
    assert _ks_statistic(a, a.copy()) == 0.0
    b = np.array([10.0, 11.0, 12.0])
    assert _ks_statistic(a, a) == 0.0
    assert _ks_statistic(a, b) == 1.0
    assert _ks_statistic(a, np.array([4.0, 5.0])) > 0.5


def test_column_changes_detected():
    base = _csv([
        ["a", "keep", "gone"],
        ["1", "x", "1"], ["2", "y", "2"], ["3", "x", "3"],
    ])
    curr = _csv([
        ["a", "keep", "brand_new", "num"],
        ["1", "x", "5", "1"], ["2", "y", "6", "2"],
    ])
    b_rep, b_df = _make(base)
    c_rep, c_df = _make(curr)
    cmp = compare(b_rep, b_df, c_rep, c_df)
    assert cmp["columns_added"] == ["brand_new", "num"]
    assert cmp["columns_removed"] == ["gone"]
    assert cmp["columns_added"] != []
    assert any(sig["kind"] == "column_removed" for sig in cmp["signals"])


def test_type_change_and_missing_rate_shift():
    base = _csv([
        ["num", "other"],
        ["10", "x"], ["20", "y"], ["30", "x"],
    ])
    curr = _csv([
        ["num", "other"],
        ["alpha", "x"], ["beta", "y"], ["gamma", "x"],
    ])
    b_rep, b_df = _make(base)
    c_rep, c_df = _make(curr)
    cmp = compare(b_rep, b_df, c_rep, c_df)
    assert any(t["column"] == "num" for t in cmp["type_changes"])
    t = next(t for t in cmp["type_changes"] if t["column"] == "num")
    assert t["baseline"] != t["current"]


def test_missing_rate_delta_and_signals():
    base = _csv([["k", "pad"]] + [[str(i), "x"] for i in range(20)])
    curr = _csv([["k", "pad"]] + [[str(i) if i < 12 else "", "x"] for i in range(20)])
    b_rep, b_df = _make(base)
    c_rep, c_df = _make(curr)
    cmp = compare(b_rep, b_df, c_rep, c_df)
    m = cmp["missing_rates"][0]
    assert m["column"] == "k"
    assert m["baseline_pct"] == 0.0
    assert abs(m["delta_pp"] - 40.0) < 0.1
    assert any(s["kind"] == "missing_rate_changed" for s in cmp["signals"])


def test_numeric_distribution_shift_and_ks_signal():
    base = _csv([["v", "pad"]] + [[str(10 + i), "x"] for i in range(30)])
    curr = _csv([["v", "pad"]] + [[str(100 + i), "x"] for i in range(30)])
    b_rep, b_df = _make(base)
    c_rep, c_df = _make(curr)
    cmp = compare(b_rep, b_df, c_rep, c_df)
    assert cmp["numeric_distribution"], "expected numeric comparison"
    n = cmp["numeric_distribution"][0]
    assert n["ks_statistic"] == 1.0
    assert n["baseline"]["mean"] != n["current"]["mean"]
    assert any(s["kind"] == "distribution_shift" for s in cmp["signals"])


def test_note_and_meta_present():
    base = _csv([["a"], ["1"], ["2"]])
    curr = _csv([["a"], ["1"], ["2"]])
    b_rep, b_df = _make(base)
    c_rep, c_df = _make(curr)
    cmp = compare(b_rep, b_df, c_rep, c_df)
    assert "signals for review" in cmp["note"]
    assert cmp["baseline"]["scan_id"]
    assert cmp["current"]["scan_id"]
