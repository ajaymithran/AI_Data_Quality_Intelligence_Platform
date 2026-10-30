"""Tests for the universal CSV data-quality scanner.

Covers: value-based type inference, every built-in detector, dimensions,
report exports, the read-only guarantee, and the modular detector
interface (including the future-ML slot, which must stay silent).
"""
import hashlib
import pathlib

import pandas as pd
import pytest

from universal import (
    DIMENSIONS, Detector, MLAnomalyDetector, ScanContext, register_detector,
    render, reset_registry, scan_bytes, scan_csv, scan_df, registered_detectors,
)
from universal.inference import classify_value, profile_column
from universal.schema import reset_ids

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
HEALTH = DATA / "sample_universal_health.csv"


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_registry()
    reset_ids()
    yield
    reset_registry()
    reset_ids()


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).fillna("").astype(str)


def _by_rule(report, rule_id):
    return [f for f in report.findings if f["rule_id"] == rule_id]


# ---------------------------------------------------------------- inference

@pytest.mark.parametrize("value,family", [
    ("", "empty"),
    ("   ", "blank"),
    ("N/A", "placeholder"),
    (" - ", "placeholder"),
    ("unknown", "placeholder"),
    ("42", "integer"),
    ("-7", "integer"),
    ("3.14", "float"),
    ("yes", "boolean"),
    ("NO", "boolean"),
    ("2026-01-05", "date"),
    ("05/01/2026", "date"),
    ("2026-01-05 10:30:00", "datetime"),
    ("hello", "text"),
    ("2026-13-45", "text"),          # impossible date -> just text
    ("12abc", "text"),
])
def test_classify_value_families(value, family):
    assert classify_value(value) == family


def test_type_inference_ignores_column_names():
    """A column called 'temperature_c' holding text must be typed as text,
    and one called 'notes' holding numbers must be typed as numbers."""
    df = _df([{"temperature_c": v} for v in ("hello", "world", "!", "??", "ok")])
    prof, _ = profile_column("temperature_c", df["temperature_c"])
    assert prof.inferred_type == "text"

    df2 = _df([{"notes": str(i)} for i in range(1, 6)])
    prof2, _ = profile_column("notes", df2["notes"])
    assert prof2.inferred_type == "integer"


def test_mixed_type_profile_records_minorities():
    df = _df([{"x": str(i)} for i in range(98)] + [{"x": "abc"}, {"x": "def"}])
    prof, _ = profile_column("x", df["x"])
    assert prof.inferred_type == "integer"
    assert prof.mixed_types == {"integer": 98, "text": 2}


def test_id_like_detection():
    df = _df([{"k": f"ID-{i}"} for i in range(10)])
    prof, _ = profile_column("k", df["k"])
    assert prof.id_like is True


# ---------------------------------------------------------------- detectors

def test_missing_values_detected_with_row_evidence():
    df = _df([{"a": "1"}, {"a": ""}, {"a": "3"}])
    rep = scan_df(df, filename="t.csv")
    hits = [f for f in _by_rule(rep, "MISSING_VALUES") if f["row"] is not None]
    assert len(hits) == 1
    assert hits[0]["row"] == 2 and hits[0]["severity"] in ("WARNING", "CRITICAL")
    assert hits[0]["dimension"] == "completeness"


def test_blank_strings_detected():
    df = _df([{"a": "1"}, {"a": "   "}, {"a": "x"}])
    hits = _by_rule(scan_df(df), "BLANK_STRINGS")
    assert len(hits) == 1 and hits[0]["row"] == 2


def test_placeholder_values_detected():
    df = _df([{"a": "1"}, {"a": "N/A"}, {"a": "null"}, {"a": "-"}])
    assert len(_by_rule(scan_df(df), "PLACEHOLDER_VALUES")) == 3


def test_duplicate_rows_detected_with_twin_reference():
    rows = [{"a": "1", "b": "x"}, {"a": "2", "b": "y"},
            {"a": "1", "b": "x"}, {"a": "3", "b": "z"}]
    hits = _by_rule(scan_df(_df(rows)), "DUPLICATE_ROWS")
    assert len(hits) == 1
    assert hits[0]["row"] == 3 and "rows 1 and 3" in hits[0]["evidence"]


def test_inconsistent_types_flagged():
    rows = [{"v": str(i)} for i in range(98)] + [{"v": "abc"}, {"v": "!!"}]
    hits = _by_rule(scan_df(_df(rows)), "INCONSISTENT_TYPES")
    assert {h["row"] for h in hits} == {99, 100}
    assert all(h["dimension"] == "consistency" for h in hits)


def test_invalid_date_format_flagged():
    rows = [{"d": f"2026-01-{i:02d}"} for i in range(1, 10)] + [{"d": "2026-02-31T09:00"}]
    hits = _by_rule(scan_df(_df(rows)), "INVALID_FORMAT")
    assert any(h["row"] == 10 for h in hits)


def test_boolean_vocabulary_violation():
    rows = [{"b": "yes"}, {"b": "no"}, {"b": "yes"}, {"b": "no"}, {"b": "MAYBE"}]
    hits = [f for f in _by_rule(scan_df(_df(rows)), "INVALID_FORMAT") if f["row"] == 5]
    assert hits and "boolean" in hits[0]["message"]


def test_impossible_calendar_date_in_text_column():
    df = _df([{"t": v} for v in ("alpha", "beta", "gamma", "delta", "2026-13-45")])
    hits = [f for f in _by_rule(scan_df(df), "INVALID_FORMAT") if f["row"] == 5]
    assert hits and "calendar" in hits[0]["message"]


def test_padded_whitespace_flagged():
    df = _df([{"s": "ok"}, {"s": " ok"}, {"s": "ok "}, {"s": "ok"}, {"s": "ok"}])
    hits = _by_rule(scan_df(df), "PADDED_WHITESPACE")
    assert {h["row"] for h in hits} == {2, 3}


def test_case_variants_flagged():
    df = _df([{"w": "ICU"}, {"w": "ICU"}, {"w": "icu"}, {"w": "ward"}, {"w": "ward"}])
    hits = _by_rule(scan_df(df), "CASE_VARIANTS")
    assert [h["row"] for h in hits] == [3]


def test_case_variants_skips_high_cardinality_text():
    df = _df([{"note": f"unique note {i}"} for i in range(10)])
    assert _by_rule(scan_df(df), "CASE_VARIANTS") == []


def test_outlier_detection():
    vals = [10, 11, 9, 12, 10, 11, 9, 12, 10, 999]
    hits = [f for f in _by_rule(scan_df(_df([{"n": str(v)} for v in vals])),
                                "SUSPICIOUS_VALUES") if f["row"] == 10]
    assert hits and hits[0]["severity"] == "INFO"


def test_future_date_flagged():
    df = _df([{"d": "2026-01-01"}, {"d": "2999-12-31"}])
    hits = [f for f in _by_rule(scan_df(df), "SUSPICIOUS_VALUES") if f["row"] == 2]
    assert hits and "future" in hits[0]["message"]


def test_evidence_cap_appends_summary():
    cap = 5
    df = _df([{"c": ""} for _ in range(50)])
    rep = scan_df(df, config={"row_evidence_cap": cap})
    row_hits = [f for f in _by_rule(rep, "MISSING_VALUES") if f["row"] is not None]
    assert len(row_hits) == cap
    summary = [f for f in _by_rule(rep, "MISSING_VALUES")
               if f["row"] is None and "suppressed" in f["evidence"]]
    assert summary and str(50 - cap) in summary[0]["message"]


# ---------------------------------------------------------------- report

def test_report_structure_and_dimensions():
    d = scan_csv(HEALTH).to_dict()
    assert set(d["dimensions"]) == set(DIMENSIONS)
    assert all(0 <= v <= 100 for v in d["dimensions"].values())
    assert d["rows"] == 21
    assert len(d["columns"]) == 8
    assert set(d["severity_counts"]) == {"INFO", "WARNING", "CRITICAL"}
    assert d["ml"]["status"] == "not_implemented"
    assert "deterministic rules only" in d["ml"]["note"]


def test_no_ml_claims_in_rendered_reports():
    rep = scan_csv(HEALTH)
    for fmt in ("json", "html"):
        payload, _, _ = render(rep, fmt)
        text = payload.decode("utf-8").lower()
        assert "machine learning" not in text
        assert "model accuracy" not in text


def test_health_sample_detects_every_issue_class():
    rules = {f["rule_id"] for f in scan_csv(HEALTH).findings}
    assert "MISSING_VALUES" in rules          # empty glucose (P-015), empty ward (P-020)
    assert "BLANK_STRINGS" in rules           # whitespace-only notes (P-009)
    assert "PLACEHOLDER_VALUES" in rules      # N/A, unknown
    assert "DUPLICATE_ROWS" in rules          # P-005 twice
    assert "INCONSISTENT_TYPES" in rules      # age='thirty', 'sixty'
    assert "INVALID_FORMAT" in rules          # visit_date=2026-13-45, discharged=MAYBE
    assert "PADDED_WHITESPACE" in rules       # '   follow-up'
    assert "CASE_VARIANTS" in rules           # ICU/icu, Ward-A/ward-a
    assert "SUSPICIOUS_VALUES" in rules       # temperature -4.0, future date


def test_render_all_formats():
    rep = scan_csv(HEALTH)
    for fmt, expect in (("json", b"dimensions"), ("csv", b"finding_id"),
                        ("html", b"Data quality report")):
        payload, media, name = render(rep, fmt)
        assert expect in payload
        assert name.endswith(f".{fmt}")
    with pytest.raises(ValueError):
        render(rep, "pdf")


def test_source_file_never_modified():
    before = hashlib.sha256(HEALTH.read_bytes()).hexdigest()
    scan_csv(HEALTH)
    scan_csv(HEALTH.read_bytes())
    assert hashlib.sha256(HEALTH.read_bytes()).hexdigest() == before


def test_scan_bytes_and_empty_csv_errors():
    rep = scan_bytes(b"a,b\n1,2\n3,4\n", filename="mini.csv")
    assert rep.rows == 2 and rep.filename == "mini.csv"
    with pytest.raises(ValueError):
        scan_bytes(b"")
    with pytest.raises(ValueError):
        scan_bytes(b",,\n")


def test_duplicate_column_names_renamed_not_crashing():
    rep = scan_bytes(b"x,x\n1,2\n3,4\n")
    assert [c.name for c in rep.columns] == ["x", "x_2"]
    assert any("renamed" in n for n in rep.notes)


# ---------------------------------------------------------------- modular interface

def test_custom_detector_plugs_in_and_runs():
    class RowCountDetector(Detector):
        rule_id = "CUSTOM_ROW_COUNT"

        def run(self, ctx):
            from universal.schema import make_finding
            return [make_finding(self.rule_id, "(dataset)", "INFO",
                                 f"{ctx.rows} rows", dimension="plausibility")]

    register_detector(RowCountDetector())
    assert any(d.rule_id == "CUSTOM_ROW_COUNT" for d in registered_detectors())
    rep = scan_df(_df([{"a": "1"}, {"b": "2"}]))
    assert any(f["rule_id"] == "CUSTOM_ROW_COUNT" for f in rep.findings)


def test_re_registering_same_rule_id_replaces_not_duplicates():
    class A(Detector):
        rule_id = "DUP_TEST"

        def run(self, ctx):
            return []

    register_detector(A())
    register_detector(A())
    assert sum(1 for d in registered_detectors() if d.rule_id == "DUP_TEST") == 1


def test_ml_detector_slot_is_silent_and_labeled():
    df = _df([{"a": "1"}, {"a": ""}, {"a": "99999"}])
    ctx = ScanContext(df=df, profiles=[], config={})
    assert MLAnomalyDetector().run(ctx) == []          # no ML runs in this version
    rep = scan_df(df)
    assert rep.ml["status"] == "not_implemented"
    assert MLAnomalyDetector in [type(d) for d in registered_detectors()] or \
        not any(f["rule_id"] == "ML_ANOMALY" for f in rep.findings)


# ---------------------------------------------------------------- retail regression

def test_retail_samples_still_scan_via_universal_scanner():
    """The old demo CSVs must also work — universality includes them."""
    for name in ("sample_valid.csv", "sample_corrupted.csv"):
        rep = scan_csv(DATA / name)
        assert rep.rows > 0 and len(rep.columns) == 8
        assert all(0 <= v <= 100 for v in rep.dimensions.values())
