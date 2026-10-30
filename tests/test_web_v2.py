"""Tests for the v2 dashboard endpoints: history, anomalies, baselines,
comparison, remediation, settings and static assets.

All uses mock data — no scan results are treated as production data.
"""
import io
from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from universal.web import build_app, router

DATA = Path(__file__).resolve().parent.parent / "data"
HEALTH = DATA / "sample_universal_health.csv"


def _big_csv(n: int = 60) -> bytes:
    buf = io.StringIO()
    buf.write("value,label\n")
    for i in range(n):
        buf.write(f"{'0.5' if i < n - 3 else '500'},l{i}\n")
    return buf.getvalue().encode()


@pytest.fixture()
def client():
    app = build_app()
    with TestClient(app) as c:
        yield c


def _scan(client, path=HEALTH):
    with open(path, "rb") as fh:
        r = client.post("/api/universal/scan",
                        files={"file": (path.name, fh, "text/csv")})
    assert r.status_code == 200
    return r.json()


def test_scan_response_includes_preview_and_release_state(client):
    d = _scan(client)
    assert "preview" in d and d["preview"]["rows"]
    assert d["release_state"] in ("PASS", "PASS WITH WARNINGS", "BLOCKED")
    assert isinstance(d["overall_score"], (int, float))
    assert 0 <= d["overall_score"] <= 100


def test_scan_config_json_is_accepted(client):
    with open(HEALTH, "rb") as fh:
        r = client.post("/api/universal/scan",
                        files={"file": ("h.csv", fh, "text/csv")},
                        data={"config": '{"row_evidence_cap": 3}'})
    assert r.status_code == 200
    with open(HEALTH, "rb") as fh:
        r2 = client.post("/api/universal/scan",
                         files={"file": ("h.csv", fh, "text/csv")})
    d1, d2 = r.json(), r2.json()
    # a tighter evidence cap must reduce stored row-level findings
    assert d1["finding_total"] != 0 and d1["finding_total"] <= d2["finding_total"]


def test_scans_history_and_full_report(client):
    d = _scan(client)
    hist = client.get("/api/universal/scans").json()
    assert any(h["scan_id"] == d["scan_id"] for h in hist)
    full = client.get(f"/api/universal/scans/{d['scan_id']}").json()
    assert full["rows"] == d["rows"] and "preview" in full
    assert client.delete(f"/api/universal/scans/{d['scan_id']}").json()["deleted"]
    assert client.get(f"/api/universal/scans/{d['scan_id']}").status_code == 404


def test_sample_data_matches_bundled_csv(client):
    r = client.get("/api/universal/sample-data")
    assert r.status_code == 200
    assert "attachment" in r.headers["content-disposition"]
    assert r.content.strip().startswith(b"patient_id,visit_date")


def test_anomaly_endpoint_skips_small_dataset_with_reason(client):
    d = _scan(client)          # 21 rows -> below MIN_ROWS=50
    r = client.post(f"/api/universal/scan/{d['scan_id']}/anomalies")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "skipped" and "rows" in body["reason"]
    # cached result retrievable
    assert client.get(f"/api/universal/scan/{d['scan_id']}/anomalies").json()["scan_id"] == d["scan_id"]
    # a suitable larger fake dataset runs end-to-end
    r2 = client.post("/api/universal/scan",
                     files={"file": ("big.csv", _big_csv(60), "text/csv")})
    big = r2.json()
    ra = client.post(f"/api/universal/scan/{big['scan_id']}/anomalies?contamination=0.05")
    assert ra.status_code == 200
    a = ra.json()
    if a["status"] == "ok":    # sklearn present
        assert a["flagged_count"] >= 1
        assert "value" in a["features_used"]
        assert a["limitations"]


def test_anomaly_never_found_before_run(client):
    d = _scan(client)
    assert client.get(f"/api/universal/scan/{d['scan_id']}/anomalies").status_code == 404
    assert client.get("/api/universal/scan/USCAN-0000000/anomalies").status_code == 404


def test_baseline_save_list_compare_and_delete(client):
    d = _scan(client)
    rb = client.post(f"/api/universal/baselines/from-scan?scan_id={d['scan_id']}")
    assert rb.status_code == 200
    b = rb.json()
    assert b["baseline_id"] and b["scan_id"] == d["scan_id"]
    assert any(x["baseline_id"] == b["baseline_id"]
               for x in client.get("/api/universal/baselines").json())

    cmp = client.get("/api/universal/compare",
                     params={"scan_id": d["scan_id"], "baseline_id": b["baseline_id"]})
    assert cmp.status_code == 200
    body = cmp.json()
    assert "signals" in body and "thresholds" in body
    assert "descriptive" in body["note"] or "review" in body["note"]

    assert client.delete(f"/api/universal/baselines/{b['baseline_id']}").json()["deleted"]
    assert client.get("/api/universal/compare",
                      params={"scan_id": d["scan_id"], "baseline_id": b["baseline_id"]}).status_code == 404
    assert client.get("/api/universal/compare",
                      params={"scan_id": d["scan_id"], "baseline_id": "nope"}).status_code == 404


def test_baseline_upload_endpoint_rejects_empty(client):
    r = client.post("/api/universal/baselines",
                    files={"file": ("e.csv", b"", "text/csv")})
    assert r.status_code == 422


def test_remediation_flow_end_to_end(client):
    d = _scan(client)
    rem = client.get(f"/api/universal/scan/{d['scan_id']}/remediation").json()
    assert rem["note"] and "original" in rem["note"]
    suggestions = rem["suggestions"]
    certain = [s["suggestion_id"] for s in suggestions if s["confidence"] == "certain"]
    assert suggestions                      # the health sample has fixable issues
    assert all(s["status"] == "pending" for s in suggestions)

    if certain:
        apply = client.post(f"/api/universal/scan/{d['scan_id']}/remediation/apply",
                            json={"accepted_ids": certain})
        assert apply.status_code == 200
        body = apply.json()
        assert body["applied"] >= 1
        assert body["cleaned_ready"]
        assert all(e.get("reason") for e in body["changelog"])

        # cleaned copy downloads
        rc = client.get(f"/api/universal/scan/{d['scan_id']}/cleaned.csv")
        assert rc.status_code == 200 and "attachment" in rc.headers["content-disposition"]
        df_clean = pd.read_csv(io.BytesIO(rc.content), dtype=str, keep_default_na=False)

        # changelogs
        for fmt, token in (("csv", "suggestion_id"), ("json", "\"changes\"")):
            rch = client.get(f"/api/universal/scan/{d['scan_id']}/changelog.{fmt}")
            assert rch.status_code == 200
            assert token.encode() in rch.content or token in rch.text

    # unknown suggestion ids rejected
    r_bad = client.post(f"/api/universal/scan/{d['scan_id']}/remediation/apply",
                        json={"accepted_ids": ["S-99999"]})
    assert r_bad.status_code == 422


def test_source_files_untouched_by_whole_flow(client):
    before = HEALTH.read_bytes()
    d = _scan(client)
    rem = client.get(f"/api/universal/scan/{d['scan_id']}/remediation").json()
    certain = [s["suggestion_id"] for s in rem["suggestions"]
               if s["confidence"] == "certain"]
    if certain:
        client.post(f"/api/universal/scan/{d['scan_id']}/remediation/apply",
                    json={"accepted_ids": certain})
        client.get(f"/api/universal/scan/{d['scan_id']}/cleaned.csv")
    client.post(f"/api/universal/baselines/from-scan?scan_id={d['scan_id']}")
    assert HEALTH.read_bytes() == before


def test_cleaned_download_before_apply_is_404(client):
    d = _scan(client)
    assert client.get(f"/api/universal/scan/{d['scan_id']}/cleaned.csv").status_code == 404
    assert client.get(f"/api/universal/scan/{d['scan_id']}/changelog.csv").status_code == 404


def test_settings_reports_configuration_without_secrets(client):
    s = client.get("/api/universal/settings").json()
    assert set(s) >= {"gemini_configured", "gemini_role", "gemini_model",
                      "anomaly_available", "versions", "privacy", "supported_formats"}
    assert s["ai_provider"].startswith("Google Gemini")
    # never leaks a key value: only short status labels are reported
    for v in s.values():
        if isinstance(v, str):
            assert not v.startswith("xai-") and len(v) < 200
    assert s["gemini_role"].startswith("advisory")
    assert s["gemini_model"]
    assert "xlsx" in s["supported_formats"]


def test_static_assets_served(client):
    r = client.get("/static/styles.css")
    assert r.status_code == 200 and "text/css" in r.headers["content-type"]
    r = client.get("/static/app.js")
    assert r.status_code == 200 and "javascript" in r.headers["content-type"]
    assert client.get("/static/nope.txt").status_code == 404


def test_index_serves_dashboard_html(client):
    html = client.get("/universal").text
    assert "Universal CSV Data-Quality Scanner" in html
    assert "never modified" in html
    assert "static/styles.css" in html and "static/app.js" in html
