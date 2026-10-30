"""Tests for the universal scanner web interface (upload -> scan -> download)."""
import pathlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from universal.web import build_app, router

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
HEALTH = DATA / "sample_universal_health.csv"


@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as c:
        yield c


def test_index_page_served(client):
    r = client.get("/universal")
    assert r.status_code == 200
    assert "Universal CSV Data-Quality Scanner" in r.text
    assert "never modified" in r.text


def test_standalone_app_serves_ui_at_root():
    app = build_app()
    with TestClient(app) as c:
        assert c.get("/").status_code == 200
        assert c.get("/universal").status_code == 200


def test_scan_endpoint_returns_report(client):
    with HEALTH.open("rb") as fh:
        r = client.post("/api/universal/scan",
                        files={"file": ("sample_universal_health.csv", fh, "text/csv")})
    assert r.status_code == 200
    d = r.json()
    assert d["rows"] == 21
    assert d["finding_total"] > 0
    assert d["ml"]["status"] == "not_implemented"
    assert set(d["dimensions"]) == {"completeness", "uniqueness", "validity",
                                    "consistency", "plausibility"}


def test_download_reports_all_formats(client):
    with HEALTH.open("rb") as fh:
        d = client.post("/api/universal/scan",
                        files={"file": ("h.csv", fh, "text/csv")}).json()
    for fmt, token in (("json", b"dimensions"), ("csv", b"finding_id"),
                       ("html", b"Data quality report")):
        r = client.get(f"/api/universal/scan/{d['scan_id']}/report.{fmt}")
        assert r.status_code == 200
        assert token in r.content
        assert "attachment" in r.headers["content-disposition"]


def test_empty_upload_rejected(client):
    r = client.post("/api/universal/scan", files={"file": ("e.csv", b"", "text/csv")})
    assert r.status_code == 422


def test_missing_scan_download_404(client):
    r = client.get("/api/universal/scan/USCAN-0000000/report.json")
    assert r.status_code == 404
