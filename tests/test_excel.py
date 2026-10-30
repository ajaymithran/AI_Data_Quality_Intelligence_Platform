"""Tests for Excel (.xlsx) ingestion — same scan path, values-only."""
import io

import openpyxl
import pytest

from universal import scan_bytes
from universal.scanner import read_raw_df


def _xlsx(rows):
    wb = openpyxl.Workbook()
    ws = wb.active
    for row in rows:
        ws.append(row)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def test_excel_scan_and_read():
    data = _xlsx([
        ["num", "when", "flag"],
        [42, "2026-01-05", "yes"],
        [3.5, "2026-01-06", "no"],
        [100.0, "n/a", "  "],      # row with loose typing
    ])
    rep = scan_bytes(data, filename="book.xlsx")
    assert rep.rows == 3
    assert rep.notes and any("excel" in n.lower() for n in rep.notes)
    assert [c.name for c in rep.columns] == ["num", "when", "flag"]
    df = read_raw_df(data, "book.xlsx")
    assert df.iat[2, 0] == "100"          # integral float rendered cleanly
    assert df.iat[2, 1] == "n/a"          # placeholder not swallowed by na_values
    assert df.iat[2, 2] == "  "           # raw cell preserved as string


def test_excel_findings_match_csv_equivalent():
    xlsx = _xlsx([
        ["value"], [10], [11], [12], [10], ["bad"],
    ])
    rep = scan_bytes(xlsx, filename="book.xlsx")
    rules = {f["rule_id"] for f in rep.findings}
    assert rules                       # detectors run on excel data too


def test_excel_empty_errors():
    wb = openpyxl.Workbook()
    buf = io.BytesIO()
    wb.save(buf)
    with pytest.raises(ValueError):
        read_raw_df(buf.getvalue(), "book.xlsx")
