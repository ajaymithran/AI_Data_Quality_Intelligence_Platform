"""Smoke tests for the universal scanner CLI."""
import pathlib

import pytest

from universal.cli import main

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
HEALTH = DATA / "sample_universal_health.csv"


def test_cli_summary_runs(capsys):
    code = main([str(HEALTH)])
    out = capsys.readouterr().out
    assert code == 0
    assert "dimensions" in out
    assert "completeness" in out
    assert "person_id" in out or "patient_id" in out


def test_cli_json_stdout(capsysbinary):
    code = main([str(HEALTH), "--format", "json"])
    assert code == 0
    assert b"dimensions" in capsysbinary.readouterr().out


def test_cli_save_writes_all_reports(tmp_path):
    code = main([str(HEALTH), "--save", str(tmp_path), "--format", "summary"])
    assert code == 0
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["sample_universal_health_quality_report.csv",
                     "sample_universal_health_quality_report.html",
                     "sample_universal_health_quality_report.json"]


def test_cli_fail_on_critical(tmp_path):
    assert main([str(HEALTH), "--format", "summary", "--fail-on", "critical"]) == 1
    assert main([str(HEALTH), "--format", "summary", "--fail-on", "never"]) == 0


def test_cli_missing_file(capsys):
    assert main(["definitely/not/here.csv"]) == 2
    assert "file not found" in capsys.readouterr().err
