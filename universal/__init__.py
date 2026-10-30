"""Universal CSV data-quality scanner.

Works on ANY CSV: column types are inferred from values (never from
names), every column is inspected, and the source file is never modified.

Quick start::

    from universal import scan_csv, render

    report = scan_csv("any_file.csv")          # path, bytes, or DataFrame
    print(report.dimensions)                   # completeness, uniqueness, ...
    payload, media, name = render(report, "html")

Extension point for the future ML detector::

    from universal import Detector, register_detector

    class MyMLDetector(Detector):
        rule_id = "MY_ML_CHECK"
        def run(self, ctx): ...

    register_detector(MyMLDetector())

No machine learning ships in this version.
"""
from .context import DEFAULT_CONFIG, ScanContext
from .detectors import (Detector, MLAnomalyDetector, register_detector,
                        registered_detectors, reset_registry)
from .report import render, to_csv, to_html, to_json
from .scanner import read_raw_df, scan_bytes, scan_csv, scan_df
from .schema import DIMENSIONS, SEVERITIES, ColumnProfile, Finding, ScanReport
from .ai_analyst import analyze_with_gemini, analyze_with_grok
from .ai_schemas import AIAnalysisReport, IssueExplanation, ReviewableSuggestion, OverallAISummary
from .anomaly import run_anomaly_detection
from .drift import compare as compare_baselines
from .remediation import apply_changes, build_suggestions

__all__ = [
    "scan_csv", "scan_bytes", "scan_df", "read_raw_df",
    "ScanReport", "Finding", "ColumnProfile",
    "SEVERITIES", "DIMENSIONS", "DEFAULT_CONFIG", "ScanContext",
    "Detector", "MLAnomalyDetector", "register_detector",
    "registered_detectors", "reset_registry",
    "render", "to_json", "to_csv", "to_html",
    "analyze_with_gemini", "analyze_with_grok", "AIAnalysisReport", "IssueExplanation",
    "ReviewableSuggestion", "OverallAISummary",
    "run_anomaly_detection", "compare_baselines",
    "build_suggestions", "apply_changes",
]
