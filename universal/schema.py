"""Data structures for the universal CSV data-quality scanner.

Everything here is column-agnostic: no retail (or any domain) assumptions.
Severity levels match the existing ``src.issue_registry`` vocabulary
(INFO / WARNING / CRITICAL) so both scanners speak the same language.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field, asdict
from typing import Optional

SEVERITIES = ("INFO", "WARNING", "CRITICAL")

# data-quality dimensions reported per scan (all 0-100, higher is better)
DIMENSIONS = ("completeness", "uniqueness", "validity", "consistency", "plausibility")

_counter = itertools.count(1)


def reset_ids() -> None:
    """Test helper: restart finding numbering."""
    global _counter
    _counter = itertools.count(1)


@dataclass
class Finding:
    """One detected data-quality issue, with row-level evidence.

    ``row`` is the 1-based data row number (header not counted), or None
    for column-level findings (e.g. 'column is 40% missing').
    """
    finding_id: str
    rule_id: str
    column: str
    severity: str
    message: str
    row: Optional[int] = None
    value: Optional[str] = None
    evidence: str = ""
    suggested_action: str = ""
    dimension: str = "validity"   # which DQ dimension this finding hurts

    def to_dict(self) -> dict:
        return asdict(self)


def make_finding(rule_id: str, column: str, severity: str, message: str,
                 *, row: int | None = None, value: str | None = None,
                 evidence: str = "", suggested_action: str = "",
                 dimension: str = "validity") -> Finding:
    if severity not in SEVERITIES:
        raise ValueError(f"bad severity {severity!r}; expected one of {SEVERITIES}")
    if dimension not in DIMENSIONS:
        raise ValueError(f"bad dimension {dimension!r}; expected one of {DIMENSIONS}")
    if row is not None:
        shown = "" if value is None else repr(value)
        evidence = evidence or f"row {row}: value={shown}"
    return Finding(
        finding_id=f"F-{next(_counter):05d}",
        rule_id=rule_id,
        column=column,
        severity=severity,
        message=message,
        row=row,
        value=value,
        evidence=evidence,
        suggested_action=suggested_action,
        dimension=dimension,
    )


@dataclass
class ColumnProfile:
    """Automatic profile of a single column (name-agnostic)."""
    name: str
    inferred_type: str          # integer|float|numeric|boolean|date|datetime|text|empty
    non_null: int = 0
    missing: int = 0            # truly empty fields
    blank: int = 0              # whitespace-only fields
    placeholder: int = 0        # N/A, null, -, unknown, ...
    unique: int = 0
    unique_pct: float = 0.0
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    top_values: list = field(default_factory=list)   # [(value, count), ...] top 5
    samples: list = field(default_factory=list)      # first 3 non-blank values
    mixed_types: dict = field(default_factory=dict)  # family -> count (when mixed)
    date_format: Optional[str] = None                # dominant date format, if any
    id_like: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScanReport:
    """Full result of scanning one CSV file. Fully serialisable."""
    scan_id: str
    filename: str
    generated_at: str
    rows: int
    columns: list                       # list[ColumnProfile]
    findings: list                      # list[dict] (Finding.to_dict())
    dimensions: dict                    # dimension -> 0..100
    severity_counts: dict               # INFO/WARNING/CRITICAL -> count
    detectors: list                     # rule ids that ran
    ml: dict                            # status of the (future) ML detector slot
    notes: list = field(default_factory=list)
    finding_total: int = 0              # total before any per-rule cap
    truncated: bool = False             # True if findings were capped

    def to_dict(self) -> dict:
        return asdict(self)
