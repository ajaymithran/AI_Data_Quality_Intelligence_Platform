"""Shared scan context passed to every detector.

Holds the raw (all-string) DataFrame plus pre-computed parsing work so
detectors never repeat expensive type/date parsing and never have to
assume anything about column names.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .schema import ColumnProfile


@dataclass
class ScanContext:
    #: raw CSV values as strings (missing cells are empty strings "")
    df: pd.DataFrame
    #: automatic per-column profiles (types, nulls, uniqueness, ...)
    profiles: list[ColumnProfile]
    #: parsed numeric values per column (NaN where not numeric)
    numeric: dict[str, pd.Series] = field(default_factory=dict)
    #: parsed datetime values per column, using the column's dominant format
    dates: dict[str, pd.Series] = field(default_factory=dict)
    #: value-family series per column (classify-once, shared by detectors)
    families: dict[str, pd.Series] = field(default_factory=dict)
    #: whitespace-stripped values per column (computed once)
    stripped: dict[str, pd.Series] = field(default_factory=dict)
    #: scan-level configuration (thresholds, evidence caps, ...)
    config: dict = field(default_factory=dict)

    def families_for(self, column: str) -> pd.Series:
        """Family series for a column, computed lazily if not supplied."""
        fam = self.families.get(column)
        if fam is None:
            from .inference import value_families
            fam = value_families(self.df[column])
            self.families[column] = fam
        return fam

    def stripped_for(self, column: str) -> pd.Series:
        """Whitespace-stripped values for a column, computed lazily."""
        ser = self.stripped.get(column)
        if ser is None:
            ser = self.df[column].str.strip()
            self.stripped[column] = ser
        return ser

    @property
    def rows(self) -> int:
        return len(self.df)

    @property
    def columns(self) -> list[str]:
        return list(self.df.columns)

    def profile(self, column: str) -> ColumnProfile:
        for p in self.profiles:
            if p.name == column:
                return p
        raise KeyError(column)

    @property
    def row_evidence_cap(self) -> int:
        """Max row-level findings kept per (rule, column) pair."""
        return int(self.config.get("row_evidence_cap", 25))


DEFAULT_CONFIG = {
    # fraction of a column's real values that must share one type family
    # before we consider the column's type "dominant"
    "type_dominance": 0.90,
    # minority families at or above this share are flagged as inconsistent
    "mixed_type_min_share": 0.01,
    "mixed_type_min_count": 2,
    # IQR fence multiplier for outlier detection (rule-of-thumb: 1.5-3.0)
    "outlier_iqr_k": 3.0,
    "outlier_min_values": 8,
    # a date at least this many days in the future is suspicious
    "future_date_days": 1,
    # column missing-rate above which the column itself is CRITICAL
    "critical_missing_rate": 0.30,
    # per (rule, column) cap on row-level findings
    "row_evidence_cap": 25,
}
