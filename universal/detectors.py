"""Detector framework: the modular interface for all checks.

A detector is any class implementing ``Detector.run(ctx) -> list[Finding]``.
Built-in detectors are deterministic rules (no ML, no training, no
"anomaly models"). The registry also keeps a slot for a future
ML-based detector: it implements the very same interface, so adding it
later requires zero changes elsewhere.

Performance: detectors reuse the per-column family series computed once
during inference (``ctx.families_for(col)``) and locate rows with
vectorised boolean masks, so nothing re-classifies values row by row.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .context import ScanContext
from .inference import BOOLEAN_TOKENS, is_impossible_date, temporal_shape
from .schema import Finding, make_finding


class Detector:
    """Base class for every detector (deterministic rules and future ML)."""

    rule_id: str = "UNKNOWN"

    def run(self, ctx: ScanContext) -> list[Finding]:  # pragma: no cover - interface
        raise NotImplementedError


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

_REGISTRY: list[Detector] = []


def register_detector(detector: Detector) -> None:
    """Add a detector to the scan pipeline (replaces same rule_id)."""
    _REGISTRY[:] = [d for d in _REGISTRY if d.rule_id != detector.rule_id]
    _REGISTRY.append(detector)


def registered_detectors() -> list[Detector]:
    return list(_REGISTRY)


def reset_registry() -> None:
    """Restore the default built-in detector set (used by tests)."""
    _REGISTRY.clear()
    for d in default_detectors():
        register_detector(d)


def default_detectors() -> list["Detector"]:
    return [
        MissingValuesDetector(),
        BlankStringDetector(),
        PlaceholderDetector(),
        DuplicateRowDetector(),
        InconsistentTypeDetector(),
        InvalidFormatDetector(),
        PaddedWhitespaceDetector(),
        CaseVariantDetector(),
        SuspiciousValueDetector(),
    ]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _rows_from_mask(series: pd.Series, mask) -> list[tuple[int, str]]:
    """(1-based data row number, raw value) for positions where mask is True."""
    mask_arr = np.asarray(mask, dtype=bool)
    positions = np.flatnonzero(mask_arr)
    if positions.size == 0:
        return []
    values = series.to_numpy()
    return [(int(p) + 1, values[p]) for p in positions]


def _evidence(findings: list[Finding], cap: int) -> list[Finding]:
    """Cap row-level findings per (rule, column), appending a summary."""
    counts: dict[str, int] = {}
    first: dict[str, Finding] = {}
    kept: list[Finding] = []
    for f in findings:
        if f.row is None:
            kept.append(f)
            continue
        col = f.column
        n = counts.get(col, 0) + 1
        counts[col] = n
        if n <= cap:
            kept.append(f)
            first.setdefault(col, f)
    for col, n in counts.items():
        overflow = n - cap
        if overflow > 0:
            meta = first[col]
            kept.append(make_finding(
                meta.rule_id, col, "INFO",
                f"{n} row-level findings in this column; "
                f"showing first {cap} ({overflow} more not listed)",
                evidence=f"{overflow} additional rows suppressed by evidence cap",
                dimension=meta.dimension,
            ))
    return kept


def fam_label(family: str) -> str:
    return {
        "integer": "number (integer)", "float": "number (decimal)",
        "boolean": "true/false flag", "date": "date", "datetime": "date+time",
        "text": "text", "placeholder": "placeholder token",
    }.get(family, family)


def dominant_count(prof, dominant: str) -> int:
    return prof.mixed_types.get(dominant, 0)


# --------------------------------------------------------------------------
# built-in deterministic detectors (no ML)
# --------------------------------------------------------------------------

class MissingValuesDetector(Detector):
    """Truly empty cells. Column-level CRITICAL when the rate is high."""

    rule_id = "MISSING_VALUES"

    def run(self, ctx: ScanContext) -> list[Finding]:
        crit_rate = ctx.config["critical_missing_rate"]
        out: list[Finding] = []
        for prof in ctx.profiles:
            if prof.missing == 0:
                continue
            rate = prof.missing / max(ctx.rows, 1)
            cell_sev = "CRITICAL" if rate >= crit_rate else "WARNING"
            s = ctx.df[prof.name]
            rows = _rows_from_mask(s, s == "")
            for row, _ in rows:
                out.append(make_finding(
                    self.rule_id, prof.name, cell_sev,
                    "missing value (empty cell)",
                    row=row, value="",
                    suggested_action="fill from source or drop the row",
                    dimension="completeness",
                ))
            out.append(make_finding(
                self.rule_id, prof.name, "WARNING",
                f"column is {rate * 100:.1f}% missing "
                f"({prof.missing}/{ctx.rows} cells empty)",
                dimension="completeness",
            ))
        return _evidence(out, ctx.row_evidence_cap)


class BlankStringDetector(Detector):
    """Values that look present but are only whitespace."""

    rule_id = "BLANK_STRINGS"

    def run(self, ctx: ScanContext) -> list[Finding]:
        out: list[Finding] = []
        for prof in ctx.profiles:
            if prof.blank == 0:
                continue
            s = ctx.df[prof.name]
            rows = _rows_from_mask(s, (s != "") & (ctx.stripped_for(prof.name) == ""))
            for row, val in rows:
                out.append(make_finding(
                    self.rule_id, prof.name, "WARNING",
                    "blank string (whitespace only)",
                    row=row, value=val,
                    suggested_action="treat as missing; trim and fill",
                    dimension="completeness",
                ))
        return _evidence(out, ctx.row_evidence_cap)


class PlaceholderDetector(Detector):
    """Tokens that masquerade as data: N/A, null, -, unknown, ..."""

    rule_id = "PLACEHOLDER_VALUES"

    def run(self, ctx: ScanContext) -> list[Finding]:
        out: list[Finding] = []
        for prof in ctx.profiles:
            if prof.placeholder == 0:
                continue
            s = ctx.df[prof.name]
            rows = _rows_from_mask(s, ctx.families_for(prof.name) == "placeholder")
            for row, val in rows:
                out.append(make_finding(
                    self.rule_id, prof.name, "WARNING",
                    f"placeholder token '{val.strip()}' recorded as a value",
                    row=row, value=val,
                    suggested_action="decide policy: treat as missing or map to a real category",
                    dimension="completeness",
                ))
        return _evidence(out, ctx.row_evidence_cap)


class DuplicateRowDetector(Detector):
    """Exact duplicate data rows (all columns equal)."""

    rule_id = "DUPLICATE_ROWS"

    def run(self, ctx: ScanContext) -> list[Finding]:
        dup_all = ctx.df.duplicated(keep=False)
        if not dup_all.any():
            return []
        positions = np.flatnonzero(dup_all.to_numpy())
        sub = ctx.df.iloc[positions]
        first_seen: dict[tuple, int] = {}
        out: list[Finding] = []
        for local, row in enumerate(sub.itertuples(index=False, name=None)):
            gpos = int(positions[local])
            key = tuple(row)
            if key not in first_seen:
                first_seen[key] = gpos + 1
                continue
            out.append(make_finding(
                self.rule_id, "(row)", "CRITICAL",
                f"row {gpos + 1} duplicates row {first_seen[key]}",
                row=gpos + 1, value=" | ".join(str(v) for v in row)[:120],
                evidence=f"rows {first_seen[key]} and {gpos + 1} identical across all "
                         f"{len(ctx.columns)} columns",
                suggested_action="de-duplicate; keep the first occurrence",
                dimension="uniqueness",
            ))
        return out


class InconsistentTypeDetector(Detector):
    """Minority type families inside a mostly-uniform column."""

    rule_id = "INCONSISTENT_TYPES"

    def run(self, ctx: ScanContext) -> list[Finding]:
        min_share = ctx.config["mixed_type_min_share"]
        min_count = ctx.config["mixed_type_min_count"]
        out: list[Finding] = []
        for prof in ctx.profiles:
            real_total = sum(prof.mixed_types.values())
            if real_total == 0:
                continue
            dominant = max(prof.mixed_types, key=prof.mixed_types.get)
            fam = ctx.families_for(prof.name)
            for family, count in prof.mixed_types.items():
                if family == dominant:
                    continue
                share = count / real_total
                if share < min_share or count < min_count:
                    continue
                rows = _rows_from_mask(ctx.df[prof.name], fam == family)
                for row, val in rows:
                    out.append(make_finding(
                        self.rule_id, prof.name, "WARNING",
                        f"value has type '{fam_label(family)}' but column is mostly "
                        f"'{fam_label(dominant)}' ({dominant_count(prof, dominant)}/{real_total})",
                        row=row, value=val,
                        suggested_action=f"coerce to {fam_label(dominant)} or split the column",
                        dimension="consistency",
                    ))
        return _evidence(out, ctx.row_evidence_cap)


class PaddedWhitespaceDetector(Detector):
    """Values with leading/trailing whitespace (' x' vs 'x')."""

    rule_id = "PADDED_WHITESPACE"

    def run(self, ctx: ScanContext) -> list[Finding]:
        out: list[Finding] = []
        for prof in ctx.profiles:
            s = ctx.df[prof.name]
            stripped = ctx.stripped_for(prof.name)
            rows = _rows_from_mask(s, (stripped != "") & (s != stripped))
            for row, val in rows:
                out.append(make_finding(
                    self.rule_id, prof.name, "INFO",
                    "value has leading/trailing whitespace",
                    row=row, value=val,
                    suggested_action="trim whitespace when ingesting this column",
                    dimension="consistency",
                ))
        return _evidence(out, ctx.row_evidence_cap)


class CaseVariantDetector(Detector):
    """The same categorical token spelled with different casing."""

    rule_id = "CASE_VARIANTS"

    def run(self, ctx: ScanContext) -> list[Finding]:
        out: list[Finding] = []
        for prof in ctx.profiles:
            if prof.inferred_type not in ("text", "boolean"):
                continue
            if prof.unique_pct > 90:          # free text / identifiers: skip
                continue
            s = ctx.df[prof.name]
            fam = ctx.families_for(prof.name)
            stripped = ctx.stripped_for(prof.name)
            distinct = stripped[~fam.isin(("empty", "blank", "placeholder"))]
            if distinct.empty:
                continue
            counts = distinct.value_counts()
            groups: dict[str, list[str]] = {}
            for spelling in counts.index:
                groups.setdefault(str(spelling).lower(), []).append(str(spelling))
            flagged: dict[str, str] = {}
            for spellings in groups.values():
                if len(spellings) < 2:
                    continue
                canonical = max(spellings, key=lambda sp: int(counts.get(sp, 0)))
                for sp in spellings:
                    if sp != canonical:
                        flagged[sp] = canonical
            if not flagged:
                continue
            for row, val in _rows_from_mask(s, stripped.isin(set(flagged))):
                canonical = flagged.get(str(val).strip(), "")
                out.append(make_finding(
                    self.rule_id, prof.name, "WARNING",
                    f"inconsistent capitalisation; column also uses '{canonical}'",
                    row=row, value=val,
                    suggested_action=f"standardise on '{canonical}'",
                    dimension="consistency",
                ))
        return _evidence(out, ctx.row_evidence_cap)


class InvalidFormatDetector(Detector):
    """Values that break the column's dominant format.

    * date/datetime columns: values that look temporal but do not match
      the dominant format, or are impossible calendar dates (2026-13-45);
    * boolean columns: values outside the detected yes/no vocabulary.
    """

    rule_id = "INVALID_FORMAT"

    def run(self, ctx: ScanContext) -> list[Finding]:
        out: list[Finding] = []
        for prof in ctx.profiles:
            series = ctx.df[prof.name]
            if prof.inferred_type in ("date", "datetime") and prof.date_format:
                fmt = prof.date_format
                for row, val in self._mismatched_dates(
                        ctx.stripped_for(prof.name), series, fmt):
                    out.append(make_finding(
                        self.rule_id, prof.name, "WARNING",
                        f"date does not match dominant format '{fmt}'",
                        row=row, value=val,
                        suggested_action=f"normalise to {fmt} or correct the value",
                        dimension="validity",
                    ))
            elif prof.inferred_type == "boolean":
                allowed = {str(v).strip().lower() for v, _ in prof.top_values
                           if str(v).strip().lower() in BOOLEAN_TOKENS}
                fam = ctx.families_for(prof.name)
                strip = ctx.stripped_for(prof.name)
                bad = ((strip != "") & (~strip.str.lower().isin(allowed))
                       & (~fam.isin(("boolean", "placeholder"))))
                for row, val in _rows_from_mask(series, bad):
                    out.append(make_finding(
                        self.rule_id, prof.name, "WARNING",
                        f"value outside boolean vocabulary {sorted(allowed)}",
                        row=row, value=val,
                        suggested_action="map to the column's yes/no vocabulary",
                        dimension="validity",
                    ))
        out.extend(self._impossible_dates(ctx))
        return _evidence(out, ctx.row_evidence_cap)

    @staticmethod
    def _mismatched_dates(stripped: pd.Series, series: pd.Series,
                          fmt: str) -> list[tuple[int, str]]:
        out: list[tuple[int, str]] = []
        for local, text in enumerate(stripped):
            if text == "":
                continue
            shape = temporal_shape(text)
            if shape is None:
                continue
            if shape[2] is None or shape[0] != fmt:
                out.append((local + 1, series.iat[local]))
        return out

    @staticmethod
    def _impossible_dates(ctx: ScanContext) -> list[Finding]:
        """Values shaped like a date but with an impossible day/month."""
        out: list[Finding] = []
        # a value can only be an impossible date if it starts with a
        # date-shaped prefix; the vectorised pre-filter skips whole
        # columns (ids, names, free text) without a per-row scan.
        shape_prefix = r"^\d{1,4}[-/.]\d{1,2}[-/.]"
        for prof in ctx.profiles:
            if prof.inferred_type in ("date", "datetime"):
                continue
            series = ctx.df[prof.name]
            if len(series) == 0:
                continue
            pre = ctx.stripped_for(prof.name).str.contains(
                shape_prefix, regex=True, na=False)
            if not pre.any():
                continue
            for pos in np.flatnonzero(pre.to_numpy()):
                val = series.iat[pos]
                if is_impossible_date(val):
                    out.append(make_finding(
                        InvalidFormatDetector.rule_id, prof.name, "WARNING",
                        "value is shaped like a date but is not a real calendar date",
                        row=int(pos) + 1, value=val,
                        suggested_action="correct the impossible date (month/day out of range)",
                        dimension="validity",
                    ))
        return out


class SuspiciousValueDetector(Detector):
    """Statistical red flags on numeric and date columns.

    * numeric outliers beyond the IQR fence (k * interquartile range)
    * dates in the future relative to scan time

    These are *suspicious*, not proven wrong — severity stays INFO/WARNING.
    """

    rule_id = "SUSPICIOUS_VALUES"

    def run(self, ctx: ScanContext) -> list[Finding]:
        out: list[Finding] = []
        out.extend(self._numeric_outliers(ctx))
        out.extend(self._future_dates(ctx))
        return _evidence(out, ctx.row_evidence_cap)

    def _numeric_outliers(self, ctx: ScanContext) -> list[Finding]:
        k = ctx.config["outlier_iqr_k"]
        min_vals = ctx.config["outlier_min_values"]
        out: list[Finding] = []
        for prof in ctx.profiles:
            if prof.inferred_type not in ("integer", "float"):
                continue
            num = ctx.numeric.get(prof.name)
            if num is None:
                continue
            valid = num.dropna()
            if len(valid) < min_vals:
                continue
            q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lo, hi = q1 - k * iqr, q3 + k * iqr
            series = ctx.df[prof.name]
            for row, val in _rows_from_mask(series, (num < lo) | (num > hi)):
                out.append(make_finding(
                    self.rule_id, prof.name, "INFO",
                    f"numeric outlier beyond IQR fence [{lo:.4g}, {hi:.4g}]",
                    row=row, value=val,
                    evidence=f"expected range roughly {lo:.4g}..{hi:.4g} "
                             f"(IQR k={k}, n={len(valid)})",
                    suggested_action="check this value; it is unusual for the column",
                    dimension="plausibility",
                ))
        return out

    def _future_dates(self, ctx: ScanContext) -> list[Finding]:
        horizon = int(ctx.config["future_date_days"])
        now = datetime.now()
        cutoff = now + timedelta(days=horizon)
        out: list[Finding] = []
        for prof in ctx.profiles:
            dates = ctx.dates.get(prof.name)
            if dates is None:
                continue
            series = ctx.df[prof.name]
            for local, dt_val in enumerate(dates):
                if dt_val is None or dt_val is pd.NaT:
                    continue
                if dt_val > cutoff:
                    ahead = (dt_val.date() - now.date()).days
                    out.append(make_finding(
                        self.rule_id, prof.name, "WARNING",
                        f"date is {ahead} days in the future",
                        row=local + 1, value=series.iat[local],
                        suggested_action="verify the date; a future date may be a typo",
                        dimension="plausibility",
                    ))
        return out


class MLAnomalyDetector(Detector):
    """Future ML slot. Intentionally silent — this version has no ML."""

    rule_id = "ML_ANOMALY"

    def run(self, ctx: ScanContext) -> list[Finding]:  # pragma: no cover
        return []


reset_registry()
