"""Automatic, name-agnostic type inference.

Everything is decided from the *values* in a column — never from its
name — so any CSV (retail, health, finance, sensor, survey, ...) is
inspected the same way.

Value families (a value belongs to exactly one):

  empty         the field is truly absent ("" in the raw CSV)
  blank         whitespace only ("   ")
  placeholder   looks present but stands for missing: N/A, null, -, unknown, ...
  integer       42, -7, +12
  float         3.14, -0.5, 1e3
  boolean       true/false/yes/no/y/n (case-insensitive)
  date          looks like a calendar date with a real day/month
  datetime      looks like a date plus a time
  text          anything else

The *dominant* family becomes the column's inferred type. Minority
families among real values are kept in ``ColumnProfile.mixed_types``
so the inconsistent-type detector can flag them with row evidence.

Performance notes
-----------------
CSV files can be large (hundreds of thousands of rows). Two rules keep
inference fast and stable:

* dates are recognised with a single precompiled regular expression and
  the ``datetime`` constructor — never ``datetime.strptime``, which
  recompiles its format regex on every single call;
* every value is classified once per column (the resulting family series
  is shared with all detectors) and classification is memoised, so
  repeated values (dates, categories, flags) cost nothing extra.
"""
from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
from functools import lru_cache

import pandas as pd

from .schema import ColumnProfile

INT_RE = re.compile(r"^[+-]?\d+$")
FLOAT_RE = re.compile(r"^[+-]?(\d+\.\d*|\.\d+|\d+[eE][+-]?\d+|\d+\.\d*[eE][+-]?\d+)$")

PLACEHOLDER_TOKENS = {
    "n/a", "na", "null", "none", "nil", "nan", "-", "--", "?", ".",
    "unknown", "tbd", "missing", "not available", "#n/a", "(blank)",
}
BOOLEAN_TOKENS = {"true", "false", "yes", "no", "y", "n"}

# Kept for API/report compatibility. Actual recognition is done by the
# regex below; the strings here are the canonical formats it reports.
DATE_FORMATS = [
    "%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y",
    "%d.%m.%Y", "%Y%m%d",
]
DATETIME_FORMATS = [
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%d/%m/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S.%f",
]

# One regex recognises every supported date/datetime shape at once:
#   <a><sep><b><sep><c> [T or space HH:MM[:SS[.fff]]]
# where a/b/c are day/month/year in some order. ``temporal_shape``
# resolves the order (a 4-digit first group is the year; otherwise the
# 4-digit last group is the year and day/month are disambiguated by
# whichever of the first two groups is > 12).
_TEMPORAL_RE = re.compile(
    r"^(?P<a>\d{1,4})[-/.](?P<b>\d{1,2})[-/.](?P<c>\d{1,4})"
    r"(?:[ T](?P<H>\d{1,2}):(?P<M>\d{1,2})(?::(?P<S>\d{1,2}))?(?:\.\d+)?)?$"
)


@lru_cache(maxsize=262_144)
def temporal_shape(s: str) -> tuple[str, bool, datetime | None] | None:
    """Recognise a date/datetime shape.

    Returns ``(format, has_time, dt)`` or ``None`` when the value is not
    date-shaped at all. ``dt`` is ``None`` for shapes that can never be a
    real calendar date (e.g. ``2026-13-45``).
    """
    m = _TEMPORAL_RE.match(s)
    if m is None:
        return None
    a, b, c = m.group("a"), m.group("b"), m.group("c")
    if len(a) == 4:
        fmt = "%Y-%m-%d"
        year, month, day = int(a), int(b), int(c)
    elif len(c) == 4:
        x, y = int(a), int(b)
        if x > 12:                       # first group must be the day
            fmt = "%d/%m/%Y"
            day, month = x, y
        elif y > 12:                     # second group must be the day
            fmt = "%m/%d/%Y"
            month, day = x, y
        else:                            # ambiguous -> prefer day-first
            fmt = "%d/%m/%Y"
            day, month = x, y
        year = int(c)
    else:
        return None

    has_time = m.group("H") is not None
    if has_time:
        fmt += " %H:%M" + (":%S" if m.group("S") is not None else "")
    try:
        dt = datetime(year, month, day,
                      int(m.group("H") or 0), int(m.group("M") or 0), int(m.group("S") or 0))
    except ValueError:
        dt = None
    return fmt, has_time, dt


@lru_cache(maxsize=262_144)
def classify_value(v: str) -> str:
    """Classify one raw CSV string into a value family (memoised)."""
    if v == "":
        return "empty"
    s = v.strip()
    if s == "":
        return "blank"
    low = s.lower()
    if low in PLACEHOLDER_TOKENS:
        return "placeholder"
    if low in BOOLEAN_TOKENS:
        return "boolean"
    if INT_RE.match(s):
        return "integer"
    if FLOAT_RE.match(s):
        return "float"
    shape = temporal_shape(s)
    if shape is not None and shape[2] is not None:
        return "datetime" if shape[1] else "date"
    return "text"


def value_families(series: pd.Series) -> pd.Series:
    """Classify a whole column once; reused by every detector."""
    return series.map(classify_value)


def _looks_temporal(s: str) -> bool:
    """True when the value has a recognised date/datetime shape."""
    return temporal_shape(s) is not None


def _match_temporal(s: str, formats: list[str]) -> str | None:
    """Return the format if ``s`` is a *valid* date in one of ``formats``."""
    shape = temporal_shape(s)
    if shape is None or shape[2] is None:
        return None
    return shape[0] if shape[0] in formats else None


def is_impossible_date(v: str) -> bool:
    """A value shaped like a date but with an impossible day/month."""
    text = v.strip()
    if not text or not text[0].isdigit():
        return False          # fast path: date shapes always start with a digit
    shape = temporal_shape(text)
    return shape is not None and shape[2] is None


def parse_series(series: pd.Series, families: list[str],
                 fam_series: pd.Series | None = None) -> pd.Series:
    """Numeric view of a column: ints/floats parsed, everything else NaN."""
    if fam_series is None:
        fam_series = value_families(series)
    mask = fam_series.isin(families)
    out = pd.Series(float("nan"), index=series.index, dtype="float64")
    if mask.any():
        out[mask] = pd.to_numeric(series[mask].str.strip(), errors="coerce")
    return out


def parse_dates(series: pd.Series, fmt: str) -> pd.Series:
    """Datetime view of a column using one explicit format.

    Parsed element-wise (object dtype of ``datetime``/NaT) so far-future
    dates beyond nanosecond-resolution limits (e.g. year 2999) cannot
    overflow the way ``pd.to_datetime`` would.
    """
    vals = []
    append = vals.append
    for v in series:
        shape = temporal_shape(v.strip())
        if shape is not None and shape[2] is not None and shape[0] == fmt:
            append(shape[2])
        else:
            append(pd.NaT)
    return pd.Series(vals, index=series.index, dtype="object")


def profile_column(name: str, series: pd.Series,
                   fam_series: pd.Series | None = None) -> tuple[ColumnProfile, dict]:
    """Infer the type of one column from its values.

    Returns (profile, extra) where extra carries parse caches:
      {"numeric": pd.Series, "dates": pd.Series|None, "date_format": str|None}
    """
    if fam_series is None:
        fam_series = value_families(series)
    families = fam_series.value_counts().to_dict()
    stripped = series.str.strip()

    n = len(series)
    missing = families.get("empty", 0)
    blank = families.get("blank", 0)
    placeholders = families.get("placeholder", 0)
    non_null = n - missing - blank

    real = Counter({k: v for k, v in families.items()
                    if k not in ("empty", "blank", "placeholder")})
    real_mask = ~fam_series.isin(("empty", "blank", "placeholder"))

    profile = ColumnProfile(name=name, inferred_type="empty",
                            non_null=non_null, missing=missing,
                            blank=blank, placeholder=placeholders)

    if real:
        top_family, top_count = real.most_common(1)[0]
        dominance = top_count / max(sum(real.values()), 1)

        if top_family in ("integer", "float"):
            profile.inferred_type = "float" if "float" in real else "integer"
        else:
            profile.inferred_type = top_family

        distinct = series[real_mask]
        real_total = int(real_mask.sum())
        profile.unique = int(stripped[real_mask].nunique())
        profile.unique_pct = round(profile.unique / max(real_total, 1) * 100, 1)
        counts = stripped[real_mask].value_counts()
        profile.top_values = [[str(k), int(v)] for k, v in counts.head(5).items()]
        profile.samples = [str(v) for v in distinct.head(3)]

        if len(real) > 1:
            profile.mixed_types = dict(real)

        profile.id_like = real_total >= 5 and profile.unique / max(real_total, 1) > 0.98

        if dominance < 0.5:
            profile.inferred_type = "mixed"

    extra: dict = {"numeric": None, "dates": None, "date_format": None,
                   "stripped": stripped}

    if profile.inferred_type in ("integer", "float"):
        numeric = parse_series(series, ["integer", "float"], fam_series)
        extra["numeric"] = numeric
        if numeric.notna().any():
            profile.min = float(numeric.min())
            profile.max = float(numeric.max())
            profile.mean = round(float(numeric.mean()), 6)

    if profile.inferred_type in ("date", "datetime"):
        fmt = _dominant_format(stripped, fam_series)
        extra["date_format"] = fmt
        profile.date_format = fmt
        if fmt:
            extra["dates"] = parse_dates(series, fmt)

    return profile, extra


def _dominant_format(stripped: pd.Series, fam_series: pd.Series) -> str | None:
    """The date format that matches the most valid date values."""
    counts: Counter = Counter()
    for v, f in zip(stripped, fam_series):
        if f in ("date", "datetime"):
            shape = temporal_shape(v)
            if shape is not None and shape[2] is not None:
                counts[shape[0]] += 1
    if not counts:
        return None
    total = sum(counts.values())
    fmt, hits = counts.most_common(1)[0]
    return fmt if hits / total >= 0.5 else None


def profile_frame(df: pd.DataFrame) -> tuple[list[ColumnProfile], dict]:
    """Profile every column of an all-string DataFrame.

    Returns (profiles, cache) where ``cache[col]["families"]`` is the
    per-value family series shared with the detectors.
    """
    profiles: list[ColumnProfile] = []
    cache: dict[str, dict] = {}
    for col in df.columns:
        fam_series = value_families(df[col])
        prof, extra = profile_column(col, df[col], fam_series)
        extra["families"] = fam_series
        profiles.append(prof)
        cache[col] = extra
    return profiles, cache
