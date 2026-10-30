"""Reviewable remediation: suggestions, apply, changelog.

Suggestions are derived ONLY from deterministic findings and only when a
safe action can be inferred from the finding itself (e.g. trimming
whitespace). Ambiguous cases are always offered as manual review — no
value is invented. Applying changes builds a SEPARATE cleaned copy in
memory; the original upload is never touched.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

# rule_id -> confidence-level behaviour
REVIEW_REASON = {
    "PLACEHOLDER_VALUES": "decide the column policy: treat as missing or "
                          "map to a real category",
    "MISSING_VALUES": "no value can be inferred; fill from source or drop",
    "INCONSISTENT_TYPES": "type conflicts need a human decision",
    "INVALID_FORMAT": "the correct value is not knowable from the data "
                      "alone; verify against the source",
    "SUSPICIOUS_VALUES": "statistically unusual; verify before changing",
}


def _next_id(n: int) -> str:
    return f"S-{n:05d}"


def build_suggestions(report, df: pd.DataFrame,
                      *, cap_per_rule_column: int = 25) -> list[dict[str, Any]]:
    """Turn deterministic findings into reviewable suggestions.

    Row-level findings with an inferable action produce concrete
    before/after suggestions; everything else becomes a manual-review row.
    """
    suggestions: list[dict[str, Any]] = []
    seen_cells: set[tuple[str, int]] = set()
    counts: dict[tuple[str, str], int] = {}
    n = 0

    def _add(rule, column, row, action, before, after, reason,
             finding_ids, confidence):
        nonlocal n
        n += 1
        suggestions.append({
            "suggestion_id": _next_id(n),
            "rule_id": rule,
            "column": column,
            "row": row,
            "action": action,             # replace_value | drop_row | review
            "before": before,
            "after": after,
            "reason": reason,
            "finding_ids": finding_ids,
            "confidence": confidence,     # certain | review
            "status": "pending",          # pending | accepted | rejected | applied
        })

    for f in report.findings:
        rule, col, row = f["rule_id"], f["column"], f["row"]
        key = (rule, col)
        counts[key] = counts.get(key, 0) + 1
        if rule != "DUPLICATE_ROWS" and col not in df.columns:
            continue
        if row is None or row < 1 or row > len(df):
            continue
        if (key, row) in seen_cells or counts.get(key, 0) > cap_per_rule_column:
            continue
        if rule == "DUPLICATE_ROWS":
            before = ""
        else:
            raw = df.iat[row - 1, df.columns.get_loc(col)]
            before = "" if raw is None else str(raw)

        if rule == "PADDED_WHITESPACE":
            after = before.strip()
            if before == after:
                continue
            seen_cells.add((key, row))
            _add(rule, col, row, "replace_value", before, after,
                 "remove leading/trailing whitespace so values match the "
                 "column's canonical spelling",
                 [f["finding_id"]], "certain")
        elif rule == "BLANK_STRINGS":
            seen_cells.add((key, row))
            _add(rule, col, row, "replace_value", before, "",
                 "normalise whitespace-only cells to empty so they are "
                 "counted as missing",
                 [f["finding_id"]], "certain")
        elif rule == "CASE_VARIANTS":
            marker = "standardise on '"
            idx = f.get("suggested_action", "").find(marker)
            if idx == -1:
                continue
            canonical = f["suggested_action"][idx + len(marker):].rstrip("'")
            if before == canonical:
                continue
            seen_cells.add((key, row))
            _add(rule, col, row, "replace_value", before, canonical,
                 f"standardise capitalisation to the column's dominant "
                 f"spelling '{canonical}'",
                 [f["finding_id"]], "certain")
        elif rule == "DUPLICATE_ROWS":
            seen_cells.add((key, row))
            _add(rule, col, row, "drop_row", before, None,
                 f"this row duplicates an earlier row; keep the first "
                 f"occurrence ({f['evidence']})",
                 [f["finding_id"]], "certain")
        elif rule in REVIEW_REASON:
            seen_cells.add((key, row))
            _add(rule, col, row, "review", before, None,
                 REVIEW_REASON[rule], [f["finding_id"]], "review")

    return suggestions


def apply_changes(df: pd.DataFrame, suggestions: list[dict[str, Any]],
                  accepted_ids: list[str]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Apply ONLY accepted, value-safe suggestions to a COPY of df.

    Returns (cleaned_df, changelog). suggestion.status fields are updated
    in place so the caller can persist final states. Review items can
    never apply (marking them accepted registers them as rejected
    decisions in the changelog). Rows are dropped after value
    replacements, descending, with later changelog rows shifted.
    """
    accepted = [s for s in suggestions if s["suggestion_id"] in accepted_ids]
    for s in suggestions:
        if s["suggestion_id"] not in accepted_ids:
            s["status"] = "pending"
        elif s["action"] == "review":
            s["status"] = "rejected"
        else:
            s["status"] = "applied"

    cleaned = df.copy()
    cols = list(cleaned.columns)
    changelog: list[dict[str, Any]] = []

    # 1) value replacements (row references still valid)
    for s in accepted:
        if s["action"] != "replace_value":
            continue
        row, col = s["row"], s["column"]
        if row is None or row < 1 or row > len(cleaned) or col not in cols:
            continue
        cleaned.iat[row - 1, cols.index(col)] = \
            "" if s["after"] is None else s["after"]
        changelog.append({
            "row": row, "column": col, "action": s["action"],
            "before": s["before"], "after": s["after"],
            "reason": s["reason"], "suggestion_id": s["suggestion_id"],
            "rule_id": s["rule_id"],
        })

    # 2) row drops, descending, so references of remaining drops stay valid
    for s in sorted((s for s in accepted if s["action"] == "drop_row"),
                    key=lambda s: -int(s["row"] or 0)):
        row = int(s["row"] or 0)
        if row < 1 or row > len(cleaned):
            continue
        cleaned = cleaned.drop(index=cleaned.index[row - 1])
        cleaned.reset_index(drop=True, inplace=True)
        for e in (entry for entry in changelog if entry["row"] > row):
            e["row"] -= 1
        changelog.append({
            "row": row, "column": "(row)", "action": "drop_row",
            "before": s["before"], "after": None,
            "reason": s["reason"], "suggestion_id": s["suggestion_id"],
            "rule_id": s["rule_id"],
        })

    return cleaned, changelog


def changelog_to_csv(changelog: list[dict[str, Any]]) -> bytes:
    """Spreadsheet-friendly change log for audit purposes."""
    import io
    buf = io.StringIO()
    buf.write("suggestion_id,rule_id,row,column,action,before,after,reason\n")
    for e in changelog:
        def _esc(v: Any) -> str:
            if v is None:
                return ""
            txt = str(v).replace('"', '""')
            return f'"{txt}"' if any(c in txt for c in (",", "\n", '"')) else txt
        buf.write(
            f"{_esc(e['suggestion_id'])},{_esc(e['rule_id'])},{e['row']},"
            f"{_esc(e['column'])},{_esc(e['action'])},{_esc(e['before'])},"
            f"{_esc(e['after'])},{_esc(e['reason'])}\n")
    return buf.getvalue().encode("utf-8")
