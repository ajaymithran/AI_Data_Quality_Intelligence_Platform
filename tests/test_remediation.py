"""Tests for reviewable remediation: suggestions, apply, changelog.

Applying changes must build a separate cleaned copy — the source df is
never mutated, and manual-review items can never be applied.
"""
import pandas as pd
import pytest

from universal import scan_df
from universal.remediation import apply_changes, build_suggestions, changelog_to_csv


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).fillna("").astype(str)


def _by_rule(suggestions, rule_id):
    return [s for s in suggestions if s["rule_id"] == rule_id]


def test_pad_whitespace_suggestion_is_certain_with_after_value():
    df = _df([{"v": " ok"}, {"v": "ok "}, {"v": "ok"}])
    rep = scan_df(df)
    sug = build_suggestions(rep, df)
    pads = _by_rule(sug, "PADDED_WHITESPACE")
    assert len(pads) == 2
    assert pads[0]["confidence"] == "certain"
    assert pads[0]["action"] == "replace_value"
    assert pads[0]["after"] == "ok"


def test_blank_and_case_and_dup_suggestions():
    df = _df([
        {"w": "  ", "c": "icu"},    # blank string in w
        {"w": "x", "c": "ICU"},     # case variant (canonical 'icu')
        {"w": "y", "c": "icu"},
        {"w": "y", "c": "icu"},     # row 4 duplicates row 3
    ])
    rep = scan_df(df)
    sug = build_suggestions(rep, df)
    blanks = _by_rule(sug, "BLANK_STRINGS")
    cases = _by_rule(sug, "CASE_VARIANTS")
    dups = _by_rule(sug, "DUPLICATE_ROWS")
    assert blanks and blanks[0]["after"] == ""
    assert cases and cases[0]["after"] == "icu"
    assert dups and dups[0]["action"] == "drop_row" and dups[0]["row"] == 4


def test_review_only_rules_cannot_auto_apply():
    df = _df([{"a": "1"}, {"a": ""}, {"a": "N/A"}])
    rep = scan_df(df)
    sug = build_suggestions(rep, df)
    review = [s for s in sug if s["confidence"] == "review"]
    assert review
    assert all(s["action"] == "review" and s["after"] is None for s in review)
    # applying a review suggestion never changes data
    cleaned, changelog = apply_changes(
        df, sug, [review[0]["suggestion_id"]])
    assert changelog == []
    assert cleaned.equals(df.copy())


def test_apply_certain_suggestions_cleans_copy_without_touching_source():
    df = _df([{"v": " ok"}, {"v": "ok "}, {"v": "ok"}])
    rep = scan_df(df)
    sug = build_suggestions(rep, df)
    certain = [s["suggestion_id"] for s in sug if s["confidence"] == "certain"]
    cleaned, changelog = apply_changes(df, sug, certain)
    # source untouched
    assert list(df["v"]) == list(_df([{"v": " ok"}, {"v": "ok "}, {"v": "ok"}])["v"])
    # cleaned copy fixed
    assert list(cleaned["v"]) == ["ok", "ok", "ok"]
    assert len(changelog) == 2
    assert {e["before"] for e in changelog} == {" ok", "ok "}
    #statuses updated on the suggestion dicts
    assert all(s["status"] == "applied"
               for s in sug if s["suggestion_id"] in certain)


def test_duplicate_row_drop_and_row_shift():
    rows = [{"a": "1", "b": "x"}, {"a": "2", "b": "y"},
            {"a": "1", "b": "x"}, {"a": "3", "b": "z"}]
    df = _df(rows)
    rep = scan_df(df)
    sug = build_suggestions(rep, df)
    dups = _by_rule(sug, "DUPLICATE_ROWS")
    assert len(dups) == 1 and dups[0]["row"] == 3
    cleaned, changelog = apply_changes(
        df, sug, [dups[0]["suggestion_id"]])
    assert len(cleaned) == 3
    assert list(cleaned["a"]) == ["1", "2", "3"]
    assert changelog[0]["action"] == "drop_row"


def test_reject_leaves_pending_others_and_applies_only_accepted():
    df = _df([{"v": " x"}, {"v": "x "}, {"v": "x"}])
    rep = scan_df(df)
    sug = build_suggestions(rep, df)
    ids = [s["suggestion_id"] for s in sug]
    cleaned, changelog = apply_changes(df, sug, [ids[0]])
    assert len(changelog) == 1
    statuses = {s["status"] for s in sug}
    assert "applied" in statuses and "pending" in statuses


def test_changelog_csv_roundtrip_content():
    df = _df([{"v": " ok"}, {"v": "ok"}])
    rep = scan_df(df)
    sug = build_suggestions(rep, df)
    cleaned, changelog = apply_changes(
        df, sug, [s["suggestion_id"] for s in sug if s["confidence"] == "certain"])
    csv_bytes = changelog_to_csv(changelog)
    text = csv_bytes.decode("utf-8")
    assert "suggestion_id,rule_id,row,column,action,before,after,reason" in text
    assert "ok" in text
