"""Tests for the Cursor-powered AI Data Quality Analyst."""
import os
from unittest.mock import patch

import pandas as pd
import pytest

from universal import analyze_with_grok, scan_csv, scan_df
from universal.ai_analyst import _compute_deterministic_release_status, _build_evidence


DATA = __import__("pathlib").Path(__file__).resolve().parent.parent / "data"
HEALTH = DATA / "sample_universal_health.csv"
CORRUPTED = DATA / "sample_corrupted.csv"

VALID_PAYLOAD = """{
  "issue_explanations": [
    {
      "issue_id": "E-001",
      "rule_id": "MISSING_VALUES",
      "issue_name": "Missing values",
      "plain_english_explanation": "Some cells are empty.",
      "what_happened": "Empty cells found in the data.",
      "affected_records": [{"row": 5, "column": "glucose", "value": ""}],
      "affected_columns": ["glucose"],
      "potential_downstream_effects": ["Reports may show incomplete metrics"],
      "severity": "High",
      "severity_justification": "Affects completeness of key metrics",
      "likely_causes": ["HYPOTHESIS: Data entry omission"],
      "investigation_steps": ["Check source system"],
      "repair_steps": ["Confirm with source or impute per policy"],
      "can_be_safely_automated": false,
      "automation_rationale": "Requires business confirmation",
      "how_to_verify_correction": ["Re-scan after correction"],
      "how_to_prevent_recurrence": ["Add validation at entry"],
      "confidence": 0.8,
      "limitations": ["Limited context"],
      "missing_information": ["Source system rules"],
      "evidence_used": [{"finding_id": "F-00001", "source": "detector"}]
    }
  ],
  "suggestions": [
    {
      "suggestion_id": "S-001",
      "issue_id": "E-001",
      "issue": "Missing values in glucose",
      "evidence": "Row 5 has empty glucose",
      "proposed_action": "Review with clinical/data owner",
      "expected_result": "Decision on fill/drop",
      "risk": "Incorrect imputation could bias results",
      "approval_required": "review",
      "verification_status": "pending",
      "safe_to_auto_apply": false,
      "notes": "Do not auto-apply"
    }
  ],
  "overall_summary": {
    "most_important_risks": ["Data completeness issues"],
    "affected_columns": ["glucose"],
    "unresolved_issues": ["MISSING_VALUES"],
    "recommended_priority_order": ["MISSING_VALUES", "OTHERS"],
    "release_recommendation_text": "Address critical issues before release.",
    "ai_explanation_of_status": "Deterministic rules determined status; issues need review.",
    "confidence_overall": 0.75,
    "limitations": ["Based on structured evidence only"]
  }
}"""


def test_missing_api_key_returns_safe_message():
    with patch.dict(os.environ, {}, clear=True):
        report = scan_csv(HEALTH)
        df = pd.read_csv(HEALTH, dtype=str, keep_default_na=False)
        result = analyze_with_grok(report, df=df)
        assert result.status == "no_key"
        assert "GEMINI_API_KEY" in result.message
        assert result.overall_summary is not None
        assert result.overall_summary.deterministic_release_status in ["PASS", "PASS WITH WARNINGS", "BLOCKED"]


def test_placeholder_api_key_treated_as_missing():
    with patch.dict(os.environ, {"GEMINI_API_KEY": "your_api_key_here"}, clear=False):
        report = scan_csv(HEALTH)
        result = analyze_with_grok(report, df=pd.DataFrame())
        assert result.status == "no_key"


def test_deterministic_release_status_blocked_on_critical():
    report = scan_csv(CORRUPTED)
    status, reason = _compute_deterministic_release_status(report)
    # Corrupted sample has issues; should be deterministic
    assert status in ["PASS WITH WARNINGS", "BLOCKED"]
    assert isinstance(reason, str) and reason


def test_issue_free_data_message():
    df = pd.DataFrame({"a": ["1", "2", "3"], "b": ["x", "y", "z"]})
    report = scan_df(df)
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False):
        result = analyze_with_grok(report, df=df)
        assert result.status == "ok"
        assert result.issue_explanations == []
        assert result.overall_summary is not None
        assert "no issues within the tested rules" in result.overall_summary.release_recommendation_text.lower()
        # Must not claim perfect
        assert "perfect" not in result.overall_summary.release_recommendation_text.lower()


def _patched_chat(payload_or_exc):
    def fake(prompt, key, model, **kwargs):
        if isinstance(payload_or_exc, Exception):
            raise payload_or_exc
        return payload_or_exc
    return patch("universal.ai_analyst._grok_chat", side_effect=fake)


def test_api_failure_handled_gracefully():
    report = scan_csv(HEALTH)
    df = pd.read_csv(HEALTH, dtype=str, keep_default_na=False)

    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False):
        with _patched_chat(RuntimeError("Cursor API HTTP 429: rate limit exceeded")):
            result = analyze_with_grok(report, df=df)
            assert result.status == "api_error"
            assert "quota" in result.message.lower() or "rate" in result.message.lower()
            assert result.overall_summary is not None
            assert result.overall_summary.deterministic_release_status in ["PASS", "PASS WITH WARNINGS", "BLOCKED"]


def test_http_error_status_mapping():
    report = scan_csv(HEALTH)
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False):
        with _patched_chat(RuntimeError("Cursor API HTTP 401: invalid bearer")):
            result = analyze_with_grok(report, df=pd.DataFrame())
            assert result.status == "no_key"       # auth failure mapped to no_key
            assert "unauthorized" in result.message.lower() or "invalid" in result.message.lower()


def test_incorrect_api_key_status_mapping():
    report = scan_csv(HEALTH)
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False):
        with _patched_chat(RuntimeError("Cursor API HTTP 400: Incorrect API key provided")):
            result = analyze_with_grok(report, df=pd.DataFrame())
            assert result.status == "no_key"
            assert "unauthorized" in result.message.lower() or "invalid" in result.message.lower()


def test_valid_analysis_with_mocked_response():
    report = scan_csv(HEALTH)
    df = pd.read_csv(HEALTH, dtype=str, keep_default_na=False)

    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False):
        with _patched_chat(VALID_PAYLOAD):
            result = analyze_with_grok(report, df=df)
            assert result.status == "ok"
            assert result.model_used              # grok model recorded
            assert len(result.issue_explanations) == 1
            assert result.issue_explanations[0].issue_name == "Missing values"
            assert result.issue_explanations[0].can_be_safely_automated is False
            assert len(result.suggestions) == 1
            assert result.suggestions[0].safe_to_auto_apply is False
            assert result.suggestions[0].approval_required == "review"
            assert result.overall_summary is not None
            # Deterministic status must be set (not left to the model)
            assert result.overall_summary.deterministic_release_status in ["PASS", "PASS WITH WARNINGS", "BLOCKED"]
            assert result.overall_summary.deterministic_rationale != ""


def test_truncated_json_response_is_repaired():
    from universal.ai_analyst import _validate_and_parse_response

    report = scan_csv(HEALTH)
    evidence = _build_evidence(report, pd.DataFrame(), max_issues=12, max_examples=1)
    truncated = '{"issue_explanations":[{"issue_name":"Missing values","plain_english_explanation":"Some values are missing'

    issue_exps, suggestions, overall = _validate_and_parse_response(truncated, evidence)
    assert len(issue_exps) == 1
    assert issue_exps[0].issue_name == "Missing values"
    assert suggestions == []
    assert overall is None


def test_empty_gemini_lists_receive_deterministic_fallbacks():
    report = scan_csv(CORRUPTED)
    df = pd.read_csv(CORRUPTED, dtype=str, keep_default_na=False)
    payload = '{"issue_explanations":[],"suggestions":[],"overall_summary":{"release_recommendation_text":"","ai_explanation_of_status":"","most_important_risks":[]}}'

    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False):
        with _patched_chat(payload):
            result = analyze_with_grok(report, df=df)

    assert result.status == "ok"
    assert result.issue_explanations
    assert result.suggestions
    assert result.overall_summary is not None
    assert result.overall_summary.release_recommendation_text.strip()


def test_malformed_response_handled_gracefully():
    report = scan_csv(HEALTH)
    df = pd.read_csv(HEALTH, dtype=str, keep_default_na=False)

    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False):
        with _patched_chat("not valid json {{{"):
            result = analyze_with_grok(report, df=df)
            # Should not crash; should return error status with deterministic info
            assert result.status in ["api_error", "error"]
            assert result.overall_summary is not None
            assert result.overall_summary.deterministic_release_status in ["PASS", "PASS WITH WARNINGS", "BLOCKED"]


def test_legacy_key_accepts_gemini_explanation_field():
    """Old clients may emit gemini_explanation_of_status; we still parse it."""
    report = scan_csv(HEALTH)
    legacy = VALID_PAYLOAD.replace("ai_explanation_of_status", "gemini_explanation_of_status")
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False):
        with _patched_chat(legacy):
            result = analyze_with_grok(report, df=pd.read_csv(HEALTH, dtype=str, keep_default_na=False))
            assert result.overall_summary.ai_explanation_of_status != ""


@pytest.mark.skip(reason="Legacy Grok HTTP test replaced by Cursor bridge coverage")
def test_grok_chat_request_shape_and_parsing(monkeypatch):
    """_grok_chat posts to {base}/chat/completions with Bearer auth and grok model."""
    import httpx
    from universal.ai_analyst import _grok_chat

    captured = {}

    def fake_post(url, *, headers=None, json=None, timeout=None, **kw):
        captured["url"] = url
        captured["auth"] = (headers or {}).get("Authorization", "")
        captured["body"] = json
        captured["sends_user_content"] = json["messages"][0]["content"].startswith("You are")
        captured["model"] = json["model"]

        class R:
            status_code = 200
            def json(self):
                return {"choices": [{"message": {"content": "hello"}}]}
        return R()

    monkeypatch.setattr(httpx, "post", fake_post)
    text = _grok_chat("You are a test prompt", "sk-test", "grok-4.7")
    assert text == "hello"
    assert captured["url"] == "https://api.x.ai/v1/chat/completions"
    assert captured["auth"] == "Bearer sk-test"
    assert captured["model"] == "grok-4.7"
    assert captured["sends_user_content"] is True


def test_evidence_does_not_include_full_dataset():
    report = scan_csv(HEALTH)
    df = pd.read_csv(HEALTH, dtype=str, keep_default_na=False)
    evidence = _build_evidence(report, df)
    # Evidence should be structured, not the full dataframe
    assert isinstance(evidence, dict)
    assert "findings" in evidence
    assert "profiles" in evidence
    # Should not contain all rows as a big dump
    assert evidence["rows"] == len(df)
    # findings limited
    assert len(evidence["findings"]) <= 50


def test_suggestions_never_auto_apply():
    report = scan_csv(HEALTH)
    df = pd.read_csv(HEALTH, dtype=str, keep_default_na=False)

    payload = """{
      "issue_explanations": [],
      "suggestions": [
        {
          "suggestion_id": "S-001",
          "issue_id": "E-001",
          "issue": "test",
          "evidence": "test",
          "proposed_action": "test",
          "expected_result": "test",
          "risk": "test",
          "approval_required": "yes",
          "verification_status": "pending",
          "safe_to_auto_apply": true,
          "notes": "test"
        }
      ],
      "overall_summary": {
        "most_important_risks": [],
        "affected_columns": [],
        "unresolved_issues": [],
        "recommended_priority_order": [],
        "release_recommendation_text": "test",
        "ai_explanation_of_status": "test",
        "confidence_overall": 0.5,
        "limitations": []
      }
    }"""

    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False):
        with _patched_chat(payload):
            result = analyze_with_grok(report, df=df)
            assert result.status == "ok"
            assert len(result.suggestions) == 1
            # Must be forced to False regardless of model response
            assert result.suggestions[0].safe_to_auto_apply is False
            assert result.suggestions[0].verification_status == "pending"


def test_env_gemini_model_override():
    from universal.ai_analyst import _get_model
    with patch.dict(os.environ, {"GEMINI_MODEL": "gemini-2.0-flash"}, clear=False):
        assert _get_model(None) == "gemini-2.0-flash"
    with patch.dict(os.environ, {"GEMINI_MODEL": ""}, clear=False):
        assert _get_model(None) == "gemini-3.8-flash"
