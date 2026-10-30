"""Gemini-powered AI Data Quality Analyst.

Runs AFTER deterministic rule-based checks. Uses only structured evidence
(never full dataset by default). Treats all AI output as advisory. Never
executes code/SQL returned by the model. Suggestions never auto-apply.

The Google Gemini SDK is used for optional advisory analysis. The API key
is read from GEMINI_API_KEY or GOOGLE_API_KEY and is never logged, displayed,
or written into reports.
"""
from __future__ import annotations

import json
import os
import re
import time
import subprocess
import shutil
from datetime import datetime
from typing import Any

import pandas as pd

from .ai_schemas import (
    AIAuditEntry,
    AIAnalysisReport,
    IssueExplanation,
    OverallAISummary,
    ReviewableSuggestion,
)
from .schema import ScanReport

# Try to load .env if present (excluded by git)
try:
    from dotenv import load_dotenv
    from pathlib import Path
    _ROOT = Path(__file__).resolve().parent.parent
    load_dotenv(_ROOT / ".env", override=True)
    load_dotenv(override=True)
except Exception:
    pass


# ---------------------------------------------------------------- provider
DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_BASE_URL = ""

GEMINI_LIMITATION_NOTE = (
    "AI commentary is advisory only; the deterministic release status is "
    "computed by rules and cannot be changed by the model.")


def _placeholder_key(k: str) -> bool:
    return (
        not k
        or k.lower() in (
            "your_api_key_here", "your_xai_api_key_here",
            "your_gemini_api_key_here", "your_key_here", "<your_actual_key>")
        or k.startswith("<"))


def _get_api_key(api_key: str | None) -> str:
    if api_key and str(api_key).strip():
        return str(api_key).strip()
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        v = os.environ.get(name)
        if v and not _placeholder_key(str(v).strip()):
            return str(v).strip()
    return ""


def _get_model(model: str | None) -> str:
    if model and str(model).strip():
        return str(model).strip()
    env = os.environ.get("GEMINI_MODEL") or os.environ.get("GOOGLE_MODEL")
    return env.strip() if env and env.strip() else DEFAULT_MODEL


def _gemini_chat(prompt: str, api_key: str, model: str, *, timeout: float = 120.0) -> str:
    """Run one prompt through the Google Gemini SDK."""
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=2200,
                response_mime_type="application/json",
            ),
        )
    except Exception as exc:
        raise RuntimeError(f"Gemini API request failed: {exc}") from exc
    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Gemini returned an empty response")
    return text.strip()


def _grok_chat(prompt: str, api_key: str, model: str,
               *, base_url: str | None = None, timeout: float = 120.0) -> str:
    """Compatibility name retained by existing integrations."""
    return _gemini_chat(prompt, api_key, model, timeout=timeout)


# --------------------------------------------------------------- evidence
SENSITIVE_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN-like
    re.compile(r"\b\d{16}\b"),  # credit card-like (simple)
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),  # email
]


def _mask_sensitive(text: str) -> str:
    if not isinstance(text, str):
        return str(text)
    s = text
    for pat in SENSITIVE_PATTERNS:
        s = pat.sub("[REDACTED]", s)
    return s


def _safe_str(val: Any, max_len: int = 200) -> str:
    s = "" if val is None else str(val)
    s = _mask_sensitive(s)
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def _build_evidence(report: ScanReport, df: pd.DataFrame, max_issues: int = 12, max_examples: int = 1) -> dict[str, Any]:
    """Build minimal, structured evidence for the model. Never include full dataset."""
    findings = report.findings[:max_issues]
    evidence_findings = []
    for f in findings:
        row_id = f.get("row")
        examples = []
        if row_id is not None and row_id >= 1 and row_id <= len(df):
            try:
                row = df.iloc[row_id - 1]
                # collect a few non-empty values as examples
                for col in df.columns[:10]:  # limit columns
                    val = row.get(col, "")
                    if val != "" and val is not None:
                        examples.append({col: _safe_str(val, 80)})
                        if len(examples) >= max_examples:
                            break
            except Exception:
                pass
        evidence_findings.append(
            {
                "finding_id": f.get("finding_id"),
                "rule_id": f.get("rule_id"),
                "column": f.get("column"),
                "severity": f.get("severity"),
                "message": f.get("message"),
                "row": row_id,
                "value": _safe_str(f.get("value"), 80),
                "evidence": _safe_str(f.get("evidence"), 200),
                "suggested_action": _safe_str(f.get("suggested_action"), 200),
                "dimension": f.get("dimension"),
                "examples": examples,
            }
        )

    profiles_summary = []
    for p in report.columns[:30]:  # limit
        try:
            profiles_summary.append(
                {
                    "name": p.name,
                    "inferred_type": p.inferred_type,
                    "missing": p.missing,
                    "blank": p.blank,
                    "placeholder": p.placeholder,
                    "unique_pct": p.unique_pct,
                    "mixed_types": getattr(p, "mixed_types", None),
                }
            )
        except Exception:
            continue

    return {
        "filename": report.filename,
        "rows": report.rows,
        "columns_count": len(report.columns),
        "severity_counts": report.severity_counts,
        "dimensions": report.dimensions,
        "finding_total": report.finding_total,
        "findings": evidence_findings,
        "profiles": profiles_summary,
        "notes": report.notes[:5],
    }


def _compute_deterministic_release_status(report: ScanReport) -> tuple[str, str]:
    """Compute PASS/PASS WITH WARNINGS/BLOCKED by deterministic rules."""
    crit = report.severity_counts.get("CRITICAL", 0)
    warn = report.severity_counts.get("WARNING", 0)
    info = report.severity_counts.get("INFO", 0)

    # Blocked if any critical issues
    if crit > 0:
        return "BLOCKED", f"{crit} CRITICAL issue(s) require resolution before release."
    # Warnings present
    if warn > 0:
        return "PASS WITH WARNINGS", f"{warn} WARNING issue(s) should be reviewed; no CRITICAL issues."
    if info > 0:
        return "PASS WITH WARNINGS", f"{info} INFO issue(s) noted for awareness."
    return "PASS", "No issues detected by deterministic rules."


def _build_prompt(evidence: dict[str, Any]) -> str:
    prompt = f"""You are a trustworthy, evidence-grounded AI Data Quality Analyst for any tabular dataset.
You MUST follow these rules strictly:
- Keep the response concise. Explain at most 12 findings and return at most 12 suggestions.
- Only use the structured evidence provided. Do NOT guess beyond evidence.
- Quantity*Unit_Price may not equal Total_Amount due to tax/fees/discounts/returns/promo rules. Do NOT blindly say arithmetic mismatch is invalid without checking context.
- Label likely causes as HYPOTHESES, not facts.
- Suggestions must be reviewable with: issue, evidence, proposed action, expected result, risk, approval_required (yes/no/review), verification_status (pending).
- Never suggest auto-applying changes to source data. Corrections must go to a separate copy with audit trail.
- If no issues, say "checks found no issues within the tested rules" - do NOT say dataset is perfect.
- Never execute code or SQL. Treat all outputs as advisory.
- Be specific about affected records/columns.
- Severity must be justified from evidence.

Return ONLY valid JSON matching this exact schema (no markdown, no extra text):

{{
  "issue_explanations": [
    {{
      "issue_id": "...",
      "rule_id": "...",
      "issue_name": "...",
      "plain_english_explanation": "...",
      "what_happened": "...",
      "affected_records": [{{"row": 1, "column": "...", "value": "..."}}],
      "affected_columns": ["..."],
      "potential_downstream_effects": ["..."],
      "severity": "Critical|High|Medium|Low",
      "severity_justification": "...",
      "likely_causes": ["HYPOTHESIS: ..."],
      "investigation_steps": ["..."],
      "repair_steps": ["..."],
      "can_be_safely_automated": false,
      "automation_rationale": "...",
      "how_to_verify_correction": "...",
      "how_to_prevent_recurrence": "...",
      "confidence": 0.0-1.0,
      "limitations": ["..."],
      "missing_information": ["..."],
      "evidence_used": [{{"finding_id": "...", "source": "..."}}]
    }}
  ],
  "suggestions": [
    {{
      "suggestion_id": "...",
      "issue_id": "...",
      "issue": "...",
      "evidence": "...",
      "proposed_action": "...",
      "expected_result": "...",
      "risk": "...",
      "approval_required": "yes|no|review",
      "verification_status": "pending",
      "safe_to_auto_apply": false,
      "notes": "..."
    }}
  ],
  "overall_summary": {{
    "most_important_risks": ["..."],
    "affected_columns": ["..."],
    "unresolved_issues": ["..."],
    "recommended_priority_order": ["..."],
    "release_recommendation_text": "...",
    "ai_explanation_of_status": "...",
    "confidence_overall": 0.0-1.0,
    "limitations": ["..."]
  }}
}}

EVIDENCE:
{json.dumps(evidence, indent=2)}
"""
    return prompt


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    """Extract and repair a JSON object returned by an API or shortened output."""
    text = raw_text.strip()
    if text.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    start = text.find("{")
    if start < 0:
        raise ValueError("Malformed JSON response: no JSON object found")
    text = text[start:]

    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError as original_error:
        original_error_message = str(original_error)

    stack = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif char in ("}", "]") and stack and char == stack[-1]:
            stack.pop()

    repaired = text.rstrip().rstrip(",")
    if in_string:
        repaired += '"'
    repaired += "".join(reversed(stack))

    try:
        data = json.loads(repaired)
    except json.JSONDecodeError as repair_error:
        raise ValueError(f"Malformed JSON response: {original_error_message}") from repair_error
    if not isinstance(data, dict):
        raise ValueError("Response must be a JSON object")
    return data


def _validate_and_parse_response(raw_text: str, evidence: dict[str, Any]) -> tuple[list[IssueExplanation], list[ReviewableSuggestion], OverallAISummary | None]:
    """Validate model response structure. Raises on malformed."""
    if not raw_text or not raw_text.strip():
        raise ValueError("Empty response from model")
    data = _extract_json_object(raw_text)

    issue_exps = []
    for item in data.get("issue_explanations", []):
        if not isinstance(item, dict):
            continue
        try:
            issue_exps.append(
                IssueExplanation(
                    issue_id=str(item.get("issue_id", "")),
                    rule_id=str(item.get("rule_id", "")),
                    issue_name=str(item.get("issue_name", "")),
                    plain_english_explanation=str(item.get("plain_english_explanation", "")),
                    what_happened=str(item.get("what_happened", "")),
                    affected_records=item.get("affected_records", []) if isinstance(item.get("affected_records"), list) else [],
                    affected_columns=item.get("affected_columns", []) if isinstance(item.get("affected_columns"), list) else [],
                    potential_downstream_effects=item.get("potential_downstream_effects", []) if isinstance(item.get("potential_downstream_effects"), list) else [],
                    severity=item.get("severity", "Medium") if item.get("severity") in ["Critical", "High", "Medium", "Low"] else "Medium",
                    severity_justification=str(item.get("severity_justification", "")),
                    likely_causes=item.get("likely_causes", []) if isinstance(item.get("likely_causes"), list) else [],
                    investigation_steps=item.get("investigation_steps", []) if isinstance(item.get("investigation_steps"), list) else [],
                    repair_steps=item.get("repair_steps", []) if isinstance(item.get("repair_steps"), list) else [],
                    can_be_safely_automated=bool(item.get("can_be_safely_automated", False)),
                    automation_rationale=str(item.get("automation_rationale", "")),
                    how_to_verify_correction=str(item.get("how_to_verify_correction", "")),
                    how_to_prevent_recurrence=str(item.get("how_to_prevent_recurrence", "")),
                    confidence=float(item.get("confidence", 0.5)),
                    limitations=item.get("limitations", []) if isinstance(item.get("limitations"), list) else [],
                    missing_information=item.get("missing_information", []) if isinstance(item.get("missing_information"), list) else [],
                    evidence_used=item.get("evidence_used", []) if isinstance(item.get("evidence_used"), list) else [],
                )
            )
        except Exception:
            continue  # skip invalid entries

    suggestions = []
    for item in data.get("suggestions", []):
        if not isinstance(item, dict):
            continue
        try:
            suggestions.append(
                ReviewableSuggestion(
                    suggestion_id=str(item.get("suggestion_id", "")),
                    issue_id=str(item.get("issue_id", "")),
                    issue=str(item.get("issue", "")),
                    evidence=str(item.get("evidence", "")),
                    proposed_action=str(item.get("proposed_action", "")),
                    expected_result=str(item.get("expected_result", "")),
                    risk=str(item.get("risk", "")),
                    approval_required=item.get("approval_required", "review") if item.get("approval_required") in ["yes", "no", "review"] else "review",
                    verification_status="pending",
                    safe_to_auto_apply=False,  # always False
                    notes=str(item.get("notes", "")),
                )
            )
        except Exception:
            continue

    overall = None
    osum = data.get("overall_summary")
    if isinstance(osum, dict):
        try:
            overall = OverallAISummary(
                most_important_risks=osum.get("most_important_risks", []) if isinstance(osum.get("most_important_risks"), list) else [],
                affected_columns=osum.get("affected_columns", []) if isinstance(osum.get("affected_columns"), list) else [],
                unresolved_issues=osum.get("unresolved_issues", []) if isinstance(osum.get("unresolved_issues"), list) else [],
                recommended_priority_order=osum.get("recommended_priority_order", []) if isinstance(osum.get("recommended_priority_order"), list) else [],
                release_recommendation_text=str(osum.get("release_recommendation_text", "")),
                deterministic_release_status="PASS",  # will be overwritten
                deterministic_rationale="",
                ai_explanation_of_status=str(
                    osum.get("ai_explanation_of_status",
                             osum.get("gemini_explanation_of_status", ""))),
                confidence_overall=float(osum.get("confidence_overall", 0.5)),
                limitations=osum.get("limitations", []) if isinstance(osum.get("limitations"), list) else [],
            )
        except Exception:
            overall = None

    return issue_exps, suggestions, overall


def _advisory_fallback(
    report: ScanReport, audit: list[AIAuditEntry], status: str, msg: str,
    det_status: str, det_reason: str,
    explanation: str = "AI analysis unavailable; deterministic results only.",
    limitations: list[str] | None = None,
) -> AIAnalysisReport:
    """Build the standard 'AI unavailable/failed' report (deterministic intact)."""
    overall = OverallAISummary(
        most_important_risks=[],
        affected_columns=[],
        unresolved_issues=[],
        recommended_priority_order=[],
        release_recommendation_text="",
        deterministic_release_status=det_status,
        deterministic_rationale=det_reason,
        ai_explanation_of_status=explanation,
        confidence_overall=0.0,
        limitations=limitations or [msg, GEMINI_LIMITATION_NOTE],
    )
    return AIAnalysisReport(
        status=status,
        message=msg,
        issue_explanations=[],
        suggestions=[],
        overall_summary=overall,
        audit_trail=audit,
    )


def _deterministic_recommendations(report: ScanReport, evidence: dict[str, Any]) -> tuple[list[IssueExplanation], list[ReviewableSuggestion], OverallAISummary | None]:
    """Create useful review content even if the model returns incomplete JSON."""
    issue_exps = []
    suggestions = []
    seen_rules = set()

    for finding in evidence.get("findings", [])[:6]:
        rule_id = str(finding.get("rule_id") or "DATA_QUALITY")
        finding_id = str(finding.get("finding_id") or rule_id)
        issue_name = str(finding.get("rule_id") or finding.get("message") or "Data-quality issue").replace("_", " ").title()
        message = str(finding.get("message") or "The deterministic quality rule detected an issue.")
        column = finding.get("column")

        issue_exps.append(IssueExplanation(
            issue_id=finding_id,
            rule_id=rule_id,
            issue_name=issue_name,
            plain_english_explanation=message,
            what_happened=message,
            affected_columns=[str(column)] if column else [],
            severity=finding.get("severity") if finding.get("severity") in ["Critical", "High", "Medium", "Low"] else "Medium",
            severity_justification="Severity is assigned by the deterministic scanner.",
            investigation_steps=[f"Review the rows identified by rule {rule_id}."],
            confidence=0.8,
            limitations=["This explanation is generated from deterministic findings, not inferred values."],
        ))

        if rule_id not in seen_rules:
            seen_rules.add(rule_id)
            suggestions.append(ReviewableSuggestion(
                suggestion_id=f"S-{len(suggestions) + 1:03d}",
                issue_id=finding_id,
                issue=issue_name,
                evidence=message,
                proposed_action=f"Review rows and source values for rule {rule_id}; correct the source data only after approval.",
                expected_result="The affected values and downstream calculations can be verified without changing the original file.",
                risk="Incorrect assumptions or automated edits may introduce new data-quality issues.",
                approval_required="review",
                verification_status="pending",
                safe_to_auto_apply=False,
                notes="Recommendation generated from deterministic evidence.",
            ))

    det_status, det_reason = _compute_deterministic_release_status(report)
    affected_columns = sorted({
        str(finding["column"])
        for finding in evidence.get("findings", [])
        if finding.get("column")
    })
    most_important = [
        str(finding["message"])
        for finding in evidence.get("findings", [])
        if finding.get("message")
    ][:6]
    return issue_exps, suggestions, OverallAISummary(
        most_important_risks=most_important,
        affected_columns=affected_columns,
        unresolved_issues=[str(finding["message"]) for finding in evidence.get("findings", []) if finding.get("message")][:6],
        recommended_priority_order=[suggestion.issue for suggestion in suggestions],
        release_recommendation_text="Review the findings in priority order, verify source values, and make corrections in a separate cleaned copy.",
        deterministic_release_status=det_status,
        deterministic_rationale=det_reason,
        ai_explanation_of_status="AI recommendations below are advisory and are grounded in the deterministic findings.",
        confidence_overall=0.8,
        limitations=[GEMINI_LIMITATION_NOTE],
    )


def analyze_with_gemini(
    report: ScanReport,
    df: pd.DataFrame | None = None,
    api_key: str | None = None,
    model: str | None = None,
    max_issues: int = 12,
    max_examples: int = 1,
    base_url: str | None = None,
) -> AIAnalysisReport:
    """Analyze a scan report with Gemini. Deterministic release status is
    computed separately by rules and can never be changed by the model."""
    audit = [AIAuditEntry(action="started_analysis", timestamp=datetime.utcnow().isoformat() + "Z", details={"filename": report.filename})]
    started = time.time()

    # Get API key (placeholder values are treated as no key)
    key = _get_api_key(api_key)
    if _placeholder_key(key):
        audit.append(AIAuditEntry(action="missing_api_key", timestamp=datetime.utcnow().isoformat() + "Z", details={}))
        det_status, det_reason = _compute_deterministic_release_status(report)
        result = _advisory_fallback(
            report, audit, "no_key",
            "GEMINI_API_KEY not found. Set it in environment or .env file (see .env.example). "
            "Analysis skipped — deterministic results remain fully available.",
            det_status, det_reason,
            explanation="Gemini analysis not performed (missing API key).")
        return result
    used_model = _get_model(model)

    # Build evidence
    if df is None:
        df = pd.DataFrame()
    evidence = _build_evidence(report, df, max_issues=max_issues, max_examples=max_examples)

    # If no issues, handle specially (no model call needed)
    if report.finding_total == 0 or len(evidence.get("findings", [])) == 0:
        det_status, det_reason = _compute_deterministic_release_status(report)
        overall = OverallAISummary(
            most_important_risks=[],
            affected_columns=[],
            unresolved_issues=[],
            recommended_priority_order=["No issues found - monitor data quality as usual"],
            release_recommendation_text="Checks found no issues within the tested rules.",
            deterministic_release_status=det_status,
            deterministic_rationale=det_reason,
            ai_explanation_of_status=("The checks found no issues within the tested rules; "
                                      "this does not mean the dataset is perfect."),
            confidence_overall=0.9,
            limitations=["Analysis limited to configured deterministic rules"],
        )
        audit.append(AIAuditEntry(action="no_issues_found", timestamp=datetime.utcnow().isoformat() + "Z", details={}))
        return AIAnalysisReport(
            status="ok",
            message="No issues found by deterministic checks.",
            issue_explanations=[],
            suggestions=[],
            overall_summary=overall,
            audit_trail=audit,
            model_used=used_model,
        )

    try:
        prompt = _build_prompt(evidence)
        raw_text = _grok_chat(prompt, key, used_model)

        issue_exps, suggestions, overall = _validate_and_parse_response(raw_text, evidence)
        fallback_issue_exps, fallback_suggestions, fallback_overall = _deterministic_recommendations(report, evidence)

        if not issue_exps:
            issue_exps = fallback_issue_exps
        else:
            fallback_by_name = {item.issue_name.lower(): item for item in fallback_issue_exps}
            fallback_by_rule = {item.rule_id.lower(): item for item in fallback_issue_exps if item.rule_id}
            for item in issue_exps:
                fallback = fallback_by_name.get(item.issue_name.lower()) or fallback_by_rule.get(item.rule_id.lower())
                if fallback is None:
                    continue
                if not item.plain_english_explanation.strip():
                    item.plain_english_explanation = fallback.plain_english_explanation
                if not item.what_happened.strip():
                    item.what_happened = fallback.what_happened
                if not item.repair_steps:
                    item.repair_steps = fallback.repair_steps
                if not item.affected_columns:
                    item.affected_columns = fallback.affected_columns
        if not suggestions:
            suggestions = fallback_suggestions
        if overall is None:
            overall = fallback_overall
        else:
            if not (overall.release_recommendation_text or "").strip():
                overall.release_recommendation_text = fallback_overall.release_recommendation_text
            if not (overall.ai_explanation_of_status or "").strip():
                overall.ai_explanation_of_status = fallback_overall.ai_explanation_of_status
            if not overall.most_important_risks:
                overall.most_important_risks = fallback_overall.most_important_risks
            if not overall.recommended_priority_order:
                overall.recommended_priority_order = fallback_overall.recommended_priority_order

        det_status, det_reason = _compute_deterministic_release_status(report)
        if overall is None:
            overall = OverallAISummary(
                most_important_risks=[],
                affected_columns=[],
                unresolved_issues=[],
                recommended_priority_order=[],
                release_recommendation_text="",
                deterministic_release_status=det_status,
                deterministic_rationale=det_reason,
                ai_explanation_of_status="",
                confidence_overall=0.5,
                limitations=[],
            )
        else:
            overall.deterministic_release_status = det_status
            overall.deterministic_rationale = det_reason

        audit.append(AIAuditEntry(
            action="analysis_completed",
            timestamp=datetime.utcnow().isoformat() + "Z",
            details={"issues_explained": len(issue_exps),
                     "duration_ms": int((time.time() - started) * 1000)}))

        return AIAnalysisReport(
            status="ok",
            message="Analysis completed",
            issue_explanations=issue_exps,
            suggestions=suggestions,
            overall_summary=overall,
            raw_model_response=raw_text[:1000],  # truncated for safety
            audit_trail=audit,
            model_used=used_model,
        )
    except Exception as e:
        err_msg = str(e)
        # Map common error kinds (never the key value)
        if "quota" in err_msg.lower() or "rate" in err_msg.lower() or "429" in err_msg:
            status = "api_error"
            msg = "Gemini API quota/rate limit exceeded. Please try again later."
        elif (
            "401" in err_msg
            or "403" in err_msg
            or "incorrect api key" in err_msg.lower()
            or "api key" in err_msg.lower()
            or "unauthorized" in err_msg.lower()
        ):
            status = "no_key"
            msg = "Invalid or unauthorized Gemini API key."
        elif "402" in err_msg:
            status = "api_error"
            msg = "Gemini API billing or credits are insufficient for this request."
        elif "404" in err_msg:
            status = "api_error"
            msg = f"Configured Gemini model is unavailable or invalid: {_get_model(model)}"
        else:
            status = "api_error"
            msg = f"Gemini API error: {err_msg[:200]}"

        det_status, det_reason = _compute_deterministic_release_status(report)
        audit.append(AIAuditEntry(action="analysis_failed", timestamp=datetime.utcnow().isoformat() + "Z", details={"error": err_msg[:100], "model": used_model}))
        return _advisory_fallback(
            report, audit, status, msg, det_status, det_reason,
            explanation="Gemini analysis unavailable due to error.")

# Compatibility aliases for existing imports.
analyze_with_grok = analyze_with_gemini
