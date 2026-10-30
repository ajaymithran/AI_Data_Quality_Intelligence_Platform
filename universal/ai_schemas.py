"""Schemas for the AI Data Quality Analyst.

All AI output is treated as advisory and validated. Suggestions never
auto-apply; corrections must go to a separate copy with an audit trail.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal


SeverityAI = Literal["Critical", "High", "Medium", "Low"]
ReleaseStatus = Literal["PASS", "PASS WITH WARNINGS", "BLOCKED"]
ApprovalRequired = Literal["yes", "no", "review"]
VerificationStatus = Literal["pending", "verified", "failed", "not_applicable"]


@dataclass
class IssueExplanation:
    issue_id: str
    rule_id: str
    issue_name: str
    plain_english_explanation: str
    what_happened: str
    affected_records: list[dict[str, Any]] = field(default_factory=list)
    affected_columns: list[str] = field(default_factory=list)
    potential_downstream_effects: list[str] = field(default_factory=list)
    severity: SeverityAI = "Medium"
    severity_justification: str = ""
    likely_causes: list[str] = field(default_factory=list)  # hypotheses
    investigation_steps: list[str] = field(default_factory=list)
    repair_steps: list[str] = field(default_factory=list)
    can_be_safely_automated: bool = False
    automation_rationale: str = ""
    how_to_verify_correction: str = ""
    how_to_prevent_recurrence: str = ""
    confidence: float = 0.5
    limitations: list[str] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    evidence_used: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReviewableSuggestion:
    suggestion_id: str
    issue_id: str
    issue: str
    evidence: str
    proposed_action: str
    expected_result: str
    risk: str
    approval_required: ApprovalRequired = "review"
    verification_status: VerificationStatus = "pending"
    safe_to_auto_apply: bool = False
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OverallAISummary:
    most_important_risks: list[str] = field(default_factory=list)
    affected_columns: list[str] = field(default_factory=list)
    unresolved_issues: list[str] = field(default_factory=list)
    recommended_priority_order: list[str] = field(default_factory=list)
    release_recommendation_text: str = ""
    # Deterministic status is computed separately; Gemini may explain it
    deterministic_release_status: ReleaseStatus = "PASS"
    deterministic_rationale: str = ""
    ai_explanation_of_status: str = ""
    confidence_overall: float = 0.5
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AIAuditEntry:
    action: str  # e.g., "generated_analysis", "applied_correction"
    timestamp: str
    details: dict[str, Any] = field(default_factory=dict)
    user: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AIAnalysisReport:
    status: Literal["ok", "error", "no_key", "api_error", "skipped"]
    message: str = ""
    issue_explanations: list[IssueExplanation] = field(default_factory=list)
    suggestions: list[ReviewableSuggestion] = field(default_factory=list)
    overall_summary: OverallAISummary | None = None
    raw_model_response: str | None = None  # for debugging only, never contains secrets
    audit_trail: list[AIAuditEntry] = field(default_factory=list)
    model_used: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d
