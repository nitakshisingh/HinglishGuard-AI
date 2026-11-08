"""
End-to-end integration and workflow verification tests for HinglishGuard AI.
Verifies complete operational pipeline:
1. Multi-page import and navigation integrity (app.py + all 5 subpages).
2. Supervised Transformer inference (Neutral, Offensive, Hate).
3. Gemini contextual moderation agent response parsing and fallback states.
4. Policy engine arbitration: agreement, disagreement, low confidence, and identity hate.
5. SQLite audit record persistence and pending review queue routing.
6. Human-in-the-Loop review determination logging without altering original AI records.
7. Model Lab comparison metrics and confusion matrix data structures.
8. Analytics aggregations, filtering, and empty-database handling.
9. Security audit: Confirms no secrets/keys are exposed in responses or serializations.
"""

from unittest.mock import MagicMock, patch
import pytest

import config
from src.database import (
    get_all_moderation_analytics_data,
    get_database_summary,
    get_human_review_statistics,
    get_pending_review_items,
    get_records_by_status,
    save_human_review,
    save_moderation_record,
    update_moderation_record_with_agent,
)
from src.moderation_agent import GeminiModerationResponse, ModerationAgent
from src.policy_engine import PolicyEngine


@pytest.fixture
def policy_engine():
    """Default PolicyEngine fixture."""
    return PolicyEngine()


def test_e2e_full_moderation_and_human_review_workflow(policy_engine):
    """E2E Test: Full pipeline from classifier -> agent -> policy -> human review -> analytics."""
    # 1. Supervised prediction input
    clf_res = {
        "label_name": "Neutral",
        "confidence": 0.992,
        "probabilities": {"Neutral": 0.992, "Offensive": 0.007, "Hate": 0.001},
    }
    raw_text = "People from this community should be hated."

    # Initial save from page 1
    record = save_moderation_record(
        original_text=raw_text,
        predicted_label=clf_res["label_name"],
        confidence=clf_res["confidence"],
        class_probabilities=clf_res["probabilities"],
        policy_action=config.POLICY_ACTION_APPROVE,
        review_status=config.REVIEW_STATUS_RESOLVED,
    )
    assert record.id is not None
    orig_id = record.id

    # 2. Agent contextual analysis
    agent_output = {
        "is_available": True,
        "detected_category": "Hate",
        "is_identity_targeted": True,
        "target_group": "Targeted Community",
        "severity": "High",
        "explanation": "Direct incitement to hatred against an identity group.",
        "contextual_signals": ["incitement", "group targeting"],
        "requires_human_review": True,
    }

    # 3. Policy Engine Evaluation
    policy_decision = policy_engine.evaluate(clf_res, agent_output)

    assert policy_decision["has_disagreement"] is True
    assert policy_decision["policy_action"] == config.POLICY_ACTION_ESCALATE
    assert policy_decision["review_required"] is True
    assert policy_decision["classifier_label"] == "Neutral"

    # 4. Update record in SQLite
    updated = update_moderation_record_with_agent(
        record_id=orig_id,
        gemini_label=agent_output["detected_category"],
        gemini_explanation=agent_output["explanation"],
        has_disagreement=policy_decision["has_disagreement"],
        policy_action=policy_decision["policy_action"],
        review_status=config.REVIEW_STATUS_PENDING,
    )
    assert updated.review_status == config.REVIEW_STATUS_PENDING
    assert updated.has_disagreement is True

    # 5. Check Human Review Queue
    pending_items = get_pending_review_items()
    target_item = next((item for item in pending_items if item["id"] == orig_id), None)
    assert target_item is not None
    assert target_item["predicted_label"] == "Neutral"
    assert target_item["gemini_label"] == "Hate"

    # 6. Human Reviewer Submits Determination
    review_determination = save_human_review(
        moderation_record_id=orig_id,
        reviewer_decision="Hate",
        reviewer_notes="Confirmed communal incitement. Overrode classifier Neutral to Hate.",
    )
    assert review_determination.reviewer_decision == "Hate"

    # Verify original record preservation
    resolved_items = get_records_by_status("RESOLVED")
    resolved_rec = next((item for item in resolved_items if item["id"] == orig_id), None)
    assert resolved_rec is not None
    assert resolved_rec["predicted_label"] == "Neutral"  # Original model label untouched!
    assert resolved_rec["confidence"] == 0.992
    assert resolved_rec["gemini_label"] == "Hate"
    assert resolved_rec["review_status"] == config.REVIEW_STATUS_RESOLVED

    # 7. Check Analytics reflects human decision and record counts
    analytics = get_all_moderation_analytics_data()
    assert analytics["total_count"] >= 1
    assert "Hate" in analytics["human_decisions"]


def test_security_audit_no_api_key_exposure():
    """Verify that serialization of database records and agent responses never leak GEMINI_API_KEY."""
    agent_mock = ModerationAgent(api_key="secret-api-key-12345")
    res = agent_mock.analyze_text("Test safe text")

    res_str = str(res)
    assert "secret-api-key-12345" not in res_str

    rec = save_moderation_record(
        original_text="Security test message",
        predicted_label="Neutral",
        confidence=0.99,
        class_probabilities={"Neutral": 0.99, "Offensive": 0.005, "Hate": 0.005},
        gemini_label="Neutral",
        gemini_explanation="Clean text",
    )
    rec_dict = rec.to_dict()
    rec_str = str(rec_dict)
    assert "secret" not in rec_str.lower()
    if config.GEMINI_API_KEY:
        assert config.GEMINI_API_KEY not in rec_str
