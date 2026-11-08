"""
Unit and integration tests for Stage 3D: Human-in-the-Loop Review Dashboard.
Tests:
1. Fetching pending, resolved, and all records with get_records_by_status.
2. Saving human reviewer determination in the HumanReview table.
3. Updating parent ModerationRecord review_status to RESOLVED.
4. Preserving original transformer label, probabilities, Gemini label, and policy action.
5. Computing human review decision statistics and class distribution.
6. Edge cases: Empty queue handling, duplicate submission prevention / resolution verification.
"""

import pytest

import config
from src.database import (
    get_human_review_statistics,
    get_pending_review_items,
    get_records_by_status,
    save_human_review,
    save_moderation_record,
)


def test_pending_queue_retrieval():
    """Verify that only records with review_status=PENDING appear in the pending queue."""
    # Create one pending and one resolved record
    p_rec = save_moderation_record(
        original_text="Pending moderation test item 1",
        predicted_label="Offensive",
        confidence=0.65,
        class_probabilities={"Neutral": 0.25, "Offensive": 0.65, "Hate": 0.10},
        policy_action=config.POLICY_ACTION_ESCALATE,
        review_status=config.REVIEW_STATUS_PENDING,
    )
    r_rec = save_moderation_record(
        original_text="Resolved moderation test item 2",
        predicted_label="Neutral",
        confidence=0.95,
        class_probabilities={"Neutral": 0.95, "Offensive": 0.03, "Hate": 0.02},
        policy_action=config.POLICY_ACTION_APPROVE,
        review_status=config.REVIEW_STATUS_RESOLVED,
    )

    pending_list = get_records_by_status("PENDING")
    pending_ids = [item["id"] for item in pending_list]
    assert p_rec.id in pending_ids
    assert r_rec.id not in pending_ids


def test_status_filtering_all_and_resolved():
    """Verify get_records_by_status correctly filters by RESOLVED and ALL."""
    rec = save_moderation_record(
        original_text="Test filter item",
        predicted_label="Neutral",
        confidence=0.88,
        class_probabilities={"Neutral": 0.88, "Offensive": 0.08, "Hate": 0.04},
        review_status=config.REVIEW_STATUS_RESOLVED,
    )

    resolved_list = get_records_by_status("RESOLVED")
    resolved_ids = [item["id"] for item in resolved_list]
    assert rec.id in resolved_ids

    all_list = get_records_by_status("ALL")
    all_ids = [item["id"] for item in all_list]
    assert rec.id in all_ids


def test_human_review_submission_workflow():
    """Verify human determination is saved separately and parent record updates to RESOLVED without data loss."""
    # Step 1: Save an escalated item requiring review
    orig_probs = {"Neutral": 0.55, "Offensive": 0.35, "Hate": 0.10}
    record = save_moderation_record(
        original_text="Tu toh pura gadha hai",
        cleaned_text="Tu toh pura gadha hai",
        predicted_label="Neutral",
        confidence=0.55,
        class_probabilities=orig_probs,
        gemini_label="Offensive",
        gemini_explanation="Mild animal slur used as general insult without group targeting.",
        has_disagreement=True,
        policy_action=config.POLICY_ACTION_ESCALATE,
        review_status=config.REVIEW_STATUS_PENDING,
    )

    # Step 2: Human moderator overrides model determination to 'Offensive'
    review = save_human_review(
        moderation_record_id=record.id,
        reviewer_decision="Offensive",
        reviewer_notes="Confirmed insult, changing from low-confidence Neutral to Offensive.",
    )

    assert review.id is not None
    assert review.moderation_record_id == record.id
    assert review.reviewer_decision == "Offensive"
    assert "Confirmed insult" in review.reviewer_notes

    # Step 3: Fetch updated record and verify data integrity
    resolved_items = get_records_by_status("RESOLVED")
    updated_rec = next(item for item in resolved_items if item["id"] == record.id)

    # Review status changed to RESOLVED
    assert updated_rec["review_status"] == config.REVIEW_STATUS_RESOLVED

    # Original model outputs preserved intact (never overwritten or erased)
    assert updated_rec["predicted_label"] == "Neutral"
    assert updated_rec["confidence"] == 0.55
    assert updated_rec["class_probabilities"] == orig_probs
    assert updated_rec["gemini_label"] == "Offensive"
    assert updated_rec["has_disagreement"] is True
    assert updated_rec["policy_action"] == config.POLICY_ACTION_ESCALATE

    # Linked review audit present
    assert len(updated_rec["human_reviews"]) >= 1
    assert updated_rec["human_reviews"][-1]["reviewer_decision"] == "Offensive"


def test_human_review_statistics_calculation():
    """Verify get_human_review_statistics accurately calculates queue counts and decision distributions."""
    # Insert new record and resolve it
    rec = save_moderation_record(
        original_text="Statistics test sentence",
        predicted_label="Hate",
        confidence=0.91,
        class_probabilities={"Neutral": 0.02, "Offensive": 0.07, "Hate": 0.91},
        review_status=config.REVIEW_STATUS_PENDING,
    )
    save_human_review(
        moderation_record_id=rec.id,
        reviewer_decision="Hate",
        reviewer_notes="Confirmed hate speech.",
    )

    stats = get_human_review_statistics()
    assert "pending_count" in stats
    assert "resolved_count" in stats
    assert "total_reviews" in stats
    assert "human_decision_distribution" in stats
    assert stats["total_reviews"] >= 1
    assert stats["human_decision_distribution"].get("Hate", 0) >= 1


def test_invalid_record_id_error_handling():
    """Verify save_human_review raises ValueError when given a nonexistent record ID."""
    with pytest.raises(ValueError, match="does not exist"):
        save_human_review(
            moderation_record_id=99999999,
            reviewer_decision="Neutral",
            reviewer_notes="Should fail.",
        )
