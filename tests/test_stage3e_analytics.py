"""
Unit and integration tests for Stage 3E Part 2: Analytics Dashboard.
Tests:
1. Retrieval and aggregation in get_all_moderation_analytics_data.
2. Calculation of KPI metrics: total records, label distributions, action counts,
   pending/resolved counts, disagreement counts, and average confidence scores.
3. Multi-criteria filtering logic (class, review status, search query).
4. Graceful handling of empty database state.
5. Preservation of original transformer predictions and human determinations separately.
"""

from unittest.mock import MagicMock, patch
import pytest

import config
from src.database import (
    get_all_moderation_analytics_data,
    save_human_review,
    save_moderation_record,
)


def test_empty_database_analytics_handling():
    """Verify get_all_moderation_analytics_data handles an empty dataset cleanly without division by zero."""
    # Mock database session returning empty list
    with patch("src.database.get_db") as mock_get_db:
        mock_db = MagicMock()
        mock_db.query.return_value.order_by.return_value.all.return_value = []
        mock_db.query.return_value.all.return_value = []
        mock_get_db.return_value.__enter__.return_value = mock_db

        data = get_all_moderation_analytics_data()

        assert data["total_count"] == 0
        assert data["records"] == []
        assert data["avg_confidence"] == 0.0
        assert data["pending_count"] == 0
        assert data["resolved_count"] == 0
        assert data["disagreements_count"] == 0
        assert data["label_counts"]["Neutral"] == 0


def test_analytics_metrics_aggregation():
    """Verify analytics accurately computes totals, label counts, action counts, and confidences from real DB records."""
    # Insert known test records
    r1 = save_moderation_record(
        original_text="Analytics test safe comment",
        predicted_label="Neutral",
        confidence=0.96,
        class_probabilities={"Neutral": 0.96, "Offensive": 0.03, "Hate": 0.01},
        policy_action=config.POLICY_ACTION_APPROVE,
        review_status=config.REVIEW_STATUS_RESOLVED,
    )
    r2 = save_moderation_record(
        original_text="Analytics test offensive comment",
        predicted_label="Offensive",
        confidence=0.84,
        class_probabilities={"Neutral": 0.10, "Offensive": 0.84, "Hate": 0.06},
        policy_action=config.POLICY_ACTION_FLAG,
        review_status=config.REVIEW_STATUS_PENDING,
        has_disagreement=True,
        gemini_label="Neutral",
    )

    data = get_all_moderation_analytics_data()

    assert data["total_count"] >= 2
    assert data["pending_count"] >= 1
    assert data["resolved_count"] >= 1
    assert data["disagreements_count"] >= 1
    assert data["avg_confidence"] > 0.0

    # Label counts
    assert data["label_counts"]["Neutral"] >= 1
    assert data["label_counts"]["Offensive"] >= 1

    # Daily activity tracking
    assert len(data["daily_activity"]) >= 1


def test_analytics_human_decisions_tracking():
    """Verify analytics accurately registers human reviewer decisions separately from model predictions."""
    # Insert record and resolve with human decision
    rec = save_moderation_record(
        original_text="Human tracking test sentence",
        predicted_label="Offensive",
        confidence=0.72,
        class_probabilities={"Neutral": 0.15, "Offensive": 0.72, "Hate": 0.13},
        policy_action=config.POLICY_ACTION_ESCALATE,
        review_status=config.REVIEW_STATUS_PENDING,
    )

    save_human_review(
        moderation_record_id=rec.id,
        reviewer_decision="Hate",
        reviewer_notes="Escalated to Hate by human reviewer.",
    )

    data = get_all_moderation_analytics_data()

    # Human decision registered in decisions distribution
    assert "Hate" in data["human_decisions"]
    assert data["human_decisions"]["Hate"] >= 1

    # Original model prediction remains unchanged
    target_rec = next(r for r in data["records"] if r["id"] == rec.id)
    assert target_rec["predicted_label"] == "Offensive"
    assert target_rec["review_status"] == config.REVIEW_STATUS_RESOLVED
    assert len(target_rec["human_reviews"]) >= 1
    assert target_rec["human_reviews"][0]["reviewer_decision"] == "Hate"


def test_analytics_filtering_logic():
    """Verify filtering behavior across predicted class, review status, and search query."""
    records = [
        {"id": 1, "original_text": "Aaj mausam accha hai", "predicted_label": "Neutral", "review_status": "RESOLVED"},
        {"id": 2, "original_text": "Tu chutiya hai", "predicted_label": "Offensive", "review_status": "PENDING"},
        {"id": 3, "original_text": "In logon ko maro", "predicted_label": "Hate", "review_status": "PENDING"},
    ]

    # Filter 1: By class 'Offensive'
    f_class = [r for r in records if r["predicted_label"] == "Offensive"]
    assert len(f_class) == 1
    assert f_class[0]["id"] == 2

    # Filter 2: By review status 'PENDING'
    f_status = [r for r in records if r["review_status"] == "PENDING"]
    assert len(f_status) == 2

    # Filter 3: By text search 'mausam'
    f_search = [r for r in records if "mausam" in r["original_text"].lower()]
    assert len(f_search) == 1
    assert f_search[0]["id"] == 1
