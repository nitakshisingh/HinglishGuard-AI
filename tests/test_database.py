"""
Unit tests for database module (src.database) and configuration (config).
"""

import os
import pytest
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import config
import src.database as db_module
from src.database import (
    Base,
    ModerationRecord,
    HumanReview,
    init_db,
    save_moderation_record,
    get_moderation_history,
    get_pending_review_items,
    save_human_review,
    update_review_status,
)


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path, monkeypatch):
    """Create an isolated temporary SQLite database for each test."""
    test_db_file = tmp_path / "test_moderation.db"
    test_db_url = f"sqlite:///{test_db_file.as_posix()}"

    # Override database URL and engine in db_module
    monkeypatch.setattr(config, "DATABASE_URL", test_db_url)
    monkeypatch.setattr(db_module, "engine", create_engine(test_db_url, connect_args={"check_same_thread": False}))
    monkeypatch.setattr(db_module, "SessionLocal", sessionmaker(autocommit=False, autoflush=False, bind=db_module.engine))

    # Initialize tables
    init_db()
    yield


class TestDatabaseModule:
    """Test suite for database schema and data operations."""

    def test_save_and_retrieve_moderation_record(self):
        """Test creating a moderation record and retrieving it in history."""
        sample_probs = {"Neutral": 0.95, "Offensive": 0.04, "Hate": 0.01}
        record = save_moderation_record(
            original_text="Aaj ka din bahut accha hai",
            cleaned_text="Aaj ka din bahut accha hai",
            predicted_label="Neutral",
            confidence=0.95,
            class_probabilities=sample_probs,
            policy_action=config.POLICY_ACTION_APPROVE,
            review_status=config.REVIEW_STATUS_RESOLVED,
        )

        assert record.id is not None
        assert record.predicted_label == "Neutral"
        assert record.confidence == 0.95
        assert not record.has_disagreement

        history = get_moderation_history(limit=10)
        assert len(history) == 1
        item = history[0]
        assert item["original_text"] == "Aaj ka din bahut accha hai"
        assert item["class_probabilities"] == sample_probs

    def test_pending_review_queue_and_human_resolution(self):
        """Test that escalated items appear in pending queue and can be resolved by a human."""
        # Save a pending item with disagreement
        record = save_moderation_record(
            original_text="Ambiguous statement with insults",
            predicted_label="Neutral",
            confidence=0.55,
            class_probabilities={"Neutral": 0.55, "Offensive": 0.40, "Hate": 0.05},
            gemini_label="Offensive",
            gemini_explanation="Contains uncivil slang",
            has_disagreement=True,
            policy_action=config.POLICY_ACTION_ESCALATE,
            review_status=config.REVIEW_STATUS_PENDING,
        )

        # Check pending queue
        pending = get_pending_review_items(limit=10)
        assert len(pending) == 1
        assert pending[0]["id"] == record.id
        assert pending[0]["has_disagreement"] is True
        assert pending[0]["review_status"] == config.REVIEW_STATUS_PENDING

        # Human moderator reviews and submits decision
        review = save_human_review(
            moderation_record_id=record.id,
            reviewer_decision="Offensive",
            reviewer_notes="Confirmed colloquial vulgarity",
        )

        assert review.id is not None
        assert review.reviewer_decision == "Offensive"

        # Check that pending queue is now empty
        updated_pending = get_pending_review_items(limit=10)
        assert len(updated_pending) == 0

        # Check that original record is still preserved with updated review_status
        history = get_moderation_history(limit=10)
        assert len(history) == 1
        assert history[0]["review_status"] == config.REVIEW_STATUS_RESOLVED
        assert history[0]["predicted_label"] == "Neutral"  # Original model label is never overwritten

    def test_update_review_status(self):
        """Test manually updating review status of an item."""
        record = save_moderation_record(
            original_text="Random text",
            predicted_label="Neutral",
            confidence=0.9,
            class_probabilities={"Neutral": 0.9, "Offensive": 0.1, "Hate": 0.0},
            review_status=config.REVIEW_STATUS_PENDING,
        )

        success = update_review_status(record.id, config.REVIEW_STATUS_SKIPPED)
        assert success is True

        pending = get_pending_review_items()
        assert len(pending) == 0

        # Updating non-existent record returns False
        assert update_review_status(999999, config.REVIEW_STATUS_RESOLVED) is False


class TestConfigModule:
    """Test suite for configuration parameters and thresholds."""

    def test_config_constants_exist(self):
        """Verify default configuration constants are properly defined."""
        assert config.GEMINI_MODEL is not None
        assert isinstance(config.UNCERTAINTY_THRESHOLD, float)
        assert isinstance(config.HIGH_CONFIDENCE_THRESHOLD, float)
        assert 0.0 <= config.UNCERTAINTY_THRESHOLD <= 1.0
        assert 0.0 <= config.HIGH_CONFIDENCE_THRESHOLD <= 1.0
        assert config.CLASS_NAMES == ["Neutral", "Offensive", "Hate"]
        assert config.ID_TO_LABEL[0] == "Neutral"
        assert config.ID_TO_LABEL[1] == "Offensive"
        assert config.ID_TO_LABEL[2] == "Hate"
