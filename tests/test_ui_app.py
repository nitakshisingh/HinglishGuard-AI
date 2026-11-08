"""
Unit tests for Stage 3 UI helpers and application logic.
Mocks model inference and Streamlit contexts to ensure test suites run fast and
safely without loading large model weights into memory.
"""

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

import config
from src.database import get_database_summary, save_moderation_record

# Dynamic import for module starting with a number
analyze_text_path = Path(__file__).resolve().parent.parent / "pages" / "1_Analyze_Text.py"
spec = importlib.util.spec_from_file_location("analyze_text_module", analyze_text_path)
analyze_text_mod = importlib.util.module_from_spec(spec)
# Patch streamlit before exec to avoid headless UI init issues
# Mock decorator factory for cache_resource
def dummy_decorator(*args, **kwargs):
    def wrapper(f):
        return f
    return wrapper

with patch("streamlit.set_page_config"), patch("streamlit.cache_resource", dummy_decorator):
    spec.loader.exec_module(analyze_text_mod)

get_prediction_explanation = analyze_text_mod.get_prediction_explanation


def test_get_prediction_explanation_hate():
    """Verify explanation generator provides transparent, accurate rationale for Hate predictions."""
    probs = {"Neutral": 0.05, "Offensive": 0.15, "Hate": 0.80}
    explanation = get_prediction_explanation("Hate", 0.80, probs)
    assert "detected high probability (80.0%) of targeted hate speech" in explanation
    assert "flagged for strict moderation review" in explanation


def test_get_prediction_explanation_offensive():
    """Verify explanation generator clarifies general vulgarity/insults for Offensive predictions."""
    probs = {"Neutral": 0.10, "Offensive": 0.85, "Hate": 0.05}
    explanation = get_prediction_explanation("Offensive", 0.85, probs)
    assert "identified general profanity, vulgarity, or personal insults" in explanation
    assert "without broad identity or group targeting" in explanation


def test_get_prediction_explanation_neutral_high_conf():
    """Verify explanation generator approves benign conversation with high confidence."""
    probs = {"Neutral": 0.95, "Offensive": 0.03, "Hate": 0.02}
    explanation = get_prediction_explanation("Neutral", 0.95, probs)
    assert "benign conversational discourse" in explanation
    assert "Safe for publication (Approved)" in explanation


def test_get_prediction_explanation_neutral_low_conf():
    """Verify low-confidence neutral classifications explain uncertainty."""
    probs = {"Neutral": 0.55, "Offensive": 0.35, "Hate": 0.10}
    explanation = get_prediction_explanation("Neutral", 0.55, probs)
    assert "low confidence (55.0%)" in explanation
    assert "marked for verification" in explanation


def test_database_summary_aggregation():
    """Verify get_database_summary accurately calculates record counts and distributions."""
    # Insert test records
    save_moderation_record(
        original_text="Test neutral 1",
        predicted_label="Neutral",
        confidence=0.92,
        class_probabilities={"Neutral": 0.92, "Offensive": 0.05, "Hate": 0.03},
        policy_action=config.POLICY_ACTION_APPROVE,
        review_status=config.REVIEW_STATUS_RESOLVED,
    )
    save_moderation_record(
        original_text="Test hate 1",
        predicted_label="Hate",
        confidence=0.88,
        class_probabilities={"Neutral": 0.02, "Offensive": 0.10, "Hate": 0.88},
        has_disagreement=True,
        policy_action=config.POLICY_ACTION_ESCALATE,
        review_status=config.REVIEW_STATUS_PENDING,
    )

    summary = get_database_summary()
    assert summary["total_records"] >= 2
    assert summary["pending_reviews"] >= 1
    assert summary["disagreements_count"] >= 1
    assert summary["label_counts"]["Neutral"] >= 1
    assert summary["label_counts"]["Hate"] >= 1


def test_system_status_check():
    """Verify system status checker accurately inspects model artifacts without loading them."""
    from app import check_system_status

    status = check_system_status()
    assert isinstance(status, dict)
    assert "transformer" in status
    assert "baseline" in status
    assert "gemini" in status
    assert status["transformer"] is True
    assert status["baseline"] is True
