"""
Integration tests for Stage 3C: Gemini AI Moderation Agent and Policy Engine integration.
Tests agreement, disagreement, implicit identity-hate detection, neutral community mentions,
counterspeech, negation, quoted statements, missing API key fallback, API failure fallback,
and SQLite persistence + human review routing with mocks (no live API calls or 1.11GB model loads).
"""

from unittest.mock import MagicMock, patch
import pytest

import config
from src.database import (
    get_moderation_history,
    get_pending_review_items,
    save_moderation_record,
    update_moderation_record_with_agent,
)
from src.moderation_agent import GeminiModerationResponse, ModerationAgent
from src.policy_engine import PolicyEngine


@pytest.fixture
def policy_engine():
    """Default PolicyEngine fixture."""
    return PolicyEngine()


def test_classifier_and_gemini_agreement_neutral(policy_engine):
    """Test agreement on benign Neutral text leads to APPROVE and no human review."""
    clf = {"label_name": "Neutral", "confidence": 0.95, "probabilities": {"Neutral": 0.95, "Offensive": 0.03, "Hate": 0.02}}
    agent = {
        "is_available": True,
        "detected_category": "Neutral",
        "is_identity_targeted": False,
        "target_group": None,
        "severity": "Low",
        "explanation": "Friendly, benign conversation.",
        "contextual_signals": ["friendly greeting"],
        "requires_human_review": False,
    }

    decision = policy_engine.evaluate(clf, agent)

    assert decision["policy_action"] == config.POLICY_ACTION_APPROVE
    assert decision["review_required"] is False
    assert decision["has_disagreement"] is False
    assert decision["final_category"] == "Neutral"


def test_classifier_and_gemini_disagreement_escalation(policy_engine):
    """Disagreement between classifier and Gemini must set has_disagreement=True and ESCALATE."""
    clf = {"label_name": "Offensive", "confidence": 0.82, "probabilities": {"Neutral": 0.10, "Offensive": 0.82, "Hate": 0.08}}
    agent = {
        "is_available": True,
        "detected_category": "Neutral",
        "is_identity_targeted": False,
        "severity": "Low",
        "explanation": "Colloquial friendly banter between peers, not hostile insult.",
        "requires_human_review": False,
    }

    decision = policy_engine.evaluate(clf, agent)

    assert decision["has_disagreement"] is True
    assert decision["policy_action"] == config.POLICY_ACTION_ESCALATE
    assert decision["review_required"] is True
    assert decision["classifier_label"] == "Offensive"
    assert decision["gemini_label"] == "Neutral"


def test_problematic_implicit_identity_hate_example(policy_engine):
    """Verify that 'People from this community should be hated' escalates via agent disagreement and identity attack."""
    # Classifier misclassified as Neutral (99.2%)
    clf = {"label_name": "Neutral", "confidence": 0.992, "probabilities": {"Neutral": 0.992, "Offensive": 0.007, "Hate": 0.001}}
    # Gemini accurately detects implicit identity incitement
    agent = {
        "is_available": True,
        "detected_category": "Hate",
        "is_identity_targeted": True,
        "target_group": "Targeted Community",
        "severity": "High",
        "explanation": "Promotes collective hostility and hatred targeting an identity community.",
        "contextual_signals": ["identity targeting", "incitement to hatred"],
        "requires_human_review": True,
    }

    decision = policy_engine.evaluate(clf, agent)

    # Must escalate to human review and preserve classifier label
    assert decision["policy_action"] == config.POLICY_ACTION_ESCALATE
    assert decision["review_required"] is True
    assert decision["has_disagreement"] is True
    assert decision["classifier_label"] == "Neutral"
    assert decision["gemini_label"] == "Hate"
    assert "identity-targeted hate speech" in decision["reason"]


def test_neutral_community_mentions_not_flagged(policy_engine):
    """Mentions of community or religion without hate or hostility must remain Neutral / Approved."""
    clf = {"label_name": "Neutral", "confidence": 0.98, "probabilities": {"Neutral": 0.98, "Offensive": 0.01, "Hate": 0.01}}
    agent = {
        "is_available": True,
        "detected_category": "Neutral",
        "is_identity_targeted": False,
        "target_group": "Community",
        "severity": "Low",
        "explanation": "Respectful discussion of community events.",
        "contextual_signals": ["positive sentiment"],
        "requires_human_review": False,
    }

    decision = policy_engine.evaluate(clf, agent)

    assert decision["policy_action"] == config.POLICY_ACTION_APPROVE
    assert decision["review_required"] is False
    assert decision["has_disagreement"] is False


def test_counterspeech_against_hate(policy_engine):
    """Counterspeech opposing hatred should be approved when both models recognize peaceful intent."""
    clf = {"label_name": "Neutral", "confidence": 0.96, "probabilities": {"Neutral": 0.96, "Offensive": 0.03, "Hate": 0.01}}
    agent = {
        "is_available": True,
        "detected_category": "Neutral",
        "is_identity_targeted": False,
        "severity": "Low",
        "explanation": "Counterspeech actively discouraging communal hatred.",
        "contextual_signals": ["counterspeech", "anti-hate"],
        "requires_human_review": False,
    }

    decision = policy_engine.evaluate(clf, agent)

    assert decision["policy_action"] == config.POLICY_ACTION_APPROVE
    assert decision["review_required"] is False
    assert not decision["has_disagreement"]


def test_negated_hostility(policy_engine):
    """Negated hostility ('I do not hate this community') must be evaluated as Neutral."""
    clf = {"label_name": "Neutral", "confidence": 0.94, "probabilities": {"Neutral": 0.94, "Offensive": 0.04, "Hate": 0.02}}
    agent = {
        "is_available": True,
        "detected_category": "Neutral",
        "is_identity_targeted": False,
        "severity": "Low",
        "explanation": "Negation phrase explicitly rejecting hatred.",
        "contextual_signals": ["negation"],
        "requires_human_review": False,
    }

    decision = policy_engine.evaluate(clf, agent)

    assert decision["policy_action"] == config.POLICY_ACTION_APPROVE
    assert decision["review_required"] is False


def test_quoted_hateful_statement_ambiguity(policy_engine):
    """Quoted hatred flagged by agent for review must trigger ESCALATE_TO_HUMAN_REVIEW."""
    clf = {"label_name": "Neutral", "confidence": 0.91, "probabilities": {"Neutral": 0.91, "Offensive": 0.06, "Hate": 0.03}}
    agent = {
        "is_available": True,
        "detected_category": "Uncertain",
        "is_identity_targeted": True,
        "target_group": "Reported Group",
        "severity": "Medium",
        "explanation": "Quoting hate speech in a reporting or criticism context requires human judgment.",
        "contextual_signals": ["quoted speech", "meta-discussion"],
        "requires_human_review": True,
    }

    decision = policy_engine.evaluate(clf, agent)

    assert decision["policy_action"] == config.POLICY_ACTION_ESCALATE
    assert decision["review_required"] is True


def test_missing_gemini_api_key_graceful_fallback(policy_engine):
    """When Gemini API key is missing or unconfigured, system falls back to classifier without fabrication."""
    agent_mock = ModerationAgent(api_key="")
    agent_res = agent_mock.analyze_text("Some text")

    assert agent_res["is_available"] is False
    assert agent_res["status"] == "UNAVAILABLE"
    assert agent_res["detected_category"] is None

    clf = {"label_name": "Neutral", "confidence": 0.89, "probabilities": {"Neutral": 0.89, "Offensive": 0.08, "Hate": 0.03}}
    decision = policy_engine.evaluate(clf, agent_res)

    assert decision["policy_action"] == config.POLICY_ACTION_APPROVE
    assert decision["gemini_label"] is None
    assert "Gemini analysis unavailable" in decision["reason"]


def test_gemini_api_failure_fallback(policy_engine):
    """When Gemini raises a network or API exception, system handles it gracefully and routes safely."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = TimeoutError("API Gateway Timeout")

    agent = ModerationAgent(api_key="valid-dummy-key", client=mock_client)
    agent_res = agent.analyze_text("Test sentence")

    assert agent_res["is_available"] is False
    assert agent_res["status"] == "ERROR"
    assert agent_res["requires_human_review"] is True

    clf = {"label_name": "Offensive", "confidence": 0.78, "probabilities": {"Neutral": 0.12, "Offensive": 0.78, "Hate": 0.10}}
    decision = policy_engine.evaluate(clf, agent_res)

    assert decision["gemini_label"] is None
    assert decision["policy_action"] == config.POLICY_ACTION_FLAG


def test_sqlite_persistence_and_pending_review_routing():
    """Verify that records with review_required=True are saved and properly retrievable from the pending review queue."""
    # Step 1: Initial classification record saved
    rec = save_moderation_record(
        original_text="People from this community should be hated.",
        predicted_label="Neutral",
        confidence=0.992,
        class_probabilities={"Neutral": 0.992, "Offensive": 0.007, "Hate": 0.001},
        cleaned_text="People from this community should be hated.",
        gemini_label=None,
        gemini_explanation=None,
        has_disagreement=False,
        policy_action=config.POLICY_ACTION_APPROVE,
        review_status=config.REVIEW_STATUS_RESOLVED,
    )
    assert rec.id is not None

    # Step 2: Agent analysis runs and discovers disagreement/identity attack -> updates record
    updated = update_moderation_record_with_agent(
        record_id=rec.id,
        gemini_label="Hate",
        gemini_explanation="High severity identity-targeted hate speech.",
        has_disagreement=True,
        policy_action=config.POLICY_ACTION_ESCALATE,
        review_status=config.REVIEW_STATUS_PENDING,
    )

    assert updated is not None
    assert updated.gemini_label == "Hate"
    assert updated.has_disagreement is True
    assert updated.policy_action == config.POLICY_ACTION_ESCALATE
    assert updated.review_status == config.REVIEW_STATUS_PENDING

    # Step 3: Verify it appears in pending review queue
    pending_items = get_pending_review_items()
    matching = [item for item in pending_items if item["id"] == rec.id]
    assert len(matching) == 1
    assert matching[0]["has_disagreement"] is True
    assert matching[0]["predicted_label"] == "Neutral"
    assert matching[0]["gemini_label"] == "Hate"
