"""
Unit tests for Deterministic Policy Engine in HinglishGuard AI.
Tests policy arbitration between classifier predictions and Gemini signals,
uncertainty routing, disagreement detection, identity-hate escalations,
and non-punitive action contracts.
"""

import pytest

import config
from src.policy_engine import PolicyEngine


@pytest.fixture
def policy_engine():
    """Default PolicyEngine fixture."""
    return PolicyEngine(
        uncertainty_threshold=0.70,
        high_confidence_threshold=0.85,
    )


def test_confident_neutral_agreement_approves(policy_engine):
    """High confidence Neutral agreement between classifier and agent should APPROVE."""
    clf = {"label_name": "Neutral", "confidence": 0.92, "probabilities": {"Neutral": 0.92, "Offensive": 0.05, "Hate": 0.03}}
    agent = {"is_available": True, "detected_category": "Neutral", "is_identity_targeted": False, "severity": "Low", "requires_human_review": False}

    decision = policy_engine.evaluate(clf, agent)

    assert decision["policy_action"] == config.POLICY_ACTION_APPROVE
    assert decision["review_required"] is False
    assert decision["final_category"] == "Neutral"
    assert not decision["has_disagreement"]


def test_confident_offensive_agreement_flags(policy_engine):
    """Confident Offensive agreement should FLAG content for filtering/warning without immediate human queue."""
    clf = {"label_name": "Offensive", "confidence": 0.88, "probabilities": {"Neutral": 0.08, "Offensive": 0.88, "Hate": 0.04}}
    agent = {"is_available": True, "detected_category": "Offensive", "is_identity_targeted": False, "severity": "Medium", "requires_human_review": False}

    decision = policy_engine.evaluate(clf, agent)

    assert decision["policy_action"] == config.POLICY_ACTION_FLAG
    assert decision["review_required"] is False
    assert decision["final_category"] == "Offensive"
    assert not decision["has_disagreement"]


def test_confident_hate_agreement_escalates(policy_engine):
    """Confident Hate agreement should FLAG content and require review."""
    clf = {"label_name": "Hate", "confidence": 0.91, "probabilities": {"Neutral": 0.02, "Offensive": 0.07, "Hate": 0.91}}
    # When agent does not report review required and not identity targeted, Rule 5 applies
    agent = {"is_available": True, "detected_category": "Hate", "is_identity_targeted": False, "severity": "High", "requires_human_review": False}

    decision = policy_engine.evaluate(clf, agent)

    # In policy engine, Hate automatically flags and marks review_required = True
    assert decision["policy_action"] == config.POLICY_ACTION_FLAG
    assert decision["review_required"] is True
    assert decision["final_category"] == "Hate"


def test_low_confidence_escalates_to_human_review(policy_engine):
    """Classifier predictions below UNCERTAINTY_THRESHOLD (0.70) must escalate to human review."""
    clf = {"label_name": "Neutral", "confidence": 0.58, "probabilities": {"Neutral": 0.58, "Offensive": 0.32, "Hate": 0.10}}
    agent = {"is_available": True, "detected_category": "Neutral", "is_identity_targeted": False, "severity": "Low", "requires_human_review": False}

    decision = policy_engine.evaluate(clf, agent)

    assert decision["policy_action"] == config.POLICY_ACTION_ESCALATE
    assert decision["review_required"] is True
    assert "Low classifier confidence" in decision["reason"]


def test_disagreement_detection_escalates_to_human_review(policy_engine):
    """When classifier and agent disagree, disagreement must be flagged and routed to human review."""
    # Classifier thinks Neutral (confident 0.82), but Agent detects Hate
    clf = {"label_name": "Neutral", "confidence": 0.82, "probabilities": {"Neutral": 0.82, "Offensive": 0.10, "Hate": 0.08}}
    agent = {"is_available": True, "detected_category": "Hate", "is_identity_targeted": False, "severity": "High", "requires_human_review": True}

    decision = policy_engine.evaluate(clf, agent)

    assert decision["has_disagreement"] is True
    assert decision["policy_action"] == config.POLICY_ACTION_ESCALATE
    assert decision["review_required"] is True
    assert decision["classifier_label"] == "Neutral"
    assert decision["gemini_label"] == "Hate"
    assert "Disagreement detected" in decision["reason"]


def test_identity_targeted_hate_always_escalates(policy_engine):
    """Identity-targeted attacks must escalate to human review immediately."""
    clf = {"label_name": "Hate", "confidence": 0.95, "probabilities": {"Neutral": 0.01, "Offensive": 0.04, "Hate": 0.95}}
    agent = {"is_available": True, "detected_category": "Hate", "is_identity_targeted": True, "target_group": "Minority", "severity": "High", "requires_human_review": True}

    decision = policy_engine.evaluate(clf, agent)

    assert decision["policy_action"] == config.POLICY_ACTION_ESCALATE
    assert decision["review_required"] is True
    assert decision["final_category"] == "Hate"
    assert "identity-targeted hate speech" in decision["reason"]


def test_agent_unavailable_fallback_relies_on_classifier(policy_engine):
    """When agent is unavailable, policy relies on classifier with notes, never assumes safe."""
    clf = {"label_name": "Offensive", "confidence": 0.86, "probabilities": {"Neutral": 0.06, "Offensive": 0.86, "Hate": 0.08}}
    agent = {"is_available": False, "status": "UNAVAILABLE", "detected_category": None}

    decision = policy_engine.evaluate(clf, agent)

    assert decision["has_disagreement"] is False
    assert decision["gemini_label"] is None
    assert decision["policy_action"] == config.POLICY_ACTION_FLAG
    assert "Gemini analysis unavailable" in decision["reason"]


def test_agent_uncertain_flag_escalates_to_review(policy_engine):
    """When agent detects ambiguity ('Uncertain'), it routes to human review."""
    clf = {"label_name": "Neutral", "confidence": 0.75, "probabilities": {"Neutral": 0.75, "Offensive": 0.20, "Hate": 0.05}}
    agent = {"is_available": True, "detected_category": "Uncertain", "is_identity_targeted": False, "severity": "Low", "requires_human_review": True}

    decision = policy_engine.evaluate(clf, agent)

    assert decision["policy_action"] == config.POLICY_ACTION_ESCALATE
    assert decision["review_required"] is True
    assert "ambiguous or borderline linguistic context" in decision["reason"]
