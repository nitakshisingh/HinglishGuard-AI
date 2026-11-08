"""
Policy Engine Module for HinglishGuard AI.

Implements deterministic content moderation rules by synthesizing:
1. Supervised model predictions (XLM-RoBERTa or TF-IDF Baseline) & probabilities.
2. Optional Gemini LLM contextual analysis.
3. Predefined safety rules, uncertainty thresholds, and disagreement tracking.

Guiding Principles:
- Preserves the original classifier label (never silently overwritten).
- Detects and logs disagreements between the classifier and Gemini agent.
- Routes low-confidence (< UNCERTAINTY_THRESHOLD) or conflicting predictions to Human Review.
- Non-punitive: Employs APPROVE, FLAG, or ESCALATE_TO_HUMAN_REVIEW without auto-deleting content.
- Safe fallbacks when Gemini analysis is unavailable (never assumes missing agent = clean content).
"""

from typing import Any, Dict, List, Optional
import config


class PolicyEngine:
    """Deterministic rule and arbitration engine for HinglishGuard AI."""

    def __init__(
        self,
        uncertainty_threshold: float = config.UNCERTAINTY_THRESHOLD,
        high_confidence_threshold: float = config.HIGH_CONFIDENCE_THRESHOLD,
    ):
        self.uncertainty_threshold = uncertainty_threshold
        self.high_confidence_threshold = high_confidence_threshold

    def evaluate(
        self,
        classifier_result: Dict[str, Any],
        agent_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluate classifier output and optional agent analysis against moderation policy.

        Args:
            classifier_result: Dict from classifier prediction containing:
                - label_name: 'Neutral', 'Offensive', 'Hate'
                - confidence: float (0.0 to 1.0)
                - probabilities: dict of class name -> prob
            agent_result: Optional dict from ModerationAgent.analyze_text() containing:
                - is_available: bool
                - detected_category: 'Neutral', 'Offensive', 'Hate', 'Uncertain', or None
                - is_identity_targeted: bool
                - severity: 'Low', 'Medium', 'High'
                - requires_human_review: bool

        Returns:
            Dictionary containing:
                - classifier_label: str ('Neutral', 'Offensive', 'Hate')
                - gemini_label: Optional[str]
                - has_disagreement: bool
                - final_category: str
                - policy_action: 'APPROVE', 'FLAG', or 'ESCALATE_TO_HUMAN_REVIEW'
                - review_required: bool
                - reason: str (human-readable explanation of the policy decision)
        """
        clf_label = classifier_result.get("label_name", "Neutral")
        confidence = float(classifier_result.get("confidence", 0.0))
        agent_data = agent_result or {"is_available": False}

        agent_available = agent_data.get("is_available", False)
        agent_label = agent_data.get("detected_category") if agent_available else None
        is_targeted = agent_data.get("is_identity_targeted", False) if agent_available else False
        agent_severity = agent_data.get("severity", "Low") if agent_available else "Low"
        agent_wants_review = agent_data.get("requires_human_review", False) if agent_available else False

        # 1. Detect and record disagreement
        has_disagreement = False
        if agent_available and agent_label and agent_label != "Uncertain":
            # Compare non-uncertain agent category to classifier label
            if agent_label != clf_label:
                has_disagreement = True

        # Policy arbitration variables
        policy_action = config.POLICY_ACTION_APPROVE
        review_required = False
        reasons: List[str] = []
        final_category = clf_label  # Supervised model label is the default primary anchor

        # --- Rule 1: High Severity Identity-Targeted Hate Escalation ---
        if agent_available and is_targeted and (agent_label == "Hate" or clf_label == "Hate"):
            policy_action = config.POLICY_ACTION_ESCALATE
            review_required = True
            final_category = "Hate"
            reasons.append("High-severity identity-targeted hate speech detected.")

        # --- Rule 2: Classifier vs. Agent Disagreement Escalation ---
        elif has_disagreement:
            policy_action = config.POLICY_ACTION_ESCALATE
            review_required = True
            # Flag disagreement; retain classifier label but mark conflict clearly
            reasons.append(
                f"Disagreement detected: Classifier predicted '{clf_label}' ({confidence*100:.1f}%), "
                f"while LLM Agent identified '{agent_label}'."
            )

        # --- Rule 3: Low Classifier Confidence (< UNCERTAINTY_THRESHOLD) ---
        elif confidence < self.uncertainty_threshold:
            policy_action = config.POLICY_ACTION_ESCALATE
            review_required = True
            reasons.append(
                f"Low classifier confidence ({confidence*100:.1f}% < {self.uncertainty_threshold*100:.1f}% threshold)."
            )

        # --- Rule 4: Agent explicitly flagged uncertainty or human review required ---
        elif agent_available and (agent_label == "Uncertain" or agent_wants_review):
            policy_action = config.POLICY_ACTION_ESCALATE
            review_required = True
            reasons.append("LLM Agent flagged ambiguous or borderline linguistic context requiring human review.")

        # --- Rule 5: High Confidence Hate Speech Confirmation ---
        elif clf_label == "Hate":
            # Confirmed Hate: Action is FLAG (non-punitive; marks for moderator inspection)
            policy_action = config.POLICY_ACTION_FLAG
            review_required = True
            final_category = "Hate"
            reasons.append(f"Classified as Hate Speech with {confidence*100:.1f}% confidence.")

        # --- Rule 6: High Confidence Offensive / Profanity ---
        elif clf_label == "Offensive":
            # Offensive / Profane content: Action is FLAG for notice
            policy_action = config.POLICY_ACTION_FLAG
            review_required = False
            final_category = "Offensive"
            reasons.append(f"Identified as Offensive/Profane language with {confidence*100:.1f}% confidence.")

        # --- Rule 7: Clean / Neutral Content ---
        else:
            # Confident Neutral content
            policy_action = config.POLICY_ACTION_APPROVE
            review_required = False
            final_category = "Neutral"
            reasons.append(f"Classified as Neutral with {confidence*100:.1f}% confidence.")

        # Note when Gemini was unavailable
        if not agent_available:
            reasons.append("Note: Decision based solely on classifier (Gemini analysis unavailable).")

        return {
            "classifier_label": clf_label,
            "gemini_label": agent_label,
            "has_disagreement": has_disagreement,
            "final_category": final_category,
            "policy_action": policy_action,
            "review_required": review_required,
            "confidence": confidence,
            "reason": " | ".join(reasons),
        }


# Global singleton instance
_DEFAULT_POLICY_ENGINE = None


def get_default_policy_engine() -> PolicyEngine:
    """Return singleton instance of PolicyEngine."""
    global _DEFAULT_POLICY_ENGINE
    if _DEFAULT_POLICY_ENGINE is None:
        _DEFAULT_POLICY_ENGINE = PolicyEngine()
    return _DEFAULT_POLICY_ENGINE


def evaluate_policy(
    classifier_result: Dict[str, Any],
    agent_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate classifier and agent outputs using the default policy engine.

    Args:
        classifier_result: Prediction dictionary from classifier.
        agent_result: Optional analysis dictionary from moderation agent.

    Returns:
        Structured policy evaluation decision.
    """
    engine = get_default_policy_engine()
    return engine.evaluate(classifier_result, agent_result)
