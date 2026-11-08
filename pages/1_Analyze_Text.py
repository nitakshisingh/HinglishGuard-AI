"""
HinglishGuard AI - Text Analysis Interface (Page 1).

Allows single text input in code-mixed Hindi-English (Hinglish), tokenizes and
evaluates using the fine-tuned XLM-RoBERTa model, displays confidence, class
probabilities, preprocessing diffs, and transparent explanations, and saves
audit records to the SQLite database.

Stage 3C Enhancement:
Includes 'Run AI Contextual Analysis' to orchestrate the Google Gemini Moderation Agent
and Deterministic Policy Engine for deep contextual reasoning, identity targeting,
disagreement detection, and non-punitive policy routing.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd

# Add workspace root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from src.preprocessing import clean_text
from src.transformer_classifier import TransformerClassifier
from src.moderation_agent import get_default_moderation_agent
from src.policy_engine import get_default_policy_engine
from src.database import (
    init_db,
    save_moderation_record,
    update_moderation_record_with_agent,
)

# Initialize page config
st.set_page_config(
    page_title="Analyze Text - HinglishGuard AI",
    page_icon="🔍",
    layout="wide",
)

# Ensure database tables exist
init_db()


@st.cache_resource(show_spinner="Loading XLM-RoBERTa Model into memory...")
def load_transformer_model():
    """Cache the fine-tuned transformer model instance to avoid repeated disk I/O."""
    model_path = str(config.TRANSFORMER_MODEL_DIR)
    if not (config.TRANSFORMER_MODEL_DIR / "model.safetensors").exists():
        return None
    try:
        return TransformerClassifier(model_dir=model_path)
    except Exception as exc:
        st.error(f"Failed to load transformer model: {exc}")
        return None


def get_prediction_explanation(predicted_label: str, confidence: float, probabilities: dict) -> str:
    """Generate a clear, factual, and transparent explanation based on classifier outputs."""
    neutral_prob = probabilities.get("Neutral", 0.0)
    offensive_prob = probabilities.get("Offensive", 0.0)
    hate_prob = probabilities.get("Hate", 0.0)

    if predicted_label == "Hate":
        explanation = (
            f"The model detected high probability ({hate_prob*100:.1f}%) of targeted hate speech, "
            f"communal antagonism, or discriminatory hostility. "
            f"Confidence is {confidence*100:.1f}%. Content is flagged for strict moderation review."
        )
    elif predicted_label == "Offensive":
        explanation = (
            f"The model identified general profanity, vulgarity, or personal insults ({offensive_prob*100:.1f}%) "
            f"without broad identity or group targeting. Confidence is {confidence*100:.1f}%. "
            f"Recommended policy action is to flag or apply profanity filtering."
        )
    else:  # Neutral
        if confidence < config.UNCERTAINTY_THRESHOLD:
            explanation = (
                f"The model classified this text as Neutral with low confidence ({confidence*100:.1f}%). "
                f"Offensive probability is {offensive_prob*100:.1f}% and Hate probability is {hate_prob*100:.1f}%. "
                f"Because confidence is below {config.UNCERTAINTY_THRESHOLD*100:.0f}%, this sample is marked for verification."
            )
        else:
            explanation = (
                f"The model detected benign conversational discourse with {confidence*100:.1f}% confidence. "
                f"Hate ({hate_prob*100:.1f}%) and Offensive ({offensive_prob*100:.1f}%) signals are negligible. "
                f"Safe for publication (Approved)."
            )

    return explanation


def main():
    st.title("🔍 Analyze Hinglish Text")
    st.markdown(
        """
        Classify code-mixed Hindi-English (Hinglish) text into **Neutral**, **Offensive**, or **Hate**
        using the production fine-tuned **XLM-RoBERTa** transformer model, followed by optional
        **Gemini AI Contextual Analysis** and deterministic policy arbitration.
        """
    )
    st.markdown("---")

    # Load model
    classifier = load_transformer_model()

    if classifier is None:
        st.error(
            "⚠️ **XLM-RoBERTa model checkpoint not found or failed to load.**\n\n"
            f"Expected location: `{config.TRANSFORMER_MODEL_DIR}`\n\n"
            "Please ensure `model.safetensors`, `config.json`, and tokenizer files are present."
        )
        return

    # Initialize session state for multi-step analysis
    if "current_analysis" not in st.session_state:
        st.session_state.current_analysis = None

    # Text Input Form
    with st.form("analyze_text_form", clear_on_submit=False):
        user_text = st.text_area(
            "Enter Hinglish sentence or post to analyze:",
            height=140,
            placeholder="e.g. Aaj mausam bahut accha hai, ya phir koi text paste karein...",
            help="Type or paste code-mixed Hinglish text here.",
        )
        submit_button = st.form_submit_button("🚀 Analyze Text", use_container_width=True)

    # Process Initial Transformer Analysis
    if submit_button:
        cleaned_input = clean_text(user_text) if user_text else ""

        if not user_text or not user_text.strip():
            st.warning("⚠️ Please enter some text before analyzing. Blank input cannot be classified.")
            st.session_state.current_analysis = None
            return

        with st.spinner("Analyzing text with XLM-RoBERTa..."):
            try:
                results = classifier.predict(user_text)
                if not results:
                    st.error("Inference returned no output.")
                    return

                res = results[0]
                label = res["label_name"]
                conf = res["confidence"]
                probs = res["probabilities"]

                # Initial classifier policy assessment
                if conf < config.UNCERTAINTY_THRESHOLD:
                    policy_action = config.POLICY_ACTION_ESCALATE
                    review_status = config.REVIEW_STATUS_PENDING
                elif label == "Hate":
                    policy_action = config.POLICY_ACTION_FLAG
                    review_status = config.REVIEW_STATUS_PENDING
                elif label == "Offensive":
                    policy_action = config.POLICY_ACTION_FLAG
                    review_status = config.REVIEW_STATUS_RESOLVED
                else:
                    policy_action = config.POLICY_ACTION_APPROVE
                    review_status = config.REVIEW_STATUS_RESOLVED

                # Save record to database
                saved_record = save_moderation_record(
                    original_text=user_text,
                    predicted_label=label,
                    confidence=conf,
                    class_probabilities=probs,
                    cleaned_text=cleaned_input,
                    gemini_label=None,
                    gemini_explanation=None,
                    has_disagreement=False,
                    policy_action=policy_action,
                    review_status=review_status,
                )

                st.session_state.current_analysis = {
                    "record_id": saved_record.id,
                    "original_text": user_text,
                    "cleaned_text": cleaned_input,
                    "label": label,
                    "confidence": conf,
                    "probabilities": probs,
                    "policy_action": policy_action,
                    "review_status": review_status,
                    "agent_result": None,
                    "policy_result": None,
                }

            except Exception as err:
                st.error(f"❌ An error occurred during inference: {err}")
                return

    # Render Current Analysis Results if available in session state
    analysis = st.session_state.current_analysis
    if analysis:
        st.success(f"✅ Transformer Analysis complete! (Record ID #{analysis['record_id']} logged to database)")

        # Section 1: Transformer Classification
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            st.subheader("Classification Outcome (XLM-RoBERTa)")
            label = analysis["label"]
            conf = analysis["confidence"]
            action = analysis["policy_action"]
            status_val = analysis["review_status"]

            if label == "Hate":
                st.error(f"### 🚨 Predicted: **{label}**")
            elif label == "Offensive":
                st.warning(f"### ⚠️ Predicted: **{label}**")
            else:
                st.success(f"### 🟢 Predicted: **{label}**")

            st.markdown(f"**Model Confidence:** `{conf*100:.2f}%`")
            st.markdown(f"**Baseline Policy Action:** `{action}`")
            st.markdown(f"**Review Queue Status:** `{status_val}`")

        with res_col2:
            st.subheader("Class Probability Distribution")
            import plotly.graph_objects as go
            probs = analysis["probabilities"]
            categories = ["Neutral", "Offensive", "Hate"]
            values = [probs.get(c, 0.0) * 100 for c in categories]
            colors = ["#2ecc71", "#f39c12", "#e74c3c"]  # Green, Orange, Red

            fig = go.Figure(
                data=[
                    go.Bar(
                        x=categories,
                        y=values,
                        text=[f"{v:.1f}%" for v in values],
                        textposition="auto",
                        marker_color=colors,
                    )
                ]
            )
            fig.update_layout(
                yaxis_title="Probability (%)",
                yaxis_range=[0, 100],
                margin=dict(l=20, r=20, t=30, b=20),
                height=280,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Explanation Block
        st.markdown("---")
        st.subheader("💡 Model Reasoning & Explanation")
        explanation = get_prediction_explanation(label, conf, probs)
        st.info(explanation)

        # Expandable Preprocessing Details
        with st.expander("🛠️ View Text Preprocessing Pipeline"):
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.markdown("**Original Input:**")
                st.code(analysis["original_text"], language="text")
            with p_col2:
                st.markdown("**Normalized Text (`clean_text()`):**")
                st.code(analysis["cleaned_text"], language="text")
            st.caption("Applied: NFKC Unicode normalization, URL strip, HTML unescaping, repeated character capping, lowercase preservation.")

        # Section 2: AI Contextual Analysis (Stage 3C)
        st.markdown("---")
        st.subheader("🧠 Deep Contextual AI Moderation (Google Gemini)")
        st.caption(
            "Perform multi-layered semantic evaluation to detect implicit identity targeting, "
            "euphemisms, sarcasm, and code-mixed cultural nuances."
        )

        agent_btn = st.button("🤖 Run AI Contextual Analysis", type="primary", use_container_width=True)

        if agent_btn:
            with st.spinner("Executing Gemini AI contextual analysis and policy arbitration..."):
                agent = get_default_moderation_agent()
                agent_res = agent.analyze_text(analysis["original_text"])

                # Run policy engine arbitration
                clf_dict = {
                    "label_name": analysis["label"],
                    "confidence": analysis["confidence"],
                    "probabilities": analysis["probabilities"],
                }
                policy_engine = get_default_policy_engine()
                policy_decision = policy_engine.evaluate(clf_dict, agent_res)

                # Persist updated determinations into SQLite
                update_moderation_record_with_agent(
                    record_id=analysis["record_id"],
                    gemini_label=agent_res.get("detected_category"),
                    gemini_explanation=agent_res.get("explanation"),
                    has_disagreement=policy_decision["has_disagreement"],
                    policy_action=policy_decision["policy_action"],
                    review_status=config.REVIEW_STATUS_PENDING if policy_decision["review_required"] else config.REVIEW_STATUS_RESOLVED,
                )

                # Update session state
                analysis["agent_result"] = agent_res
                analysis["policy_result"] = policy_decision
                analysis["policy_action"] = policy_decision["policy_action"]
                analysis["review_status"] = config.REVIEW_STATUS_PENDING if policy_decision["review_required"] else config.REVIEW_STATUS_RESOLVED

        # Display Gemini & Policy Results if completed
        if analysis.get("agent_result") and analysis.get("policy_result"):
            agent_res = analysis["agent_result"]
            p_res = analysis["policy_result"]

            st.markdown("#### 📋 AI Contextual Assessment")

            if not agent_res.get("is_available"):
                st.warning(f"⚠️ **Gemini Agent Unavailable**: {agent_res.get('explanation')}")
                st.info("The system safely fell back to the supervised classifier. Gemini responses were not fabricated.")
            else:
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Gemini Category", agent_res.get("detected_category", "N/A"))
                with c2:
                    st.metric("Identity Targeted?", "🚨 Yes" if agent_res.get("is_identity_targeted") else "No")
                with c3:
                    st.metric("Target Group", agent_res.get("target_group") or "None Identified")
                with c4:
                    st.metric("Severity", agent_res.get("severity", "Low"))

                st.markdown(f"**Semantic Explanation:** {agent_res.get('explanation')}")

                if agent_res.get("contextual_signals"):
                    st.markdown(f"**Linguistic Signals:** `{', '.join(agent_res['contextual_signals'])}`")

            # Policy Arbitration Display
            st.markdown("#### ⚖️ Policy Engine Determination")

            if p_res.get("has_disagreement"):
                st.error(
                    f"⚠️ **Disagreement Detected!** "
                    f"Classifier predicted **{analysis['label']}** ({analysis['confidence']*100:.1f}%), "
                    f"while Gemini identified **{p_res.get('gemini_label')}**."
                )
            else:
                st.success("✅ **Consensus Reached**: Classifier and Gemini signals are aligned.")

            # Final Action Card
            act = p_res.get("policy_action")
            if act == config.POLICY_ACTION_ESCALATE:
                st.error(f"### Final Action: **{act}** 🚨\n\nContent routed to Human Moderator queue for review.")
            elif act == config.POLICY_ACTION_FLAG:
                st.warning(f"### Final Action: **{act}** ⚠️\n\nContent flagged for warning/filtering.")
            else:
                st.success(f"### Final Action: **{act}** 🟢\n\nContent approved for publication.")

            st.markdown(f"**Policy Reason:** *{p_res.get('reason')}*")


if __name__ == "__main__":
    main()
