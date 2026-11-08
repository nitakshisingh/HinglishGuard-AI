"""
HinglishGuard AI - Deep AI Moderation Agent Interface (Page 3).

Provides end-to-end multi-agent evaluation:
1. Supervised fine-tuned XLM-RoBERTa classification.
2. Google Gemini contextual linguistic analysis (identity targeting, sarcasm, negation, slurs).
3. Deterministic Policy Engine arbitration.
4. Transparent side-by-side audit view with disagreement detection.
5. SQLite audit record persistence with automatic Human Review Queue routing.
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
)

# Initialize page config
st.set_page_config(
    page_title="AI Moderation Agent - HinglishGuard AI",
    page_icon="🤖",
    layout="wide",
)

# Ensure database tables exist
init_db()


@st.cache_resource(show_spinner="Loading XLM-RoBERTa Model into memory...")
def load_transformer_model():
    """Cache the fine-tuned transformer model instance."""
    model_path = str(config.TRANSFORMER_MODEL_DIR)
    if not (config.TRANSFORMER_MODEL_DIR / "model.safetensors").exists():
        return None
    try:
        return TransformerClassifier(model_dir=model_path)
    except Exception as exc:
        st.error(f"Failed to load transformer model: {exc}")
        return None


def main():
    st.title("🤖 Multi-Agent AI Moderation Hub")
    st.markdown(
        """
        Evaluate code-mixed Hindi-English (Hinglish), Hindi, and English discourse using
        the **hybrid multi-agent moderation pipeline**:
        - **Supervised Transformer (`xlm-roberta-base`)**: Fast lexical prior and confidence scoring.
        - **Gemini Moderation Agent (`google.genai`)**: Nuanced semantic reasoning, intent analysis, and identity target detection.
        - **Deterministic Policy Engine**: Rule-based arbitration, disagreement enforcement, and non-punitive routing.
        """
    )
    st.markdown("---")

    classifier = load_transformer_model()
    if classifier is None:
        st.error("⚠️ XLM-RoBERTa model checkpoint not found or failed to load. Please check `models/transformers/`.")
        return

    # Text Input Form
    with st.form("agent_moderation_form", clear_on_submit=False):
        user_text = st.text_area(
            "Enter text for full hybrid evaluation (Hinglish, Hindi, or English):",
            height=140,
            placeholder="e.g. People from this community should be hated, ya koi aur Hinglish post...",
            help="Enter text to evaluate across both the Transformer and Gemini AI Agent.",
        )
        submit_btn = st.form_submit_button("⚡ Run Full AI Moderation Pipeline", type="primary", use_container_width=True)

    if submit_btn:
        if not user_text or not user_text.strip():
            st.warning("⚠️ Please provide text before executing moderation.")
            return

        cleaned_text = clean_text(user_text)

        with st.spinner("Executing Transformer classification, Gemini LLM reasoning, and Policy Engine arbitration..."):
            try:
                # 1. Supervised Transformer Inference
                clf_results = classifier.predict(user_text)
                if not clf_results:
                    st.error("Transformer inference produced no output.")
                    return
                clf_res = clf_results[0]
                clf_label = clf_res["label_name"]
                clf_conf = clf_res["confidence"]
                clf_probs = clf_res["probabilities"]

                # 2. Gemini Agent Analysis
                agent = get_default_moderation_agent()
                agent_res = agent.analyze_text(user_text)

                # 3. Policy Engine Evaluation
                policy_engine = get_default_policy_engine()
                policy_decision = policy_engine.evaluate(
                    classifier_result={
                        "label_name": clf_label,
                        "confidence": clf_conf,
                        "probabilities": clf_probs,
                    },
                    agent_result=agent_res,
                )

                # 4. Save to Database
                review_status = (
                    config.REVIEW_STATUS_PENDING
                    if policy_decision["review_required"]
                    else config.REVIEW_STATUS_RESOLVED
                )
                saved_record = save_moderation_record(
                    original_text=user_text,
                    predicted_label=clf_label,
                    confidence=clf_conf,
                    class_probabilities=clf_probs,
                    cleaned_text=cleaned_text,
                    gemini_label=agent_res.get("detected_category"),
                    gemini_explanation=agent_res.get("explanation"),
                    has_disagreement=policy_decision["has_disagreement"],
                    policy_action=policy_decision["policy_action"],
                    review_status=review_status,
                )

                st.success(f"✅ Moderation complete! Audit Record #{saved_record.id} saved to database.")

                # Render Side-by-Side Outputs
                col_left, col_right = st.columns(2)

                # LEFT: Transformer Model Output
                with col_left:
                    st.subheader("1️⃣ XLM-RoBERTa Classifier")
                    st.caption("Supervised text representation model")

                    if clf_label == "Hate":
                        st.error(f"### Predicted: **{clf_label}**")
                    elif clf_label == "Offensive":
                        st.warning(f"### Predicted: **{clf_label}**")
                    else:
                        st.success(f"### Predicted: **{clf_label}**")

                    st.markdown(f"**Confidence:** `{clf_conf*100:.2f}%`")

                    # Probabilities Bar Chart using Plotly
                    import plotly.graph_objects as go
                    categories = ["Neutral", "Offensive", "Hate"]
                    values = [clf_probs.get(c, 0.0) * 100 for c in categories]
                    colors = ["#2ecc71", "#f39c12", "#e74c3c"]

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
                        margin=dict(l=20, r=20, t=20, b=20),
                        height=240,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # RIGHT: Gemini Agent Output
                with col_right:
                    st.subheader("2️⃣ Gemini Moderation Agent")
                    st.caption(f"Contextual reasoning model (`{config.GEMINI_MODEL}`)")

                    if not agent_res.get("is_available"):
                        st.warning("⚠️ **Gemini Agent Unavailable**")
                        st.info(f"{agent_res.get('explanation')}\n\n*Note: Falling back gracefully to classifier alone without fabricating analysis.*")
                    else:
                        g_label = agent_res.get("detected_category", "N/A")
                        if g_label == "Hate":
                            st.error(f"### Detected: **{g_label}**")
                        elif g_label == "Offensive":
                            st.warning(f"### Detected: **{g_label}**")
                        elif g_label == "Uncertain":
                            st.info(f"### Detected: **{g_label}**")
                        else:
                            st.success(f"### Detected: **{g_label}**")

                        m1, m2, m3 = st.columns(3)
                        with m1:
                            st.metric("Identity Attack?", "🚨 Yes" if agent_res.get("is_identity_targeted") else "No")
                        with m2:
                            st.metric("Target Group", agent_res.get("target_group") or "None")
                        with m3:
                            st.metric("Severity", agent_res.get("severity", "Low"))

                        st.markdown(f"**Explanation:** {agent_res.get('explanation')}")
                        if agent_res.get("contextual_signals"):
                            st.markdown(f"**Signals:** `{', '.join(agent_res['contextual_signals'])}`")
                        st.markdown(f"**Agent Review Flag:** `{'Yes' if agent_res.get('requires_human_review') else 'No'}`")

                # Policy Arbitration Section
                st.markdown("---")
                st.subheader("⚖️ Policy Engine Arbitration & Determination")

                p_action = policy_decision["policy_action"]
                disagreement = policy_decision["has_disagreement"]

                if disagreement:
                    st.error(
                        f"🚨 **Disagreement Detected!** "
                        f"Classifier predicted **{clf_label}** ({clf_conf*100:.1f}%), "
                        f"while Gemini identified **{agent_res.get('detected_category')}**."
                    )
                else:
                    st.success("✅ **Alignment**: Classifier and Gemini signals reached consensus.")

                act_col1, act_col2 = st.columns([1, 2])
                with act_col1:
                    if p_action == config.POLICY_ACTION_ESCALATE:
                        st.error(f"### Action: **{p_action}**\n\n📌 Routed to Human Review Queue")
                    elif p_action == config.POLICY_ACTION_FLAG:
                        st.warning(f"### Action: **{p_action}**\n\n⚠️ Content flagged with warning")
                    else:
                        st.success(f"### Action: **{p_action}**\n\n🟢 Content approved")

                with act_col2:
                    st.markdown(f"**Preserved Supervised Label:** `{clf_label}` *(Never silently overwritten)*")
                    st.markdown(f"**Review Queue Status:** `{review_status}`")
                    st.markdown(f"**Policy Rationale:** *{policy_decision['reason']}*")

            except Exception as err:
                st.error(f"❌ An error occurred during pipeline execution: {err}")


if __name__ == "__main__":
    main()
