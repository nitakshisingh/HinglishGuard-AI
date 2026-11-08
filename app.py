"""
HinglishGuard AI - Main Application & Executive Dashboard.

Hybrid Hinglish Hate Speech Detection & Intelligent Content Moderation System.
Provides system health monitoring, model artifact verification, moderation metrics,
and multi-page navigation across the platform.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd

# Add workspace root to Python path
sys.path.append(str(Path(__file__).resolve().parent))

import config
from src.database import init_db, get_database_summary, get_moderation_history

# Initialize page configuration
st.set_page_config(
    page_title="HinglishGuard AI - Content Moderation Hub",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ensure database tables exist
init_db()


def check_system_status():
    """Verify availability of model artifacts and configured services."""
    transformer_exists = (
        config.TRANSFORMER_MODEL_DIR.exists()
        and (config.TRANSFORMER_MODEL_DIR / "model.safetensors").exists()
    )
    baseline_exists = (
        config.BASELINE_MODEL_PATH.exists()
        and config.BASELINE_VEC_PATH.exists()
    )
    gemini_status = config.IS_GEMINI_AVAILABLE

    return {
        "transformer": transformer_exists,
        "baseline": baseline_exists,
        "gemini": gemini_status,
    }


def main():
    # Sidebar Navigation & Overview
    st.sidebar.title("🛡️ HinglishGuard AI")
    st.sidebar.markdown(
        """
        **Hybrid Moderation System**
        *Fine-Tuned XLM-RoBERTa + Gemini Agent + Policy Engine*
        """
    )
    st.sidebar.markdown("---")
    st.sidebar.info(
        "📌 **Navigate to:**\n"
        "- **1. Analyze Text**: Single Hinglish text analysis\n"
        "- **2. Model Lab**: Model comparison & benchmarks\n"
        "- **3. AI Moderation Agent**: Deep LLM context analysis\n"
        "- **4. Human Review**: HITL escalation queue\n"
        "- **5. Analytics**: System-wide moderation telemetry"
    )

    # Main Dashboard Header
    st.title("🛡️ HinglishGuard AI Dashboard")
    st.markdown(
        """
        Welcome to **HinglishGuard AI**, an enterprise-grade content safety platform designed specifically for
        code-mixed Hindi-English (**Hinglish**) social discourse. Combining deep transformer representation
        learning (`xlm-roberta-base`) with contextual LLM reasoning and deterministic policy enforcement.
        """
    )

    st.markdown("---")

    # Section 1: System & Model Infrastructure Health
    st.subheader("⚙️ System Infrastructure & Model Health")
    status = check_system_status()

    col1, col2, col3 = st.columns(3)

    with col1:
        if status["transformer"]:
            st.success("✅ **XLM-RoBERTa Model**\n\nStatus: *Active & Loaded locally*\n\nArch: `xlm-roberta-base`")
        else:
            st.error("❌ **XLM-RoBERTa Model**\n\nStatus: *Artifacts Missing*\n\nCheck `models/transformers/`")

    with col2:
        if status["baseline"]:
            st.success("✅ **TF-IDF Baseline**\n\nStatus: *Ready*\n\nLogistic Regression (Char n-grams)")
        else:
            st.warning("⚠️ **TF-IDF Baseline**\n\nStatus: *Missing Artifacts*\n\nTrain baseline first")

    with col3:
        if status["gemini"]:
            st.success(f"✅ **Gemini Agent**\n\nStatus: *Configured*\n\nModel: `{config.GEMINI_MODEL}`")
        else:
            st.warning("⚠️ **Gemini Agent**\n\nStatus: *Unconfigured/Fallback*\n\nSet `GEMINI_API_KEY` in `.env`")

    st.markdown("---")

    # Section 2: Moderation Summary Metrics
    st.subheader("📊 Moderation Metrics & Queue Telemetry")
    summary = get_database_summary()

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(label="Total Moderated Records", value=summary["total_records"])
    with m2:
        st.metric(label="Pending Human Review", value=summary["pending_reviews"], delta=f"{summary['pending_reviews']} items" if summary["pending_reviews"] else None)
    with m3:
        st.metric(label="Resolved Reviews", value=summary["resolved_reviews"])
    with m4:
        st.metric(label="Agent Disagreements", value=summary["disagreements_count"])

    # Distribution breakdown
    if summary["total_records"] > 0:
        c1, c2 = st.columns(2)
        import plotly.graph_objects as go
        with c1:
            st.caption("Class Distribution of Processed Content")
            l_cats = list(summary["label_counts"].keys())
            l_vals = list(summary["label_counts"].values())
            fig_labels = go.Figure(data=[go.Bar(x=l_cats, y=l_vals, marker_color=["#2ecc71", "#f39c12", "#e74c3c"])])
            fig_labels.update_layout(yaxis_title="Count", margin=dict(l=20, r=20, t=20, b=20), height=250)
            st.plotly_chart(fig_labels, use_container_width=True)
        with c2:
            st.caption("Policy Actions Taken")
            a_cats = list(summary["policy_action_counts"].keys())
            a_vals = list(summary["policy_action_counts"].values())
            fig_actions = go.Figure(data=[go.Bar(x=a_cats, y=a_vals, marker_color="#3498db")])
            fig_actions.update_layout(yaxis_title="Count", margin=dict(l=20, r=20, t=20, b=20), height=250)
            st.plotly_chart(fig_actions, use_container_width=True)

    st.markdown("---")

    # Section 3: Recent Moderation Activity
    st.subheader("🕒 Recent Moderation Activity")
    recent = get_moderation_history(limit=10)

    if not recent:
        st.info("ℹ️ No moderation activity logged yet. Visit **1. Analyze Text** in the sidebar to run your first classification.")
    else:
        # Render clean HTML table to avoid st.dataframe's pyarrow dependency
        import html
        table_rows = []
        for r in recent:
            probs = r.get("class_probabilities", {})
            hate_prob = f"{probs.get('Hate', 0.0)*100:.1f}%" if probs else "N/A"
            table_rows.append({
                "ID": str(r["id"]),
                "Original Text": r["original_text"][:60] + ("..." if len(r["original_text"]) > 60 else ""),
                "Prediction": r["predicted_label"],
                "Confidence": f"{r['confidence']*100:.1f}%",
                "Hate Prob": hate_prob,
                "Policy Action": r["policy_action"],
                "Review Status": r["review_status"],
                "Disagreement": "⚠️ Yes" if r["has_disagreement"] else "No",
                "Timestamp (UTC)": r["created_at"][:19].replace("T", " ") if r["created_at"] else "N/A",
            })

        headers = ["ID", "Original Text", "Prediction", "Confidence", "Hate Prob", "Policy Action", "Review Status", "Disagreement", "Timestamp (UTC)"]
        html_table = '<div style="overflow-x:auto;"><table style="width:100%; border-collapse: collapse; font-size: 0.9em;">'
        html_table += '<thead style="background-color: #262730; color: white;"><tr>'
        for h in headers:
            html_table += f'<th style="padding: 10px; border: 1px solid #444; text-align: left;">{html.escape(h)}</th>'
        html_table += '</tr></thead><tbody>'
        for row in table_rows:
            html_table += '<tr style="border-bottom: 1px solid #333;">'
            for h in headers:
                val = row[h]
                color_style = ""
                if h == "Prediction":
                    if val == "Hate":
                        color_style = "color: #ff4b4b; font-weight: bold;"
                    elif val == "Offensive":
                        color_style = "color: #ffa421; font-weight: bold;"
                    else:
                        color_style = "color: #21c354; font-weight: bold;"
                html_table += f'<td style="padding: 8px; border: 1px solid #333; {color_style}">{html.escape(str(val))}</td>'
            html_table += '</tr>'
        html_table += '</tbody></table></div>'

        st.markdown(html_table, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.caption("🛡️ HinglishGuard AI • Built with PyTorch, Hugging Face Transformers, Google GenAI SDK, and Streamlit.")


if __name__ == "__main__":
    main()
