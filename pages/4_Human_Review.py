"""
HinglishGuard AI - Human-in-the-Loop (HITL) Review Dashboard (Page 4).

Provides an operational moderation workflow for human reviewers:
1. Review pending moderation records routed by the Policy Engine.
2. Inspect original text, preprocessed tokens, model probabilities, Gemini reasoning,
   identity-targeting assessments, and disagreement alerts.
3. Confirm model predictions or adjust classification determinations (Neutral, Offensive, Hate).
4. Provide audit notes and record human decisions in the HumanReview table.
5. Updates moderation record status to RESOLVED without altering or deleting original model signals.
6. Filter records (Pending, Resolved, All) and view human determination statistics.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Add workspace root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from src.database import (
    init_db,
    get_records_by_status,
    get_human_review_statistics,
    save_human_review,
)

# Initialize page config
st.set_page_config(
    page_title="Human Review Queue - HinglishGuard AI",
    page_icon="⚖️",
    layout="wide",
)

# Ensure database tables exist
init_db()


def main():
    st.title("⚖️ Human-in-the-Loop (HITL) Moderation Queue")
    st.markdown(
        """
        Review, verify, and arbitrate content escalated by the **Policy Engine** due to low classifier confidence,
        model-agent disagreement, or detected identity attacks. Original model and agent predictions
        are permanently preserved; reviewer determinations are recorded in a separate audit log.
        """
    )
    st.markdown("---")

    # Fetch Review Statistics
    stats = get_human_review_statistics()

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(
            label="Pending In Queue",
            value=stats["pending_count"],
            delta=f"{stats['pending_count']} awaiting review" if stats["pending_count"] else None,
            delta_color="inverse",
        )
    with m2:
        st.metric(label="Resolved Records", value=stats["resolved_count"])
    with m3:
        st.metric(label="Total Human Determinations Logged", value=stats["total_reviews"])

    # Distribution of Human Decisions (if any exist)
    if stats["human_decision_distribution"]:
        with st.expander("📊 Human Decision Breakdown (Resolved Records)", expanded=False):
            dist = stats["human_decision_distribution"]
            fig_hist = go.Figure(
                data=[
                    go.Bar(
                        x=list(dist.keys()),
                        y=list(dist.values()),
                        text=list(dist.values()),
                        textposition="auto",
                        marker_color=["#2ecc71", "#f39c12", "#e74c3c", "#3498db"][: len(dist)],
                    )
                ]
            )
            fig_hist.update_layout(
                yaxis_title="Decisions Count",
                margin=dict(l=20, r=20, t=20, b=20),
                height=220,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")

    # Filter selection
    filter_col1, filter_col2 = st.columns([1, 3])
    with filter_col1:
        status_filter = st.selectbox(
            "Filter Queue by Status:",
            options=["PENDING", "RESOLVED", "ALL"],
            index=0,
            help="Show records waiting for moderation or already resolved.",
        )

    # Fetch matching records
    records = get_records_by_status(status_filter=status_filter, limit=50)

    if not records:
        if status_filter == "PENDING":
            st.success("🎉 **Queue is clear!** No moderation records currently require human review.")
            st.info("When low-confidence predictions or agent disagreements occur in **Analyze Text**, they will appear here.")
        else:
            st.info(f"ℹ️ No records found matching filter '{status_filter}'.")
        return

    st.subheader(f"📑 Moderation Items ({len(records)} displayed)")

    for idx, r in enumerate(records):
        rec_id = r["id"]
        is_pending = r["review_status"] == config.REVIEW_STATUS_PENDING
        has_disagree = r["has_disagreement"]

        # Card header styling
        status_badge = "🔴 PENDING REVIEW" if is_pending else "🟢 RESOLVED"
        disagree_badge = "⚠️ DISAGREEMENT" if has_disagree else "✅ CONSENSUS"

        with st.container():
            # Bordered card
            st.markdown(
                f"""
                <div style="border: 1px solid #444; border-radius: 8px; padding: 15px; margin-bottom: 20px; background-color: #1e1e24;">
                    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #333; padding-bottom: 8px;">
                        <span style="font-size: 1.1em; font-weight: bold; color: #f1f2f6;">Record #{rec_id}</span>
                        <span>
                            <span style="background-color: {'#e74c3c' if is_pending else '#2ecc71'}; color: white; padding: 3px 8px; border-radius: 4px; font-size: 0.8em; margin-right: 8px;">{status_badge}</span>
                            <span style="background-color: {'#e67e22' if has_disagree else '#27ae60'}; color: white; padding: 3px 8px; border-radius: 4px; font-size: 0.8em;">{disagree_badge}</span>
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Record details in 2 columns
            c_left, c_right = st.columns([3, 2])

            with c_left:
                st.markdown("**Original Text:**")
                st.info(r["original_text"])

                if r.get("cleaned_text") and r["cleaned_text"] != r["original_text"]:
                    with st.expander("View Preprocessed Text"):
                        st.code(r["cleaned_text"], language="text")

                # Disagreement Alert
                if has_disagree:
                    st.error(
                        f"🚨 **Disagreement**: Transformer classified as **{r['predicted_label']}** "
                        f"while Gemini identified as **{r.get('gemini_label')}**."
                    )

                # Gemini Details
                if r.get("gemini_explanation"):
                    st.markdown(f"**Gemini Reasoning:** *{r['gemini_explanation']}*")

            with c_right:
                st.markdown("**Model Telemetry:**")
                st.markdown(f"- **Transformer Label:** `{r['predicted_label']}` ({r['confidence']*100:.1f}%)")
                st.markdown(f"- **Gemini Label:** `{r.get('gemini_label') or 'Unavailable'}`")
                st.markdown(f"- **Policy Action:** `{r['policy_action']}`")
                st.markdown(f"- **Logged At:** `{r['created_at'][:19].replace('T', ' ')} UTC`")

                # Probability distribution chart
                probs = r.get("class_probabilities", {})
                if probs:
                    cats = ["Neutral", "Offensive", "Hate"]
                    vals = [probs.get(c, 0.0) * 100 for c in cats]
                    mini_fig = go.Figure(
                        data=[
                            go.Bar(
                                x=cats,
                                y=vals,
                                text=[f"{v:.0f}%" for v in vals],
                                textposition="auto",
                                marker_color=["#2ecc71", "#f39c12", "#e74c3c"],
                            )
                        ]
                    )
                    mini_fig.update_layout(
                        yaxis_range=[0, 100],
                        margin=dict(l=10, r=10, t=10, b=10),
                        height=140,
                    )
                    st.plotly_chart(mini_fig, use_container_width=True, key=f"chart_{rec_id}_{idx}")

            # Human Review Form (for PENDING items)
            if is_pending:
                with st.form(key=f"review_form_{rec_id}"):
                    st.markdown("#### ✍️ Human Moderator Determination")

                    # Options matching requirements
                    decision_choice = st.radio(
                        "Assign Final Classification Decision:",
                        options=[
                            f"Confirm Model Decision ({r['predicted_label']})",
                            "Change classification to Neutral",
                            "Change classification to Offensive",
                            "Change classification to Hate",
                        ],
                        index=0,
                        key=f"decision_radio_{rec_id}",
                    )

                    notes = st.text_area(
                        "Moderator Rationale / Audit Notes (Optional):",
                        placeholder="e.g. Sarcasm detected; benign cultural slang; communal insult confirmed...",
                        key=f"notes_{rec_id}",
                    )

                    submit_review = st.form_submit_button("💾 Submit Final Human Determination", type="primary")

                    if submit_review:
                        # Map selection to normalized class
                        if "Neutral" in decision_choice:
                            final_decision = "Neutral"
                        elif "Offensive" in decision_choice:
                            final_decision = "Offensive"
                        elif "Hate" in decision_choice:
                            final_decision = "Hate"
                        else:
                            final_decision = r["predicted_label"]

                        # Save to HumanReview table and update status to RESOLVED
                        save_human_review(
                            moderation_record_id=rec_id,
                            reviewer_decision=final_decision,
                            reviewer_notes=notes.strip() if notes else None,
                        )

                        st.success(f"✅ Record #{rec_id} resolved as **{final_decision}**! Refreshed queue.")
                        st.rerun()

            else:
                # Display existing human review audit logs for resolved records
                reviews = r.get("human_reviews", [])
                if reviews:
                    with st.expander(f"📋 Prior Human Determinations ({len(reviews)})", expanded=True):
                        for rev in reviews:
                            st.markdown(
                                f"- **Decision:** `{rev['reviewer_decision']}` | "
                                f"**Timestamp:** `{rev['reviewed_at'][:19].replace('T', ' ')} UTC`"
                            )
                            if rev.get("reviewer_notes"):
                                st.markdown(f"  *Notes:* {rev['reviewer_notes']}")

            st.markdown("---")


if __name__ == "__main__":
    main()
