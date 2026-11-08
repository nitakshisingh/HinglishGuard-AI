"""
HinglishGuard AI - System-Wide Moderation Analytics Dashboard (Page 5).

Provides comprehensive telemetry, operational metrics, and audit reporting
derived directly from live SQLite database records:
1. Executive KPI Cards: Total texts analyzed, class distribution counts, pending/resolved reviews,
   agent disagreements, and average model confidence.
2. Interactive Plotly Visualizations:
   - Moderation class distribution (Neutral, Offensive, Hate).
   - Policy action breakdown (APPROVE, FLAG, ESCALATE_TO_HUMAN_REVIEW).
   - Moderation activity throughput over time (daily trends).
   - Human reviewer determination distribution.
   - Transformer vs. Gemini disagreement breakdown.
3. Multi-Criteria Filtering:
   - Date range selector.
   - Predicted class filter.
   - Review status filter (Pending, Resolved, All).
4. Searchable Moderation Audit History:
   - Full-text search over original Hinglish discourse.
   - Clean, responsive HTML audit table avoiding pyarrow DLL blocks on Windows.
5. Graceful empty-state handling for fresh databases.
"""

import sys
from pathlib import Path
from datetime import datetime
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import html

# Add workspace root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from src.database import init_db, get_all_moderation_analytics_data

# Initialize page config
st.set_page_config(
    page_title="Moderation Analytics - HinglishGuard AI",
    page_icon="📈",
    layout="wide",
)

# Ensure database tables exist
init_db()


def main():
    st.title("📈 System-Wide Moderation Analytics")
    st.markdown(
        """
        Operational telemetry, model performance metrics, and compliance audit trail
        derived in real time from **HinglishGuard AI** production database records.
        """
    )
    st.markdown("---")

    # Fetch live analytics data
    analytics = get_all_moderation_analytics_data()
    total_records = analytics["total_count"]

    if total_records == 0:
        st.info("ℹ️ **No moderation records logged yet.**")
        st.markdown(
            """
            The database is currently empty. Visit **1. Analyze Text** or **3. AI Moderation Agent**
            in the sidebar to classify Hinglish text and generate live moderation telemetry.
            """
        )
        return

    # Section 1: Executive KPI Cards
    st.subheader("📌 Executive Telemetry & KPI Overview")

    l_counts = analytics["label_counts"]
    avg_conf = analytics["avg_confidence"]

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.metric("Total Analyzed", f"{total_records:,}")
    with kpi2:
        st.metric("Neutral (Safe)", f"{l_counts.get('Neutral', 0):,}", f"{(l_counts.get('Neutral', 0)/total_records)*100:.1f}%")
    with kpi3:
        st.metric("Offensive", f"{l_counts.get('Offensive', 0):,}", f"{(l_counts.get('Offensive', 0)/total_records)*100:.1f}%")
    with kpi4:
        st.metric("Hate Speech", f"{l_counts.get('Hate', 0):,}", f"{(l_counts.get('Hate', 0)/total_records)*100:.1f}%", delta_color="inverse")
    with kpi5:
        st.metric("Avg Confidence", f"{avg_conf*100:.1f}%")

    sub_k1, sub_k2, sub_k3, sub_k4 = st.columns(4)
    with sub_k1:
        st.metric("Pending Human Review", f"{analytics['pending_count']:,}", delta=f"{analytics['pending_count']} awaiting" if analytics['pending_count'] else None, delta_color="inverse")
    with sub_k2:
        st.metric("Resolved Human Reviews", f"{analytics['resolved_count']:,}")
    with sub_k3:
        st.metric("Model-Agent Disagreements", f"{analytics['disagreements_count']:,}", delta=f"{(analytics['disagreements_count']/total_records)*100:.1f}% of total" if total_records else None, delta_color="off")
    with sub_k4:
        total_rev = len(analytics.get("human_decisions", {}))
        st.metric("Human Determinations Logged", f"{sum(analytics.get('human_decisions', {}).values()):,}")

    st.markdown("---")

    # Section 2: Interactive Plotly Visualizations
    st.subheader("📊 Interactive Telemetry Visualizations")

    row1_c1, row1_c2 = st.columns(2)

    with row1_c1:
        st.markdown("**1. Moderation Class Distribution**")
        classes = ["Neutral", "Offensive", "Hate"]
        c_vals = [l_counts.get(c, 0) for c in classes]
        colors = ["#2ecc71", "#f39c12", "#e74c3c"]
        fig_classes = go.Figure(
            data=[
                go.Bar(
                    x=classes,
                    y=c_vals,
                    text=[f"{v} ({v/total_records*100:.1f}%)" if total_records else "0" for v in c_vals],
                    textposition="auto",
                    marker_color=colors,
                )
            ]
        )
        fig_classes.update_layout(
            yaxis_title="Count",
            margin=dict(l=20, r=20, t=20, b=20),
            height=260,
        )
        st.plotly_chart(fig_classes, use_container_width=True)

    with row1_c2:
        st.markdown("**2. Policy Engine Actions Breakdown**")
        a_counts = analytics["action_counts"]
        act_names = list(a_counts.keys())
        act_vals = list(a_counts.values())
        fig_actions = go.Figure(
            data=[
                go.Bar(
                    x=act_names,
                    y=act_vals,
                    text=[f"{v} ({v/total_records*100:.1f}%)" if total_records else "0" for v in act_vals],
                    textposition="auto",
                    marker_color=["#2ecc71", "#f39c12", "#e74c3c"][: len(act_names)],
                )
            ]
        )
        fig_actions.update_layout(
            yaxis_title="Count",
            margin=dict(l=20, r=20, t=20, b=20),
            height=260,
        )
        st.plotly_chart(fig_actions, use_container_width=True)

    row2_c1, row2_c2 = st.columns(2)

    with row2_c1:
        st.markdown("**3. Moderation Throughput Over Time**")
        daily = analytics["daily_activity"]
        if daily:
            sorted_days = sorted(daily.keys())
            day_counts = [daily[d] for d in sorted_days]
            fig_time = go.Figure(
                data=[
                    go.Scatter(
                        x=sorted_days,
                        y=day_counts,
                        mode="lines+markers",
                        line=dict(color="#3498db", width=3),
                        marker=dict(size=7, color="#2980b9"),
                    )
                ]
            )
            fig_time.update_layout(
                xaxis_title="Date",
                yaxis_title="Volume",
                margin=dict(l=20, r=20, t=20, b=20),
                height=260,
            )
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.info("No timeline data available.")

    with row2_c2:
        st.markdown("**4. Human Moderator Determinations Distribution**")
        h_decisions = analytics["human_decisions"]
        if h_decisions:
            h_labels = list(h_decisions.keys())
            h_vals = list(h_decisions.values())
            fig_human = go.Figure(
                data=[
                    go.Bar(
                        x=h_labels,
                        y=h_vals,
                        text=[str(v) for v in h_vals],
                        textposition="auto",
                        marker_color=["#2ecc71", "#f39c12", "#e74c3c", "#9b59b6"][: len(h_labels)],
                    )
                ]
            )
            fig_human.update_layout(
                yaxis_title="Decisions Count",
                margin=dict(l=20, r=20, t=20, b=20),
                height=260,
            )
            st.plotly_chart(fig_human, use_container_width=True)
        else:
            st.info("No human determinations recorded yet. Visit **4. Human Review** to arbitrate pending items.")

    st.markdown("---")

    # Section 3: Multi-Criteria Filter Controls
    st.subheader("🔍 Searchable Moderation Audit Log")

    all_records = analytics["records"]

    f_col1, f_col2, f_col3 = st.columns(3)

    with f_col1:
        class_filter = st.selectbox(
            "Filter by Predicted Class:",
            options=["ALL", "Neutral", "Offensive", "Hate"],
            index=0,
        )

    with f_col2:
        status_filter = st.selectbox(
            "Filter by Review Status:",
            options=["ALL", "PENDING", "RESOLVED"],
            index=0,
        )

    with f_col3:
        search_query = st.text_input(
            "Search Text Content:",
            placeholder="Type keyword or phrase to search...",
            help="Filter audit history by matching substring in original text.",
        )

    # Apply filters
    filtered_records = []
    for r in all_records:
        # Class filter
        if class_filter != "ALL" and r.get("predicted_label") != class_filter:
            continue
        # Status filter
        if status_filter != "ALL" and r.get("review_status") != status_filter:
            continue
        # Search query
        if search_query and search_query.strip():
            q = search_query.strip().lower()
            orig = (r.get("original_text") or "").lower()
            if q not in orig:
                continue
        filtered_records.append(r)

    st.caption(f"Showing **{len(filtered_records)}** of **{total_records}** total records")

    # Render Styled HTML Table
    if not filtered_records:
        st.warning("⚠️ No records match your current filter criteria.")
    else:
        table_rows = []
        for r in filtered_records[:100]:  # Limit displayed rows to 100
            probs = r.get("class_probabilities", {})
            hate_prob = f"{probs.get('Hate', 0.0)*100:.1f}%" if probs else "N/A"
            gemini_display = r.get("gemini_label") or "N/A"
            table_rows.append({
                "ID": str(r["id"]),
                "Original Text": r["original_text"][:65] + ("..." if len(r["original_text"]) > 65 else ""),
                "Transformer Label": r["predicted_label"],
                "Confidence": f"{r['confidence']*100:.1f}%",
                "Hate Prob": hate_prob,
                "Gemini Label": gemini_display,
                "Policy Action": r["policy_action"],
                "Review Status": r["review_status"],
                "Disagreement": "⚠️ Yes" if r["has_disagreement"] else "No",
                "Logged (UTC)": r["created_at"][:19].replace("T", " ") if r["created_at"] else "N/A",
            })

        headers = [
            "ID",
            "Original Text",
            "Transformer Label",
            "Confidence",
            "Hate Prob",
            "Gemini Label",
            "Policy Action",
            "Review Status",
            "Disagreement",
            "Logged (UTC)",
        ]

        html_table = '<div style="overflow-x:auto;"><table style="width:100%; border-collapse: collapse; font-size: 0.88em;">'
        html_table += '<thead style="background-color: #262730; color: white;"><tr>'
        for h in headers:
            html_table += f'<th style="padding: 9px; border: 1px solid #444; text-align: left;">{html.escape(h)}</th>'
        html_table += '</tr></thead><tbody>'

        for row in table_rows:
            html_table += '<tr style="border-bottom: 1px solid #333;">'
            for h in headers:
                val = row[h]
                style = ""
                if h == "Transformer Label":
                    if val == "Hate":
                        style = "color: #ff4b4b; font-weight: bold;"
                    elif val == "Offensive":
                        style = "color: #ffa421; font-weight: bold;"
                    else:
                        style = "color: #21c354; font-weight: bold;"
                elif h == "Disagreement" and "Yes" in val:
                    style = "color: #ff4b4b; font-weight: bold;"
                elif h == "Review Status" and val == "PENDING":
                    style = "color: #ff4b4b; font-weight: bold;"
                elif h == "Review Status" and val == "RESOLVED":
                    style = "color: #21c354;"

                html_table += f'<td style="padding: 8px; border: 1px solid #333; {style}">{html.escape(str(val))}</td>'
            html_table += '</tr>'

        html_table += '</tbody></table></div>'
        st.markdown(html_table, unsafe_allow_html=True)

    # Section 4: Average Confidence Summary Table
    st.markdown("---")
    st.subheader("🎯 Mean Prediction Confidence by Category")
    avg_by_label = analytics["avg_confidence_by_label"]
    conf_cards = st.columns(3)
    with conf_cards[0]:
        st.metric("Neutral Mean Confidence", f"{avg_by_label.get('Neutral', 0.0)*100:.1f}%")
    with conf_cards[1]:
        st.metric("Offensive Mean Confidence", f"{avg_by_label.get('Offensive', 0.0)*100:.1f}%")
    with conf_cards[2]:
        st.metric("Hate Mean Confidence", f"{avg_by_label.get('Hate', 0.0)*100:.1f}%")


if __name__ == "__main__":
    main()
