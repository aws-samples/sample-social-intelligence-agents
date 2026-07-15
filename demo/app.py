"""Streamlit demo -- social intelligence lead generation via Amazon Bedrock AgentCore streaming.

Streams real-time agent events from the deployed Amazon Bedrock AgentCore Runtime endpoint.
Shows live activity feed with node transitions, tool calls, and results.

Usage:
    streamlit run demo/app.py

Modules:
    config.py       -- Configuration, env loading, constants
    data.py         -- DynamoDB data layer and config checks
    streaming.py    -- AgentCore streaming client and SSE parser
    components.py   -- Reusable UI rendering components
"""

import csv
import io
import json
import os
import sys
import time
from urllib.parse import urlparse

import streamlit as st
from markupsafe import escape as _esc

# Ensure demo/ sibling modules are importable when Streamlit runs this file directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from components import (  # noqa: E402
    extract_result_preview,
    format_elapsed,
    render_empty_state,
    render_lead_card,
    score_bar,
    score_dot,
)
from config import (  # noqa: E402
    AWS_REGION,
    LEAD_TYPES,
    SCORE_HIGH_THRESHOLD,
    SCORE_MID_THRESHOLD,
    VERTICALS,
)
from data import check_config, get_existing_lead_ids, get_leads, normalize_trends  # noqa: E402
from streaming import invoke_agentcore_streaming  # noqa: E402

from social_intelligence.tools.email_renderer import render_email_html  # noqa: E402

# ---------------------------------------------------------------------------
# Page config and global styles
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AnyCompany -- Social Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stApp { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px; padding: 20px; margin: 8px 0;
        border: 1px solid #2d3748;
    }
    .metric-card h3 { color: #a0aec0; font-size: 11px; text-transform: uppercase;
        letter-spacing: 1px; margin: 0 0 8px; font-weight: 500; }
    .metric-card p { color: #ffffff; font-size: 28px; font-weight: 700; margin: 0; }
    .metric-card .delta { font-size: 12px; margin-top: 4px; }
    .lead-card {
        background: #1a202c; border-radius: 10px; padding: 18px; margin: 10px 0;
        border-left: 4px solid #4299e1; transition: transform 0.15s;
    }
    .lead-card:hover { transform: translateX(2px); }
    .score-high { border-left-color: #38a169; }
    .score-mid { border-left-color: #d69e2e; }
    .score-low { border-left-color: #e53e3e; }
    .tag-badge {
        display: inline-block; background: #2d3748; color: #e2e8f0;
        font-size: 11px; padding: 2px 8px; border-radius: 10px; margin: 2px;
    }
    .dist-row {
        display: flex; align-items: center; margin: 6px 0; font-size: 13px;
    }
    .dist-label { color: #a0aec0; min-width: 80px; }
    .dist-count { color: #e2e8f0; min-width: 100px; text-align: right; font-weight: 500; }
    .config-ok { color: #68d391; font-size: 12px; }
    .config-err { color: #fc8181; font-size: 12px; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🔍 Social Intelligence")
    st.caption("Multi-agent prospect discovery and outreach")
    st.divider()

    st.markdown("### 🎯 Lead Type")
    lead_type = st.radio(
        "Target",
        list(LEAD_TYPES.keys()),
        format_func=lambda x: f"{LEAD_TYPES[x]['icon']} {x}",
        label_visibility="collapsed",
    )
    st.caption(LEAD_TYPES[lead_type]["description"])
    st.divider()

    st.markdown("### 🏢 Industry Vertical")
    selected_verticals = st.multiselect(
        "Verticals",
        list(VERTICALS.keys()),
        default=["AdTech & Programmatic", "SaaS & Developer Tools"],
        format_func=lambda x: f"{VERTICALS[x]['icon']} {x}",
        label_visibility="collapsed",
    )
    st.divider()

    st.markdown("### ⚙️ Agent Config")
    pattern = st.selectbox(
        "Orchestration",
        ["graph", "swarm"],
        help="Graph: faster, deterministic DAG. Swarm: autonomous handoffs.",
    )
    skip_existing = st.checkbox(
        "Skip existing leads",
        value=True,
        help="Include known lead IDs in the prompt so agents skip already-processed prospects.",
    )
    st.divider()

    # Connection status
    st.markdown("### 📡 Connection")
    config_issues = check_config()
    if config_issues:
        for issue in config_issues:
            # Escape the message before HTML interpolation — it embeds AWS_PROFILE,
            # an env-controlled value, so render it as inert text to prevent XSS.
            st.markdown(f'<span class="config-err">❌ {_esc(issue)}</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="config-ok">✅ AWS credentials OK</span>', unsafe_allow_html=True)
        from config import AGENT_ARN

        runtime_id = AGENT_ARN.split("/")[-1] if AGENT_ARN else "not set"
        st.markdown(f'<span class="config-ok">✅ Runtime: {_esc(runtime_id)}</span>', unsafe_allow_html=True)
    st.divider()

    st.caption(f"Region: {AWS_REGION}")
    st.caption("Model: Claude Sonnet 4 | Mode: Amazon Bedrock AgentCore Streaming")


# ---------------------------------------------------------------------------
# Main content -- 3 tabs
# ---------------------------------------------------------------------------

tab_discover, tab_leads, tab_emails = st.tabs(["🚀 Discover", "📋 Leads", "✉️ Emails"])


# ---------------------------------------------------------------------------
# Tab 1: Discover -- Agent Pipeline
# ---------------------------------------------------------------------------

with tab_discover:
    st.markdown("### Run Agent Pipeline")
    st.caption(f"Multi-agent pipeline via Amazon Bedrock AgentCore Runtime ({pattern} pattern)")

    # Build prompt with optional dedup context
    base_prompt = (
        f"Find recent tech launches and AI tool announcements relevant to "
        f"{', '.join(selected_verticals)}. Focus on {lead_type.lower()} prospects. "
        f"Score each prospect and generate personalized outreach emails for those scoring {SCORE_MID_THRESHOLD}+."
    )

    dedup_context = ""
    if skip_existing:
        existing_ids = get_existing_lead_ids()
        if existing_ids:
            ids_str = ", ".join(existing_ids[:50])
            dedup_context = f"\n\nALREADY PROCESSED (skip these prospect IDs): {ids_str}"

    custom_prompt = st.text_area(
        "Prompt",
        value=base_prompt + dedup_context,
        height=120,
        key="agent_prompt",
    )

    col_btn, col_session = st.columns([1, 2])
    with col_btn:
        run_clicked = st.button(
            "🚀 Run Pipeline",
            use_container_width=True,
            type="primary",
            disabled=bool(config_issues),
        )
    with col_session:
        session_id = st.text_input(
            "Session ID (for memory)",
            value="",
            placeholder="Leave blank for no memory",
            label_visibility="collapsed",
        )

    if config_issues and run_clicked:
        st.warning("Fix configuration issues in the sidebar before running the pipeline.")

    if run_clicked and not config_issues:
        events: list[dict] = []
        st.session_state["agent_events"] = events
        st.session_state["agent_result"] = ""

        progress_placeholder = st.empty()
        progress_placeholder.markdown(
            '<div style="background:#1a202c;border-radius:8px;padding:14px 18px;margin:8px 0;'
            'border:1px solid #2d3748;text-align:center;">'
            '<span style="color:#63b3ed;font-size:13px;font-weight:500;">'
            "⏳ Starting pipeline...</span></div>",
            unsafe_allow_html=True,
        )

        st.markdown("##### Activity Feed")
        event_container = st.container()

        pipeline_start = time.time()

        try:
            result = invoke_agentcore_streaming(
                prompt=custom_prompt,
                pattern=pattern,
                events=events,
                session_id=session_id,
                live_container=event_container,
                status_placeholder=progress_placeholder,
            )
            st.session_state["agent_result"] = result

            elapsed = time.time() - pipeline_start

            # Invalidate leads cache so Leads/Emails tabs show fresh data
            get_leads.clear()

            # Compute summary stats
            has_errors = any(e.get("type") == "error" for e in events)
            has_pipeline_end = any(e.get("type") == "pipeline_end" for e in events)
            completed_nodes = sum(1 for e in events if e.get("type") == "node_end")
            tool_calls = sum(1 for e in events if e.get("type") == "tool_end")
            tool_errors = sum(
                1
                for e in events
                if e.get("type") == "tool_end" and extract_result_preview(e.get("result_preview", ""))[0] == "error"
            )

            summary_parts = []
            if completed_nodes:
                summary_parts.append(f"{completed_nodes} phases")
            if tool_calls:
                err_text = f" ({tool_errors} errors)" if tool_errors else ""
                summary_parts.append(f"{tool_calls} tool calls{err_text}")
            summary_parts.append(format_elapsed(elapsed))
            summary_text = " · ".join(summary_parts)

            if has_errors:
                progress_placeholder.markdown(
                    f'<div style="background:#742a2a;border-radius:8px;padding:14px 18px;margin:8px 0;'
                    f'border:1px solid #e53e3e;text-align:center;">'
                    f'<span style="color:#fed7d7;font-size:13px;font-weight:500;">'
                    f"⚠️ Pipeline finished with errors · {summary_text}</span></div>",
                    unsafe_allow_html=True,
                )
            elif has_pipeline_end:
                progress_placeholder.markdown(
                    f'<div style="background:#1c4532;border-radius:8px;padding:14px 18px;margin:8px 0;'
                    f'border:1px solid #38a169;text-align:center;">'
                    f'<span style="color:#68d391;font-size:13px;font-weight:500;">'
                    f"✅ Pipeline complete · {summary_text}</span></div>",
                    unsafe_allow_html=True,
                )
            else:
                progress_placeholder.markdown(
                    f'<div style="background:#1c4532;border-radius:8px;padding:14px 18px;margin:8px 0;'
                    f'border:1px solid #38a169;text-align:center;">'
                    f'<span style="color:#68d391;font-size:13px;font-weight:500;">'
                    f"✅ Stream closed · {summary_text}</span></div>",
                    unsafe_allow_html=True,
                )
        except Exception as e:
            elapsed = time.time() - pipeline_start
            progress_placeholder.error(f"Pipeline error ({format_elapsed(elapsed)}): {e}")
            events.append({"type": "error", "message": str(e), "ts": time.time()})

        if st.session_state.get("agent_result"):
            with st.expander("📄 Raw Result", expanded=False):
                st.text(st.session_state["agent_result"][:5000])


# ---------------------------------------------------------------------------
# Tab 2: Leads Database
# ---------------------------------------------------------------------------

with tab_leads:
    st.markdown("### 📋 Leads Database")

    col_refresh, col_export, _ = st.columns([1, 1, 4])
    with col_refresh:
        if st.button("🔄 Refresh", use_container_width=True, key="refresh_leads"):
            get_leads.clear()
            st.rerun()

    leads = get_leads()

    with col_export:
        if leads:
            buf = io.StringIO()
            fieldnames = [
                "prospect_id",
                "product_name",
                "score",
                "confidence",
                "signal_strength",
                "data_quality",
                "author",
                "source_url",
                "reasoning",
                "email_subject",
                "discovered_at",
            ]
            writer = csv.DictWriter(buf, fieldnames=fieldnames)
            writer.writeheader()
            for lead in leads:
                row = {}
                for k in fieldnames:
                    v = lead.get(k, "")
                    # DynamoDB returns Decimal for numbers — convert for clean CSV
                    if hasattr(v, "as_integer_ratio"):  # Decimal/float
                        v = int(v) if v == int(v) else float(v)
                    row[k] = v
                writer.writerow(row)
            st.download_button(
                "📥 Export CSV",
                data=buf.getvalue(),
                file_name="social_intel_leads.csv",
                mime="text/csv",
                use_container_width=True,
            )

    if not leads:
        render_empty_state(
            "📋",
            "No leads yet",
            "Run a discovery pipeline in the Discover tab to populate the database.",
        )
    else:
        scores = [int(lead.get("score", 0)) for lead in leads]
        avg_score = sum(scores) / len(scores) if scores else 0
        high_count = sum(1 for s in scores if s >= SCORE_HIGH_THRESHOLD)
        mid_count = sum(1 for s in scores if SCORE_MID_THRESHOLD <= s < SCORE_HIGH_THRESHOLD)
        low_count = sum(1 for s in scores if s < SCORE_MID_THRESHOLD)

        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(
            f'<div class="metric-card"><h3>Total Leads</h3><p>{len(leads)}</p></div>',
            unsafe_allow_html=True,
        )
        m2.markdown(
            f'<div class="metric-card"><h3>Avg Score</h3><p>{avg_score:.0f}</p></div>',
            unsafe_allow_html=True,
        )
        m3.markdown(
            f'<div class="metric-card"><h3>High Quality (≥80)</h3>'
            f'<p>{high_count} <span class="delta" style="color:#38a169;">'
            f"({100 * high_count // len(leads)}%)</span></p></div>",
            unsafe_allow_html=True,
        )
        m4.markdown(
            f'<div class="metric-card"><h3>Latest Discovery</h3>'
            f'<p style="font-size:18px;">{_esc(leads[0].get("discovered_at", "")[:10])}</p></div>',
            unsafe_allow_html=True,
        )

        # Score distribution
        st.markdown("#### Score Distribution")
        total = len(leads)
        for label, count, color, dot in [
            ("80-100", high_count, "#38a169", "🟢"),
            ("60-79", mid_count, "#d69e2e", "🟡"),
            ("< 60", low_count, "#e53e3e", "🔴"),
        ]:
            pct = (count / total * 100) if total else 0
            st.markdown(
                f'<div class="dist-row">'
                f'<span class="dist-label">{dot} {label}</span>'
                f"{score_bar(count, total, color)}"
                f'<span class="dist-count">{count} leads ({pct:.0f}%)</span></div>',
                unsafe_allow_html=True,
            )

        st.divider()

        # Filters
        st.markdown("#### Filters")
        fc1, fc2, fc3 = st.columns([1, 1, 2])
        with fc1:
            score_filter = st.selectbox(
                "Score Range",
                ["All", "≥ 80 (High)", "60-79 (Mid)", "< 60 (Low)"],
                key="score_filter",
            )
        with fc2:
            quality_filter = st.selectbox(
                "Data Quality",
                ["All", "high", "medium", "low"],
                key="quality_filter",
            )
        with fc3:
            search_term = st.text_input(
                "Search products",
                placeholder="Type to filter...",
                key="lead_search",
            )

        # Apply filters
        filtered = leads
        if score_filter == "≥ 80 (High)":
            filtered = [lead for lead in filtered if int(lead.get("score", 0)) >= SCORE_HIGH_THRESHOLD]
        elif score_filter == "60-79 (Mid)":
            filtered = [
                lead for lead in filtered if SCORE_MID_THRESHOLD <= int(lead.get("score", 0)) < SCORE_HIGH_THRESHOLD
            ]
        elif score_filter == "< 60 (Low)":
            filtered = [lead for lead in filtered if int(lead.get("score", 0)) < SCORE_MID_THRESHOLD]
        if quality_filter != "All":
            filtered = [lead for lead in filtered if lead.get("data_quality", "medium") == quality_filter]
        if search_term:
            term = search_term.lower()
            filtered = [
                lead
                for lead in filtered
                if term in lead.get("product_name", "").lower()
                or term in lead.get("author", "").lower()
                or term in lead.get("reasoning", "").lower()
            ]

        st.caption(f"Showing {len(filtered)} of {len(leads)} leads")

        for lead in filtered:
            render_lead_card(lead)


# ---------------------------------------------------------------------------
# Tab 3: Email Preview
# ---------------------------------------------------------------------------

with tab_emails:
    st.markdown("### ✉️ Email Preview")
    st.caption("Rendered HTML emails ready for Amazon SES delivery")

    email_leads = [lead for lead in get_leads() if lead.get("email_body")]

    if not email_leads:
        render_empty_state(
            "✉️",
            "No emails generated yet",
            "Run a discovery pipeline in the Discover tab. Emails are generated for prospects scoring 60+.",
        )
    else:

        def _lead_label(i: int) -> str:
            name = email_leads[i].get("product_name", "Unknown")
            sc = email_leads[i].get("score", 0)
            return f"{score_dot(int(sc))} {name} (score: {sc})"

        selected = st.selectbox(
            "Select Lead",
            range(len(email_leads)),
            format_func=_lead_label,
            key="email_select",
        )
        lead = email_leads[selected]

        col_meta, col_email = st.columns([1, 3])

        with col_meta:
            score = int(lead.get("score", 0))
            confidence = float(lead.get("confidence", 0))
            quality = lead.get("data_quality", "medium")

            st.markdown("#### Lead Metadata")
            st.markdown(
                f'<div style="background:#1a202c;border-radius:8px;padding:14px;'
                f'border:1px solid #2d3748;">'
                f'<div style="margin-bottom:10px;"><span style="color:#718096;font-size:11px;'
                f'text-transform:uppercase;">Score</span><br>'
                f'<span style="color:#e2e8f0;font-size:22px;font-weight:700;">'
                f"{score_dot(score)} {score}/100</span></div>"
                f'<div style="margin-bottom:10px;"><span style="color:#718096;font-size:11px;'
                f'text-transform:uppercase;">Confidence</span><br>'
                f'<span style="color:#e2e8f0;font-size:18px;font-weight:600;">'
                f"{confidence:.0%}</span></div>"
                f'<div style="margin-bottom:10px;"><span style="color:#718096;font-size:11px;'
                f'text-transform:uppercase;">Data Quality</span><br>'
                f'<span style="color:#e2e8f0;font-size:16px;">{_esc(quality.title())}</span></div>'
                f"</div>",
                unsafe_allow_html=True,
            )

            trend_list = normalize_trends(lead.get("top_trends"))
            if trend_list:
                st.markdown("#### Trending Signals")
                for t in trend_list[:5]:
                    st.markdown(
                        f'<span class="tag-badge" style="margin:2px 0;">{_esc(t)}</span>',
                        unsafe_allow_html=True,
                    )

            src_url = lead.get("source_url", "")
            if src_url and urlparse(str(src_url)).scheme in ("http", "https"):
                st.markdown("#### Source")
                st.markdown(f"[🔗 View on source]({src_url})")

        with col_email:
            st.markdown(f"**Subject:** {lead.get('email_subject', '')}")
            st.divider()

            token_list = normalize_trends(lead.get("top_trends"))

            html_result = render_email_html(
                prospect_id=lead.get("prospect_id", ""),
                subject=lead.get("email_subject", ""),
                body=lead.get("email_body", ""),
                personalization_tokens=token_list,
                score=int(lead.get("score", 0)),
                confidence=float(lead.get("confidence", 0)),
                data_quality=lead.get("data_quality", "medium"),
            )
            html_data = json.loads(html_result)
            html_content = html_data.get("html", "")

            if html_content:
                st.components.v1.html(html_content, height=650, scrolling=True)
                st.download_button(
                    "📥 Download HTML",
                    data=html_content,
                    file_name=f"email_{lead.get('prospect_id', 'draft')}.html",
                    mime="text/html",
                    use_container_width=True,
                )
