"""Reusable UI components for the Streamlit demo."""

import json
import re
from urllib.parse import urlparse

import streamlit as st
from config import (
    NODE_ICONS,
    NODE_LABELS,
    SCORE_HIGH_THRESHOLD,
    SCORE_MID_THRESHOLD,
    TOOL_LABELS,
)
from markupsafe import escape as _esc

# ---------------------------------------------------------------------------
# Tool display helpers
# ---------------------------------------------------------------------------


def clean_tool_name(raw_name: str) -> str:
    """Strip gateway prefix (e.g. 'social-intel-tools___hackernews_trending' -> 'hackernews_trending')."""
    return raw_name.split("___", 1)[-1] if "___" in raw_name else raw_name


def tool_display(raw_name: str) -> tuple[str, str]:
    """Return (icon, friendly_label) for a tool name."""
    clean = clean_tool_name(raw_name)
    if clean in TOOL_LABELS:
        return TOOL_LABELS[clean]
    return ("🔧", clean.replace("_", " ").title())


def extract_result_preview(raw_text: str) -> tuple[str, str]:
    """Parse API gateway JSON wrapper and return (status_label, preview_text)."""
    if not raw_text:
        return ("success", "")
    try:
        outer = json.loads(raw_text)
        if isinstance(outer, dict) and "statusCode" in outer:
            code = outer.get("statusCode", 200)
            body_str = outer.get("body", "")
            if isinstance(body_str, str):
                try:
                    body = json.loads(body_str)
                except (json.JSONDecodeError, TypeError):
                    body = body_str
            else:
                body = body_str

            if code >= 500:
                return ("error", str(body.get("error", "Server error")) if isinstance(body, dict) else "Server error")
            if code >= 400:
                return ("warning", str(body.get("error", "Client error")) if isinstance(body, dict) else "Client error")
            if isinstance(body, dict):
                return ("success", _summarize_body(body))
            return ("success", str(body)[:120])
    except (json.JSONDecodeError, TypeError):
        pass

    if '"statusCode"' in raw_text:
        code_match = re.search(r'"statusCode"\s*:\s*(\d+)', raw_text)
        code = int(code_match.group(1)) if code_match else 200
        if code >= 500:
            err_match = re.search(r'"error"\s*:\s*"([^"]*)"', raw_text)
            return ("error", err_match.group(1) if err_match else "Server error")
        if code >= 400:
            return ("warning", "Client error")
        titles = re.findall(r'"title"\s*:\s*"([^"]{3,50})', raw_text)
        for key in ("stories", "articles", "leads"):
            if f'"{key}"' in raw_text:
                summary = f"{key.title()} found"
                if titles:
                    summary += f": {', '.join(titles[:2])}"
                return ("success", summary)
        return ("success", "Data received")

    try:
        direct = json.loads(raw_text)
        if isinstance(direct, dict):
            return ("success", _summarize_body(direct))
    except (json.JSONDecodeError, TypeError):
        pass

    return ("success", raw_text[:120])


def _summarize_body(body: dict) -> str:
    """Create a human-readable one-liner from a tool response body."""
    for key, label in [
        ("stories", "stories"),
        ("articles", "articles"),
        ("leads", "existing leads"),
        ("results", "results"),
        ("repositories", "repos"),
        ("videos", "videos"),
        ("questions", "questions"),
        ("posts", "posts"),
    ]:
        if key in body:
            items = body[key]
            count = len(items) if isinstance(items, list) else 0
            if count and key == "stories":
                titles = [s.get("title", "")[:50] for s in items[:2] if isinstance(s, dict)]
                return f"{count} {label}" + (f": {', '.join(titles)}" if titles else "")
            return f"{count} {label}" if count else f"No {label} found"
    if "stored" in body:
        return "Lead stored" if body["stored"] else f"Store failed: {body.get('error', 'unknown')}"
    if "html" in body:
        return "HTML rendered"
    if "status" in body:
        return str(body["status"])
    return ", ".join(list(body.keys())[:3])


# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------


def score_color(score: int) -> str:
    """Return the CSS hex color for the given numeric score."""
    if score >= SCORE_HIGH_THRESHOLD:
        return "#38a169"
    if score >= SCORE_MID_THRESHOLD:
        return "#d69e2e"
    return "#e53e3e"


def score_dot(score: int) -> str:
    """Return the colored-circle emoji for the given numeric score."""
    if score >= SCORE_HIGH_THRESHOLD:
        return "🟢"
    if score >= SCORE_MID_THRESHOLD:
        return "🟡"
    return "🔴"


def score_bar(count: int, total: int, color: str) -> str:
    """Return an HTML bar for score distribution."""
    pct = (count / total * 100) if total > 0 else 0
    return (
        f'<div style="background:#2d3748;border-radius:4px;height:10px;'
        f'overflow:hidden;flex:1;margin:0 8px;">'
        f'<div style="background:{color};height:100%;width:{pct}%;'
        f'border-radius:4px;transition:width 0.3s;"></div></div>'
    )


def format_elapsed(seconds: float) -> str:
    """Format elapsed seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.0f}s"


# ---------------------------------------------------------------------------
# Event renderer (activity feed)
# ---------------------------------------------------------------------------


def render_event(evt: dict) -> None:
    """Render a single event in the activity feed."""
    t = evt.get("type", "")
    if t == "node_start":
        nid = evt.get("node_id", "")
        icon = NODE_ICONS.get(nid, "▶️")
        label = _esc(NODE_LABELS.get(nid, nid))
        st.markdown(
            f'<div style="background:linear-gradient(135deg,#1a365d 0%,#1e3a5f 100%);'
            f"border-radius:8px;padding:12px 16px;margin:10px 0 4px;"
            f'border-left:4px solid #4299e1;display:flex;align-items:center;gap:10px;">'
            f'<span style="font-size:18px;">{icon}</span>'
            f'<div><span style="color:#e2e8f0;font-size:14px;font-weight:600;">{label}</span>'
            f'<span style="color:#63b3ed;font-size:11px;margin-left:8px;'
            f'text-transform:uppercase;letter-spacing:0.5px;">running</span></div></div>',
            unsafe_allow_html=True,
        )
    elif t == "node_end":
        nid = evt.get("node_id", "")
        icon = NODE_ICONS.get(nid, "✅")
        label = _esc(NODE_LABELS.get(nid, nid))
        st.markdown(
            f'<div style="background:linear-gradient(135deg,#1c4532 0%,#22543d 100%);'
            f"border-radius:8px;padding:12px 16px;margin:4px 0 10px;"
            f'border-left:4px solid #38a169;display:flex;align-items:center;gap:10px;">'
            f'<span style="font-size:18px;">{icon}</span>'
            f'<div><span style="color:#e2e8f0;font-size:14px;font-weight:600;">{label}</span>'
            f'<span style="color:#68d391;font-size:11px;margin-left:8px;'
            f'text-transform:uppercase;letter-spacing:0.5px;">complete</span></div></div>',
            unsafe_allow_html=True,
        )
    elif t == "tool_start":
        raw_name = evt.get("tool_name", "")
        icon, friendly = tool_display(raw_name)
        inputs = evt.get("input_summary", {})
        param_html = ""
        if inputs:
            chips = []
            for k, v in list(inputs.items())[:3]:
                v_str = str(v)[:35] + "..." if len(str(v)) > 35 else str(v)
                chips.append(
                    f'<span style="display:inline-block;background:#4a5568;color:#cbd5e0;'
                    f'font-size:10px;padding:2px 8px;border-radius:10px;margin:2px 3px 0 0;">'
                    f"{_esc(k)}={_esc(v_str)}</span>"
                )
            param_html = f'<div style="margin-top:4px;">{"".join(chips)}</div>'
        st.markdown(
            f'<div style="background:#2d3748;border-radius:6px;padding:8px 12px;'
            f"margin:3px 0 3px 28px;font-size:12px;color:#cbd5e0;"
            f'border-left:2px solid #4a5568;">'
            f'<span style="color:#a0aec0;">⏳</span> '
            f"{icon} <strong>{_esc(friendly)}</strong>"
            f"{param_html}</div>",
            unsafe_allow_html=True,
        )
    elif t == "tool_end":
        raw_name = evt.get("tool_name", "")
        icon, friendly = tool_display(raw_name)
        raw_preview = evt.get("result_preview", "")
        status_label, preview = extract_result_preview(raw_preview)
        status_cfg = {"error": ("#e53e3e", "❌"), "warning": ("#d69e2e", "⚠️")}
        color, result_icon = status_cfg.get(status_label, ("#38a169", "✅"))
        preview_html = ""
        if preview:
            preview_html = (
                f'<div style="color:#a0aec0;font-size:11px;margin-top:3px;'
                f'font-style:italic;padding-left:2px;">{_esc(preview[:120])}</div>'
            )
        st.markdown(
            f'<div style="background:#2d3748;border-radius:6px;padding:8px 12px;'
            f"margin:3px 0 3px 28px;font-size:12px;color:#cbd5e0;"
            f'border-left:2px solid {color};">'
            f"{result_icon} {icon} <strong>{_esc(friendly)}</strong> "
            f'<span style="background:{color};color:#fff;font-size:10px;'
            f'padding:1px 7px;border-radius:10px;margin-left:4px;">{_esc(status_label)}</span>'
            f"{preview_html}</div>",
            unsafe_allow_html=True,
        )
    elif t == "structured_output":
        nid = evt.get("node_id", "")
        keys = evt.get("keys", [])
        label = _esc(NODE_LABELS.get(nid, nid))
        st.markdown(
            f'<div style="background:#2d3748;border-radius:6px;padding:8px 12px;'
            f"margin:3px 0 3px 28px;font-size:12px;color:#b794f4;"
            f'border-left:2px solid #805ad5;">'
            f"📦 <strong>{label}</strong> produced structured output: {_esc(', '.join(keys))}</div>",
            unsafe_allow_html=True,
        )
    elif t == "info":
        msg = _esc(evt.get("message", ""))
        st.markdown(
            f'<div style="font-size:12px;color:#718096;margin:4px 0 4px 28px;">ℹ️ {msg}</div>',
            unsafe_allow_html=True,
        )
    elif t == "error":
        msg = _esc(evt.get("message", ""))
        st.markdown(
            f'<div style="background:#742a2a;border-radius:8px;padding:10px 14px;margin:6px 0;'
            f'border-left:4px solid #e53e3e;font-size:13px;color:#fed7d7;">'
            f"❌ {msg}</div>",
            unsafe_allow_html=True,
        )
    elif t == "pipeline_end":
        st.markdown(
            '<div style="background:linear-gradient(135deg,#1c4532 0%,#22543d 100%);'
            "border-radius:8px;padding:12px 16px;margin:10px 0;text-align:center;"
            'border:1px solid #38a169;">'
            '<span style="color:#68d391;font-size:14px;font-weight:600;">'
            "🏁 Pipeline Complete</span></div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Lead card renderer
# ---------------------------------------------------------------------------


def render_lead_card(lead: dict) -> None:
    """Render a single lead as a styled card with expandable email."""
    score = int(lead.get("score", 0))
    color = score_color(score)
    if score >= SCORE_HIGH_THRESHOLD:
        cls = "score-high"
    elif score >= SCORE_MID_THRESHOLD:
        cls = "score-mid"
    else:
        cls = "score-low"
    product = _esc(lead.get("product_name", "Unknown"))
    author = _esc(lead.get("author", ""))
    reasoning = _esc(lead.get("reasoning", "")[:300])
    discovered = _esc(lead.get("discovered_at", "")[:19].replace("T", " "))
    signal = _esc(lead.get("signal_strength", "moderate"))
    quality = _esc(lead.get("data_quality", "medium"))
    confidence = float(lead.get("confidence", 0))

    # Validate URL scheme before embedding in href to prevent javascript: injection.
    # Use double-quote attribute delimiters so _esc() properly neutralises embedded quotes.
    raw_url = lead.get("source_url", "")
    safe_url = raw_url if urlparse(str(raw_url)).scheme in ("http", "https") else ""
    source_link = (
        f'<a href="{_esc(safe_url)}" target="_blank" rel="noopener noreferrer" '
        f'style="color:#63b3ed;font-size:11px;margin-left:12px;text-decoration:none;">🔗 Source</a>'
        if safe_url
        else ""
    )

    trends = lead.get("top_trends", "")
    trend_tags = ""
    if trends:
        tags = [t.strip() for t in str(trends).split(",") if t.strip()][:4]
        trend_tags = "".join(f'<span class="tag-badge">{_esc(tag)}</span>' for tag in tags)

    st.markdown(
        f'<div class="lead-card {cls}">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
        f'<div style="display:flex;align-items:center;gap:10px;">'
        f'<span style="font-size:20px;">{score_dot(score)}</span>'
        f'<div><span style="color:#e2e8f0;font-size:17px;font-weight:600;">{product}</span>'
        f'<span style="color:#718096;font-size:12px;margin-left:10px;">by {author}</span></div></div>'
        f'<span style="background:{color};color:#fff;padding:5px 14px;'
        f'border-radius:14px;font-size:14px;font-weight:700;">{score}</span></div>'
        f'<p style="color:#a0aec0;font-size:13px;margin:10px 0 8px 30px;line-height:1.5;">{reasoning}</p>'
        f'<div style="margin:4px 0 8px 30px;">{trend_tags}</div>'
        f'<div style="margin-left:30px;display:flex;flex-wrap:wrap;gap:12px;align-items:center;">'
        f'<span style="color:#718096;font-size:11px;">📅 {discovered}</span>'
        f'<span style="color:#718096;font-size:11px;">📡 {signal}</span>'
        f'<span style="color:#718096;font-size:11px;">📊 {quality}</span>'
        f'<span style="color:#718096;font-size:11px;">🎯 {confidence:.0%} confidence</span>'
        f"{source_link}</div></div>",
        unsafe_allow_html=True,
    )

    email_body = lead.get("email_body", "")
    if email_body:
        label = f"📧 {lead.get('email_subject', '')[:60]}" if lead.get("email_subject") else "📧 View Email"
        with st.expander(label):
            st.markdown(f"**Subject:** {lead.get('email_subject', '')}")
            st.markdown(email_body)


# ---------------------------------------------------------------------------
# Empty state placeholders
# ---------------------------------------------------------------------------


def render_empty_state(icon: str, title: str, description: str) -> None:
    """Render a centered empty state placeholder.

    All text arguments are HTML-escaped before rendering, so this function is
    safe to call with any string (including user- or database-derived values).
    Pass plain text only; it will be rendered as inert text, not HTML.

    Args:
        icon: Emoji or plain text icon character.
        title: Short heading text (escaped).
        description: Descriptive body text (escaped).
    """
    st.markdown(
        f'<div style="text-align:center;padding:60px 20px;color:#718096;">'
        f'<div style="font-size:48px;margin-bottom:16px;">{_esc(icon)}</div>'
        f'<p style="font-size:16px;margin-bottom:8px;">{_esc(title)}</p>'
        f'<p style="font-size:13px;">{_esc(description)}</p></div>',
        unsafe_allow_html=True,
    )
