"""Configuration, constants, and environment loading for the Streamlit demo."""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# .env file loading — uses python-dotenv if available, skips gracefully if not
# ---------------------------------------------------------------------------

_env_path = Path(__file__).resolve().parent.parent / ".env"
try:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=_env_path, override=False)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# AWS / App settings
# ---------------------------------------------------------------------------

AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
AWS_PROFILE = os.environ.get("AWS_PROFILE", "default")
LEADS_TABLE = os.environ.get("LEADS_TABLE_NAME", "social-intel-leads")
AGENT_ARN = os.environ.get("AGENTCORE_AGENT_ARN", "")

# ---------------------------------------------------------------------------
# Score thresholds
# ---------------------------------------------------------------------------

SCORE_HIGH_THRESHOLD = 80
SCORE_MID_THRESHOLD = 60

# ---------------------------------------------------------------------------
# Node display mappings (graph node IDs -> human labels)
# ---------------------------------------------------------------------------

NODE_LABELS: dict[str, str] = {
    "research": "Trend Research",
    "search": "Search Specialist",
    "analysis": "Analysis",
    "email": "Email Generation",
    "trend_researcher": "Trend Research",
    "search_specialist": "Search Specialist",
    "analyst": "Analysis",
    "email_generator": "Email Generation",
}

NODE_ICONS: dict[str, str] = {
    "research": "🔬",
    "search": "🔍",
    "analysis": "📊",
    "email": "✉️",
    "trend_researcher": "🔬",
    "search_specialist": "🔍",
    "analyst": "📊",
    "email_generator": "✉️",
}

# ---------------------------------------------------------------------------
# Tool display mappings
# ---------------------------------------------------------------------------

TOOL_LABELS: dict[str, tuple[str, str]] = {
    "hackernews_trending": ("🟠", "Hacker News"),
    "youtube_trending": ("🔴", "YouTube Trends"),
    "devto_trending": ("📝", "Dev.to"),
    "wikipedia_summary": ("📚", "Wikipedia"),
    "github_search": ("🐙", "GitHub"),
    "lobsters_trending": ("🦞", "Lobsters"),
    "producthunt_trending": ("🚀", "Product Hunt"),
    "reddit_search": ("🟣", "Reddit"),
    "stackoverflow_search": ("📋", "Stack Overflow"),
    "check_existing_leads": ("📋", "Check Leads DB"),
    "store_lead": ("💾", "Store Lead"),
    "retrieve_brand_knowledge": ("📖", "Brand Knowledge"),
    "render_email_html_tool": ("✉️", "Render Email"),
}

# ---------------------------------------------------------------------------
# Sidebar options
# ---------------------------------------------------------------------------

VERTICALS: dict[str, dict[str, str]] = {
    "AdTech & Programmatic": {"icon": "📡", "description": "Ad exchanges, DSPs, SSPs, programmatic platforms"},
    "Social Media & Creator Economy": {
        "icon": "🎨",
        "description": "Social platforms, influencer tools, creator monetization",
    },
    "MarTech & Analytics": {"icon": "📊", "description": "Marketing automation, attribution, customer data platforms"},
    "E-commerce & D2C": {"icon": "🛒", "description": "E-commerce platforms, Shopify apps, D2C brands"},
    "SaaS & Developer Tools": {"icon": "🔧", "description": "B2B SaaS, API platforms, developer infrastructure"},
    "AI & Machine Learning": {"icon": "🤖", "description": "AI startups, ML platforms, LLM applications"},
    "Media & Publishing": {"icon": "📰", "description": "Digital publishers, CMS, newsletter platforms"},
    "Gaming & Entertainment": {"icon": "🎮", "description": "Game studios, streaming, game monetization"},
}

LEAD_TYPES: dict[str, dict[str, str]] = {
    "Advertisers": {"icon": "💰", "description": "Companies that buy ads"},
    "Publishers": {"icon": "📱", "description": "Apps/sites with audiences"},
    "Both": {"icon": "🔄", "description": "Could be either advertiser or publisher"},
}
