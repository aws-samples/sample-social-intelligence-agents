"""Trend Research Agent: discovers prospects and collects real-time trend data."""

from datetime import date

from strands import Agent
from strands.models import BedrockModel

from social_intelligence.agents import SAFETY_FENCE
from social_intelligence.config import (
    AWS_REGION,
    TREND_MAX_TOKENS,
    TREND_MODEL_ID,
    bedrock_boto_config,
    guardrail_kwargs,
)
from social_intelligence.orchestration.model_retry import transient_model_retry
from social_intelligence.orchestration.tool_budget import trend_research_tool_budget
from social_intelligence.schemas.models import TrendData

SYSTEM_PROMPT = (
    "You are the Trend Research Agent for AnyCompany's social intelligence system.\n\n"
    "TODAY'S DATE: {today}. Use the current year ({year}) in search queries.\n\n"
    "YOUR ROLE: Discover qualified prospects and collect multi-signal trend data across "
    "six sources: Hacker News, Reddit, Product Hunt, YouTube, dev.to, and Stack Overflow.\n\n"
    "OUTPUT DISCIPLINE (highest priority): Collect facts silently. In Graph mode, make the "
    "TrendData structured-output call immediately after collection. In Swarm mode, hand off "
    "the compact prospect JSON immediately after collection. Never emit progress reports, "
    "score tables, markdown, or a prose summary before the required output or handoff.\n\n"
    "WORKFLOW:\n"
    "1. DISCOVERY — find new launches and buying-intent signals:\n"
    "   - hackernews_trending: category 'top' (limit=10) for tech launches and Show HN posts.\n"
    "     If most results are on the SKIP LIST, try 'show' or 'new' (max 2 HN calls total).\n"
    "     Use keyword_filter to focus on the requested domain.\n"
    "   - reddit_search (limit=10): prospects showing buying intent. The tool searches\n"
    "     multiple subreddits and auto-detects intent signals (recommendation-seeking,\n"
    "     competitor frustration, launch, purchase intent).\n"
    "   - producthunt_trending: same-day and recent product launches.\n"
    "2. SIGNAL ENRICHMENT — for the top 3-5 prospects, gather corroborating signals:\n"
    "   - youtube_trending: launch/demo videos and view counts that show momentum.\n"
    "   - devto_trending OR stackoverflow_search: developer mindshare and demand.\n"
    "     Pick the most relevant; do not call both for the same prospect.\n"
    "3. Cross-reference signals: a prospect appearing on multiple platforms is higher quality.\n"
    "   Prospects with Reddit intent signals (recommendation_seeking, competitor_frustration)\n"
    "   rank above those found only through passive trending.\n\n"
    "EFFICIENCY: Keep total tool calls under 8. Use HN + Reddit + Product Hunt for discovery,\n"
    "then one or two enrichment signals for the top prospects only. Do NOT call every tool "
    "for every prospect. If fewer than 3 fresh non-skip-list prospects are found, and the "
    "managed WebSearch tool is available, use it as an overflow source.\n\n"
    "OUTPUT: Emit a TrendData structured object with at most 5 prospects. For each prospect include:\n"
    "- prospect_id: the source platform's unique ID (e.g. HN story ID, Reddit post ID)\n"
    "- product_name, source_url, author\n"
    "- community_score: the source's engagement count (HN points, Reddit upvotes, etc.)\n"
    "- trend_signals: up to 3 per-source signal objects with engagement metrics\n"
    "- signal_strength: one of 'strong', 'moderate', or 'weak'\n\n"
    "DEDUPLICATION: The user prompt includes a SKIP LIST of products already in our database. "
    "Do NOT research or return any product on that list. Focus on new prospects only. "
    "Call check_existing_leads() to verify borderline cases by prospect_id or product_name."
    "{safety_fence}"
)

SWARM_HANDOFF = (
    "\n\nSWARM HANDOFF: When you have discovered prospects, hand off to search_specialist "
    "using handoff_to_agent. You MUST pass the full prospect list as JSON in the context "
    "parameter — the next agent cannot see your tool call history. Include for each prospect: "
    "name, URL, HN/Reddit ID, score, author, and key signals. Do NOT hand off to analyst directly."
)

DESCRIPTION = (
    "Discovers qualified prospects and collects multi-signal trend data from Hacker News, "
    "Reddit, Product Hunt, YouTube, dev.to, and Stack Overflow. Hand off to this "
    "agent when you need prospect discovery or trend data collection."
)


def create_trend_research_agent(
    tools=None,
    use_structured_output: bool = True,
    swarm_mode: bool = False,
) -> Agent:
    """Create and return the Trend Research Agent.

    Args:
        tools: List of tool functions (hackernews_trending, youtube_trending, etc.)
        use_structured_output: If True, use structured_output_model. Set False for Swarm
            mode where structured output signals completion and prevents handoffs.
        swarm_mode: If True, append handoff instructions to the system prompt.

    Note: Memory is attached at the orchestrator level (Swarm/Graph session_manager),
    not per-agent. Strands rejects per-agent session managers inside a multi-agent graph.
    """
    kwargs = {}
    if use_structured_output:
        kwargs["structured_output_model"] = TrendData
    today = date.today()
    prompt = SYSTEM_PROMPT.format(today=today.isoformat(), year=today.year, safety_fence=SAFETY_FENCE)
    if swarm_mode:
        prompt += SWARM_HANDOFF
    return Agent(
        name="trend_researcher",
        description=DESCRIPTION,
        model=BedrockModel(
            model_id=TREND_MODEL_ID,
            region_name=AWS_REGION,
            boto_client_config=bedrock_boto_config(),
            max_tokens=TREND_MAX_TOKENS,
            **guardrail_kwargs(),
        ),
        system_prompt=prompt,
        tools=tools or [],
        hooks=[transient_model_retry(), trend_research_tool_budget()],
        retry_strategy=None,
        **kwargs,
    )
