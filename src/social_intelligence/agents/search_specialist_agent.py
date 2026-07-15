"""Search Specialist Agent: enriches prospect data with multi-source research."""

from datetime import date

from strands import Agent
from strands.models import BedrockModel

from social_intelligence.agents import SAFETY_FENCE
from social_intelligence.config import (
    AWS_REGION,
    SEARCH_MAX_TOKENS,
    SEARCH_MODEL_ID,
    bedrock_boto_config,
    guardrail_kwargs,
)
from social_intelligence.orchestration.model_retry import transient_model_retry
from social_intelligence.orchestration.tool_budget import search_specialist_tool_budget
from social_intelligence.schemas.models import EnrichmentData

SYSTEM_PROMPT = (
    "You are the Search Specialist Agent for AnyCompany's social intelligence system.\n\n"
    "TODAY'S DATE: {today}. Always use the current year ({year}) in search queries.\n\n"
    "YOUR ROLE: Enrich prospect data with supplementary research from multiple sources.\n\n"
    "OUTPUT DISCIPLINE (highest priority): Gather facts silently. In Graph mode, make the "
    "EnrichmentData structured-output call immediately after collection. In Swarm mode, "
    "hand off the compact enrichment JSON immediately after collection. Never emit progress "
    "reports, score tables, markdown, or a prose summary before the required output or handoff.\n\n"
    "WORKFLOW:\n"
    "1. Use wikipedia_summary for: background on the company, technology, or industry\n"
    "2. Use github_search for: open-source presence, star counts, community activity\n"
    "3. Use ONE of lobsters_trending or stackoverflow_search for additional signal\n\n"
    "EFFICIENCY: Plan across ALL handoff prospects before calling tools. You have a hard\n"
    "four-call research budget: at most one wikipedia_summary, at most two github_search,\n"
    "and exactly one of lobsters_trending or stackoverflow_search for the entire task.\n"
    "Do not call both community sources, repeat an empty query, or make a follow-up tool\n"
    "call after those four calls. The runtime's SKIP LIST is authoritative for enrichment;\n"
    "store_lead performs the final atomic duplicate check. Focus on the highest-signal\n"
    "prospects, and complete the structured output from evidence already collected.\n\n"
    "OUTPUT: Emit an EnrichmentData structured object. For each prospect include:\n"
    "- prospect_id: matches the identifier from TrendData\n"
    "- background: factual summary under 240 characters\n"
    "- recent_news: most recent source-backed development under 240 characters (empty if unavailable)\n"
    "- competitors: up to 3 competitor names\n"
    "- oss_summary: open-source activity under 240 characters\n"
    "- talking_points: up to 2 concrete, source-backed talking points, each under 180 characters\n"
    "- evidence: up to 3 source-backed facts, each under 240 characters\n\n"
    "Prefer data from the last 90 days over historical context. "
    "A specific recent fact (funding round, product launch, version release) is worth "
    "more than generic background for outreach personalization.\n\n"
    "DEDUPLICATION: Before starting enrichment, apply the supplied SKIP LIST. Skip products "
    "already listed with a score >= 60 and discovered within the last 7 days. The SKIP LIST is "
    "authoritative for enrichment; store_lead performs the final atomic duplicate check at "
    "persistence. Do not describe any prospect as database-verified."
    "{safety_fence}"
)

SWARM_HANDOFF = (
    "\n\nSWARM HANDOFF: When enrichment is done, hand off to analyst using handoff_to_agent. "
    "You MUST pass the full enriched prospect data as JSON in the context parameter — "
    "the next agent cannot see your tool call history. Include for each prospect: "
    "name, background, competitors, GitHub stats, talking points, and all data from "
    "the trend_researcher's handoff."
)

DESCRIPTION = (
    "Enriches prospect data with Wikipedia context, GitHub open-source "
    "intelligence, Lobste.rs tech community discussions, and Stack Overflow "
    "demand signals. Hand off to this agent AFTER trend discovery for deeper research."
)


def create_search_specialist_agent(
    tools=None,
    use_structured_output: bool = True,
    swarm_mode: bool = False,
) -> Agent:
    """Create and return the Search Specialist Agent.

    Args:
        tools: List of tool functions (web_search, wikipedia_summary, etc.)
        use_structured_output: If True, use structured_output_model. Set False for Swarm
            mode where structured output signals completion and prevents handoffs.
        swarm_mode: If True, append handoff instructions to the system prompt.

    Note: Memory is attached at the orchestrator level (Swarm/Graph session_manager),
    not per-agent. Strands rejects per-agent session managers inside a multi-agent graph.
    """
    kwargs = {}
    if use_structured_output:
        kwargs["structured_output_model"] = EnrichmentData
    today = date.today()
    prompt = SYSTEM_PROMPT.format(today=today.isoformat(), year=today.year, safety_fence=SAFETY_FENCE)
    if swarm_mode:
        prompt += SWARM_HANDOFF
    return Agent(
        name="search_specialist",
        description=DESCRIPTION,
        model=BedrockModel(
            model_id=SEARCH_MODEL_ID,
            region_name=AWS_REGION,
            boto_client_config=bedrock_boto_config(),
            max_tokens=SEARCH_MAX_TOKENS,
            **guardrail_kwargs(),
        ),
        system_prompt=prompt,
        tools=tools or [],
        hooks=[transient_model_retry(), search_specialist_tool_budget()],
        retry_strategy=None,
        **kwargs,
    )
