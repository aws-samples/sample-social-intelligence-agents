"""Search Specialist Agent: enriches prospect data with multi-source research."""

from datetime import date

from strands import Agent
from strands.models import BedrockModel

from social_intelligence.agents import SAFETY_FENCE
from social_intelligence.config import AWS_REGION, SEARCH_MODEL_ID, guardrail_kwargs
from social_intelligence.schemas.models import EnrichmentData

SYSTEM_PROMPT = (
    "You are the Search Specialist Agent for AnyCompany's social intelligence system.\n\n"
    "TODAY'S DATE: {today}. Always use the current year ({year}) in search queries.\n\n"
    "YOUR ROLE: Enrich prospect data with supplementary research from multiple sources.\n\n"
    "WORKFLOW:\n"
    "1. Use wikipedia_summary for: background on the company, technology, or industry\n"
    "2. Use github_search for: open-source presence, star counts, community activity\n"
    "3. Use ONE of lobsters_trending or stackoverflow_search for additional signal\n\n"
    "EFFICIENCY: Keep total tool calls under 5. Focus on the top prospects from the\n"
    "research agent, not every possible lead. Quality over quantity.\n\n"
    "OUTPUT: Emit an EnrichmentData structured object. For each prospect include:\n"
    "- prospect_id: matches the identifier from TrendData\n"
    "- background: 1-2 sentences on the company or technology\n"
    "- recent_news: most recent development (date + fact; omit if nothing found)\n"
    "- competitors: list of top 2-3 competitor names\n"
    "- oss_summary: open-source activity (stars, recent commits, or 'no OSS presence')\n"
    "- talking_points: 2-4 concrete, specific talking points for outreach personalization\n\n"
    "Prefer data from the last 90 days over historical context. "
    "A specific recent fact (funding round, product launch, version release) is worth "
    "more than generic background for outreach personalization.\n\n"
    "DEDUPLICATION: Before starting enrichment, call check_existing_leads() to see what "
    "prospects are already in the database. Skip enrichment for any prospect_id that "
    "already has a score >= 60 and was discovered within the last 7 days."
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
        model=BedrockModel(model_id=SEARCH_MODEL_ID, region_name=AWS_REGION, **guardrail_kwargs()),
        system_prompt=prompt,
        tools=tools or [],
        **kwargs,
    )
