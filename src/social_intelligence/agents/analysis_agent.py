"""Analysis Agent: scores prospect-trend relevance and prioritizes prospects."""

import json
import logging
from pathlib import Path

from strands import Agent
from strands.models import BedrockModel

from social_intelligence.agents import SAFETY_FENCE
from social_intelligence.config import ANALYSIS_MODEL_ID, AWS_REGION, guardrail_kwargs
from social_intelligence.schemas.models import ScoredProspectList

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ICP profile: loaded from config/icp_profile.json at agent-creation time.
# Falls back to the hardcoded block below if the file is missing, preserving
# exact blog behavior.
# ---------------------------------------------------------------------------

_DEFAULT_ICP_BLOCK = (
    "ICP FIT: Score against AnyCompany's ideal customer profile:\n"
    "- Developer tool or SaaS product (high fit)\n"
    "- Active open-source presence with growing community (high fit)\n"
    "- Recently launched or in growth phase (high fit)\n"
    "- B2B focus with technical buyer persona (high fit)\n"
    "- Consumer-only product with no developer angle (low fit)\n"
    "Add +10 to the final score for strong ICP fit, -10 for poor fit."
)


def _load_icp_block() -> str:
    """Load ICP criteria from config/icp_profile.json, falling back to the default block.

    Resolves the config directory relative to this file, walking up to the repo root
    (the directory that contains src/), so behavior is CWD-independent.

    Returns:
        Formatted ICP block string ready for injection into the system prompt.
    """
    # Walk up from agents/ -> social_intelligence/ -> src/ -> repo root
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    icp_path = repo_root / "config" / "icp_profile.json"
    try:
        with icp_path.open() as fh:
            data = json.load(fh)
        lines = ["ICP FIT: Score against AnyCompany's ideal customer profile:"]
        for item in data.get("high_fit", []):
            lines.append(f"- {item} (high fit)")
        for item in data.get("low_fit", []):
            lines.append(f"- {item} (low fit)")
        bonus = data.get("score_bonus", 10)
        penalty = data.get("score_penalty", 10)
        lines.append(f"Add +{bonus} to the final score for strong ICP fit, -{penalty} for poor fit.")
        return "\n".join(lines)
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        logger.debug("ICP profile not found or invalid at %s; using default", icp_path)
        return _DEFAULT_ICP_BLOCK


SYSTEM_PROMPT_TEMPLATE = (
    "You are the Analysis Agent for AnyCompany's social intelligence system.\n\n"
    "YOUR ROLE: Score every prospect passed to you and emit a ScoredProspectList "
    "structured object. That structured output is your entire deliverable — do NOT "
    "narrate scores in prose.\n\n"
    "SCORING CRITERIA (0-100 total):\n"
    "- Topical alignment (0-25): How well does the prospect's product align with "
    "current trends? Are they in a growing or declining space?\n"
    "- Timing relevance (0-20): Is this the right moment for outreach? Recent launches, "
    "funding rounds, or trend spikes increase timing score.\n"
    "- Engagement potential (0-20): How likely is the prospect to respond? High HN scores, "
    "active GitHub repos, and dev.to engagement indicate receptiveness.\n"
    "- Intent signals (0-20): Does the prospect show buying intent? Look for: "
    "recommendation-seeking posts, competitor frustration, budget discussions, "
    "tool evaluation threads, or job postings for roles that signal purchasing needs. "
    "Reddit intent_signals fields are pre-classified; weight them heavily.\n"
    "- Data quality (0-15): How complete and consistent is the collected data? "
    "Multi-source confirmation increases quality score.\n\n"
    "TEMPORAL DECAY: Weight signals by freshness.\n"
    "- Signals < 24 hours old: 1.5x weight\n"
    "- Signals 1-3 days old: 1.2x weight\n"
    "- Signals 3-7 days old: 1.0x weight (baseline)\n"
    "- Signals > 7 days old: 0.5x weight\n"
    "Apply temporal decay before summing category scores.\n\n"
    "{icp_block}\n\n"
    "SCORE CALIBRATION: Scores above 80 require strong multi-signal confirmation AND "
    "at least one intent signal. Flag data inconsistencies in the reasoning field.\n\n"
    "REQUIRED STRUCTURED OUTPUT — this is a hard contract:\n"
    "Your response MUST be a ScoredProspectList object containing a 'prospects' list.\n"
    "Every prospect from the research and enrichment agents must appear in the list.\n"
    "Do NOT omit any prospect. Do NOT narrate scores in prose such as "
    "'Prospect X scored 88/100' — the structured fields are the authoritative output.\n\n"
    "Each entry in prospects[] requires ALL of the following fields:\n"
    "  prospect_id   : str  — the identifier passed in from TrendData (e.g. HN story ID)\n"
    "  product_name  : str  — the product or project name\n"
    "  score         : int  — integer 0-100 (sum of weighted category scores + ICP adjustment)\n"
    "  confidence    : float — 0.0 to 1.0 (your certainty in the score given data completeness)\n"
    "  reasoning     : str  — exactly 2-3 sentences explaining the score; cite specific signals\n"
    "  top_trends    : list[str] — 1-5 trend strings most relevant for email personalization\n"
    "  intent_signals: list[str] — buying-intent signals detected (empty list if none found)\n"
    "  icp_fit       : str  — one of: 'strong', 'medium', or 'weak'\n"
    "  data_quality  : str  — one of: 'high', 'medium', or 'low'\n\n"
    "After you have computed all scores, emit ONLY the ScoredProspectList structured object. "
    "Do not add commentary, summaries, or markdown outside the structured fields."
    "{safety_fence}"
)

SWARM_HANDOFF = (
    "\n\nSWARM HANDOFF: After scoring ALL prospects, you MUST hand off to email_generator "
    "using handoff_to_agent. Pass the full scored prospect data as JSON in the context "
    "parameter — the next agent cannot see your conversation. Include for each prospect: "
    "prospect_id, product_name, score, confidence, reasoning, top_trends, intent_signals, "
    "icp_fit, and data_quality. Do NOT finish without handing off."
)

DESCRIPTION = (
    "Scores prospect-trend relevance on a 0-100 scale using structured criteria. "
    "Hand off to this agent AFTER trend discovery and enrichment are complete. "
    "After scoring, hands off to the email generation agent."
)


def create_analysis_agent(
    use_structured_output: bool = True,
    swarm_mode: bool = False,
) -> Agent:
    """Create and return the Analysis Agent.

    Args:
        use_structured_output: If True, use structured_output_model. Set False for Swarm
            mode where structured output signals completion and prevents handoffs.
        swarm_mode: If True, append handoff instructions to the system prompt.

    Note: Memory is attached at the orchestrator level (Swarm/Graph session_manager),
    not per-agent. Strands rejects per-agent session managers inside a multi-agent graph.
    """
    kwargs = {}
    if use_structured_output:
        kwargs["structured_output_model"] = ScoredProspectList
    icp_block = _load_icp_block()
    prompt = SYSTEM_PROMPT_TEMPLATE.format(icp_block=icp_block, safety_fence=SAFETY_FENCE)
    if swarm_mode:
        prompt += SWARM_HANDOFF
    return Agent(
        name="analyst",
        description=DESCRIPTION,
        model=BedrockModel(model_id=ANALYSIS_MODEL_ID, region_name=AWS_REGION, **guardrail_kwargs()),
        system_prompt=prompt,
        tools=[],
        **kwargs,
    )
