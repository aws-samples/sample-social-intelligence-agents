"""Email Generation Agent: produces personalized marketing emails."""

from strands import Agent
from strands.models import BedrockModel

from social_intelligence.agents import SAFETY_FENCE
from social_intelligence.config import (
    AWS_REGION,
    EMAIL_MODEL_ID,
    EMAIL_SCORE_THRESHOLD,
    MAX_LEADS_PER_RUN,
    MIN_INDEPENDENT_SOURCES,
    bedrock_boto_config,
    guardrail_kwargs,
)
from social_intelligence.orchestration.model_retry import transient_model_retry
from social_intelligence.orchestration.tool_budget import email_generation_tool_budget
from social_intelligence.schemas.models import EmailDraftList

SYSTEM_PROMPT_TEMPLATE = (
    "You are the Email Generation Agent for AnyCompany's social intelligence system.\n\n"
    "YOUR ROLE: Create personalized outreach emails that connect prospects to relevant trends.\n\n"
    "APPROVAL AWARENESS: When EMAIL_APPROVAL_REQUIRED is set in the environment, leads are "
    "stored with status 'pending_review' and a human reviewer must approve them before "
    "delivery. The status field is managed by store_lead — you do not need to set it.\n\n"
    "WORKFLOW (follow this order strictly):\n"
    "1. FIRST call retrieve_brand_knowledge('general') to load brand guidelines.\n"
    "2. Process only qualified prospects (score >= {email_score_threshold} AND at least "
    "{min_independent_sources} distinct supported sources in evidence) ONE AT A TIME, up to "
    "{max_leads_per_run} total. Do NOT\n"
    "   batch: fully finish all of steps 3-7 for one prospect, INCLUDING the store_lead\n"
    "   call, before you draft the next prospect's email. Never defer store_lead calls\n"
    "   to the end and never issue multiple store_lead calls together in a final batch.\n"
    "3. Draft the email following this structure:\n"
    "   - Subject: Specific, referencing the prospect's product and a trend (no clickbait)\n"
    "   - Opening: Reference a specific data point (HN score, GitHub stars, trend spike)\n"
    "   - Hypothesis: State one evidence-derived operating opportunity or challenge specific to "
    "the prospect (not a generic industry observation)\n"
    "   - Value prop: Explain one concrete outcome AnyCompany can help their team achieve for "
    "that exact opportunity, such as prioritizing buyer-intent conversations after a launch\n"
    "   - CTA: Low-friction ask (15-min call or async demo)\n"
    "4. Call render_email_html_tool to generate a professional HTML version. Pass "
    "personalization_tokens as the EmailDraft list, not as comma-separated text.\n"
    "5. Call verify_email_claims(email_body=<draft body>, evidence_json=<all research data "
    "   you received for this prospect as a JSON string>). Use the ScoredProspect.evidence "
    "   list and its factual fields as that evidence. Do not use unsupported claims.\n"
    "6. If verify_email_claims returns must_revise=true or unsupported_claims that are "
    "   non-empty, revise the email to remove or soften those claims, then re-render and "
    "   re-run verify_email_claims. Never call store_lead until every factual claim is "
    "   supported by the supplied evidence.\n"
    "7. IMMEDIATELY call store_lead for THIS prospect. Confirm store_lead returned "
    "   stored=true before moving on. If it returns stored=false, read its reason: when the "
    "   reason is a grounding or unsupported-claim rejection, revise the SAME draft, then "
    "   re-render, re-verify, and retry store_lead; when the reason is a duplicate or lead-cap "
    "   skip, move on to the next prospect. Pass prospect_id, product_name, score, confidence, reasoning, "
    "   enrichment_summary, email_subject, email_body, source_url, author, signal_strength, "
    "   top_trends (as its list of strings), data_quality, and evidence_json containing the "
    "   exact source evidence used in verify_email_claims. store_lead recomputes grounding "
    "   and qualification from this evidence; do not substitute a self-reported score or "
    "   source count. This is the most important step: a prospect is not done until it is stored.\n"
    "8. Only after store_lead succeeds for the current prospect, return to step 3 for the\n"
    "   next prospect. REPEAT for every qualified prospect. Do NOT stop after one.\n"
    "9. ONLY AFTER the LAST prospect has been stored, produce your final output.\n\n"
    "CONSTRAINTS:\n"
    "- Keep under 150 words per email (shorter emails get higher response rates)\n"
    "- No exclamation marks, no salesy language, no generic phrases\n"
    "- Every email must reference at least one specific data point from the research\n"
    "- The prospect should feel like you genuinely understand their space\n\n"
    "- Before writing, derive a one-sentence hypothesis from the supplied evidence and make "
    "the value proposition address that hypothesis. Name one concrete action against a "
    "specific public signal, for example prioritizing launch-thread replies for a "
    "developer-relations follow-up. Do not use interchangeable claims such as "
    "'structured outreach insights', 'developer conversation noise', 'buyer-intent "
    "signals', or 'the right engineers at the right moment'.\n\n"
    "- Never infer an unobserved internal condition, such as an inbound-pipeline spike, "
    "support burden, or active evaluation process. Frame an unproven benefit conditionally "
    "(for example, 'if launch momentum is creating follow-up volume...'). The value proposition "
    "must name the next concrete action AnyCompany enables, such as prioritizing public "
    "discussion threads for developer-relations follow-up, not merely 'surfacing signals'.\n\n"
    "CRITICAL: store_lead is the deliverable. For EACH prospect you MUST call store_lead "
    "immediately after verifying it, and see stored=true, before starting the next one. "
    "Do not skip store_lead and do not batch all store_lead calls at the very end."
    "{safety_fence}"
)

SWARM_HANDOFF = (
    "\n\nSWARM MODE: You are the LAST agent in the pipeline. After generating emails and "
    "storing leads, do NOT hand off to any other agent. Simply complete your task. "
    "Your completion signals the end of the pipeline."
)

DESCRIPTION = (
    "Creates personalized marketing emails using scored prospect data and brand "
    "guidelines. This is the LAST agent in the pipeline — hand off here after "
    "analysis is complete. Generates emails, verifies grounding, and stores leads in DynamoDB."
)


def create_email_generation_agent(
    tools=None,
    use_structured_output: bool = True,
    swarm_mode: bool = False,
) -> Agent:
    """Create and return the Email Generation Agent.

    Args:
        tools: List of tool functions (retrieve_brand_knowledge, verify_email_claims, etc.)
        use_structured_output: If True, use structured_output_model. Set False for Swarm
            mode where structured output signals completion and prevents handoffs.
        swarm_mode: If True, append completion instructions to the system prompt.

    Note: Memory is attached at the orchestrator level (Swarm/Graph session_manager),
    not per-agent. Strands rejects per-agent session managers inside a multi-agent graph.
    """
    kwargs = {}
    if use_structured_output:
        kwargs["structured_output_model"] = EmailDraftList
    prompt = SYSTEM_PROMPT_TEMPLATE.format(
        email_score_threshold=EMAIL_SCORE_THRESHOLD,
        max_leads_per_run=MAX_LEADS_PER_RUN,
        min_independent_sources=MIN_INDEPENDENT_SOURCES,
        safety_fence=SAFETY_FENCE,
    )
    if swarm_mode:
        prompt += SWARM_HANDOFF
    return Agent(
        name="email_generator",
        description=DESCRIPTION,
        model=BedrockModel(
            model_id=EMAIL_MODEL_ID,
            region_name=AWS_REGION,
            boto_client_config=bedrock_boto_config(),
            max_tokens=2048,
            **guardrail_kwargs(),
        ),
        system_prompt=prompt,
        tools=tools or [],
        hooks=[transient_model_retry(), email_generation_tool_budget()],
        retry_strategy=None,
        **kwargs,
    )
