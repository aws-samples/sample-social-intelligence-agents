"""AgentCore Runtime entrypoint: multi-agent social intelligence system.

Agents discover tools via Amazon Bedrock AgentCore Gateway (MCP protocol with IAM auth).
Agent-side tools (DynamoDB, email renderer, brand KB, grounding gate) are imported directly
since they run in the agent process, not behind the Gateway.

Orchestration patterns:
- 'graph' (default): Deterministic DAG with parallel entry and conditional edges
- 'swarm': Autonomous agent collaboration with dynamic handoffs

Human-in-the-loop: set EMAIL_APPROVAL_REQUIRED=true to store leads as 'pending_review'
status. store_lead in dynamodb_tool reads this env var and applies the status; the email
agent prompt is aware of this behavior and documents it to the model.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from mcp_proxy_for_aws.client import aws_iam_streamablehttp_client
from strands.multiagent import GraphBuilder, Swarm
from strands.multiagent.base import Status
from strands.tools.mcp.mcp_client import MCPClient

from social_intelligence.agents.analysis_agent import create_analysis_agent
from social_intelligence.agents.email_generation_agent import create_email_generation_agent
from social_intelligence.agents.search_specialist_agent import create_search_specialist_agent
from social_intelligence.agents.trend_research_agent import create_trend_research_agent

# Agent-side tools: run in agent process, not behind Gateway
from social_intelligence.tools.brand_knowledge import retrieve_brand_knowledge
from social_intelligence.tools.dynamodb_tool import check_existing_leads, reset_lead_counter, store_lead
from social_intelligence.tools.email_renderer import render_email_html_tool
from social_intelligence.tools.grounding_gate import verify_email_claims

if TYPE_CHECKING:
    from strands.multiagent.graph import GraphState

try:
    from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig
    from bedrock_agentcore.memory.integrations.strands.session_manager import (
        AgentCoreMemorySessionManager,
    )

    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False

from social_intelligence.config import AWS_REGION

logger = logging.getLogger(__name__)

# Configure logging only when running as the AgentCore entrypoint, not when imported by tests.
if not logging.root.handlers:
    logging.basicConfig(level=logging.INFO)
MEMORY_ID = os.environ.get("AGENTCORE_MEMORY_ID", "")
GATEWAY_URL = os.environ.get("GATEWAY_URL", "")

# Score threshold for email generation: prospects below this are skipped
EMAIL_SCORE_THRESHOLD = int(os.environ.get("EMAIL_SCORE_THRESHOLD", "60"))


# ---------------------------------------------------------------------------
# Tool loading: Gateway tools + agent-side tools
# ---------------------------------------------------------------------------


@contextmanager
def _gateway_tools():
    """Load tools from AgentCore Gateway with proper lifecycle management.

    Yields a dict mapping agent role to tool list. The MCP client is cleaned
    up when the context manager exits, preventing resource leaks.
    """
    logger.info("Connecting to AgentCore Gateway: %s", GATEWAY_URL)
    mcp_client = MCPClient(
        lambda: aws_iam_streamablehttp_client(
            endpoint=GATEWAY_URL,
            aws_region=AWS_REGION,
            aws_service="bedrock-agentcore",
        )
    )
    try:
        mcp_client.__enter__()
        gateway_tools = mcp_client.list_tools_sync()
        logger.info("Loaded %d tools from Gateway", len(gateway_tools))

        yield {
            "trend": gateway_tools + [check_existing_leads],
            "enrichment": gateway_tools + [check_existing_leads],
            "email": [
                retrieve_brand_knowledge,
                render_email_html_tool,
                verify_email_claims,
                store_lead,
            ]
            + gateway_tools,
        }
    finally:
        try:
            mcp_client.__exit__(None, None, None)
        except Exception:
            logger.debug("MCP client cleanup error (non-fatal)", exc_info=True)


# ---------------------------------------------------------------------------
# Orchestration builders
# ---------------------------------------------------------------------------


def _all_dependencies_complete(required_nodes: list[str]):
    """Conditional edge: proceed only when ALL listed nodes have completed."""

    def check(state: GraphState) -> bool:
        return all(nid in state.results and state.results[nid].status == Status.COMPLETED for nid in required_nodes)

    return check


def _score_above_threshold(state: GraphState) -> bool:
    """Conditional edge: only generate email if analysis produced high-scoring prospects.

    The analysis node uses ScoredProspectList structured output. We check whether
    any prospect scored above the threshold. Defaults to False on parse failure
    to avoid generating emails for unscored prospects.
    """
    analysis_result = state.results.get("analysis")
    if not analysis_result or analysis_result.status != Status.COMPLETED:
        return False

    try:
        # state.results["analysis"] is a NodeResult. Its .result is an AgentResult
        # (or a nested MultiAgentResult). NodeResult.get_agent_results() flattens to
        # the list of AgentResult objects; each carries .structured_output, which is
        # the ScoredProspectList Pydantic model the analysis agent produced.
        agent_results = []
        if hasattr(analysis_result, "get_agent_results"):
            agent_results = analysis_result.get_agent_results()
        elif hasattr(analysis_result, "result"):
            agent_results = [analysis_result.result]

        # Diagnostic: surface whether structured_output is present and its type, so
        # the score-gating path is observable in CloudWatch without logging content.
        so_types = [type(getattr(ar, "structured_output", None)).__name__ for ar in agent_results]
        logger.info("Score gate: %d agent result(s), structured_output types=%s", len(agent_results), so_types)

        max_score = _max_score_from_structured(agent_results)
        if max_score is not None:
            logger.info(
                "Analysis max score (structured_output): %d (threshold: %d)",
                max_score,
                EMAIL_SCORE_THRESHOLD,
            )
            return max_score >= EMAIL_SCORE_THRESHOLD

        # Text fallback when structured_output is absent (the analysis agent sometimes
        # narrates scores in prose instead of emitting the Pydantic model). We scan the
        # full result text for both JSON-shaped and prose-shaped prospect scores, gated
        # so HN/Reddit community scores are never matched.
        joined_text = " ".join(str(getattr(ar, "structured_output", "") or "") for ar in agent_results)
        full_text = f"{joined_text} {getattr(analysis_result, 'result', analysis_result)!s}"
        max_score = _max_prospect_score_in_text(full_text)
        if max_score is not None:
            logger.info("Analysis max score (text fallback): %d (threshold: %d)", max_score, EMAIL_SCORE_THRESHOLD)
            return max_score >= EMAIL_SCORE_THRESHOLD

        logger.warning("Could not find prospect-context scores in analysis output, skipping email generation")
    except (ValueError, TypeError, AttributeError, KeyError):
        logger.warning("Could not parse analysis scores, skipping email generation")
    return False


def _max_score_from_structured(agent_results: list) -> int | None:
    """Extract the highest prospect score from AgentResult.structured_output.

    Reads the ScoredProspectList (or single ScoredProspect) Pydantic model that
    the analysis agent attaches to each AgentResult, normalizing via model_dump().
    Returns None when no structured score is present.
    """
    scores: list[int] = []
    for ar in agent_results:
        structured = getattr(ar, "structured_output", None)
        if structured is None:
            continue
        if hasattr(structured, "model_dump"):
            try:
                structured = structured.model_dump()
            except (TypeError, ValueError):
                continue
        if isinstance(structured, dict):
            prospects = structured.get("prospects")
            if isinstance(prospects, list):
                scores.extend(int(p["score"]) for p in prospects if isinstance(p, dict) and "score" in p)
            elif "score" in structured:
                scores.append(int(structured["score"]))
    return max(scores) if scores else None


def _max_prospect_score_in_text(text: str) -> int | None:
    """Return the highest analysis prospect score found in text, or None.

    The analysis agent may express scores two ways:
    1. JSON-shaped: "score": N near a ScoredProspect-only field
       (reasoning/confidence/icp_fit/data_quality), within ~400 chars, either order.
    2. Prose-shaped: "score: 88", "(Score: 72)", "scored 88/100", or "88/100".

    Both forms require an analysis-scoring context so raw HN/Reddit community
    scores (which appear near comments/author, never near these markers) are
    excluded. Scores are clamped to the valid 0-100 range.
    """
    import re

    scores: list[int] = []

    # Form 1: JSON "score": N co-located with a ScoredProspect-only field.
    marker = r'"(?:reasoning|confidence|icp_fit|data_quality)"'
    json_score = r'"score"\s*:\s*(\d{1,3})'
    for pat in (
        rf"{json_score}[^{{}}]{{0,400}}{marker}",
        rf"{marker}[^{{}}]{{0,400}}{json_score}",
    ):
        scores.extend(int(m) for m in re.findall(pat, text))

    # Form 2: prose scores. "score: 88", "Score: 72", "scored 88", "88/100".
    for m in re.findall(r"(?i)\bscored?\s*[:=]?\s*(\d{1,3})\b(?:\s*/\s*100)?", text):
        scores.append(int(m))
    for m in re.findall(r"\b(\d{1,3})\s*/\s*100\b", text):
        scores.append(int(m))

    valid = [s for s in scores if 0 <= s <= 100]
    return max(valid) if valid else None


def _build_graph(tools: dict, session_manager=None):
    """Build the Graph orchestrator (deterministic DAG).

    DAG topology:
        research ──┐
                   ├──→ analysis (waits for both) ──→ email (if score ≥ threshold)
        search   ──┘

    Memory attaches at the graph level via set_session_manager. Strands persists
    graph state and per-node conversations to AgentCore Memory; agents must NOT
    carry their own session_manager (Strands rejects that inside a graph).
    """
    trend = create_trend_research_agent(tools=tools["trend"])
    search = create_search_specialist_agent(tools=tools["enrichment"])
    analysis = create_analysis_agent()
    email = create_email_generation_agent(tools=tools["email"])

    builder = GraphBuilder()
    builder.add_node(trend, "research")
    builder.add_node(search, "search")
    builder.add_node(analysis, "analysis")
    builder.add_node(email, "email")

    # Parallel entry: research and search run concurrently
    builder.set_entry_point("research")
    builder.set_entry_point("search")

    # Analysis waits for both research and search to complete
    wait_for_both = _all_dependencies_complete(["research", "search"])
    builder.add_edge("research", "analysis", condition=wait_for_both)
    builder.add_edge("search", "analysis", condition=wait_for_both)

    # Email only runs if analysis found high-scoring prospects
    builder.add_edge("analysis", "email", condition=_score_above_threshold)

    builder.set_execution_timeout(1200)  # 20 min, well under AgentCore's 60-min streaming limit
    builder.set_max_node_executions(20)
    builder.set_node_timeout(300)  # 5 min per node

    if session_manager:
        builder.set_session_manager(session_manager)

    return builder.build()


def _build_swarm(tools: dict, session_manager=None):
    """Build the Swarm orchestrator (autonomous agent collaboration).

    Swarm mode disables structured_output_model on agents because structured
    output signals task completion in Strands, which would prevent handoffs
    between agents. Each agent gets swarm_mode=True for explicit handoff instructions.

    Memory attaches at the swarm level via the session_manager argument. Agents
    must NOT carry their own session_manager (Strands rejects that inside a swarm).
    """
    # Email agent also gets check_existing_leads in swarm mode for proactive dedup
    email_tools = tools["email"] + [check_existing_leads]
    agents = [
        create_trend_research_agent(
            tools=tools["trend"],
            use_structured_output=False,
            swarm_mode=True,
        ),
        create_search_specialist_agent(
            tools=tools["enrichment"],
            use_structured_output=False,
            swarm_mode=True,
        ),
        create_analysis_agent(
            use_structured_output=False,
            swarm_mode=True,
        ),
        create_email_generation_agent(
            tools=email_tools,
            use_structured_output=False,
            swarm_mode=True,
        ),
    ]
    return Swarm(
        agents,
        entry_point=agents[0],
        max_handoffs=15,
        max_iterations=15,
        execution_timeout=1200.0,  # 20 min, well under AgentCore's 60-min streaming limit
        node_timeout=300.0,  # 5 min per agent turn
        repetitive_handoff_detection_window=8,
        repetitive_handoff_min_unique_agents=3,
        session_manager=session_manager,
    )


# ---------------------------------------------------------------------------
# Prompt augmentation: inject existing leads as a hard skip list
# ---------------------------------------------------------------------------


def _augment_prompt_with_skip_list(prompt: str) -> str:
    """Fetch existing lead names from DynamoDB and append a skip list to the prompt.

    This ensures agents discover NEW prospects instead of re-processing known ones.
    The skip list is injected at the prompt level so every agent in the pipeline sees it.

    Strategy: Scans a bounded window of recent items (projected to product_name only)
    with a segment-limited approach. The scan reads at most 150 items in a single page,
    keeping RCU cost under 25 even at scale. This is acceptable because:
    - MAX_LEADS_PER_RUN = 3 means the table grows at ~3 items/invocation
    - TTL (365 days) keeps the table from growing unboundedly
    - ProjectionExpression limits per-item cost to ~1 RCU per 4KB page

    For tables with 10K+ items, consider adding a GSI on discovered_at to query
    only the last 7 days instead of scanning.
    """
    from social_intelligence.tools.dynamodb_tool import _get_table

    try:
        table = _get_table()
        leads: list[dict] = []
        # Single-page bounded scan projecting only product_name to limit RCU cost.
        # FilterExpression is not used because DynamoDB charges RCU on the raw scan
        # regardless of filtering: it's cheaper to fetch 150 items and filter in-memory.
        scan_kwargs: dict = {
            "Limit": 150,
            "ProjectionExpression": "product_name",
            "ConsistentRead": False,  # eventually consistent costs half the RCU
        }
        resp = table.scan(**scan_kwargs)
        leads.extend(resp.get("Items", []))

        if not leads:
            return prompt

        # Deduplicate product names
        seen_names: set[str] = set()
        skip_names: list[str] = []
        for lead in leads:
            name = str(lead.get("product_name", "")).strip()
            name_lower = name.lower()
            if name and name_lower not in seen_names:
                seen_names.add(name_lower)
                skip_names.append(name)

        if not skip_names:
            return prompt

        skip_list = ", ".join(skip_names)
        logger.info("Skip list loaded: %d products", len(skip_names))
        return (
            f"{prompt}\n\n"
            f"SKIP LIST — these products are already in our database. "
            f"Do NOT research, score, or generate emails for any of them. "
            f"Find DIFFERENT prospects instead: {skip_list}"
        )
    except Exception:
        logger.debug("Could not build skip list", exc_info=True)
        return prompt


# ---------------------------------------------------------------------------
# Payload + session helpers (pure logic, unit-testable without AWS)
# ---------------------------------------------------------------------------

DEFAULT_PROMPT = "Find recent AI tool launches and generate outreach emails"


def _resolve_payload(payload: dict | None) -> tuple[str, str, str, str]:
    """Extract (prompt, pattern, session_id, actor_id) from the request payload.

    Applies defaults and normalizes the pattern to 'graph' or 'swarm'. Tolerates
    a missing or non-dict payload so a malformed request degrades to the default
    graph run rather than raising.
    """
    data = payload if isinstance(payload, dict) else {}
    prompt = str(data.get("prompt") or DEFAULT_PROMPT)
    pattern = "swarm" if str(data.get("pattern", "graph")).lower() == "swarm" else "graph"
    session_id = str(data.get("session_id") or "")
    actor_id = str(data.get("actor_id") or "")
    return prompt, pattern, session_id, actor_id


def _memory_enabled(session_id: str) -> bool:
    """Return True when AgentCore Memory should be wired for this invocation.

    Memory requires the SDK to be importable, a deployed memory id in the
    environment, and a caller-supplied session id. Absent any of these the
    pipeline degrades gracefully to a stateless run.
    """
    return bool(MEMORY_AVAILABLE and MEMORY_ID and session_id)


def _resolve_actor_id(actor_id: str) -> str:
    """Return the caller actor id, or a stable per-day default when unset."""
    if actor_id:
        return actor_id
    from datetime import datetime, timezone

    return f"user_{datetime.now(timezone.utc).strftime('%Y%m%d')}"


def _build_session_manager(session_id: str, actor_id: str):
    """Build an AgentCoreMemorySessionManager, or None when memory is disabled.

    Pure up to the final SDK constructor call: namespace/retrieval config and the
    resolved actor id are computed here so they can be unit-tested. Retrieved
    records are injected as <prior_context> data, never as instructions.
    """
    if not _memory_enabled(session_id):
        return None

    from bedrock_agentcore.memory.integrations.strands.config import RetrievalConfig

    mem_config = AgentCoreMemoryConfig(
        memory_id=MEMORY_ID,
        session_id=session_id,
        actor_id=_resolve_actor_id(actor_id),
        # Keys match the strategy namespaces defined in the CDK stack.
        retrieval_config={
            "/actors/{actorId}/sessions/{sessionId}/prospects": RetrievalConfig(top_k=5, relevance_score=0.4),
            "/actors/{actorId}/brand": RetrievalConfig(top_k=3, relevance_score=0.4),
        },
        context_tag="prior_context",
    )
    logger.info("Memory enabled: memory_id=%s, session_id=%s", MEMORY_ID, session_id)
    return AgentCoreMemorySessionManager(agentcore_memory_config=mem_config, region_name=AWS_REGION)


def _build_orchestrator(pattern: str, tools: dict, session_manager):
    """Build the Swarm or Graph orchestrator for the requested pattern."""
    if pattern == "swarm":
        return _build_swarm(tools, session_manager)
    return _build_graph(tools, session_manager)


# Per-run diagnostics: observable in CloudWatch to distinguish a real pipeline
# run from an empty/no-op run (nodes start but agents never call the model).
_RUN_DIAG: dict = {"started": set(), "completed": set(), "tool_calls": 0}


def _reset_run_diag() -> None:
    """Reset per-invocation pipeline diagnostics."""
    _RUN_DIAG["started"] = set()
    _RUN_DIAG["completed"] = set()
    _RUN_DIAG["tool_calls"] = 0


def _event_type(event: dict) -> str:
    """Best-effort extraction of a Strands multi-agent event type string."""
    if not isinstance(event, dict):
        return ""
    # Strands has used both 'multiagent_*' and 'multi_agent_*' spellings; normalize.
    return str(event.get("type", "")).replace("multi_agent_", "multiagent_")


def _log_pipeline_event(event: dict) -> None:
    """Record node lifecycle + tool-call signals for one streamed event.

    Logs node start/stop (with stop_reason when present) and counts tool calls,
    so an empty run (nodes start, zero tool calls, no model output) is visible in
    CloudWatch. Fully defensive: never raises on an unexpected event shape.
    """
    try:
        etype = _event_type(event)
        if etype == "multiagent_node_start":
            node = event.get("node_id", "?")
            _RUN_DIAG["started"].add(node)
            logger.info("Node start: %s", node)
        elif etype in ("multiagent_node_stop", "multiagent_node_complete"):
            node = event.get("node_id", "?")
            _RUN_DIAG["completed"].add(node)
            result = event.get("result")
            stop_reason = getattr(result, "stop_reason", None) if result is not None else None
            logger.info("Node stop: %s (stop_reason=%s)", node, stop_reason)
        elif etype == "multiagent_node_stream":
            inner = event.get("event", {})
            msg = inner.get("message", {}) if isinstance(inner, dict) else {}
            for block in msg.get("content", []) if isinstance(msg, dict) else []:
                if isinstance(block, dict) and block.get("toolUse", {}).get("name"):
                    _RUN_DIAG["tool_calls"] += 1
    except Exception:
        logger.debug("pipeline-event logging error (non-fatal)", exc_info=True)


# ---------------------------------------------------------------------------
# AgentCore entrypoint
# ---------------------------------------------------------------------------

app = BedrockAgentCoreApp()


@app.entrypoint
async def invoke(payload, context=None):  # noqa: ARG001 (context required by AgentCore)
    """Stream events from the multi-agent pipeline.

    Payload:
        prompt (str): User request
        pattern (str): 'graph' or 'swarm'
        session_id (str): Optional session ID for memory
        actor_id (str): Optional actor ID for memory
    """
    if not GATEWAY_URL:
        raise RuntimeError("GATEWAY_URL environment variable is required but not set")

    prompt, pattern, session_id, actor_id = _resolve_payload(payload)
    logger.info("Invoking pattern=%s, prompt_length=%d", pattern, len(prompt))

    # Reset per-run lead counter and pipeline diagnostics so each invocation is fresh.
    reset_lead_counter()
    _reset_run_diag()

    # Build skip list from existing leads so agents discover NEW prospects
    prompt = _augment_prompt_with_skip_list(prompt)

    # Optional memory integration: attaches at the orchestrator level.
    session_manager = _build_session_manager(session_id, actor_id)

    with _gateway_tools() as tools:
        orchestrator = _build_orchestrator(pattern, tools, session_manager)
        async for event in orchestrator.stream_async(prompt):
            _log_pipeline_event(event)
            yield event
        logger.info(
            "Pipeline finished: nodes_started=%s, nodes_completed=%s, tool_calls=%d",
            sorted(_RUN_DIAG["started"]),
            sorted(_RUN_DIAG["completed"]),
            _RUN_DIAG["tool_calls"],
        )


if __name__ == "__main__":
    app.run()
