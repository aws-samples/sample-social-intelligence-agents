"""AgentCore Runtime entrypoint: multi-agent social intelligence system.

Agents discover tools via Amazon Bedrock AgentCore Gateway (MCP protocol with IAM auth).
Agent-side tools (DynamoDB, email renderer, brand KB, grounding gate) are imported directly
since they run in the agent process, not behind the Gateway.

Orchestration patterns:
- 'graph' (default): Deterministic DAG. Research discovers prospects, search enriches
  those prospects, analysis scores them, and email runs when a prospect scores >=
  EMAIL_SCORE_THRESHOLD.
- 'swarm': Autonomous agent collaboration with dynamic handoffs

Invocation modes: stream events synchronously (default), or pass {"background": true}
to run as an AgentCore async task (HealthyBusy) and read results from DynamoDB by run_id.

Human-in-the-loop: set EMAIL_APPROVAL_REQUIRED=true to store leads as 'pending_review'
status. store_lead in dynamodb_tool reads this env var and applies the status; the email
agent prompt is aware of this behavior and documents it to the model.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import os
import threading
import uuid
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from hashlib import sha256
from typing import TYPE_CHECKING

from bedrock_agentcore import BedrockAgentCoreApp
from mcp_proxy_for_aws.client import aws_iam_streamablehttp_client
from strands.multiagent import GraphBuilder, Swarm
from strands.multiagent.base import Status
from strands.tools.mcp.mcp_client import MCPClient

from social_intelligence.agents.analysis_agent import create_analysis_agent
from social_intelligence.agents.email_generation_agent import create_email_generation_agent
from social_intelligence.agents.search_specialist_agent import create_search_specialist_agent
from social_intelligence.agents.trend_research_agent import create_trend_research_agent
from social_intelligence.orchestration.qualification_gate import assess_email_eligibility

# Agent-side tools: run in agent process, not behind Gateway
from social_intelligence.tools.brand_knowledge import retrieve_brand_knowledge
from social_intelligence.tools.dynamodb_tool import (
    check_existing_leads,
    claim_url,
    get_run_output_metrics,
    persist_analysis_scores,
    persist_run_status,
    persist_scored_prospects,
    reset_lead_counter,
    set_run_isolation,
    set_run_session_id,
    store_lead,
)
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

from social_intelligence.config import (
    AWS_REGION,
    EMAIL_SCORE_THRESHOLD,
    MIN_INDEPENDENT_SOURCES,
    ORCHESTRATION_NODE_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)

_APPLICATION_LOG_LEVEL = getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO)
# AgentCore can install a root handler before this module loads. Attach an application
# stdout handler so diagnostics are emitted even when that handler filters INFO records.
_application_logger = logging.getLogger("social_intelligence")
_application_logger.setLevel(_APPLICATION_LOG_LEVEL)
if not _application_logger.handlers:
    _application_handler = logging.StreamHandler()
    _application_handler.setLevel(_APPLICATION_LOG_LEVEL)
    _application_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    _application_logger.addHandler(_application_handler)
    _application_logger.propagate = False

# Configure logging only when running as the AgentCore entrypoint, not when imported by tests.
if not logging.root.handlers:
    logging.basicConfig(level=_APPLICATION_LOG_LEVEL)
MEMORY_ID = os.environ.get("AGENTCORE_MEMORY_ID", "")
GATEWAY_URL = os.environ.get("GATEWAY_URL", "")

_TREND_GATEWAY_TOOL_NAMES = frozenset(
    {
        "hackernews_trending",
        "reddit_search",
        "producthunt_trending",
        "youtube_trending",
        "devto_trending",
        "stackoverflow_search",
    }
)
_TREND_OPTIONAL_GATEWAY_TOOL_NAMES = frozenset({"WebSearch"})
_ENRICHMENT_GATEWAY_TOOL_NAMES = frozenset(
    {
        "wikipedia_summary",
        "github_search",
        "lobsters_trending",
        "stackoverflow_search",
    }
)
_AGENTCORE_GATEWAY_TOOL_DELIMITER = "___"

# ---------------------------------------------------------------------------
# Tool loading: Gateway tools + agent-side tools
# ---------------------------------------------------------------------------


def _gateway_tool_name(tool: object) -> str:
    """Return a Gateway schema name, or an empty string if unavailable.

    AgentCore Gateway presents Lambda-target tools over MCP as
    ``target_name___tool_name``. The role allow-lists intentionally use the
    unprefixed schema operation names, so normalize at this boundary without
    modifying the tool object that Strands invokes.
    """
    name = getattr(tool, "tool_name", "")
    if isinstance(name, str) and name:
        return _logical_gateway_tool_name(name)

    spec = getattr(tool, "tool_spec", None)
    if isinstance(spec, dict):
        spec_name = spec.get("name", "")
        if isinstance(spec_name, str):
            return _logical_gateway_tool_name(spec_name)
    return ""


def _logical_gateway_tool_name(name: str) -> str:
    """Strip the AgentCore Gateway target prefix from a public MCP tool name."""
    if _AGENTCORE_GATEWAY_TOOL_DELIMITER not in name:
        return name
    return name.split(_AGENTCORE_GATEWAY_TOOL_DELIMITER, 1)[1]


def _select_gateway_tools(
    gateway_tools: list,
    allowed_names: frozenset[str],
    optional_names: frozenset[str] = frozenset(),
) -> list:
    """Restrict an agent to its required tools and any available optional tools."""
    selected_names = allowed_names | optional_names
    selected = [tool for tool in gateway_tools if _gateway_tool_name(tool) in selected_names]
    available = {_gateway_tool_name(tool) for tool in gateway_tools}
    missing = allowed_names - available
    if missing:
        logger.warning("Gateway is missing expected tools: %s", sorted(missing))
    return selected


@contextmanager
def _gateway_tools():
    """Load tools from AgentCore Gateway with proper lifecycle management.

    Yields a dict mapping agent role to tool list. The MCP client is cleaned
    up when the context manager exits, preventing resource leaks.
    """
    logger.info("Connecting to AgentCore Gateway: %s", GATEWAY_URL)
    with MCPClient(
        lambda: aws_iam_streamablehttp_client(
            endpoint=GATEWAY_URL,
            aws_region=AWS_REGION,
            aws_service="bedrock-agentcore",
        )
    ) as mcp_client:
        gateway_tools = mcp_client.list_tools_sync()
        logger.info("Loaded %d tools from Gateway", len(gateway_tools))
        trend_tools = _select_gateway_tools(
            gateway_tools,
            _TREND_GATEWAY_TOOL_NAMES,
            _TREND_OPTIONAL_GATEWAY_TOOL_NAMES,
        )
        enrichment_tools = _select_gateway_tools(gateway_tools, _ENRICHMENT_GATEWAY_TOOL_NAMES)

        yield {
            "trend": trend_tools + [check_existing_leads, claim_url],
            "enrichment": enrichment_tools,
            "email": [
                retrieve_brand_knowledge,
                render_email_html_tool,
                verify_email_claims,
                store_lead,
            ],
        }


# ---------------------------------------------------------------------------
# Orchestration builders
# ---------------------------------------------------------------------------


def _all_dependencies_complete(required_nodes: list[str]):
    """Return an edge condition that delays routing until every named node completes.

    Strands evaluates outgoing edges after each completed batch and schedules a target
    when any current edge is traversable. The research-to-analysis edge remains false
    while search runs; the search-to-analysis edge becomes true only after both results
    exist, so analysis receives both dependency outputs.
    """

    def check(state: GraphState) -> bool:
        return all(nid in state.results and state.results[nid].status == Status.COMPLETED for nid in required_nodes)

    return check


def _score_above_threshold(state: GraphState) -> bool:
    """Conditional edge: run email only when analysis produced an eligible prospect.

    The analysis node uses ScoredProspectList structured output. A candidate needs
    both the configured score and corroboration from independent source evidence.
    Defaults to False on parse failure to avoid generating emails for unscored or
    uncorroborated prospects.
    """
    analysis_result = state.results.get("analysis")
    if not analysis_result or analysis_result.status != Status.COMPLETED:
        return False

    try:
        # NodeResult.get_agent_results() flattens nested multi-agent output into
        # AgentResult objects; each carries ScoredProspectList in structured_output.
        agent_results = _agent_results_from_node_result(analysis_result)

        # Diagnostic: surface whether structured_output is present and its type, so
        # the score-gating path is observable in CloudWatch without logging content.
        so_types = [type(getattr(ar, "structured_output", None)).__name__ for ar in agent_results]
        logger.info("Score gate: %d agent result(s), structured_output types=%s", len(agent_results), so_types)

        prospects = _scored_prospects_from_structured(agent_results)
        if prospects:
            eligible_count = sum(
                assess_email_eligibility(
                    prospect.get("score"),
                    prospect.get("evidence", []),
                    score_threshold=EMAIL_SCORE_THRESHOLD,
                    min_independent_sources=MIN_INDEPENDENT_SOURCES,
                ).email_eligible
                for prospect in prospects
            )
            logger.info(
                "Email gate: %d/%d prospect(s) meet score >= %d and %d independent-source requirement",
                eligible_count,
                len(prospects),
                EMAIL_SCORE_THRESHOLD,
                MIN_INDEPENDENT_SOURCES,
            )
            return eligible_count > 0

        logger.warning("Analysis emitted no valid structured prospects; skipping email generation")
    except ValueError, TypeError, AttributeError, KeyError:
        logger.warning("Could not parse analysis scores, skipping email generation")
    return False


def _agent_results_from_node_result(node_result) -> list:
    """Return flattened Strands AgentResult objects from a NodeResult-like value."""
    if hasattr(node_result, "get_agent_results"):
        return node_result.get_agent_results()
    if hasattr(node_result, "result"):
        return [node_result.result]
    return [node_result]


def _scored_prospects_from_structured(agent_results: list) -> list[dict]:
    """Extract schema-valid prospect dictionaries from AgentResult.structured_output."""
    prospects: list[dict] = []
    for agent_result in agent_results:
        structured = getattr(agent_result, "structured_output", None)
        if structured is None:
            continue
        if hasattr(structured, "model_dump"):
            try:
                structured = structured.model_dump()
            except TypeError, ValueError:
                continue
        if not isinstance(structured, dict):
            continue

        raw_prospects = structured.get("prospects")
        if not isinstance(raw_prospects, list):
            continue
        for prospect in raw_prospects:
            if not isinstance(prospect, dict):
                continue
            score = prospect.get("score")
            if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 100:
                continue
            prospects.append(prospect)
    return prospects


def _analysis_scores_event(event: dict) -> dict | None:
    """Create an opt-in SSE event from the typed result of the analysis graph node."""
    if _event_type(event) != "multiagent_node_stop" or event.get("node_id") != "analysis":
        return None

    # Strands 1.47 places the completed NodeResult in node_result. Keep the
    # result fallback for compatibility with older emitted event recordings.
    node_result = event.get("node_result", event.get("result"))
    status = getattr(node_result, "status", None)
    if status is not None and getattr(status, "value", status) != Status.COMPLETED.value:
        return None

    prospects = _scored_prospects_from_structured(_agent_results_from_node_result(node_result))
    event_prospects = []
    for prospect in prospects:
        qualification = assess_email_eligibility(
            prospect.get("score"),
            prospect.get("evidence", []),
            score_threshold=EMAIL_SCORE_THRESHOLD,
            min_independent_sources=MIN_INDEPENDENT_SOURCES,
        )
        event_prospect = {
            key: prospect[key]
            for key in (
                "prospect_id",
                "product_name",
                "score",
                "confidence",
                "icp_fit",
                "score_breakdown",
                "data_quality",
                "signal_strength",
                "evidence",
            )
            if key in prospect
        }
        event_prospect["independent_source_count"] = qualification.independent_source_count
        event_prospect["email_eligible"] = qualification.email_eligible
        event_prospects.append(event_prospect)
    return {
        "type": "analysis_scores",
        "prospects": event_prospects,
    }


def _build_graph(tools: dict, session_manager=None):
    """Build the Graph orchestrator (deterministic DAG).

    DAG topology:
        research ──→ search ──→ analysis ──→ email (if score ≥ threshold)

    Search requires the exact ``TrendData`` prospect IDs emitted by research. It must
    therefore depend on research rather than run as an independent entry point. Long
    runs use the background invocation mode, so preserving this data contract does not
    rely on an SSE response stream remaining open.

    Memory attaches at the graph level via set_session_manager. Multi-agent session
    managers persist orchestrator state and execution history, rather than attaching
    separate conversation history to every agent.
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

    # Search receives the complete TrendData output and enriches those exact prospects.
    builder.set_entry_point("research")
    builder.add_edge("research", "search")

    # Strands routes via any traversable outgoing edge, rather than a native AND-join.
    # Keep both dependencies so analysis receives both outputs, and make only the
    # search edge eligible once research and search have both completed.
    wait_for_research_and_search = _all_dependencies_complete(["research", "search"])
    builder.add_edge("research", "analysis", condition=wait_for_research_and_search)
    builder.add_edge("search", "analysis", condition=wait_for_research_and_search)

    # Email only runs if analysis found high-scoring prospects
    builder.add_edge("analysis", "email", condition=_score_above_threshold)

    builder.set_execution_timeout(1200)  # 20 min, well under AgentCore's 60-min streaming limit
    builder.set_max_node_executions(20)
    # Leaves room for the bounded three-attempt model retry policy and persistence.
    builder.set_node_timeout(ORCHESTRATION_NODE_TIMEOUT_SECONDS)

    if session_manager:
        builder.set_session_manager(session_manager)

    return builder.build()


def _build_swarm(tools: dict, session_manager=None):
    """Build the Swarm orchestrator (autonomous agent collaboration).

    Swarm mode disables structured_output_model on agents because structured
    output signals task completion in Strands, which would prevent handoffs
    between agents. Each agent gets swarm_mode=True for explicit handoff instructions.

    Memory attaches at the swarm level via the session_manager argument, preserving
    orchestration state while the explicit handoff context carries prospect evidence.
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
            tools=[persist_scored_prospects],
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
        # The intended path requires three handoffs. Allow two bounded corrections for a
        # malformed handoff, while the deterministic recovery below handles an early exit.
        max_handoffs=5,
        max_iterations=6,
        execution_timeout=1200.0,  # 20 min, well under AgentCore's 60-min streaming limit
        node_timeout=float(ORCHESTRATION_NODE_TIMEOUT_SECONDS),
        repetitive_handoff_detection_window=3,
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

    Uses the ``dedup-partition-discovered-at-index`` GSI, where every lead has a
    constant dedup partition and ``discovered_at`` is the sort key. Querying the
    index is deterministic and returns the newest 150 names without a table scan.
    """
    from social_intelligence.tools.dynamodb_tool import _get_table, get_run_session_id, run_isolation_enabled

    try:
        table = _get_table()
        resp = table.query(
            IndexName="dedup-partition-discovered-at-index",
            KeyConditionExpression="dedup_partition = :partition",
            ExpressionAttributeValues={":partition": "LEAD"},
            ScanIndexForward=False,
            Limit=150,
        )
        leads = resp.get("Items", [])

        # Under per-run isolation, only skip products THIS run already stored, so parallel
        # eval runs against a shared table do not starve each other of prospects.
        if run_isolation_enabled():
            this_run = get_run_session_id()
            leads = [ld for ld in leads if str(ld.get("session_id", "")) == this_run]

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


def _payload_flag(value: object) -> bool:
    """Parse an optional API boolean without treating the string ``false`` as true."""
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _memory_enabled(session_id: str) -> bool:
    """Return True when AgentCore Memory should be wired for this invocation.

    Memory requires the SDK to be importable, a deployed memory id in the
    environment, and a caller-supplied session id. Absent any of these the
    pipeline degrades gracefully to a stateless run.
    """
    return bool(MEMORY_AVAILABLE and MEMORY_ID and session_id)


def _resolve_actor_id(actor_id: str, session_id: str) -> str:
    """Return the caller actor id, or an anonymous identity scoped to its session."""
    if actor_id:
        return actor_id

    # Actor IDs scope AgentCore Memory namespaces. Hash rather than reuse a shared
    # "anonymous" actor so callers that omit actor_id cannot read one another's memory.
    return f"anonymous_{sha256(session_id.encode('utf-8')).hexdigest()[:24]}"


def _build_session_manager(session_id: str, actor_id: str):
    """Build an AgentCoreMemorySessionManager, or None when memory is disabled.

    The returned manager is entered around each invocation so the AgentCore SDK
    flushes buffered memory turns. Retrieved records are injected as
    <prior_context> data, never as instructions.
    """
    if not _memory_enabled(session_id):
        return None

    from bedrock_agentcore.memory.integrations.strands.config import RetrievalConfig

    mem_config = AgentCoreMemoryConfig(
        memory_id=MEMORY_ID,
        session_id=session_id,
        actor_id=_resolve_actor_id(actor_id, session_id),
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
# Each invocation gets a separate mapping, avoiding cross-request log pollution.
_RUN_DIAG: ContextVar[dict | None] = ContextVar("social_intelligence_run_diagnostics", default=None)


def _current_run_diag() -> dict:
    """Return diagnostics for this request, initializing direct test calls."""
    diagnostics = _RUN_DIAG.get()
    if diagnostics is None:
        diagnostics = {"started": set(), "completed": set(), "tool_calls": 0}
        _RUN_DIAG.set(diagnostics)
    return diagnostics


def _reset_run_diag() -> None:
    """Reset per-invocation pipeline diagnostics."""
    _RUN_DIAG.set({"started": set(), "completed": set(), "tool_calls": 0})


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
        diagnostics = _current_run_diag()
        etype = _event_type(event)
        if etype == "multiagent_node_start":
            node = event.get("node_id", "?")
            diagnostics["started"].add(node)
            logger.info("Node start: %s", node)
        elif etype in ("multiagent_node_stop", "multiagent_node_complete"):
            node = event.get("node_id", "?")
            diagnostics["completed"].add(node)
            node_result = event.get("node_result", event.get("result"))
            result = getattr(node_result, "result", node_result)
            stop_reason = getattr(result, "stop_reason", None) if result is not None else None
            logger.info("Node stop: %s (stop_reason=%s)", node, stop_reason)
        elif etype == "multiagent_node_stream":
            inner = event.get("event", {})
            msg = inner.get("message", {}) if isinstance(inner, dict) else {}
            for block in msg.get("content", []) if isinstance(msg, dict) else []:
                if isinstance(block, dict) and block.get("toolUse", {}).get("name"):
                    diagnostics["tool_calls"] += 1
    except Exception:
        logger.debug("pipeline-event logging error (non-fatal)", exc_info=True)


def _terminal_result_status(event: dict) -> str:
    """Extract a normalized completion status from a Strands terminal result event."""
    result = event.get("result")
    status = result.get("status") if isinstance(result, dict) else getattr(result, "status", None)
    return str(getattr(status, "value", status) or "").lower()


def _require_completed_result(event: dict, pattern: str) -> None:
    """Reject a terminal event that does not represent a completed orchestration."""
    status = _terminal_result_status(event)
    if status != Status.COMPLETED.value:
        raise RuntimeError(f"{pattern} orchestration ended with status {status or 'missing'}")


def _swarm_recovery_reason() -> str | None:
    """Return a deterministic-recovery reason when Swarm violated its output contract."""
    metrics = get_run_output_metrics()
    if not metrics.score_persistence_calls:
        return "the analyst did not persist a scored-prospect handoff"
    if metrics.scores_persisted != metrics.scores_requested:
        return (
            "score persistence was incomplete "
            f"({metrics.scores_persisted}/{metrics.scores_requested} records persisted)"
        )
    if metrics.leads_stored and not metrics.email_eligible_scores:
        raise RuntimeError("swarm persisted an email lead without an email-eligible analysis score")
    if metrics.email_eligible_scores and not metrics.leads_stored:
        return "email-eligible analysis scores did not produce a stored lead"
    return None


# ---------------------------------------------------------------------------
# AgentCore entrypoint
# ---------------------------------------------------------------------------

app = BedrockAgentCoreApp()


async def _run_pipeline(prompt: str, pattern: str, session_manager, include_scored_prospects: bool):
    """Run the orchestrator, persist analysis scores, and yield each pipeline event.

    Shared by the synchronous streaming path and the background (async) path. Scores are
    persisted to DynamoDB the moment the analysis node completes, so results survive a
    dropped response stream or a fully detached background run.

    Args:
        prompt: The (skip-list-augmented) user prompt.
        pattern: 'graph' or 'swarm'.
        session_manager: An active AgentCore memory session manager, or None.
        include_scored_prospects: When True, also yield the typed analysis_scores event.

    Yields:
        Orchestrator events, plus the analysis_scores event when opted in.
    """
    with _gateway_tools() as tools:
        orchestrator = _build_orchestrator(pattern, tools, session_manager)
        deferred_swarm_result = None
        swarm_execution_error: Exception | None = None
        try:
            async for event in orchestrator.stream_async(prompt):
                _log_pipeline_event(event)
                analysis_scores = _analysis_scores_event(event)
                if analysis_scores is not None:
                    # Persist before yielding the node-stop event. A client may disconnect
                    # immediately after receiving it, but the score rows must remain durable.
                    persisted = persist_analysis_scores(analysis_scores["prospects"])
                    logger.info("Persisted %d analysis score record(s) for the run", persisted)
                if _event_type(event) == "multiagent_result":
                    _require_completed_result(event, pattern)
                    if pattern == "swarm":
                        # Do not let a no-op Swarm terminal event end the response stream
                        # before output invariants are checked below.
                        deferred_swarm_result = event
                        continue
                yield event
                if analysis_scores is not None and include_scored_prospects:
                    yield analysis_scores
        except Exception as exc:
            # A Swarm node can fail mid-run (for example, Bedrock model throttling that
            # outlasts the retry budget) and never emit a terminal result. Route that into
            # the same deterministic Graph recovery below instead of failing the whole run.
            # The Graph pattern has no comparable recovery, so its errors still surface.
            if pattern != "swarm":
                raise
            swarm_execution_error = exc
            logger.warning("Swarm execution failed before a terminal result: %s", exc, exc_info=True)

        if pattern == "swarm":
            recovery_reason = _swarm_recovery_reason()
            if recovery_reason is None and swarm_execution_error is not None:
                recovery_reason = f"swarm execution failed before completion: {swarm_execution_error}"
            if deferred_swarm_result is not None and recovery_reason is None:
                yield deferred_swarm_result
            elif recovery_reason is None:
                raise RuntimeError("swarm orchestration ended without a terminal result event")
            else:
                # Snapshot the Swarm-phase counters before the Graph fallback re-runs
                # persistence on the same run-scoped state, so the recovery event and the
                # final cumulative log line remain attributable to each phase.
                swarm_metrics = get_run_output_metrics()
                logger.warning(
                    "Swarm recovery started: %s (swarm phase: scores_persisted=%d/%d, "
                    "eligible_scores=%d, leads_stored=%d)",
                    recovery_reason,
                    swarm_metrics.scores_persisted,
                    swarm_metrics.scores_requested,
                    swarm_metrics.email_eligible_scores,
                    swarm_metrics.leads_stored,
                )
                yield {"type": "swarm_recovery", "reason": recovery_reason, "fallback_pattern": "graph"}
                fallback = _build_graph(tools, session_manager)
                async for event in fallback.stream_async(prompt):
                    _log_pipeline_event(event)
                    analysis_scores = _analysis_scores_event(event)
                    if analysis_scores is not None:
                        # Preserve the same durability guarantee in the Graph fallback.
                        persisted = persist_analysis_scores(analysis_scores["prospects"])
                        logger.info("Persisted %d recovery analysis score record(s) for the run", persisted)
                    if _event_type(event) == "multiagent_result":
                        _require_completed_result(event, "graph recovery")
                    yield event
                    if analysis_scores is not None and include_scored_prospects:
                        yield analysis_scores

        diagnostics = _current_run_diag()
        metrics = get_run_output_metrics()
        logger.info(
            "Pipeline finished: nodes_started=%s, nodes_completed=%s, tool_calls=%d, "
            "score_persistence_calls=%d, scores_persisted=%d/%d, eligible_scores=%d, leads_stored=%d",
            sorted(diagnostics["started"]),
            sorted(diagnostics["completed"]),
            diagnostics["tool_calls"],
            metrics.score_persistence_calls,
            metrics.scores_persisted,
            metrics.scores_requested,
            metrics.email_eligible_scores,
            metrics.leads_stored,
        )


def _run_pipeline_in_background(prompt: str, pattern: str, session_manager, task_id: int) -> None:
    """Drive the pipeline to completion in a detached thread, then clear the async task.

    Started via contextvars.copy_context().run so the run-scoped state (session id, dedup
    isolation, lead counter, diagnostics) — held in ContextVars that do NOT propagate to a
    plain thread — is carried into the background run. This lets the entrypoint return
    immediately while AgentCore keeps the session HealthyBusy until the task completes; all
    results persist to DynamoDB inside _run_pipeline.

    Args:
        prompt: The skip-list-augmented user prompt.
        pattern: 'graph' or 'swarm'.
        session_manager: An active memory session manager, or None.
        task_id: The AgentCore async-task id to complete when the run finishes.
    """

    succeeded = False
    execution_path = pattern
    try:

        async def _drive_with_path_tracking() -> None:
            nonlocal execution_path
            manager_context = session_manager if session_manager is not None else nullcontext()
            with manager_context as active_session_manager:
                async for event in _run_pipeline(
                    prompt,
                    pattern,
                    active_session_manager,
                    include_scored_prospects=False,
                ):
                    if isinstance(event, dict) and event.get("type") == "swarm_recovery":
                        execution_path = "graph_recovery"

        asyncio.run(_drive_with_path_tracking())
        succeeded = True
    except Exception:
        logger.exception("Background pipeline run failed")
    finally:
        persist_run_status(succeeded, execution_path=execution_path)
        app.complete_async_task(task_id)


@app.entrypoint
async def invoke(payload, context=None):  # noqa: ARG001 (context required by AgentCore)
    """Run the multi-agent pipeline, streaming events or running in the background.

    Payload:
        prompt (str): User request
        pattern (str): 'graph' or 'swarm'
        session_id (str): Optional session ID for memory
        actor_id (str): Optional actor ID for memory
        include_scored_prospects (bool): Emit typed analysis scores for evaluation clients
        background (bool): When true, start the run as an AgentCore async task and return
            immediately with the run_id. The runtime stays HealthyBusy until the pipeline
            finishes server-side; results are read back from DynamoDB by run_id. This
            avoids the response-stream idle-disconnect on long multi-agent runs.
    """
    if not GATEWAY_URL:
        raise RuntimeError("GATEWAY_URL environment variable is required but not set")

    prompt, pattern, session_id, actor_id = _resolve_payload(payload)
    logger.info("Invoking pattern=%s, prompt_length=%d", pattern, len(prompt))

    # Reset per-run lead counter and pipeline diagnostics so each invocation is fresh.
    reset_lead_counter()
    _reset_run_diag()
    # Stamp every run with an id, even for one-shot calls without AgentCore Memory.
    # This also lets frontier claims distinguish the current run from another runtime.
    data = payload if isinstance(payload, dict) else {}
    run_id = str(data.get("run_id") or session_id or uuid.uuid4())
    set_run_session_id(run_id)
    # Optional per-run dedup isolation: when the caller sets isolate=true (and a run_id),
    # existing-lead checks and the skip list are scoped to this run so parallel eval runs
    # against the shared table do not starve each other. Defaults off (production behavior).
    set_run_isolation(_payload_flag(data.get("isolate")) and bool(run_id))
    include_scored_prospects = _payload_flag(data.get("include_scored_prospects"))
    background = _payload_flag(data.get("background"))

    # Build skip list from existing leads so agents discover NEW prospects
    prompt = _augment_prompt_with_skip_list(prompt)

    # Optional memory integration: attaches at the orchestrator level and is entered
    # for the full invocation so AgentCore flushes buffered state before returning.
    session_manager = _build_session_manager(session_id, actor_id)

    if background:
        # Long multi-agent runs outlive the response stream's idle window. Track the run
        # as an AgentCore async task (keeps the session HealthyBusy, not idle-killed) and
        # drive it in a detached thread. copy_context() carries the run-scoped ContextVars
        # into that thread; results are persisted to DynamoDB and read back by run_id.
        task_id = app.add_async_task("pipeline_run", {"run_id": run_id, "pattern": pattern})
        context_snapshot = contextvars.copy_context()
        threading.Thread(
            target=lambda: context_snapshot.run(_run_pipeline_in_background, prompt, pattern, session_manager, task_id),
            daemon=True,
        ).start()
        yield {"type": "async_started", "run_id": run_id, "task_id": task_id}
        return

    # Synchronous streaming path (interactive clients such as the demo UI).
    manager_context = session_manager if session_manager is not None else nullcontext()
    with manager_context as active_session_manager:
        async for event in _run_pipeline(prompt, pattern, active_session_manager, include_scored_prospects):
            yield event


if __name__ == "__main__":
    app.run()
