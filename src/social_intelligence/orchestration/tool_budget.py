"""Hard per-invocation tool-call budgets for Strands agents.

Prompts explain the desired workflow, but the SDK hook is the enforcement point:
calls over budget are converted to tool errors before any external side effect occurs.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Collection, Mapping
from threading import Lock
from typing import Any

from strands.hooks import BeforeInvocationEvent, BeforeToolCallEvent, HookProvider, HookRegistry

from social_intelligence.config import MAX_LEADS_PER_RUN

logger = logging.getLogger(__name__)

_SWARM_HANDOFF_TOOL = "handoff_to_agent"
_AGENTCORE_GATEWAY_TOOL_DELIMITER = "___"
_STRUCTURED_OUTPUT_TOOLS = frozenset(
    {
        "TrendData",
        "EnrichmentData",
        "ScoredProspectList",
        "EmailDraftList",
    }
)


def _logical_tool_name(tool_name: str) -> str:
    """Return the schema tool name from an AgentCore Gateway tool name.

    AgentCore Gateway exposes Lambda-target tools to MCP clients as
    ``target_name___tool_name``. Budgets are defined against the schema names so
    they remain stable if the target name changes.
    """
    if _AGENTCORE_GATEWAY_TOOL_DELIMITER not in tool_name:
        return tool_name
    return tool_name.split(_AGENTCORE_GATEWAY_TOOL_DELIMITER, 1)[1]


class ToolCallBudget(HookProvider):
    """Enforce total and per-tool call limits for one Strands agent invocation."""

    def __init__(
        self,
        *,
        agent_name: str,
        max_total_calls: int,
        max_calls_per_tool: Mapping[str, int] | None = None,
        default_per_tool_limit: int | None = None,
        exempt_tools: Collection[str] = (_SWARM_HANDOFF_TOOL, *_STRUCTURED_OUTPUT_TOOLS),
    ) -> None:
        if not isinstance(agent_name, str) or not agent_name.strip():
            raise ValueError("agent_name must be non-empty")
        self._validate_limit("max_total_calls", max_total_calls)
        if default_per_tool_limit is not None:
            self._validate_limit("default_per_tool_limit", default_per_tool_limit, allow_zero=True)

        limits = dict(max_calls_per_tool or {})
        for tool_name, limit in limits.items():
            if not isinstance(tool_name, str) or not tool_name.strip():
                raise ValueError("tool names must be non-empty")
            self._validate_limit(f"max_calls_per_tool[{tool_name!r}]", limit, allow_zero=True)
        if any(not isinstance(tool_name, str) or not tool_name for tool_name in exempt_tools):
            raise ValueError("exempt tool names must be non-empty strings")

        self.agent_name = agent_name
        self.max_total_calls = max_total_calls
        self.max_calls_per_tool = limits
        self.default_per_tool_limit = default_per_tool_limit
        self.exempt_tools = frozenset(exempt_tools)
        self._lock = Lock()
        self._total_calls = 0
        self._calls_by_tool: Counter[str] = Counter()

    @staticmethod
    def _validate_limit(name: str, value: int, *, allow_zero: bool = False) -> None:
        minimum = 0 if allow_zero else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            comparison = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be a {comparison} integer")

    def register_hooks(self, registry: HookRegistry, **_: Any) -> None:
        """Register SDK lifecycle callbacks using the public Strands hook contract."""
        registry.add_callback(BeforeInvocationEvent, self.reset_counts)
        registry.add_callback(BeforeToolCallEvent, self.enforce)

    def reset_counts(self, _: BeforeInvocationEvent) -> None:
        """Start each agent invocation with a fresh budget."""
        with self._lock:
            self._total_calls = 0
            self._calls_by_tool.clear()

    def enforce(self, event: BeforeToolCallEvent) -> None:
        """Cancel a tool call that exceeds the configured invocation budget."""
        invoked_tool_name = event.tool_use.get("name")
        if not isinstance(invoked_tool_name, str) or not invoked_tool_name:
            event.cancel_tool = "Tool call rejected because its name is unavailable."
            logger.warning("Rejected unnamed tool call for agent=%s", self.agent_name)
            return
        tool_name = _logical_tool_name(invoked_tool_name)
        if tool_name in self.exempt_tools:
            return

        with self._lock:
            if self._total_calls >= self.max_total_calls:
                event.cancel_tool = (
                    f"Tool-call budget exhausted for {self.agent_name}. "
                    "Use evidence already collected and complete this stage without more tools."
                )
                logger.warning(
                    "Tool budget exhausted: agent=%s total_limit=%d attempted_tool=%s",
                    self.agent_name,
                    self.max_total_calls,
                    tool_name,
                )
                return

            tool_limit = self.max_calls_per_tool.get(tool_name, self.default_per_tool_limit)
            calls_for_tool = self._calls_by_tool[tool_name]
            if tool_limit is not None and calls_for_tool >= tool_limit:
                event.cancel_tool = (
                    f"Tool '{tool_name}' reached its limit for {self.agent_name}. "
                    "Use evidence already collected and do not repeat this tool call."
                )
                logger.warning(
                    "Per-tool budget exhausted: agent=%s tool=%s tool_limit=%d",
                    self.agent_name,
                    tool_name,
                    tool_limit,
                )
                return

            self._total_calls += 1
            self._calls_by_tool[tool_name] += 1


def trend_research_tool_budget() -> ToolCallBudget:
    """Return the bounded source-research budget for the trend agent."""
    return ToolCallBudget(
        agent_name="trend_researcher",
        max_total_calls=8,
        max_calls_per_tool={
            "hackernews_trending": 2,
            "reddit_search": 2,
            "producthunt_trending": 2,
            "youtube_trending": 1,
            "devto_trending": 1,
            "stackoverflow_search": 1,
            "check_existing_leads": 3,
            "claim_url": 5,
        },
        default_per_tool_limit=1,
    )


def search_specialist_tool_budget() -> ToolCallBudget:
    """Return the four-call cross-prospect enrichment budget for the search agent."""
    return ToolCallBudget(
        agent_name="search_specialist",
        max_total_calls=4,
        max_calls_per_tool={
            "wikipedia_summary": 1,
            "github_search": 2,
            "lobsters_trending": 1,
            "stackoverflow_search": 1,
        },
        default_per_tool_limit=1,
    )


def analysis_tool_budget() -> ToolCallBudget:
    """Return the score-persistence budget for the swarm analysis handoff.

    Score persistence now canonicalizes the top-line score and icp_adjustment from the
    bounded score_breakdown, so ordinary arithmetic drift no longer rejects the batch and
    the common case succeeds on the first call. The budget still allows two
    self-corrections: a genuinely malformed row (e.g. a missing field or an out-of-range
    category) returns field-level ``errors`` the analyst can repair on retry, and this
    headroom keeps such a repair from tripping the swarm-to-graph recovery fallback.
    """
    return ToolCallBudget(
        agent_name="analyst",
        max_total_calls=3,
        max_calls_per_tool={"persist_scored_prospects": 3},
        default_per_tool_limit=0,
    )


def email_generation_tool_budget() -> ToolCallBudget:
    """Return the finite render, verify, and persist budget for outreach."""
    revision_attempts_per_lead = 2
    return ToolCallBudget(
        agent_name="email_generator",
        max_total_calls=1 + (2 * revision_attempts_per_lead + 2) * MAX_LEADS_PER_RUN,
        max_calls_per_tool={
            "retrieve_brand_knowledge": 1,
            "render_email_html_tool": revision_attempts_per_lead * MAX_LEADS_PER_RUN,
            "verify_email_claims": revision_attempts_per_lead * MAX_LEADS_PER_RUN,
            "store_lead": revision_attempts_per_lead * MAX_LEADS_PER_RUN,
            "check_existing_leads": MAX_LEADS_PER_RUN,
        },
        default_per_tool_limit=0,
    )
