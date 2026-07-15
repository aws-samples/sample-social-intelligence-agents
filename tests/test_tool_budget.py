"""Tests for SDK-enforced per-invocation Strands tool budgets."""

from unittest.mock import MagicMock

import pytest
from strands.hooks import BeforeInvocationEvent, BeforeToolCallEvent

from social_intelligence.config import MAX_LEADS_PER_RUN
from social_intelligence.orchestration.tool_budget import (
    ToolCallBudget,
    email_generation_tool_budget,
    search_specialist_tool_budget,
)


def _before_tool_call(tool_name: str) -> BeforeToolCallEvent:
    """Build a real public Strands hook event for a synthetic tool request."""
    return BeforeToolCallEvent(
        agent=MagicMock(),
        selected_tool=None,
        tool_use={"toolUseId": f"tool-{tool_name}", "name": tool_name, "input": {}},
        invocation_state={},
    )


def _before_invocation() -> BeforeInvocationEvent:
    return BeforeInvocationEvent(agent=MagicMock(), invocation_state={})


def test_total_limit_cancels_the_next_call() -> None:
    budget = ToolCallBudget(agent_name="test", max_total_calls=2)
    budget.reset_counts(_before_invocation())

    assert budget.enforce(_before_tool_call("first")) is None
    assert budget.enforce(_before_tool_call("second")) is None
    blocked = _before_tool_call("third")
    budget.enforce(blocked)

    assert blocked.cancel_tool
    assert "budget exhausted" in str(blocked.cancel_tool)


def test_per_tool_limit_and_invocation_reset() -> None:
    budget = ToolCallBudget(agent_name="test", max_total_calls=5, max_calls_per_tool={"search": 1})
    budget.reset_counts(_before_invocation())
    budget.enforce(_before_tool_call("search"))
    blocked = _before_tool_call("search")
    budget.enforce(blocked)
    assert "reached its limit" in str(blocked.cancel_tool)

    budget.reset_counts(_before_invocation())
    fresh = _before_tool_call("search")
    budget.enforce(fresh)
    assert fresh.cancel_tool is False


def test_handoff_is_exempt_and_unknown_tools_can_be_denied() -> None:
    budget = ToolCallBudget(
        agent_name="test",
        max_total_calls=1,
        max_calls_per_tool={"known_tool": 1},
        default_per_tool_limit=0,
    )
    budget.reset_counts(_before_invocation())

    handoff = _before_tool_call("handoff_to_agent")
    budget.enforce(handoff)
    blocked = _before_tool_call("unexpected_tool")
    budget.enforce(blocked)
    allowed = _before_tool_call("known_tool")
    budget.enforce(allowed)

    assert handoff.cancel_tool is False
    assert "reached its limit" in str(blocked.cancel_tool)
    assert allowed.cancel_tool is False


def test_gateway_prefix_uses_the_schema_tool_budget() -> None:
    budget = ToolCallBudget(agent_name="test", max_total_calls=2, max_calls_per_tool={"search": 1})
    budget.reset_counts(_before_invocation())

    budget.enforce(_before_tool_call("social-intel-tools___search"))
    blocked = _before_tool_call("social-intel-tools___search")
    budget.enforce(blocked)

    assert "reached its limit" in str(blocked.cancel_tool)


def test_structured_output_is_exempt_from_external_tool_budgets() -> None:
    budget = ToolCallBudget(agent_name="test", max_total_calls=1)
    budget.reset_counts(_before_invocation())

    budget.enforce(_before_tool_call("EnrichmentData"))
    budget.enforce(_before_tool_call("external_tool"))
    blocked = _before_tool_call("second_external_tool")
    budget.enforce(blocked)

    assert blocked.cancel_tool
    assert "budget exhausted" in str(blocked.cancel_tool)


def test_invalid_budget_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        ToolCallBudget(agent_name="test", max_total_calls=0)
    with pytest.raises(ValueError, match="non-empty"):
        ToolCallBudget(agent_name="", max_total_calls=1)


def test_email_budget_covers_two_complete_persistence_attempts_per_lead() -> None:
    budget = email_generation_tool_budget()

    assert budget.max_calls_per_tool["retrieve_brand_knowledge"] == 1
    assert budget.max_calls_per_tool["render_email_html_tool"] == 2 * MAX_LEADS_PER_RUN
    assert budget.max_calls_per_tool["verify_email_claims"] == 2 * MAX_LEADS_PER_RUN
    assert budget.max_calls_per_tool["store_lead"] == 2 * MAX_LEADS_PER_RUN
    assert budget.max_total_calls == 1 + (6 * MAX_LEADS_PER_RUN)


def test_search_budget_enforces_the_four_call_cross_prospect_plan() -> None:
    budget = search_specialist_tool_budget()

    assert budget.max_total_calls == 4
    assert budget.max_calls_per_tool == {
        "wikipedia_summary": 1,
        "github_search": 2,
        "lobsters_trending": 1,
        "stackoverflow_search": 1,
    }
