"""Coverage tests for previously-untested paths.

Covers:
- _http.py retry/backoff (503→200 retry; non-retryable 404)
- _secrets.py TTL cache (cache hit; cache-miss after clear)
- dynamodb_tool.store_lead dedup by prospect_id, happy path, MAX_LEADS_PER_RUN cap
- grounding_gate.verify_email_claims guardrail skipped when env vars absent
- entrypoint orchestration builders and conditional edge helpers
- email_renderer.render_email_html XSS escaping, compliance footer flag
"""

import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# 1. _http.py retry / backoff
# ---------------------------------------------------------------------------


class TestHttpRetryBackoff:
    """get_with_retry retries on transient 5xx and skips retry on 4xx."""

    def test_retries_on_503_then_returns_200(self):
        from social_intelligence.tools._http import get_with_retry

        resp_503 = MagicMock()
        resp_503.status_code = 503
        resp_200 = MagicMock()
        resp_200.status_code = 200

        with patch("social_intelligence.tools._http.httpx.get", side_effect=[resp_503, resp_200]) as mock_get:
            with patch("social_intelligence.tools._http.time.sleep") as mock_sleep:
                result = get_with_retry("https://api.github.com/repos/org/repo")

        assert result.status_code == 200
        assert mock_get.call_count == 2
        mock_sleep.assert_called_once()

    def test_no_retry_on_non_retryable_404(self):
        from social_intelligence.tools._http import get_with_retry

        resp_404 = MagicMock()
        resp_404.status_code = 404

        with patch("social_intelligence.tools._http.httpx.get", return_value=resp_404) as mock_get:
            with patch("social_intelligence.tools._http.time.sleep") as mock_sleep:
                result = get_with_retry("https://api.github.com/repos/org/missing")

        assert result.status_code == 404
        assert mock_get.call_count == 1
        mock_sleep.assert_not_called()

    def test_sleep_uses_exponential_backoff_on_two_503s(self):
        from social_intelligence.tools._http import _BACKOFF_BASE, get_with_retry

        resp_503 = MagicMock()
        resp_503.status_code = 503
        resp_200 = MagicMock()
        resp_200.status_code = 200

        with patch("social_intelligence.tools._http.httpx.get", side_effect=[resp_503, resp_503, resp_200]):
            with patch("social_intelligence.tools._http.time.sleep") as mock_sleep:
                result = get_with_retry("https://api.github.com/repos/org/slow")

        assert result.status_code == 200
        sleep_calls = [c.args[0] for c in mock_sleep.call_args_list]
        # First sleep = base * 2^0, second = base * 2^1
        assert sleep_calls[0] == pytest.approx(_BACKOFF_BASE * 1)
        assert sleep_calls[1] == pytest.approx(_BACKOFF_BASE * 2)


# ---------------------------------------------------------------------------
# 2. _secrets.py TTL cache
# ---------------------------------------------------------------------------


class TestSecretsTTLCache:
    """get_secret caches the value and calls the client only once per secret."""

    def setup_method(self):
        from social_intelligence.tools import _secrets

        _secrets._cache.clear()

    def teardown_method(self):
        from social_intelligence.tools import _secrets

        _secrets._cache.clear()

    def test_second_call_is_cache_hit(self):
        from social_intelligence.tools import _secrets

        mock_client = MagicMock()
        mock_client.get_secret_value.return_value = {"SecretString": "v1"}

        with patch.object(_secrets, "_get_client", return_value=mock_client):
            first = _secrets.get_secret("test-secret")
            second = _secrets.get_secret("test-secret")

        assert first == "v1"
        assert second == "v1"
        assert mock_client.get_secret_value.call_count == 1

    def test_clearing_cache_forces_new_call(self):
        from social_intelligence.tools import _secrets

        mock_client = MagicMock()
        mock_client.get_secret_value.return_value = {"SecretString": "v1"}

        with patch.object(_secrets, "_get_client", return_value=mock_client):
            _secrets.get_secret("test-secret")

        _secrets._cache.clear()

        mock_client.get_secret_value.return_value = {"SecretString": "v2"}
        with patch.object(_secrets, "_get_client", return_value=mock_client):
            result = _secrets.get_secret("test-secret")

        assert result == "v2"
        assert mock_client.get_secret_value.call_count == 2

    def test_different_secrets_each_call_client(self):
        from social_intelligence.tools import _secrets

        mock_client = MagicMock()
        mock_client.get_secret_value.side_effect = [
            {"SecretString": "alpha"},
            {"SecretString": "beta"},
        ]

        with patch.object(_secrets, "_get_client", return_value=mock_client):
            a = _secrets.get_secret("secret-a")
            b = _secrets.get_secret("secret-b")

        assert a == "alpha"
        assert b == "beta"
        assert mock_client.get_secret_value.call_count == 2

    def test_error_is_not_cached_fail_closed(self):
        """A fetch error propagates and is NOT cached, so a retry re-fetches."""
        from social_intelligence.tools import _secrets

        mock_client = MagicMock()
        mock_client.get_secret_value.side_effect = RuntimeError("AccessDenied")

        with patch.object(_secrets, "_get_client", return_value=mock_client):
            with pytest.raises(RuntimeError):
                _secrets.get_secret("boom")

        # The failure must not have populated the cache.
        assert "boom" not in _secrets._cache

        # A subsequent successful call re-fetches (error was not cached).
        mock_client.get_secret_value.side_effect = None
        mock_client.get_secret_value.return_value = {"SecretString": "recovered"}
        with patch.object(_secrets, "_get_client", return_value=mock_client):
            assert _secrets.get_secret("boom") == "recovered"

    def test_secret_value_is_never_logged(self, caplog):
        """The secret VALUE must never appear in log output (only the id may)."""
        import logging

        from social_intelligence.tools import _secrets

        mock_client = MagicMock()
        mock_client.get_secret_value.return_value = {"SecretString": "super-secret-value-xyz"}

        with caplog.at_level(logging.DEBUG):
            with patch.object(_secrets, "_get_client", return_value=mock_client):
                _secrets.get_secret("test-secret")

        assert "super-secret-value-xyz" not in caplog.text


# ---------------------------------------------------------------------------
# 3. dynamodb_tool.store_lead dedup + cap
# ---------------------------------------------------------------------------


class TestStoreLeadDedupAndCap:
    """store_lead deduplication and per-run cap enforcement."""

    def setup_method(self):
        from social_intelligence.tools import dynamodb_tool

        dynamodb_tool.reset_lead_counter()
        # Restore default cap in case a previous test changed it
        dynamodb_tool.MAX_LEADS_PER_RUN = 3

    def teardown_method(self):
        from social_intelligence.tools import dynamodb_tool

        dynamodb_tool.reset_lead_counter()
        dynamodb_tool.MAX_LEADS_PER_RUN = 3

    def test_prospect_id_dedup_skips_put_item(self):
        from social_intelligence.tools import dynamodb_tool

        mock_table = MagicMock()
        # prospect_id query returns an existing item
        mock_table.query.return_value = {
            "Items": [{"prospect_id": "hn-1", "discovered_at": "2025-01-01T00:00:00+00:00"}]
        }

        with patch.object(dynamodb_tool, "_get_table", return_value=mock_table):
            with patch.object(dynamodb_tool, "_find_by_product_name", return_value=None):
                result = json.loads(dynamodb_tool.store_lead.__wrapped__(prospect_id="hn-1", product_name="Acme"))

        assert result["stored"] is False
        assert "duplicate" in result["reason"].lower()
        mock_table.put_item.assert_not_called()

    def test_happy_path_stores_and_increments_counter(self):
        from social_intelligence.tools import dynamodb_tool

        mock_table = MagicMock()
        mock_table.query.return_value = {"Items": []}
        mock_table.put_item.return_value = {}

        with patch.object(dynamodb_tool, "_get_table", return_value=mock_table):
            with patch.object(dynamodb_tool, "_find_by_product_name", return_value=None):
                result = json.loads(
                    dynamodb_tool.store_lead.__wrapped__(prospect_id="hn-99", product_name="BrandNew", score=75)
                )

        assert result["stored"] is True
        assert result["prospect_id"] == "hn-99"
        assert result["score"] == 75
        assert result["leads_stored_this_run"] == 1
        mock_table.put_item.assert_called_once()

    def test_max_leads_cap_blocks_after_limit(self):
        from social_intelligence.tools import dynamodb_tool

        dynamodb_tool.MAX_LEADS_PER_RUN = 1
        dynamodb_tool.reset_lead_counter()

        mock_table = MagicMock()
        mock_table.query.return_value = {"Items": []}
        mock_table.put_item.return_value = {}

        with patch.object(dynamodb_tool, "_get_table", return_value=mock_table):
            with patch.object(dynamodb_tool, "_find_by_product_name", return_value=None):
                # First store — should succeed
                r1 = json.loads(
                    dynamodb_tool.store_lead.__wrapped__(prospect_id="hn-a", product_name="ProdA", score=80)
                )
                # Second store — should be capped
                r2 = json.loads(
                    dynamodb_tool.store_lead.__wrapped__(prospect_id="hn-b", product_name="ProdB", score=80)
                )

        assert r1["stored"] is True
        assert r2["stored"] is False
        assert "cap" in r2["reason"].lower()
        assert mock_table.put_item.call_count == 1

    def test_reset_lead_counter_allows_fresh_stores(self):
        from social_intelligence.tools import dynamodb_tool

        dynamodb_tool.MAX_LEADS_PER_RUN = 1
        dynamodb_tool.reset_lead_counter()

        mock_table = MagicMock()
        mock_table.query.return_value = {"Items": []}
        mock_table.put_item.return_value = {}

        with patch.object(dynamodb_tool, "_get_table", return_value=mock_table):
            with patch.object(dynamodb_tool, "_find_by_product_name", return_value=None):
                dynamodb_tool.store_lead.__wrapped__(prospect_id="hn-x", product_name="X", score=70)
                # Cap hit
                r_capped = json.loads(
                    dynamodb_tool.store_lead.__wrapped__(prospect_id="hn-y", product_name="Y", score=70)
                )

        assert r_capped["stored"] is False

        # Reset — cap should lift
        dynamodb_tool.reset_lead_counter()
        with patch.object(dynamodb_tool, "_get_table", return_value=mock_table):
            with patch.object(dynamodb_tool, "_find_by_product_name", return_value=None):
                mock_table.query.return_value = {"Items": []}
                r_fresh = json.loads(
                    dynamodb_tool.store_lead.__wrapped__(prospect_id="hn-z", product_name="Z", score=70)
                )

        assert r_fresh["stored"] is True


# ---------------------------------------------------------------------------
# 4. grounding_gate.verify_email_claims guardrail skip
# ---------------------------------------------------------------------------


class TestGroundingGateGuardrailSkip:
    """verify_email_claims skips boto3 call when GUARDRAIL_ID/VERSION are absent."""

    def test_guardrail_skipped_and_returns_valid_score_dict(self):
        from social_intelligence.tools.grounding_gate import verify_email_claims

        env_clean = {k: v for k, v in os.environ.items() if k not in ("GUARDRAIL_ID", "GUARDRAIL_VERSION")}

        with patch.dict(os.environ, env_clean, clear=True):
            with patch("boto3.client") as mock_boto:
                result = json.loads(
                    verify_email_claims.__wrapped__(
                        email_body="Great work on your launch.",
                        evidence_json="{}",
                    )
                )

        mock_boto.assert_not_called()
        assert result["grounding_score"] == 1.0
        assert result["unsupported_claims"] == []
        assert result["guardrail_action"] is None

    def test_guardrail_skipped_with_numeric_claims_still_scores(self):
        """Numeric claims are still evaluated by structured fallback, not guardrail."""
        from social_intelligence.tools.grounding_gate import verify_email_claims

        env_clean = {k: v for k, v in os.environ.items() if k not in ("GUARDRAIL_ID", "GUARDRAIL_VERSION")}

        with patch.dict(os.environ, env_clean, clear=True):
            with patch("boto3.client") as mock_boto:
                result = json.loads(
                    verify_email_claims.__wrapped__(
                        email_body="Congrats on 2400 stars!",
                        evidence_json=json.dumps({"github": "The project crossed 2400 stars."}),
                    )
                )

        mock_boto.assert_not_called()
        assert result["grounding_score"] == 1.0
        assert "2400" in result["supported"] or any("2400" in s for s in result["supported"])


# ---------------------------------------------------------------------------
# 5. entrypoint orchestration builders and conditional edges
# ---------------------------------------------------------------------------

# Module-level fixture: entrypoint needs GATEWAY_URL at import time.
# We insert it into sys.modules under a scoped patch so the test module cache
# is isolated from other test files.

_ENTRYPOINT_MODULE = "entrypoint"


def _import_entrypoint():
    """Import (or return cached) entrypoint with required env + module stubs."""
    if _ENTRYPOINT_MODULE in sys.modules:
        return sys.modules[_ENTRYPOINT_MODULE]

    stubs = {
        "bedrock_agentcore": MagicMock(),
        "bedrock_agentcore.runtime": MagicMock(),
        "mcp_proxy_for_aws": MagicMock(),
        "mcp_proxy_for_aws.client": MagicMock(),
        "strands.tools.mcp": MagicMock(),
        "strands.tools.mcp.mcp_client": MagicMock(),
    }
    for mod, mock in stubs.items():
        if mod not in sys.modules:
            sys.modules[mod] = mock

    with patch.dict(os.environ, {"GATEWAY_URL": "https://example.com/mcp", "AWS_DEFAULT_REGION": "us-east-1"}):
        import importlib

        ep = importlib.import_module(_ENTRYPOINT_MODULE)
    return ep


class TestEntrypointOrchestration:
    """Conditional edge helpers and graph/swarm builders."""

    @pytest.fixture(autouse=True)
    def _ep(self):
        self.ep = _import_entrypoint()

    def _make_node_result(self, score):
        """Build a fake completed NodeResult exposing get_agent_results().

        Mirrors the real Strands shape: NodeResult.get_agent_results() returns
        AgentResult objects, each with a Pydantic structured_output. We use the
        real ScoredProspectList model so model_dump() behaves like production.
        """
        from strands.multiagent.base import Status

        from social_intelligence.schemas.models import ScoredProspect, ScoredProspectList

        structured = ScoredProspectList(
            prospects=[
                ScoredProspect(
                    prospect_id="hn-1",
                    product_name="Acme",
                    score=score,
                    confidence=0.9,
                    reasoning="strong multi-signal",
                )
            ]
        )
        agent_result = SimpleNamespace(structured_output=structured)
        return SimpleNamespace(
            status=Status.COMPLETED,
            result=agent_result,
            get_agent_results=lambda: [agent_result],
        )

    def test_score_above_threshold_true(self):
        node = self._make_node_result(88)
        state = SimpleNamespace(results={"analysis": node})
        assert self.ep._score_above_threshold(state) is True

    def test_score_below_threshold_false(self):
        node = self._make_node_result(40)
        state = SimpleNamespace(results={"analysis": node})
        assert self.ep._score_above_threshold(state) is False

    def test_score_missing_analysis_false(self):
        state = SimpleNamespace(results={})
        assert self.ep._score_above_threshold(state) is False

    def test_score_from_structured_output_dict_path(self):
        """The gate reads structured_output via model_dump(), not text parsing."""
        max_score = self.ep._max_score_from_structured(self._make_node_result(75).get_agent_results())
        assert max_score == 75

    def test_text_fallback_matches_scored_prospect_only(self):
        """Text fallback matches ScoredProspect scores, not HN/Reddit community scores."""
        # A ScoredProspect-shaped blob: score near reasoning/confidence -> matched.
        prospect_text = '{"prospect_id": "x", "score": 82, "confidence": 0.8, "reasoning": "good"}'
        assert self.ep._max_prospect_score_in_text(prospect_text) == 82
        # A raw HN story: score near comments/author -> NOT matched (returns None).
        hn_text = '{"title": "Show HN", "score": 297, "comments": 71, "author": "dev"}'
        assert self.ep._max_prospect_score_in_text(hn_text) is None

    def test_text_fallback_matches_prose_scores(self):
        """The fallback also catches prose-form analysis scores (not just JSON)."""
        assert self.ep._max_prospect_score_in_text("Prospect A scored 88/100, strong fit") == 88
        assert self.ep._max_prospect_score_in_text("OpenAI DayBreak (score: 72) recommendation_seeking") == 72
        # Out-of-range community scores in prose are excluded (>100).
        assert self.ep._max_prospect_score_in_text("hit HN with a score of 297 and 71 comments") is None

    def test_all_dependencies_complete_both_done(self):
        from strands.multiagent.base import Status

        checker = self.ep._all_dependencies_complete(["research", "search"])
        node = SimpleNamespace(status=Status.COMPLETED)
        state = SimpleNamespace(results={"research": node, "search": node})
        assert checker(state) is True

    def test_all_dependencies_complete_one_missing(self):
        from strands.multiagent.base import Status

        checker = self.ep._all_dependencies_complete(["research", "search"])
        node = SimpleNamespace(status=Status.COMPLETED)
        state = SimpleNamespace(results={"research": node})
        assert checker(state) is False

    def test_build_graph_returns_graph_object(self):
        tools = {"trend": [], "enrichment": [], "email": []}
        graph = self.ep._build_graph(tools)
        assert type(graph).__name__ == "Graph"

    def test_build_swarm_returns_swarm_object(self):
        from strands.multiagent import Swarm

        tools = {"trend": [], "enrichment": [], "email": []}
        swarm = self.ep._build_swarm(tools)
        assert isinstance(swarm, Swarm)


class TestEntrypointDiagnostics:
    """Per-run pipeline diagnostics that make an empty/no-op run observable."""

    @pytest.fixture(autouse=True)
    def _ep(self):
        self.ep = _import_entrypoint()
        self.ep._reset_run_diag()

    def test_event_type_normalizes_spelling(self):
        assert self.ep._event_type({"type": "multi_agent_node_stop"}) == "multiagent_node_stop"
        assert self.ep._event_type({"type": "multiagent_node_start"}) == "multiagent_node_start"
        assert self.ep._event_type("not-a-dict") == ""

    def test_tracks_node_lifecycle_and_tool_calls(self):
        self.ep._log_pipeline_event({"type": "multiagent_node_start", "node_id": "research"})
        self.ep._log_pipeline_event({"type": "multi_agent_node_start", "node_id": "search"})
        self.ep._log_pipeline_event(
            {"type": "multiagent_node_stream", "event": {"message": {"content": [{"toolUse": {"name": "hn"}}]}}}
        )
        self.ep._log_pipeline_event({"type": "multiagent_node_stop", "node_id": "research"})
        assert self.ep._RUN_DIAG["started"] == {"research", "search"}
        assert self.ep._RUN_DIAG["completed"] == {"research"}
        assert self.ep._RUN_DIAG["tool_calls"] == 1

    def test_log_pipeline_event_tolerates_bad_shapes(self):
        # Must never raise, regardless of event shape.
        for bad in (None, "x", 42, {}, {"type": "multiagent_node_stop"}, {"type": "unknown"}):
            self.ep._log_pipeline_event(bad)
        assert self.ep._RUN_DIAG["tool_calls"] == 0

    def test_reset_clears_diagnostics(self):
        self.ep._log_pipeline_event({"type": "multiagent_node_start", "node_id": "n"})
        self.ep._reset_run_diag()
        assert self.ep._RUN_DIAG == {"started": set(), "completed": set(), "tool_calls": 0}


class TestEntrypointPayloadAndSession:
    """Pure payload/session helpers extracted from the async invoke handler."""

    @pytest.fixture(autouse=True)
    def _ep(self):
        self.ep = _import_entrypoint()

    def test_resolve_payload_defaults_on_none(self):
        prompt, pattern, sid, actor = self.ep._resolve_payload(None)
        assert prompt == self.ep.DEFAULT_PROMPT
        assert pattern == "graph"
        assert sid == "" and actor == ""

    def test_resolve_payload_reads_all_fields(self):
        result = self.ep._resolve_payload({"prompt": "hi", "pattern": "SWARM", "session_id": "s1", "actor_id": "a1"})
        assert result == ("hi", "swarm", "s1", "a1")

    def test_resolve_payload_unknown_pattern_falls_back_to_graph(self):
        assert self.ep._resolve_payload({"pattern": "nonsense"})[1] == "graph"

    def test_resolve_payload_non_dict_is_tolerated(self):
        prompt, pattern, sid, actor = self.ep._resolve_payload("not-a-dict")
        assert prompt == self.ep.DEFAULT_PROMPT and pattern == "graph"

    def test_memory_disabled_without_session_id(self):
        # MEMORY_ID is unset in the test environment, so memory is always disabled.
        assert self.ep._memory_enabled("") is False
        assert self.ep._memory_enabled("s1") is False

    def test_resolve_actor_id_uses_caller_value(self):
        assert self.ep._resolve_actor_id("analyst-7") == "analyst-7"

    def test_resolve_actor_id_defaults_to_dated_user(self):
        actor = self.ep._resolve_actor_id("")
        assert actor.startswith("user_") and len(actor) == len("user_") + 8

    def test_build_session_manager_none_when_disabled(self):
        assert self.ep._build_session_manager("s1", "a1") is None

    def test_build_orchestrator_dispatches_by_pattern(self):
        from strands.multiagent import Swarm

        tools = {"trend": [], "enrichment": [], "email": []}
        assert type(self.ep._build_orchestrator("graph", tools, None)).__name__ == "Graph"
        assert isinstance(self.ep._build_orchestrator("swarm", tools, None), Swarm)


# ---------------------------------------------------------------------------
# 6. email_renderer.render_email_html
# ---------------------------------------------------------------------------


class TestEmailRenderer:
    """HTML rendering, XSS escaping, and compliance footer."""

    def test_returns_rendered_true_and_has_html(self):
        from social_intelligence.tools.email_renderer import render_email_html

        result = json.loads(render_email_html(prospect_id="p-1", subject="Hello World", body="Body text here."))
        assert result["rendered"] is True
        assert "html" in result
        assert len(result["html"]) > 50

    def test_xss_script_tag_is_escaped(self):
        from social_intelligence.tools.email_renderer import render_email_html

        result = json.loads(
            render_email_html(
                prospect_id="p-xss",
                subject="<script>alert(1)</script>",
                body="Normal body.",
            )
        )
        html = result["html"]
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html

    def test_compliance_footer_absent_by_default(self):
        from social_intelligence.tools.email_renderer import render_email_html

        env_clean = {k: v for k, v in os.environ.items() if k != "COMPLIANCE_FOOTER_REQUIRED"}
        with patch.dict(os.environ, env_clean, clear=True):
            result = json.loads(render_email_html(prospect_id="p-2", subject="Hi", body="Footer test."))

        assert "Unsubscribe" not in result["html"]

    def test_compliance_footer_present_when_env_set(self):
        from social_intelligence.tools.email_renderer import render_email_html

        with patch.dict(os.environ, {"COMPLIANCE_FOOTER_REQUIRED": "true"}):
            result = json.loads(render_email_html(prospect_id="p-3", subject="Hi", body="Footer test."))

        assert "Unsubscribe" in result["html"]
        assert "123 Main St" in result["html"]

    def test_prospect_id_and_subject_in_result(self):
        from social_intelligence.tools.email_renderer import render_email_html

        result = json.loads(render_email_html(prospect_id="p-meta", subject="My Subject", body="Content."))
        assert result["prospect_id"] == "p-meta"
        assert result["subject"] == "My Subject"
