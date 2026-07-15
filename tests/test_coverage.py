"""Coverage tests for previously-untested paths.

Covers:
- _http.py retry/backoff (503→200 retry; non-retryable 404)
- _secrets.py TTL cache (cache hit; cache-miss after clear)
- dynamodb_tool.store_lead dedup by prospect_id, happy path, MAX_LEADS_PER_RUN cap
- grounding_gate.verify_email_claims guardrail skipped when env vars absent
- entrypoint orchestration builders and conditional edge helpers
- email_renderer.render_email_html XSS escaping, compliance footer flag
"""

import asyncio
import json
import os
import sys
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _score_breakdown_for(score: int) -> dict[str, int]:
    """Build a valid medium-ICP breakdown for a bounded score."""
    remaining = score
    values: dict[str, int] = {"icp_adjustment": 0}
    for name, cap in (
        ("topical_alignment", 25),
        ("timing_relevance", 20),
        ("engagement_potential", 20),
        ("intent_signal_strength", 20),
        ("data_quality", 15),
    ):
        contribution = min(cap, max(0, remaining))
        values[name] = contribution
        remaining -= contribution
    return values


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
        mock_table.meta.client.transact_write_items.assert_not_called()

    def test_happy_path_stores_and_increments_counter(self):
        from social_intelligence.tools import dynamodb_tool

        mock_table = MagicMock()
        mock_table.query.return_value = {"Items": []}
        mock_table.put_item.return_value = {}

        with patch.object(dynamodb_tool, "_get_table", return_value=mock_table):
            with patch.object(dynamodb_tool, "_find_by_product_name", return_value=None):
                result = json.loads(
                    dynamodb_tool.store_lead.__wrapped__(
                        prospect_id="hn-99",
                        product_name="BrandNew",
                        score=75,
                        top_trends=["recent launch", "open-source momentum"],
                    )
                )

        assert result["stored"] is True
        assert result["prospect_id"] == "hn-99"
        assert result["score"] == 75
        assert result["leads_stored_this_run"] == 1
        writes = mock_table.meta.client.transact_write_items.call_args.kwargs["TransactItems"]
        assert len(writes) == 3
        assert writes[-1]["Put"]["Item"]["top_trends"] == ["recent launch", "open-source momentum"]

    def test_run_session_id_stamped_on_stored_lead(self):
        """store_lead writes the run session id set by the entrypoint onto the item."""
        from social_intelligence.tools import dynamodb_tool

        mock_table = MagicMock()
        mock_table.query.return_value = {"Items": []}
        mock_table.put_item.return_value = {}

        dynamodb_tool.reset_lead_counter()
        dynamodb_tool.set_run_session_id("sess-abc-123")
        try:
            with patch.object(dynamodb_tool, "_get_table", return_value=mock_table):
                with patch.object(dynamodb_tool, "_find_by_product_name", return_value=None):
                    dynamodb_tool.store_lead.__wrapped__(prospect_id="hn-s1", product_name="SessProd", score=70)
        finally:
            dynamodb_tool.set_run_session_id("")  # reset shared state for other tests

        writes = mock_table.meta.client.transact_write_items.call_args.kwargs["TransactItems"]
        stored_item = writes[-1]["Put"]["Item"]
        assert stored_item["session_id"] == "sess-abc-123"
        assert stored_item["dedup_partition"] == "LEAD"

    def test_persist_analysis_scores_writes_score_partition_records(self):
        """Analysis scores persist under the SCORE partition, stamped with the run session,
        so evaluation reads them back even for sub-threshold prospects that never email."""
        from social_intelligence.tools import dynamodb_tool

        mock_table = MagicMock()
        dynamodb_tool.set_run_session_id("run-scores-1")
        try:
            with patch.object(dynamodb_tool, "_get_table", return_value=mock_table):
                written = dynamodb_tool.persist_analysis_scores(
                    [
                        {
                            "prospect_id": "p1",
                            "product_name": "Alpha",
                            "score": 88,
                            "score_breakdown": _score_breakdown_for(88),
                        },
                        {
                            "prospect_id": "p2",
                            "product_name": "Beta",
                            "score": 30,
                            "score_breakdown": _score_breakdown_for(30),
                        },
                        {
                            "prospect_id": "p3",
                            "product_name": "Bad",
                            "score": 150,
                            "score_breakdown": _score_breakdown_for(100),
                        },  # out of range, skipped
                    ]
                )
        finally:
            dynamodb_tool.set_run_session_id("")

        assert written == 2
        items = [call.kwargs["Item"] for call in mock_table.put_item.call_args_list]
        assert {it["score"] for it in items} == {88, 30}
        assert all(it["dedup_partition"] == "SCORE" for it in items)
        assert all(it["session_id"] == "run-scores-1" for it in items)
        metrics = dynamodb_tool.get_run_output_metrics()
        assert metrics.scores_persisted == 2
        assert metrics.email_eligible_scores == 0

    def test_persist_scored_prospects_accepts_native_swarm_handoff_objects(self):
        """The swarm-only persistence tool stores low scores as well as qualified leads."""
        from social_intelligence.tools import dynamodb_tool

        mock_table = MagicMock()
        dynamodb_tool.set_run_session_id("swarm-run-1")
        try:
            with patch.object(dynamodb_tool, "_get_table", return_value=mock_table):
                result = json.loads(
                    dynamodb_tool.persist_scored_prospects.__wrapped__(
                        [
                            {
                                "prospect_id": "high",
                                "product_name": "High",
                                "score": 81,
                                "confidence": 0.9,
                                "score_breakdown": _score_breakdown_for(81),
                                "reasoning": "High-quality recent launch signal.",
                            },
                            {
                                "prospect_id": "low",
                                "product_name": "Low",
                                "score": 24,
                                "confidence": 0.5,
                                "score_breakdown": _score_breakdown_for(24),
                                "reasoning": "Low ICP fit and weak evidence.",
                            },
                        ]
                    )
                )
        finally:
            dynamodb_tool.set_run_session_id("")

        assert result == {"stored": True, "persisted": 2}
        items = [call.kwargs["Item"] for call in mock_table.put_item.call_args_list]
        assert {item["score"] for item in items} == {24, 81}
        schema = dynamodb_tool.persist_scored_prospects.tool_spec["inputSchema"]
        assert schema["properties"]["prospects"]["type"] == "array"
        assert "prospects_json" not in schema["properties"]

    def test_persist_analysis_scores_keeps_duplicate_source_ids_distinct(self):
        """Duplicate IDs in one model response must not overwrite a score record."""
        from social_intelligence.tools import dynamodb_tool

        mock_table = MagicMock()
        dynamodb_tool.set_run_session_id("run-duplicates-1")
        try:
            with patch.object(dynamodb_tool, "_get_table", return_value=mock_table):
                written = dynamodb_tool.persist_analysis_scores(
                    [
                        {
                            "prospect_id": "same-id",
                            "product_name": "Alpha",
                            "score": 80,
                            "score_breakdown": _score_breakdown_for(80),
                        },
                        {
                            "prospect_id": "same-id",
                            "product_name": "Beta",
                            "score": 35,
                            "score_breakdown": _score_breakdown_for(35),
                        },
                    ]
                )
        finally:
            dynamodb_tool.set_run_session_id("")

        assert written == 2
        keys = [call.kwargs["Item"]["prospect_id"] for call in mock_table.put_item.call_args_list]
        assert len(set(keys)) == 2

    def test_persist_scored_prospects_requires_and_persists_score_breakdown(self):
        """Swarm score persistence validates the same arithmetic as Graph output."""
        from social_intelligence.tools import dynamodb_tool

        mock_table = MagicMock()
        dynamodb_tool.set_run_session_id("minimal-score-run")
        try:
            with patch.object(dynamodb_tool, "_get_table", return_value=mock_table):
                result = json.loads(
                    dynamodb_tool.persist_scored_prospects.__wrapped__(
                        [
                            {
                                "prospect_id": "minimal",
                                "score": 70,
                                "score_breakdown": _score_breakdown_for(70),
                            }
                        ]
                    )
                )
        finally:
            dynamodb_tool.set_run_session_id("")

        assert result == {"stored": True, "persisted": 1}
        item = mock_table.put_item.call_args.kwargs["Item"]
        assert item["prospect_id"].startswith("score::minimal-score-run::0::")
        assert item["score_breakdown"] == _score_breakdown_for(70)

    def test_persist_scored_prospects_rejects_missing_identity(self):
        """The compact persistence contract still rejects unusable score rows."""
        from social_intelligence.tools import dynamodb_tool

        result = json.loads(dynamodb_tool.persist_scored_prospects.__wrapped__([{"score": 70}]))

        assert result["stored"] is False
        assert "score persistence contract" in result["reason"]

    def test_persist_scored_prospects_accepts_an_empty_completed_analysis(self):
        """A valid no-prospect analysis is distinct from a missing Swarm handoff."""
        from social_intelligence.tools import dynamodb_tool

        dynamodb_tool.reset_lead_counter()
        result = json.loads(dynamodb_tool.persist_scored_prospects.__wrapped__([]))

        assert result == {"stored": True, "persisted": 0}
        metrics = dynamodb_tool.get_run_output_metrics()
        assert metrics.score_persistence_calls == 1
        assert metrics.scores_requested == 0
        assert metrics.scores_persisted == 0

    def test_persist_run_status_writes_session_visible_completion_marker(self):
        from social_intelligence.tools import dynamodb_tool

        mock_table = MagicMock()
        dynamodb_tool.set_run_session_id("run-complete-1")
        try:
            with patch.object(dynamodb_tool, "_get_table", return_value=mock_table):
                dynamodb_tool.persist_run_status(False, execution_path="graph_recovery")
        finally:
            dynamodb_tool.set_run_session_id("")

        item = mock_table.put_item.call_args.kwargs["Item"]
        assert item["product_name"] == "__run_status__:failed"
        assert item["session_id"] == "run-complete-1"
        assert item["execution_path"] == "graph_recovery"

    def test_run_isolation_ignores_other_runs_product_match(self):
        """Under isolation, a product stored by a different run is not a duplicate."""
        from social_intelligence.tools import dynamodb_tool

        # GSI returns a key hit; the full item belongs to a DIFFERENT run's session.
        mock_table = MagicMock()
        mock_table.query.return_value = {"Items": [{"prospect_id": "hn-x", "discovered_at": "t"}]}
        other = {"prospect_id": "hn-x", "product_name": "Dup", "session_id": "other-run"}
        mock_table.get_item.return_value = {"Item": other}

        dynamodb_tool.set_run_session_id("my-run")
        dynamodb_tool.set_run_isolation(True)
        try:
            hit = dynamodb_tool._find_by_product_name(mock_table, "Dup")
        finally:
            dynamodb_tool.set_run_isolation(False)
            dynamodb_tool.set_run_session_id("")

        assert hit is None  # different run's lead is not a duplicate for this run

    def test_run_isolation_matches_same_run_product(self):
        """Under isolation, a product stored by THIS run is still a duplicate."""
        from social_intelligence.tools import dynamodb_tool

        mock_table = MagicMock()
        mock_table.query.return_value = {"Items": [{"prospect_id": "hn-y", "discovered_at": "t"}]}
        mine = {"prospect_id": "hn-y", "product_name": "Dup", "session_id": "my-run"}
        mock_table.get_item.return_value = {"Item": mine}

        dynamodb_tool.set_run_session_id("my-run")
        dynamodb_tool.set_run_isolation(True)
        try:
            hit = dynamodb_tool._find_by_product_name(mock_table, "Dup")
        finally:
            dynamodb_tool.set_run_isolation(False)
            dynamodb_tool.set_run_session_id("")

        assert hit is not None
        assert hit["prospect_id"] == "hn-y"

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
        assert mock_table.meta.client.transact_write_items.call_count == 1

    def test_concurrent_calls_reserve_lead_capacity_before_dynamodb_work(self):
        """ContextVar-backed child threads share one reservation counter."""
        from social_intelligence.tools import dynamodb_tool

        dynamodb_tool.MAX_LEADS_PER_RUN = 1
        dynamodb_tool.reset_lead_counter()
        query_started = threading.Event()
        release_query = threading.Event()

        class BlockingTable:
            def __init__(self):
                self.writes: list[dict] = []
                self.name = "test-leads"
                self.meta = SimpleNamespace(client=SimpleNamespace(transact_write_items=self._write_transaction))

            def query(self, **_):
                query_started.set()
                release_query.wait(timeout=1)
                return {"Items": []}

            def _write_transaction(self, *, TransactItems):
                self.writes.append(TransactItems[-1]["Put"]["Item"])

        table = BlockingTable()

        def store(prospect_id: str) -> dict:
            return json.loads(
                dynamodb_tool.store_lead.__wrapped__(
                    prospect_id=prospect_id,
                    product_name=prospect_id,
                    score=80,
                )
            )

        async def run_concurrently():
            first = asyncio.create_task(asyncio.to_thread(store, "hn-first"))
            assert await asyncio.to_thread(query_started.wait, 1)
            second = asyncio.create_task(asyncio.to_thread(store, "hn-second"))
            await asyncio.sleep(0)
            release_query.set()
            return await asyncio.gather(first, second)

        with patch.object(dynamodb_tool, "_get_table", return_value=table):
            with patch.object(dynamodb_tool, "_find_by_product_name", return_value=None):
                first, second = asyncio.run(run_concurrently())

        assert sum(result["stored"] for result in (first, second)) == 1
        assert len(table.writes) == 1

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


class TestGroundingGateVerification:
    """verify_email_claims is independent of inline model guardrail settings."""

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

    def _make_node_result(self, score, include_corroborating_evidence: bool = True):
        """Build a fake completed NodeResult exposing get_agent_results().

        Mirrors the real Strands shape: NodeResult.get_agent_results() returns
        AgentResult objects, each with a Pydantic structured_output. We use the
        real ScoredProspectList model so model_dump() behaves like production.
        """
        from strands.multiagent.base import Status

        from social_intelligence.schemas.models import EvidenceItem, ScoredProspect, ScoredProspectList

        evidence = []
        if include_corroborating_evidence:
            evidence = [
                EvidenceItem(source="hackernews", url="https://news.ycombinator.com/item?id=1", fact="Launched"),
                EvidenceItem(source="github", url="https://github.com/acme/project", fact="Active repository"),
            ]

        structured = ScoredProspectList(
            prospects=[
                ScoredProspect(
                    prospect_id="hn-1",
                    product_name="Acme",
                    score=score,
                    score_breakdown=_score_breakdown_for(score),
                    confidence=0.9,
                    reasoning="strong multi-signal",
                    evidence=evidence,
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

    def test_score_without_independent_sources_false(self):
        node = self._make_node_result(88, include_corroborating_evidence=False)
        state = SimpleNamespace(results={"analysis": node})
        assert self.ep._score_above_threshold(state) is False

    def test_score_missing_analysis_false(self):
        state = SimpleNamespace(results={})
        assert self.ep._score_above_threshold(state) is False

    def test_score_from_structured_output_dict_path(self):
        """The gate reads structured_output via model_dump(), not text parsing."""
        prospects = self.ep._scored_prospects_from_structured(self._make_node_result(75).get_agent_results())
        assert [prospect["score"] for prospect in prospects] == [75]

    def test_text_only_analysis_output_never_passes_email_gate(self):
        """Raw HN or prose scores cannot trigger outreach without typed output."""
        from strands.multiagent.base import Status

        text_result = SimpleNamespace(structured_output=None)
        node = SimpleNamespace(
            status=Status.COMPLETED,
            result="HN scored 82 points today.",
            get_agent_results=lambda: [text_result],
        )
        state = SimpleNamespace(results={"analysis": node})
        assert self.ep._score_above_threshold(state) is False

    def test_analysis_scores_event_uses_node_result_structured_output(self):
        node = self._make_node_result(75)
        event = {
            "type": "multiagent_node_stop",
            "node_id": "analysis",
            "node_result": node,
        }

        scores_event = self.ep._analysis_scores_event(event)

        assert scores_event["type"] == "analysis_scores"
        assert len(scores_event["prospects"]) == 1
        prospect = scores_event["prospects"][0]
        assert prospect["prospect_id"] == "hn-1"
        assert prospect["score"] == 75
        assert prospect["score_breakdown"] == _score_breakdown_for(75)
        assert prospect["independent_source_count"] == 2
        assert prospect["email_eligible"] is True

    def test_pipeline_persists_analysis_scores_before_yielding_node_stop(self, monkeypatch):
        """A disconnected streaming client cannot lose the completed analysis score rows."""
        from contextlib import nullcontext

        from strands.multiagent.base import Status

        node_stop = {
            "type": "multiagent_node_stop",
            "node_id": "analysis",
            "node_result": self._make_node_result(75),
        }
        terminal = {"type": "multiagent_result", "result": SimpleNamespace(status=Status.COMPLETED)}

        class _FakeOrchestrator:
            async def stream_async(self, _prompt):
                yield node_stop
                yield terminal

        persisted: list[list[dict]] = []
        monkeypatch.setattr(
            self.ep,
            "_gateway_tools",
            lambda: nullcontext({"trend": [], "enrichment": [], "email": []}),
        )
        monkeypatch.setattr(self.ep, "_build_orchestrator", lambda *_: _FakeOrchestrator())
        monkeypatch.setattr(
            self.ep,
            "persist_analysis_scores",
            lambda prospects: persisted.append(prospects) or len(prospects),
        )

        async def consume_first_event():
            stream = self.ep._run_pipeline("prompt", "graph", None, include_scored_prospects=False)
            event = await anext(stream)
            await stream.aclose()
            return event

        event = asyncio.run(consume_first_event())

        assert event is node_stop
        assert len(persisted) == 1
        assert persisted[0][0]["prospect_id"] == "hn-1"

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
        assert graph.node_timeout == 960

    def test_graph_passes_discovered_prospects_to_search_then_analysis(self):
        """Search must receive TrendData before producing matching EnrichmentData."""
        tools = {"trend": [], "enrichment": [], "email": []}
        graph = self.ep._build_graph(tools)

        assert {node.node_id for node in graph.entry_points} == {"research"}
        assert {node.node_id for node in graph.nodes["search"].dependencies} == {"research"}
        assert {node.node_id for node in graph.nodes["analysis"].dependencies} == {"research", "search"}

    def test_build_swarm_returns_swarm_object(self):
        from strands.multiagent import Swarm

        tools = {"trend": [], "enrichment": [], "email": []}
        swarm = self.ep._build_swarm(tools)
        assert isinstance(swarm, Swarm)
        assert swarm.max_handoffs == 5
        assert swarm.max_iterations == 6
        assert swarm.node_timeout == 960

    def test_gateway_tool_selection_keeps_roles_scoped(self):
        tools = [
            SimpleNamespace(tool_name="hackernews_trending"),
            SimpleNamespace(tool_name="social-intel-tools___reddit_search"),
            SimpleNamespace(tool_name="github_search"),
            SimpleNamespace(tool_name="wikipedia_summary"),
            SimpleNamespace(tool_name="social-intel-web-search___WebSearch"),
        ]

        trend = self.ep._select_gateway_tools(
            tools,
            self.ep._TREND_GATEWAY_TOOL_NAMES,
            self.ep._TREND_OPTIONAL_GATEWAY_TOOL_NAMES,
        )
        enrichment = self.ep._select_gateway_tools(tools, self.ep._ENRICHMENT_GATEWAY_TOOL_NAMES)

        assert [tool.tool_name for tool in trend] == [
            "hackernews_trending",
            "social-intel-tools___reddit_search",
            "social-intel-web-search___WebSearch",
        ]
        assert [tool.tool_name for tool in enrichment] == ["github_search", "wikipedia_summary"]


class TestEntrypointDiagnostics:
    """Per-run pipeline diagnostics that make an empty/no-op run observable."""

    @pytest.fixture(autouse=True)
    def _ep(self):
        from social_intelligence.tools import dynamodb_tool

        self.ep = _import_entrypoint()
        self.ep._reset_run_diag()
        dynamodb_tool.reset_lead_counter()

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
        diagnostics = self.ep._current_run_diag()
        assert diagnostics["started"] == {"research", "search"}
        assert diagnostics["completed"] == {"research"}
        assert diagnostics["tool_calls"] == 1

    def test_log_pipeline_event_tolerates_bad_shapes(self):
        # Must never raise, regardless of event shape.
        for bad in (None, "x", 42, {}, {"type": "multiagent_node_stop"}, {"type": "unknown"}):
            self.ep._log_pipeline_event(bad)
        assert self.ep._current_run_diag()["tool_calls"] == 0

    def test_reset_clears_diagnostics(self):
        self.ep._log_pipeline_event({"type": "multiagent_node_start", "node_id": "n"})
        self.ep._reset_run_diag()
        assert self.ep._current_run_diag() == {"started": set(), "completed": set(), "tool_calls": 0}

    def test_parallel_invocations_keep_diagnostics_separate(self):
        async def invoke(node_id: str):
            self.ep._reset_run_diag()
            self.ep._log_pipeline_event({"type": "multiagent_node_start", "node_id": node_id})
            await asyncio.sleep(0)
            return self.ep._current_run_diag()

        async def run_parallel():
            return await asyncio.gather(invoke("research"), invoke("email"))

        first, second = asyncio.run(run_parallel())

        assert first == {"started": {"research"}, "completed": set(), "tool_calls": 0}
        assert second == {"started": {"email"}, "completed": set(), "tool_calls": 0}

    def test_swarm_recovery_requires_analysis_handoff(self):
        assert self.ep._swarm_recovery_reason() == "the analyst did not persist a scored-prospect handoff"

    def test_swarm_recovery_requires_lead_for_eligible_score(self):
        from social_intelligence.tools import dynamodb_tool

        state = dynamodb_tool._current_run_state()
        with state.lock:
            state.score_persistence_calls = 1
            state.scores_requested = 1
            state.scores_persisted = 1
            state.email_eligible_scores = 1

        assert self.ep._swarm_recovery_reason() == "email-eligible analysis scores did not produce a stored lead"

    def test_terminal_result_requires_completed_status(self):
        from strands.multiagent.base import Status

        completed = {"result": SimpleNamespace(status=Status.COMPLETED)}
        self.ep._require_completed_result(completed, "graph")

        with pytest.raises(RuntimeError, match="status failed"):
            self.ep._require_completed_result({"result": SimpleNamespace(status=Status.FAILED)}, "graph")

    def test_swarm_no_op_recovers_before_emitting_terminal_result(self):
        """A no-op Swarm must not close the stream before deterministic recovery runs."""
        from contextlib import contextmanager

        from strands.multiagent.base import Status

        class _Orchestrator:
            def __init__(self, events):
                self.events = events

            async def stream_async(self, _prompt):
                for event in self.events:
                    yield event

        @contextmanager
        def gateway_tools():
            yield {"trend": [], "enrichment": [], "email": []}

        swarm = _Orchestrator(
            [
                {"type": "multiagent_node_start", "node_id": "trend_researcher"},
                {"type": "multiagent_result", "result": SimpleNamespace(status=Status.COMPLETED)},
            ]
        )
        graph = _Orchestrator(
            [
                {"type": "multiagent_node_start", "node_id": "research"},
                {"type": "multiagent_result", "result": SimpleNamespace(status=Status.COMPLETED)},
            ]
        )

        async def collect_events():
            return [
                event
                async for event in self.ep._run_pipeline(
                    "find prospects",
                    "swarm",
                    session_manager=None,
                    include_scored_prospects=False,
                )
            ]

        with patch.object(self.ep, "_gateway_tools", gateway_tools):
            with patch.object(self.ep, "_build_orchestrator", return_value=swarm):
                with patch.object(self.ep, "_build_graph", return_value=graph):
                    events = asyncio.run(collect_events())

        assert [event["type"] for event in events] == [
            "multiagent_node_start",
            "swarm_recovery",
            "multiagent_node_start",
            "multiagent_result",
        ]
        assert events[1]["fallback_pattern"] == "graph"

    def test_swarm_midrun_failure_recovers_via_graph(self):
        """A Swarm node crash (for example, model throttling) falls back to Graph."""
        from contextlib import contextmanager

        from strands.multiagent.base import Status

        class _CrashingSwarm:
            async def stream_async(self, _prompt):
                yield {"type": "multiagent_node_start", "node_id": "trend_researcher"}
                raise RuntimeError("ModelThrottledException: ApplyGuardrail limit exceeded")

        class _Graph:
            async def stream_async(self, _prompt):
                yield {"type": "multiagent_node_start", "node_id": "research"}
                yield {"type": "multiagent_result", "result": SimpleNamespace(status=Status.COMPLETED)}

        @contextmanager
        def gateway_tools():
            yield {"trend": [], "enrichment": [], "email": []}

        async def collect_events():
            return [
                event
                async for event in self.ep._run_pipeline(
                    "find prospects",
                    "swarm",
                    session_manager=None,
                    include_scored_prospects=False,
                )
            ]

        with patch.object(self.ep, "_gateway_tools", gateway_tools):
            with patch.object(self.ep, "_build_orchestrator", return_value=_CrashingSwarm()):
                with patch.object(self.ep, "_build_graph", return_value=_Graph()):
                    events = asyncio.run(collect_events())

        types = [event["type"] for event in events]
        assert "swarm_recovery" in types
        assert types[-1] == "multiagent_result"
        recovery = next(event for event in events if event["type"] == "swarm_recovery")
        assert recovery["fallback_pattern"] == "graph"
        # A mid-run crash with no persisted scores reports the missing-handoff reason;
        # what matters is that the crash routes into Graph recovery rather than failing
        # the whole run. The explicit crash reason is used when scores were persisted.
        assert recovery["reason"]

    def test_swarm_crash_after_clean_metrics_reports_execution_error(self):
        """A crash when output invariants already hold uses the explicit crash reason."""
        from contextlib import contextmanager

        from strands.multiagent.base import Status

        class _CrashingSwarm:
            async def stream_async(self, _prompt):
                yield {"type": "multiagent_node_start", "node_id": "email_generator"}
                raise RuntimeError("ModelThrottledException: ApplyGuardrail limit exceeded")

        class _Graph:
            async def stream_async(self, _prompt):
                yield {"type": "multiagent_result", "result": SimpleNamespace(status=Status.COMPLETED)}

        @contextmanager
        def gateway_tools():
            yield {"trend": [], "enrichment": [], "email": []}

        async def collect_events():
            return [
                event
                async for event in self.ep._run_pipeline(
                    "find prospects",
                    "swarm",
                    session_manager=None,
                    include_scored_prospects=False,
                )
            ]

        # Force the metrics-based reason to be clean so the explicit crash reason is used.
        with patch.object(self.ep, "_gateway_tools", gateway_tools):
            with patch.object(self.ep, "_build_orchestrator", return_value=_CrashingSwarm()):
                with patch.object(self.ep, "_build_graph", return_value=_Graph()):
                    with patch.object(self.ep, "_swarm_recovery_reason", return_value=None):
                        events = asyncio.run(collect_events())

        recovery = next(event for event in events if event["type"] == "swarm_recovery")
        assert "failed before completion" in recovery["reason"]
        assert events[-1]["type"] == "multiagent_result"

    def test_graph_midrun_failure_is_not_swallowed(self):
        """The Graph pattern has no recovery, so a mid-run failure must propagate."""
        from contextlib import contextmanager

        class _CrashingGraph:
            async def stream_async(self, _prompt):
                yield {"type": "multiagent_node_start", "node_id": "research"}
                raise RuntimeError("graph node failed")

        @contextmanager
        def gateway_tools():
            yield {"trend": [], "enrichment": [], "email": []}

        async def collect_events():
            return [
                event
                async for event in self.ep._run_pipeline(
                    "find prospects",
                    "graph",
                    session_manager=None,
                    include_scored_prospects=False,
                )
            ]

        with patch.object(self.ep, "_gateway_tools", gateway_tools):
            with patch.object(self.ep, "_build_orchestrator", return_value=_CrashingGraph()):
                with pytest.raises(RuntimeError, match="graph node failed"):
                    asyncio.run(collect_events())

    def test_swarm_valid_empty_analysis_emits_its_terminal_result(self):
        """A completed, explicitly empty analysis is not retried as a missing handoff."""
        from contextlib import contextmanager

        from strands.multiagent.base import Status

        from social_intelligence.tools.dynamodb_tool import RunOutputMetrics

        class _Orchestrator:
            async def stream_async(self, _prompt):
                yield {"type": "multiagent_result", "result": SimpleNamespace(status=Status.COMPLETED)}

        @contextmanager
        def gateway_tools():
            yield {"trend": [], "enrichment": [], "email": []}

        async def collect_events():
            return [
                event
                async for event in self.ep._run_pipeline(
                    "find prospects",
                    "swarm",
                    session_manager=None,
                    include_scored_prospects=False,
                )
            ]

        completed_empty = RunOutputMetrics(
            score_persistence_calls=1,
            scores_requested=0,
            scores_persisted=0,
            email_eligible_scores=0,
            leads_stored=0,
        )
        with patch.object(self.ep, "_gateway_tools", gateway_tools):
            with patch.object(self.ep, "_build_orchestrator", return_value=_Orchestrator()):
                with patch.object(self.ep, "get_run_output_metrics", return_value=completed_empty):
                    events = asyncio.run(collect_events())

        assert [event["type"] for event in events] == ["multiagent_result"]

    def test_background_swarm_recovery_persists_actual_execution_path(self):
        """A recovered run must not be indistinguishable from native Swarm success."""

        async def recovered_events():
            yield {"type": "swarm_recovery", "fallback_pattern": "graph"}

        with patch.object(self.ep, "_run_pipeline", return_value=recovered_events()):
            with patch.object(self.ep, "persist_run_status") as persist_status:
                with patch.object(self.ep.app, "complete_async_task"):
                    self.ep._run_pipeline_in_background("find prospects", "swarm", None, task_id=7)

        persist_status.assert_called_once_with(True, execution_path="graph_recovery")


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
        assert self.ep._resolve_actor_id("analyst-7", "session-1") == "analyst-7"

    def test_resolve_actor_id_defaults_to_session_scoped_anonymous_identity(self):
        actor = self.ep._resolve_actor_id("", "session-1")
        assert actor.startswith("anonymous_")
        assert actor == self.ep._resolve_actor_id("", "session-1")
        assert actor != self.ep._resolve_actor_id("", "session-2")

    def test_build_session_manager_none_when_disabled(self):
        assert self.ep._build_session_manager("s1", "a1") is None

    def test_build_orchestrator_dispatches_by_pattern(self):
        from strands.multiagent import Swarm

        tools = {"trend": [], "enrichment": [], "email": []}
        assert type(self.ep._build_orchestrator("graph", tools, None)).__name__ == "Graph"
        assert isinstance(self.ep._build_orchestrator("swarm", tools, None), Swarm)


class TestDynamoDbRunContext:
    """Each concurrent AgentCore request must retain its own tool-run metadata."""

    def test_parallel_invocations_keep_session_and_isolation_separate(self):
        from social_intelligence.tools import dynamodb_tool

        async def invoke(session_id: str, isolate: bool):
            dynamodb_tool.reset_lead_counter()
            dynamodb_tool.set_run_session_id(session_id)
            dynamodb_tool.set_run_isolation(isolate)
            await asyncio.sleep(0)
            return dynamodb_tool.get_run_session_id(), dynamodb_tool.run_isolation_enabled()

        async def run_parallel():
            return await asyncio.gather(
                invoke("run-a", True),
                invoke("run-b", False),
            )

        first, second = asyncio.run(run_parallel())

        assert first == ("run-a", True)
        assert second == ("run-b", False)


# ---------------------------------------------------------------------------
# 6. email_renderer.render_email_html
# ---------------------------------------------------------------------------


class TestEmailRenderer:
    """HTML rendering, XSS escaping, and compliance footer."""

    def test_tool_schema_accepts_personalization_token_arrays(self):
        from social_intelligence.tools.dynamodb_tool import store_lead
        from social_intelligence.tools.email_renderer import render_email_html_tool

        def has_array_branch(schema: dict) -> bool:
            return schema.get("type") == "array" or any(
                branch.get("type") == "array" for branch in schema.get("anyOf", [])
            )

        store_schema = store_lead.tool_spec["inputSchema"]["json"]["properties"]["top_trends"]
        render_schema = render_email_html_tool.tool_spec["inputSchema"]["json"]["properties"]["personalization_tokens"]

        assert has_array_branch(store_schema)
        assert has_array_branch(render_schema)

    def test_tool_renders_personalization_token_arrays(self):
        from social_intelligence.tools.email_renderer import render_email_html_tool

        result = json.loads(
            render_email_html_tool.__wrapped__(
                prospect_id="p-token",
                subject="A specific subject",
                body="A concise body.",
                personalization_tokens=["HN score: 120", "GitHub stars: 500"],
            )
        )

        assert "HN score: 120" in result["html"]
        assert "GitHub stars: 500" in result["html"]

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
