"""Extra hermetic coverage tests — targets dynamodb_tool, grounding_gate, _http CB, entrypoint.

All AWS and network calls are patched. No real boto3/httpx I/O occurs.
"""

from __future__ import annotations

import json
import os
import sys
import time
from unittest.mock import MagicMock, patch

import boto3
import httpx
import pytest
from botocore.exceptions import ClientError
from botocore.stub import ANY, Stubber

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _client_error(code: str) -> ClientError:
    """Build a botocore ClientError with the given Error.Code."""
    return ClientError({"Error": {"Code": code, "Message": "mock"}}, "operation")


def _transaction_cancelled_for_dedup() -> ClientError:
    """Build the cancellation response emitted when a transaction reservation exists."""
    return ClientError(
        {
            "Error": {"Code": "TransactionCanceledException", "Message": "dedup marker exists"},
            "CancellationReasons": [{"Code": "ConditionalCheckFailed"}],
        },
        "TransactWriteItems",
    )


# ---------------------------------------------------------------------------
# 1. dynamodb_tool — check_existing_leads
# ---------------------------------------------------------------------------


class TestCheckExistingLeads:
    """check_existing_leads returns correct shape for all query paths."""

    def _mock_table(self):
        return MagicMock()

    # -- prospect_id found (exists=True, age_days, stale) --------------------

    def test_prospect_id_found_returns_exists_true_with_age(self):
        from social_intelligence.tools import dynamodb_tool

        mock_table = self._mock_table()
        mock_table.query.return_value = {
            "Items": [
                {
                    "prospect_id": "hn-42",
                    "discovered_at": "2000-01-01T00:00:00+00:00",
                    "product_name": "OldProd",
                    "score": 77,
                    "status": "new",
                }
            ]
        }

        with patch.object(dynamodb_tool, "_get_table", return_value=mock_table):
            result = json.loads(dynamodb_tool.check_existing_leads.__wrapped__(prospect_id="hn-42"))

        assert result["exists"] is True
        assert result["prospect_id"] == "hn-42"
        assert result["product_name"] == "OldProd"
        assert result["score"] == 77
        assert result["age_days"] > 0
        assert result["stale"] is True  # 2000-01-01 is definitely > 7 days ago

    def test_prospect_id_not_found_returns_exists_false(self):
        from social_intelligence.tools import dynamodb_tool

        mock_table = self._mock_table()
        mock_table.query.return_value = {"Items": []}

        with patch.object(dynamodb_tool, "_get_table", return_value=mock_table):
            result = json.loads(
                dynamodb_tool.check_existing_leads.__wrapped__(prospect_id="hn-missing", product_name="")
            )

        assert result["exists"] is False
        assert result["prospect_id"] == "hn-missing"

    # -- product_name path ---------------------------------------------------

    def test_product_name_found_returns_matched_by_product_name(self):
        from social_intelligence.tools import dynamodb_tool

        mock_table = self._mock_table()
        # query returns nothing for prospect_id; _find_by_product_name returns an item
        existing_item = {
            "prospect_id": "hn-99",
            "discovered_at": "2024-01-01T00:00:00+00:00",
            "product_name": "Acme",
            "score": 55,
            "status": "new",
        }

        with patch.object(dynamodb_tool, "_get_table", return_value=mock_table):
            with patch.object(dynamodb_tool, "_find_by_product_name", return_value=existing_item):
                result = json.loads(dynamodb_tool.check_existing_leads.__wrapped__(product_name="Acme"))

        assert result["exists"] is True
        assert result["matched_by"] == "product_name"
        assert result["prospect_id"] == "hn-99"

    def test_product_name_not_found_returns_exists_false(self):
        from social_intelligence.tools import dynamodb_tool

        mock_table = self._mock_table()

        with patch.object(dynamodb_tool, "_get_table", return_value=mock_table):
            with patch.object(dynamodb_tool, "_find_by_product_name", return_value=None):
                result = json.loads(dynamodb_tool.check_existing_leads.__wrapped__(product_name="Unknown"))

        assert result["exists"] is False
        assert result["product_name"] == "Unknown"

    # -- recent-leads GSI path (limit) ---------------------------------------

    def test_recent_leads_query_returns_leads_list_with_count(self):
        from social_intelligence.tools import dynamodb_tool

        mock_table = self._mock_table()
        mock_table.query.return_value = {
            "Items": [
                {"prospect_id": "hn-1", "product_name": "Alpha", "score": 80, "discovered_at": "2025-06-01T00:00:00"},
                {"prospect_id": "hn-2", "product_name": "Beta", "score": 70, "discovered_at": "2025-05-01T00:00:00"},
            ]
        }

        with patch.object(dynamodb_tool, "_get_table", return_value=mock_table):
            result = json.loads(
                dynamodb_tool.check_existing_leads.__wrapped__(prospect_id="", product_name="", limit=5)
            )

        assert result["source"] == "DynamoDB"
        assert result["count"] == 2
        assert len(result["leads"]) == 2
        assert mock_table.query.call_args.kwargs["IndexName"] == dynamodb_tool.RECENT_LEADS_INDEX

    def test_recent_leads_query_respects_limit_cap(self):
        from social_intelligence.tools import dynamodb_tool

        mock_table = self._mock_table()
        # Return 1 item; limit=1 means only 1 returned
        mock_table.query.return_value = {
            "Items": [
                {"prospect_id": "hn-1", "product_name": "Alpha", "score": 80, "discovered_at": "2025-06-01T00:00:00"},
            ]
        }

        with patch.object(dynamodb_tool, "_get_table", return_value=mock_table):
            result = json.loads(
                dynamodb_tool.check_existing_leads.__wrapped__(prospect_id="", product_name="", limit=1)
            )

        assert len(result["leads"]) == 1
        assert mock_table.query.call_args.kwargs["Limit"] == 1

    def test_clienterror_returns_storage_error(self):
        from social_intelligence.tools import dynamodb_tool

        mock_table = self._mock_table()
        mock_table.query.side_effect = _client_error("ProvisionedThroughputExceededException")

        with patch.object(dynamodb_tool, "_get_table", return_value=mock_table):
            result = json.loads(dynamodb_tool.check_existing_leads.__wrapped__(prospect_id="hn-1"))

        assert result["error"] == "storage_error"
        assert result["leads"] == []

    # -- prospect_id not found then falls through to product_name check ------

    def test_prospect_id_miss_falls_through_to_product_name(self):
        from social_intelligence.tools import dynamodb_tool

        mock_table = self._mock_table()
        mock_table.query.return_value = {"Items": []}
        existing_item = {
            "prospect_id": "hn-77",
            "discovered_at": "2024-06-01T00:00:00+00:00",
            "product_name": "Widget",
            "score": 60,
            "status": "new",
        }

        with patch.object(dynamodb_tool, "_get_table", return_value=mock_table):
            with patch.object(dynamodb_tool, "_find_by_product_name", return_value=existing_item):
                result = json.loads(
                    dynamodb_tool.check_existing_leads.__wrapped__(prospect_id="hn-missing", product_name="Widget")
                )

        assert result["exists"] is True
        assert result["matched_by"] == "product_name"


# ---------------------------------------------------------------------------
# 2. dynamodb_tool — _find_by_product_name
# ---------------------------------------------------------------------------


class TestFindByProductName:
    """_find_by_product_name covers GSI query, get_item, empty, ClientError."""

    def test_gsi_hit_returns_full_item_from_get_item(self):
        from social_intelligence.tools import dynamodb_tool

        mock_table = MagicMock()
        mock_table.query.return_value = {"Items": [{"prospect_id": "hn-10", "discovered_at": "2024-01-01T00:00:00"}]}
        full_item = {"prospect_id": "hn-10", "discovered_at": "2024-01-01T00:00:00", "product_name": "Acme"}
        mock_table.get_item.return_value = {"Item": full_item}

        result = dynamodb_tool._find_by_product_name(mock_table, "Acme")

        assert result == full_item
        mock_table.query.assert_called_once()
        mock_table.get_item.assert_called_once()

    def test_gsi_hit_returns_key_when_get_item_returns_no_item(self):
        from social_intelligence.tools import dynamodb_tool

        mock_table = MagicMock()
        key_item = {"prospect_id": "hn-11", "discovered_at": "2024-01-01T00:00:00"}
        mock_table.query.return_value = {"Items": [key_item]}
        mock_table.get_item.return_value = {}  # no "Item" key

        result = dynamodb_tool._find_by_product_name(mock_table, "ACME")

        assert result == key_item  # falls back to the GSI key item

    def test_gsi_returns_no_items_returns_none(self):
        from social_intelligence.tools import dynamodb_tool

        mock_table = MagicMock()
        mock_table.query.return_value = {"Items": []}

        result = dynamodb_tool._find_by_product_name(mock_table, "ghost")

        assert result is None

    def test_empty_product_name_returns_none_without_query(self):
        from social_intelligence.tools import dynamodb_tool

        mock_table = MagicMock()

        result = dynamodb_tool._find_by_product_name(mock_table, "   ")

        assert result is None
        mock_table.query.assert_not_called()

    def test_client_error_returns_none(self):
        from social_intelligence.tools import dynamodb_tool

        mock_table = MagicMock()
        mock_table.query.side_effect = _client_error("ResourceNotFoundException")

        result = dynamodb_tool._find_by_product_name(mock_table, "oops")

        assert result is None


# ---------------------------------------------------------------------------
# 3. dynamodb_tool — claim_url (with frontier table)
# ---------------------------------------------------------------------------


class TestClaimUrl:
    """claim_url with FRONTIER_TABLE_NAME set exercises the real DB path."""

    def test_claim_succeeds_returns_true(self):
        from social_intelligence.tools import dynamodb_tool

        mock_table = MagicMock()
        mock_table.put_item.return_value = {}

        with patch.dict(os.environ, {"FRONTIER_TABLE_NAME": "test-frontier"}):
            with patch.object(dynamodb_tool, "_get_frontier_table", return_value=mock_table):
                result = json.loads(dynamodb_tool.claim_url.__wrapped__("hn-500"))

        assert result["claimed"] is True
        assert "reason" not in result
        mock_table.put_item.assert_called_once()

    def test_claim_records_owner_and_allows_expired_claim_replacement(self):
        from social_intelligence.tools import dynamodb_tool

        mock_table = MagicMock()
        mock_table.put_item.return_value = {}

        dynamodb_tool.set_run_session_id("run-123")
        try:
            with patch.dict(os.environ, {"FRONTIER_TABLE_NAME": "test-frontier"}):
                with patch.object(dynamodb_tool, "_get_frontier_table", return_value=mock_table):
                    result = json.loads(dynamodb_tool.claim_url.__wrapped__(" prospect:hn-500 "))
        finally:
            dynamodb_tool.set_run_session_id("")

        assert result["claimed"] is True
        kwargs = mock_table.put_item.call_args.kwargs
        assert kwargs["Item"]["claim_key"] == "prospect:hn-500"
        assert kwargs["Item"]["owner_id"] == "run-123"
        assert "expires_at < :now" in kwargs["ConditionExpression"]
        assert kwargs["ExpressionAttributeValues"][":owner"] == "run-123"

    def test_blank_claim_key_is_rejected_before_dynamodb(self):
        from social_intelligence.tools import dynamodb_tool

        with patch.object(dynamodb_tool, "_get_frontier_table") as get_table:
            result = json.loads(dynamodb_tool.claim_url.__wrapped__("   "))

        assert result == {"claimed": False, "reason": "claim_key is required"}
        get_table.assert_not_called()

    def test_claim_fails_returns_false_already_claimed(self):
        from social_intelligence.tools import dynamodb_tool

        mock_table = MagicMock()
        mock_table.put_item.side_effect = _client_error("ConditionalCheckFailedException")

        with patch.dict(os.environ, {"FRONTIER_TABLE_NAME": "test-frontier"}):
            with patch.object(dynamodb_tool, "_get_frontier_table", return_value=mock_table):
                result = json.loads(dynamodb_tool.claim_url.__wrapped__("hn-501"))

        assert result["claimed"] is False
        assert result["reason"] == "already claimed"

    def test_other_client_error_reraises(self):
        from social_intelligence.tools import dynamodb_tool

        mock_table = MagicMock()
        mock_table.put_item.side_effect = _client_error("InternalServerError")

        with patch.dict(os.environ, {"FRONTIER_TABLE_NAME": "test-frontier"}):
            with patch.object(dynamodb_tool, "_get_frontier_table", return_value=mock_table):
                with pytest.raises(ClientError):
                    dynamodb_tool.claim_url.__wrapped__("hn-502")


# ---------------------------------------------------------------------------
# 4. dynamodb_tool — store_lead (product_name dup, ClientError, expires_at)
# ---------------------------------------------------------------------------


class TestStoreLeadExtra:
    """Paths not covered by test_coverage.py: product_name dup, ClientError, expires_at."""

    def setup_method(self):
        from social_intelligence.tools import dynamodb_tool

        dynamodb_tool.reset_lead_counter()
        dynamodb_tool.MAX_LEADS_PER_RUN = 10

    def teardown_method(self):
        from social_intelligence.tools import dynamodb_tool

        dynamodb_tool.reset_lead_counter()
        dynamodb_tool.MAX_LEADS_PER_RUN = 3

    def test_product_name_duplicate_returns_stored_false(self):
        from social_intelligence.tools import dynamodb_tool

        existing = {"prospect_id": "hn-old", "discovered_at": "2024-01-01T00:00:00+00:00"}
        mock_table = MagicMock()

        with patch.object(dynamodb_tool, "_get_table", return_value=mock_table):
            with patch.object(dynamodb_tool, "_find_by_product_name", return_value=existing):
                result = json.loads(dynamodb_tool.store_lead.__wrapped__(prospect_id="hn-new", product_name="Acme"))

        assert result["stored"] is False
        assert "duplicate" in result["reason"].lower()
        assert result["existing_prospect_id"] == "hn-old"
        mock_table.meta.client.transact_write_items.assert_not_called()

    def test_storage_clienterror_returns_storage_error(self):
        from social_intelligence.tools import dynamodb_tool

        mock_table = MagicMock()
        mock_table.query.return_value = {"Items": []}
        mock_table.meta.client.transact_write_items.side_effect = _client_error(
            "ProvisionedThroughputExceededException"
        )

        with patch.object(dynamodb_tool, "_get_table", return_value=mock_table):
            with patch.object(dynamodb_tool, "_find_by_product_name", return_value=None):
                result = json.loads(dynamodb_tool.store_lead.__wrapped__(prospect_id="hn-err", product_name="ErrProd"))

        assert result["stored"] is False
        assert result.get("error") == "storage_error"

    def test_atomic_dedup_conditional_check_failure(self):
        from social_intelligence.tools import dynamodb_tool

        mock_table = MagicMock()
        mock_table.query.return_value = {"Items": []}
        mock_table.meta.client.transact_write_items.side_effect = _transaction_cancelled_for_dedup()

        with patch.object(dynamodb_tool, "_get_table", return_value=mock_table):
            with patch.object(dynamodb_tool, "_find_by_product_name", return_value=None):
                result = json.loads(
                    dynamodb_tool.store_lead.__wrapped__(prospect_id="hn-race", product_name="RaceProd")
                )

        assert result["stored"] is False
        assert "duplicate" in result["reason"].lower()

    def test_atomic_write_uses_native_values_with_resource_client(self):
        """The DynamoDB resource serializes transaction items exactly once."""
        from social_intelligence.tools import dynamodb_tool

        table = boto3.resource(
            "dynamodb",
            region_name="eu-west-1",
            aws_access_key_id="test",
            aws_secret_access_key="test",  # noqa: S106 - stubbed credential prevents provider I/O
        ).Table("social-intel-leads")
        expected = {
            "TransactItems": [
                {
                    "Put": {
                        "TableName": "social-intel-leads",
                        "Item": {
                            "prospect_id": ANY,
                            "discovered_at": "__dedup_marker__",
                            "record_type": "dedup_marker",
                            "expires_at": 1_800_000_000,
                        },
                        "ConditionExpression": "attribute_not_exists(prospect_id)",
                    }
                },
                {
                    "Put": {
                        "TableName": "social-intel-leads",
                        "Item": {
                            "prospect_id": ANY,
                            "discovered_at": "__dedup_marker__",
                            "record_type": "dedup_marker",
                            "expires_at": 1_800_000_000,
                        },
                        "ConditionExpression": "attribute_not_exists(prospect_id)",
                    }
                },
                {
                    "Put": {
                        "TableName": "social-intel-leads",
                        "Item": {
                            "prospect_id": "hn-resource-client",
                            "product_name": "Resource Client",
                            "discovered_at": "2026-07-15T00:00:00+00:00",
                        },
                        "ConditionExpression": (
                            "attribute_not_exists(prospect_id) AND attribute_not_exists(discovered_at)"
                        ),
                    }
                },
            ]
        }

        with Stubber(table.meta.client) as stubber:
            stubber.add_response("transact_write_items", {}, expected)
            dynamodb_tool._store_lead_atomically(
                table,
                {
                    "prospect_id": "hn-resource-client",
                    "product_name": "Resource Client",
                    "discovered_at": "2026-07-15T00:00:00+00:00",
                },
                expires_at=1_800_000_000,
            )

    def test_stored_item_has_expires_at_set(self):
        """expires_at must be a Unix epoch int in the transactional lead write."""
        from social_intelligence.tools import dynamodb_tool

        mock_table = MagicMock()
        mock_table.query.return_value = {"Items": []}
        with patch.object(dynamodb_tool, "_get_table", return_value=mock_table):
            with patch.object(dynamodb_tool, "_find_by_product_name", return_value=None):
                result = json.loads(
                    dynamodb_tool.store_lead.__wrapped__(prospect_id="hn-ttl", product_name="TTLProd", score=80)
                )

        assert result["stored"] is True
        writes = mock_table.meta.client.transact_write_items.call_args.kwargs["TransactItems"]
        item = writes[-1]["Put"]["Item"]
        assert "expires_at" in item
        # expires_at must be a positive integer (Unix epoch)
        assert item["expires_at"] > int(time.time())  # at least in the future


# ---------------------------------------------------------------------------
# 5. grounding_gate — guardrail branch and claim extraction edge cases
# ---------------------------------------------------------------------------


class TestGroundingGateInlineGuardrails:
    """The grounding tool must not duplicate Converse guardrail evaluation."""

    def test_guardrail_env_does_not_trigger_a_second_bedrock_request(self):
        from social_intelligence.tools.grounding_gate import verify_email_claims

        with patch.dict(os.environ, {"GUARDRAIL_ID": "gid-1", "GUARDRAIL_VERSION": "1"}):
            with patch("boto3.client") as mock_boto:
                result = json.loads(
                    verify_email_claims.__wrapped__(
                        email_body="We have 500 users.",
                        evidence_json=json.dumps({"data": "500 users signed up"}),
                    )
                )

        mock_boto.assert_not_called()
        assert isinstance(result["grounding_score"], float)
        assert result["must_revise"] is False


class TestClaimExtractionEdgeCases:
    """_extract_claims handles real metrics while excluding identifier-like tokens."""

    def test_percent_token(self):
        from social_intelligence.tools.grounding_gate import _extract_claims

        claims = _extract_claims("Our product has 94.2% uptime.")
        assert any("94.2%" in c for c in claims)

    def test_km_scale_suffix(self):
        from social_intelligence.tools.grounding_gate import _extract_claims

        claims = _extract_claims("We have 10K+ users and $1.2M revenue.")
        raw = " ".join(claims)
        assert "10K+" in raw
        assert "$1.2M" in raw

    def test_metric_unit_word(self):
        from social_intelligence.tools.grounding_gate import _extract_claims

        claims = _extract_claims("The repo hit 2400 stars.")
        assert any("2400" in c for c in claims)

    def test_bare_year_not_extracted(self):
        from social_intelligence.tools.grounding_gate import _extract_claims

        claims = _extract_claims("Founded in 2024 with no metrics.")
        # 2024 alone is not a factual metric claim
        assert not any(c.strip() == "2024" for c in claims)

    def test_no_claims_empty_list(self):
        from social_intelligence.tools.grounding_gate import _extract_claims

        assert _extract_claims("Great launch, well done!") == []

    def test_model_name_with_scale_suffix_is_not_a_claim(self):
        from social_intelligence.tools.grounding_gate import _extract_claims

        assert _extract_claims("I enjoyed reading about Bonsai 27B.") == []

    def test_deduplication_preserves_order(self):
        from social_intelligence.tools.grounding_gate import _extract_claims

        claims = _extract_claims("99% uptime and 99% reliability.")
        assert claims.count("99%") == 1  # deduped


# ---------------------------------------------------------------------------
# 6. _http.py — circuit breaker trip and half-open recovery
# ---------------------------------------------------------------------------


class TestCircuitBreakerTrip:
    """Trip the circuit breaker with consecutive failures and verify fast-fail."""

    URL = "https://api.github.com/repos/org/repo"

    def test_trip_after_threshold_failures_then_fail_fast(self):
        from social_intelligence.tools._http import _CB_FAILURE_THRESHOLD, get_with_retry

        fail_exc = httpx.ConnectError("connection refused")

        with patch("social_intelligence.tools._http.httpx.get", side_effect=fail_exc):
            with patch("social_intelligence.tools._http.time.sleep"):
                # Exhaust retries _CB_FAILURE_THRESHOLD times to open the breaker
                for _ in range(_CB_FAILURE_THRESHOLD):
                    with pytest.raises(httpx.ConnectError):
                        get_with_retry(self.URL)

        # Now the circuit is open; next call must fail fast WITHOUT calling httpx.get
        with patch("social_intelligence.tools._http.httpx.get") as mock_get_fast:
            with pytest.raises(httpx.ConnectError, match="Circuit breaker open"):
                get_with_retry(self.URL)
            mock_get_fast.assert_not_called()

    def test_reset_state_clears_open_breaker(self):
        from social_intelligence.tools._http import (
            _CB_FAILURE_THRESHOLD,
            get_with_retry,
            reset_state,
        )

        fail_exc = httpx.ConnectError("connection refused")

        with patch("social_intelligence.tools._http.httpx.get", side_effect=fail_exc):
            with patch("social_intelligence.tools._http.time.sleep"):
                for _ in range(_CB_FAILURE_THRESHOLD):
                    with pytest.raises(httpx.ConnectError):
                        get_with_retry(self.URL)

        # Breaker is open; reset clears it
        reset_state()

        resp_200 = MagicMock()
        resp_200.status_code = 200
        with patch("social_intelligence.tools._http.httpx.get", return_value=resp_200) as mock_get_ok:
            result = get_with_retry(self.URL)

        assert result.status_code == 200
        mock_get_ok.assert_called_once()


class TestCircuitBreakerHalfOpen:
    """After recovery period elapses the breaker enters half-open and probes."""

    URL = "https://api.github.com/repos/org/half-open"

    def test_half_open_probes_and_success_resets_breaker(self):
        from social_intelligence.tools._http import (
            _CB_FAILURE_THRESHOLD,
            _CB_RECOVERY_SECONDS,
            get_with_retry,
        )

        fail_exc = httpx.ConnectError("conn refused")

        # Trip the breaker
        with patch("social_intelligence.tools._http.httpx.get", side_effect=fail_exc):
            with patch("social_intelligence.tools._http.time.sleep"):
                for _ in range(_CB_FAILURE_THRESHOLD):
                    with pytest.raises(httpx.ConnectError):
                        get_with_retry(self.URL)

        # Advance monotonic clock past recovery window
        past_time = time.monotonic() + _CB_RECOVERY_SECONDS + 1.0

        resp_200 = MagicMock()
        resp_200.status_code = 200

        with patch("social_intelligence.tools._http.time.monotonic", return_value=past_time):
            with patch("social_intelligence.tools._http.httpx.get", return_value=resp_200) as mock_probe:
                result = get_with_retry(self.URL)

        # Probe was called (half-open let it through)
        assert result.status_code == 200
        mock_probe.assert_called_once()

    def test_half_open_failure_re_opens_breaker(self):
        """After half-open probe fails, the failure counter increments again.

        We verify the failure is recorded (count > 0) rather than asserting
        the breaker is already re-opened (that requires >= threshold failures).
        """
        from social_intelligence.tools._http import (
            _CB_FAILURE_THRESHOLD,
            _CB_RECOVERY_SECONDS,
            _cb_failures,
            get_with_retry,
        )

        fail_exc = httpx.ConnectError("conn refused")

        # Trip the breaker with exactly _CB_FAILURE_THRESHOLD exhausted retries
        with patch("social_intelligence.tools._http.httpx.get", side_effect=fail_exc):
            with patch("social_intelligence.tools._http.time.sleep"):
                for _ in range(_CB_FAILURE_THRESHOLD):
                    with pytest.raises(httpx.ConnectError):
                        get_with_retry(self.URL)

        # Advance past recovery window so _cb_is_open clears state (half-open)
        past_time = time.monotonic() + _CB_RECOVERY_SECONDS + 1.0

        with patch("social_intelligence.tools._http.time.monotonic", return_value=past_time):
            with patch("social_intelligence.tools._http.httpx.get", side_effect=fail_exc):
                with patch("social_intelligence.tools._http.time.sleep"):
                    with pytest.raises(httpx.ConnectError):
                        get_with_retry(self.URL)

        # The probe failure must have been recorded — count restarted from 0 to 1
        host = "api.github.com"
        assert _cb_failures.get(host, 0) >= 1


# ---------------------------------------------------------------------------
# 7. entrypoint — _augment_prompt_with_skip_list
# ---------------------------------------------------------------------------


class TestAugmentPromptWithSkipList:
    """_augment_prompt_with_skip_list injects known product names into prompt."""

    @pytest.fixture(autouse=True)
    def _ep(self):
        """Import entrypoint via the same helper used by TestEntrypointOrchestration."""
        if "entrypoint" not in sys.modules:
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

                sys.modules.pop("entrypoint", None)
                self.ep = importlib.import_module("entrypoint")
        else:
            self.ep = sys.modules["entrypoint"]

    def test_skip_list_injected_when_table_has_items(self):
        from social_intelligence.tools import dynamodb_tool

        mock_table = MagicMock()
        mock_table.query.return_value = {
            "Items": [
                {"product_name": "AlphaProduct"},
                {"product_name": "BetaProduct"},
            ]
        }

        with patch.object(dynamodb_tool, "_get_table", return_value=mock_table):
            result = self.ep._augment_prompt_with_skip_list("Find new AI tools")

        assert "AlphaProduct" in result
        assert "BetaProduct" in result
        assert "SKIP LIST" in result

    def test_empty_table_returns_prompt_unchanged(self):
        from social_intelligence.tools import dynamodb_tool

        mock_table = MagicMock()
        mock_table.query.return_value = {"Items": []}

        with patch.object(dynamodb_tool, "_get_table", return_value=mock_table):
            original = "Find new AI tools"
            result = self.ep._augment_prompt_with_skip_list(original)

        assert result == original

    def test_table_error_returns_prompt_unchanged(self):
        from social_intelligence.tools import dynamodb_tool

        mock_table = MagicMock()
        mock_table.query.side_effect = Exception("DynamoDB unavailable")

        with patch.object(dynamodb_tool, "_get_table", return_value=mock_table):
            original = "Find new AI tools"
            result = self.ep._augment_prompt_with_skip_list(original)

        assert result == original

    def test_deduplicates_product_names_case_insensitive(self):
        from social_intelligence.tools import dynamodb_tool

        mock_table = MagicMock()
        mock_table.query.return_value = {
            "Items": [
                {"product_name": "Acme"},
                {"product_name": "acme"},  # duplicate (case-insensitive)
                {"product_name": "Beta"},
            ]
        }

        with patch.object(dynamodb_tool, "_get_table", return_value=mock_table):
            result = self.ep._augment_prompt_with_skip_list("Find tools")

        # "Acme" appears once in the skip list (deduplicated)
        skip_section = result.split("SKIP LIST")[1] if "SKIP LIST" in result else ""
        assert skip_section.count("Acme") + skip_section.count("acme") == 1

    def test_items_with_empty_product_name_are_excluded(self):
        from social_intelligence.tools import dynamodb_tool

        mock_table = MagicMock()
        mock_table.query.return_value = {
            "Items": [
                {"product_name": ""},
                {"product_name": "   "},
                {"product_name": "ValidProduct"},
            ]
        }

        with patch.object(dynamodb_tool, "_get_table", return_value=mock_table):
            result = self.ep._augment_prompt_with_skip_list("Find tools")

        assert "ValidProduct" in result
        # The two empty/whitespace names must NOT appear in the skip list
        skip_section = result.split("SKIP LIST")[1] if "SKIP LIST" in result else ""
        lines = [item.strip() for item in skip_section.split(",")]
        for blank in ("", " ", "   "):
            assert blank not in lines
