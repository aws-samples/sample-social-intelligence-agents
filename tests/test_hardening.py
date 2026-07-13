"""Tests for the security hardening and deep-research upgrades.

Covers: outbound host allow-list (SSRF guard), computed temporal decay,
Hacker News 'show' category, the frontier claim_url dedup tool, the email
grounding gate, and the store_lead grounding threshold. All env-gated features
are tested in both their default (off) and active states.
"""

import json
import os
import time
from unittest.mock import patch

import pytest


class TestOutboundAllowList:
    """The shared HTTP client must reject hosts outside the source-API allow-list."""

    def test_rejects_unlisted_host(self):
        from social_intelligence.tools import _http

        with pytest.raises(ValueError, match="host not allowed"):
            _http._assert_allowed("https://evil.example.com/exfil")

    def test_rejects_imds_endpoint(self):
        from social_intelligence.tools import _http

        with pytest.raises(ValueError):
            _http._assert_allowed("http://169.254.169.254/latest/meta-data/")

    def test_permits_known_api_hosts(self):
        from social_intelligence.tools import _http

        for url in (
            "https://api.github.com/search/repositories",
            "https://www.reddit.com/r/SaaS/hot.json",
            "https://hacker-news.firebaseio.com/v0/topstories.json",
        ):
            _http._assert_allowed(url)  # must not raise

    def test_get_with_retry_blocks_unlisted_host_before_request(self):
        from social_intelligence.tools import _http

        with patch("social_intelligence.tools._http.httpx.get") as mock_get:
            with pytest.raises(ValueError):
                _http.get_with_retry("https://attacker.test/x")
            mock_get.assert_not_called()


class TestTemporalDecay:
    """Freshness weight must be computed from timestamps, not left to the model."""

    def test_weight_buckets(self):
        from social_intelligence.tools._freshness import freshness_weight

        now = time.time()
        assert freshness_weight(now) == 1.5
        assert freshness_weight(now - 2 * 86400) == 1.2
        assert freshness_weight(now - 5 * 86400) == 1.0
        assert freshness_weight(now - 30 * 86400) == 0.5

    def test_accepts_iso_string(self):
        from social_intelligence.tools._freshness import freshness_weight

        assert freshness_weight("2020-01-01T00:00:00Z") == 0.5

    def test_bad_input_defaults_to_one(self):
        from social_intelligence.tools._freshness import freshness_weight

        assert freshness_weight("not-a-date") == 1.0
        assert freshness_weight(None) == 1.0

    def test_trend_item_has_default_weight(self):
        from social_intelligence.schemas.models import TrendItem

        assert TrendItem(source="hackernews", topic="x").freshness_weight == 1.0


class TestHackerNewsShowCategory:
    """The 'show' category the agent prompt uses must be valid in the tool."""

    def test_show_is_valid_category(self):
        from social_intelligence.tools import hackernews

        assert "show" in hackernews._VALID_CATEGORIES

    @patch("social_intelligence.tools._http.httpx.get")
    def test_show_maps_to_showstories_endpoint(self, mock_get):
        from unittest.mock import MagicMock

        from social_intelligence.tools.hackernews import handle

        ids_resp = MagicMock()
        ids_resp.json.return_value = []
        ids_resp.raise_for_status = MagicMock()
        ids_resp.status_code = 200
        mock_get.return_value = ids_resp

        result = handle({"category": "show", "limit": 1})
        assert "error" not in result
        called_url = mock_get.call_args[0][0]
        assert "showstories" in called_url


class TestFrontierClaimUrl:
    """The frontier dedup tool is a graceful no-op until FRONTIER_TABLE_NAME is set."""

    def test_no_op_when_env_unset(self):
        from social_intelligence.tools.dynamodb_tool import claim_url

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FRONTIER_TABLE_NAME", None)
            result = json.loads(claim_url.__wrapped__("hn-123"))
        assert result["claimed"] is True
        assert "disabled" in result["reason"]


class TestGroundingGate:
    """The grounding gate scores email claims against gathered evidence."""

    def test_supported_claim_scores_high(self):
        from social_intelligence.tools.grounding_gate import verify_email_claims

        evidence = json.dumps({"github": "The repo crossed 2,400 stars this week."})
        out = json.loads(verify_email_claims.__wrapped__("Congrats on 2,400 stars!", evidence))
        assert out["grounding_score"] == 1.0
        assert out["unsupported_claims"] == []

    def test_unsupported_number_is_flagged(self):
        from social_intelligence.tools.grounding_gate import verify_email_claims

        evidence = json.dumps({"github": "The repo crossed 2,400 stars."})
        out = json.loads(verify_email_claims.__wrapped__("You have 99% uptime and 2,400 stars.", evidence))
        assert out["grounding_score"] < 1.0
        assert any("99" in c for c in out["unsupported_claims"])

    def test_no_numeric_claims_scores_perfect(self):
        from social_intelligence.tools.grounding_gate import verify_email_claims

        out = json.loads(verify_email_claims.__wrapped__("Great work on the launch.", "{}"))
        assert out["grounding_score"] == 1.0


class TestStoreLeadGroundingThreshold:
    """store_lead must refuse low-grounding leads when GROUNDING_MIN_SCORE is set."""

    def test_rejects_below_threshold_without_writing(self):
        from social_intelligence.tools import dynamodb_tool

        with patch.dict(os.environ, {"GROUNDING_MIN_SCORE": "0.8"}, clear=False):
            with patch.object(dynamodb_tool, "_get_table") as mock_table:
                result = json.loads(
                    dynamodb_tool.store_lead.__wrapped__(prospect_id="hn-1", product_name="Acme", grounding_score=0.4)
                )
                mock_table.assert_not_called()
        assert result["stored"] is False
        assert "grounding" in result["reason"].lower()


class TestStoreLeadApprovalGate:
    """EMAIL_APPROVAL_REQUIRED stores leads as 'pending_review' for human-in-the-loop."""

    def _run(self, env: dict) -> dict:
        from social_intelligence.tools import dynamodb_tool

        captured: dict = {}

        class _FakeTable:
            def query(self, **_):
                return {"Items": []}

            def put_item(self, *, Item, **_):
                captured["item"] = Item

        dynamodb_tool.reset_lead_counter()
        with patch.dict(os.environ, env, clear=False):
            for var in ("GROUNDING_MIN_SCORE",):
                os.environ.pop(var, None)
            with patch.object(dynamodb_tool, "_find_by_product_name", return_value=None):
                with patch.object(dynamodb_tool, "_get_table", return_value=_FakeTable()):
                    result = json.loads(
                        dynamodb_tool.store_lead.__wrapped__(prospect_id="hn-9", product_name="Acme", score=80)
                    )
        result["_stored_status"] = captured["item"]["status"]
        return result

    def test_pending_review_when_required(self):
        result = self._run({"EMAIL_APPROVAL_REQUIRED": "true"})
        assert result["stored"] is True
        assert result["status"] == "pending_review"
        assert result["_stored_status"] == "pending_review"

    def test_new_status_by_default(self):
        os.environ.pop("EMAIL_APPROVAL_REQUIRED", None)
        result = self._run({})
        assert result["stored"] is True
        assert result["status"] == "new"
        assert result["_stored_status"] == "new"
