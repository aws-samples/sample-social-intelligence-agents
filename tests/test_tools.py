"""Tests for tool handlers — happy path + error paths."""

from unittest.mock import MagicMock, patch

import httpx
import pytest


def _score_breakdown_for(score: int) -> dict[str, int]:
    """Build a valid medium-ICP score breakdown for one bounded total."""
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


class TestRedditTool:
    """Tests for the Reddit search tool."""

    @patch("social_intelligence.tools.reddit.get_with_retry")
    def test_returns_posts_matching_keyword(self, mock_get):
        from social_intelligence.tools.reddit import handle

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": {
                "children": [
                    {
                        "data": {
                            "title": "Looking for AI code review tool",
                            "selftext": "Any recommendations?",
                            "permalink": "/r/SaaS/comments/abc/test",
                            "subreddit": "SaaS",
                            "score": 42,
                            "num_comments": 10,
                            "author": "testuser",
                            "created_utc": 1700000000,
                        }
                    }
                ]
            }
        }
        mock_get.return_value = mock_resp

        result = handle({"keyword": "AI", "subreddits": "SaaS", "limit": 5})

        assert result["count"] == 1
        assert result["source"] == "Reddit"
        assert result["posts"][0]["title"] == "Looking for AI code review tool"

    @patch("social_intelligence.tools.reddit.get_with_retry")
    def test_detects_recommendation_seeking_intent(self, mock_get):
        from social_intelligence.tools.reddit import handle

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": {
                "children": [
                    {
                        "data": {
                            "title": "Looking for a better monitoring tool",
                            "selftext": "",
                            "permalink": "/r/devtools/test",
                            "subreddit": "devtools",
                            "score": 10,
                            "num_comments": 5,
                            "author": "user1",
                            "created_utc": 1700000000,
                        }
                    }
                ]
            }
        }
        mock_get.return_value = mock_resp

        result = handle({"keyword": "", "subreddits": "devtools"})

        assert len(result["posts"]) == 1
        signals = result["posts"][0]["intent_signals"]
        assert any("recommendation_seeking" in s for s in signals)

    @patch("social_intelligence.tools.reddit.get_with_retry")
    def test_handles_api_failure_gracefully(self, mock_get):
        from social_intelligence.tools.reddit import handle

        mock_get.side_effect = httpx.HTTPError("Connection failed")
        result = handle({"keyword": "test", "subreddits": "SaaS"})

        assert result["count"] == 0
        assert result["posts"] == []

    def test_invalid_sort_defaults_to_hot(self):
        from social_intelligence.tools.reddit import handle

        with patch("social_intelligence.tools.reddit.get_with_retry") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"data": {"children": []}}
            mock_get.return_value = mock_resp

            handle({"sort": "INVALID"})
            # Should not raise, uses default "hot"


class TestIntentSignalDetection:
    """Tests for the Reddit intent signal detection."""

    def test_detects_competitor_frustration(self):
        from social_intelligence.tools.reddit import _detect_intent_signals

        signals = _detect_intent_signals("Frustrated with Datadog pricing", "too expensive")
        assert any("competitor_frustration" in s for s in signals)

    def test_detects_product_launch(self):
        from social_intelligence.tools.reddit import _detect_intent_signals

        signals = _detect_intent_signals("Just launched our new dev tool", "")
        assert any("product_launch" in s for s in signals)

    def test_detects_purchase_intent(self):
        from social_intelligence.tools.reddit import _detect_intent_signals

        signals = _detect_intent_signals("What's the ROI on observability tools?", "budget")
        assert any("purchase_intent" in s for s in signals)

    def test_no_signals_for_generic_post(self):
        from social_intelligence.tools.reddit import _detect_intent_signals

        signals = _detect_intent_signals("Nice weather today", "Great day outside")
        assert signals == []


class TestStackOverflowTool:
    """Tests for the Stack Overflow search tool."""

    @patch("social_intelligence.tools._http.httpx.get")
    def test_returns_questions(self, mock_get):
        from social_intelligence.tools.stackoverflow import handle

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "items": [
                {
                    "title": "How to use vector databases?",
                    "link": "https://stackoverflow.com/q/123",
                    "score": 15,
                    "answer_count": 3,
                    "view_count": 500,
                    "is_answered": True,
                    "tags": ["vector-database", "ai"],
                    "creation_date": 1700000000,
                }
            ],
            "has_more": False,
            "quota_remaining": 290,
        }
        mock_get.return_value = mock_resp

        result = handle({"query": "vector database", "limit": 5})

        assert result["count"] == 1
        assert result["source"] == "Stack Overflow"
        assert result["questions"][0]["title"] == "How to use vector databases?"

    @patch("social_intelligence.tools._http.httpx.get")
    def test_returns_error_on_api_failure(self, mock_get):
        from social_intelligence.tools.stackoverflow import handle

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError("500", request=MagicMock(), response=MagicMock())
        mock_get.return_value = mock_resp

        result = handle({"query": "test"})
        assert result["count"] == 0
        assert "error" in result
        assert result["source"] == "Stack Overflow"

    def test_invalid_sort_defaults_to_relevance(self):
        from social_intelligence.tools.stackoverflow import handle

        with patch("social_intelligence.tools._http.httpx.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"items": [], "has_more": False, "quota_remaining": 300}
            mock_get.return_value = mock_resp

            handle({"query": "test", "sort": "INVALID"})
            call_params = mock_get.call_args[1]["params"]
            assert call_params["sort"] == "relevance"


class TestHackerNewsTool:
    """Tests for the Hacker News tool."""

    @patch("social_intelligence.tools._http.httpx.get")
    def test_returns_stories(self, mock_get):
        from social_intelligence.tools.hackernews import handle

        story_resp = MagicMock()
        story_resp.json.return_value = [1001]
        story_resp.raise_for_status = MagicMock()

        item_resp = MagicMock()
        item_resp.status_code = 200
        item_resp.json.return_value = {
            "id": 1001,
            "type": "story",
            "title": "Show HN: AI Code Review Tool",
            "url": "https://example.com",
            "score": 200,
            "by": "maker",
            "descendants": 50,
            "time": 1700000000,
        }

        mock_get.side_effect = [story_resp, item_resp]

        result = handle({"category": "top", "limit": 1})

        assert result["count"] == 1
        assert result["source"] == "Hacker News"
        assert result["stories"][0]["score"] == 200

    def test_invalid_category_returns_error(self):
        from social_intelligence.tools.hackernews import handle

        result = handle({"category": "INVALID"})
        assert "error" in result
        assert result["count"] == 0
        assert result["source"] == "Hacker News"


class TestYouTubeTool:
    """Tests for the YouTube search tool."""

    @patch("social_intelligence.tools.youtube.get_secret", return_value="AIza" + "a" * 35)
    @patch("social_intelligence.tools.youtube.get_with_retry")
    def test_returns_videos(self, mock_get, _mock_secret):
        from social_intelligence.tools.youtube import handle

        search_resp = MagicMock()
        search_resp.json.return_value = {"items": [{"id": {"videoId": "abc123"}}]}
        search_resp.raise_for_status = MagicMock()

        stats_resp = MagicMock()
        stats_resp.json.return_value = {
            "items": [
                {
                    "id": "abc123",
                    "snippet": {"title": "AI Agent Tutorial", "channelTitle": "TechChan", "publishedAt": "2025-01-01"},
                    "statistics": {"viewCount": "5000", "likeCount": "200"},
                }
            ]
        }
        stats_resp.raise_for_status = MagicMock()

        mock_get.side_effect = [search_resp, stats_resp]

        result = handle({"query": "AI agents", "max_results": 1})
        assert len(result["videos"]) == 1
        assert result["videos"][0]["title"] == "AI Agent Tutorial"

    @patch("social_intelligence.tools.youtube.get_secret", return_value="AIza" + "a" * 35)
    @patch("social_intelligence.tools.youtube.get_with_retry")
    def test_returns_error_on_api_failure(self, mock_get, _mock_secret):
        from social_intelligence.tools.youtube import handle

        mock_get.side_effect = httpx.HTTPError("Connection failed")
        result = handle({"query": "test"})
        assert result["videos"] == []
        assert "error" in result

    @patch("social_intelligence.tools.youtube.get_secret", return_value="generated/bootstrap?value")
    @patch("social_intelligence.tools.youtube.get_with_retry")
    def test_invalid_bootstrap_key_skips_network_call(self, mock_get, _mock_secret):
        from social_intelligence.tools.youtube import handle

        result = handle({"query": "AI agents"})

        assert result["error"] == "not_configured"
        mock_get.assert_not_called()


class TestDevToTool:
    """Tests for the dev.to tool."""

    @patch("social_intelligence.tools.devto.get_with_retry")
    def test_returns_articles(self, mock_get):
        from social_intelligence.tools.devto import handle

        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {
                "title": "Building AI Agents",
                "url": "https://dev.to/test/building-ai-agents",
                "positive_reactions_count": 42,
                "comments_count": 5,
                "user": {"name": "DevAuthor"},
                "published_at": "2025-01-15",
                "tag_list": ["ai", "agents"],
                "reading_time_minutes": 8,
                "description": "A guide to building AI agents",
            }
        ]
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = handle({"tag": "ai", "limit": 5})
        assert result["count"] == 1
        assert result["source"] == "dev.to"
        assert result["articles"][0]["title"] == "Building AI Agents"

    @patch("social_intelligence.tools.devto.get_with_retry")
    def test_returns_error_on_api_failure(self, mock_get):
        from social_intelligence.tools.devto import handle

        mock_get.side_effect = httpx.HTTPError("Timeout")
        result = handle({"tag": "test"})
        assert result["count"] == 0
        assert "error" in result


class TestWikipediaTool:
    """Tests for the Wikipedia tool."""

    @patch("social_intelligence.tools._http.httpx.get")
    def test_returns_summary(self, mock_get):
        from social_intelligence.tools.wikipedia import handle

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "title": "Artificial intelligence",
            "extract": "AI is the simulation of human intelligence.",
            "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/AI"}},
            "description": "Branch of computer science",
        }
        mock_get.return_value = mock_resp

        result = handle({"topic": "Artificial intelligence"})
        assert result["source"] == "Wikipedia"
        assert result["title"] == "Artificial intelligence"
        assert "error" not in result

    @patch("social_intelligence.tools._http.httpx.get")
    def test_returns_error_on_api_failure(self, mock_get):
        from social_intelligence.tools.wikipedia import handle

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError("500", request=MagicMock(), response=MagicMock())
        mock_get.return_value = mock_resp

        result = handle({"topic": "nonexistent"})
        assert "error" in result
        assert result["source"] == "Wikipedia"

    @patch("social_intelligence.tools.wikipedia.get_with_retry")
    def test_not_found_returns_empty_result_without_error(self, mock_get):
        from social_intelligence.tools.wikipedia import handle

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404",
            request=httpx.Request("GET", "https://en.wikipedia.org/wiki/missing"),
            response=httpx.Response(404),
        )
        mock_get.return_value = mock_resp

        result = handle({"topic": "missing"})

        assert result["source"] == "Wikipedia"
        assert result["extract"] == ""
        assert "error" not in result


class TestGitHubTool:
    """Tests for the GitHub search tool."""

    @patch("social_intelligence.tools._http.httpx.get")
    def test_returns_repos(self, mock_get):
        from social_intelligence.tools.github import handle

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "items": [
                {
                    "full_name": "org/ai-tool",
                    "description": "An AI tool",
                    "stargazers_count": 500,
                    "forks_count": 50,
                    "language": "Python",
                    "html_url": "https://github.com/org/ai-tool",
                    "updated_at": "2025-01-15",
                    "topics": ["ai"],
                }
            ]
        }
        mock_get.return_value = mock_resp

        result = handle({"query": "ai tool", "limit": 5})
        assert result["count"] == 1
        assert result["source"] == "GitHub"
        assert result["repos"][0]["stars"] == 500

    @patch("social_intelligence.tools._http.httpx.get")
    def test_returns_error_on_api_failure(self, mock_get):
        from social_intelligence.tools.github import handle

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError("403", request=MagicMock(), response=MagicMock())
        mock_get.return_value = mock_resp

        result = handle({"query": "test"})
        assert result["count"] == 0
        assert "error" in result
        assert result["source"] == "GitHub"

    def test_invalid_sort_defaults_to_stars(self):
        from social_intelligence.tools.github import handle

        with patch("social_intelligence.tools._http.httpx.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"items": []}
            mock_get.return_value = mock_resp

            handle({"query": "test", "sort": "INVALID"})
            call_params = mock_get.call_args[1]["params"]
            assert call_params["sort"] == "stars"


class TestLobstersTool:
    """Tests for the Lobsters tool."""

    @patch("social_intelligence.tools.lobsters.get_with_retry")
    def test_returns_stories(self, mock_get):
        from social_intelligence.tools.lobsters import handle

        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {
                "title": "New Rust Feature",
                "url": "https://example.com/rust",
                "score": 30,
                "comment_count": 12,
                "submitter_user": {"username": "rustdev"},
                "tags": ["rust", "programming"],
                "created_at": "2025-01-10",
            }
        ]
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = handle({"limit": 5})
        assert result["count"] == 1
        assert result["source"] == "Lobste.rs"
        assert result["stories"][0]["title"] == "New Rust Feature"

    @patch("social_intelligence.tools.lobsters.get_with_retry")
    def test_returns_error_on_api_failure(self, mock_get):
        from social_intelligence.tools.lobsters import handle

        mock_get.side_effect = httpx.HTTPError("Connection refused")
        result = handle({"limit": 5})
        assert result["count"] == 0
        assert "error" in result
        assert result["source"] == "Lobste.rs"


class TestRegistryRoutes:
    """Tests for the tool registry."""

    def test_all_routes_are_callable(self):
        from social_intelligence.tools.registry import ROUTES

        for name, handler in ROUTES.items():
            assert callable(handler), f"Route '{name}' is not callable"

    def test_expected_routes_exist(self):
        from social_intelligence.tools.registry import ROUTES

        expected = [
            "hackernews",
            "youtube",
            "devto",
            "wikipedia",
            "github",
            "lobsters",
            "producthunt",
            "reddit",
            "stackoverflow",
        ]
        for route in expected:
            assert route in ROUTES, f"Missing route: {route}"

    def test_route_count(self):
        from social_intelligence.tools.registry import ROUTES

        assert len(ROUTES) == 9


class TestLambdaHandler:
    """Tests for the Lambda handler routing logic."""

    def test_schema_to_route_mapping_file_parseable(self):
        """Verify the handler file is valid Python and contains the expected mapping."""
        import ast
        import os

        handler_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "infra",
            "lambda",
            "handler.py",
        )
        with open(handler_path) as f:
            source = f.read()
        tree = ast.parse(source)

        # Find _SCHEMA_TO_ROUTE assignment and verify it has 11 entries
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "_SCHEMA_TO_ROUTE":
                        assert isinstance(node.value, ast.Dict)
                        assert len(node.value.keys) == 9
                        return

        pytest.fail("_SCHEMA_TO_ROUTE not found in handler.py")


class TestPydanticModels:
    """Tests for Pydantic data models."""

    def test_trend_item_defaults(self):
        from social_intelligence.schemas.models import TrendItem

        item = TrendItem(source="hackernews", topic="Test")
        assert item.engagement == 0
        assert item.intent_signals == []
        assert item.timestamp == ""

    def test_prospect_contract_retains_evidence_across_stages(self):
        from social_intelligence.schemas.models import (
            EnrichmentData,
            EvidenceItem,
            ProspectEnrichment,
            ProspectProfile,
            ScoredProspect,
            TrendData,
            TrendItem,
        )

        signal = TrendItem(
            source="hackernews",
            topic="Launch",
            url="https://news.ycombinator.com/item?id=1",
            fact="The launch reached 2400 points.",
            metric_name="points",
            metric_value="2400",
        )
        trend = TrendData(
            prospects=[
                ProspectProfile(
                    prospect_id="hn:1",
                    product_name="Acme",
                    source_url=signal.url,
                    trend_signals=[signal],
                )
            ]
        )
        enrichment = EnrichmentData(
            prospects=[
                ProspectEnrichment(
                    prospect_id="hn:1",
                    product_name="Acme",
                    evidence=[
                        EvidenceItem(
                            source="github",
                            url="https://github.com/acme/project",
                            fact="The repository has 1200 stars.",
                            metric_name="stars",
                            metric_value="1200",
                        )
                    ],
                )
            ]
        )
        scored = ScoredProspect(
            prospect_id="hn:1",
            product_name="Acme",
            source_url=trend.prospects[0].source_url,
            score=85,
            score_breakdown=_score_breakdown_for(85),
            confidence=0.9,
            reasoning="Strong, corroborated launch and open-source signals.",
            evidence=enrichment.prospects[0].evidence,
        )

        assert scored.evidence[0].metric_value == "1200"
        assert scored.source_url == "https://news.ycombinator.com/item?id=1"

    def test_scored_prospect_validation(self):
        from social_intelligence.schemas.models import ScoredProspect

        prospect = ScoredProspect(
            prospect_id="hn-123",
            score=85,
            score_breakdown=_score_breakdown_for(85),
            confidence=0.9,
            reasoning="Strong multi-signal confirmation",
        )
        assert prospect.score == 85
        assert prospect.icp_fit == "medium"
        assert prospect.data_quality == "medium"

    def test_scored_prospect_score_bounds(self):
        from pydantic import ValidationError

        from social_intelligence.schemas.models import ScoredProspect

        with pytest.raises(ValidationError):
            ScoredProspect(prospect_id="test", score=150, confidence=0.5, reasoning="Invalid")

    def test_scored_prospect_rejects_mismatched_score_breakdown(self):
        """The model cannot persist a score total that disagrees with its components."""
        from pydantic import ValidationError

        from social_intelligence.schemas.models import ScoredProspect

        with pytest.raises(ValidationError, match="score_breakdown total"):
            ScoredProspect(
                prospect_id="test",
                score=84,
                score_breakdown=_score_breakdown_for(85),
                confidence=0.5,
                reasoning="A deliberately inconsistent total.",
            )

    def test_persisted_score_canonicalizes_instead_of_rejecting(self):
        """The swarm persistence contract recomputes the score from the breakdown.

        Unlike the Graph ScoredProspect path, the Swarm analyst has no structured-output
        retry, so a drifted total or a mismatched icp_adjustment is normalized from the
        bounded components rather than rejected.
        """
        from social_intelligence.schemas.models import PersistedScore

        drifted = PersistedScore(
            prospect_id="hn-123",
            score=84,  # breakdown sums to 83
            score_breakdown=_score_breakdown_for(83),
        )
        assert drifted.score == 83

        strong = PersistedScore(
            prospect_id="hn-123",
            score=83,  # analyst omitted the strong-ICP +10
            icp_fit="strong",
            score_breakdown=_score_breakdown_for(83),
        )
        assert strong.score == 93
        assert strong.score_breakdown.icp_adjustment == 10

    def test_scored_prospect_enforces_compact_structured_output(self):
        from pydantic import ValidationError

        from social_intelligence.schemas.models import EvidenceItem, ScoredProspect

        with pytest.raises(ValidationError):
            ScoredProspect(prospect_id="test", score=50, confidence=0.5, reasoning="x" * 481)
        with pytest.raises(ValidationError):
            ScoredProspect(
                prospect_id="test",
                score=50,
                score_breakdown=_score_breakdown_for(50),
                confidence=0.5,
                reasoning="Valid concise rationale.",
                evidence=[EvidenceItem(source="hackernews")] * 5,
            )

    def test_scoring_enum_fields_reject_unknown_values(self):
        from pydantic import ValidationError

        from social_intelligence.schemas.models import ProspectProfile, ScoredProspect

        with pytest.raises(ValidationError):
            ProspectProfile(prospect_id="hn-1", product_name="Acme", signal_strength="very strong")
        with pytest.raises(ValidationError):
            ScoredProspect(
                prospect_id="hn-1",
                score=75,
                score_breakdown=_score_breakdown_for(75),
                confidence=0.8,
                reasoning="Valid evidence, invalid metadata.",
                icp_fit="excellent",
            )
        with pytest.raises(ValidationError):
            ScoredProspect(
                prospect_id="hn-1",
                score=75,
                score_breakdown=_score_breakdown_for(75),
                confidence=0.8,
                reasoning="Valid evidence, invalid metadata.",
                data_quality="verified",
            )
        with pytest.raises(ValidationError):
            ScoredProspect(
                prospect_id="hn-1",
                score=75,
                score_breakdown=_score_breakdown_for(75),
                confidence=0.8,
                reasoning="Valid evidence, invalid metadata.",
                signal_strength="high",
            )

    def test_email_draft_defaults(self):
        from social_intelligence.schemas.models import EmailDraft

        draft = EmailDraft(
            prospect_id="hn-123",
            subject="Test Subject",
            body="Test body",
        )
        assert draft.brand_compliant is True
        assert draft.personalization_tokens == []
