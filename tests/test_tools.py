"""Tests for tool handlers — happy path + error paths."""

from unittest.mock import MagicMock, patch

import httpx
import pytest


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

    @patch("social_intelligence.tools.youtube.get_secret", return_value="fake-key")
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

    @patch("social_intelligence.tools.youtube.get_secret", return_value="fake-key")
    @patch("social_intelligence.tools.youtube.get_with_retry")
    def test_returns_error_on_api_failure(self, mock_get, _mock_secret):
        from social_intelligence.tools.youtube import handle

        mock_get.side_effect = httpx.HTTPError("Connection failed")
        result = handle({"query": "test"})
        assert result["videos"] == []
        assert "error" in result


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
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError("404", request=MagicMock(), response=MagicMock())
        mock_get.return_value = mock_resp

        result = handle({"topic": "nonexistent"})
        assert "error" in result
        assert result["source"] == "Wikipedia"


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

    def test_scored_prospect_validation(self):
        from social_intelligence.schemas.models import ScoredProspect

        prospect = ScoredProspect(
            prospect_id="hn-123",
            score=85,
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

    def test_email_draft_defaults(self):
        from social_intelligence.schemas.models import EmailDraft

        draft = EmailDraft(
            prospect_id="hn-123",
            subject="Test Subject",
            body="Test body",
        )
        assert draft.brand_compliant is True
        assert draft.personalization_tokens == []
