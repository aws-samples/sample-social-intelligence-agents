"""Tests for the Product Hunt tool handler."""

from unittest.mock import MagicMock, patch

import httpx


class TestProductHuntTool:
    """Tests for the Product Hunt GraphQL tool."""

    @patch("social_intelligence.tools.producthunt.get_secret", return_value="fake-ph-token")
    @patch("social_intelligence.tools.producthunt.post_with_retry")
    def test_happy_path_returns_posts(self, mock_post, _mock_secret):
        from social_intelligence.tools.producthunt import handle

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": {
                "posts": {
                    "edges": [
                        {
                            "node": {
                                "id": "ph-001",
                                "name": "DevTool Pro",
                                "tagline": "The best dev tool",
                                "votesCount": 350,
                                "commentsCount": 42,
                                "url": "https://www.producthunt.com/posts/devtool-pro",
                                "website": "https://devtoolpro.io",
                                "createdAt": "2025-01-15T10:00:00Z",
                                "featuredAt": "2025-01-15T12:00:00Z",
                                "topics": {"edges": [{"node": {"name": "Developer Tools", "slug": "developer-tools"}}]},
                                "makers": [{"name": "Alice Dev", "username": "alicedev"}],
                            }
                        }
                    ]
                }
            }
        }
        mock_post.return_value = mock_resp

        result = handle({"topic": "developer-tools", "limit": 5})

        assert result["count"] == 1
        assert result["source"] == "Product Hunt"
        assert result["posts"][0]["name"] == "DevTool Pro"
        assert result["posts"][0]["votes"] == 350
        assert result["posts"][0]["comments"] == 42
        assert result["posts"][0]["url"] == "https://www.producthunt.com/posts/devtool-pro"
        assert result["posts"][0]["topics"] == ["Developer Tools"]
        assert result["posts"][0]["makers"] == ["Alice Dev"]
        assert result["posts"][0]["source"] == "Product Hunt"

    @patch("social_intelligence.tools.producthunt.get_secret", return_value="fake-ph-token")
    @patch("social_intelligence.tools.producthunt.post_with_retry")
    def test_returns_multiple_posts(self, mock_post, _mock_secret):
        from social_intelligence.tools.producthunt import handle

        def _node(i):
            return {
                "node": {
                    "id": f"ph-{i:03d}",
                    "name": f"Product {i}",
                    "tagline": f"Tagline {i}",
                    "votesCount": i * 10,
                    "commentsCount": i,
                    "url": f"https://producthunt.com/posts/product-{i}",
                    "website": f"https://product{i}.io",
                    "createdAt": "2025-02-01T00:00:00Z",
                    "featuredAt": None,
                    "topics": {"edges": []},
                    "makers": [],
                }
            }

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {"posts": {"edges": [_node(1), _node(2), _node(3)]}}}
        mock_post.return_value = mock_resp

        result = handle({"topic": "saas", "limit": 3})

        assert result["count"] == 3
        assert result["source"] == "Product Hunt"
        assert len(result["posts"]) == 3

    @patch("social_intelligence.tools.producthunt.get_secret", return_value="")
    def test_missing_token_returns_gracefully(self, _mock_secret):
        from social_intelligence.tools.producthunt import handle

        result = handle({"topic": "ai"})

        assert result["posts"] == []
        assert "error" in result
        assert result["error"] == "not_configured"

    @patch("social_intelligence.tools.producthunt.get_secret", return_value=None)
    def test_none_token_returns_gracefully(self, _mock_secret):
        from social_intelligence.tools.producthunt import handle

        result = handle({"topic": "ai"})

        assert result["posts"] == []
        assert "error" in result

    @patch("social_intelligence.tools.producthunt.get_secret", return_value="generated/bootstrap?value")
    @patch("social_intelligence.tools.producthunt.post_with_retry")
    def test_invalid_bootstrap_token_skips_network_call(self, mock_post, _mock_secret):
        from social_intelligence.tools.producthunt import handle

        result = handle({"topic": "ai"})

        assert result["error"] == "not_configured"
        mock_post.assert_not_called()

    @patch("social_intelligence.tools.producthunt.get_secret", return_value="fake-ph-token")
    @patch("social_intelligence.tools.producthunt.post_with_retry")
    def test_api_http_error_returns_error_shape(self, mock_post, _mock_secret):
        from social_intelligence.tools.producthunt import handle

        mock_post.side_effect = httpx.HTTPError("Connection timeout")

        result = handle({"topic": "ai", "limit": 5})

        assert result["posts"] == []
        assert result["count"] == 0
        assert result["source"] == "Product Hunt"
        assert result["error"] == "upstream_error"

    @patch("social_intelligence.tools.producthunt.get_secret", return_value="fake-ph-token")
    @patch("social_intelligence.tools.producthunt.post_with_retry")
    def test_non_200_status_returns_error_shape(self, mock_post, _mock_secret):
        from social_intelligence.tools.producthunt import handle

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500 Server Error", request=MagicMock(), response=MagicMock()
        )
        mock_post.return_value = mock_resp

        result = handle({"topic": "ai", "limit": 5})

        assert result["posts"] == []
        assert result["count"] == 0
        assert result["source"] == "Product Hunt"
        assert result["error"] == "upstream_error"

    @patch("social_intelligence.tools.producthunt.get_secret", return_value="fake-ph-token")
    @patch("social_intelligence.tools.producthunt.post_with_retry")
    def test_graphql_errors_field_returns_error_message(self, mock_post, _mock_secret):
        from social_intelligence.tools.producthunt import handle

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"errors": [{"message": "Unauthorized"}, {"message": "Rate limit exceeded"}]}
        mock_post.return_value = mock_resp

        result = handle({"topic": "ai"})

        assert result["posts"] == []
        assert result["count"] == 0
        assert result["source"] == "Product Hunt"
        assert "Unauthorized" in result["error"]

    @patch("social_intelligence.tools.producthunt.get_secret", return_value="fake-ph-token")
    @patch("social_intelligence.tools.producthunt.post_with_retry")
    def test_topic_sanitization_strips_special_chars(self, mock_post, _mock_secret):
        from social_intelligence.tools.producthunt import handle

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {"posts": {"edges": []}}}
        mock_post.return_value = mock_resp

        # Topic with special characters that re.sub should strip
        handle({"topic": "ai & ML! (tools)"})

        call_kwargs = mock_post.call_args[1]
        variables = call_kwargs["json"]["variables"]
        # Only alphanumeric, underscore, hyphen survive re.sub(r"[^a-zA-Z0-9_-]", "", ...)
        assert variables["topic"] == "aiMLtools"

    @patch("social_intelligence.tools.producthunt.get_secret", return_value="fake-ph-token")
    @patch("social_intelligence.tools.producthunt.post_with_retry")
    def test_empty_topic_omits_topic_variable(self, mock_post, _mock_secret):
        from social_intelligence.tools.producthunt import handle

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {"posts": {"edges": []}}}
        mock_post.return_value = mock_resp

        handle({"topic": "", "limit": 5})

        call_kwargs = mock_post.call_args[1]
        variables = call_kwargs["json"]["variables"]
        assert "topic" not in variables

    @patch("social_intelligence.tools.producthunt.get_secret", return_value="fake-ph-token")
    @patch("social_intelligence.tools.producthunt.post_with_retry")
    def test_invalid_order_defaults_to_votes(self, mock_post, _mock_secret):
        from social_intelligence.tools.producthunt import handle

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {"posts": {"edges": []}}}
        mock_post.return_value = mock_resp

        handle({"topic": "ai", "order": "BOGUS"})

        call_kwargs = mock_post.call_args[1]
        variables = call_kwargs["json"]["variables"]
        assert variables["order"] == "VOTES"

    @patch("social_intelligence.tools.producthunt.get_secret", return_value="fake-ph-token")
    @patch("social_intelligence.tools.producthunt.post_with_retry")
    def test_limit_clamped_to_valid_range(self, mock_post, _mock_secret):
        from social_intelligence.tools.producthunt import handle

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {"posts": {"edges": []}}}
        mock_post.return_value = mock_resp

        handle({"topic": "ai", "limit": 999})

        call_kwargs = mock_post.call_args[1]
        variables = call_kwargs["json"]["variables"]
        assert variables["first"] == 20  # clamped to max

    @patch("social_intelligence.tools.producthunt.get_secret", return_value="fake-ph-token")
    @patch("social_intelligence.tools.producthunt.post_with_retry")
    def test_featured_flag_passed_when_set(self, mock_post, _mock_secret):
        from social_intelligence.tools.producthunt import handle

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {"posts": {"edges": []}}}
        mock_post.return_value = mock_resp

        handle({"topic": "ai", "featured": True})

        call_kwargs = mock_post.call_args[1]
        variables = call_kwargs["json"]["variables"]
        assert variables["featured"] is True

    @patch("social_intelligence.tools.producthunt.get_secret", return_value="fake-ph-token")
    @patch("social_intelligence.tools.producthunt.post_with_retry")
    def test_featured_absent_when_not_provided(self, mock_post, _mock_secret):
        from social_intelligence.tools.producthunt import handle

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {"posts": {"edges": []}}}
        mock_post.return_value = mock_resp

        handle({"topic": "ai"})

        call_kwargs = mock_post.call_args[1]
        variables = call_kwargs["json"]["variables"]
        assert "featured" not in variables

    @patch("social_intelligence.tools.producthunt.get_secret", return_value="fake-ph-token")
    @patch("social_intelligence.tools.producthunt.post_with_retry")
    def test_authorization_header_uses_bearer_token(self, mock_post, _mock_secret):
        from social_intelligence.tools.producthunt import handle

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {"posts": {"edges": []}}}
        mock_post.return_value = mock_resp

        handle({"topic": "ai"})

        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["headers"]["Authorization"] == "Bearer fake-ph-token"

    @patch("social_intelligence.tools.producthunt.get_secret", return_value="fake-ph-token")
    @patch("social_intelligence.tools.producthunt.post_with_retry")
    def test_empty_edges_returns_zero_count(self, mock_post, _mock_secret):
        from social_intelligence.tools.producthunt import handle

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {"posts": {"edges": []}}}
        mock_post.return_value = mock_resp

        result = handle({"topic": "nonexistent-topic-xyz"})

        assert result["posts"] == []
        assert result["count"] == 0
        assert result["source"] == "Product Hunt"
        assert "error" not in result
