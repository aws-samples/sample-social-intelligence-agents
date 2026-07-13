"""Product Hunt GraphQL API: discover trending product launches.

Security: API token retrieved from AWS Secrets Manager at invocation time
(never hardcoded). IAM policy scopes GetSecretValue to the specific secret
ARN (social-intel/producthunt-api-token). HTTPS enforced for all API calls.
"""

import logging
import re

from ._freshness import freshness_weight
from ._http import post_with_retry
from ._secrets import get_secret

logger = logging.getLogger(__name__)

_VALID_ORDERS = {"VOTES", "RANKING", "NEWEST", "FEATURED_AT"}


def handle(params: dict) -> dict:
    """Query Product Hunt for trending products.

    Args:
        params: topic (str), order (str), limit (int), featured (bool|None)
    """
    topic = re.sub(r"[^a-zA-Z0-9_-]", "", str(params.get("topic", "")))[:100]
    order = params.get("order", "VOTES")
    if order not in _VALID_ORDERS:
        order = "VOTES"
    limit = max(1, min(int(params.get("limit", 10)), 20))
    featured = params.get("featured")

    token = get_secret("social-intel/producthunt-api-token")
    if not token:
        return {"posts": [], "error": "No PH token"}

    query = """
    query($order: PostsOrder!, $first: Int!, $topic: String, $featured: Boolean) {
        posts(order: $order, first: $first, topic: $topic, featured: $featured) {
            edges { node {
                id name tagline votesCount commentsCount url website createdAt featuredAt
                topics { edges { node { name slug } } }
                makers { name username }
            } }
        }
    }
    """
    variables: dict = {"order": order, "first": limit}
    if topic:
        variables["topic"] = topic
    if featured is not None:
        variables["featured"] = bool(featured)

    try:
        resp = post_with_retry(
            "https://api.producthunt.com/v2/api/graphql",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"query": query, "variables": variables},
            timeout=15.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        logger.exception("Product Hunt API error")
        return {"posts": [], "count": 0, "source": "Product Hunt", "error": "upstream_error"}

    if "errors" in data:
        error_msg = "; ".join(e.get("message", "Unknown") for e in data["errors"][:3])
        return {"posts": [], "count": 0, "source": "Product Hunt", "error": error_msg}

    posts = []
    for edge in data.get("data", {}).get("posts", {}).get("edges", []):
        p = edge["node"]
        posts.append(
            {
                "id": p.get("id", ""),
                "name": p.get("name", ""),
                "tagline": p.get("tagline", ""),
                "votes": p.get("votesCount", 0),
                "comments": p.get("commentsCount", 0),
                "url": p.get("url", ""),
                "website": p.get("website", ""),
                "created_at": p.get("createdAt", ""),
                "featured_at": p.get("featuredAt", ""),
                "topics": [t["node"]["name"] for t in p.get("topics", {}).get("edges", [])],
                "makers": [m.get("name", "") for m in p.get("makers", [])],
                "freshness_weight": freshness_weight(p.get("createdAt", "")),
                "source": "Product Hunt",
            }
        )
    return {"posts": posts, "count": len(posts), "source": "Product Hunt"}
