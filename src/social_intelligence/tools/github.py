"""GitHub Search API: open-source intelligence."""

import logging
from datetime import date, timedelta

from ._freshness import freshness_weight
from ._http import get_with_retry
from ._secrets import get_secret

logger = logging.getLogger(__name__)

_VALID_SORTS = {"stars", "forks", "updated"}


def _auth_headers() -> dict[str, str]:
    """Return request headers, adding an optional Bearer token when available.

    An optional token raises the GitHub rate limit from 60 to 5000 req/hr.
    Secret id: social-intel/github-token
    """
    headers: dict[str, str] = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "AnyCompanyBot/1.0",
    }
    try:
        token = get_secret("social-intel/github-token").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
    except Exception:
        # Secret unavailable: proceed unauthenticated (60 req/hr limit applies)
        logger.debug("GitHub token secret unavailable; using unauthenticated requests")
    return headers


def handle(params: dict) -> dict:
    """Search GitHub for repositories related to a topic.

    Only repos created in the past 90 days are returned to surface trending work.

    Args:
        params: query (str, max 500 chars), sort (str, one of stars/forks/updated),
            limit (int 1-10, default 5)

    Returns:
        Dict with keys: repos (list), query (str), count (int), source (str).
        On error, adds an ``error`` key and returns empty repos.
    """
    query = str(params.get("query", ""))[:500]
    sort = params.get("sort", "stars")
    if sort not in _VALID_SORTS:
        sort = "stars"
    limit = max(1, min(int(params.get("limit", 5)), 10))

    cutoff = (date.today() - timedelta(days=90)).isoformat()

    try:
        resp = get_with_retry(
            "https://api.github.com/search/repositories",
            params={"q": f"{query} created:>{cutoff}", "sort": sort, "per_page": limit},
            headers=_auth_headers(),
            timeout=15.0,
        )
        resp.raise_for_status()
    except Exception:
        logger.exception("GitHub API error for query '%s'", query)
        return {"repos": [], "query": query, "count": 0, "source": "GitHub", "error": "upstream_error"}

    repos = []
    for item in resp.json().get("items", []):
        updated_at = item["updated_at"]
        repos.append(
            {
                "name": item["full_name"],
                "description": (item.get("description") or "")[:200],
                "stars": item["stargazers_count"],
                "forks": item["forks_count"],
                "language": item.get("language", ""),
                "url": item["html_url"],
                "updated": updated_at,
                "topics": item.get("topics", [])[:5],
                "freshness_weight": freshness_weight(updated_at),
            }
        )
    return {"repos": repos, "query": query, "count": len(repos), "source": "GitHub"}
