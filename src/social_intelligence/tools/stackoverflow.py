"""Stack Overflow / Stack Exchange API: technology demand signals from Q&A trends."""

import logging

from ._freshness import freshness_weight
from ._http import get_with_retry

logger = logging.getLogger(__name__)

SE_API = "https://api.stackexchange.com/2.3"


def handle(params: dict) -> dict:
    """Search Stack Overflow for questions related to a technology or topic.

    Rising question counts for a technology indicate growing market demand.
    Unanswered questions signal unmet needs and potential opportunities.

    Args:
        params: query (str, max 300 chars), sort (str, one of activity/votes/creation/relevance),
            limit (int 1-20, default 10)

    Returns:
        Dict with keys: questions (list), query (str), count (int), has_more (bool),
        quota_remaining (int), source (str). On error, adds an ``error`` key.
    """
    query = str(params.get("query", ""))[:300]
    sort = params.get("sort", "activity")
    if sort not in {"activity", "votes", "creation", "relevance"}:
        sort = "relevance"
    limit = max(1, min(int(params.get("limit", 10)), 20))

    try:
        resp = get_with_retry(
            f"{SE_API}/search/advanced",
            params={
                "order": "desc",
                "sort": sort,
                "q": query,
                "site": "stackoverflow",
                "pagesize": limit,
                # Opaque filter "!nNPvSNdWme" includes: title, link, score, answer_count,
                # view_count, is_answered, tags, creation_date, body (truncated).
                # Regenerate at: https://api.stackexchange.com/docs/create-filter
                "filter": "!nNPvSNdWme",
            },
            timeout=15.0,
        )
        resp.raise_for_status()
    except Exception:
        logger.exception("Stack Overflow API error for query '%s'", query)
        return {"questions": [], "query": query, "count": 0, "source": "Stack Overflow", "error": "upstream_error"}

    data = resp.json()

    questions = []
    for item in data.get("items", []):
        questions.append(
            {
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "score": item.get("score", 0),
                "answer_count": item.get("answer_count", 0),
                "view_count": item.get("view_count", 0),
                "is_answered": item.get("is_answered", False),
                "tags": item.get("tags", []),
                "created": item.get("creation_date", 0),
                "freshness_weight": freshness_weight(item.get("creation_date", 0)),
            }
        )

    return {
        "questions": questions,
        "query": query,
        "count": len(questions),
        "has_more": data.get("has_more", False),
        "quota_remaining": data.get("quota_remaining", 0),
        "source": "Stack Overflow",
    }
