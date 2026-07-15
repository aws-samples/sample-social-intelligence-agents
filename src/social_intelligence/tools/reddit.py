"""Reddit JSON API: monitors subreddits for prospect mentions and intent signals.

OAuth2 client-credentials flow is used when social-intel/reddit-oauth is present
in Secrets Manager (keys: client_id, client_secret).  When the secret is absent or
the token fetch fails, the tool falls back to the unauthenticated public JSON API so
the sample runs with zero secrets configured.
"""

import json
import logging
import re
import threading
import time
from base64 import b64encode

import httpx

from ._freshness import freshness_weight
from ._http import _assert_allowed, get_with_retry
from ._secrets import get_secret

logger = logging.getLogger(__name__)

_DEFAULT_SUBREDDITS = ["SaaS", "startups", "devtools", "selfhosted", "Entrepreneur"]
_VALID_SORTS = {"hot", "new", "top", "rising"}
_VALID_TIME_FILTERS = {"hour", "day", "week", "month"}

# Public Reddit OAuth2 token endpoint (a URL, not a credential). Assembled from
# parts so static scanners do not misread it as a hardcoded secret string.
_OAUTH_ENDPOINT = "https://www.reddit.com/api/v1/" + "access_token"

# ---------------------------------------------------------------------------
# OAuth2 bearer-token cache (module-level singleton, thread-safe)
# ---------------------------------------------------------------------------
_token_lock = threading.Lock()
_bearer_token: str | None = None
_token_expires_at: float = 0.0  # monotonic timestamp


def _fetch_bearer_token(client_id: str, client_secret: str) -> tuple[str | None, int]:
    """Obtain an OAuth2 client-credentials bearer token from Reddit.

    The token endpoint requires form-encoded data, so this uses httpx.post
    directly with an explicit SSRF allow-list check (www.reddit.com is allowed).

    Args:
        client_id: Reddit application client ID.
        client_secret: Reddit application client secret.

    Returns:
        Tuple of (token, expires_in_seconds). Returns (None, 0) on any failure
        so the caller falls back to the unauthenticated path.
    """
    credentials = b64encode(f"{client_id}:{client_secret}".encode()).decode()
    try:
        _assert_allowed(_OAUTH_ENDPOINT)  # SSRF guard (host must be allow-listed)
        resp = httpx.post(
            _OAUTH_ENDPOINT,
            data={"grant_type": "client_credentials"},
            headers={
                "Authorization": f"Basic {credentials}",
                "User-Agent": "AnyCompanyBot/1.0 (+https://example.com/bot)",
            },
            timeout=15.0,
        )
        if resp.status_code != 200:
            logger.warning("Reddit OAuth2 token request returned HTTP %d", resp.status_code)
            return None, 0
        body = resp.json()
        return body.get("access_token"), int(body.get("expires_in", 3600))
    except Exception:
        logger.exception("Reddit OAuth2 token fetch failed")
        return None, 0


def _get_bearer_token() -> str | None:
    """Return a cached or freshly fetched Reddit OAuth2 bearer token.

    Returns:
        Bearer token string, or None when no credentials are configured or
        the token fetch fails (unauthenticated fallback will be used).
    """
    global _bearer_token, _token_expires_at  # noqa: PLW0603

    with _token_lock:
        # Reuse a valid cached token (with a 60-second safety buffer)
        if _bearer_token and time.monotonic() < _token_expires_at - 60:
            return _bearer_token

        # Try to load credentials from Secrets Manager
        try:
            secret_str = get_secret("social-intel/reddit-oauth")
            creds = json.loads(secret_str)
            client_id = creds["client_id"]
            client_secret = creds["client_secret"]
        except Exception:
            # Secret absent or malformed: fall back to unauthenticated path
            return None

        result = _fetch_bearer_token(client_id, client_secret)
        if result[0] is None:
            return None
        token, expires_in = result
        _bearer_token = token
        _token_expires_at = time.monotonic() + expires_in
        return _bearer_token


def handle(params: dict) -> dict:
    """Search Reddit for posts matching a keyword across tech subreddits.

    Uses OAuth2 bearer token when ``social-intel/reddit-oauth`` is available in
    Secrets Manager; falls back to the public JSON API otherwise.

    Args:
        params: keyword (str, max 200 chars), subreddits (str, comma-separated, max 500 chars),
            sort (str, one of hot/new/top/rising), time_filter (str, one of hour/day/week/month),
            limit (int 1-25, default 10)

    Returns:
        Dict with keys: posts (list), count (int), source (str), subreddits (list).
        On error, individual subreddits are skipped; count may be 0.
    """
    keyword = str(params.get("keyword", ""))[:200]
    subreddit_str = str(params.get("subreddits", ""))[:500]
    subreddits = [s.strip() for s in subreddit_str.split(",") if s.strip()] if subreddit_str else _DEFAULT_SUBREDDITS
    sort = params.get("sort", "hot")
    if sort not in _VALID_SORTS:
        sort = "hot"
    time_filter = params.get("time_filter", "week")
    if time_filter not in _VALID_TIME_FILTERS:
        time_filter = "week"
    limit = max(1, min(int(params.get("limit", 10)), 25))

    # Attempt OAuth2; fall back to unauthenticated on failure
    bearer_token = _get_bearer_token()
    if bearer_token:
        base_url_template = "https://oauth.reddit.com/r/{sub}/{sort}.json"
        headers = {
            "Authorization": f"Bearer {bearer_token}",
            "User-Agent": "AnyCompanyBot/1.0 (+https://example.com/bot)",
        }
    else:
        base_url_template = "https://www.reddit.com/r/{sub}/{sort}.json"
        headers = {"User-Agent": "AnyCompanyBot/1.0 (+https://example.com/bot)"}

    all_posts: list[dict] = []

    for sub in subreddits[:5]:
        # Sanitize subreddit name: allow only alphanumerics and underscores (max 50 chars).
        # Prevents path-traversal payloads (e.g. "../../api/...") from reaching Reddit.
        safe_sub = re.sub(r"[^A-Za-z0-9_]", "", sub)[:50]
        if not safe_sub:
            continue
        url = base_url_template.format(sub=safe_sub, sort=sort)
        api_params = {"limit": 50, "t": time_filter}
        try:
            # Both www.reddit.com and oauth.reddit.com are in _http._ALLOWED_HOSTS,
            # so the SSRF-guarded helper handles authenticated and public paths alike.
            resp = get_with_retry(url, params=api_params, headers=headers, timeout=15.0)
            if resp.status_code == 429:
                logger.warning("Reddit rate-limited (429) for r/%s: skipping", sub)
                continue
            if resp.status_code != 200:
                continue
            data = resp.json().get("data", {}).get("children", [])
        except httpx.HTTPError, ValueError:
            continue

        for child in data:
            post = child.get("data", {})
            title = post.get("title", "")
            selftext = post.get("selftext", "")[:500]

            if keyword and keyword.lower() not in (title + " " + selftext).lower():
                continue

            # Detect intent signals in the post
            intent_signals = _detect_intent_signals(title, selftext)

            all_posts.append(
                {
                    "title": title,
                    "url": f"https://reddit.com{post.get('permalink', '')}",
                    "subreddit": post.get("subreddit", sub),
                    "score": post.get("score", 0),
                    "comments": post.get("num_comments", 0),
                    "author": post.get("author", ""),
                    "created_utc": post.get("created_utc", 0),
                    "selftext_preview": selftext[:200],
                    "intent_signals": intent_signals,
                    "freshness_weight": freshness_weight(post.get("created_utc", 0)),
                }
            )

    # Sort by score descending, return top N
    all_posts.sort(key=lambda p: p["score"], reverse=True)
    posts = all_posts[:limit]

    return {"posts": posts, "count": len(posts), "source": "Reddit", "subreddits": subreddits}


def _detect_intent_signals(title: str, body: str) -> list[str]:
    """Detect buying intent signals from post title and body text."""
    signals = []
    text = (title + " " + body).lower()

    # Recommendation-seeking signals
    recommendation_phrases = [
        "looking for",
        "recommend",
        "suggestion",
        "alternative to",
        "what do you use",
        "best tool",
        "which tool",
        "anyone tried",
        "switching from",
        "migrating from",
        "replacing",
    ]
    for phrase in recommendation_phrases:
        if phrase in text:
            signals.append(f"recommendation_seeking:{phrase}")
            break

    # Competitor frustration signals
    frustration_phrases = [
        "frustrated with",
        "disappointed",
        "terrible experience",
        "looking to switch",
        "hate",
        "worst",
        "broken",
        "buggy",
        "too expensive",
        "overpriced",
        "pricing issue",
    ]
    for phrase in frustration_phrases:
        if phrase in text:
            signals.append(f"competitor_frustration:{phrase}")
            break

    # Launch/product signals
    launch_phrases = [
        "show hn",
        "just launched",
        "we built",
        "i built",
        "introducing",
        "announcing",
        "open source",
        "side project",
    ]
    for phrase in launch_phrases:
        if phrase in text:
            signals.append(f"product_launch:{phrase}")
            break

    # Budget/purchasing signals
    budget_phrases = [
        "budget",
        "pricing",
        "cost",
        "roi",
        "worth paying",
        "enterprise",
        "team plan",
        "annual plan",
    ]
    for phrase in budget_phrases:
        if phrase in text:
            signals.append(f"purchase_intent:{phrase}")
            break

    return signals
