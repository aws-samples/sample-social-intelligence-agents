"""Shared HTTP helpers with retry, circuit breaker, and an SSRF allow-list.

Security: httpx enforces TLS certificate validation by default (verify=True).
All external API URLs in this project use HTTPS. The retry logic only retries
on transient failures (429, 5xx); it does not retry on auth or client errors.

SSRF guard: _assert_allowed() enforces an outbound host allow-list. Every call
through get_with_retry or post_with_retry is validated against _ALLOWED_HOSTS
before any network I/O is attempted.

Circuit breaker: Per-host circuit breaker trips after _CB_FAILURE_THRESHOLD
consecutive failures and stays open for _CB_RECOVERY_SECONDS. While open,
requests to that host fail immediately with an httpx.HTTPError rather than
waiting for network timeouts. This prevents cascading delays when an upstream
API is down.
"""

import logging
import threading
import time
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BACKOFF_BASE = 1.0  # seconds
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# Circuit breaker configuration
_CB_FAILURE_THRESHOLD = 5  # consecutive failures before tripping
_CB_RECOVERY_SECONDS = 60.0  # time to stay open before trying again

# SSRF guard: only these hosts may be contacted by tool handlers.
_ALLOWED_HOSTS: frozenset[str] = frozenset(
    {
        "hacker-news.firebaseio.com",
        "www.reddit.com",
        "oauth.reddit.com",  # Reddit OAuth2 read API (used when reddit-oauth secret is set)
        "api.github.com",
        "en.wikipedia.org",
        "www.googleapis.com",
        "api.producthunt.com",
        "api.stackexchange.com",
        "dev.to",
        "lobste.rs",
    }
)


# ---------------------------------------------------------------------------
# Circuit breaker: per-host failure tracking
# ---------------------------------------------------------------------------

_cb_lock = threading.Lock()
_cb_failures: dict[str, int] = {}  # host → consecutive failure count
_cb_opened_at: dict[str, float] = {}  # host → monotonic time when circuit opened


def _cb_host(url: str) -> str:
    """Extract the host key for circuit breaker tracking."""
    return urlparse(url).netloc.split(":")[0]


def _cb_is_open(host: str) -> bool:
    """Check if the circuit breaker for this host is currently open.

    Returns True (open/blocking) when the host has exceeded the failure
    threshold and the recovery period has not yet elapsed.
    """
    with _cb_lock:
        opened = _cb_opened_at.get(host)
        if opened is None:
            return False
        if time.monotonic() - opened >= _CB_RECOVERY_SECONDS:
            # Recovery period elapsed: allow a single probe request (half-open)
            del _cb_opened_at[host]
            _cb_failures[host] = 0
            return False
        return True


def _cb_record_success(host: str) -> None:
    """Record a successful request, resetting the failure counter."""
    with _cb_lock:
        _cb_failures.pop(host, None)
        _cb_opened_at.pop(host, None)


def _cb_record_failure(host: str) -> None:
    """Record a failed request. Trips the breaker when threshold is reached."""
    with _cb_lock:
        count = _cb_failures.get(host, 0) + 1
        _cb_failures[host] = count
        if count >= _CB_FAILURE_THRESHOLD and host not in _cb_opened_at:
            _cb_opened_at[host] = time.monotonic()
            logger.warning(
                "Circuit breaker OPEN for %s after %d consecutive failures (recovery in %ds)",
                host,
                count,
                int(_CB_RECOVERY_SECONDS),
            )


# ---------------------------------------------------------------------------
# SSRF guard
# ---------------------------------------------------------------------------


def _assert_allowed(url: str) -> None:
    """Raise ValueError if the URL's host is not in the outbound allow-list.

    Args:
        url: The full URL to validate.

    Raises:
        ValueError: When the host is not in _ALLOWED_HOSTS.
    """
    host = urlparse(url).netloc.split(":")[0]  # strip optional port
    if host not in _ALLOWED_HOSTS:
        raise ValueError(f"host not allowed: {host}")


# ---------------------------------------------------------------------------
# Retry loop with circuit breaker integration
# ---------------------------------------------------------------------------


def _retry_loop(
    request_fn,
    url: str,
    attempt_count: int = _MAX_RETRIES,
) -> httpx.Response:
    """Run a request function with exponential backoff on transient failures.

    Integrates with the per-host circuit breaker: if the breaker is open,
    raises immediately without network I/O. On success, resets the breaker.
    On exhausted retries, records the failure for breaker tracking.
    """
    host = _cb_host(url)

    # Circuit breaker check: fail fast if the host is known-down
    if _cb_is_open(host):
        logger.warning("Circuit breaker OPEN for %s: skipping request", host)
        raise httpx.ConnectError(f"Circuit breaker open for {host}")

    last_exc: Exception | None = None
    resp: httpx.Response | None = None
    for attempt in range(attempt_count):
        try:
            resp = request_fn()
            if resp.status_code not in _RETRYABLE_STATUS:
                _cb_record_success(host)
                return resp
            logger.warning("Retryable status %d from %s (attempt %d)", resp.status_code, url, attempt + 1)
        except httpx.HTTPError as exc:
            last_exc = exc
            logger.warning("HTTP error from %s (attempt %d): %s", url, attempt + 1, exc)

        if attempt < attempt_count - 1:
            wait = _BACKOFF_BASE * (2**attempt)
            time.sleep(wait)

    # All retries exhausted: record failure for circuit breaker
    _cb_record_failure(host)

    if last_exc:
        raise last_exc
    return resp  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Test / lifecycle hook
# ---------------------------------------------------------------------------


def reset_state() -> None:
    """Reset module-level circuit-breaker state.

    Intended for test isolation: the circuit breaker is process-global by design
    (it must persist across calls in a warm Lambda/Runtime container), so tests
    must reset it between cases to avoid cross-test state leakage. Not used on
    the production path.
    """
    with _cb_lock:
        _cb_failures.clear()
        _cb_opened_at.clear()


# ---------------------------------------------------------------------------
# Public API: one-shot httpx calls guarded by the SSRF allow-list, retry, and breaker
# ---------------------------------------------------------------------------


def get_with_retry(
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: float = 15.0,
    follow_redirects: bool = False,
) -> httpx.Response:
    """HTTP GET with SSRF allow-list, retry, and per-host circuit breaker.

    Validates the target host against the SSRF allow-list before attempting
    any network I/O. Issues a one-shot httpx.get per call (no shared client);
    each tool invocation is short-lived in the agent process, so a persistent
    pool would add lifecycle complexity without a measurable win. Retries up to
    3 times on connection errors and 429/5xx; fails fast when the per-host
    circuit breaker is open.

    Args:
        url: Target URL (host must be in _ALLOWED_HOSTS).
        params: Optional query parameters.
        headers: Optional request headers.
        timeout: Request timeout in seconds.
        follow_redirects: Whether to follow HTTP redirects (default False).

    Raises:
        ValueError: If the host is not in the allow-list.
        httpx.ConnectError: If the circuit breaker is open for this host.
        httpx.HTTPError: After all retries are exhausted.
    """
    _assert_allowed(url)
    return _retry_loop(
        lambda: httpx.get(url, params=params, headers=headers, timeout=timeout, follow_redirects=follow_redirects),
        url,
    )


def post_with_retry(
    url: str,
    *,
    json: dict | None = None,
    headers: dict | None = None,
    timeout: float = 15.0,
) -> httpx.Response:
    """HTTP POST with SSRF allow-list, retry, and per-host circuit breaker.

    Validates the target host against the SSRF allow-list before attempting
    any network I/O. Issues a one-shot httpx.post per call (no shared client);
    each tool invocation is short-lived in the agent process, so a persistent
    pool would add lifecycle complexity without a measurable win. Retries up to
    3 times on connection errors and 429/5xx; fails fast when the per-host
    circuit breaker is open.

    Args:
        url: Target URL (host must be in _ALLOWED_HOSTS).
        json: Optional JSON body.
        headers: Optional request headers.
        timeout: Request timeout in seconds.

    Raises:
        ValueError: If the host is not in the allow-list.
        httpx.ConnectError: If the circuit breaker is open for this host.
        httpx.HTTPError: After all retries are exhausted.
    """
    _assert_allowed(url)
    return _retry_loop(
        lambda: httpx.post(url, json=json, headers=headers, timeout=timeout),
        url,
    )
