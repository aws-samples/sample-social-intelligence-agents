"""Shared Secrets Manager helper with TTL-based in-memory caching.

Used by the credential-consuming tools (YouTube, Product Hunt, GitHub, Reddit
OAuth2), which run in the Lambda behind the AgentCore Gateway. The secrets are
created CMK-encrypted by the CDK stack; callers tolerate an empty/absent value
and fall back to unauthenticated behavior, so the sample runs with zero
credentials configured.

Security properties:
- Secret VALUES are never logged (only the secret_id, never the SecretString).
- Fails closed: any Secrets Manager error propagates to the caller, which
  catches it and degrades to the unauthenticated path. Errors are NOT cached,
  so a transient failure does not poison the cache.
- Lazy-imports boto3 to minimize cold start impact.

Cache expires after 10 minutes to bound the window during which a rotated
secret stays valid in a warm Lambda (threat-model finding TS004). The TTL
governs Lambda-side caching only; each tool invocation is short-lived, so a
tighter TTL costs at most one extra GetSecretValue per interval. Tune via
SECRET_CACHE_TTL_SECONDS (lower for stricter rotation, higher to cut calls).

Future migration: AgentCore Identity's token vault + @requires_api_key /
@requires_access_token decorators are the managed alternative, but those run
in the AgentCore Runtime, not in this Lambda. See SECURITY.md.
"""

import os
import threading
import time

_CACHE_TTL = int(os.environ.get("SECRET_CACHE_TTL_SECONDS", "600"))  # 10 minutes

_cache: dict[str, tuple[str, float]] = {}
_lock = threading.Lock()

# Lazy singleton: created once on first use so a warm Lambda reuses the connection.
# boto3 is imported lazily to minimize cold-start overhead.
_client = None
_client_lock = threading.Lock()


def _get_client():
    """Return the module-level Secrets Manager client, creating it on first call."""
    global _client  # noqa: PLW0603
    if _client is None:
        with _client_lock:
            if _client is None:
                import boto3

                _client = boto3.client("secretsmanager")
    return _client


def get_secret(secret_id: str) -> str:
    """Retrieve a secret from AWS Secrets Manager with TTL caching.

    Uses a two-phase lock to avoid serializing all callers during the
    Secrets Manager network call:
      Phase 1: acquire lock, check cache, release immediately on hit.
      Phase 2: call Secrets Manager WITHOUT holding the lock (slow I/O).
      Phase 3: re-acquire lock to write the result, then return.

    Two concurrent misses may both fetch; that is acceptable because
    Secrets Manager is idempotent and the last writer overwrites
    with an equivalent value.
    """
    # Phase 1: fast path: return cached value if still fresh.
    with _lock:
        entry = _cache.get(secret_id)
        if entry and (time.monotonic() - entry[1]) < _CACHE_TTL:
            return entry[0]

    # Phase 2: cache miss: fetch without holding the lock so other
    # threads are not serialized behind this network call.
    client = _get_client()
    resp = client.get_secret_value(SecretId=secret_id)
    value = resp["SecretString"]

    # Phase 3: write result back under lock (last writer wins, which is fine).
    with _lock:
        _cache[secret_id] = (value, time.monotonic())

    return value
