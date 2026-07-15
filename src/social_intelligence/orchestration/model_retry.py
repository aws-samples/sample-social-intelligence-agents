"""Bounded retries for transient model transport failures.

Botocore cannot replay a Bedrock event stream once it has started. This hook uses the
public Strands ``AfterModelCallEvent`` retry contract to retry only retryable transport
and throttling failures, with a hard cap of three total model attempts.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator
from typing import Any

from botocore.exceptions import ConnectionClosedError, ConnectTimeoutError, EndpointConnectionError, ReadTimeoutError
from strands.hooks import AfterInvocationEvent, AfterModelCallEvent, HookProvider, HookRegistry
from strands.types.exceptions import ModelThrottledException
from urllib3.exceptions import ProtocolError
from urllib3.exceptions import ReadTimeoutError as Urllib3ReadTimeoutError

logger = logging.getLogger(__name__)

_RETRYABLE_EXCEPTIONS = (
    ConnectTimeoutError,
    ConnectionClosedError,
    EndpointConnectionError,
    ModelThrottledException,
    ReadTimeoutError,
    Urllib3ReadTimeoutError,
    ProtocolError,
)


def _exception_chain(exception: BaseException) -> Iterator[BaseException]:
    """Yield an exception and its causal chain once each."""
    seen: set[int] = set()
    current: BaseException | None = exception
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


class TransientModelRetry(HookProvider):
    """Retry transient Bedrock model failures with an explicit total-attempt cap."""

    def __init__(self, *, max_attempts: int = 3, initial_delay: float = 2.0, max_delay: float = 8.0) -> None:
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        if initial_delay < 0 or max_delay < initial_delay:
            raise ValueError("retry delays must be non-negative and max_delay must not be smaller than initial_delay")
        self._max_attempts = max_attempts
        self._initial_delay = initial_delay
        self._max_delay = max_delay
        self._retry_count = 0

    def register_hooks(self, registry: HookRegistry, **_: Any) -> None:
        """Register against the public Strands model and invocation lifecycle events."""
        registry.add_callback(AfterModelCallEvent, self.retry_transient_failure)
        registry.add_callback(AfterInvocationEvent, self.reset)

    async def retry_transient_failure(self, event: AfterModelCallEvent) -> None:
        """Request a retry after a retryable model transport failure."""
        if event.retry:
            return
        if event.exception is None:
            self._retry_count = 0
            return
        if not any(isinstance(error, _RETRYABLE_EXCEPTIONS) for error in _exception_chain(event.exception)):
            return
        if self._retry_count >= self._max_attempts - 1:
            logger.warning(
                "Model retry budget exhausted after %d total attempt(s): %s",
                self._max_attempts,
                type(event.exception).__name__,
            )
            return

        delay = min(self._initial_delay * (2**self._retry_count), self._max_delay)
        self._retry_count += 1
        logger.warning(
            "Retrying transient model failure: retry=%d/%d delay=%.1fs exception=%s",
            self._retry_count,
            self._max_attempts - 1,
            delay,
            type(event.exception).__name__,
        )
        if delay:
            await asyncio.sleep(delay)
        event.retry = True

    def reset(self, _: AfterInvocationEvent) -> None:
        """Reset the retry budget after every complete agent invocation."""
        self._retry_count = 0


def transient_model_retry() -> TransientModelRetry:
    """Return the shared three-total-attempt model retry policy."""
    return TransientModelRetry(max_attempts=3, initial_delay=2.0, max_delay=8.0)
