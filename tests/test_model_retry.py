"""Tests for bounded retries of transient Strands model failures."""

import asyncio
from unittest.mock import MagicMock

from strands.hooks import AfterInvocationEvent, AfterModelCallEvent
from urllib3.exceptions import ReadTimeoutError

from social_intelligence.orchestration.model_retry import TransientModelRetry


def _read_timeout() -> ReadTimeoutError:
    return ReadTimeoutError(None, "https://bedrock-runtime.example", "read timed out")


def test_retries_transient_errors_only_within_total_attempt_budget() -> None:
    retry = TransientModelRetry(max_attempts=3, initial_delay=0, max_delay=0)

    first = AfterModelCallEvent(agent=MagicMock(), exception=_read_timeout())
    asyncio.run(retry.retry_transient_failure(first))
    second = AfterModelCallEvent(agent=MagicMock(), exception=_read_timeout())
    asyncio.run(retry.retry_transient_failure(second))
    third = AfterModelCallEvent(agent=MagicMock(), exception=_read_timeout())
    asyncio.run(retry.retry_transient_failure(third))

    assert first.retry is True
    assert second.retry is True
    assert third.retry is False


def test_does_not_retry_non_transient_errors_and_resets_after_invocation() -> None:
    retry = TransientModelRetry(max_attempts=2, initial_delay=0, max_delay=0)
    non_transient = AfterModelCallEvent(agent=MagicMock(), exception=ValueError("invalid response"))
    asyncio.run(retry.retry_transient_failure(non_transient))
    assert non_transient.retry is False

    retry.reset(AfterInvocationEvent(agent=MagicMock()))
    transient = AfterModelCallEvent(agent=MagicMock(), exception=_read_timeout())
    asyncio.run(retry.retry_transient_failure(transient))
    assert transient.retry is True
