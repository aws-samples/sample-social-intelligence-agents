"""Shared pytest fixtures for the social-intelligence test suite."""

import pytest


@pytest.fixture(autouse=True)
def _reset_http_circuit_breaker():
    """Reset the _http circuit-breaker state before and after every test.

    The circuit breaker is process-global by design (it must persist across
    calls in a warm Lambda/Runtime container). Without this reset, a test that
    trips the breaker for a host leaks that open state into later tests,
    causing order-dependent flakiness. Resetting per test guarantees isolation.
    """
    from social_intelligence.tools import _http

    _http.reset_state()
    yield
    _http.reset_state()
