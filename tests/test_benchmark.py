"""Unit tests for the terminal-aware benchmark and ADOT token parsing."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def benchmark_module():
    """Load the standalone benchmark script without requiring a scripts package."""
    path = Path(__file__).parents[1] / "scripts" / "benchmark.py"
    spec = importlib.util.spec_from_file_location("benchmark_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Stream:
    """Minimal AgentCore-like streaming body for terminal-event tests."""

    def __init__(self, lines: list[bytes]):
        self._lines = lines
        self.closed = False

    def iter_lines(self, **_):
        yield from self._lines

    def close(self):
        self.closed = True


def test_consume_terminal_stream_requires_result_event(benchmark_module):
    """A drained response without the pipeline result must be treated as failed."""
    stream = _Stream([b'data: {"type":"multiagent_node_start","node_id":"research"}'])

    with pytest.raises(RuntimeError, match="terminal pipeline result"):
        benchmark_module._consume_terminal_stream(stream, timeout_seconds=1)

    assert stream.closed is True


def test_consume_terminal_stream_accepts_result_event(benchmark_module):
    """The benchmark completes only after the entrypoint's terminal event."""
    stream = _Stream(
        [
            b'data: {"type":"multiagent_node_start","node_id":"research"}',
            b'data: {"type":"multiagent_result","result":"complete"}',
        ]
    )

    benchmark_module._consume_terminal_stream(stream, timeout_seconds=1)

    assert stream.closed is True


def test_token_usage_deduplicates_spans_and_ignores_non_model_records(benchmark_module):
    """Token accounting uses distinct spans with GenAI usage attributes only."""
    spans = [
        {
            "traceId": "trace-1",
            "spanId": "span-1",
            "attributes": {
                "gen_ai.usage.input_tokens": 100,
                "gen_ai.usage.output_tokens": "25",
            },
        },
        {
            "traceId": "trace-1",
            "spanId": "span-1",
            "attributes": {
                "gen_ai.usage.input_tokens": 100,
                "gen_ai.usage.output_tokens": 25,
            },
        },
        {
            "traceId": "trace-1",
            "spanId": "span-2",
            "attributes": {
                "gen_ai.usage.input_tokens": {"value": 40},
                "gen_ai.usage.output_tokens": 10,
            },
        },
        {"traceId": "trace-1", "spanId": "event-1", "attributes": {"gen_ai.tool.name": "github_search"}},
    ]

    assert benchmark_module._token_usage_from_spans(spans) == (140, 35, 2)
    assert benchmark_module._span_cost_usd(140, 35, 3.0, 15.0) == pytest.approx(0.000945)
    assert benchmark_module._span_cost_usd(140, 35, None, 15.0) is None
