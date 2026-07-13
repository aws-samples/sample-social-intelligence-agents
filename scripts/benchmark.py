"""Benchmark the Graph and Swarm orchestration patterns head-to-head.

Invokes the deployed Amazon Bedrock AgentCore Runtime once per prospect for each
pattern, then reports latency percentiles and a per-prospect average. Token counts
and per-token cost are read from AgentCore Observability traces in Amazon CloudWatch,
not estimated here — this script measures wall-clock latency and writes a CSV you can
join against the trace data.

Prerequisites:
    export AGENTCORE_AGENT_ARN=arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/social_intel-XXXX
    export AWS_DEFAULT_REGION=us-east-1   # optional, defaults to us-east-1

Usage:
    python scripts/benchmark.py --prospects 50
    python scripts/benchmark.py --prospects 10 --patterns graph
    python scripts/benchmark.py --prospects 50 --prompt "Find recent AI agent launches"
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
import uuid
from dataclasses import dataclass, field

import boto3

REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
DEFAULT_PROMPT = "Find recent AI tool launches and generate outreach emails"


@dataclass
class PatternResult:
    """Latency samples for one orchestration pattern."""

    pattern: str
    latencies: list[float] = field(default_factory=list)
    errors: int = 0

    def summary(self) -> dict[str, float | int | str]:
        """Return latency statistics for this pattern."""
        if not self.latencies:
            return {"pattern": self.pattern, "runs": 0, "errors": self.errors}
        ordered = sorted(self.latencies)
        return {
            "pattern": self.pattern,
            "runs": len(ordered),
            "errors": self.errors,
            "avg_s": round(statistics.mean(ordered), 1),
            "p50_s": round(statistics.median(ordered), 1),
            "p95_s": round(_percentile(ordered, 95), 1),
            "max_s": round(ordered[-1], 1),
        }


def _percentile(ordered: list[float], pct: float) -> float:
    """Return the pct-th percentile of an already-sorted list (nearest-rank)."""
    if not ordered:
        return 0.0
    k = max(0, min(len(ordered) - 1, int(round((pct / 100.0) * len(ordered) + 0.5)) - 1))
    return ordered[k]


def _invoke_once(client, agent_arn: str, prompt: str, pattern: str) -> float:
    """Invoke the runtime once and return wall-clock latency in seconds.

    Raises on transport error so the caller can count it. The full response body
    is drained but not parsed — token counts come from CloudWatch traces, not here.
    """
    payload = json.dumps({"prompt": prompt, "pattern": pattern}).encode()
    start = time.perf_counter()
    response = client.invoke_agent_runtime(
        agentRuntimeArn=agent_arn,
        runtimeSessionId=str(uuid.uuid4()),
        payload=payload,
    )
    stream = response.get("response")
    if stream is not None:
        stream.read()  # drain so latency includes the full streamed run
    return time.perf_counter() - start


def run_benchmark(prospects: int, patterns: list[str], prompt: str) -> dict[str, PatternResult]:
    """Run the benchmark for each pattern and return per-pattern results."""
    agent_arn = os.environ.get("AGENTCORE_AGENT_ARN", "")
    if not agent_arn:
        print("ERROR: set AGENTCORE_AGENT_ARN to the deployed runtime ARN (see CDK output RuntimeArn).")
        sys.exit(1)

    client = boto3.client("bedrock-agentcore", region_name=REGION)
    results = {p: PatternResult(pattern=p) for p in patterns}

    for pattern in patterns:
        print(f"\n=== Pattern: {pattern} — {prospects} runs ===")
        for i in range(prospects):
            try:
                latency = _invoke_once(client, agent_arn, prompt, pattern)
                results[pattern].latencies.append(latency)
                print(f"  [{pattern}] run {i + 1}/{prospects}: {latency:.1f}s")
            except Exception as exc:  # noqa: BLE001 — benchmark must continue past a single failure
                results[pattern].errors += 1
                print(f"  [{pattern}] run {i + 1}/{prospects}: ERROR {type(exc).__name__}: {exc}")

    return results


def write_csv(results: dict[str, PatternResult], path: str) -> None:
    """Write per-run latency samples to a CSV for joining against CloudWatch trace data."""
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["pattern", "run_index", "latency_s"])
        for pattern, result in results.items():
            for idx, latency in enumerate(result.latencies):
                writer.writerow([pattern, idx, round(latency, 3)])


def main() -> None:
    """Parse arguments, run the benchmark, and print a comparison table."""
    parser = argparse.ArgumentParser(description="Benchmark Graph vs Swarm orchestration.")
    parser.add_argument("--prospects", type=int, default=50, help="Runs per pattern (default: 50)")
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["graph", "swarm"],
        choices=["graph", "swarm"],
        help="Patterns to benchmark (default: graph swarm)",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt sent to the runtime")
    parser.add_argument("--csv", default="benchmark_results.csv", help="Output CSV path")
    args = parser.parse_args()

    results = run_benchmark(args.prospects, args.patterns, args.prompt)
    write_csv(results, args.csv)

    print("\n=== Results ===")
    for pattern in args.patterns:
        print(json.dumps(results[pattern].summary()))
    print(f"\nPer-run latencies written to {args.csv}")
    print("For token counts and per-prospect cost, query AgentCore Observability traces in CloudWatch.")


if __name__ == "__main__":
    main()
