"""Benchmark Graph and Swarm with terminal-aware runtime and ADOT measurements.

The benchmark measures a complete streamed pipeline invocation, not merely a drained
HTTP response. It then obtains input and output token counts from AgentCore
Observability spans through the installed AgentCore evaluation SDK. Costs are calculated
only when the operator supplies current model-specific rates.

Prerequisites:
    export AGENTCORE_AGENT_ARN=arn:aws:bedrock-agentcore:eu-west-1:123456789012:runtime/social_intel-XXXX
    export AGENTCORE_EVENT_LOG_GROUP=/aws/bedrock-agentcore/social-intelligence/application
    export LEADS_TABLE_NAME=social-intel-leads  # optional: creates human-review worksheet

Usage:
    python scripts/benchmark.py --prospects 50
    python scripts/benchmark.py --prospects 50 --patterns graph --trace-wait-seconds 300
    python scripts/benchmark.py --input-cost-per-million 3 --output-cost-per-million 15
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import bedrock_agentcore  # noqa: F401 - registers the AgentCore botocore service model
import boto3
from bedrock_agentcore.evaluation import fetch_spans_from_cloudwatch
from botocore.config import Config

REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
DEFAULT_PROMPT = "Find recent AI tool launches and generate outreach emails"
DEFAULT_STREAM_TIMEOUT_SECONDS = 1_500
DEFAULT_TRACE_WAIT_SECONDS = 300


@dataclass
class InvocationResult:
    """One complete AgentCore Runtime invocation and optional telemetry measurements."""

    pattern: str
    run_index: int
    session_id: str
    started_at: datetime
    latency_s: float
    completed: bool
    error: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    model_calls: int | None = None
    cost_usd: float | None = None
    trace_error: str = ""


@dataclass
class PatternResult:
    """Invocation results aggregated for one orchestration pattern."""

    pattern: str
    invocations: list[InvocationResult] = field(default_factory=list)

    def summary(self) -> dict[str, float | int | str | None]:
        """Return completed-run latency and telemetry summary values."""
        successful = [result for result in self.invocations if result.completed]
        latencies = sorted(result.latency_s for result in successful)
        tokenized = [result for result in successful if result.input_tokens is not None]
        costs = [result.cost_usd for result in successful if result.cost_usd is not None]
        summary: dict[str, float | int | str | None] = {
            "pattern": self.pattern,
            "runs": len(self.invocations),
            "completed": len(successful),
            "errors": len(self.invocations) - len(successful),
        }
        if latencies:
            summary.update(
                {
                    "avg_s": round(statistics.mean(latencies), 1),
                    "p50_s": round(statistics.median(latencies), 1),
                    "p95_s": round(_percentile(latencies, 95), 1),
                    "max_s": round(latencies[-1], 1),
                }
            )
        if tokenized:
            summary.update(
                {
                    "avg_input_tokens": round(statistics.mean(result.input_tokens or 0 for result in tokenized)),
                    "avg_output_tokens": round(statistics.mean(result.output_tokens or 0 for result in tokenized)),
                    "avg_model_calls": round(statistics.mean(result.model_calls or 0 for result in tokenized), 1),
                }
            )
        if costs:
            summary["avg_cost_usd"] = round(statistics.mean(costs), 6)
        return summary


def _percentile(ordered: list[float], percentile: int) -> float:
    """Return the nearest-rank percentile of an already-sorted list."""
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, int(round((percentile / 100) * len(ordered) + 0.5)) - 1))
    return ordered[index]


def _iter_response_lines(body: Any):
    """Yield decoded response lines from an AgentCore streaming body."""
    if hasattr(body, "iter_lines"):
        for line in body.iter_lines(chunk_size=1024):
            yield line.decode("utf-8", errors="replace") if isinstance(line, bytes) else str(line)
        return

    buffer = ""
    for chunk in body:
        buffer += chunk.decode("utf-8", errors="replace") if isinstance(chunk, bytes) else str(chunk)
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            yield line
    if buffer.strip():
        yield buffer


def _consume_terminal_stream(body: Any, timeout_seconds: int) -> None:
    """Consume a response stream and require the entrypoint terminal result event."""
    deadline = time.monotonic() + timeout_seconds
    try:
        for line in _iter_response_lines(body):
            if time.monotonic() > deadline:
                raise TimeoutError(f"stream exceeded {timeout_seconds} seconds before a terminal pipeline result")

            raw = line.strip()
            if not raw:
                continue
            if raw.startswith("data: "):
                raw = raw[6:]
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue

            event_type = str(event.get("type", ""))
            if event_type == "multiagent_result":
                return
            if event_type in {"error", "multiagent_error"}:
                raise RuntimeError(str(event.get("message") or event.get("error") or "runtime emitted an error event"))
    finally:
        close = getattr(body, "close", None)
        if callable(close):
            close()

    raise RuntimeError("response ended without a terminal pipeline result")


def _invoke_once(
    client: Any,
    agent_arn: str,
    prompt: str,
    pattern: str,
    run_index: int,
    timeout_seconds: int,
) -> InvocationResult:
    """Invoke one complete pipeline and return a terminal-aware benchmark record."""
    session_id = uuid.uuid4().hex
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    try:
        response = client.invoke_agent_runtime(
            agentRuntimeArn=agent_arn,
            runtimeSessionId=session_id,
            payload=json.dumps({"prompt": prompt, "pattern": pattern, "run_id": session_id}).encode(),
            contentType="application/json",
            accept="text/event-stream, application/json",
            qualifier="DEFAULT",
        )
        body = response.get("response")
        if body is None:
            raise RuntimeError("AgentCore Runtime returned no response body")
        _consume_terminal_stream(body, timeout_seconds)
        return InvocationResult(
            pattern=pattern,
            run_index=run_index,
            session_id=session_id,
            started_at=started_at,
            latency_s=time.perf_counter() - started,
            completed=True,
        )
    except Exception as exc:  # noqa: BLE001 - benchmark records per-run failures and continues
        return InvocationResult(
            pattern=pattern,
            run_index=run_index,
            session_id=session_id,
            started_at=started_at,
            latency_s=time.perf_counter() - started,
            completed=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def run_benchmark(
    prospects: int,
    patterns: list[str],
    prompt: str,
    timeout_seconds: int,
) -> dict[str, PatternResult]:
    """Run one terminal-aware invocation per requested prospect and pattern."""
    agent_arn = os.environ.get("AGENTCORE_AGENT_ARN", "")
    if not agent_arn:
        sys.exit("ERROR: set AGENTCORE_AGENT_ARN to the deployed RuntimeArn.")

    client = boto3.client(
        "bedrock-agentcore",
        region_name=REGION,
        config=Config(read_timeout=timeout_seconds + 60, connect_timeout=10, retries={"max_attempts": 0}),
    )
    results = {pattern: PatternResult(pattern=pattern) for pattern in patterns}

    for pattern in patterns:
        print(f"\n=== Pattern: {pattern}; {prospects} complete runs ===")
        for run_index in range(1, prospects + 1):
            result = _invoke_once(client, agent_arn, prompt, pattern, run_index, timeout_seconds)
            results[pattern].invocations.append(result)
            outcome = f"{result.latency_s:.1f}s" if result.completed else f"ERROR {result.error}"
            print(f"  [{pattern}] run {run_index}/{prospects}: {outcome}")
    return results


def _as_nonnegative_int(value: object) -> int | None:
    """Coerce a span attribute value to a nonnegative integer, when possible."""
    if isinstance(value, dict):
        value = value.get("value")
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except TypeError, ValueError:
        return None
    return parsed if parsed >= 0 else None


def _token_usage_from_spans(spans: list[dict[str, Any]]) -> tuple[int, int, int]:
    """Sum token attributes from distinct model-invocation spans."""
    input_tokens = 0
    output_tokens = 0
    model_calls = 0
    seen_span_ids: set[tuple[str, str]] = set()

    for index, span in enumerate(spans):
        if not isinstance(span, dict):
            continue
        span_key = (str(span.get("traceId", "")), str(span.get("spanId", "")))
        if not all(span_key):
            span_key = ("unidentified", str(index))
        if span_key in seen_span_ids:
            continue
        seen_span_ids.add(span_key)

        attributes = span.get("attributes")
        if not isinstance(attributes, dict):
            continue
        input_value = _as_nonnegative_int(attributes.get("gen_ai.usage.input_tokens"))
        output_value = _as_nonnegative_int(attributes.get("gen_ai.usage.output_tokens"))
        if input_value is None and output_value is None:
            continue
        input_tokens += input_value or 0
        output_tokens += output_value or 0
        model_calls += 1
    return input_tokens, output_tokens, model_calls


def _span_cost_usd(
    input_tokens: int,
    output_tokens: int,
    input_cost_per_million: float | None,
    output_cost_per_million: float | None,
) -> float | None:
    """Calculate cost only when explicit current rates cover both token directions."""
    if input_cost_per_million is None or output_cost_per_million is None:
        return None
    return (input_tokens * input_cost_per_million + output_tokens * output_cost_per_million) / 1_000_000


def collect_observability(
    results: dict[str, PatternResult],
    event_log_group: str,
    trace_wait_seconds: int,
    input_cost_per_million: float | None,
    output_cost_per_million: float | None,
) -> None:
    """Attach actual ADOT token usage and optional cost to completed invocations."""
    completed = [result for group in results.values() for result in group.invocations if result.completed]
    if not completed:
        return
    if not event_log_group:
        print("\nADOT token collection skipped: set AGENTCORE_EVENT_LOG_GROUP from the CDK output.")
        return

    if trace_wait_seconds:
        print(f"\nWaiting {trace_wait_seconds}s for ADOT spans to arrive in CloudWatch...")
        time.sleep(trace_wait_seconds)

    end_time = datetime.now(timezone.utc)
    for result in completed:
        try:
            spans = fetch_spans_from_cloudwatch(
                session_id=result.session_id,
                event_log_group=event_log_group,
                start_time=result.started_at,
                end_time=end_time,
                region=REGION,
            )
            input_tokens, output_tokens, model_calls = _token_usage_from_spans(spans)
            result.input_tokens = input_tokens
            result.output_tokens = output_tokens
            result.model_calls = model_calls
            result.cost_usd = _span_cost_usd(
                input_tokens,
                output_tokens,
                input_cost_per_million,
                output_cost_per_million,
            )
        except Exception as exc:  # noqa: BLE001 - missing traces must not discard measured latency
            result.trace_error = f"{type(exc).__name__}: {exc}"


def write_csv(results: dict[str, PatternResult], path: Path) -> None:
    """Write every invocation, including failures and missing-telemetry state, to CSV."""
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "pattern",
                "run_index",
                "session_id",
                "started_at",
                "completed",
                "latency_s",
                "input_tokens",
                "output_tokens",
                "model_calls",
                "cost_usd",
                "error",
                "trace_error",
            ],
        )
        writer.writeheader()
        for group in results.values():
            for result in group.invocations:
                writer.writerow(
                    {
                        "pattern": result.pattern,
                        "run_index": result.run_index,
                        "session_id": result.session_id,
                        "started_at": result.started_at.isoformat(),
                        "completed": result.completed,
                        "latency_s": round(result.latency_s, 3),
                        "input_tokens": result.input_tokens,
                        "output_tokens": result.output_tokens,
                        "model_calls": result.model_calls,
                        "cost_usd": result.cost_usd,
                        "error": result.error,
                        "trace_error": result.trace_error,
                    }
                )


def _email_leads_for_session(table: Any, session_id: str) -> list[dict[str, Any]]:
    """Return full email-lead records associated with one benchmark session."""
    from boto3.dynamodb.conditions import Key

    records: list[dict[str, Any]] = []
    query_kwargs: dict[str, Any] = {
        "IndexName": "session-id-discovered-at-index",
        "KeyConditionExpression": Key("session_id").eq(session_id),
        "ProjectionExpression": "prospect_id, discovered_at, email_body",
    }
    while True:
        response = table.query(**query_kwargs)
        for record in response.get("Items", []):
            if not str(record.get("email_body", "")).strip():
                continue
            item = table.get_item(
                Key={"prospect_id": record["prospect_id"], "discovered_at": record["discovered_at"]}
            ).get("Item")
            if item:
                records.append(item)
        if not response.get("LastEvaluatedKey"):
            return records
        query_kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]


def write_human_review_template(
    results: dict[str, PatternResult],
    leads_table_name: str,
    path: Path,
) -> int:
    """Create a two-reviewer worksheet from persisted drafts and source evidence."""
    table = boto3.resource("dynamodb", region_name=REGION).Table(leads_table_name)
    rows: list[dict[str, Any]] = []
    for group in results.values():
        for result in group.invocations:
            if not result.completed:
                continue
            for lead in _email_leads_for_session(table, result.session_id):
                rows.append(
                    {
                        "pattern": result.pattern,
                        "run_index": result.run_index,
                        "session_id": result.session_id,
                        "prospect_id": lead.get("prospect_id", ""),
                        "product_name": lead.get("product_name", ""),
                        "email_body": lead.get("email_body", ""),
                        "evidence_json": lead.get("evidence_json", ""),
                        "reviewer_1_relevance_1_to_5": "",
                        "reviewer_1_personalization_1_to_5": "",
                        "reviewer_1_grounding_1_to_5": "",
                        "reviewer_1_notes": "",
                        "reviewer_2_relevance_1_to_5": "",
                        "reviewer_2_personalization_1_to_5": "",
                        "reviewer_2_grounding_1_to_5": "",
                        "reviewer_2_notes": "",
                    }
                )

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else _review_fieldnames())
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _review_fieldnames() -> list[str]:
    """Return the review worksheet schema even when no qualified email was generated."""
    return [
        "pattern",
        "run_index",
        "session_id",
        "prospect_id",
        "product_name",
        "email_body",
        "evidence_json",
        "reviewer_1_relevance_1_to_5",
        "reviewer_1_personalization_1_to_5",
        "reviewer_1_grounding_1_to_5",
        "reviewer_1_notes",
        "reviewer_2_relevance_1_to_5",
        "reviewer_2_personalization_1_to_5",
        "reviewer_2_grounding_1_to_5",
        "reviewer_2_notes",
    ]


def _nonnegative_rate(value: str) -> float | None:
    """Parse an optional per-million-token cost without silently inventing pricing."""
    if not value.strip():
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("cost rate must be a nonnegative number") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("cost rate must be a nonnegative number")
    return parsed


def main() -> None:
    """Parse benchmark options, run measurements, and write machine and human artifacts."""
    parser = argparse.ArgumentParser(description="Benchmark Graph vs Swarm orchestration.")
    parser.add_argument("--prospects", type=int, default=50, help="Complete runs per pattern (default: 50)")
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["graph", "swarm"],
        choices=["graph", "swarm"],
        help="Patterns to benchmark (default: graph swarm)",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt sent to the runtime")
    parser.add_argument("--csv", type=Path, default=Path("benchmark_results.csv"), help="Per-run CSV output")
    parser.add_argument(
        "--event-log-group",
        default=os.environ.get("AGENTCORE_EVENT_LOG_GROUP", ""),
        help="Runtime application event log group from the CDK output",
    )
    parser.add_argument(
        "--trace-wait-seconds",
        type=int,
        default=DEFAULT_TRACE_WAIT_SECONDS,
        help="Wait once after invocations for ADOT spans (default: 300)",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_STREAM_TIMEOUT_SECONDS,
        help="Maximum wall-clock duration for one streamed invocation (default: 1500)",
    )
    parser.add_argument(
        "--input-cost-per-million",
        type=_nonnegative_rate,
        default=_nonnegative_rate(os.environ.get("BENCHMARK_INPUT_COST_PER_MILLION", "")),
        help="Current model input-token USD rate per million; omit to leave cost blank",
    )
    parser.add_argument(
        "--output-cost-per-million",
        type=_nonnegative_rate,
        default=_nonnegative_rate(os.environ.get("BENCHMARK_OUTPUT_COST_PER_MILLION", "")),
        help="Current model output-token USD rate per million; omit to leave cost blank",
    )
    parser.add_argument(
        "--leads-table",
        default=os.environ.get("LEADS_TABLE_NAME", ""),
        help="Optional DynamoDB leads table used to create the two-reviewer worksheet",
    )
    parser.add_argument(
        "--review-csv",
        type=Path,
        default=Path("benchmark_human_review.csv"),
        help="Two-reviewer quality worksheet output",
    )
    parser.add_argument("--no-review-template", action="store_true", help="Do not query leads or write review CSV")
    args = parser.parse_args()

    if args.prospects < 1:
        parser.error("--prospects must be at least 1")
    if args.timeout_seconds < 1:
        parser.error("--timeout-seconds must be at least 1")
    if args.trace_wait_seconds < 0:
        parser.error("--trace-wait-seconds cannot be negative")
    if (args.input_cost_per_million is None) != (args.output_cost_per_million is None):
        parser.error("provide both input and output token rates to calculate cost")

    results = run_benchmark(args.prospects, args.patterns, args.prompt, args.timeout_seconds)
    collect_observability(
        results,
        args.event_log_group,
        args.trace_wait_seconds,
        args.input_cost_per_million,
        args.output_cost_per_million,
    )
    write_csv(results, args.csv)

    print("\n=== Results ===")
    for pattern in args.patterns:
        print(json.dumps(results[pattern].summary(), sort_keys=True))
    print(f"\nPer-run latency and ADOT telemetry written to {args.csv}")
    if args.input_cost_per_million is None:
        print("Cost is blank: supply current input and output per-million-token rates to calculate it.")

    if args.leads_table and not args.no_review_template:
        try:
            rows = write_human_review_template(results, args.leads_table, args.review_csv)
            print(f"Two-reviewer worksheet written to {args.review_csv} ({rows} email draft(s)).")
        except Exception as exc:  # noqa: BLE001 - a review artifact cannot invalidate core measurements
            print(f"Human-review worksheet skipped: {type(exc).__name__}: {exc}", file=sys.stderr)
    elif not args.no_review_template:
        print("Human-review worksheet skipped: set LEADS_TABLE_NAME or pass --leads-table.")


if __name__ == "__main__":
    main()
