"""Eval harness for the social intelligence lead-gen pipeline.

Two modes:
  --offline  Validates golden_set.json structure and runs the grounding_gate
             structured fallback against synthetic email/evidence pairs.
             No AWS credentials required.
  --live     Starts each golden prompt as an AgentCore background run (survives the
             response-stream idle window), waits for a successful terminal marker, then
             reads results from DynamoDB by run_id:
             scored_prospect prompts read persisted analysis scores; email_draft prompts
             read persisted drafts and score them with an LLM-as-judge on relevance and
             on grounding against the lead's recorded evidence. Requires AWS credentials
             and AGENTCORE_AGENT_ARN.

Usage:
  uv run python scripts/eval_quality.py --offline
  uv run python scripts/eval_quality.py --live
  uv run python scripts/eval_quality.py --live --limit 1

Output: a plain-text summary table printed to stdout.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_GOLDEN_SET_PATH = _REPO_ROOT / "eval" / "golden_set.json"
_GROUNDING_GATE_PATH = _REPO_ROOT / "src" / "social_intelligence" / "tools" / "grounding_gate.py"


def _load_golden_set() -> list[dict[str, Any]]:
    """Load and return the golden set from eval/golden_set.json.

    Returns:
        List of golden set item dicts.

    Raises:
        SystemExit: If the file is missing or not valid JSON.
    """
    if not _GOLDEN_SET_PATH.exists():
        sys.exit(f"ERROR: golden_set.json not found at {_GOLDEN_SET_PATH}")
    try:
        with _GOLDEN_SET_PATH.open() as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        sys.exit(f"ERROR: golden_set.json is not valid JSON: {exc}")
    return data  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Offline mode
# ---------------------------------------------------------------------------

_SYNTHETIC_CASES: list[dict[str, Any]] = [
    {
        "description": "All claims supported — high grounding score",
        "email_body": "Hi, noticed your repo hit 1200 stars on GitHub last week. Very impressive.",
        "evidence": '{"github_stars": 1200, "week": "last week", "repo": "example-tool"}',
        "expected_score_min": 0.9,
    },
    {
        "description": "Mixed claims — partial support",
        "email_body": "Your tool has 500 GitHub stars and 10K monthly downloads.",
        "evidence": '{"github_stars": 500, "description": "popular open-source tool"}',
        "expected_score_min": 0.4,
        "expected_score_max": 0.7,
    },
    {
        "description": "Unsupported numeric claim — low grounding score",
        "email_body": "With $5M in funding and 2000 GitHub stars, you're clearly growing fast.",
        "evidence": '{"product": "Example SaaS", "category": "developer tools", "github_stars": 2000}',
        "expected_score_min": 0.4,
        "expected_score_max": 0.6,
    },
    {
        "description": "No numeric claims — perfect grounding score",
        "email_body": "I noticed your product was featured on Product Hunt recently. Congrats.",
        "evidence": '{"source": "producthunt", "featured": true}',
        "expected_score_min": 1.0,
    },
    {
        "description": "Percentage claim not in evidence",
        "email_body": "Your tool reduces build times by 40%.",
        "evidence": '{"product": "BuildFast", "category": "CI/CD"}',
        "expected_score_min": 0.0,
        "expected_score_max": 0.1,
    },
]


def _validate_golden_set(items: list[dict[str, Any]]) -> list[str]:
    """Validate that every golden set item has required fields.

    Args:
        items: Loaded golden set items.

    Returns:
        List of error strings. Empty list means validation passed.
    """
    required = {"prompt", "expected_type", "min_score"}
    errors: list[str] = []
    for idx, item in enumerate(items):
        missing = required - set(item.keys())
        if missing:
            errors.append(f"Item {idx}: missing fields {missing}")
        if not isinstance(item.get("prompt", ""), str) or not item.get("prompt", "").strip():
            errors.append(f"Item {idx}: 'prompt' must be a non-empty string")
        if item.get("expected_type") not in ("email_draft", "scored_prospect"):
            errors.append(f"Item {idx}: 'expected_type' must be 'email_draft' or 'scored_prospect'")
        min_score = item.get("min_score")
        if isinstance(min_score, bool) or not isinstance(min_score, (int, float)):
            errors.append(f"Item {idx}: 'min_score' must be a number")
            continue

        expected_type = item.get("expected_type")
        lower_bound, upper_bound = (1, 10) if expected_type == "email_draft" else (0, 100)
        if not lower_bound <= min_score <= upper_bound:
            errors.append(f"Item {idx}: 'min_score' must be in [{lower_bound}, {upper_bound}] for {expected_type}")

        if "max_score" not in item:
            continue
        max_score = item["max_score"]
        if isinstance(max_score, bool) or not isinstance(max_score, (int, float)):
            errors.append(f"Item {idx}: 'max_score' must be a number")
        elif not lower_bound <= max_score <= upper_bound:
            errors.append(f"Item {idx}: 'max_score' must be in [{lower_bound}, {upper_bound}] for {expected_type}")
        elif min_score > max_score:
            errors.append(f"Item {idx}: 'min_score' cannot exceed 'max_score'")
    return errors


def _run_grounding_fallback(email_body: str, evidence_json: str) -> dict[str, Any]:
    """Run the grounding gate structured fallback using the production code path.

    Imports _extract_claims and _token_in_evidence from grounding_gate.py to ensure
    the eval harness always matches production behavior (no duplicated regex).

    Args:
        email_body: Email text to check.
        evidence_json: JSON string with evidence.

    Returns:
        Dict with grounding_score, supported, unsupported_claims keys.
    """
    # Add src to path so the import works from scripts/
    src_dir = str(_REPO_ROOT / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from social_intelligence.tools.grounding_gate import _extract_claims, _token_in_evidence

    try:
        evidence_obj = json.loads(evidence_json)
        evidence_text = json.dumps(evidence_obj)
    except json.JSONDecodeError, TypeError, ValueError:
        evidence_text = str(evidence_json)
    evidence_lower = evidence_text.lower()

    claims = _extract_claims(email_body)
    if not claims:
        return {"grounding_score": 1.0, "supported": [], "unsupported_claims": []}

    supported = [c for c in claims if _token_in_evidence(c, evidence_lower)]
    unsupported = [c for c in claims if not _token_in_evidence(c, evidence_lower)]
    grounding_score = len(supported) / len(claims)
    return {
        "grounding_score": round(grounding_score, 4),
        "supported": supported,
        "unsupported_claims": unsupported,
    }


def run_offline() -> bool:
    """Run offline validation of golden set structure and grounding gate logic.

    Returns:
        True if all checks pass, False if any check fails.
    """
    print("=" * 60)
    print("OFFLINE EVAL — social intelligence quality harness")
    print("=" * 60)

    # 1. Validate golden set structure
    print("\n[1/2] Validating golden_set.json structure ...")
    items = _load_golden_set()
    errors = _validate_golden_set(items)
    if errors:
        for err in errors:
            print(f"  FAIL: {err}")
        return False
    print(f"  PASS: {len(items)} items, all fields valid")

    # 2. Grounding gate synthetic tests
    print(f"\n[2/2] Running grounding gate on {len(_SYNTHETIC_CASES)} synthetic cases ...")
    all_pass = True
    for case in _SYNTHETIC_CASES:
        result = _run_grounding_fallback(case["email_body"], case["evidence"])
        score = result["grounding_score"]
        min_ok = score >= case["expected_score_min"]
        max_ok = score <= case.get("expected_score_max", 1.0)
        status = "PASS" if (min_ok and max_ok) else "FAIL"
        if status == "FAIL":
            all_pass = False
        bounds = f"[{case['expected_score_min']:.1f}, {case.get('expected_score_max', 1.0):.1f}]"
        print(f"  {status}  score={score:.2f} (expected {bounds})  — {case['description']}")
        if status == "FAIL":
            print(f"       supported={result['supported']}  unsupported={result['unsupported_claims']}")

    print()
    if all_pass:
        print("ALL OFFLINE CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED — see details above")
    return all_pass


# ---------------------------------------------------------------------------
# Live mode
# ---------------------------------------------------------------------------

# Max seconds to poll DynamoDB for a background run's results. Covers the graph's
# execution budget (set_execution_timeout(1200) in entrypoint.py) plus write lag, so an
# allowed-but-slow run is not cut off before it persists. Env-tunable for faster local runs.
_RESULT_POLL_TIMEOUT_S = int(os.environ.get("EVAL_POLL_TIMEOUT_SECONDS", "1260"))
_RESULT_POLL_DELAY_S = int(os.environ.get("EVAL_POLL_DELAY_SECONDS", "15"))
_RUN_STATUS_PRODUCT_PREFIX = "__run_status__:"
_RUN_STATUS_SUCCEEDED = "succeeded"
_RUN_STATUS_FAILED = "failed"
_VALID_EVAL_PATTERNS = frozenset({"graph", "swarm", "both"})

# Minimum judged grounding (1-10) an email must reach to pass, alongside its relevance
# threshold. A fluent but unsupported email should not pass on relevance alone.
_EMAIL_GROUNDING_MIN = int(os.environ.get("EVAL_GROUNDING_MIN", "5"))

# Judge model. Defaults to Claude Sonnet 4.6 via the "global." inference profile for
# stronger relevance/grounding judgments than Haiku, and so --live works in any
# supported Region (US and EU). Override with JUDGE_MODEL_ID to pin a Region-scoped
# profile or a cheaper model. Mirrors the region-agnostic MODEL_ID default in config.
_JUDGE_MODEL = os.environ.get("JUDGE_MODEL_ID", "global.anthropic.claude-sonnet-4-6")
_JUDGE_PROMPT = """You are an email quality judge. Score the following outreach email on two dimensions.
Return the scores through the submit_quality_scores tool.

Prompt that triggered the pipeline:
{prompt}

Email body:
{email_body}

Evidence gathered for this prospect (the research and enrichment the email may cite):
{evidence}

Relevance: does the email address a genuine pain point suggested by the prompt?
(1=irrelevant, 10=highly targeted)
Grounding: is every specific claim in the email supported by the evidence above?
Penalize claims (numbers, funding, metrics) that the evidence does not support.
(1=unsupported/fabricated, 10=every claim traceable to the evidence)
"""
_JUDGE_TOOL_NAME = "submit_quality_scores"
_JUDGE_TOOL_CONFIG = {
    "tools": [
        {
            "toolSpec": {
                "name": _JUDGE_TOOL_NAME,
                "description": "Submit final relevance and grounding scores for the evaluated outreach email.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "relevance": {
                                "type": "integer",
                                # Bedrock Converse strict tool schemas reject numeric range keywords.
                                # _judge_score enforces the 1-10 contract after tool-use parsing.
                                "description": (
                                    "Integer from 1 to 10: how specifically the email addresses "
                                    "the requested buyer need."
                                ),
                            },
                            "grounding": {
                                "type": "integer",
                                "description": (
                                    "Integer from 1 to 10: how completely the email claims are supported by evidence."
                                ),
                            },
                        },
                        "required": ["relevance", "grounding"],
                    }
                },
                "strict": True,
            }
        }
    ],
    "toolChoice": {"tool": {"name": _JUDGE_TOOL_NAME}},
}


def _iter_sse_json_events(chunks):
    """Yield JSON payloads from an SSE byte stream, including split event chunks."""
    buffer = ""
    for chunk in chunks:
        if isinstance(chunk, bytes):
            buffer += chunk.decode("utf-8", errors="replace")
        elif isinstance(chunk, str):
            buffer += chunk
        else:
            logger.debug("Ignoring non-text SSE chunk of type %s", type(chunk).__name__)
            continue

        # AgentCore emits standard data: JSON SSE frames. Normalize CRLF only after
        # buffering so a JSON payload split across network chunks remains intact.
        buffer = buffer.replace("\r\n", "\n")
        while "\n\n" in buffer:
            frame, buffer = buffer.split("\n\n", 1)
            data_lines = [line[5:].lstrip() for line in frame.splitlines() if line.startswith("data:")]
            if not data_lines:
                continue
            try:
                payload = json.loads("\n".join(data_lines))
            except json.JSONDecodeError:
                logger.debug("Ignoring non-JSON SSE payload")
                continue
            if isinstance(payload, dict):
                yield payload


def _start_background_run(arn: str, prompt: str, region: str, run_id: str, pattern: str = "graph") -> str:
    """Start a pipeline run as an AgentCore async task and return its acknowledged run_id.

    The runtime executes long multi-agent runs in a background thread (staying
    HealthyBusy) and persists all results to DynamoDB, so this call returns quickly with
    an ``async_started`` acknowledgment rather than holding a minutes-long SSE stream that
    would idle-disconnect. Results are read back from DynamoDB by run_id afterward.

    run_id is passed as the payload ``run_id`` (not ``session_id``) on purpose: it drives
    lead attribution WITHOUT enabling AgentCore Memory, which keys off session_id and
    would add latency and cross-prompt state to an otherwise one-shot eval run.

    Args:
        arn: AgentCore agent runtime ARN.
        prompt: User prompt to send.
        region: AWS region.
        run_id: Attribution id stamped onto every record the run persists.
        pattern: Orchestration pattern to exercise, 'graph' (default) or 'swarm'.

    Returns:
        The run_id echoed by the runtime's async_started acknowledgment.

    Raises:
        RuntimeError: If the invoke call fails or the runtime does not acknowledge the
            background run.
    """
    import boto3
    from botocore.config import Config as BotoConfig
    from botocore.exceptions import ResponseStreamingError

    client = boto3.client(
        "bedrock-agentcore",
        region_name=region,
        # AgentCore must acknowledge background work before the client can begin
        # DynamoDB polling. Permit a cold container start instead of failing the
        # invocation client while the runtime is still preparing the async task.
        config=BotoConfig(read_timeout=300, connect_timeout=10, retries={"total_max_attempts": 1}),
    )
    try:
        response = client.invoke_agent_runtime(
            agentRuntimeArn=arn,
            runtimeSessionId=f"eval-{run_id}-{uuid.uuid4().hex}",
            payload=json.dumps(
                {"prompt": prompt, "pattern": pattern, "run_id": run_id, "isolate": True, "background": True}
            ).encode(),
            contentType="application/json",
            accept="text/event-stream, application/json",
            qualifier="DEFAULT",
        )
    except Exception as exc:
        raise RuntimeError(f"invoke_agent_runtime failed: {exc}") from exc

    body = response.get("response")
    if body is None:
        raise RuntimeError("invoke_agent_runtime returned no response stream")
    acknowledged = None
    try:
        chunks = body.iter_chunks(chunk_size=4096) if hasattr(body, "iter_chunks") else body
        for event in _iter_sse_json_events(chunks):
            if event.get("type") == "async_started":
                acknowledged = str(event.get("run_id") or run_id)
                break
    except ResponseStreamingError:
        logger.debug("stream closed after background acknowledgment")
    finally:
        close = getattr(body, "close", None)
        if callable(close):
            close()
    if acknowledged is None:
        raise RuntimeError("runtime did not acknowledge the background run")
    return acknowledged


def _parse_judge_json(raw: str) -> dict[str, Any]:
    """Parse the judge's scores from a model response that may wrap the JSON.

    Stronger judge models often return the JSON object inside a markdown code fence
    or alongside a sentence of reasoning, so a bare json.loads() fails. This extracts
    the first balanced {...} object and parses that.

    Args:
        raw: The raw text content of the judge model response.

    Returns:
        The parsed JSON object as a dict.

    Raises:
        ValueError: If no JSON object can be located or parsed.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError, TypeError:
        pass
    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in judge response: {raw[:120]!r}")
    return json.loads(match.group(0))


def _judge_score(scores: dict[str, Any], name: str) -> int:
    """Return one strictly valid 1-10 LLM-judge score."""
    value = scores.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 10:
        raise ValueError(f"judge {name!r} score must be an integer from 1 to 10")
    return value


def _judge_scores_from_response(response: dict[str, Any]) -> dict[str, Any]:
    """Extract forced tool output, retaining a text fallback for model overrides."""
    content = response.get("output", {}).get("message", {}).get("content", [])
    if not isinstance(content, list):
        raise ValueError("judge response contains no content blocks")

    text_blocks: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        tool_use = block.get("toolUse")
        if isinstance(tool_use, dict) and tool_use.get("name") == _JUDGE_TOOL_NAME:
            scores = tool_use.get("input")
            if isinstance(scores, dict):
                return scores
            raise ValueError("judge tool result has no JSON input object")
        text = block.get("text")
        if isinstance(text, str):
            text_blocks.append(text)

    if text_blocks:
        return _parse_judge_json("\n".join(text_blocks))
    raise ValueError("judge did not call the required scoring tool")


def _records_from_dynamodb(region: str, run_id: str) -> list[dict[str, Any]]:
    """Read all records the pipeline persisted for a run (email leads and score rows).

    DynamoDB is the authoritative store: the SSE stream has no heartbeat and can drop
    mid-run, but the pipeline still writes results server-side. Records are attributed
    by session_id == run_id via the session GSI, so this is exact even when many prompts
    run concurrently. Emailed-lead rows carry an email_body; analysis score rows do not,
    which is how _is_lead_record / _is_score_record tell them apart. Score rows capture
    every scored prospect, including sub-threshold ones that never become emails. The
    query stays within the GSI projection (email_body, score, product_name).

    Args:
        region: AWS region of the leads table.
        run_id: The run identifier stamped onto this run's records (stored as session_id).

    Returns:
        Records (score, email_body, product_name) written by this run.
    """
    import boto3
    from boto3.dynamodb.conditions import Key

    table_name = os.environ.get("LEADS_TABLE_NAME", "social-intel-leads")
    table = boto3.resource("dynamodb", region_name=region).Table(table_name)
    records: list[dict[str, Any]] = []
    query_kwargs: dict[str, Any] = {
        "IndexName": "session-id-discovered-at-index",
        "KeyConditionExpression": Key("session_id").eq(run_id),
        # prospect_id + discovered_at are the base-table key (always on the GSI) so an
        # email lead can be fetched in full for grounding evidence via _full_lead_record.
        "ProjectionExpression": "prospect_id, discovered_at, email_body, score, product_name",
    }
    while True:
        resp = table.query(**query_kwargs)
        records.extend(resp.get("Items", []))
        start_key = resp.get("LastEvaluatedKey")
        if not start_key:
            break
        query_kwargs["ExclusiveStartKey"] = start_key
    return records


def _is_lead_record(record: dict[str, Any]) -> bool:
    """An emailed-lead record has a non-empty email_body; a score record does not."""
    return bool(str(record.get("email_body", "")).strip())


def _is_run_status_record(record: dict[str, Any]) -> bool:
    """Return whether a record marks terminal success or failure for this run."""
    return str(record.get("product_name", "")).startswith(_RUN_STATUS_PRODUCT_PREFIX)


def _terminal_run_status(records: list[dict[str, Any]]) -> str | None:
    """Return a valid terminal status from a run's persisted records, if present."""
    for record in records:
        product_name = str(record.get("product_name", ""))
        if not product_name.startswith(_RUN_STATUS_PRODUCT_PREFIX):
            continue
        status = product_name.removeprefix(_RUN_STATUS_PRODUCT_PREFIX)
        if status in {_RUN_STATUS_SUCCEEDED, _RUN_STATUS_FAILED}:
            return status
    return None


def _run_execution_path(region: str, records: list[dict[str, Any]]) -> str | None:
    """Read the terminal marker's non-projected execution path from the base table.

    The session GSI deliberately projects only evaluation fields. The terminal marker's
    primary key is still present, so one exact base-table read keeps recovery metadata
    available without widening the index.
    """
    for record in records:
        if not _is_run_status_record(record):
            continue
        execution_path = _full_lead_record(region, record).get("execution_path")
        return str(execution_path) if isinstance(execution_path, str) and execution_path else None
    return None


def _is_score_record(record: dict[str, Any]) -> bool:
    """A persisted score has a score value but is neither an emailed lead nor run marker."""
    return not _is_lead_record(record) and not _is_run_status_record(record) and record.get("score") is not None


def _full_lead_record(region: str, record: dict[str, Any]) -> dict[str, Any]:
    """Fetch a lead's full base-table item for grounding evidence.

    The session GSI projects only a few attributes, so the enrichment_summary, reasoning,
    and persisted grounding_score used to judge grounding live only on the base item.

    Args:
        region: AWS region of the leads table.
        record: A GSI-projected record carrying prospect_id and discovered_at.

    Returns:
        The full base-table item, or the projected record if the key lookup fails.
    """
    import boto3

    prospect_id = record.get("prospect_id")
    discovered_at = record.get("discovered_at")
    if not prospect_id or not discovered_at:
        return record
    table_name = os.environ.get("LEADS_TABLE_NAME", "social-intel-leads")
    table = boto3.resource("dynamodb", region_name=region).Table(table_name)
    item = table.get_item(Key={"prospect_id": prospect_id, "discovered_at": discovered_at}).get("Item")
    return item or record


def _wait_for_terminal_run(
    region: str,
    run_id: str,
    timeout_s: int,
    delay_s: int,
    cancellation_event: threading.Event | None = None,
) -> tuple[str | None, list[dict[str, Any]]]:
    """Wait for a terminal marker and return it with the run's settled records.

    A persisted lead or score is only an intermediate side effect. The evaluator waits
    until the background task declares success, then performs one delayed read to absorb
    eventual GSI propagation and sequential lead writes. A failed or timed-out run is
    never allowed to pass from partial output.

    Args:
        region: AWS region of the leads table.
        run_id: The run identifier stamped onto this run's records.
        timeout_s: Maximum total seconds to wait.
        delay_s: Seconds to wait between reads.
        cancellation_event: When set, stop polling without starting further AWS requests.

    Returns:
        A ``(status, records)`` tuple. ``status`` is ``succeeded`` or ``failed`` when
        the runtime persisted a terminal marker, otherwise ``None`` after cancellation
        or timeout.
    """
    waited = 0
    latest_records: list[dict[str, Any]] = []
    while True:
        if cancellation_event and cancellation_event.is_set():
            return None, latest_records
        latest_records = _records_from_dynamodb(region, run_id)
        status = _terminal_run_status(latest_records)
        if status is not None:
            if status == _RUN_STATUS_FAILED:
                return status, latest_records
            # The GSI is eventually consistent. The terminal marker is written after all
            # pipeline writes, but use one settle read before evaluating the full result.
            time.sleep(delay_s)
            return status, _records_from_dynamodb(region, run_id) or latest_records
        if waited >= timeout_s:
            return None, latest_records
        time.sleep(delay_s)
        waited += delay_s


def _judge_email(bedrock_client: Any, prompt: str, email_body: str, evidence: str) -> dict[str, int]:
    """Score email quality with the judge model (Claude Sonnet 4.6 by default) on Bedrock.

    Args:
        bedrock_client: boto3 bedrock-runtime client.
        prompt: Original pipeline prompt.
        email_body: Email body text to judge.
        evidence: The prospect's gathered research/enrichment, so the judge can score
            grounding against real facts rather than plausibility alone.

    Returns:
        Dict with 'relevance' and 'grounding' int scores (0 on judge failure).
    """
    judge_prompt = _JUDGE_PROMPT.format(
        prompt=prompt, email_body=email_body[:800], evidence=(evidence or "(no evidence recorded)")[:1500]
    )
    try:
        response = bedrock_client.converse(
            modelId=_JUDGE_MODEL,
            messages=[{"role": "user", "content": [{"text": judge_prompt}]}],
            inferenceConfig={"maxTokens": 200, "temperature": 0.0},
            toolConfig=_JUDGE_TOOL_CONFIG,
        )
        scores = _judge_scores_from_response(response)
        return {
            "relevance": _judge_score(scores, "relevance"),
            "grounding": _judge_score(scores, "grounding"),
        }
    except Exception:
        logger.warning("LLM judge call failed", exc_info=True)
        return {"relevance": 0, "grounding": 0}


def _evaluate_prompt(
    item: dict[str, Any],
    idx: int,
    arn: str,
    region: str,
    bedrock_client: Any,
    pattern: str = "graph",
    cancellation_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Run one golden prompt end-to-end and score it. Safe to call concurrently.

    Starts the pipeline as an AgentCore async task (background) in the given pattern, then
    reads results back from DynamoDB by run_id: scored_prospect prompts read persisted
    analysis scores, email prompts read persisted leads and judge the draft. Nothing
    depends on the response stream surviving the multi-minute run.

    Graph persists typed node scores directly. Swarm persists the analyst's complete score
    list through its dedicated agent-side tool before the analyst hands off, so both
    patterns evaluate every score, including prospects below the email threshold.

    Args:
        item: Golden-set entry with prompt, min_score, expected_type.
        idx: Zero-based index (for log labelling).
        arn: AgentCore agent runtime ARN.
        region: AWS region.
        bedrock_client: boto3 bedrock-runtime client for the judge.
        pattern: Orchestration pattern to exercise, 'graph' or 'swarm'.
        cancellation_event: Stops queued or polling work after an operator interrupt.

    Returns:
        Result dict with prompt, pattern, type, min_score, metric, pass (grounding for emails).
    """
    prompt = item["prompt"]
    min_score = item["min_score"]
    max_score = item.get("max_score")
    expected_type = item["expected_type"]
    run_id = uuid.uuid4().hex
    label = f"[{idx + 1}] ({pattern}/{expected_type})"
    if cancellation_event and cancellation_event.is_set():
        return {
            "prompt": prompt,
            "pattern": pattern,
            "type": expected_type,
            "min_score": min_score,
            "max_score": max_score,
            "metric": 0.0,
            "pass": False,
        }
    print(f"{label} start: {prompt[:50]}...")

    # Start the run in the background; the pipeline persists all results to DynamoDB.
    # timeout_s covers full server-side completion of a background multi-agent run.
    _start_background_run(arn, prompt, region, run_id, pattern)
    terminal_status, records = _wait_for_terminal_run(
        region,
        run_id,
        timeout_s=_RESULT_POLL_TIMEOUT_S,
        delay_s=_RESULT_POLL_DELAY_S,
        cancellation_event=cancellation_event,
    )
    if terminal_status != _RUN_STATUS_SUCCEEDED:
        outcome = terminal_status or "timed out waiting for a terminal marker"
        print(f"{label} FAILED RUN: {outcome}")
        return {
            "prompt": prompt,
            "pattern": pattern,
            "type": expected_type,
            "min_score": min_score,
            "max_score": max_score,
            "metric": 0.0,
            "pass": False,
        }

    execution_path = _run_execution_path(region, records)
    if pattern == "swarm" and execution_path == "graph_recovery":
        print(f"{label} FAILED RUN: Swarm violated its output contract and recovered with Graph")
        return {
            "prompt": prompt,
            "pattern": pattern,
            "type": expected_type,
            "min_score": min_score,
            "max_score": max_score,
            "metric": 0.0,
            "execution_path": execution_path,
            "pass": False,
        }

    if expected_type == "scored_prospect":
        score_rows = [record for record in records if _is_score_record(record)]
        scores = [int(r["score"]) for r in score_rows if r.get("score") is not None]
        if not scores:
            print(f"{label} NO SCORES: run persisted no scored prospects")
            return {
                "prompt": prompt,
                "pattern": pattern,
                "type": expected_type,
                "min_score": min_score,
                "max_score": max_score,
                "metric": 0.0,
                "pass": False,
            }
        top_score = max(scores)
        passed = top_score >= min_score and (max_score is None or top_score <= max_score)
        expected = f"[{min_score}, {max_score}]" if max_score is not None else f">= {min_score}"
        print(
            f"{label} {'PASS' if passed else 'FAIL'} top_score={top_score} "
            f"(expected {expected}), {len(scores)} scored prospect(s)"
        )
        return {
            "prompt": prompt,
            "pattern": pattern,
            "type": expected_type,
            "min_score": min_score,
            "max_score": max_score,
            "metric": float(top_score),
            "pass": passed,
        }

    # The terminal marker guarantees email generation has completed, so judge every
    # persisted draft rather than allowing an early first draft to mask later failures.
    leads = [record for record in records if _is_lead_record(record)]
    if not leads:
        print(f"{label} NO LEAD: pipeline persisted no email lead (below threshold or deduped)")
        return {
            "prompt": prompt,
            "pattern": pattern,
            "type": expected_type,
            "min_score": min_score,
            "max_score": max_score,
            "metric": 0.0,
            "pass": False,
        }

    # Judge every persisted draft against its preserved source evidence, rather than
    # generated reasoning or an enrichment summary that might repeat an unsupported claim.
    judged = [ld for ld in leads if str(ld.get("email_body", "")).strip()]
    if not judged:
        print(f"{label} FAIL: lead persisted but no email body ({len(leads)} lead(s))")
        return {
            "prompt": prompt,
            "pattern": pattern,
            "type": expected_type,
            "min_score": min_score,
            "max_score": max_score,
            "metric": 0.0,
            "pass": False,
        }

    rel_scores: list[int] = []
    grd_scores: list[int] = []
    for lead in judged:
        full = _full_lead_record(region, lead)
        evidence = str(full.get("evidence_json", "")).strip()
        scores = _judge_email(bedrock_client, prompt, str(full.get("email_body", "")), evidence)
        rel_scores.append(scores["relevance"])
        grd_scores.append(scores["grounding"])

    avg_rel = sum(rel_scores) / len(rel_scores)
    avg_grd = sum(grd_scores) / len(grd_scores)
    # An email passes only when it is both relevant AND grounded: a fluent but
    # unsupported email should not pass. max_score (when set) still bounds relevance.
    relevance_ok = avg_rel >= min_score and (max_score is None or avg_rel <= max_score)
    grounding_ok = avg_grd >= _EMAIL_GROUNDING_MIN
    passed = relevance_ok and grounding_ok
    expected = f"[{min_score}, {max_score}]" if max_score is not None else f">= {min_score}"
    print(
        f"{label} {'PASS' if passed else 'FAIL'} relevance={avg_rel:.1f} (expected {expected}) "
        f"grounding={avg_grd:.1f} (min {_EMAIL_GROUNDING_MIN})"
    )
    return {
        "prompt": prompt,
        "pattern": pattern,
        "type": expected_type,
        "min_score": min_score,
        "max_score": max_score,
        "metric": avg_rel,
        "grounding": avg_grd,
        "pass": passed,
    }


def _eval_patterns(value: str) -> list[str]:
    """Validate EVAL_PATTERN and expand ``both`` into its concrete runtime patterns."""
    pattern = value.strip().lower()
    if pattern not in _VALID_EVAL_PATTERNS:
        allowed = ", ".join(sorted(_VALID_EVAL_PATTERNS))
        raise ValueError(f"EVAL_PATTERN must be one of: {allowed}; got {value!r}")
    return ["graph", "swarm"] if pattern == "both" else [pattern]


def run_live(max_prompts: int | None = None, start_at: int = 1) -> bool:
    """Run live evaluation against the deployed AgentCore runtime.

    Prompts run concurrently (bounded by EVAL_CONCURRENCY, default 5). Each uses a
    unique run_id so its persisted leads are attributable even though all prompts
    write to the same table at once, cutting wall-clock from the serial sum to roughly
    the slowest single prompt.

    Returns:
        True if all golden prompts met their threshold, False otherwise.

    Args:
        max_prompts: Optional positive cap for a bounded smoke evaluation.
        start_at: One-based golden-set entry to start from. Used with ``max_prompts``
            to target a single scenario without running earlier, unrelated prompts.
    """
    arn = os.environ.get("AGENTCORE_AGENT_ARN", "")
    if not arn:
        sys.exit("ERROR: AGENTCORE_AGENT_ARN env var is required for --live mode")

    import boto3
    from botocore.config import Config

    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    concurrency = int(os.environ.get("EVAL_CONCURRENCY", "5"))
    # Which orchestration pattern(s) to evaluate: "graph" (default), "swarm", or "both".
    try:
        patterns = _eval_patterns(os.environ.get("EVAL_PATTERN", "graph"))
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")
    # Adaptive retries let botocore back off on Bedrock throttling automatically, and a
    # larger pool serves the concurrent judge/invoke calls without connection contention.
    bedrock_client = boto3.client(
        "bedrock-runtime",
        region_name=region,
        config=Config(
            retries={"total_max_attempts": 3, "mode": "adaptive"},
            max_pool_connections=max(10, concurrency * 2),
        ),
    )

    print("=" * 72)
    print("LIVE EVAL — social intelligence quality harness")
    print(f"ARN: {arn[:60]}...  concurrency={concurrency}  patterns={patterns}")
    print("=" * 72)

    indexed_items = list(enumerate(_load_golden_set()))
    errors = _validate_golden_set([item for _, item in indexed_items])
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return False
    selected_items = indexed_items[start_at - 1 :]
    if max_prompts is not None:
        selected_items = selected_items[:max_prompts]
    if not selected_items:
        print(f"ERROR: --start {start_at} is outside the golden set")
        return False

    # One work item per (pattern, golden prompt). Runs concurrently; collect results
    # indexed so the summary stays ordered and grouped by pattern.
    work = [(pattern, idx, item) for pattern in patterns for idx, item in selected_items]
    indexed: list[tuple[int, dict[str, Any]]] = []
    cancellation_event = threading.Event()
    pool = ThreadPoolExecutor(max_workers=concurrency)
    futures = {}
    try:
        futures = {
            pool.submit(_evaluate_prompt, item, idx, arn, region, bedrock_client, pattern, cancellation_event): order
            for order, (pattern, idx, item) in enumerate(work)
        }
        for future in as_completed(futures):
            order = futures[future]
            pattern, idx, item = work[order]
            try:
                indexed.append((order, future.result()))
            except Exception as exc:  # a single prompt failing must not sink the run
                logger.warning("prompt %d (%s) evaluation errored: %s", idx + 1, pattern, exc)
                indexed.append(
                    (
                        order,
                        {
                            "prompt": item["prompt"],
                            "pattern": pattern,
                            "type": item["expected_type"],
                            "min_score": item["min_score"],
                            "max_score": item.get("max_score"),
                            "metric": 0.0,
                            "pass": False,
                        },
                    )
                )
    except KeyboardInterrupt:
        cancellation_event.set()
        for future in futures:
            future.cancel()
        pool.shutdown(wait=False, cancel_futures=True)
        print("\nLive evaluation interrupted; pending runs were canceled.")
        raise
    else:
        pool.shutdown(wait=True)
    results = [r for _, r in sorted(indexed, key=lambda pair: pair[0])]

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'#':<4} {'Pass':<5} {'Pattern':<7} {'Type':<16} {'Metric':<8} {'Expected':<10} {'Prompt':<22}")
    print("-" * 80)
    all_pass = True
    for idx, r in enumerate(results):
        marker = "Y" if r["pass"] else "N"
        if not r["pass"]:
            all_pass = False
        metric_label = f"{r['metric']:.1f}"
        expected = f"{r['min_score']}-{r['max_score']}" if r.get("max_score") is not None else f">={r['min_score']}"
        pat = str(r.get("pattern", "graph"))
        print(f"{idx + 1:<4} {marker:<5} {pat:<7} {r['type']:<16} {metric_label:<8} {expected:<10} {r['prompt'][:21]}")
    print("=" * 80)
    passed_count = sum(1 for r in results if r["pass"])
    # Per-pattern tally so a mixed-pattern run reports each pattern's pass rate.
    for pat in sorted({str(r.get("pattern", "graph")) for r in results}):
        pat_results = [r for r in results if str(r.get("pattern", "graph")) == pat]
        pat_passed = sum(1 for r in pat_results if r["pass"])
        print(f"  {pat}: {pat_passed}/{len(pat_results)} met threshold")
    print(f"Result: {passed_count}/{len(results)} prompts met threshold")
    print("  (email_draft metric = judged relevance 1-10; scored_prospect metric = top score 0-100)")
    return all_pass


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse args and dispatch to offline or live eval mode."""
    # Line-buffer stdout so per-prompt progress appears immediately when the output
    # is redirected to a file or a pipe (CI logs). The live run takes minutes per
    # prompt; block buffering would otherwise hide all progress until the process exits.
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(
        description="Eval harness for the social intelligence lead-gen pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--offline", action="store_true", help="Validate golden set + run grounding gate without AWS")
    group.add_argument(
        "--live", action="store_true", help="Invoke deployed runtime + LLM-as-judge (needs AGENTCORE_AGENT_ARN)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Evaluate only the first N golden prompts in live mode (bounded smoke test).",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="One-based golden-set entry to start from in live mode (default: 1).",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.start <= 0:
        parser.error("--start must be positive")
    if args.offline and args.limit is not None:
        parser.error("--limit is supported only with --live")
    if args.offline and args.start != 1:
        parser.error("--start is supported only with --live")

    try:
        ok = run_offline() if args.offline else run_live(args.limit, args.start)
    except KeyboardInterrupt:
        return
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
