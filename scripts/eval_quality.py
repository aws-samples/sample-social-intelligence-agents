"""Eval harness for the social intelligence lead-gen pipeline.

Two modes:
  --offline  Validates golden_set.json structure and runs the grounding_gate
             structured fallback against synthetic email/evidence pairs.
             No AWS credentials required.
  --live     Invokes the deployed runtime (AGENTCORE_AGENT_ARN) for each golden
             prompt, parses scored prospects and email drafts from the SSE stream,
             then runs a cheap LLM-as-judge via Claude Haiku on Bedrock to score
             email relevance (1-10) and grounding quality. Requires AWS credentials
             and AGENTCORE_AGENT_ARN env var.

Usage:
  uv run python scripts/eval_quality.py --offline
  uv run python scripts/eval_quality.py --live

Output: a plain-text summary table printed to stdout.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import uuid
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
        if not isinstance(item.get("min_score"), (int, float)):
            errors.append(f"Item {idx}: 'min_score' must be a number")
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
    except (json.JSONDecodeError, TypeError, ValueError):
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

_HAIKU_MODEL = "us.anthropic.claude-haiku-4-5"
_JUDGE_PROMPT = """You are an email quality judge. Score the following outreach email on two dimensions.
Return ONLY a JSON object with keys "relevance" (int 1-10) and "grounding" (int 1-10).

Prompt that triggered the pipeline:
{prompt}

Email body:
{email_body}

Relevance: does the email address a genuine pain point suggested by the prompt?
(1=irrelevant, 10=highly targeted)
Grounding: are the specific claims believable and precise rather than vague?
(1=vague/generic, 10=specific and data-backed)
"""


def _invoke_agent(arn: str, prompt: str, region: str) -> list[dict[str, Any]]:
    """Invoke the AgentCore runtime and collect SSE events.

    Args:
        arn: AgentCore agent runtime ARN.
        prompt: User prompt to send.
        region: AWS region.

    Returns:
        List of parsed event dicts from the SSE stream.
    """
    import boto3

    client = boto3.client("bedrock-agentcore", region_name=region)
    session_id = str(uuid.uuid4())
    events: list[dict[str, Any]] = []
    try:
        response = client.invoke_agent_runtime(
            agentRuntimeArn=arn,
            runtimeSessionId=session_id,
            payload=json.dumps({"prompt": prompt, "pattern": "graph"}).encode(),
        )
        stream = response.get("response", b"")
        if isinstance(stream, bytes):
            for line in stream.decode().splitlines():
                if line.startswith("data:"):
                    try:
                        events.append(json.loads(line[5:].strip()))
                    except (json.JSONDecodeError, ValueError):
                        pass
    except Exception:
        logger.warning("invoke_agent_runtime failed", exc_info=True)
    return events


def _extract_emails_from_events(events: list[dict[str, Any]]) -> list[str]:
    """Extract email body strings from pipeline SSE events.

    Args:
        events: Parsed SSE event dicts.

    Returns:
        List of email body strings found in the events.
    """
    emails: list[str] = []
    for event in events:
        text = json.dumps(event)
        # Look for body fields in EmailDraft-like structures
        for match in re.finditer(r'"body"\s*:\s*"((?:[^"\\]|\\.)+)"', text):
            emails.append(match.group(1).replace("\\n", "\n"))
    return emails


def _judge_email(bedrock_client: Any, prompt: str, email_body: str) -> dict[str, int]:
    """Call Claude Haiku on Bedrock to judge email quality.

    Args:
        bedrock_client: boto3 bedrock-runtime client.
        prompt: Original pipeline prompt.
        email_body: Email body text to judge.

    Returns:
        Dict with 'relevance' and 'grounding' int scores.
    """
    judge_prompt = _JUDGE_PROMPT.format(prompt=prompt, email_body=email_body[:800])
    try:
        response = bedrock_client.converse(
            modelId=_HAIKU_MODEL,
            messages=[{"role": "user", "content": [{"text": judge_prompt}]}],
            inferenceConfig={"maxTokens": 100, "temperature": 0.0},
        )
        raw = response["output"]["message"]["content"][0]["text"]
        scores = json.loads(raw)
        return {
            "relevance": int(scores.get("relevance", 0)),
            "grounding": int(scores.get("grounding", 0)),
        }
    except Exception:
        logger.warning("LLM judge call failed", exc_info=True)
        return {"relevance": 0, "grounding": 0}


def run_live() -> bool:
    """Run live evaluation against the deployed AgentCore runtime.

    Returns:
        True if all golden prompts met their min_score, False otherwise.
    """
    arn = os.environ.get("AGENTCORE_AGENT_ARN", "")
    if not arn:
        sys.exit("ERROR: AGENTCORE_AGENT_ARN env var is required for --live mode")

    import boto3
    from botocore.config import Config

    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    # Adaptive retries let botocore back off on Bedrock throttling automatically,
    # so the judge loop needs no manual sleep pacing between calls.
    bedrock_client = boto3.client(
        "bedrock-runtime",
        region_name=region,
        config=Config(retries={"max_attempts": 5, "mode": "adaptive"}),
    )

    print("=" * 60)
    print("LIVE EVAL — social intelligence quality harness")
    print(f"ARN: {arn[:60]}...")
    print("=" * 60)

    items = _load_golden_set()
    results: list[dict[str, Any]] = []

    for idx, item in enumerate(items):
        prompt = item["prompt"]
        min_score = item["min_score"]
        print(f"\n[{idx + 1}/{len(items)}] {prompt[:70]}...")

        events = _invoke_agent(arn, prompt, region)
        emails = _extract_emails_from_events(events)

        if not emails:
            print("  WARNING: no emails found in response")
            results.append(
                {"prompt": prompt, "min_score": min_score, "avg_relevance": 0, "avg_grounding": 0, "pass": False}
            )
            continue

        rel_scores: list[int] = []
        grd_scores: list[int] = []
        for email_body in emails[:2]:  # judge up to 2 emails per prompt
            scores = _judge_email(bedrock_client, prompt, email_body)
            rel_scores.append(scores["relevance"])
            grd_scores.append(scores["grounding"])

        avg_rel = sum(rel_scores) / len(rel_scores)
        avg_grd = sum(grd_scores) / len(grd_scores)
        passed = avg_rel >= min_score
        status = "PASS" if passed else "FAIL"
        print(f"  {status}  relevance={avg_rel:.1f} grounding={avg_grd:.1f} (min={min_score})")
        results.append(
            {
                "prompt": prompt,
                "min_score": min_score,
                "avg_relevance": avg_rel,
                "avg_grounding": avg_grd,
                "pass": passed,
            }
        )

    # Summary table
    print("\n" + "=" * 60)
    print(f"{'#':<4} {'Pass':<5} {'Rel':<5} {'Grd':<5} {'Prompt':<45}")
    print("-" * 60)
    all_pass = True
    for idx, r in enumerate(results):
        marker = "Y" if r["pass"] else "N"
        if not r["pass"]:
            all_pass = False
        print(f"{idx + 1:<4} {marker:<5} {r['avg_relevance']:<5.1f} {r['avg_grounding']:<5.1f} {r['prompt'][:44]}")
    print("=" * 60)
    passed_count = sum(1 for r in results if r["pass"])
    print(f"Result: {passed_count}/{len(results)} prompts met min_score threshold")
    return all_pass


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse args and dispatch to offline or live eval mode."""
    parser = argparse.ArgumentParser(
        description="Eval harness for the social intelligence lead-gen pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--offline", action="store_true", help="Validate golden set + run grounding gate without AWS")
    group.add_argument(
        "--live", action="store_true", help="Invoke deployed runtime + LLM-as-judge (needs AGENTCORE_AGENT_ARN)"
    )
    args = parser.parse_args()

    if args.offline:
        ok = run_offline()
        sys.exit(0 if ok else 1)
    else:
        ok = run_live()
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
