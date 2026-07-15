"""Shared configuration: single source of truth for model IDs, regions, and defaults."""

import logging
import os
from collections.abc import Mapping

from botocore.config import Config

logger = logging.getLogger(__name__)


def resolve_aws_region(environment: Mapping[str, str] | None = None) -> str:
    """Resolve the AWS Region using the standard explicit-region precedence."""
    source = environment if environment is not None else os.environ
    return source.get("AWS_REGION", "").strip() or source.get("AWS_DEFAULT_REGION", "").strip() or "us-east-1"


def _bounded_int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    """Read a bounded integer environment variable, falling back to a safe default."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; using %d", name, raw, default)
        return default
    if not minimum <= value <= maximum:
        logger.warning("%s=%d is outside [%d, %d]; using %d", name, value, minimum, maximum, default)
        return default
    return value


AWS_REGION = resolve_aws_region()

# An event-stream read timeout is the maximum idle period between Bedrock response
# chunks, not the total generation time. The node timeout below reserves enough time
# for all three bounded attempts plus retry backoff and orchestration cleanup.
BEDROCK_READ_TIMEOUT_SECONDS = _bounded_int_env("BEDROCK_READ_TIMEOUT_SECONDS", 300, 30, 360)

# These values are enforced by the orchestration and persistence boundaries. Keeping
# them here also lets agent instructions match the configured runtime behavior.
EMAIL_SCORE_THRESHOLD = _bounded_int_env("EMAIL_SCORE_THRESHOLD", 60, 0, 100)
MIN_INDEPENDENT_SOURCES = _bounded_int_env("MIN_INDEPENDENT_SOURCES", 2, 1, 9)
MAX_LEADS_PER_RUN = _bounded_int_env("MAX_LEADS_PER_RUN", 3, 1, 50)
# Each agent must be able to return its bounded, typed contract in a single response.
# The prompts and Pydantic models cap record counts and field sizes; these ceilings
# leave sufficient room for a complete response without permitting unbounded retries.
TREND_MAX_TOKENS = _bounded_int_env("TREND_MAX_TOKENS", 8192, 1024, 32768)
SEARCH_MAX_TOKENS = _bounded_int_env("SEARCH_MAX_TOKENS", 8192, 1024, 32768)
# The analysis agent emits the largest structured output of the four: a ScoredProspectList
# of up to five prospects, each with a six-field score_breakdown, reasoning, an enrichment
# summary, and up to four evidence items. A 4096 ceiling truncated that payload mid-generation
# and raised MaxTokensReachedException, so keep it at 8192 to fit the full typed contract.
ANALYSIS_MAX_TOKENS = _bounded_int_env("ANALYSIS_MAX_TOKENS", 8192, 1024, 32768)

# Graph and Swarm share the same three-attempt transient model retry policy. With the
# default 300-second event-stream read timeout, 960 seconds reserves three socket-read
# attempts, exponential retry backoff, and orchestration/persistence cleanup. Raise this
# automatically if an operator increases the read timeout so the retry contract cannot
# be silently invalidated.
_MIN_NODE_TIMEOUT_SECONDS = BEDROCK_READ_TIMEOUT_SECONDS * 3 + 60
ORCHESTRATION_NODE_TIMEOUT_SECONDS = max(
    _bounded_int_env("ORCHESTRATION_NODE_TIMEOUT_SECONDS", 960, 120, 1140),
    _MIN_NODE_TIMEOUT_SECONDS,
)


def bedrock_boto_config() -> Config:
    """Return the botocore Config applied to every agent's BedrockModel client.

    Returns:
        A Config with a long read timeout (BEDROCK_READ_TIMEOUT_SECONDS) so lengthy
        structured-output generations do not time out, plus adaptive retries for
        automatic backoff on Bedrock throttling.
    """
    return Config(
        read_timeout=BEDROCK_READ_TIMEOUT_SECONDS,
        connect_timeout=10,
        # Streaming read failures happen after a response begins, where botocore cannot
        # safely retry. Agent hooks own the complete three-attempt retry budget, so each
        # HTTP attempt is single-shot here and retries are never compounded.
        retries={"total_max_attempts": 1, "mode": "adaptive"},
    )


# Amazon Bedrock cross-region inference profile for the default agent model.
# The default uses the "global." profile, which resolves in every supported Region
# (US and EU) so the sample works out of the box wherever it is deployed. Newer
# Claude models require an inference-profile prefix; bare "anthropic.claude-..." IDs
# are not on-demand invocable. Region-scoped profiles are also valid overrides:
# "us." in US Regions, "eu." in eu-west-1. Set MODEL_ID to pin a specific profile.
# Retired ("Legacy") model IDs return ResourceNotFoundException at invoke time.
MODEL_ID = os.environ.get(
    "MODEL_ID",
    "global.anthropic.claude-sonnet-4-6",
)

# Optional per-agent model overrides. All default to MODEL_ID, so the out-of-the-box
# behavior matches the blog (every agent on Claude Sonnet 4.6). Operators who want to
# tier models for cost (for example a cheaper model for discovery/enrichment triage and
# the default for email synthesis). Set these env vars without touching code:
#   TREND_MODEL_ID, SEARCH_MODEL_ID, ANALYSIS_MODEL_ID, EMAIL_MODEL_ID
TREND_MODEL_ID = os.environ.get("TREND_MODEL_ID", MODEL_ID)
SEARCH_MODEL_ID = os.environ.get("SEARCH_MODEL_ID", MODEL_ID)
ANALYSIS_MODEL_ID = os.environ.get("ANALYSIS_MODEL_ID", MODEL_ID)
EMAIL_MODEL_ID = os.environ.get("EMAIL_MODEL_ID", MODEL_ID)

# Optional Bedrock Guardrail: both ID and version must be set to activate.
# When unset (default), no guardrail kwargs are passed to BedrockModel and blog behavior
# is unchanged. Operators deploy a Bedrock Guardrail with a contextual-grounding policy
# and set these vars to enable output filtering across all four agents.
GUARDRAIL_ID = os.environ.get("GUARDRAIL_ID", "")
GUARDRAIL_VERSION = os.environ.get("GUARDRAIL_VERSION", "")


def guardrail_kwargs() -> dict:
    """Return the public Strands guardrail kwargs when both values are configured.

    Strands translates these names to the Converse ``guardrailConfig`` object.
    Synchronous processing and output redaction keep policy enforcement in the model
    request, avoiding separate ``ApplyGuardrail`` calls and their independent quota.
    It deliberately sends no ``guardContent`` blocks, so Bedrock evaluates every
    message and tool result; adding one would scope assessment to only those marked
    blocks.

    Returns:
        Dict of public Strands ``BedrockModel`` guardrail options when active, or an
        empty dict when either env var is unset. Passing an empty dict to
        ``BedrockModel(**kwargs)`` is a no-op, preserving default behavior.
    """
    if GUARDRAIL_ID and GUARDRAIL_VERSION:
        return {
            "guardrail_id": GUARDRAIL_ID,
            "guardrail_version": GUARDRAIL_VERSION,
            "guardrail_trace": "enabled",
            "guardrail_stream_processing_mode": "sync",
            "guardrail_redact_input": True,
            "guardrail_redact_output": True,
            "guardrail_redact_output_message": "[Assistant response blocked by Bedrock Guardrails.]",
        }
    return {}
