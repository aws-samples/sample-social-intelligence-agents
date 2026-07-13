"""Shared configuration: single source of truth for model IDs, regions, and defaults."""

import os

AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

# Amazon Bedrock cross-region inference profile for the default agent model.
# The "us." prefix is the US cross-region profile; newer Claude models require this
# prefix (bare "anthropic.claude-..." IDs are not on-demand invocable). Use "global."
# for the global profile. Update this to an active profile in your account/region;
# retired ("Legacy") model IDs return ResourceNotFoundException at invoke time.
MODEL_ID = os.environ.get(
    "MODEL_ID",
    "us.anthropic.claude-sonnet-4-6",
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
    """Return BedrockModel guardrail kwargs when both GUARDRAIL_ID and GUARDRAIL_VERSION are set.

    Returns:
        Dict with guardrail_id, guardrail_version, and guardrail_trace keys when active,
        or an empty dict when either env var is unset. Passing an empty dict to
        BedrockModel(**kwargs) is a no-op, preserving default (no-guardrail) behavior.
    """
    if GUARDRAIL_ID and GUARDRAIL_VERSION:
        return {
            "guardrail_id": GUARDRAIL_ID,
            "guardrail_version": GUARDRAIL_VERSION,
            "guardrail_trace": "enabled",
        }
    return {}
