"""Tests for shared runtime configuration resolution."""

import logging
from unittest.mock import MagicMock

from strands.models import BedrockModel

from social_intelligence import config
from social_intelligence.config import (
    ANALYSIS_MAX_TOKENS,
    ORCHESTRATION_NODE_TIMEOUT_SECONDS,
    SEARCH_MAX_TOKENS,
    TREND_MAX_TOKENS,
    bedrock_boto_config,
    bool_env,
    resolve_aws_region,
)


def test_aws_region_has_precedence_over_aws_default_region() -> None:
    assert resolve_aws_region({"AWS_REGION": "eu-west-1", "AWS_DEFAULT_REGION": "us-east-1"}) == "eu-west-1"


def test_aws_region_uses_default_then_safe_fallback() -> None:
    assert resolve_aws_region({"AWS_DEFAULT_REGION": "us-east-1"}) == "us-east-1"
    assert resolve_aws_region({}) == "us-east-1"


def test_bedrock_retry_policy_has_three_total_attempts() -> None:
    assert bedrock_boto_config().retries == {"total_max_attempts": 1, "mode": "adaptive"}


def test_guardrail_kwargs_match_the_strands_converse_adapter_contract(monkeypatch) -> None:
    monkeypatch.setattr(config, "GUARDRAIL_ID", "guardrail-id")
    monkeypatch.setattr(config, "GUARDRAIL_VERSION", "7")
    assert config.guardrail_kwargs() == {
        "guardrail_id": "guardrail-id",
        "guardrail_version": "7",
        "guardrail_trace": "enabled",
        "guardrail_stream_processing_mode": "sync",
        "guardrail_redact_input": True,
        "guardrail_redact_output": True,
        "guardrail_redact_output_message": "[Assistant response blocked by Bedrock Guardrails.]",
    }

    monkeypatch.setattr(config, "GUARDRAIL_VERSION", "")
    assert config.guardrail_kwargs() == {}


def test_guardrail_kwargs_generate_a_single_inline_converse_configuration(monkeypatch) -> None:
    monkeypatch.setattr(config, "GUARDRAIL_ID", "guardrail-id")
    monkeypatch.setattr(config, "GUARDRAIL_VERSION", "7")
    session = MagicMock()
    session.region_name = "eu-west-1"

    request = BedrockModel(
        boto_session=session, model_id="global.anthropic.claude-sonnet-4-6", **config.guardrail_kwargs()
    ).format_request([{"role": "user", "content": [{"text": "Find developer-tool prospects."}]}])

    assert request["guardrailConfig"] == {
        "guardrailIdentifier": "guardrail-id",
        "guardrailVersion": "7",
        "trace": "enabled",
        "streamProcessingMode": "sync",
    }


def test_bool_env_treats_falsy_spellings_as_off(monkeypatch) -> None:
    # bool("false") is True, so a flag read with bool() would turn on when an operator
    # explicitly disables it. Every falsy spelling must stay off.
    for value in ("false", "False", "0", "no", " FALSE ", ""):
        monkeypatch.setenv("SOCIAL_INTEL_TEST_FLAG", value)
        assert bool_env("SOCIAL_INTEL_TEST_FLAG") is False, value

    monkeypatch.delenv("SOCIAL_INTEL_TEST_FLAG")
    assert bool_env("SOCIAL_INTEL_TEST_FLAG") is False


def test_bool_env_accepts_the_documented_truthy_spellings(monkeypatch) -> None:
    for value in ("true", "TRUE", "1", "yes", " Yes "):
        monkeypatch.setenv("SOCIAL_INTEL_TEST_FLAG", value)
        assert bool_env("SOCIAL_INTEL_TEST_FLAG") is True, value


def test_bool_env_warns_and_stays_off_for_unrecognized_values(monkeypatch, caplog) -> None:
    monkeypatch.setenv("SOCIAL_INTEL_TEST_FLAG", "maybe")
    with caplog.at_level(logging.WARNING, logger=config.__name__):
        assert bool_env("SOCIAL_INTEL_TEST_FLAG") is False
    assert "is not a boolean" in caplog.text


def test_analysis_output_budget_has_a_safe_default() -> None:
    # The analysis agent emits the largest structured output (ScoredProspectList with a
    # per-prospect score_breakdown). 4096 truncated it and raised MaxTokensReachedException,
    # so the default must leave room for the full typed contract.
    assert ANALYSIS_MAX_TOKENS == 8192
    assert SEARCH_MAX_TOKENS == 8192
    assert TREND_MAX_TOKENS == 8192
    assert ORCHESTRATION_NODE_TIMEOUT_SECONDS == 960
