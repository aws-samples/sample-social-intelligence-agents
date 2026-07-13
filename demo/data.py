"""DynamoDB data layer and AWS configuration checks for the Streamlit demo."""

import logging

import boto3
import streamlit as st
from botocore.exceptions import ClientError, ProfileNotFound
from config import AGENT_ARN, AWS_PROFILE, AWS_REGION, LEADS_TABLE

logger = logging.getLogger(__name__)


def get_boto_session():
    """Create a boto3 session, falling back to default credentials if profile fails."""
    try:
        return boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
    except ProfileNotFound:
        logger.debug("AWS profile '%s' not found, using default credentials", AWS_PROFILE)
        return boto3.Session(region_name=AWS_REGION)


@st.cache_data(ttl=30, show_spinner=False)
def get_leads(limit: int = 50) -> list[dict]:
    """Fetch leads from Amazon DynamoDB, sorted by discovery date descending."""
    try:
        session = get_boto_session()
        table = session.resource("dynamodb", region_name=AWS_REGION).Table(LEADS_TABLE)
        leads: list[dict] = []
        scan_kwargs: dict = {"Limit": 100}  # page size
        while True:
            resp = table.scan(**scan_kwargs)
            leads.extend(resp.get("Items", []))
            if len(leads) >= limit * 3 or "LastEvaluatedKey" not in resp:
                break
            scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
        leads.sort(key=lambda x: x.get("discovered_at", ""), reverse=True)
        return leads[:limit]
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "ResourceNotFoundException":
            logger.warning("DynamoDB table '%s' not found. Deploy CDK stack first.", LEADS_TABLE)
        else:
            logger.warning("DynamoDB error: %s", e)
        return []
    except Exception as e:
        logger.warning("Failed to fetch leads: %s", e)
        return []


def get_existing_lead_ids() -> list[str]:
    """Return prospect_id values from DynamoDB for deduplication context."""
    leads = get_leads()
    return [lead.get("prospect_id", "") for lead in leads if lead.get("prospect_id")]


def check_config() -> list[str]:
    """Validate configuration and return a list of issues (empty = all good)."""
    issues = []
    if not AGENT_ARN:
        issues.append(
            "**AGENTCORE_AGENT_ARN** not set. Copy `RuntimeArn` from CDK output and add it to `.env` or export it."
        )
    try:
        session = get_boto_session()
        session.client("sts", region_name=AWS_REGION).get_caller_identity()
    except Exception as e:
        # Logs the boto exception type/message (e.g. "Unable to locate credentials"),
        # never a secret value. nosemgrep: the %s is an exception, not a credential.
        logger.debug("STS get_caller_identity failed: %s", e)  # nosemgrep: python-logger-credential-disclosure
        issues.append(
            f"**AWS credentials** not configured for profile `{AWS_PROFILE}`. "
            f"Run `aws configure` or check your profile."
        )
    return issues
