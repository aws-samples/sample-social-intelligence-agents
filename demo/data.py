"""DynamoDB data layer and AWS configuration checks for the Streamlit demo."""

import logging

import boto3
import streamlit as st
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError, ProfileNotFound
from config import AGENT_ARN, AWS_PROFILE, AWS_REGION, LEADS_TABLE

logger = logging.getLogger(__name__)


def normalize_trends(value) -> list[str]:
    """Return top_trends as a list of non-empty strings.

    store_lead persists top_trends as a list, but older records (and manual writes)
    may hold a comma-separated string. Accept both so lead rendering never crashes on
    a list (str.split would raise) or shows list syntax as badge text.

    Args:
        value: The raw top_trends value from a lead record (list, str, or None).

    Returns:
        A list of trimmed, non-empty trend strings.
    """
    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        items = value.split(",")
    else:
        return []
    return [str(item).strip() for item in items if str(item).strip()]


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
        capped_limit = max(1, min(int(limit), 50))
        # The GSI returns the newest lead keys in order. Fetch the full records by
        # primary key because the dashboard renders fields not projected by the index.
        index_items = table.query(
            IndexName="dedup-partition-discovered-at-index",
            KeyConditionExpression=Key("dedup_partition").eq("LEAD"),
            ScanIndexForward=False,
            Limit=capped_limit,
        ).get("Items", [])
        leads: list[dict] = []
        for item in index_items:
            lead = table.get_item(Key={"prospect_id": item["prospect_id"], "discovered_at": item["discovered_at"]}).get(
                "Item"
            )
            if lead:
                leads.append(lead)
        return leads
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
