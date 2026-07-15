"""Brand knowledge base tool for email personalization.

Returns AnyCompany brand guidelines. In production, this could read from Amazon S3
or a vector database. For the sample, it returns inline brand knowledge.

This is an agent-side tool: it runs in the agent process, not behind
the Amazon API Gateway/AWS Lambda, because it doesn't call external APIs.
"""

import json

from strands import tool


@tool
def retrieve_brand_knowledge(topic: str = "general") -> str:
    """Retrieve brand knowledge base content for email personalization.

    Args:
        topic: Knowledge area ('general', 'tone', 'value_prop', or 'guidelines').

    Returns:
        JSON string with brand guidelines and messaging specifications.
    """
    return json.dumps(
        {
            "topic": topic,
            "content": {
                "brand_name": "AnyCompany",
                "tone": "Professional, warm, concise. No jargon. Be genuine.",
                "value_prop": (
                    "Use public launch and community signals to prioritize concrete "
                    "sales and developer-relations follow-up"
                ),
                "concrete_outcomes": [
                    "rank launch comments and discussion threads for developer-relations follow-up",
                    "prioritize public replies that indicate product adoption questions",
                    "route high-signal community conversations to the appropriate outreach owner",
                ],
                "cta_style": "Low-friction: suggest a 15-minute call or async demo",
                "avoid": [
                    "Salesy language",
                    "Generic templates",
                    "Exclamation marks",
                    "Unverified claims about the prospect's internal operations",
                    "Abstract claims about social intelligence or buyer-intent signals without a concrete action",
                ],
            },
        }
    )
