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
                "value_prop": "AI-powered social intelligence for smarter outreach",
                "cta_style": "Low-friction: suggest a 15-minute call or async demo",
                "avoid": ["Salesy language", "Generic templates", "Exclamation marks"],
            },
        }
    )
