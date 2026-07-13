"""Pydantic models for structured data flow between agents.

These models serve dual purposes:
1. Inter-agent data contracts validated at handoff boundaries
2. Strands structured_output_model for type-safe agent responses
"""

from pydantic import BaseModel, Field


class TrendItem(BaseModel):
    """A single trend data point from any social source."""

    source: str = Field(
        description="Data source: hackernews, youtube, devto, "
        "github, lobsters, wikipedia, reddit, stackoverflow, producthunt",
    )
    topic: str = Field(description="Trend topic or title")
    engagement: int = Field(default=0, description="Engagement metric (views, likes, score)")
    url: str = Field(default="", description="Source URL")
    timestamp: str = Field(default="", description="ISO timestamp for temporal decay weighting")
    intent_signals: list[str] = Field(
        default_factory=list,
        description="Detected intent signals: recommendation_seeking, "
        "competitor_frustration, product_launch, purchase_intent",
    )
    freshness_weight: float = Field(
        default=1.0,
        description="Temporal decay weight from source timestamp: 1.5 <24h, 1.2 <72h, 1.0 <168h, 0.5 older",
    )


class ProspectProfile(BaseModel):
    """A prospect discovered from Hacker News or other tech communities."""

    prospect_id: str = Field(description="Unique identifier (e.g. HN story ID)")
    product_name: str = Field(description="Product or project name")
    tagline: str = Field(default="")
    author: str = Field(default="", description="Author or maker username")
    website: str = Field(default="")
    category: str = Field(default="")
    score: int = Field(default=0, description="Community score or upvotes")


class TrendData(BaseModel):
    """Output of the Trend Research Agent: structured output model."""

    prospects: list[ProspectProfile] = Field(default_factory=list, description="Discovered prospects")
    trends: list[TrendItem] = Field(default_factory=list, description="Collected trend signals")
    signal_strength: str = Field(
        default="moderate",
        description="Overall multi-signal strength: strong, moderate, weak",
    )


class EnrichmentArticle(BaseModel):
    """A single enrichment article from web search."""

    title: str
    url: str
    snippet: str = ""
    source: str = Field(default="", description="Source: wikipedia, github, lobsters, stackoverflow")


class EnrichmentData(BaseModel):
    """Output of the Search Specialist Agent: structured output model."""

    articles: list[EnrichmentArticle] = Field(default_factory=list)
    competitors: list[str] = Field(default_factory=list, description="Top 2-3 competitors identified")
    key_talking_points: list[str] = Field(default_factory=list, description="Key talking points for outreach")


class ScoredProspect(BaseModel):
    """A single scored prospect from the Analysis Agent."""

    prospect_id: str = Field(description="Prospect identifier from TrendData")
    product_name: str = Field(default="", description="Product or project name")
    score: int = Field(ge=0, le=100, description="Relevance score 0-100")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the score")
    reasoning: str = Field(description="2-3 sentence explanation of the score")
    top_trends: list[str] = Field(default_factory=list, description="Most relevant trends for email personalization")
    intent_signals: list[str] = Field(default_factory=list, description="Detected buying intent signals")
    icp_fit: str = Field(default="medium", description="ICP fit: strong, medium, or weak")
    data_quality: str = Field(default="medium", description="Assessment: high, medium, or low")


class ScoredProspectList(BaseModel):
    """Output of the Analysis Agent: list of scored prospects."""

    prospects: list[ScoredProspect] = Field(
        description="All scored prospects. Include every prospect from research, not just the top one."
    )


class EmailDraft(BaseModel):
    """A single email draft for one prospect."""

    prospect_id: str = Field(description="Prospect identifier")
    subject: str = Field(description="Email subject line")
    body: str = Field(description="Email body text, under 150 words")
    personalization_tokens: list[str] = Field(
        default_factory=list,
        description="Specific data points referenced in the email (e.g. 'HN score: 342')",
    )
    brand_compliant: bool = Field(default=True, description="Whether email passes brand guidelines check")


class EmailDraftList(BaseModel):
    """Output of the Email Generation Agent: list of email drafts."""

    drafts: list[EmailDraft] = Field(
        description="One email draft per qualified prospect (score >= 60). Call store_lead for each."
    )
