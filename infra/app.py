"""CDK app entry point for the social intelligence infrastructure."""

import os

import aws_cdk as cdk
import cdk_nag
from stacks.social_intelligence_stack import SocialIntelligenceStack

app = cdk.App()

# Add AWS Solutions cdk-nag checks
cdk.Aspects.of(app).add(cdk_nag.AwsSolutionsChecks(verbose=True))

SocialIntelligenceStack(
    app,
    "SocialIntelligenceStack",
    env=cdk.Environment(
        region=os.environ.get("CDK_DEFAULT_REGION", "us-east-1"),
        account=os.environ["CDK_DEFAULT_ACCOUNT"],
    ),
)
app.synth()
