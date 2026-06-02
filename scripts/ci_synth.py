#!/usr/bin/env python3
"""Offline CDK synth for CI: uses a dummy account so no AWS credentials are required."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aws_cdk import App, Environment

from infrastructure.application_stack import OmniSummaryApplicationStack
from infrastructure.foundation_stack import OmniSummaryFoundationStack
from shared import Config


def main() -> None:
    config = Config.load()
    env = Environment(account="123456789012", region=config.aws.region)
    app = App()

    foundation = OmniSummaryFoundationStack(
        app,
        f"{config.aws.project_name}-{config.aws.stage}-foundation",
        config=config,
        alert_email=os.getenv("ALERT_EMAIL", "ci@example.com"),
        env=env,
    )
    OmniSummaryApplicationStack(
        app,
        f"{config.aws.project_name}-{config.aws.stage}-application",
        config=config,
        foundation=foundation,
        slack_signing_secret=os.getenv("SLACK_SIGNING_SECRET", "ci"),
        slack_bot_token=os.getenv("SLACK_BOT_TOKEN", "ci"),
        tavily_api_key=os.getenv("TAVILY_API_KEY", "ci"),
        openai_api_key=os.getenv("OPENAI_API_KEY", "ci"),
        env=env,
    )
    app.synth()
    print("CDK synth succeeded (offline CI mode)")


if __name__ == "__main__":
    main()
