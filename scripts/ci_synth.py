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
    # Clear vpc_id so the foundation stack CREATES a VPC instead of Vpc.from_lookup, which needs
    # real credentials/context and can't run against the dummy CI account. This keeps `cdk synth`
    # (via the pinned CLI) fully offline while still exercising the CLI↔aws-cdk-lib schema handshake
    # — the exact break a bare in-process app.synth() misses.
    config.aws.vpc_id = ""
    # Env-agnostic (no concrete account) so creating a VPC uses CDK's dummy AZs (Fn::GetAZs / the
    # 2-AZ stub) instead of an STS-backed availability-zone lookup — keeps synth fully offline.
    env = Environment(region=config.aws.region)
    app = App()

    foundation = OmniSummaryFoundationStack(
        app,
        f"{config.aws.project_name}-{config.aws.stage}-foundation",
        config=config,
        alert_email=os.getenv("ALERT_EMAIL", "ci@example.com"),
        env=env,
    )
    # Pass dummy NON-EMPTY values for the SAME arg set as deploy.py (incl. a sha256 image ref) so
    # the synthesized resource graph MATCHES production — the conditional SSM-param loop and the
    # `sha256:`-prefix image-pin branch (the exact deploy mechanism) are then exercised in CI, not
    # only on a real deploy.
    OmniSummaryApplicationStack(
        app,
        f"{config.aws.project_name}-{config.aws.stage}-application",
        config=config,
        foundation=foundation,
        slack_signing_secret=os.getenv("SLACK_SIGNING_SECRET", "ci"),
        slack_bot_token=os.getenv("SLACK_BOT_TOKEN", "ci"),
        slack_channel_id=os.getenv("SLACK_CHANNEL_ID", "ci"),
        tavily_api_key=os.getenv("TAVILY_API_KEY", "ci"),
        openai_api_key=os.getenv("OPENAI_API_KEY", "ci"),
        youtube_api_key=os.getenv("YOUTUBE_API_KEY", "ci"),
        threads_access_token=os.getenv("THREADS_ACCESS_TOKEN", "ci"),
        threads_user_id=os.getenv("THREADS_USER_ID", "ci"),
        agentcore_image_ref=os.getenv("AGENTCORE_IMAGE_REF", "sha256:" + "0" * 64),
        digest_image_ref=os.getenv("DIGEST_IMAGE_REF", "sha256:" + "0" * 64),
        env=env,
    )
    app.synth()
    print("CDK synth succeeded (offline CI mode)")


if __name__ == "__main__":
    main()
