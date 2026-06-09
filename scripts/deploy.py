#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import boto3
from aws_cdk import App, Environment

from infrastructure.application_stack import OmniSummaryApplicationStack
from infrastructure.foundation_stack import OmniSummaryFoundationStack
from shared import Config


def main():
    config = Config.load()

    boto_session = boto3.Session(
        region_name=config.aws.region,
        profile_name=config.aws.profile or None,
    )
    account_id = boto_session.client("sts").get_caller_identity()["Account"]

    env = Environment(account=account_id, region=config.aws.region)

    app = App()

    foundation = OmniSummaryFoundationStack(
        app,
        f"{config.aws.project_name}-{config.aws.stage}-foundation",
        config=config,
        alert_email=os.getenv("ALERT_EMAIL", ""),
        env=env,
    )

    OmniSummaryApplicationStack(
        app,
        f"{config.aws.project_name}-{config.aws.stage}-application",
        config=config,
        foundation=foundation,
        slack_signing_secret=os.getenv("SLACK_SIGNING_SECRET", ""),
        slack_bot_token=os.getenv("SLACK_BOT_TOKEN", ""),
        slack_channel_id=os.getenv("SLACK_CHANNEL_ID", ""),
        tavily_api_key=os.getenv("TAVILY_API_KEY", ""),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        youtube_api_key=os.getenv("YOUTUBE_API_KEY", ""),
        threads_access_token=os.getenv("THREADS_ACCESS_TOKEN", ""),
        threads_user_id=os.getenv("THREADS_USER_ID", ""),
        agentcore_image_ref=os.getenv("AGENTCORE_IMAGE_REF", ""),
        digest_image_ref=os.getenv("DIGEST_IMAGE_REF", ""),
        env=env,
    )

    app.synth()


if __name__ == "__main__":
    main()
