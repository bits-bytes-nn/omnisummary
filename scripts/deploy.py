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
        # No secrets are passed in: a CloudFormation template cannot hold a SecureString, so the
        # values would land in plaintext in cdk.out, the CDK staging bucket and GetTemplate. The
        # stack creates the parameters holding a placeholder; run scripts/put_secrets.py after the
        # deploy to write the real values as SecureStrings.
        agentcore_image_ref=os.getenv("AGENTCORE_IMAGE_REF", ""),
        digest_image_ref=os.getenv("DIGEST_IMAGE_REF", ""),
        env=env,
    )

    app.synth()


if __name__ == "__main__":
    main()
