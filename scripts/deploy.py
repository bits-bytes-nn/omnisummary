#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aws_cdk import App, Environment

from infrastructure.application_stack import OmniSummaryApplicationStack
from infrastructure.foundation_stack import OmniSummaryFoundationStack
from shared import Config


def main():
    config = Config.load()

    # The account the CDK CLI resolved from ITS credentials (the ones it deploys with), not a second
    # STS call through config.aws.profile, which can name a different account.
    account = os.environ.get("CDK_DEFAULT_ACCOUNT")
    if not account:
        raise SystemExit("CDK_DEFAULT_ACCOUNT is unset: run this through the CDK CLI (npx cdk deploy -a ...)")
    env = Environment(account=account, region=config.aws.region)

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
        digest_image_ref=os.getenv("DIGEST_IMAGE_REF", ""),
        env=env,
    )

    app.synth()


if __name__ == "__main__":
    main()
