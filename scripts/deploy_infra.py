import os
import sys
from pathlib import Path

import aws_cdk as core
import boto3

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from infrastructure import OmniSummaryApplicationStack, OmniSummaryFoundationStack
from shared import Config, EnvVars, get_account_id, logger


def main() -> None:
    try:
        config = Config.load()
        profile_name = os.environ.get(EnvVars.AWS_PROFILE_NAME.value)

        logger.info(
            "Deploying infrastructure for '%s' in '%s' stage",
            config.resources.project_name,
            config.resources.stage,
        )

        boto_session = boto3.Session(
            region_name=config.resources.default_region_name,
            profile_name=profile_name,
        )
        account_id = get_account_id(boto_session)

        env = core.Environment(
            account=account_id,
            region=config.resources.default_region_name,
        )

        app = core.App()

        base_name = f"OmniSummary{config.resources.stage.capitalize()}"

        foundation_stack = OmniSummaryFoundationStack(
            app,
            f"{base_name}FoundationStack",
            config=config,
            env=env,
        )

        slack_business_channel_ids_str = os.getenv(EnvVars.SLACK_BUSINESS_CHANNEL_IDS.value)
        slack_personal_channel_ids_str = os.getenv(EnvVars.SLACK_PERSONAL_CHANNEL_IDS.value)

        application_stack = OmniSummaryApplicationStack(
            app,
            f"{base_name}ApplicationStack",
            config=config,
            foundation_stack=foundation_stack,
            langchain_api_key=os.getenv(EnvVars.LANGCHAIN_API_KEY.value),
            slack_business_token=os.getenv(EnvVars.SLACK_BUSINESS_TOKEN.value),
            slack_business_channel_ids=(
                slack_business_channel_ids_str.split(",") if slack_business_channel_ids_str else None
            ),
            slack_personal_token=os.getenv(EnvVars.SLACK_PERSONAL_TOKEN.value),
            slack_personal_channel_ids=(
                slack_personal_channel_ids_str.split(",") if slack_personal_channel_ids_str else None
            ),
            slack_signing_secret=os.getenv(EnvVars.SLACK_SIGNING_SECRET.value),
            upstage_api_key=os.getenv(EnvVars.UPSTAGE_API_KEY.value),
            env=env,
        )
        application_stack.add_dependency(foundation_stack)

        app.synth()

        logger.info("Infrastructure deployment completed successfully")

    except Exception as e:
        logger.error("Error occurred during deployment: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
