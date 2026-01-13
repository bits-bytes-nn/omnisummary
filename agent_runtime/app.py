import os
from typing import Any

import boto3
from bedrock_agentcore.runtime import BedrockAgentCoreApp

from agent import create_summarization_agent, tool_state_context
from shared import EnvVars, SSMParams, get_ssm_param_value, logger

app = BedrockAgentCoreApp()

AWS_DEFAULT_REGION = os.getenv(EnvVars.AWS_DEFAULT_REGION.value, "us-west-2")
PROJECT_NAME = os.getenv(EnvVars.PROJECT_NAME.value, "omnisummary")
STAGE = os.getenv(EnvVars.STAGE.value, "dev")
SSM_PARAM_NAME_PREFIX = f"/{PROJECT_NAME}/{STAGE}"

SSM_TO_ENV_MAPPING = {
    SSMParams.LANGCHAIN_API_KEY: EnvVars.LANGCHAIN_API_KEY,
    SSMParams.SLACK_BUSINESS_TOKEN: EnvVars.SLACK_BUSINESS_TOKEN,
    SSMParams.SLACK_BUSINESS_CHANNEL_IDS: EnvVars.SLACK_BUSINESS_CHANNEL_IDS,
    SSMParams.SLACK_PERSONAL_TOKEN: EnvVars.SLACK_PERSONAL_TOKEN,
    SSMParams.SLACK_PERSONAL_CHANNEL_IDS: EnvVars.SLACK_PERSONAL_CHANNEL_IDS,
    SSMParams.SLACK_SIGNING_SECRET: EnvVars.SLACK_SIGNING_SECRET,
    SSMParams.UPSTAGE_API_KEY: EnvVars.UPSTAGE_API_KEY,
}


def _load_ssm_parameters_to_env(session: boto3.Session) -> None:
    logger.info("Loading SSM parameters...")
    for ssm_param, env_var in SSM_TO_ENV_MAPPING.items():
        param_name = f"{SSM_PARAM_NAME_PREFIX}/{ssm_param.value}"
        param_value = get_ssm_param_value(session, param_name)
        if param_value:
            os.environ[env_var.value] = param_value
        else:
            logger.warning("SSM parameter '%s' not found or empty", param_name)
    logger.info("SSM parameters loaded.")


@app.entrypoint
def invoke(payload: dict[str, Any]) -> str:
    try:
        boto_session = boto3.Session(region_name=AWS_DEFAULT_REGION)
        _load_ssm_parameters_to_env(boto_session)

        user_input = payload["prompt"]
        channel_id = payload.get("channel_id")

        logger.info("User input: '%s'", user_input[:100])
        logger.info("Slack context: channel_id='%s'", channel_id)

        with tool_state_context(channel_id):
            agent = create_summarization_agent()
            logger.info("Created new agent instance for this request")

            response = agent(user_input)
            logger.info("Agent response received")

            result = response.message["content"][0].get("text")
            if not result:
                logger.error("No text content in agent response")
                raise ValueError("No text content in agent response")

            logger.info("Agent execution completed successfully")
            return result

    except (KeyError, IndexError) as e:
        logger.error("Failed to parse payload or agent response: %s", e, exc_info=True)
        raise ValueError(f"Invalid payload or agent response structure: {e}") from e
    except Exception as e:
        logger.error("Agent execution failed: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    app.run()
