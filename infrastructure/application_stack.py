from pathlib import Path
from typing import Any

from aws_cdk import CfnOutput, Duration, Stack
from aws_cdk import aws_apigateway as apigw
from aws_cdk import aws_bedrockagentcore as bedrockagentcore
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_ssm as ssm
from aws_cdk.aws_lambda_python_alpha import PythonFunction
from constructs import Construct

from shared import Config, EnvVars, SSMParams

from .foundation_stack import OmniSummaryFoundationStack


class OmniSummaryApplicationStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        config: Config,
        foundation_stack: OmniSummaryFoundationStack,
        langchain_api_key: str | None = None,
        slack_business_token: str | None = None,
        slack_business_channel_ids: list[str] | None = None,
        slack_personal_token: str | None = None,
        slack_personal_channel_ids: list[str] | None = None,
        slack_signing_secret: str | None = None,
        upstage_api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.config = config
        self.project_name = config.resources.project_name
        self.stage = config.resources.stage
        self.project_root = Path(__file__).parent.parent
        self.foundation_stack = foundation_stack

        self._store_ssm_parameters(
            langchain_api_key,
            slack_business_token,
            slack_business_channel_ids,
            slack_personal_token,
            slack_personal_channel_ids,
            slack_signing_secret,
            upstage_api_key,
        )

        self.agentcore_runtime = self._create_agentcore_runtime()
        self.slack_event_handler_function = self._create_slack_event_handler_function()
        self.api_gateway = self._create_api_gateway()

        self._create_outputs()

    def _get_resource_name(self, suffix: str) -> str:
        return f"{self.project_name}-{self.stage}-{suffix}"

    def _create_agentcore_runtime(self) -> bedrockagentcore.CfnRuntime:
        runtime = bedrockagentcore.CfnRuntime(
            self,
            "SummaryBotAgentCoreRuntime",
            agent_runtime_name=self._get_resource_name("summary-bot").replace("-", "_"),
            agent_runtime_artifact=bedrockagentcore.CfnRuntime.AgentRuntimeArtifactProperty(
                container_configuration=bedrockagentcore.CfnRuntime.ContainerConfigurationProperty(
                    container_uri=f"{self.foundation_stack.ecr_repository.repository_uri}:{self.config.infrastructure.agentcore_image_tag}"
                )
            ),
            network_configuration=bedrockagentcore.CfnRuntime.NetworkConfigurationProperty(network_mode="PUBLIC"),
            protocol_configuration="HTTP",
            role_arn=self.foundation_stack.agentcore_role.role_arn,
            description=f"Summary Bot AgentCore Runtime for '{self.stack_name}'",
            environment_variables={
                EnvVars.AWS_BEDROCK_REGION.value: self.config.resources.bedrock_region_name or self.region,
                EnvVars.AWS_DEFAULT_REGION.value: self.config.resources.default_region_name or self.region,
                EnvVars.PROJECT_NAME.value: self.project_name,
                EnvVars.STAGE.value: self.stage,
            },
        )

        runtime.node.add_dependency(self.foundation_stack.build_trigger)
        return runtime

    def _create_slack_event_handler_function(self) -> lambda_.Function:
        environment = {
            EnvVars.AGENTCORE_RUNTIME_ARN.value: self.agentcore_runtime.attr_agent_runtime_arn,
            EnvVars.DDB_TABLE_NAME.value: self.foundation_stack.event_dedup_table.table_name,
            EnvVars.EVENT_DEDUPLICATION_TTL_SEC.value: str(self.config.infrastructure.event_deduplication_ttl_sec),
            EnvVars.PROJECT_NAME.value: self.project_name,
            EnvVars.SLACK_SIGNATURE_EXPIRATION_SEC.value: str(
                self.config.infrastructure.slack_signature_expiration_sec
            ),
            EnvVars.STAGE.value: self.stage,
        }

        return PythonFunction(
            self,
            "SummaryBotSlackEventHandlerLambdaFunction",
            entry=str(self.project_root / "lambda_handlers"),
            index="slack_event_handler.py",
            handler="handler",
            runtime=lambda_.Runtime.PYTHON_3_12,
            timeout=Duration.seconds(self.config.infrastructure.lambda_timeout_seconds),
            memory_size=self.config.infrastructure.lambda_memory_mb,
            role=self.foundation_stack.lambda_role,
            environment=environment,
            function_name=self._get_resource_name("slack-event-handler-lambda"),
        )

    def _create_api_gateway(self) -> apigw.RestApi:
        api = apigw.RestApi(
            self,
            "SummaryBotAPI",
            rest_api_name=self._get_resource_name("summary-bot"),
            description="API Gateway for Summary Bot",
            deploy_options=apigw.StageOptions(stage_name=self.stage),
        )

        integration = apigw.LambdaIntegration(self.slack_event_handler_function)
        slack_resource = api.root.add_resource("slack")
        events_resource = slack_resource.add_resource("events")
        events_resource.add_method("POST", integration)

        return api

    def _store_ssm_parameters(
        self,
        langchain_api_key: str | None,
        slack_business_token: str | None,
        slack_business_channel_ids: list[str] | None,
        slack_personal_token: str | None,
        slack_personal_channel_ids: list[str] | None,
        slack_signing_secret: str | None,
        upstage_api_key: str | None,
    ) -> None:
        ssm_params_to_create = {
            SSMParams.LANGCHAIN_API_KEY: langchain_api_key,
            SSMParams.SLACK_BUSINESS_TOKEN: slack_business_token,
            SSMParams.SLACK_BUSINESS_CHANNEL_IDS: slack_business_channel_ids,
            SSMParams.SLACK_PERSONAL_TOKEN: slack_personal_token,
            SSMParams.SLACK_PERSONAL_CHANNEL_IDS: slack_personal_channel_ids,
            SSMParams.SLACK_SIGNING_SECRET: slack_signing_secret,
            SSMParams.UPSTAGE_API_KEY: upstage_api_key,
        }

        descriptions = {
            SSMParams.LANGCHAIN_API_KEY: "Langchain API Key",
            SSMParams.SLACK_BUSINESS_TOKEN: "Slack Business Token",
            SSMParams.SLACK_BUSINESS_CHANNEL_IDS: "Slack Business Channel IDs",
            SSMParams.SLACK_PERSONAL_TOKEN: "Slack Personal Token",
            SSMParams.SLACK_PERSONAL_CHANNEL_IDS: "Slack Personal Channel IDs",
            SSMParams.SLACK_SIGNING_SECRET: "Slack Signing Secret",
            SSMParams.UPSTAGE_API_KEY: "Upstage API Key",
        }

        for param_enum, param_value in ssm_params_to_create.items():
            if param_value:
                param_name = f"/{self.project_name}/{self.stage}/{param_enum.value}"
                string_value = ",".join(param_value) if isinstance(param_value, list) else param_value
                ssm.StringParameter(
                    self,
                    f"SsmParam{param_enum.name}",
                    parameter_name=param_name,
                    string_value=string_value,
                    description=descriptions.get(param_enum, "Managed by CDK"),
                    tier=ssm.ParameterTier.STANDARD,
                )

    def _create_outputs(self) -> None:
        CfnOutput(
            self,
            "SummaryBotAgentCoreRuntimeArn",
            value=self.agentcore_runtime.attr_agent_runtime_arn,
            description="Summary Bot AgentCore Runtime ARN",
        )
        CfnOutput(
            self,
            "SummaryBotSlackWebhookUrl",
            value=f"{self.api_gateway.url}slack/events",
            description="Summary Bot Slack Events API webhook URL",
        )
