from __future__ import annotations

from pathlib import Path

from aws_cdk import CfnOutput, Duration, Stack, Tags
from aws_cdk import aws_apigateway as apigw
from aws_cdk import aws_events as events
from aws_cdk import aws_events_targets as targets
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_ssm as ssm
from aws_cdk.aws_bedrockagentcore import CfnRuntime
from constructs import Construct

from shared import Config

from .foundation_stack import OmniSummaryFoundationStack


class OmniSummaryApplicationStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        config: Config,
        foundation: OmniSummaryFoundationStack,
        slack_signing_secret: str = "",
        slack_bot_token: str = "",
        tavily_api_key: str = "",
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        project_name = config.aws.project_name
        stage = config.aws.stage
        bedrock_region = config.aws.bedrock_region

        Tags.of(self).add("Project", project_name)
        Tags.of(self).add("Stage", stage)

        project_root = Path(__file__).parent.parent

        ssm_params = {
            "slack-signing-secret": slack_signing_secret,
            "slack-bot-token": slack_bot_token,
            "tavily-api-key": tavily_api_key,
        }
        for name, value in ssm_params.items():
            if value:
                ssm.StringParameter(
                    self,
                    f"Ssm-{name}",
                    parameter_name=f"/{project_name}/{stage}/{name}",
                    string_value=value,
                )

        agentcore_runtime = CfnRuntime(
            self,
            "AgentCoreRuntime",
            agent_runtime_name=f"{project_name}_{stage}_followup",
            agent_runtime_artifact={
                "containerConfiguration": {"containerUri": f"{foundation.ecr_repo.repository_uri}@sha256:96eede6755e1496685d0f2c46a449e08e12a45ba3b6492dc258b873cee56fd4f"}
            },
            network_configuration={"networkMode": "PUBLIC"},
            protocol_configuration="HTTP",
            role_arn=foundation.agentcore_role.role_arn,
            environment_variables={
                "AWS_BEDROCK_REGION": bedrock_region,
                "STATE_BUCKET": foundation.state_bucket.bucket_name,
                "S3_PREFIX": f"{config.aws.s3_prefix}/digest_state" if config.aws.s3_prefix else "digest_state",
                "PROJECT_NAME": project_name,
                "STAGE": stage,
            },
        )

        digest_lambda = lambda_.DockerImageFunction(
            self,
            "DigestPipelineLambda",
            function_name=f"{project_name}-{stage}-digest",
            code=lambda_.DockerImageCode.from_ecr(
                foundation.ecr_repo,
                tag_or_digest="latest",
                cmd=["lambda_handlers.digest_handler.handler"],
            ),
            timeout=Duration.minutes(15),
            memory_size=1024,
            role=foundation.lambda_role,
            vpc=foundation.vpc,
            vpc_subnets=foundation.vpc_subnets,
            environment={
                "STATE_BUCKET": foundation.state_bucket.bucket_name,
                "S3_PREFIX": f"{config.aws.s3_prefix}/digest_state" if config.aws.s3_prefix else "digest_state",
                "RSSHUB_BASE_URL": "http://rsshub.omnisummary.local:1200",
                "AWS_BEDROCK_REGION": bedrock_region,
                "PROJECT_NAME": project_name,
                "STAGE": stage,
            },
        )

        slack_lambda = lambda_.Function(
            self,
            "SlackEventLambda",
            function_name=f"{project_name}-{stage}-slack-events",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="slack_event_handler.handler",
            code=lambda_.Code.from_asset(str(project_root / "lambda_handlers")),
            timeout=Duration.seconds(60),
            memory_size=128,
            role=foundation.lambda_role,
            environment={
                "AGENTCORE_RUNTIME_ARN": agentcore_runtime.attr_agent_runtime_arn,
                "DDB_TABLE_NAME": foundation.dedup_table.table_name,
                "PROJECT_NAME": project_name,
                "STAGE": stage,
            },
        )

        api = apigw.RestApi(
            self,
            "SlackApi",
            rest_api_name=f"{project_name}-{stage}-slack",
            deploy_options=apigw.StageOptions(stage_name=stage),
        )
        slack_resource = api.root.add_resource("slack").add_resource("events")
        slack_resource.add_method("POST", apigw.LambdaIntegration(slack_lambda))

        events.Rule(
            self,
            "DailyDigestRule",
            rule_name=f"{project_name}-{stage}-daily-digest",
            schedule=events.Schedule.cron(hour="13", minute="0"),
            targets=[targets.LambdaFunction(digest_lambda)],
        )

        CfnOutput(self, "SlackWebhookUrl", value=f"{api.url}slack/events")
        CfnOutput(self, "AgentCoreArn", value=agentcore_runtime.attr_agent_runtime_arn)
        CfnOutput(self, "StateBucket", value=foundation.state_bucket.bucket_name)
