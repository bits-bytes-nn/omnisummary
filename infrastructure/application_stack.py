from __future__ import annotations

from pathlib import Path

from aws_cdk import CfnOutput, Duration, Stack, Tags
from aws_cdk import aws_apigateway as apigw
from aws_cdk import aws_cloudwatch as cloudwatch
from aws_cdk import aws_cloudwatch_actions as cw_actions
from aws_cdk import aws_events as events
from aws_cdk import aws_events_targets as targets
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_ssm as ssm
from aws_cdk import aws_wafv2 as wafv2
from aws_cdk.aws_bedrockagentcore import CfnRuntime
from constructs import Construct

from shared import Config
from shared.constants import RSSHUB_PORT

from .foundation_stack import OmniSummaryFoundationStack


class OmniSummaryApplicationStack(Stack):
    # Service-discovery DNS name for the Fargate RSSHub service injected into the digest
    # Lambda. Overridable at synth via the `rsshub_base_url` CDK context value so
    # dev/staging/prod can target different endpoints without editing this stack.
    DEFAULT_RSSHUB_BASE_URL = f"http://rsshub.omnisummary.local:{RSSHUB_PORT}"

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        config: Config,
        foundation: OmniSummaryFoundationStack,
        slack_signing_secret: str = "",
        slack_bot_token: str = "",
        slack_channel_id: str = "",
        tavily_api_key: str = "",
        openai_api_key: str = "",
        threads_access_token: str = "",
        threads_user_id: str = "",
        agentcore_image_ref: str = "",
        digest_image_ref: str = "",
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        project_name = config.aws.project_name
        stage = config.aws.stage
        bedrock_region = config.aws.bedrock_region
        rsshub_base_url = self.node.try_get_context("rsshub_base_url") or self.DEFAULT_RSSHUB_BASE_URL

        Tags.of(self).add("Project", project_name)
        Tags.of(self).add("Stage", stage)

        project_root = Path(__file__).parent.parent

        ssm_params = {
            "slack-signing-secret": slack_signing_secret,
            "slack-bot-token": slack_bot_token,
            "slack-channel-id": slack_channel_id,
            "tavily-api-key": tavily_api_key,
            "openai-api-key": openai_api_key,
            "threads-access-token": threads_access_token,
            "threads-user-id": threads_user_id,
        }
        # CloudFormation cannot create SecureString SSM parameters (AWS::SSM::Parameter
        # supports only String/StringList). These hold low-sensitivity API tokens; access
        # is restricted by the scoped ssm:GetParameter* IAM policy on /{project}/{stage}/*.
        # Promote to Secrets Manager if higher-sensitivity credentials are added.
        for name, value in ssm_params.items():
            if value:
                ssm.StringParameter(
                    self,
                    f"Ssm-{name}",
                    parameter_name=f"/{project_name}/{stage}/{name}",
                    string_value=value,
                )

        image_ref = (agentcore_image_ref or "arm64").lstrip("@")
        if image_ref.startswith("sha256:"):
            container_uri = f"{foundation.ecr_repo.repository_uri}@{image_ref}"
        else:
            container_uri = f"{foundation.ecr_repo.repository_uri}:{image_ref}"

        agentcore_runtime = CfnRuntime(
            self,
            "AgentCoreRuntime",
            agent_runtime_name=f"{project_name}_{stage}_followup",
            agent_runtime_artifact={"containerConfiguration": {"containerUri": container_uri}},
            network_configuration={"networkMode": "PUBLIC"},
            protocol_configuration="HTTP",
            role_arn=foundation.agentcore_role.role_arn,
            environment_variables={
                "AWS_BEDROCK_REGION": bedrock_region,
                "PROJECT_NAME": project_name,
                "STAGE": stage,
                "MEMORY_ID": foundation.memory_id,
                # recall_trends reads trends.json from the same S3 location the digest
                # pipeline writes; without these the agent falls back to a local store
                # that has no trends in the container and recall silently returns nothing.
                "STATE_BUCKET": foundation.state_bucket.bucket_name,
                "S3_PREFIX": f"{config.aws.s3_prefix}/digest_state" if config.aws.s3_prefix else "digest_state",
            },
        )

        # Pin to the pushed image digest when provided (DIGEST_IMAGE_REF). A bare
        # ":latest" tag string never changes in the template, so CloudFormation would
        # not redeploy the function after a new push; a digest forces the update.
        digest_tag_or_digest = (digest_image_ref or "latest").lstrip("@")

        # Daily-visual Lambda: invoked asynchronously by the digest Lambda so its
        # LLM-editor + Tavily + gpt-image work (~1-2 min) stays off the digest's
        # critical path. Same image, loads ranked items from AgentCore Memory.
        visual_lambda = lambda_.DockerImageFunction(
            self,
            "DailyVisualLambda",
            function_name=f"{project_name}-{stage}-visual",
            code=lambda_.DockerImageCode.from_ecr(
                foundation.ecr_repo,
                tag_or_digest=digest_tag_or_digest,
                cmd=["lambda_handlers.visual_handler.handler"],
            ),
            # gpt-image render (~90s) + Threads root post + reply-indexing retries (the root
            # isn't addressable as a reply target for tens of seconds) can exceed 5 min; 10 gives
            # the flat reply chain room to finish so comments don't get cut off by a timeout.
            timeout=Duration.minutes(10),
            memory_size=512,
            role=foundation.lambda_role,
            vpc=foundation.vpc,
            vpc_subnets=foundation.vpc_subnets,
            environment={
                "AWS_BEDROCK_REGION": bedrock_region,
                "PROJECT_NAME": project_name,
                "STAGE": stage,
                "MEMORY_ID": foundation.memory_id,
            },
        )

        digest_lambda = lambda_.DockerImageFunction(
            self,
            "DigestPipelineLambda",
            function_name=f"{project_name}-{stage}-digest",
            code=lambda_.DockerImageCode.from_ecr(
                foundation.ecr_repo,
                tag_or_digest=digest_tag_or_digest,
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
                "RSSHUB_BASE_URL": rsshub_base_url,
                "AWS_BEDROCK_REGION": bedrock_region,
                "PROJECT_NAME": project_name,
                "STAGE": stage,
                "ALERT_SNS_TOPIC_ARN": foundation.alerts_topic.topic_arn,
                "MEMORY_ID": foundation.memory_id,
                "VISUAL_FUNCTION_NAME": visual_lambda.function_name,
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
            deploy_options=apigw.StageOptions(
                stage_name=stage,
                throttling_rate_limit=config.aws.api_throttle_rate_limit,
                throttling_burst_limit=config.aws.api_throttle_burst_limit,
                metrics_enabled=True,
            ),
        )
        slack_resource = api.root.add_resource("slack").add_resource("events")
        slack_resource.add_method("POST", apigw.LambdaIntegration(slack_lambda))

        self._attach_waf(api, project_name, stage, config.aws.waf_rate_limit)
        self._add_alarms(digest_lambda, slack_lambda, api, foundation)

        events.Rule(
            self,
            "DailyDigestRule",
            rule_name=f"{project_name}-{stage}-daily-digest",
            schedule=events.Schedule.cron(
                hour=config.aws.digest_cron_hour,
                minute=config.aws.digest_cron_minute,
            ),
            targets=[targets.LambdaFunction(digest_lambda)],
        )

        # Threads long-lived tokens expire after 60 days; refresh on a schedule well
        # inside that window so the self-posting token never lapses. Same image (carries
        # shared + httpx); writes the renewed token back to SSM.
        threads_refresh_lambda = lambda_.DockerImageFunction(
            self,
            "ThreadsRefreshLambda",
            function_name=f"{project_name}-{stage}-threads-refresh",
            code=lambda_.DockerImageCode.from_ecr(
                foundation.ecr_repo,
                tag_or_digest=digest_tag_or_digest,
                cmd=["lambda_handlers.threads_refresh_handler.handler"],
            ),
            timeout=Duration.minutes(1),
            memory_size=256,
            role=foundation.lambda_role,
            environment={"PROJECT_NAME": project_name, "STAGE": stage},
        )
        events.Rule(
            self,
            "ThreadsTokenRefreshRule",
            rule_name=f"{project_name}-{stage}-threads-refresh",
            schedule=events.Schedule.rate(Duration.days(config.aws.threads_token_refresh_days)),
            targets=[targets.LambdaFunction(threads_refresh_lambda)],
        )

        CfnOutput(self, "SlackWebhookUrl", value=f"{api.url}slack/events")
        CfnOutput(self, "AgentCoreArn", value=agentcore_runtime.attr_agent_runtime_arn)
        CfnOutput(self, "StateBucket", value=foundation.state_bucket.bucket_name)

    def _add_alarms(
        self,
        digest_lambda: lambda_.IFunction,
        slack_lambda: lambda_.IFunction,
        api: apigw.RestApi,
        foundation: OmniSummaryFoundationStack,
    ) -> None:
        alarm_action = cw_actions.SnsAction(foundation.alerts_topic)
        for name, fn in (("Digest", digest_lambda), ("SlackEvent", slack_lambda)):
            alarm = fn.metric_errors(period=Duration.minutes(5)).create_alarm(
                self,
                f"{name}ErrorsAlarm",
                threshold=1,
                evaluation_periods=1,
                treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
            )
            alarm.add_alarm_action(alarm_action)

        api_5xx = api.metric_server_error(period=Duration.minutes(5))
        api_alarm = api_5xx.create_alarm(
            self,
            "ApiServerErrorAlarm",
            threshold=1,
            evaluation_periods=1,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
        )
        api_alarm.add_alarm_action(alarm_action)

    def _attach_waf(self, api: apigw.RestApi, project_name: str, stage: str, rate_limit: int) -> None:
        managed_groups = [
            ("AWSManagedRulesCommonRuleSet", 1),
            ("AWSManagedRulesKnownBadInputsRuleSet", 2),
            ("AWSManagedRulesAmazonIpReputationList", 3),
        ]
        managed_rules = [
            wafv2.CfnWebACL.RuleProperty(
                name=group_name,
                priority=priority,
                override_action=wafv2.CfnWebACL.OverrideActionProperty(none={}),
                statement=wafv2.CfnWebACL.StatementProperty(
                    managed_rule_group_statement=wafv2.CfnWebACL.ManagedRuleGroupStatementProperty(
                        vendor_name="AWS",
                        name=group_name,
                    )
                ),
                visibility_config=wafv2.CfnWebACL.VisibilityConfigProperty(
                    sampled_requests_enabled=True,
                    cloud_watch_metrics_enabled=True,
                    metric_name=group_name,
                ),
            )
            for group_name, priority in managed_groups
        ]
        rate_rule = wafv2.CfnWebACL.RuleProperty(
            name="RateLimit",
            priority=0,
            action=wafv2.CfnWebACL.RuleActionProperty(block={}),
            statement=wafv2.CfnWebACL.StatementProperty(
                rate_based_statement=wafv2.CfnWebACL.RateBasedStatementProperty(
                    limit=rate_limit,
                    aggregate_key_type="IP",
                )
            ),
            visibility_config=wafv2.CfnWebACL.VisibilityConfigProperty(
                sampled_requests_enabled=True,
                cloud_watch_metrics_enabled=True,
                metric_name="RateLimit",
            ),
        )
        web_acl = wafv2.CfnWebACL(
            self,
            "SlackApiWebAcl",
            scope="REGIONAL",
            default_action=wafv2.CfnWebACL.DefaultActionProperty(allow={}),
            visibility_config=wafv2.CfnWebACL.VisibilityConfigProperty(
                sampled_requests_enabled=True,
                cloud_watch_metrics_enabled=True,
                metric_name=f"{project_name}-{stage}-waf",
            ),
            rules=[rate_rule, *managed_rules],
        )
        wafv2.CfnWebACLAssociation(
            self,
            "SlackApiWebAclAssociation",
            resource_arn=api.deployment_stage.stage_arn,
            web_acl_arn=web_acl.attr_arn,
        )
