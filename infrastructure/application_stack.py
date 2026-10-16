from __future__ import annotations

from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack, Tags
from aws_cdk import aws_cloudwatch as cloudwatch
from aws_cdk import aws_cloudwatch_actions as cw_actions
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_events as events
from aws_cdk import aws_events_targets as targets
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_lambda_destinations as destinations
from aws_cdk import aws_logs as logs
from aws_cdk import aws_sqs as sqs
from aws_cdk import aws_ssm as ssm
from constructs import Construct

from shared import Config
from shared.constants import DIGEST_ITEMS_METRIC, METRIC_NAMESPACE, RSSHUB_PORT, SSM_PLACEHOLDER, SSM_SECRET_ENV_VARS
from shared.metrics import metric_dimensions

from .foundation_stack import OmniSummaryFoundationStack


class OmniSummaryApplicationStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        config: Config,
        foundation: OmniSummaryFoundationStack,
        digest_image_ref: str = "",
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        project_name = config.aws.project_name
        stage = config.aws.stage
        state_prefix = f"{config.aws.s3_prefix}/digest_state" if config.aws.s3_prefix else "digest_state"

        Tags.of(self).add("Project", project_name)
        Tags.of(self).add("Stage", stage)

        # The parameter PATHS are stack-managed (so the scoped ssm:GetParameter* policy and the
        # names stay in one place) but the VALUES never are. CloudFormation cannot create
        # SecureString parameters, so this stack used to pass the real tokens in as `string_value`
        # — which wrote the live Tavily/OpenAI/YouTube keys and the Threads access token in
        # PLAINTEXT into the template, and therefore into cdk.out/*.template.json, the CDK
        # staging bucket and every cloudformation:GetTemplate response.
        #
        # Each parameter is created holding SSM_PLACEHOLDER instead, and `scripts/put_secrets.py`
        # replaces it with the real value as a SecureString after the deploy (it DELETES the
        # placeholder String and re-creates it encrypted, because SSM rejects a type change on an
        # overwrite). Re-deploys do NOT clobber it: CloudFormation only updates a resource whose
        # template properties changed, and the placeholder never changes — so the SecureString value
        # and type survive.
        #
        # Therefore: NEVER add, rename or re-value a property of these resources (not even a
        # Description). Any property change makes CloudFormation PutParameter this resource again,
        # writing the placeholder over the live secret — every affected feature then reads its
        # secret as unset until put_secrets.py is re-run. tests/test_infrastructure.py pins the
        # rendered property set for exactly this reason.
        for name in SSM_SECRET_ENV_VARS:
            ssm.StringParameter(
                self,
                f"Ssm-{name}",
                parameter_name=f"/{project_name}/{stage}/{name}",
                string_value=SSM_PLACEHOLDER,
            )

        # Pin to the pushed image digest when provided (DIGEST_IMAGE_REF). A bare
        # ":latest" tag string never changes in the template, so CloudFormation would
        # not redeploy the function after a new push; a digest forces the update.
        digest_tag_or_digest = (digest_image_ref or "latest").lstrip("@")

        # DLQ for failed async Lambda invokes (defined in foundation so the on_failure send grant
        # on the shared lambda_role stays intra-stack). With retry_attempts=0, a failed run lands
        # here for inspection/replay instead of auto-retrying. The visual handler also carries a
        # per-date Threads idempotency marker (ThreadsPostLedger), so even a retry after a
        # POST-publish timeout won't double-post; retry_attempts=0 keeps that belt-and-suspenders.
        async_dlq = foundation.async_dlq

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
            # gpt-image render + Threads root post + reply-indexing retries (an IMAGE
            # root isn't addressable as a reply target for several MINUTES) can exceed 10 min; 15
            # (the Lambda max) gives the flat reply chain room to finish so comments aren't cut off.
            timeout=Duration.minutes(15),
            memory_size=512,
            role=foundation.lambda_role,
            # Outside the VPC: everything it calls (Bedrock, S3, AgentCore Memory, SNS, OpenAI, Threads)
            # is a public endpoint, so the VPC only added an ENI and a NAT dependency to the digest's
            # only Threads delivery path. The digest Lambda stays inside for the RSSHub service.
            # No auto-retry: the ThreadsPostLedger idempotency marker already blocks a re-post,
            # and retry_attempts=0 avoids re-running the costly gpt-image render on failure.
            # Failures go to the DLQ for manual replay instead.
            retry_attempts=0,
            on_failure=destinations.SqsDestination(async_dlq),
            log_group=self._log_group("DailyVisualLambdaLogGroup", f"{project_name}-{stage}-visual"),
            environment={
                "PROJECT_NAME": project_name,
                "STAGE": stage,
                "MEMORY_ID": foundation.memory_id,
                # The visual Lambda is the Threads publish path, so it — not the digest Lambda —
                # is what knows whether the post actually landed. It notifies on a partial reply
                # chain / failed delivery; without this the notice is a silent no-op.
                "ALERT_SNS_TOPIC_ARN": foundation.alerts_topic.topic_arn,
                # The visual maker keeps the Threads post ledger and hosts the image in the state
                # bucket; without these it falls through to the config-profile branch and crashes on
                # the 'research' profile, which doesn't exist in Lambda.
                "STATE_BUCKET": foundation.state_bucket.bucket_name,
                "S3_PREFIX": state_prefix,
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
            # No auto-retry on the daily cron invoke: the pipeline is not idempotent (a retry
            # re-posts the digest). A failure goes to the DLQ for inspection/manual replay.
            retry_attempts=0,
            on_failure=destinations.SqsDestination(async_dlq),
            log_group=self._log_group("DigestPipelineLambdaLogGroup", f"{project_name}-{stage}-digest"),
            environment={
                "STATE_BUCKET": foundation.state_bucket.bucket_name,
                "S3_PREFIX": state_prefix,
                "RSSHUB_BASE_URL": foundation.rsshub_base_url,
                "PROJECT_NAME": project_name,
                "STAGE": stage,
                "ALERT_SNS_TOPIC_ARN": foundation.alerts_topic.topic_arn,
                "MEMORY_ID": foundation.memory_id,
                "VISUAL_FUNCTION_NAME": visual_lambda.function_name,
            },
        )

        # The digest Lambda resolves rsshub.omnisummary.local and connects to the Fargate RSSHub
        # service, but the service's security group had no ingress rule for the Lambda's SG, so
        # every X feed fetch silently timed out in AWS (the S3 park file is what actually supplied
        # X items). Added from THIS stack — which already depends on foundation — so the rule is
        # a plain ingress addition with no new SG/service and no cross-stack cycle. It is written
        # as an explicit CfnSecurityGroupIngress rather than connections.allow_from(), because the
        # latter attaches the rule to the RSSHub SG's own stack (foundation) and CDK then rejects
        # the fnd -> app SG reference as a dependency cycle. The Lambda SG allows all egress by
        # default, so ingress is the only missing half.
        ec2.CfnSecurityGroupIngress(
            self,
            "RSSHubIngressFromDigestLambda",
            group_id=foundation.rsshub_service.connections.security_groups[0].security_group_id,
            source_security_group_id=digest_lambda.connections.security_groups[0].security_group_id,
            ip_protocol="tcp",
            from_port=RSSHUB_PORT,
            to_port=RSSHUB_PORT,
            description="Digest Lambda to RSSHub",
        )

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
            # The handler re-raises on failure so the Errors alarm fires; without an explicit
            # value Lambda would retry the async (EventBridge) invoke twice, re-calling the
            # Threads refresh endpoint. One attempt + the alarm is what we want.
            retry_attempts=0,
            log_group=self._log_group("ThreadsRefreshLambdaLogGroup", f"{project_name}-{stage}-threads-refresh"),
            environment={"PROJECT_NAME": project_name, "STAGE": stage},
        )
        events.Rule(
            self,
            "ThreadsTokenRefreshRule",
            rule_name=f"{project_name}-{stage}-threads-refresh",
            schedule=events.Schedule.rate(Duration.days(config.aws.threads_token_refresh_days)),
            targets=[targets.LambdaFunction(threads_refresh_lambda)],
        )

        # Answers readers' comments on the account's recent posts. Outside the VPC like the visual
        # Lambda: it calls only public endpoints (Threads, Bedrock, S3). One run handles at most
        # pipeline.comment_reply_max_per_run comments, one model call and one post each.
        comment_reply_lambda = lambda_.DockerImageFunction(
            self,
            "CommentReplyLambda",
            function_name=f"{project_name}-{stage}-comment-replies",
            code=lambda_.DockerImageCode.from_ecr(
                foundation.ecr_repo,
                tag_or_digest=digest_tag_or_digest,
                cmd=["lambda_handlers.comment_reply_handler.handler"],
            ),
            # Up to comment_reply_max_per_run comments, each a model call plus a publish whose retry
            # ladder alone can take 30s.
            timeout=Duration.minutes(10),
            memory_size=512,
            role=foundation.lambda_role,
            # One run at a time: two runs reading the same conversation would both answer a comment.
            reserved_concurrent_executions=1,
            # A failed comment stays unrecorded and the next scheduled run retries it.
            retry_attempts=0,
            log_group=self._log_group("CommentReplyLambdaLogGroup", f"{project_name}-{stage}-comment-replies"),
            environment={
                "PROJECT_NAME": project_name,
                "STAGE": stage,
                "STATE_BUCKET": foundation.state_bucket.bucket_name,
                "S3_PREFIX": state_prefix,
            },
        )
        events.Rule(
            self,
            "CommentReplyRule",
            rule_name=f"{project_name}-{stage}-comment-replies",
            schedule=events.Schedule.rate(Duration.minutes(config.aws.comment_reply_interval_minutes)),
            targets=[targets.LambdaFunction(comment_reply_lambda)],
        )

        # Alarm on ALL Lambdas. Created here so every Lambda already exists.
        self._add_alarms(
            {
                "Digest": digest_lambda,
                "Visual": visual_lambda,
                "ThreadsRefresh": threads_refresh_lambda,
                "CommentReply": comment_reply_lambda,
            },
            foundation,
            async_dlq,
            project_name=project_name,
            stage=stage,
        )

        CfnOutput(self, "StateBucket", value=foundation.state_bucket.bucket_name)

    def _add_alarms(
        self,
        lambdas: dict[str, lambda_.Function],
        foundation: OmniSummaryFoundationStack,
        async_dlq: sqs.Queue,
        *,
        project_name: str,
        stage: str,
    ) -> None:
        # The EMF-backed metrics below are published with these dimensions (shared/metrics.py), so
        # the alarm MUST carry the same map: an undimensioned alarm reads every deployment's
        # datapoints as one series, which made a dev run of 5 items keep prod's empty-digest alarm
        # green on a day prod shipped nothing.
        emf_dimensions = metric_dimensions(project_name, stage)
        alarm_action = cw_actions.SnsAction(foundation.alerts_topic)
        for name, fn in lambdas.items():
            # Crashes (raised exceptions) — Errors metric.
            errors = fn.metric_errors(period=Duration.minutes(5)).create_alarm(
                self,
                f"{name}ErrorsAlarm",
                threshold=1,
                evaluation_periods=1,
                treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
            )
            errors.add_alarm_action(alarm_action)
            # Timeouts don't count as Errors, so also alarm when max Duration nears the configured
            # timeout (90%). Catches the visual-Lambda "ran out of time mid Threads post" case.
            timeout_ms = fn.timeout.to_milliseconds() if fn.timeout else 0
            if timeout_ms:
                duration = fn.metric_duration(period=Duration.minutes(5), statistic="Maximum").create_alarm(
                    self,
                    f"{name}TimeoutAlarm",
                    threshold=timeout_ms * 0.9,
                    evaluation_periods=1,
                    treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
                )
                duration.add_alarm_action(alarm_action)

        # Symptomless failure: the digest ran with no error/timeout but published an empty digest,
        # OR didn't run at all that day. The EMF metric appears once per daily run; a 24h window with
        # missing-data=BREACHING catches both a 0 count and a skipped run. NOTE: CloudWatch caps an
        # alarm at evaluation_periods * period <= 86400s (1 day), so this MUST stay <= 24h — 25h is
        # rejected by PutMetricAlarm at deploy. Missing-data=BREACHING already covers the skipped-run
        # case without needing extra slack.
        empty_digest = cloudwatch.Metric(
            namespace=METRIC_NAMESPACE,
            metric_name=DIGEST_ITEMS_METRIC,
            dimensions_map=emf_dimensions,
            period=Duration.hours(24),
            statistic="Maximum",
        ).create_alarm(
            self,
            "EmptyDigestAlarm",
            threshold=1,
            comparison_operator=cloudwatch.ComparisonOperator.LESS_THAN_THRESHOLD,
            evaluation_periods=1,
            treat_missing_data=cloudwatch.TreatMissingData.BREACHING,
        )
        empty_digest.add_alarm_action(alarm_action)

        # Any message in the async DLQ means a digest/visual run failed (retries are off) and is
        # waiting for inspection/replay — alarm so it doesn't sit unnoticed.
        dlq_alarm = async_dlq.metric_approximate_number_of_messages_visible(period=Duration.minutes(5)).create_alarm(
            self,
            "AsyncDLQAlarm",
            threshold=1,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
            evaluation_periods=1,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
        )
        dlq_alarm.add_alarm_action(alarm_action)

    def _log_group(self, construct_id: str, function_name: str) -> logs.LogGroup:
        """A function's log group, DECLARED, instead of the deprecated `log_retention=` prop.

        That prop does not set a property — it renders a `Custom::LogRetention` Lambda per function,
        each with its own execution role holding `logs:PutRetentionPolicy` AND
        `logs:DeleteRetentionPolicy` on Resource `"*"`. Four extra Lambdas, four extra roles, and a
        wildcard permission that can silently shorten or drop the retention of ANY log group in a
        shared account, all to set one property CloudWatch exposes on the group itself. Declaring the
        group also brings it inside the template-wide wildcard-resource and retention assertions.

        CloudFormation will not adopt a log group Lambda already created implicitly for an existing
        function; such a group has to be deleted (or imported) before the deploy that declares it.
        """
        return logs.LogGroup(
            self,
            construct_id,
            log_group_name=f"/aws/lambda/{function_name}",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY,
        )
