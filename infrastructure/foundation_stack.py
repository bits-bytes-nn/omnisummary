from __future__ import annotations

import json

from aws_cdk import Duration, RemovalPolicy, Stack, Tags
from aws_cdk import aws_budgets as budgets
from aws_cdk import aws_ce as ce
from aws_cdk import aws_ecr as ecr
from aws_cdk import aws_iam as iam
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_sns as sns
from aws_cdk import aws_sns_subscriptions as subs
from aws_cdk import aws_sqs as sqs
from aws_cdk.aws_bedrockagentcore import CfnMemory
from constructs import Construct

from shared import Config
from shared.naming import digest_memory_name, state_bucket_name


class OmniSummaryFoundationStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        config: Config,
        alert_email: str = "",
        editions: tuple[str, ...] = (),
        edition_prefixes: tuple[str, ...] = (),
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        project_name = config.aws.project_name
        stage = config.aws.stage

        Tags.of(self).add("Project", project_name)
        Tags.of(self).add("Stage", stage)

        # The stack owns the state bucket, on every stage: it holds the post ledger that stops a second
        # post, so it is versioned (old versions kept 30 days), retained when the stack goes, private and
        # TLS-only. A shared pre-existing bucket carried none of that.
        self.state_bucket = s3.Bucket(
            self,
            "StateBucket",
            bucket_name=state_bucket_name(config.aws.state_bucket_name, project_name, stage),
            removal_policy=RemovalPolicy.RETAIN,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            versioned=True,
            lifecycle_rules=[s3.LifecycleRule(noncurrent_version_expiration=Duration.days(30))],
        )

        self.ecr_repo = ecr.Repository(
            self,
            "AgentEcrRepo",
            repository_name=f"{project_name}-{stage}-agent",
            removal_policy=RemovalPolicy.DESTROY,
            # Headroom over the pushes between deploys: an expired image is the digest the Lambdas run on.
            lifecycle_rules=[ecr.LifecycleRule(max_image_count=30)],
            # Basic scanning (no charge): an OS package CVE in the base image is otherwise invisible.
            image_scan_on_push=True,
        )

        ssm_read_statement = iam.PolicyStatement(
            actions=["ssm:GetParameter"],
            resources=[f"arn:aws:ssm:{self.region}:{self.account}:parameter/{project_name}/{stage}/*"],
        )
        bedrock_invoke_statement = iam.PolicyStatement(
            # CountTokens powers token counting/truncation (replaces the local tiktoken heuristic);
            # without it the calls AccessDeny and silently fall back to a char/4 estimate.
            # No streaming action: every Lambda call is a non-streaming Converse (only the local
            # research CLI streams, under the operator's own credentials).
            actions=["bedrock:InvokeModel", "bedrock:CountTokens"],
            resources=[
                "arn:aws:bedrock:*::foundation-model/*",
                f"arn:aws:bedrock:*:{self.account}:inference-profile/*",
                # This project's APPLICATION inference profiles (scripts/put_inference_profiles.py)
                # are a DIFFERENT ARN resource type from the system-defined ones above. The model
                # resolver prefers them so on-demand token spend carries the Project cost-allocation
                # tag — omitting this makes every Bedrock call AccessDenied the moment a profile
                # exists, which is the whole digest.
                f"arn:aws:bedrock:*:{self.account}:application-inference-profile/*",
            ],
        )
        # ListInferenceProfiles: the cross-region helper resolves a model ID to a profile before
        # invoking (without it, a bare model ID that on-demand Bedrock rejects). GetInferenceProfile:
        # langchain-aws reads the model behind an application-profile ARN.
        bedrock_profile_statement = iam.PolicyStatement(
            actions=["bedrock:GetInferenceProfile", "bedrock:ListInferenceProfiles"],
            resources=["*"],
        )
        # One role per kind of work, so the Lambda that reads strangers' comments holds neither the
        # token write nor the publish fan-out: each edition's digest and visual Lambdas share one role
        # (lambda_role for AI/ML, edition_roles[name] otherwise), the comment-reply and token-refresh
        # Lambdas get only what they call. Per edition, so one edition cannot rewrite the other's S3 ledgers
        # and park files or fire the other's publish Lambda. The AgentCore memory is shared: editions keep
        # apart by session prefix, not by IAM.
        self.lambda_role = iam.Role(
            self,
            "LambdaRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
        )
        self.edition_roles: dict[str, iam.Role] = {"": self.lambda_role}
        for edition in editions:
            self.edition_roles[edition] = iam.Role(
                self,
                f"{edition.capitalize()}LambdaRole",
                assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            )
        pipeline_roles = list(self.edition_roles.values())
        self.comment_reply_role = iam.Role(
            self,
            "CommentReplyRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
        )
        self.token_refresh_role = iam.Role(
            self,
            "TokenRefreshRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
        )
        # Logs for this project's own functions only: the VPC-access managed policy this replaced
        # granted logs:* writes on every group in the account (no Lambda runs in a VPC any more).
        logs_statement = iam.PolicyStatement(
            actions=["logs:CreateLogStream", "logs:PutLogEvents"],
            resources=[f"arn:aws:logs:{self.region}:{self.account}:log-group:/aws/lambda/{project_name}-{stage}-*:*"],
        )
        # The comment-reply and refresh Lambdas read only the Threads token and user id, not the
        # OpenAI/Tavily/YouTube keys under the same path.
        threads_secrets_read = iam.PolicyStatement(
            actions=["ssm:GetParameter"],
            resources=[
                f"arn:aws:ssm:{self.region}:{self.account}:parameter/{project_name}/{stage}/{name}"
                for name in ("threads-access-token", "threads-user-id")
            ],
        )
        for role in (*pipeline_roles, self.comment_reply_role, self.token_refresh_role):
            role.add_to_policy(logs_statement)
        for role in pipeline_roles:
            role.add_to_policy(ssm_read_statement)
        self.comment_reply_role.add_to_policy(threads_secrets_read)
        self.token_refresh_role.add_to_policy(threads_secrets_read)
        for role in (*pipeline_roles, self.comment_reply_role):
            role.add_to_policy(bedrock_invoke_statement)
            role.add_to_policy(bedrock_profile_statement)
        # Scope object access to the project's own prefix instead of the entire bucket, and per edition
        # below. The root prefix provably covers
        # everything the project touches: {prefix}/digest_state/* (state_store), {prefix}/<file>
        # park files (adapters.collectors.base.park_file_key) and {prefix}/threads/*.png (daily visual).
        # Exactly the calls the code makes: get_object/put_object (and the presigned GET Threads
        # fetches, signed by this role). ListBucket only so a missing key reads as NoSuchKey rather
        # than AccessDenied. It cannot carry an s3:prefix condition: S3 checks it on a GetObject for a
        # missing key, where no prefix exists, so the condition would turn every NoSuchKey into
        # AccessDenied. grant_read_write also handed out delete, legal-hold and retention actions.
        # An empty s3_prefix keeps the bucket-wide object scope.
        prefix = config.aws.s3_prefix.strip("/")
        root = f"{prefix}/" if prefix else ""
        # The comment-reply Lambda writes one object, its own ledger: it cannot touch the post ledger,
        # the published-URL ledger, trends or the park files. (COMMENT_REPLIES_KEY under digest_state.)
        # An edition lives under its own prefix inside the AI/ML one (deploy.py checks), so the AI/ML
        # role's grant would cover it: an explicit Deny keeps the AI/ML Lambdas out of it.
        # deploy.py always passes each edition's own prefix; without them (tests, bootstrap) the edition
        # is assumed to live at {s3_prefix}/{edition}, the layout deploy.py enforces.
        prefixes = dict(zip(editions, edition_prefixes, strict=True)) if edition_prefixes else {}
        edition_roots = {
            edition: (prefixes.get(edition) or f"{root}{edition}").strip("/") + "/" for edition in editions
        }
        object_scopes = {
            self.lambda_role: f"{root}*",
            self.comment_reply_role: f"{root}digest_state/comment_replies.json",
            **{self.edition_roles[edition]: f"{edition_root}*" for edition, edition_root in edition_roots.items()},
        }
        for role, objects in object_scopes.items():
            role.add_to_policy(
                iam.PolicyStatement(
                    actions=["s3:GetObject", "s3:PutObject"],
                    resources=[self.state_bucket.arn_for_objects(objects)],
                )
            )
        if edition_roots:
            self.lambda_role.add_to_policy(
                iam.PolicyStatement(
                    effect=iam.Effect.DENY,
                    actions=["s3:GetObject", "s3:PutObject"],
                    resources=[self.state_bucket.arn_for_objects(f"{r}*") for r in edition_roots.values()],
                )
            )
        for role in (*pipeline_roles, self.comment_reply_role):
            role.add_to_policy(
                iam.PolicyStatement(
                    actions=["s3:ListBucket"],
                    resources=[self.state_bucket.bucket_arn],
                )
            )
        # The digest Lambda's only cross-function call is the async fan-out to the daily-visual
        # Lambda, so grant exactly that one function instead of every {project}-{stage}-* function
        # (which included the token-refresh Lambda). Built from the literal name the application
        # stack assigns, so foundation takes NO dependency on it — importing the function object here
        # would create a foundation -> application cycle.
        for edition, role in self.edition_roles.items():
            role.add_to_policy(
                iam.PolicyStatement(
                    actions=["lambda:InvokeFunction"],
                    resources=[
                        f"arn:aws:lambda:{self.region}:{self.account}:function:{project_name}-{stage}-"
                        + (f"{edition}-" if edition else "")
                        + "visual"
                    ],
                )
            )
        # The Threads token-refresh Lambda writes the renewed token back to its own SSM param.
        self.token_refresh_role.add_to_policy(
            iam.PolicyStatement(
                actions=["ssm:PutParameter"],
                resources=[
                    f"arn:aws:ssm:{self.region}:{self.account}:parameter/{project_name}/{stage}/threads-access-token"
                ],
            )
        )

        self.alerts_topic = sns.Topic(self, "AlertsTopic", topic_name=f"{project_name}-{stage}-alerts")
        if alert_email:
            self.alerts_topic.add_subscription(subs.EmailSubscription(alert_email))
        if config.aws.s3_sync_user:
            # The local sync writes the park files (X, YouTube) under the project's prefix and nothing else.
            iam.Policy(
                self,
                "S3SyncPolicy",
                statements=[
                    iam.PolicyStatement(
                        actions=["s3:PutObject"], resources=[self.state_bucket.arn_for_objects(f"{root}*")]
                    )
                ],
                users=[iam.User.from_user_name(self, "S3SyncUser", config.aws.s3_sync_user)],
            )
        if alert_email and config.aws.monthly_budget_usd:
            self._spend_guards(project_name, stage, config.aws.monthly_budget_usd, alert_email)
        for role in pipeline_roles:
            self.alerts_topic.grant_publish(role)

        # Dead-letter queue for failed async (EventBridge / fire-and-forget) Lambda invokes. Lives
        # here, with the per-edition pipeline roles, so the on_failure send grant stays intra-stack (an
        # app-stack DLQ would make foundation depend on the app stack — a cycle).
        self.async_dlq = sqs.Queue(
            self,
            "AsyncInvokeDLQ",
            queue_name=f"{project_name}-{stage}-async-dlq",
            retention_period=Duration.days(14),
        )

        # Short-term event memory only: digest snapshots are stored/reloaded via CreateEvent/ListEvents.
        # With no extraction strategy the memory never invokes a model, so it needs no execution role.
        self.memory = CfnMemory(
            self,
            "DigestMemory",
            name=digest_memory_name(project_name, stage),
            event_expiry_duration=90,
            description="OmniSummary digest snapshot state",
        )
        # dev IS the live deployment: a stack delete or replacement must not take 90 days of snapshots.
        self.memory.apply_removal_policy(RemovalPolicy.RETAIN)
        self.memory_id = self.memory.attr_memory_id

        # Scoped to THIS stack's memory instead of memory/* — otherwise the role could read and write
        # the event history of every AgentCore memory in the account, which in a shared account is
        # other projects' conversation state. The `/*` companion covers the per-session
        # sub-resources the data-plane actions address. Session ids are deterministic
        # (`digest-YYYY-MM-DD`), so no ListSessions.
        for role in pipeline_roles:
            role.add_to_policy(
                iam.PolicyStatement(
                    actions=["bedrock-agentcore:CreateEvent", "bedrock-agentcore:ListEvents"],
                    resources=[self.memory.attr_memory_arn, f"{self.memory.attr_memory_arn}/*"],
                )
            )

    def _spend_guards(self, project_name: str, stage: str, monthly_usd: float, email: str) -> None:
        """A budget and an anomaly monitor on everything tagged Project=<project_name>: the account-wide
        budget could not tell this project's spend from the account's."""
        subscriber = budgets.CfnBudget.SubscriberProperty(subscription_type="EMAIL", address=email)
        budgets.CfnBudget(
            self,
            "SpendBudget",
            budget=budgets.CfnBudget.BudgetDataProperty(
                budget_name=f"{project_name}-{stage}-monthly",
                budget_type="COST",
                time_unit="MONTHLY",
                budget_limit=budgets.CfnBudget.SpendProperty(amount=monthly_usd, unit="USD"),
                cost_filters={"TagKeyValue": [f"user:Project${project_name}"]},
            ),
            notifications_with_subscribers=[
                budgets.CfnBudget.NotificationWithSubscribersProperty(
                    notification=budgets.CfnBudget.NotificationProperty(
                        notification_type=kind, comparison_operator="GREATER_THAN", threshold=percent
                    ),
                    subscribers=[subscriber],
                )
                for kind, percent in (("ACTUAL", 80), ("FORECASTED", 100))
            ],
        )
        monitor = ce.CfnAnomalyMonitor(
            self,
            "SpendAnomalyMonitor",
            monitor_name=f"{project_name}-{stage}",
            monitor_type="CUSTOM",
            monitor_specification=json.dumps({"Tags": {"Key": "Project", "Values": [project_name]}}),
        )
        ce.CfnAnomalySubscription(
            self,
            "SpendAnomalySubscription",
            subscription_name=f"{project_name}-{stage}",
            frequency="DAILY",
            monitor_arn_list=[monitor.attr_monitor_arn],
            subscribers=[ce.CfnAnomalySubscription.SubscriberProperty(type="EMAIL", address=email)],
            # An anomaly worth a day of the monthly budget.
            threshold_expression=json.dumps(
                {
                    "Dimensions": {
                        "Key": "ANOMALY_TOTAL_IMPACT_ABSOLUTE",
                        "MatchOptions": ["GREATER_THAN_OR_EQUAL"],
                        "Values": [str(round(monthly_usd / 30, 2))],
                    }
                }
            ),
        )
