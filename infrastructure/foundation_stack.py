from __future__ import annotations

import os

from aws_cdk import Duration, RemovalPolicy, Stack, Tags
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecr as ecr
from aws_cdk import aws_ecs as ecs
from aws_cdk import aws_iam as iam
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_servicediscovery as sd
from aws_cdk import aws_sns as sns
from aws_cdk import aws_sns_subscriptions as subs
from aws_cdk import aws_sqs as sqs
from aws_cdk import aws_ssm as ssm
from aws_cdk.aws_bedrockagentcore import CfnMemory
from constructs import Construct

from shared import Config
from shared.constants import RSSHUB_PORT, SSM_PLACEHOLDER, SSM_RSSHUB_SECRET_ENV_VARS
from shared.memory import digest_memory_name


class OmniSummaryFoundationStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        config: Config,
        alert_email: str = "",
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        project_name = config.aws.project_name
        stage = config.aws.stage
        is_prod = stage == "prod"

        Tags.of(self).add("Project", project_name)
        Tags.of(self).add("Stage", stage)

        if config.aws.subnet_ids:
            self.vpc_subnets = ec2.SubnetSelection(
                subnets=[
                    ec2.Subnet.from_subnet_id(self, f"Subnet{i}", sid) for i, sid in enumerate(config.aws.subnet_ids)
                ]
            )
        else:
            self.vpc_subnets = ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS)

        if config.aws.vpc_id:
            self.vpc = ec2.Vpc.from_lookup(self, "Vpc", vpc_id=config.aws.vpc_id)
        else:
            self.vpc = ec2.Vpc(
                self,
                "Vpc",
                max_azs=2,
                nat_gateways=1,
                subnet_configuration=[
                    ec2.SubnetConfiguration(name="Public", subnet_type=ec2.SubnetType.PUBLIC, cidr_mask=24),
                    ec2.SubnetConfiguration(
                        name="Private", subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS, cidr_mask=24
                    ),
                ],
            )

        if config.aws.state_bucket_name:
            self.state_bucket = s3.Bucket.from_bucket_name(self, "StateBucket", config.aws.state_bucket_name)
        else:
            self.state_bucket = s3.Bucket(
                self,
                "StateBucket",
                bucket_name=f"{project_name}-{stage}-state",
                removal_policy=RemovalPolicy.RETAIN if is_prod else RemovalPolicy.DESTROY,
                auto_delete_objects=not is_prod,
                encryption=s3.BucketEncryption.S3_MANAGED,
                block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
                enforce_ssl=True,
                versioned=True,
            )

        self.ecr_repo = ecr.Repository(
            self,
            "AgentEcrRepo",
            repository_name=f"{project_name}-{stage}-agent",
            removal_policy=RemovalPolicy.DESTROY,
            lifecycle_rules=[ecr.LifecycleRule(max_image_count=10)],
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
        self.lambda_role = iam.Role(
            self,
            "LambdaRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaVPCAccessExecutionRole"),
            ],
        )
        self.lambda_role.add_to_policy(ssm_read_statement)
        self.lambda_role.add_to_policy(bedrock_invoke_statement)
        self.lambda_role.add_to_policy(bedrock_profile_statement)
        # Scope object access to the project's own prefix instead of the entire bucket. The bucket
        # can be a pre-existing shared one (config.aws.state_bucket_name), so a bucket-wide
        # grant hands every other tenant's objects to this role. The root prefix provably covers
        # everything the project touches: {prefix}/digest_state/* (state_store), {prefix}/<file>
        # park files (collectors.base.park_file_key) and {prefix}/threads/*.png (daily visual).
        # Exactly the calls the code makes: get_object/put_object (and the presigned GET Threads
        # fetches, signed by this role). ListBucket only so a missing key reads as NoSuchKey rather
        # than AccessDenied. It cannot carry an s3:prefix condition: S3 checks it on a GetObject for a
        # missing key, where no prefix exists, so the condition would turn every NoSuchKey into
        # AccessDenied. grant_read_write also handed out delete, legal-hold and retention actions.
        # An empty s3_prefix keeps the bucket-wide object scope.
        prefix = config.aws.s3_prefix.strip("/")
        self.lambda_role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:GetObject", "s3:PutObject"],
                resources=[self.state_bucket.arn_for_objects(f"{prefix}/*" if prefix else "*")],
            )
        )
        self.lambda_role.add_to_policy(
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
        self.lambda_role.add_to_policy(
            iam.PolicyStatement(
                actions=["lambda:InvokeFunction"],
                resources=[f"arn:aws:lambda:{self.region}:{self.account}:function:{project_name}-{stage}-visual"],
            )
        )
        # The Threads token-refresh Lambda writes the renewed token back to its own SSM param.
        self.lambda_role.add_to_policy(
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
        self.alerts_topic.grant_publish(self.lambda_role)

        # Dead-letter queue for failed async (EventBridge / fire-and-forget) Lambda invokes. Lives
        # here, with the shared lambda_role, so the on_failure send grant stays intra-stack (an
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
        self.memory_id = self.memory.attr_memory_id

        # Scoped to THIS stack's memory instead of memory/* — otherwise the role could read and write
        # the event history of every AgentCore memory in the account, which in a shared account is
        # other projects' conversation state. The `/*` companion covers the per-session
        # sub-resources the data-plane actions address. Session ids are deterministic
        # (`digest-YYYY-MM-DD`), so no ListSessions.
        self.lambda_role.add_to_policy(
            iam.PolicyStatement(
                actions=["bedrock-agentcore:CreateEvent", "bedrock-agentcore:ListEvents"],
                resources=[self.memory.attr_memory_arn, f"{self.memory.attr_memory_arn}/*"],
            )
        )

        self.ecs_cluster = ecs.Cluster(self, "EcsCluster", vpc=self.vpc)

        namespace_name, rsshub_service = "omnisummary.local", "rsshub"
        # The digest Lambda's RSSHub endpoint, built from the same two names the service registers.
        self.rsshub_base_url = f"http://{rsshub_service}.{namespace_name}:{RSSHUB_PORT}"
        namespace = sd.PrivateDnsNamespace(
            self,
            "ServiceNamespace",
            name=namespace_name,
            vpc=self.vpc,
        )

        # Created here, holding a placeholder, so they exist before the RSSHub service that consumes
        # them starts. scripts/put_secrets.py writes the real values as SecureStrings.
        rsshub_secret_params = {
            env_key: ssm.StringParameter(
                self,
                f"Ssm-{name}",
                parameter_name=f"/{project_name}/{stage}/{name}",
                string_value=SSM_PLACEHOLDER,
            )
            for name, env_key in SSM_RSSHUB_SECRET_ENV_VARS.items()
        }

        rsshub_task = ecs.FargateTaskDefinition(self, "RSSHubTask", memory_limit_mib=2048, cpu=1024)
        rsshub_task.add_container(
            "RSSHubContainer",
            image=ecs.ContainerImage.from_registry("diygod/rsshub:latest"),
            port_mappings=[ecs.PortMapping(container_port=RSSHUB_PORT)],
            # With no retention the CDK-generated group also carries DeletionPolicy: Retain, so this
            # was the one log group in either stack kept forever — and the only one nothing asserted.
            logging=ecs.LogDrivers.aws_logs(stream_prefix="rsshub", log_retention=logs.RetentionDays.ONE_MONTH),
            environment={
                "NODE_ENV": "production",
                "CACHE_TYPE": "memory",
                "PROXY_URI": os.environ.get("RSSHUB_PROXY_URI", ""),
                "PROXY_STRATEGY": "all",
            },
            # The X session cookies authenticate as the account, so they are NOT template material:
            # passed through `environment` they were written in plaintext into the task definition
            # and therefore into the CloudFormation template. `secrets` puts only the parameter ARN
            # there and the ECS agent fetches (and decrypts) the value at task start.
            # scripts/put_secrets.py writes these parameters as SecureStrings.
            secrets={env_key: ecs.Secret.from_ssm_parameter(param) for env_key, param in rsshub_secret_params.items()},
        )

        self.rsshub_service = ecs.FargateService(
            self,
            "RSSHubService",
            cluster=self.ecs_cluster,
            task_definition=rsshub_task,
            desired_count=config.aws.rsshub_desired_count,
            assign_public_ip=False,
            vpc_subnets=self.vpc_subnets,
            cloud_map_options=ecs.CloudMapOptions(
                name=rsshub_service,
                cloud_map_namespace=namespace,
                dns_record_type=sd.DnsRecordType.A,
            ),
        )
