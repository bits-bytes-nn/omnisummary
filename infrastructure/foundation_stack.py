from __future__ import annotations

import os

from aws_cdk import Duration, RemovalPolicy, Stack, Tags
from aws_cdk import aws_codebuild as codebuild
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecr as ecr
from aws_cdk import aws_ecs as ecs
from aws_cdk import aws_iam as iam
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_servicediscovery as sd
from aws_cdk import aws_sns as sns
from aws_cdk import aws_sns_subscriptions as subs
from aws_cdk import aws_sqs as sqs
from aws_cdk.aws_bedrockagentcore import CfnMemory
from constructs import Construct

from shared import Config
from shared.constants import RSSHUB_PORT


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

        self.dedup_table = dynamodb.Table(
            self,
            "EventDedup",
            table_name=f"{project_name}-{stage}-event-dedup",
            partition_key=dynamodb.Attribute(name="event_id", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            time_to_live_attribute="ttl",
            removal_policy=RemovalPolicy.DESTROY,
            encryption=dynamodb.TableEncryption.AWS_MANAGED,
            point_in_time_recovery_specification=dynamodb.PointInTimeRecoverySpecification(
                point_in_time_recovery_enabled=is_prod
            ),
        )

        ssm_read_statement = iam.PolicyStatement(
            actions=["ssm:GetParameter", "ssm:GetParameters", "ssm:GetParametersByPath"],
            resources=[f"arn:aws:ssm:{self.region}:{self.account}:parameter/{project_name}/{stage}/*"],
        )
        bedrock_invoke_statement = iam.PolicyStatement(
            actions=["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
            resources=[
                "arn:aws:bedrock:*::foundation-model/*",
                f"arn:aws:bedrock:*:{self.account}:inference-profile/*",
            ],
        )
        # The cross-region helper resolves a model ID to its inference-profile ARN
        # before invoking. Without these, it AccessDenies on ListInferenceProfiles and
        # falls back to a bare model ID, which Bedrock rejects for on-demand throughput.
        bedrock_profile_statement = iam.PolicyStatement(
            actions=["bedrock:GetInferenceProfile", "bedrock:ListInferenceProfiles"],
            resources=["*"],
        )
        logs_statement = iam.PolicyStatement(
            actions=["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
            resources=[
                f"arn:aws:logs:{self.region}:{self.account}:log-group:/aws/lambda/{project_name}-{stage}-*",
                f"arn:aws:logs:{self.region}:{self.account}:log-group:/aws/bedrock-agentcore/*",
            ],
        )

        self.agentcore_role = iam.Role(
            self,
            "AgentCoreRole",
            assumed_by=iam.ServicePrincipal("bedrock-agentcore.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEC2ContainerRegistryReadOnly"),
            ],
        )
        self.agentcore_role.add_to_policy(ssm_read_statement)
        self.agentcore_role.add_to_policy(bedrock_invoke_statement)
        self.agentcore_role.add_to_policy(bedrock_profile_statement)
        self.agentcore_role.add_to_policy(logs_statement)
        self.state_bucket.grant_read_write(self.agentcore_role)

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
        self.lambda_role.add_to_policy(logs_statement)
        self.state_bucket.grant_read_write(self.lambda_role)
        self.dedup_table.grant_read_write_data(self.lambda_role)
        self.lambda_role.add_to_policy(
            iam.PolicyStatement(
                actions=["lambda:InvokeFunction"],
                resources=[f"arn:aws:lambda:{self.region}:{self.account}:function:{project_name}-{stage}-*"],
            )
        )
        self.lambda_role.add_to_policy(
            iam.PolicyStatement(
                actions=["bedrock-agentcore:InvokeAgentRuntime"],
                resources=[f"arn:aws:bedrock-agentcore:{self.region}:{self.account}:runtime/*"],
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

        memory_exec_role = iam.Role(
            self,
            "MemoryExecutionRole",
            assumed_by=iam.ServicePrincipal("bedrock-agentcore.amazonaws.com"),
        )
        memory_exec_role.add_to_policy(bedrock_invoke_statement)
        memory_exec_role.add_to_policy(bedrock_profile_statement)

        # Short-term event memory only: digest snapshots are stored/reloaded via
        # CreateEvent/ListEvents. Trend memory now lives in the structured trends.json
        # (StateStore), so the semantic-extraction strategy and RetrieveMemoryRecords
        # are no longer needed.
        self.memory = CfnMemory(
            self,
            "DigestMemory",
            name=f"{project_name}_{stage}_digest_state".replace("-", "_"),
            event_expiry_duration=90,
            description="OmniSummary digest snapshot state",
            memory_execution_role_arn=memory_exec_role.role_arn,
        )
        self.memory_id = self.memory.attr_memory_id

        memory_data_statement = iam.PolicyStatement(
            actions=[
                "bedrock-agentcore:CreateEvent",
                "bedrock-agentcore:ListEvents",
                "bedrock-agentcore:ListSessions",
            ],
            resources=[f"arn:aws:bedrock-agentcore:{self.region}:{self.account}:memory/*"],
        )
        self.lambda_role.add_to_policy(memory_data_statement)
        self.agentcore_role.add_to_policy(memory_data_statement)

        self.ecs_cluster = ecs.Cluster(self, "EcsCluster", vpc=self.vpc)

        namespace = sd.PrivateDnsNamespace(
            self,
            "ServiceNamespace",
            name="omnisummary.local",
            vpc=self.vpc,
        )

        rsshub_task = ecs.FargateTaskDefinition(self, "RSSHubTask", memory_limit_mib=2048, cpu=1024)
        rsshub_task.add_container(
            "RSSHubContainer",
            image=ecs.ContainerImage.from_registry("diygod/rsshub:latest"),
            port_mappings=[ecs.PortMapping(container_port=RSSHUB_PORT)],
            logging=ecs.LogDrivers.aws_logs(stream_prefix="rsshub"),
            environment={
                "NODE_ENV": "production",
                "CACHE_TYPE": "memory",
                "TWITTER_AUTH_TOKEN": os.environ.get("TWITTER_AUTH_TOKEN", ""),
                "TWITTER_CT0": os.environ.get("TWITTER_CT0", ""),
                "PROXY_URI": os.environ.get("RSSHUB_PROXY_URI", ""),
                "PROXY_STRATEGY": "all",
            },
        )

        self.rsshub_service = ecs.FargateService(
            self,
            "RSSHubService",
            cluster=self.ecs_cluster,
            task_definition=rsshub_task,
            desired_count=1,
            assign_public_ip=False,
            vpc_subnets=self.vpc_subnets,
            cloud_map_options=ecs.CloudMapOptions(
                name="rsshub",
                cloud_map_namespace=namespace,
                dns_record_type=sd.DnsRecordType.A,
            ),
        )

        self.codebuild_project = codebuild.Project(
            self,
            "ImageBuild",
            project_name=f"{project_name}-{stage}-agent-build",
            environment=codebuild.BuildEnvironment(
                build_image=codebuild.LinuxArmBuildImage.AMAZON_LINUX_2_STANDARD_3_0,
                compute_type=codebuild.ComputeType.LARGE,
                privileged=True,
            ),
            environment_variables={
                "AWS_ACCOUNT_ID": codebuild.BuildEnvironmentVariable(value=self.account),
                "AWS_DEFAULT_REGION": codebuild.BuildEnvironmentVariable(value=self.region),
                "IMAGE_REPO_NAME": codebuild.BuildEnvironmentVariable(value=self.ecr_repo.repository_name),
                "IMAGE_TAG": codebuild.BuildEnvironmentVariable(value="latest"),
            },
            build_spec=codebuild.BuildSpec.from_object(
                {
                    "version": "0.2",
                    "phases": {
                        "pre_build": {
                            "commands": [
                                "aws ecr get-login-password --region $AWS_DEFAULT_REGION | "
                                "docker login --username AWS --password-stdin "
                                "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com",
                            ]
                        },
                        "build": {
                            "commands": [
                                "docker build -t $IMAGE_REPO_NAME:$IMAGE_TAG .",
                                "docker tag $IMAGE_REPO_NAME:$IMAGE_TAG "
                                "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/"
                                "$IMAGE_REPO_NAME:$IMAGE_TAG",
                            ]
                        },
                        "post_build": {
                            "commands": [
                                "docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/"
                                "$IMAGE_REPO_NAME:$IMAGE_TAG",
                            ]
                        },
                    },
                }
            ),
        )
        self.ecr_repo.grant_push(self.codebuild_project)
