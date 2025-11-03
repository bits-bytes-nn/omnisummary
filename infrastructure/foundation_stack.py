from pathlib import Path
from typing import Any

from aws_cdk import (
    BundlingOptions,
    CustomResource,
    Duration,
    RemovalPolicy,
    Stack,
    Tags,
)
from aws_cdk import aws_codebuild as codebuild
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecr as ecr
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_s3_assets as s3_assets
from constructs import Construct

from shared import Config


class OmniSummaryFoundationStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        config: Config,
        **kwargs: Any,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.config = config
        self.project_name = config.resources.project_name
        self.stage = config.resources.stage
        self.project_root = Path(__file__).parent.parent

        self._add_tags()

        self._configure_vpc()

        self.ecr_repository = self._create_ecr_repository()
        self.event_dedup_table = self._create_event_deduplication_table()

        self.agentcore_role = self._create_agentcore_role()
        self.lambda_role = self._create_lambda_role()

        self.source_asset = self._create_source_asset()
        self.codebuild_project = self._create_codebuild_project()
        self.build_trigger = self._create_build_trigger()

    def _add_tags(self) -> None:
        for key, value in {
            "ProjectName": self.project_name,
            "Stage": self.stage,
            "CostCenter": self.project_name,
            "ManagedBy": "CDK",
        }.items():
            Tags.of(self).add(key, value)

    def _configure_vpc(self) -> None:
        vpc_id = self.config.resources.vpc_id
        subnet_ids = self.config.resources.subnet_ids

        if vpc_id and subnet_ids:
            self.vpc = ec2.Vpc.from_lookup(self, "BaseVPC", vpc_id=vpc_id)
            self.vpc_subnets = ec2.SubnetSelection(
                subnets=[ec2.Subnet.from_subnet_id(self, f"Subnet{i}", sid) for i, sid in enumerate(subnet_ids)]
            )
        else:
            self.vpc = ec2.Vpc(
                self,
                "BaseVPC",
                max_azs=2,
                nat_gateways=1,
                subnet_configuration=[
                    ec2.SubnetConfiguration(name="Public", subnet_type=ec2.SubnetType.PUBLIC, cidr_mask=24),
                    ec2.SubnetConfiguration(
                        name="Private",
                        subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                        cidr_mask=24,
                    ),
                ],
            )
            self.vpc_subnets = ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS)

    def _get_resource_name(self, suffix: str) -> str:
        return f"{self.project_name}-{self.stage}-{suffix}"

    def _create_ecr_repository(self) -> ecr.Repository:
        removal_policy = RemovalPolicy.DESTROY if self.config.resources.stage == "dev" else RemovalPolicy.RETAIN
        return ecr.Repository(
            self,
            "SummaryBotAgentRepository",
            repository_name=self._get_resource_name("summary-bot"),
            removal_policy=removal_policy,
            lifecycle_rules=[ecr.LifecycleRule(max_image_count=10, description="Keep last 10 images")],
        )

    def _create_event_deduplication_table(self) -> dynamodb.Table:
        removal_policy = RemovalPolicy.DESTROY if self.config.resources.stage == "dev" else RemovalPolicy.RETAIN
        return dynamodb.Table(
            self,
            "SummaryBotTable",
            table_name=self._get_resource_name("summary-bot"),
            partition_key=dynamodb.Attribute(name="event_id", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            time_to_live_attribute="ttl",
            removal_policy=removal_policy,
        )

    def _create_agentcore_role(self) -> iam.Role:
        managed_policies = [
            iam.ManagedPolicy.from_aws_managed_policy_name(name)
            for name in [
                "AmazonBedrockFullAccess",
                "AmazonEC2ContainerRegistryFullAccess",
                "AmazonSSMFullAccess",
                "CloudWatchLogsFullAccess",
            ]
        ]
        return iam.Role(
            self,
            "SummaryBotAgentCoreRole",
            assumed_by=iam.ServicePrincipal("bedrock-agentcore.amazonaws.com"),
            description="Execution role for Summary Bot AgentCore Runtime",
            managed_policies=managed_policies,
            role_name=self._get_resource_name("agentcore"),
        )

    def _create_lambda_role(self) -> iam.Role:
        managed_policies = [
            iam.ManagedPolicy.from_aws_managed_policy_name(name)
            for name in [
                "AmazonBedrockFullAccess",
                "AmazonDynamoDBFullAccess",
                "AmazonSSMFullAccess",
                "CloudWatchLogsFullAccess",
            ]
        ]
        role = iam.Role(
            self,
            "SummaryBotLambdaRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            description="Execution role for Summary Bot Slack Event Handler Lambda Function",
            managed_policies=managed_policies,
            role_name=self._get_resource_name("slack-event-handler-lambda"),
        )

        role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["bedrock-agentcore:InvokeAgentRuntime"],
                resources=["*"],
            )
        )

        role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["lambda:InvokeFunction"],
                resources=[
                    f"arn:aws:lambda:{self.region}:{self.account}:function:{self._get_resource_name('slack-event-handler-lambda')}"
                ],
            )
        )

        role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["dynamodb:PutItem", "dynamodb:GetItem", "dynamodb:Query"],
                resources=[
                    f"arn:aws:dynamodb:{self.region}:{self.account}:table/{self._get_resource_name('summary-bot')}"
                ],
            )
        )

        return role

    def _create_source_asset(self) -> s3_assets.Asset:
        return s3_assets.Asset(
            self,
            "SummaryBotSourceAsset",
            path=str(self.project_root),
            exclude=[
                "*.pyc",
                "__pycache__",
                "cdk.out",
                ".venv",
                "venv",
                ".mypy_cache",
                ".pytest_cache",
                ".git",
                "*.egg-info",
                ".claude",
                ".idea",
                ".run",
                "cdk.context.json",
                "docs",
                "logs",
            ],
        )

    def _create_codebuild_project(self) -> codebuild.Project:
        managed_policies = [
            iam.ManagedPolicy.from_aws_managed_policy_name(name)
            for name in [
                "AmazonEC2ContainerRegistryFullAccess",
                "AmazonS3FullAccess",
                "CloudWatchLogsFullAccess",
            ]
        ]
        role = iam.Role(
            self,
            "SummaryBotCodeBuildRole",
            role_name=self._get_resource_name("codebuild"),
            assumed_by=iam.ServicePrincipal("codebuild.amazonaws.com"),
            description="Execution role for Summary Bot CodeBuild project",
            managed_policies=managed_policies,
        )

        build_spec = codebuild.BuildSpec.from_object(
            {
                "version": "0.2",
                "phases": {
                    "pre_build": {
                        "commands": [
                            "echo Logging in to Amazon ECR...",
                            "aws ecr get-login-password --region $AWS_DEFAULT_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com",
                        ]
                    },
                    "build": {
                        "commands": [
                            "echo Build started on `date`",
                            "echo Building the Docker image for ARM64...",
                            "docker build -t $IMAGE_REPO_NAME:$IMAGE_TAG .",
                            "docker tag $IMAGE_REPO_NAME:$IMAGE_TAG $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/$IMAGE_REPO_NAME:$IMAGE_TAG",
                        ]
                    },
                    "post_build": {
                        "commands": [
                            "echo Build completed on `date`",
                            "echo Pushing the Docker image...",
                            "docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/$IMAGE_REPO_NAME:$IMAGE_TAG",
                            "echo ARM64 Docker image pushed successfully",
                        ]
                    },
                },
            }
        )

        return codebuild.Project(
            self,
            "SummaryBotImageBuildProject",
            project_name=self._get_resource_name("summary-bot"),
            description=f"Build agent Docker image for '{self.stack_name}'",
            role=role,
            environment=codebuild.BuildEnvironment(
                build_image=codebuild.LinuxArmBuildImage.AMAZON_LINUX_2_STANDARD_3_0,
                compute_type=codebuild.ComputeType.LARGE,
                privileged=True,
            ),
            source=codebuild.Source.s3(
                bucket=self.source_asset.bucket,
                path=self.source_asset.s3_object_key,
            ),
            build_spec=build_spec,
            environment_variables={
                "AWS_ACCOUNT_ID": codebuild.BuildEnvironmentVariable(value=self.account),
                "AWS_DEFAULT_REGION": codebuild.BuildEnvironmentVariable(
                    value=self.config.resources.default_region_name or self.region
                ),
                "IMAGE_REPO_NAME": codebuild.BuildEnvironmentVariable(value=self.ecr_repository.repository_name),
                "IMAGE_TAG": codebuild.BuildEnvironmentVariable(value=self.config.infrastructure.agentcore_image_tag),
                "STACK_NAME": codebuild.BuildEnvironmentVariable(value=self.stack_name),
            },
        )

    def _create_build_trigger(self) -> CustomResource:
        function = lambda_.Function(
            self,
            "SummaryBotBuildTriggerFunction",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="build_trigger.handler",
            timeout=Duration.minutes(15),
            code=lambda_.Code.from_asset(
                str(self.project_root / "lambda_handlers"),
                bundling=BundlingOptions(
                    image=lambda_.Runtime.PYTHON_3_12.bundling_image,
                    command=[
                        "bash",
                        "-c",
                        "pip install aws-lambda-powertools urllib3 boto3 -t /asset-output && " "cp -r . /asset-output",
                    ],
                ),
            ),
            initial_policy=[
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=["codebuild:StartBuild", "codebuild:BatchGetBuilds"],
                    resources=[self.codebuild_project.project_arn],
                )
            ],
            function_name=self._get_resource_name("build-trigger-lambda"),
        )

        return CustomResource(
            self,
            "SummaryBotBuildTrigger",
            service_token=function.function_arn,
            properties={"ProjectName": self.codebuild_project.project_name},
        )
