from __future__ import annotations

import os

from aws_cdk import RemovalPolicy, Stack, Tags
from aws_cdk import aws_codebuild as codebuild
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecr as ecr
from aws_cdk import aws_ecs as ecs
from aws_cdk import aws_iam as iam
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_servicediscovery as sd
from constructs import Construct

from shared import Config


class OmniSummaryFoundationStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        config: Config,
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
        )

        self.agentcore_role = iam.Role(
            self,
            "AgentCoreRole",
            assumed_by=iam.ServicePrincipal("bedrock-agentcore.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonBedrockFullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEC2ContainerRegistryReadOnly"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSSMReadOnlyAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("CloudWatchLogsFullAccess"),
            ],
        )
        self.state_bucket.grant_read_write(self.agentcore_role)

        self.lambda_role = iam.Role(
            self,
            "LambdaRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaVPCAccessExecutionRole"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonBedrockFullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSSMReadOnlyAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("CloudWatchLogsFullAccess"),
            ],
        )
        self.state_bucket.grant_read_write(self.lambda_role)
        self.dedup_table.grant_read_write_data(self.lambda_role)
        self.lambda_role.add_to_policy(iam.PolicyStatement(actions=["lambda:InvokeFunction"], resources=["*"]))

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
            port_mappings=[ecs.PortMapping(container_port=1200)],
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
