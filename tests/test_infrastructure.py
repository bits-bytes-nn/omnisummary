import json
from pathlib import Path

import pytest
from aws_cdk import App, Environment
from aws_cdk.assertions import Match, Template

from infrastructure.application_stack import OmniSummaryApplicationStack
from infrastructure.foundation_stack import OmniSummaryFoundationStack
from shared import Config
from shared.constants import ALL_SSM_SECRET_ENV_VARS, RSSHUB_PORT, SSM_PLACEHOLDER

# The TRACKED config, matching what scripts/ci_synth.py synths: config/config.yaml is gitignored,
# so asserting against Config.load() checked a different stack locally than in CI (where it fell
# back to bare code defaults).
CONFIG_TEMPLATE = Path(__file__).resolve().parent.parent / "config" / "config-template.yaml"


@pytest.fixture(scope="module")
def templates():
    config = Config.from_yaml(str(CONFIG_TEMPLATE))
    config.aws.state_bucket_name = ""  # force CDK-created bucket to assert hardening
    env = Environment(account="123456789012", region=config.aws.region)
    app = App()
    foundation = OmniSummaryFoundationStack(app, "fnd", config=config, alert_email="alerts@example.com", env=env)
    application = OmniSummaryApplicationStack(
        app,
        "app",
        config=config,
        foundation=foundation,
        env=env,
    )
    return Template.from_stack(foundation), Template.from_stack(application)


class TestNoSecretsInTheTemplate:
    """A CloudFormation template is not a secret store: it is written to cdk.out, uploaded to the
    CDK staging bucket, and returned verbatim by cloudformation:GetTemplate. The stack used to pass
    the real Slack bot token, Tavily/OpenAI/YouTube keys, the Threads access token and the X session
    cookies straight into it. Only placeholders and ARNs may appear."""

    def test_every_ssm_parameter_holds_only_the_placeholder(self, templates):
        foundation, application = templates
        for template in (foundation, application):
            params = template.find_resources("AWS::SSM::Parameter")
            for logical_id, resource in params.items():
                assert resource["Properties"]["Value"] == SSM_PLACEHOLDER, logical_id

    def test_all_expected_secret_parameters_exist(self, templates):
        foundation, application = templates
        names = set()
        for template in (foundation, application):
            for resource in template.find_resources("AWS::SSM::Parameter").values():
                names.add(resource["Properties"]["Name"].rsplit("/", 1)[-1])
        assert set(ALL_SSM_SECRET_ENV_VARS) <= names

    def test_rsshub_reads_x_cookies_as_secrets_not_plain_environment(self, templates):
        foundation, _ = templates
        task_defs = foundation.find_resources("AWS::ECS::TaskDefinition")
        container = next(iter(task_defs.values()))["Properties"]["ContainerDefinitions"][0]
        env_names = {e["Name"] for e in container.get("Environment", [])}
        assert not {"TWITTER_AUTH_TOKEN", "TWITTER_CT0"} & env_names
        assert {s["Name"] for s in container["Secrets"]} == {"TWITTER_AUTH_TOKEN", "TWITTER_CT0"}


class TestRSSHubServiceScale:
    def test_desired_count_comes_from_config_and_defaults_to_zero(self, templates):
        # The digest never reaches this service: RSSHubCollector returns the S3 park file before it
        # would even probe RSSHub (see TestParkedItems), and the local sync cron refreshes that file
        # before every run — so a running task is pure cost. The task DEFINITION stays deployed, so
        # raising aws.rsshub_desired_count to 1 restores the in-AWS fallback.
        foundation, _ = templates
        services = foundation.find_resources("AWS::ECS::Service")
        assert len(services) == 1
        assert next(iter(services.values()))["Properties"]["DesiredCount"] == 0
        foundation.resource_count_is("AWS::ECS::TaskDefinition", 1)


class TestSlackLambdaLeastPrivilege:
    """The Slack-events Lambda is the only internet-reachable entry point, and it used to run with
    the pipeline's role — Bedrock model invocation, the state bucket, SNS publish, ssm:PutParameter
    on the Threads token, and InvokeFunction on every function in the project."""

    def test_slack_lambda_does_not_share_the_pipeline_role(self, templates):
        _, application = templates
        fns = application.find_resources("AWS::Lambda::Function")
        roles = {
            props["Properties"].get("FunctionName", ""): json.dumps(props["Properties"]["Role"])
            for props in fns.values()
            if props["Properties"].get("FunctionName")
        }
        slack = next(v for k, v in roles.items() if k.endswith("slack-events"))
        pipeline = next(v for k, v in roles.items() if k.endswith("-digest"))
        assert "SlackEventRole" in slack
        assert slack != pipeline

    def test_slack_role_has_no_bedrock_model_or_bucket_access(self, templates):
        foundation, _ = templates
        policies = foundation.find_resources("AWS::IAM::Policy")
        actions: list[str] = []
        for props in policies.values():
            if "SlackEventRole" not in json.dumps(props["Properties"].get("Roles", "")):
                continue
            for statement in props["Properties"]["PolicyDocument"]["Statement"]:
                action = statement.get("Action")
                actions.extend([action] if isinstance(action, str) else action)
        assert actions, "expected an inline policy attached to SlackEventRole"
        assert not [a for a in actions if a.startswith("bedrock:")]
        assert not [a for a in actions if a.startswith("s3:")]
        assert not [a for a in actions if a.startswith("sns:")]
        assert "ssm:PutParameter" not in actions
        # It still needs its own four capabilities.
        assert "bedrock-agentcore:InvokeAgentRuntime" in actions
        assert "lambda:InvokeFunction" in actions
        assert "ssm:GetParameter" in actions


class TestFoundationStack:
    def test_sns_topic_created(self, templates):
        foundation, _ = templates
        foundation.resource_count_is("AWS::SNS::Topic", 1)

    def test_email_subscription(self, templates):
        foundation, _ = templates
        foundation.has_resource_properties(
            "AWS::SNS::Subscription",
            {"Protocol": "email", "Endpoint": "alerts@example.com"},
        )

    def test_s3_bucket_encrypted_and_versioned(self, templates):
        foundation, _ = templates
        foundation.has_resource_properties(
            "AWS::S3::Bucket",
            {
                "VersioningConfiguration": {"Status": "Enabled"},
                "BucketEncryption": Match.any_value(),
                "PublicAccessBlockConfiguration": Match.any_value(),
            },
        )

    def test_dynamodb_encrypted(self, templates):
        foundation, _ = templates
        foundation.has_resource_properties(
            "AWS::DynamoDB::Table",
            {"SSESpecification": {"SSEEnabled": True}},
        )

    def test_no_broad_managed_policies(self, templates):
        foundation, _ = templates
        rendered = str(foundation.find_resources("AWS::IAM::Role"))
        assert "AmazonSSMReadOnlyAccess" not in rendered
        assert "AmazonBedrockFullAccess" not in rendered
        assert "CloudWatchLogsFullAccess" not in rendered

    def test_scoped_logs_policy(self, templates):
        foundation, _ = templates
        rendered = str(foundation.find_resources("AWS::IAM::Policy"))
        assert "logs:CreateLogStream" in rendered
        assert "logs:PutLogEvents" in rendered

    def test_scoped_ssm_policy(self, templates):
        foundation, _ = templates
        rendered = str(foundation.find_resources("AWS::IAM::Policy"))
        assert "ssm:GetParameter" in rendered
        assert "bedrock:InvokeModel" in rendered
        assert "bedrock:ListInferenceProfiles" in rendered

    def test_sensitive_actions_not_wildcard_resource(self, templates):
        foundation, _ = templates
        policies = foundation.find_resources("AWS::IAM::Policy")
        sensitive = ("ssm:GetParameter", "logs:PutLogEvents", "bedrock-agentcore:CreateEvent")
        for policy in policies.values():
            for stmt in policy["Properties"]["PolicyDocument"]["Statement"]:
                actions = stmt.get("Action", [])
                actions = [actions] if isinstance(actions, str) else actions
                if any(any(s in a for s in sensitive) for a in actions):
                    assert stmt.get("Resource") != "*", f"sensitive action scoped to *: {actions}"

    def test_ssm_resource_scoped_to_project_path(self, templates):
        foundation, _ = templates
        rendered = str(foundation.find_resources("AWS::IAM::Policy"))
        # the scoped SSM ARN must reference the /{project}/{stage}/ parameter path
        assert "parameter/omnisummary/dev/" in rendered

    def test_agentcore_memory_resource(self, templates):
        foundation, _ = templates
        foundation.resource_count_is("AWS::BedrockAgentCore::Memory", 1)

    def test_memory_data_plane_permissions(self, templates):
        foundation, _ = templates
        rendered = str(foundation.find_resources("AWS::IAM::Policy"))
        assert "bedrock-agentcore:CreateEvent" in rendered
        # Recall is gone (trends live in trends.json); RetrieveMemoryRecords removed.
        assert "bedrock-agentcore:RetrieveMemoryRecords" not in rendered


class TestApplicationStack:
    def test_waf_web_acl(self, templates):
        _, app = templates
        app.resource_count_is("AWS::WAFv2::WebACL", 1)
        app.resource_count_is("AWS::WAFv2::WebACLAssociation", 1)

    def test_waf_has_rate_limit_rule(self, templates):
        _, app = templates
        acls = app.find_resources("AWS::WAFv2::WebACL")
        rendered = str(acls)
        assert "RateBasedStatement" in rendered
        assert "AWSManagedRulesCommonRuleSet" in rendered

    def test_waf_enforces_block_not_just_monitor(self, templates):
        _, app = templates
        acl = next(iter(app.find_resources("AWS::WAFv2::WebACL").values()))
        props = acl["Properties"]
        # default allow, rate-limit rule actually blocks (not Count/monitor-only)
        assert "Allow" in props["DefaultAction"]
        rate_rule = next(r for r in props["Rules"] if r["Name"] == "RateLimit")
        assert "Block" in rate_rule["Action"]
        assert rate_rule["Statement"]["RateBasedStatement"]["Limit"] > 0

    def test_cloudwatch_alarms(self, templates):
        _, app = templates
        # 4 lambdas (digest, slack, visual, threads-refresh) × (errors + timeout) + api 5xx
        # + empty-digest + async-DLQ + agent-errors = 12
        app.resource_count_is("AWS::CloudWatch::Alarm", 12)
        # the symptomless-failure alarms specifically exist (count alone wouldn't catch a swap)
        app.has_resource_properties("AWS::CloudWatch::Alarm", {"MetricName": "DigestItemsPublished"})
        app.has_resource_properties("AWS::CloudWatch::Alarm", {"MetricName": "AgentErrors"})
        app.has_resource_properties("AWS::CloudWatch::Alarm", {"MetricName": "ApproximateNumberOfMessagesVisible"})

    def test_openai_ssm_param(self, templates):
        _, app = templates
        params = app.find_resources("AWS::SSM::Parameter")
        names = {v["Properties"]["Name"] for v in params.values()}
        assert "/omnisummary/dev/openai-api-key" in names

    def test_digest_lambda_has_alert_topic_env(self, templates):
        _, app = templates
        funcs = app.find_resources("AWS::Lambda::Function")
        has_env = any(
            "ALERT_SNS_TOPIC_ARN" in v["Properties"].get("Environment", {}).get("Variables", {}) for v in funcs.values()
        )
        assert has_env

    def test_rsshub_ingress_from_digest_lambda(self, templates):
        # Without an ingress rule on the RSSHub service SG the digest Lambda cannot reach the
        # Fargate RSSHub service at all (every X feed fetch times out).
        _, app = templates
        app.has_resource_properties(
            "AWS::EC2::SecurityGroupIngress",
            {
                "IpProtocol": "tcp",
                "FromPort": RSSHUB_PORT,
                "ToPort": RSSHUB_PORT,
                "GroupId": Match.any_value(),
                "SourceSecurityGroupId": Match.any_value(),
            },
        )

    def test_every_lambda_disables_async_retries(self, templates):
        # The handlers re-raise so Errors alarms / the DLQ fire; retries must stay off on ALL of
        # them (a retried digest re-posts, a retried refresh re-calls the Threads endpoint).
        _, app = templates
        configs = app.find_resources("AWS::Lambda::EventInvokeConfig")
        assert len(configs) == 4  # digest, visual, slack-events, threads-refresh
        assert all(v["Properties"]["MaximumRetryAttempts"] == 0 for v in configs.values())

    def test_api_gateway_throttling(self, templates):
        _, app = templates
        app.has_resource_properties(
            "AWS::ApiGateway::Stage",
            {"MethodSettings": Match.array_with([Match.object_like({"ThrottlingRateLimit": Match.any_value()})])},
        )


class TestLeastPrivilegeGrants:
    """The pipeline role held s3:* on the WHOLE bucket (which can be a pre-existing shared one) and
    lambda:InvokeFunction on every {project}-{stage}-* function, including the internet-facing Slack
    handler. Both are scoped down."""

    @pytest.fixture(scope="class")
    def prefixed_foundation(self):
        config = Config.from_yaml(str(CONFIG_TEMPLATE))
        config.aws.state_bucket_name = ""
        config.aws.s3_prefix = "omnisummary"
        app = App()
        stack = OmniSummaryFoundationStack(
            app,
            "fnd-prefixed",
            config=config,
            env=Environment(account="123456789012", region=config.aws.region),
        )
        return Template.from_stack(stack)

    @staticmethod
    def _statements(template):
        for policy in template.find_resources("AWS::IAM::Policy").values():
            yield from policy["Properties"]["PolicyDocument"]["Statement"]

    def test_object_access_is_scoped_to_the_project_prefix(self, prefixed_foundation):
        object_arns = [
            json.dumps(r)
            for stmt in self._statements(prefixed_foundation)
            for a in (stmt["Action"] if isinstance(stmt["Action"], list) else [stmt["Action"]])
            if a.startswith("s3:GetObject") or a.startswith("s3:PutObject")
            for r in (stmt["Resource"] if isinstance(stmt["Resource"], list) else [stmt["Resource"]])
        ]
        keyspaces = [arn for arn in object_arns if "/*" in arn]
        assert keyspaces, "expected S3 object grants"
        # Every object key space must carry the project prefix — state_store's digest_state, the
        # collectors' park files and the daily visual's Threads images all live under it. A bare
        # "<bucket>/*" would be the whole (possibly shared, pre-existing) bucket. The bucket-level
        # ARN itself stays for List, which CDK adds and which is deliberately left alone.
        assert all('"/omnisummary/*"' in arn for arn in keyspaces), keyspaces

    def test_invoke_function_is_scoped_to_the_visual_function(self, templates):
        foundation, _ = templates
        invoke_resources = [
            stmt["Resource"]
            for stmt in self._statements(foundation)
            if "lambda:InvokeFunction" in (stmt["Action"] if isinstance(stmt["Action"], list) else [stmt["Action"]])
        ]
        rendered = json.dumps(invoke_resources)
        assert "function:omnisummary-dev-visual" in rendered
        assert "function:omnisummary-dev-*" not in rendered

    def test_visual_lambda_can_publish_the_delivery_alert(self, templates):
        # The visual Lambda is the Threads publish path, so it is what notices a partial reply
        # chain; without the topic ARN in its env the notice is a silent no-op.
        _, app = templates
        funcs = app.find_resources("AWS::Lambda::Function")
        visual = next(v for v in funcs.values() if v["Properties"].get("FunctionName") == "omnisummary-dev-visual")
        assert "ALERT_SNS_TOPIC_ARN" in visual["Properties"]["Environment"]["Variables"]
