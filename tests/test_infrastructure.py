import pytest
from aws_cdk import App, Environment
from aws_cdk.assertions import Match, Template

from infrastructure.application_stack import OmniSummaryApplicationStack
from infrastructure.foundation_stack import OmniSummaryFoundationStack
from shared import Config


@pytest.fixture(scope="module")
def templates():
    config = Config.load()
    config.aws.state_bucket_name = ""  # force CDK-created bucket to assert hardening
    env = Environment(account="123456789012", region=config.aws.region)
    app = App()
    foundation = OmniSummaryFoundationStack(app, "fnd", config=config, alert_email="alerts@example.com", env=env)
    application = OmniSummaryApplicationStack(
        app,
        "app",
        config=config,
        foundation=foundation,
        reddit_client_id="rid",
        reddit_client_secret="rsec",
        openai_api_key="oai",
        tavily_api_key="tav",
        env=env,
    )
    return Template.from_stack(foundation), Template.from_stack(application)


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

    def test_agentcore_memory_resource(self, templates):
        foundation, _ = templates
        foundation.resource_count_is("AWS::BedrockAgentCore::Memory", 1)

    def test_memory_data_plane_permissions(self, templates):
        foundation, _ = templates
        rendered = str(foundation.find_resources("AWS::IAM::Policy"))
        assert "bedrock-agentcore:CreateEvent" in rendered
        assert "bedrock-agentcore:RetrieveMemoryRecords" in rendered


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

    def test_cloudwatch_alarms(self, templates):
        _, app = templates
        # 2 lambda error alarms + 1 api 5xx alarm
        app.resource_count_is("AWS::CloudWatch::Alarm", 3)

    def test_reddit_and_openai_ssm_params(self, templates):
        _, app = templates
        params = app.find_resources("AWS::SSM::Parameter")
        names = {v["Properties"]["Name"] for v in params.values()}
        assert "/omnisummary/dev/reddit-client-id" in names
        assert "/omnisummary/dev/reddit-client-secret" in names
        assert "/omnisummary/dev/openai-api-key" in names

    def test_digest_lambda_has_alert_topic_env(self, templates):
        _, app = templates
        funcs = app.find_resources("AWS::Lambda::Function")
        has_env = any(
            "ALERT_SNS_TOPIC_ARN" in v["Properties"].get("Environment", {}).get("Variables", {}) for v in funcs.values()
        )
        assert has_env

    def test_api_gateway_throttling(self, templates):
        _, app = templates
        app.has_resource_properties(
            "AWS::ApiGateway::Stage",
            {"MethodSettings": Match.array_with([Match.object_like({"ThrottlingRateLimit": Match.any_value()})])},
        )
