import json
from pathlib import Path

import pytest
from aws_cdk import App, Environment
from aws_cdk.assertions import Match, Template

from infrastructure.application_stack import OmniSummaryApplicationStack
from infrastructure.foundation_stack import OmniSummaryFoundationStack
from shared import Config
from shared.constants import ALL_SSM_SECRET_ENV_VARS, METRIC_NAMESPACE, RSSHUB_PORT, SSM_PLACEHOLDER
from shared.metrics import metric_dimensions

# The TRACKED config, matching what scripts/ci_synth.py synths: config/config.yaml is gitignored,
# so asserting against Config.load() checked a different stack locally than in CI (where it fell
# back to bare code defaults).
CONFIG_TEMPLATE = Path(__file__).resolve().parent.parent / "config" / "config-template.yaml"

# CloudWatch rejects PutMetricAlarm when Period * EvaluationPeriods exceeds one day; the empty-digest
# alarm sits right on that edge (a 24h period), which is why application_stack carries a note about it.
_CLOUDWATCH_MAX_ALARM_WINDOW_SEC = 86400
# The share of a function's configured Timeout its Duration alarm fires at (application_stack's 0.9):
# a timeout does not count as an Error, so this is the only signal for "ran out of time mid-post".
_TIMEOUT_ALARM_THRESHOLD_RATIO = 0.9


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
        # Equality, not a subset: a stack-created secret parameter that scripts/put_secrets.py does
        # not know about would stay a plaintext String placeholder forever, and nothing else would
        # ever notice — the "these are SecureStrings" claim has to cover every one of them.
        assert names == set(ALL_SSM_SECRET_ENV_VARS)

    def test_secret_parameter_properties_are_pinned(self, templates):
        # put_secrets.py owns these parameters' VALUES out-of-band (it deletes the placeholder String
        # and re-creates it as a SecureString). CloudFormation only leaves that alone while the
        # resource's template properties are unchanged — so ANY new/renamed property here would make
        # the next deploy write the placeholder back over the live secret. Pin the property set so
        # such an edit fails here instead of in production.
        foundation, application = templates
        for template in (foundation, application):
            for logical_id, resource in template.find_resources("AWS::SSM::Parameter").items():
                # Tags are stack-wide and already deployed; everything else is pinned.
                assert set(resource["Properties"]) <= {"Name", "Type", "Value", "Tags"}, logical_id
                assert {"Name", "Type", "Value"} <= set(resource["Properties"]), logical_id
                assert resource["Properties"]["Type"] == "String", logical_id

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

    def test_slack_role_reads_only_its_two_slack_parameters(self, templates):
        # The shared ssm_read_statement covers parameter/{project}/{stage}/* — the OpenAI/Tavily/
        # YouTube keys, the Threads token and the X cookies. The handler reads exactly the signing
        # secret and the bot token, so the only internet-reachable component gets exactly those.
        foundation, _ = templates
        resources: list[str] = []
        for props in foundation.find_resources("AWS::IAM::Policy").values():
            if "SlackEventRole" not in json.dumps(props["Properties"].get("Roles", "")):
                continue
            for statement in props["Properties"]["PolicyDocument"]["Statement"]:
                action = statement.get("Action")
                actions = [action] if isinstance(action, str) else action
                if not any(a.startswith("ssm:") for a in actions):
                    continue
                resource = statement.get("Resource")
                resources.extend([resource] if isinstance(resource, str) else resource)
        assert resources, "expected an ssm statement on SlackEventRole"
        assert not [r for r in resources if r.endswith("/dev/*")]
        suffixes = sorted(r.rsplit("/", 1)[-1] for r in resources)
        assert suffixes == ["slack-bot-token", "slack-signing-secret"]


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
        # BOTH templates: inspecting only the foundation's roles missed the four Custom::LogRetention
        # providers the deprecated `log_retention=` prop rendered into the APPLICATION stack, each
        # with logs:PutRetentionPolicy + logs:DeleteRetentionPolicy on Resource "*" — an
        # audit-trail-tampering primitive in a shared account.
        sensitive = (
            "ssm:GetParameter",
            "logs:PutLogEvents",
            "logs:PutRetentionPolicy",
            "logs:DeleteRetentionPolicy",
            "bedrock-agentcore:CreateEvent",
        )
        for template in templates:
            for policy in template.find_resources("AWS::IAM::Policy").values():
                for stmt in policy["Properties"]["PolicyDocument"]["Statement"]:
                    actions = stmt.get("Action", [])
                    actions = [actions] if isinstance(actions, str) else actions
                    if any(any(s in a for s in sensitive) for a in actions):
                        assert stmt.get("Resource") != "*", f"sensitive action scoped to *: {actions}"

    def test_no_log_retention_custom_resource(self, templates):
        # `log_retention=` is deprecated and does not set a property: it renders one Lambda + role per
        # function purely to call PutRetentionPolicy. Declaring the LogGroup deletes all of it.
        for template in templates:
            template.resource_count_is("Custom::LogRetention", 0)

    def test_every_log_group_has_a_retention(self, templates):
        # The RSSHub container's group had no RetentionInDays AND DeletionPolicy: Retain — the one log
        # group in either stack kept forever. Assert over BOTH templates so the next log producer
        # cannot reintroduce an unbounded one.
        for template in templates:
            groups = template.find_resources("AWS::Logs::LogGroup")
            assert groups
            for logical_id, group in groups.items():
                assert group["Properties"].get("RetentionInDays"), f"{logical_id} retains logs forever"

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

    def test_memory_data_plane_scoped_to_this_stacks_memory(self, templates):
        # memory/* handed both roles the event history of every AgentCore memory in the account,
        # although this stack's own memory ARN is in scope where the statement is built.
        foundation, _ = templates
        memory_logical_id = next(iter(foundation.find_resources("AWS::BedrockAgentCore::Memory")))
        for props in foundation.find_resources("AWS::IAM::Policy").values():
            for statement in props["Properties"]["PolicyDocument"]["Statement"]:
                action = statement.get("Action")
                actions = [action] if isinstance(action, str) else action
                if "bedrock-agentcore:CreateEvent" not in actions:
                    continue
                rendered_resource = json.dumps(statement["Resource"])
                assert ":memory/*" not in rendered_resource
                assert memory_logical_id in rendered_resource


class TestNoPrivilegedNoOpBuildProject:
    def test_no_codebuild_project_is_created(self, templates):
        # The stack used to create a privileged CodeBuild project (docker-in-docker) with ecr
        # grant_push and NO source, so its `docker build .` ran in an empty directory. Nothing in the
        # repo referenced it and the README's resource table omitted it, yet docs/design.md advertised
        # it as a capability — a privileged no-op is strictly worse than no build project.
        foundation, application = templates
        for template in (foundation, application):
            template.resource_count_is("AWS::CodeBuild::Project", 0)


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

    def test_every_alarm_notifies_the_alerts_topic(self, templates):
        # The only alarm assertions were a resource COUNT and three MetricName checks — nothing said
        # an alarm actually notifies anyone. An alarm with no AlarmActions is a dashboard widget:
        # it goes red and no one is told, which is indistinguishable from having no alarm at all.
        _, app = templates
        alarms = app.find_resources("AWS::CloudWatch::Alarm")
        assert alarms
        for logical_id, alarm in alarms.items():
            actions = alarm["Properties"].get("AlarmActions") or []
            assert actions, f"{logical_id} has no AlarmActions"
            assert all("Ref" in action or "Fn::ImportValue" in action for action in actions), logical_id

    def test_no_alarm_exceeds_cloudwatchs_one_day_evaluation_window(self, templates):
        # application_stack carries an explicit deploy-time hazard note: CloudWatch rejects
        # PutMetricAlarm when Period * EvaluationPeriods exceeds 86400s. Nothing checked it, so the
        # template synthesized clean and was rejected at deploy — the worst place to find out.
        _, app = templates
        for logical_id, alarm in app.find_resources("AWS::CloudWatch::Alarm").items():
            props = alarm["Properties"]
            window = int(props["Period"]) * int(props["EvaluationPeriods"])
            assert window <= _CLOUDWATCH_MAX_ALARM_WINDOW_SEC, f"{logical_id} evaluates {window}s"

    def test_each_timeout_alarm_tracks_its_own_functions_timeout(self, templates):
        # A timeout alarm whose threshold drifts from the function's configured Timeout either fires
        # on healthy runs or never fires at all. Read BOTH numbers out of the template so a Duration
        # change on one Lambda cannot silently leave its alarm behind.
        _, app = templates
        functions = app.find_resources("AWS::Lambda::Function")
        timeout_alarms = {
            logical_id: alarm
            for logical_id, alarm in app.find_resources("AWS::CloudWatch::Alarm").items()
            if alarm["Properties"].get("MetricName") == "Duration"
        }
        assert timeout_alarms
        for logical_id, alarm in timeout_alarms.items():
            dimensions = {d["Name"]: d["Value"] for d in alarm["Properties"]["Dimensions"]}
            # The dimension is a Ref to the function resource, so the alarm and the Timeout it must
            # track are read out of the SAME template rather than from a literal repeated in the test.
            function_ref = dimensions["FunctionName"]["Ref"]
            timeout_sec = functions[function_ref]["Properties"]["Timeout"]
            assert alarm["Properties"]["Threshold"] == pytest.approx(
                timeout_sec * 1000 * _TIMEOUT_ALARM_THRESHOLD_RATIO
            ), logical_id

    def test_emf_alarms_read_this_deployments_datapoints_only(self, templates):
        # The EMF records are dimensioned by project/stage (shared/metrics.py). An undimensioned
        # alarm aggregates every deployment into one series: a dev run of 5 items kept prod's
        # Maximum<1 empty-digest alarm green on a day prod shipped nothing, and dev agent failures
        # paged on prod's error alarm.
        _, app = templates
        expected = [{"Name": name, "Value": value} for name, value in metric_dimensions("omnisummary", "dev").items()]
        assert expected, "the dimension map must not be empty"
        for metric_name in ("DigestItemsPublished", "AgentErrors"):
            app.has_resource_properties(
                "AWS::CloudWatch::Alarm",
                {"Namespace": METRIC_NAMESPACE, "MetricName": metric_name, "Dimensions": expected},
            )

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
    @classmethod
    def prefixed_foundation(cls):
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


class TestBedrockCostAttributionPermissions:
    """The model resolver prefers this project's APPLICATION inference profile so on-demand token
    spend carries the Project cost-allocation tag (there is no taggable resource behind InvokeModel
    otherwise). That ARN is a DIFFERENT resource type from the system-defined inference profiles, so
    omitting it makes every Bedrock call AccessDenied the moment a profile exists — the whole digest."""

    def test_invoke_covers_application_inference_profiles(self, templates):
        foundation, _ = templates
        for policy in foundation.find_resources("AWS::IAM::Policy").values():
            for statement in policy["Properties"]["PolicyDocument"]["Statement"]:
                actions = statement.get("Action", [])
                actions = [actions] if isinstance(actions, str) else actions
                if "bedrock:InvokeModel" not in actions:
                    continue
                rendered = json.dumps(statement.get("Resource", []))
                assert "application-inference-profile/*" in rendered
                assert "inference-profile/*" in rendered  # system-defined ones still work
                return
        raise AssertionError("no statement granting bedrock:InvokeModel was found")


# A pushed image digest, as `export DIGEST_IMAGE_REF=sha256:...` supplies it.
_PUSHED_DIGEST = "sha256:" + "ab" * 32


@pytest.fixture(scope="module")
def pinned_template():
    """The stack as a real deploy renders it: both image refs pinned to a pushed digest.

    The default fixture passes NO refs, so every assertion in this file runs against the `latest`
    rendering and the pinned branch never renders at all — while pinning is what makes a deploy
    take effect. CloudFormation only updates a Lambda whose template properties CHANGED, so a
    constant tag string means the function silently keeps running last week's image; that is a
    recorded deploy-only failure mode, invisible to a test that never asks for a pin."""
    config = Config.from_yaml(str(CONFIG_TEMPLATE))
    env = Environment(account="123456789012", region=config.aws.region)
    app = App()
    foundation = OmniSummaryFoundationStack(app, "fnd-pinned", config=config, alert_email="a@example.com", env=env)
    application = OmniSummaryApplicationStack(
        app,
        "app-pinned",
        config=config,
        foundation=foundation,
        agentcore_image_ref=_PUSHED_DIGEST,
        digest_image_ref=_PUSHED_DIGEST,
        env=env,
    )
    return Template.from_stack(application)


class TestThePushedImageDigestReachesEveryFunction:
    """Three Lambdas share one image, and each takes the pin separately. A refactor that drops it on
    one of them ships with CI green unless the digest rendering is asserted per function."""

    @staticmethod
    def _image_uris(template):
        # Only the image-backed functions; the Slack ingress ships as a zip (no ImageUri at all).
        funcs = template.find_resources("AWS::Lambda::Function")
        return {
            logical_id: json.dumps(func["Properties"]["Code"]["ImageUri"])
            for logical_id, func in funcs.items()
            if "ImageUri" in func["Properties"]["Code"]
        }

    def test_every_lambda_image_uri_carries_the_digest(self, pinned_template):
        image_uris = self._image_uris(pinned_template)
        # digest + visual + threads-refresh: each takes the pin separately.
        assert len(image_uris) == 3, image_uris
        for logical_id, uri in image_uris.items():
            assert f"@{_PUSHED_DIGEST}" in uri, logical_id
            assert ":latest" not in uri, logical_id

    def test_the_unpinned_stack_falls_back_to_the_latest_tag(self, templates):
        _, app = templates
        image_uris = self._image_uris(app)
        assert len(image_uris) == 3, image_uris
        for logical_id, uri in image_uris.items():
            assert ":latest" in uri, logical_id

    def test_the_agentcore_runtime_uses_at_for_a_digest_and_colon_for_a_tag(self, pinned_template, templates):
        pinned = pinned_template.find_resources("AWS::BedrockAgentCore::Runtime")
        uri = json.dumps(next(iter(pinned.values()))["Properties"]["AgentRuntimeArtifact"])
        assert f"@{_PUSHED_DIGEST}" in uri
        _, app = templates
        default = app.find_resources("AWS::BedrockAgentCore::Runtime")
        assert ":arm64" in json.dumps(next(iter(default.values()))["Properties"]["AgentRuntimeArtifact"])
