from unittest.mock import MagicMock, patch


class TestAgentPromptCaching:
    def test_bedrock_model_built_with_auto_cache(self):
        captured = {}

        def fake_bedrock_model(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        with (
            patch("agent.agent.BedrockModel", side_effect=fake_bedrock_model),
            patch("agent.agent.Agent", return_value=MagicMock(tool_names=["t"])),
            patch("agent.agent.boto3.Session", return_value=MagicMock()),
            patch(
                "agent.agent.BedrockCrossRegionModelHelper.get_cross_region_model_id",
                return_value="global.anthropic.claude-sonnet-4-6",
            ),
        ):
            from agent.agent import create_digest_agent

            create_digest_agent()

        cache_config = captured.get("cache_config")
        assert cache_config is not None
        assert cache_config.strategy == "auto"


class TestAgentToolInjection:
    def test_injected_tools_override_default(self):
        captured = {}

        def fake_agent(**kwargs):
            captured.update(kwargs)
            return MagicMock(tool_names=["custom"])

        sentinel_tools = [object()]
        with (
            patch("agent.agent.BedrockModel", return_value=MagicMock()),
            patch("agent.agent.Agent", side_effect=fake_agent),
            patch("agent.agent.boto3.Session", return_value=MagicMock()),
            patch(
                "agent.agent.BedrockCrossRegionModelHelper.get_cross_region_model_id",
                return_value="global.anthropic.claude-sonnet-4-6",
            ),
        ):
            from agent.agent import create_digest_agent

            create_digest_agent(tools=sentinel_tools)

        assert captured.get("tools") is sentinel_tools

    def test_default_tools_used_when_none(self):
        captured = {}

        def fake_agent(**kwargs):
            captured.update(kwargs)
            return MagicMock(tool_names=["t"])

        with (
            patch("agent.agent.BedrockModel", return_value=MagicMock()),
            patch("agent.agent.Agent", side_effect=fake_agent),
            patch("agent.agent.boto3.Session", return_value=MagicMock()),
            patch(
                "agent.agent.BedrockCrossRegionModelHelper.get_cross_region_model_id",
                return_value="global.anthropic.claude-sonnet-4-6",
            ),
        ):
            from agent.agent import create_digest_agent

            create_digest_agent()

        assert len(captured.get("tools")) == 6


class TestAgentMaxTokensFallback:
    def test_warns_when_model_info_missing(self):
        with (
            patch("agent.agent.BedrockModel", return_value=MagicMock()),
            patch("agent.agent.Agent", return_value=MagicMock(tool_names=["t"])),
            patch("agent.agent.boto3.Session", return_value=MagicMock()),
            patch(
                "agent.agent.BedrockCrossRegionModelHelper.get_cross_region_model_id",
                return_value="global.anthropic.claude-sonnet-4-6",
            ),
            patch("agent.agent._LANGUAGE_MODEL_INFO", {}),
            patch("agent.agent.logger") as log,
        ):
            from agent.agent import create_digest_agent

            create_digest_agent()

        assert log.warning.called
