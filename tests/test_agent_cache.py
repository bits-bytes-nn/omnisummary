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
