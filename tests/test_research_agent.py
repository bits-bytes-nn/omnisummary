from unittest.mock import MagicMock, patch


class TestAgentPromptCaching:
    def test_bedrock_model_built_with_auto_cache(self):
        captured = {}

        def fake_bedrock_model(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        with (
            patch("agent.research_agent.BedrockModel", side_effect=fake_bedrock_model),
            patch("agent.research_agent.Agent", return_value=MagicMock(tool_names=["t"])),
            patch("agent.research_agent.boto3.Session", return_value=MagicMock()),
            patch(
                "agent.research_agent.BedrockCrossRegionModelHelper.get_cross_region_model_id",
                return_value="global.anthropic.claude-sonnet-4-6",
            ),
        ):
            from agent.research_agent import create_research_agent

            create_research_agent()

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
            patch("agent.research_agent.BedrockModel", return_value=MagicMock()),
            patch("agent.research_agent.Agent", side_effect=fake_agent),
            patch("agent.research_agent.boto3.Session", return_value=MagicMock()),
            patch(
                "agent.research_agent.BedrockCrossRegionModelHelper.get_cross_region_model_id",
                return_value="global.anthropic.claude-sonnet-4-6",
            ),
        ):
            from agent.research_agent import create_research_agent

            create_research_agent(tools=sentinel_tools)

        assert captured.get("tools") is sentinel_tools

    def test_default_tools_used_when_none(self):
        captured = {}

        def fake_agent(**kwargs):
            captured.update(kwargs)
            return MagicMock(tool_names=["t"])

        with (
            patch("agent.research_agent.BedrockModel", return_value=MagicMock()),
            patch("agent.research_agent.Agent", side_effect=fake_agent),
            patch("agent.research_agent.boto3.Session", return_value=MagicMock()),
            patch(
                "agent.research_agent.BedrockCrossRegionModelHelper.get_cross_region_model_id",
                return_value="global.anthropic.claude-sonnet-4-6",
            ),
        ):
            from agent.research_agent import create_research_agent

            create_research_agent()

        # Assert tool IDENTITIES, not a magic count — a 1-for-1 wrong swap would pass a count check.
        from agent.research_tools import (
            attach_image,
            community_search,
            deliver_report,
            read_url,
            recall_trends,
            search_papers,
            web_search,
        )

        expected = {
            web_search,
            community_search,
            search_papers,
            read_url,
            recall_trends,
            attach_image,
            deliver_report,
        }
        assert set(captured.get("tools")) == expected


class TestVoiceInjection:
    def test_persona_and_knobs_injected_into_prompt(self):
        captured = {}

        def fake_agent(**kwargs):
            captured.update(kwargs)
            return MagicMock(tool_names=["t"])

        with (
            patch("agent.research_agent.BedrockModel", return_value=MagicMock()),
            patch("agent.research_agent.Agent", side_effect=fake_agent),
            patch("agent.research_agent.boto3.Session", return_value=MagicMock()),
            patch(
                "agent.research_agent.BedrockCrossRegionModelHelper.get_cross_region_model_id",
                return_value="global.anthropic.claude-sonnet-4-6",
            ),
        ):
            from agent.research_agent import create_research_agent
            from shared import Config

            create_research_agent()
            cfg = Config.load()

        prompt = captured["system_prompt"]
        # All placeholders substituted — a dropped/mis-named key would leave a literal brace.
        assert "{voice_guidance}" not in prompt
        assert "{research_breadth}" not in prompt
        assert "{korean_style_rules}" not in prompt
        # The digest narrator persona is actually injected (the headline feature).
        token = cfg.pipeline.digest_voice_guidance.split()[0]
        assert token in prompt
        assert str(cfg.agent.research_slack_target_words) in prompt

    def test_shares_korean_style_rules_with_digest(self):
        # The same KOREAN_STYLE_RULES block must back BOTH the research prompt and the digest
        # language rules, so the two features can't drift on register / colon-ban / translationese.
        from agent.research_agent import SYSTEM_PROMPT_TEMPLATE
        from shared import KOREAN_STYLE_RULES, Config

        cfg = Config.load()
        assert KOREAN_STYLE_RULES in cfg.pipeline.digest_language_rules
        assert "{korean_style_rules}" in SYSTEM_PROMPT_TEMPLATE


class TestAgentMaxTokensFallback:
    def test_warns_when_model_info_missing(self):
        with (
            patch("agent.research_agent.BedrockModel", return_value=MagicMock()),
            patch("agent.research_agent.Agent", return_value=MagicMock(tool_names=["t"])),
            patch("agent.research_agent.boto3.Session", return_value=MagicMock()),
            patch(
                "agent.research_agent.BedrockCrossRegionModelHelper.get_cross_region_model_id",
                return_value="global.anthropic.claude-sonnet-4-6",
            ),
            patch("agent.research_agent._LANGUAGE_MODEL_INFO", {}),
            patch("agent.research_agent.logger") as log,
        ):
            from agent.research_agent import create_research_agent

            create_research_agent()

        assert log.warning.called
