from unittest.mock import MagicMock, patch


class TestAgentPromptCaching:
    """Strands resolves strategy="auto" by substring-matching the model id for "claude"/"anthropic".
    In production the id is an application inference profile ARN whose trailing segment is opaque, so
    "auto" concluded the model could not cache and dropped every cache point — a research turn
    re-sends the system prompt plus all accumulated tool results, so this is silently expensive.
    The previous version of this test mocked the resolver to return "global.anthropic.claude-sonnet-5",
    which contains "anthropic", and therefore could never have caught it."""

    PROFILE_ARN = "arn:aws:bedrock:us-west-2:998601677581:application-inference-profile/pgqjcyolwk7q"

    def _build_with(self, resolved_model_id):
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
                return_value=resolved_model_id,
            ),
        ):
            from agent.research_agent import create_research_agent

            create_research_agent()
        return captured

    def test_cache_strategy_is_stated_not_sniffed(self):
        cache_config = self._build_with(self.PROFILE_ARN).get("cache_config")
        assert cache_config is not None
        # Must NOT be "auto": that is the value whose model-id sniff fails on an ARN.
        assert cache_config.strategy == "anthropic"

    def test_caching_survives_an_opaque_model_id(self):
        # The point of stating the strategy is that it cannot depend on the id's spelling, so the
        # same strategy must come out for a profile ARN and for a plain model id.
        for model_id in (self.PROFILE_ARN, "global.anthropic.claude-sonnet-5"):
            assert self._build_with(model_id)["cache_config"].strategy == "anthropic"

    def test_strands_would_disable_caching_for_the_deployed_model_id_under_auto(self):
        # Pins the upstream behaviour this works around: if Strands ever taught "auto" to recognise
        # a profile ARN, this test fails and the explicit strategy can be revisited.
        from strands.models.bedrock import BedrockModel

        sniffed = BedrockModel._cache_strategy.fget(MagicMock(config={"model_id": self.PROFILE_ARN}))
        assert sniffed is None


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
                return_value="global.anthropic.claude-sonnet-5",
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
                return_value="global.anthropic.claude-sonnet-5",
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
            recall_digest,
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
            recall_digest,
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
                return_value="global.anthropic.claude-sonnet-5",
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

    def test_threads_post_cap_comes_from_the_renderer_constant(self):
        # The prompt used to spell the per-post limit as a literal "500" while the renderer enforced
        # THREADS_MAX_POST_CHARS, so raising the channel's cap in one place left the other behind and
        # the agent wrote to a budget the renderer no longer used.
        from agent.research_agent import SYSTEM_PROMPT_TEMPLATE

        assert "{threads_max_post_chars}" in SYSTEM_PROMPT_TEMPLATE
        assert "500" not in SYSTEM_PROMPT_TEMPLATE

    def test_threads_scope_is_narrowed_not_compressed(self):
        # A 6-post cap gave Threads ~2.5k chars of prose against Slack's ~8.7k, and the prompt still
        # demanded the same facts, figures and conclusions. The only way to obey both was to cut the
        # explanation, which is what made Threads reports read as assertion lists. The instruction
        # now trades scope for depth, so it must NOT ask for parity again.
        from agent.research_agent import SYSTEM_PROMPT_TEMPLATE

        assert "FEWER points" in SYSTEM_PROMPT_TEMPLATE
        assert "COMPRESSED" not in SYSTEM_PROMPT_TEMPLATE
        assert "same facts, figures, sources, and conclusions" not in SYSTEM_PROMPT_TEMPLATE

    def test_todays_date_is_injected_from_the_clock(self):
        # The prompt interpolated persona, knobs and the tool menu but never the date, so the model
        # judged "latest" against its training prior and had to INVENT a date for recall_digest
        # (which then silently recalled nothing).
        from datetime import datetime
        from zoneinfo import ZoneInfo

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
                return_value="global.anthropic.claude-sonnet-5",
            ),
        ):
            from agent.research_agent import create_research_agent
            from shared import Config

            create_research_agent()
            cfg = Config.load()

        prompt = captured["system_prompt"]
        today = datetime.now(ZoneInfo(cfg.aws.timezone)).date().isoformat()
        assert f"Today is {today}" in prompt
        assert cfg.aws.timezone in prompt
        assert "{today}" not in prompt

    def test_tool_menu_is_derived_from_the_bound_tools(self):
        # The prompt used to carry a hand-written numbered tool list beside the separately hardcoded
        # list in create_research_agent, with nothing keeping them in agreement: a renamed, added or
        # dropped tool left the model reading a menu it could no longer call. Every bound tool must
        # appear, with its real arguments, and nothing else may.
        from agent.research_agent import _render_tools_block
        from agent.research_tools import (
            attach_image,
            community_search,
            deliver_report,
            read_url,
            recall_digest,
            recall_trends,
            search_papers,
            web_search,
        )

        tools = [
            web_search,
            community_search,
            search_papers,
            read_url,
            recall_trends,
            recall_digest,
            attach_image,
            deliver_report,
        ]
        block = _render_tools_block(tools)
        for tool in tools:
            assert f"{tool.tool_name}(" in block
        # arguments come from the real signature, not prose
        assert "web_search(query, recency)" in block
        assert "deliver_report(report, channel)" in block
        # one line per tool, numbered, and no stale name survives
        assert len(block.splitlines()) == len(tools)
        assert "make_visual" not in block and "get_detail" not in block

    def test_a_renamed_tool_changes_the_menu(self):
        # The point of deriving the block: dropping a tool must remove it, with no edit anywhere.
        from agent.research_agent import _render_tools_block
        from agent.research_tools import web_search

        assert "deliver_report" not in _render_tools_block([web_search])

    def test_shares_korean_style_rules_with_digest(self):
        # The same KOREAN_STYLE_RULES block must back BOTH the research prompt and the digest
        # language rules, so the two features can't drift on register / colon-ban / translationese.
        from agent.research_agent import SYSTEM_PROMPT_TEMPLATE
        from shared import KOREAN_STYLE_RULES, Config

        cfg = Config.load()
        assert KOREAN_STYLE_RULES in cfg.pipeline.digest_language_rules
        assert "{korean_style_rules}" in SYSTEM_PROMPT_TEMPLATE


class TestAgentTemperatureGating:
    def test_temperature_omitted_for_sonnet_5_default(self):
        # config.agent.model_id defaults to Sonnet 5 (supports_temperature=False), which 400s on
        # a non-default temperature. The agent must omit it, mirroring the Bedrock factory gate.
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
                return_value="global.anthropic.claude-sonnet-5",
            ),
        ):
            from agent.research_agent import create_research_agent

            create_research_agent()

        assert "temperature" not in captured

    def test_temperature_sent_when_model_accepts_it(self):
        from shared.utils import LanguageModelInfo

        captured = {}

        def fake_bedrock_model(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        temp_ok = LanguageModelInfo(context_window_size=200000, max_output_tokens=64000, supports_temperature=True)
        with (
            patch("agent.research_agent.BedrockModel", side_effect=fake_bedrock_model),
            patch("agent.research_agent.Agent", return_value=MagicMock(tool_names=["t"])),
            patch("agent.research_agent.boto3.Session", return_value=MagicMock()),
            patch(
                "agent.research_agent.BedrockCrossRegionModelHelper.get_cross_region_model_id",
                return_value="global.anthropic.claude-sonnet-4-6",
            ),
        ):
            import agent.research_agent as ra
            from shared.config import Config

            model_id = Config.load().agent.model_id
            with patch.dict(ra.LANGUAGE_MODEL_INFO, {model_id: temp_ok}, clear=False):
                ra.create_research_agent()

        assert captured.get("temperature") == 0.0


class TestAgentMaxTokensFallback:
    def test_warns_when_model_info_missing(self):
        with (
            patch("agent.research_agent.BedrockModel", return_value=MagicMock()),
            patch("agent.research_agent.Agent", return_value=MagicMock(tool_names=["t"])),
            patch("agent.research_agent.boto3.Session", return_value=MagicMock()),
            patch(
                "agent.research_agent.BedrockCrossRegionModelHelper.get_cross_region_model_id",
                return_value="global.anthropic.claude-sonnet-5",
            ),
            patch("agent.research_agent.LANGUAGE_MODEL_INFO", {}),
            patch("agent.research_agent.logger") as log,
        ):
            from agent.research_agent import create_research_agent

            create_research_agent()

        assert log.warning.called
