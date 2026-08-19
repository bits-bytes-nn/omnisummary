from unittest.mock import MagicMock, patch

import pytest

from shared.constants import LanguageModelId
from shared.utils import LANGUAGE_MODEL_INFO, TOKEN_COUNT_MODEL, BedrockLanguageModelFactory


def _factory(client=None):
    with patch("shared.utils.boto3.Session") as session:
        session.return_value.client.return_value = client or MagicMock()
        session.return_value.region_name = "us-west-2"
        session.return_value.profile_name = None
        return BedrockLanguageModelFactory(region_name="us-west-2")


class TestCountTokens:
    def test_always_counts_with_supported_base_model_id(self):
        # CountTokens only works on some base models; we always count with the supported Sonnet
        # base id (shared tokenizer), never the caller's model, and strip any cross-region prefix.
        client = MagicMock()
        client.count_tokens.return_value = {"inputTokens": 42}
        f = _factory(client)
        assert f.count_tokens("some text") == 42
        assert client.count_tokens.call_args.kwargs["modelId"] == TOKEN_COUNT_MODEL.value

    def test_falls_back_to_char_estimate_on_api_error(self):
        client = MagicMock()
        client.count_tokens.side_effect = RuntimeError("throttled")
        f = _factory(client)
        assert f.count_tokens("x" * 40) == 10  # len//4 fallback

    def test_truncate_to_tokens_binary_searches_to_fit(self):
        client = MagicMock()
        # token count ≈ chars/5 for this fake, so a 1000-char text over a 20-token budget truncates
        client.count_tokens.side_effect = lambda modelId, input: {
            "inputTokens": len(input["converse"]["messages"][0]["content"][0]["text"]) // 5
        }
        f = _factory(client)
        out = f.truncate_to_tokens("가 " * 500, 20)
        assert len(out) < 1000
        assert f.count_tokens(out) <= 20

    def test_truncate_returns_text_when_within_budget(self):
        client = MagicMock()
        client.count_tokens.return_value = {"inputTokens": 5}
        f = _factory(client)
        assert f.truncate_to_tokens("short", 100) == "short"

    def test_count_tokens_memoizes_identical_text(self):
        # The same text is counted many times (prompt building across stages, truncate's binary
        # search); repeat calls must hit the instance cache, not re-bill the Bedrock API.
        client = MagicMock()
        client.count_tokens.return_value = {"inputTokens": 7}
        f = _factory(client)
        assert f.count_tokens("repeated") == 7
        assert f.count_tokens("repeated") == 7
        assert client.count_tokens.call_count == 1  # second call served from cache

    def test_count_tokens_degrades_after_first_failure(self):
        # After CountTokens fails once, the rest of the run must use the char estimate WITHOUT
        # hammering the failing API — otherwise truncate's binary search amplifies one blip into
        # a storm of failing round-trips.
        client = MagicMock()
        client.count_tokens.side_effect = RuntimeError("throttled")
        f = _factory(client)
        assert f.count_tokens("a" * 40) == 10  # first: tries API, fails, estimates
        assert f.count_tokens("b" * 80) == 20  # second: short-circuits to estimate
        assert f.count_tokens("c" * 12) == 3
        assert client.count_tokens.call_count == 1  # API called only once, then degraded


class TestTemperatureGating:
    def test_opus_48_omits_temperature(self):
        # Opus 4.7/4.8 reject the temperature param -> must not be sent.
        f = _factory()
        info = LANGUAGE_MODEL_INFO[LanguageModelId.CLAUDE_V4_8_OPUS]
        cfg = f._build_model_config(info, "global.anthropic.claude-opus-4-8", True)
        assert "temperature" not in cfg
        assert cfg["max_tokens"] > 0

    def test_sonnet_46_includes_temperature(self):
        f = _factory()
        info = LANGUAGE_MODEL_INFO[LanguageModelId.CLAUDE_V4_6_SONNET]
        cfg = f._build_model_config(info, "global.anthropic.claude-sonnet-4-6", True)
        assert "temperature" in cfg

    def test_non_cross_region_opus_48_omits_temperature_and_top_k(self):
        # Models that reject sampling params must omit BOTH temperature and top_k on the
        # non-converse (ChatBedrock) path, or Bedrock 400s.
        f = _factory()
        info = LANGUAGE_MODEL_INFO[LanguageModelId.CLAUDE_V4_8_OPUS]
        cfg = f._build_model_config(info, "anthropic.claude-opus-4-8", False)
        assert "temperature" not in cfg["model_kwargs"]
        assert "top_k" not in cfg["model_kwargs"]

    def test_non_cross_region_sonnet_46_includes_top_k(self):
        # A sampling-param-accepting model DOES get top_k on the non-converse path.
        f = _factory()
        info = LANGUAGE_MODEL_INFO[LanguageModelId.CLAUDE_V4_6_SONNET]
        cfg = f._build_model_config(info, "anthropic.claude-sonnet-4-6", False)
        assert cfg["model_kwargs"]["top_k"] == BedrockLanguageModelFactory.DEFAULT_TOP_K

    def test_sonnet_5_omits_temperature(self):
        # Sonnet 5 rejects non-default sampling params (400), same as Opus 4.7/4.8. It is now
        # the default digest/agent/trend model, so lock the gating in explicitly.
        f = _factory()
        info = LANGUAGE_MODEL_INFO[LanguageModelId.CLAUDE_V5_SONNET]
        cfg = f._build_model_config(info, "global.anthropic.claude-sonnet-5", True)
        assert "temperature" not in cfg
        assert cfg["max_tokens"] > 0

    def test_sonnet_5_omits_top_k_on_non_converse(self):
        f = _factory()
        info = LANGUAGE_MODEL_INFO[LanguageModelId.CLAUDE_V5_SONNET]
        cfg = f._build_model_config(info, "anthropic.claude-sonnet-5", False)
        assert "temperature" not in cfg["model_kwargs"]
        assert "top_k" not in cfg["model_kwargs"]


class TestThinkingFormat:
    """Newer models (Sonnet 5, Opus 4.7/4.8) reject the legacy
    thinking.type='enabled' + budget_tokens form and require adaptive thinking.
    Guards against re-emitting the deprecated shape when thinking is enabled."""

    def test_adaptive_model_emits_adaptive_thinking(self):
        f = _factory()
        info = LANGUAGE_MODEL_INFO[LanguageModelId.CLAUDE_V5_SONNET]
        assert info.uses_adaptive_thinking is True
        cfg = f._build_model_config(info, "global.anthropic.claude-sonnet-5", True, enable_thinking=True)
        amrf = cfg["additional_model_request_fields"]
        assert amrf["thinking"] == {"type": "adaptive"}
        assert "budget_tokens" not in amrf["thinking"]
        assert amrf["output_config"]["effort"] == f.DEFAULT_THINKING_EFFORT

    def test_adaptive_model_honors_thinking_effort(self):
        f = _factory()
        info = LANGUAGE_MODEL_INFO[LanguageModelId.CLAUDE_V4_8_OPUS]
        cfg = f._build_model_config(
            info,
            "global.anthropic.claude-opus-4-8",
            True,
            enable_thinking=True,
            thinking_effort="high",
        )
        assert cfg["additional_model_request_fields"]["output_config"]["effort"] == "high"

    def test_legacy_model_keeps_budget_form(self):
        f = _factory()
        info = LANGUAGE_MODEL_INFO[LanguageModelId.CLAUDE_V4_6_SONNET]
        assert info.uses_adaptive_thinking is False
        cfg = f._build_model_config(info, "global.anthropic.claude-sonnet-4-6", True, enable_thinking=True)
        thinking = cfg["additional_model_request_fields"]["thinking"]
        assert thinking["type"] == "enabled"
        assert thinking["budget_tokens"] == f.DEFAULT_THINKING_BUDGET_TOKENS


class TestOpus5Registry:
    """Opus 5 is selectable from config. Its capability flags were VERIFIED against Converse on
    global.anthropic.claude-opus-5, not inferred from the version number: a `temperature` param and
    the legacy thinking.type="enabled"/budget_tokens form both return ValidationException, while
    thinking.type="adaptive" + output_config.effort succeeds. Getting these wrong 400s every call."""

    def test_opus_5_is_registered(self):
        assert LanguageModelId.CLAUDE_V5_OPUS in LANGUAGE_MODEL_INFO

    def test_opus_5_rejects_sampling_params_and_uses_adaptive_thinking(self):
        info = LANGUAGE_MODEL_INFO[LanguageModelId.CLAUDE_V5_OPUS]
        assert info.supports_temperature is False
        assert info.uses_adaptive_thinking is True
        assert info.supports_thinking is True
        assert info.supports_prompt_caching is True

    def test_opus_5_matches_the_other_claude_5_family_gates(self):
        # Same shape as Sonnet 5 / Opus 4.8 — a new family member that silently differs on these two
        # flags is the failure mode this pins.
        opus5 = LANGUAGE_MODEL_INFO[LanguageModelId.CLAUDE_V5_OPUS]
        for sibling in (LanguageModelId.CLAUDE_V5_SONNET, LanguageModelId.CLAUDE_V4_8_OPUS):
            other = LANGUAGE_MODEL_INFO[sibling]
            assert opus5.supports_temperature == other.supports_temperature
            assert opus5.uses_adaptive_thinking == other.uses_adaptive_thinking


class TestModelClassConstructorSurface:
    """Every other test here asserts on the config DICT, so a langchain-aws upgrade that renamed or
    dropped a kwarg still passed the whole suite — nothing ever fed the dict to the real class. Both
    ChatBedrockConverse and ChatBedrock set pydantic extra="forbid", so actually constructing them
    is what turns a silent framework break into a red build. No network: the boto client is a mock."""

    CASES = [
        (LanguageModelId.CLAUDE_V4_8_OPUS, {}),
        (LanguageModelId.CLAUDE_V5_SONNET, {}),
        (LanguageModelId.CLAUDE_V5_OPUS, {}),
        (LanguageModelId.CLAUDE_V5_SONNET, {"enable_thinking": True}),
        (LanguageModelId.CLAUDE_V4_6_SONNET, {"enable_thinking": True}),  # legacy budget_tokens form
        (LanguageModelId.CLAUDE_V5_SONNET, {"supports_1m_context_window": True}),
        (LanguageModelId.CLAUDE_V5_SONNET, {"enable_performance_optimization": True}),
    ]

    def test_every_built_config_constructs_both_model_classes(self):
        from langchain_aws import ChatBedrock, ChatBedrockConverse

        f = _factory()
        # conftest's hermetic_env blocks boto3.client outright; these classes build one in a
        # validator, so opt in with a mock. Still no network — nothing is invoked.
        with patch("boto3.client", return_value=MagicMock()):
            for model, kwargs in self.CASES:
                info = LANGUAGE_MODEL_INFO[model]
                for use_converse, cls in ((True, ChatBedrockConverse), (False, ChatBedrock)):
                    cfg = f._build_model_config(info, model.value, use_converse, stage="ranking", **kwargs)
                    cls(**cfg)  # raises on an unknown/renamed kwarg

    def test_the_application_profile_arn_config_constructs(self):
        # The ARN path adds `provider`, which only exists on the converse class — and it is the path
        # that only runs in a deployed account, so a break here would surface in production first.
        from langchain_aws import ChatBedrockConverse

        f = _factory()
        info = LANGUAGE_MODEL_INFO[LanguageModelId.CLAUDE_V5_SONNET]
        cfg = f._build_model_config(
            info, "arn:aws:bedrock:us-west-2:1:application-inference-profile/abc", True, stage="digest"
        )
        with patch("boto3.client", return_value=MagicMock()):
            ChatBedrockConverse(**cfg)


class TestTokenUsageAttribution:
    """Cost Explorer bills per MODEL, and several pipeline stages share one model — so a token total
    could not be traced to the stage that spent it. Every model carries a usage logger tagged with
    its stage; telemetry must never be able to fail a generation."""

    def test_stage_tagged_usage_logger_is_attached(self):
        from shared.utils import _TokenUsageLogger

        f = _factory()
        info = LANGUAGE_MODEL_INFO[LanguageModelId.CLAUDE_V5_SONNET]
        cfg = f._build_model_config(info, "global.anthropic.claude-sonnet-5", True, stage="ranking")
        loggers = [c for c in cfg["callbacks"] if isinstance(c, _TokenUsageLogger)]
        assert [h.stage for h in loggers] == ["ranking"]
        assert loggers[0].model_id == "global.anthropic.claude-sonnet-5"

    def test_untagged_call_is_labelled_rather_than_dropped(self):
        from shared.utils import _TokenUsageLogger

        f = _factory()
        info = LANGUAGE_MODEL_INFO[LanguageModelId.CLAUDE_V5_SONNET]
        cfg = f._build_model_config(info, "global.anthropic.claude-sonnet-5", True)
        assert [c.stage for c in cfg["callbacks"] if isinstance(c, _TokenUsageLogger)] == ["unattributed"]

    def test_a_caller_supplied_callback_is_kept(self):
        from shared.utils import _TokenUsageLogger

        mine = MagicMock()
        f = _factory()
        info = LANGUAGE_MODEL_INFO[LanguageModelId.CLAUDE_V5_SONNET]
        cfg = f._build_model_config(info, "global.anthropic.claude-sonnet-5", True, callbacks=[mine], stage="digest")
        assert mine in cfg["callbacks"]
        assert any(isinstance(c, _TokenUsageLogger) for c in cfg["callbacks"])

    def test_a_response_without_usage_metadata_is_survivable(self):
        from shared.utils import _TokenUsageLogger

        handler = _TokenUsageLogger("digest", "global.anthropic.claude-sonnet-5")
        handler.on_llm_end(MagicMock(generations=[], llm_output=None))  # must not raise
        handler.on_llm_end(object())  # not even the expected shape


class TestApplicationProfileResolution:
    """On-demand Bedrock has no taggable resource, so token spend is unattributable in a shared
    account. The resolver prefers this project's tagged APPLICATION inference profile — and because
    resolution happens in one place, both the LangChain factory and the Strands research agent get
    it. It must degrade silently: cost reporting may never stop a generation."""

    def test_profile_name_is_deterministic_and_slugged(self, monkeypatch):
        from shared.utils import BedrockCrossRegionModelHelper as H

        monkeypatch.setenv("PROJECT_NAME", "omnisummary")
        monkeypatch.setenv("STAGE", "dev")
        assert (
            H.application_profile_name(LanguageModelId.CLAUDE_V4_8_OPUS) == "omnisummary-dev-anthropic-claude-opus-4-8"
        )
        # dots and colons in a model id are not valid in a profile name
        name = H.application_profile_name(LanguageModelId.CLAUDE_V3_HAIKU)
        assert "." not in name and ":" not in name

    def test_matching_profile_is_preferred(self, monkeypatch):
        from shared.utils import BedrockCrossRegionModelHelper as H

        monkeypatch.setenv("PROJECT_NAME", "omnisummary")
        monkeypatch.setenv("STAGE", "dev")
        arn = "arn:aws:bedrock:us-west-2:1:application-inference-profile/abc"
        client = MagicMock()
        client.get_paginator.return_value.paginate.return_value = [
            {
                "inferenceProfileSummaries": [
                    {"inferenceProfileName": "someone-elses", "inferenceProfileArn": "arn:other"},
                    {"inferenceProfileName": "omnisummary-dev-anthropic-claude-opus-4-8", "inferenceProfileArn": arn},
                ]
            }
        ]
        session = MagicMock()
        session.client.return_value = client
        got = H._application_profile_arn(session, LanguageModelId.CLAUDE_V4_8_OPUS, "us-west-2", "fallback-id")
        assert got == arn

    def test_absent_profile_keeps_the_system_defined_id(self, monkeypatch):
        from shared.utils import BedrockCrossRegionModelHelper as H

        monkeypatch.setenv("PROJECT_NAME", "omnisummary")
        client = MagicMock()
        client.get_paginator.return_value.paginate.return_value = [{"inferenceProfileSummaries": []}]
        session = MagicMock()
        session.client.return_value = client
        assert (
            H._application_profile_arn(session, LanguageModelId.CLAUDE_V5_SONNET, "us-west-2", "global.anthropic.x")
            == "global.anthropic.x"
        )

    def test_a_denied_lookup_keeps_the_system_defined_id(self):
        from shared.utils import BedrockCrossRegionModelHelper as H

        session = MagicMock()
        session.client.side_effect = RuntimeError("AccessDenied")
        assert (
            H._application_profile_arn(session, LanguageModelId.CLAUDE_V5_SONNET, "us-west-2", "global.anthropic.x")
            == "global.anthropic.x"
        )

    def test_an_arn_model_id_names_its_provider(self):
        # ChatBedrockConverse refuses an ARN without a provider ("Model provider should be supplied
        # when passing a model ARN as model_id"), which is exactly what a profile ARN is.
        f = _factory()
        info = LANGUAGE_MODEL_INFO[LanguageModelId.CLAUDE_V5_SONNET]
        cfg = f._build_model_config(info, "arn:aws:bedrock:us-west-2:1:application-inference-profile/abc", True)
        assert cfg["provider"] == "anthropic"
        assert "provider" not in f._build_model_config(info, "global.anthropic.claude-sonnet-5", True)


def _resolution_client(*available: str) -> MagicMock:
    """A bedrock control-plane client whose SYSTEM_DEFINED profiles are exactly `available`, and
    which has no APPLICATION profile (so resolution stops at the system-defined id)."""
    client = MagicMock()
    client.list_inference_profiles.return_value = {
        "inferenceProfileSummaries": [{"inferenceProfileId": profile_id} for profile_id in available]
    }
    client.get_paginator.return_value.paginate.return_value = [{"inferenceProfileSummaries": []}]
    return client


def _session(client: MagicMock) -> MagicMock:
    session = MagicMock()
    session.client.return_value = client
    return session


class TestCrossRegionResolution:
    """Which model id every stage actually bills against. The global/regional/standard ladder and its
    broad except decide that, and nothing exercised them: get_model is mocked in every consumer test.
    A ladder that silently always returned the standard id would look identical in the logs."""

    MODEL = LanguageModelId.CLAUDE_V5_SONNET

    def test_a_global_profile_wins(self):
        from shared.utils import BedrockCrossRegionModelHelper as H

        client = _resolution_client(f"global.{self.MODEL.value}", f"us.{self.MODEL.value}")
        assert H.get_cross_region_model_id(_session(client), self.MODEL, "us-west-2") == f"global.{self.MODEL.value}"

    def test_the_regional_profile_is_the_second_choice(self):
        from shared.utils import BedrockCrossRegionModelHelper as H

        client = _resolution_client(f"us.{self.MODEL.value}")
        assert H.get_cross_region_model_id(_session(client), self.MODEL, "us-west-2") == f"us.{self.MODEL.value}"

    def test_asia_pacific_regions_use_the_apac_prefix(self):
        # ap-northeast-2's profiles are named apac.*, not ap.* — the one region family whose prefix
        # is not the first two characters of its name.
        from shared.utils import BedrockCrossRegionModelHelper as H

        client = _resolution_client(f"apac.{self.MODEL.value}")
        assert H.get_cross_region_model_id(_session(client), self.MODEL, "ap-northeast-2") == f"apac.{self.MODEL.value}"

    def test_other_regions_use_their_two_letter_prefix(self):
        from shared.utils import BedrockCrossRegionModelHelper as H

        client = _resolution_client(f"eu.{self.MODEL.value}")
        assert H.get_cross_region_model_id(_session(client), self.MODEL, "eu-west-1") == f"eu.{self.MODEL.value}"

    def test_no_cross_region_profile_falls_back_to_the_standard_id(self):
        from shared.utils import BedrockCrossRegionModelHelper as H

        client = _resolution_client()
        assert H.get_cross_region_model_id(_session(client), self.MODEL, "us-west-2") == self.MODEL.value

    def test_a_denied_listing_falls_back_to_the_standard_id(self):
        # ListInferenceProfiles is denied / throttled: the digest must still run against the plain
        # model id rather than fail at model construction.
        from shared.utils import BedrockCrossRegionModelHelper as H

        client = MagicMock()
        client.list_inference_profiles.side_effect = RuntimeError("AccessDenied")
        client.get_paginator.return_value.paginate.return_value = [{"inferenceProfileSummaries": []}]
        assert H.get_cross_region_model_id(_session(client), self.MODEL, "us-west-2") == self.MODEL.value

    def test_resolution_is_cached_per_model_and_region(self):
        # ranker/digest/trend/refine each build a model; without the cache every build pays the
        # round-trip (and re-risks the AccessDenied path).
        from shared.utils import BedrockCrossRegionModelHelper as H

        client = _resolution_client(f"global.{self.MODEL.value}")
        session = _session(client)
        H.get_cross_region_model_id(session, self.MODEL, "us-west-2")
        H.get_cross_region_model_id(session, self.MODEL, "us-west-2")
        assert client.list_inference_profiles.call_count == 1
        H.get_cross_region_model_id(session, self.MODEL, "eu-west-1")  # a different region is not cached
        assert client.list_inference_profiles.call_count > 1


class TestGetModel:
    """The public entry point every stage calls. Consumer tests all mock it, so nothing asserted
    which model class it builds or that the resolved id reaches the model."""

    @staticmethod
    def _built(model_id: LanguageModelId, resolved: str, **kwargs):
        from shared.utils import BedrockCrossRegionModelHelper as H

        f = _factory()
        with patch.object(H, "get_cross_region_model_id", return_value=resolved):
            with patch("boto3.client", return_value=MagicMock()):
                return f.get_model(model_id, **kwargs)

    def test_a_cross_region_id_builds_the_converse_class_with_that_id(self):
        from langchain_aws import ChatBedrockConverse

        resolved = f"global.{LanguageModelId.CLAUDE_V5_SONNET.value}"
        model = self._built(LanguageModelId.CLAUDE_V5_SONNET, resolved, stage="digest")
        assert isinstance(model, ChatBedrockConverse)
        assert model.model_id == resolved

    def test_a_standard_id_without_thinking_builds_the_legacy_class(self):
        from langchain_aws import ChatBedrock

        model_id = LanguageModelId.CLAUDE_V3_5_SONNET
        model = self._built(model_id, model_id.value, stage="digest")
        assert isinstance(model, ChatBedrock)
        assert model.model_id == model_id.value

    def test_thinking_on_a_supporting_model_forces_the_converse_class(self):
        from langchain_aws import ChatBedrockConverse

        model_id = LanguageModelId.CLAUDE_V5_SONNET
        model = self._built(model_id, model_id.value, stage="ranking", enable_thinking=True)
        assert isinstance(model, ChatBedrockConverse)

    def test_the_stage_reaches_the_usage_logger(self):
        # The bill is per MODEL and several stages share one, so the stage tag is the only way a
        # token total is attributable at all.
        from shared.utils import _TokenUsageLogger

        model = self._built(LanguageModelId.CLAUDE_V5_SONNET, "global.anthropic.claude-sonnet-5", stage="ranking")
        loggers = [c for c in (model.callbacks or []) if isinstance(c, _TokenUsageLogger)]
        assert [c.stage for c in loggers] == ["ranking"]

    def test_an_unregistered_model_is_rejected(self):
        f = _factory()
        with patch.object(f, "get_model_info", return_value=None):
            with pytest.raises(ValueError, match="Unsupported language model ID"):
                f.get_model(LanguageModelId.CLAUDE_V5_SONNET)
