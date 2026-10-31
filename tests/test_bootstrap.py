from unittest.mock import MagicMock, patch

from adapters.openai_images import OpenAIImageRenderer
from adapters.threads import ThreadsGateway
from entrypoints import bootstrap
from shared.config import Config


class TestDailyVisualWiring:
    def test_the_maker_gets_the_real_adapters_configured_from_config(self):
        config = Config()
        config.pipeline.visual_image_quality = "high"
        with patch.object(bootstrap, "create_state_store", return_value=MagicMock()):
            maker = bootstrap.daily_visual_maker(config, MagicMock())
        assert isinstance(maker.threads, ThreadsGateway)
        renderer = maker.generator.renderer
        assert isinstance(renderer, OpenAIImageRenderer)
        assert (renderer.image_model, renderer.image_quality) == (config.pipeline.image_model, "high")
        assert renderer.timeout_sec == config.pipeline.visual_image_timeout_sec

    def test_a_state_store_that_cannot_be_built_still_yields_a_maker(self):
        # The day still publishes, without the Threads ledger and the comic history.
        with patch.object(bootstrap, "create_state_store", side_effect=RuntimeError("no bucket")):
            maker = bootstrap.daily_visual_maker(Config(), MagicMock())
        assert maker.threads_ledger is None and maker.comic_log is None


class TestLanguageModels:
    def test_bedrock_runs_in_the_bedrock_region_on_the_profile_only_where_it_resolves(self):
        config = Config()
        with patch.object(bootstrap, "available_boto_profile", return_value=None) as profile:
            with patch.object(bootstrap.boto3, "Session") as session:
                with patch.object(bootstrap, "BedrockLanguageModelFactory") as factory:
                    bootstrap.language_models(config)
        profile.assert_called_once_with(config.aws.profile)
        session.assert_called_once_with(profile_name=None, region_name=config.aws.bedrock_region)
        assert factory.call_args.kwargs["region_name"] == config.aws.bedrock_region


class TestCollectors:
    """The --sources override, the unknown-source warning, the disabled-source skip, and the language
    models reaching web_search alone."""

    @staticmethod
    def _config() -> Config:
        config = Config()
        for cfg in (
            config.collectors.rss,
            config.collectors.reddit,
            config.collectors.youtube,
            config.collectors.web_search,
        ):
            cfg.enabled = True
        config.collectors.rsshub.enabled = False
        return config

    def test_enabled_sources_are_collected_when_no_override_is_given(self):
        labels = [name for name, _ in bootstrap.collectors(self._config(), MagicMock())]
        assert labels == ["reddit", "rss", "web_search", "youtube"]  # rsshub is disabled

    def test_sources_override_selects_exactly_what_was_asked_for(self):
        labels = [name for name, _ in bootstrap.collectors(self._config(), MagicMock(), ["rss", "reddit"])]
        assert labels == ["rss", "reddit"]  # the caller's order, not the map's

    def test_an_unknown_source_is_warned_about_and_skipped(self):
        with patch.object(bootstrap, "logger") as log:
            labels = [name for name, _ in bootstrap.collectors(self._config(), MagicMock(), ["rss", "nope"])]
        assert labels == ["rss"]
        assert any("Unknown source" in str(c.args) for c in log.warning.call_args_list)

    def test_an_explicitly_requested_but_disabled_source_is_skipped(self):
        # --sources rsshub on a deployment where rsshub is disabled must NOT start it.
        labels = [name for name, _ in bootstrap.collectors(self._config(), MagicMock(), ["rsshub", "rss"])]
        assert labels == ["rss"]

    def test_only_web_search_receives_the_language_models(self):
        from collectors.web_search import WebSearchCollector

        models = MagicMock()
        built = dict(bootstrap.collectors(self._config(), models, ["web_search", "rss"]))
        assert isinstance(built["web_search"], WebSearchCollector)
        # The models are only ever used to build the query-refine model; without them, web_search
        # silently loses query refinement.
        assert built["web_search"]._llm is models.get_model.return_value
        models.get_model.assert_called_once()
        assert models.get_model.call_args.kwargs["stage"] == "query-refine"
