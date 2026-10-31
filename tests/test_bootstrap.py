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
