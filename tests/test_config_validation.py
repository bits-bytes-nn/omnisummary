import re
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from shared.config import Config, PipelineConfig, RedditCollectorConfig, YouTubeCollectorConfig, get_config
from shared.utils import LANGUAGE_MODEL_INFO

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
CONFIG_TEMPLATE = CONFIG_DIR / "config-template.yaml"

# EVERY config file in the repo, parametrized. config/config.yaml is gitignored, so a test that
# skipped without it asserted nothing in CI — the model-registry-sync and code-default-vs-deployed
# invariants only ever ran on one laptop. The tracked template is always present, so the assertion
# is never vacuous, and a developer's local config.yaml is checked on top of it.
CONFIG_FILES = sorted(CONFIG_DIR.glob("*.yaml"))
CONFIG_IDS = [p.name for p in CONFIG_FILES]


class TestStrictConfig:
    def test_unknown_key_rejected(self):
        # A typo'd config key must fail loudly (extra="forbid"), not be silently dropped and fall
        # back to a code default — critical for the delivery toggles.
        with pytest.raises(ValidationError):
            PipelineConfig(enable_thread_post=True)  # typo of enable_threads_post
        with pytest.raises(ValidationError):
            Config(pipeline={"min_scor": 0.5})  # typo of min_score

    @pytest.mark.parametrize("config_path", CONFIG_FILES, ids=CONFIG_IDS)
    def test_every_config_file_loads_under_strict_validation(self, config_path):
        # Every config file must contain only known keys (guards against a strict-mode regression
        # where a real key isn't modeled). Parametrized over the DIRECTORY, so the tracked template
        # always makes this non-vacuous: config.yaml is gitignored and the old skip meant CI checked
        # nothing at all.
        cfg = Config.from_yaml(str(config_path))
        assert cfg.pipeline.top_n >= 1
        assert cfg.aws.project_name

    def test_config_template_loads_under_strict_validation(self):
        # The template is what a new deployment copies to config.yaml, so it must itself validate.
        # Loaded via from_yaml on the template PATH — Config.load() reads the shipped config.yaml
        # and would pass no matter how broken the template is (the old vacuous assertion).
        cfg = Config.from_yaml(str(CONFIG_TEMPLATE))
        assert cfg.pipeline.top_n >= 1
        assert cfg.aws.project_name

    def test_config_template_ships_no_live_sources(self):
        # Placeholder-only lists: a template must not silently start collecting from somewhere,
        # and the code defaults must not fill them in either.
        cfg = Config.from_yaml(str(CONFIG_TEMPLATE))
        assert cfg.collectors.rss.feeds == []
        assert cfg.collectors.reddit.subreddits == []
        assert cfg.collectors.youtube.channels == []
        assert cfg.collectors.rsshub.accounts == []

    def test_template_is_the_config_ci_synths(self):
        # config/config.yaml is gitignored, so a Config.load()-based synth silently fell back to
        # bare code defaults in CI and proved nothing about the config anyone actually deploys.
        # Keep the CI synth pinned to the tracked template.
        synth_src = (Path(__file__).resolve().parent.parent / "scripts" / "ci_synth.py").read_text()
        assert "config-template.yaml" in synth_src
        assert "Config.from_yaml(str(CONFIG_TEMPLATE))" in synth_src
        assert "config = Config.load()" not in synth_src

    def test_reddit_subreddits_default_is_empty(self):
        # A live source list must come from config.yaml, never from a code default (which would
        # keep collecting if the config key were ever dropped or typo'd out).
        assert RedditCollectorConfig().subreddits == []

    def test_top_n_lower_bound(self):
        with pytest.raises(ValidationError):
            PipelineConfig(top_n=0)
        with pytest.raises(ValidationError):
            PipelineConfig(top_n=-3)


class TestGetConfigCache:
    def test_parses_the_yaml_once(self):
        # Config.load() re-reads and re-validates the whole YAML on every call; a single research
        # run did that dozens of times (once per tool invocation). get_config() must parse once.
        with patch.object(Config, "load", wraps=Config.load) as load:
            first = get_config()
            second = get_config()
        assert first is second
        assert load.call_count == 1

    def test_cache_clear_reloads(self):
        first = get_config()
        get_config.cache_clear()
        assert get_config() is not first


class TestConfiguredModelsAreRegistered:
    def test_template_models_have_registry_info(self):
        # The tracked template is what a new deployment ships with, so its models must be
        # registered too — and unlike config.yaml it always exists (nothing to skip).
        cfg = Config.from_yaml(str(CONFIG_TEMPLATE))
        configured = {
            cfg.pipeline.ranking_model,
            cfg.pipeline.digest_model,
            cfg.pipeline.trend_model,
            cfg.collectors.web_search.refine_model,
            cfg.agent.model_id,
        }
        missing = [m.value for m in configured if m not in LANGUAGE_MODEL_INFO]
        assert not missing, f"template models missing from LANGUAGE_MODEL_INFO: {missing}"

    @pytest.mark.parametrize("config_path", CONFIG_FILES, ids=CONFIG_IDS)
    def test_every_configured_model_has_registry_info(self, config_path):
        # A model set in config that lacks a LANGUAGE_MODEL_INFO entry passes Pydantic load
        # (valid enum) but hits the runtime max_tokens/gating fallback with only a warning.
        # This locks the two in sync so a Sonnet-5-style bump can't half-land. Parametrized over
        # every config file: gated on config.yaml alone it never ran in CI.
        cfg = Config.from_yaml(str(config_path))
        configured = {
            cfg.pipeline.ranking_model,
            cfg.pipeline.digest_model,
            cfg.pipeline.trend_model,
            cfg.collectors.web_search.refine_model,
            cfg.agent.model_id,
        }
        missing = [m.value for m in configured if m not in LANGUAGE_MODEL_INFO]
        assert not missing, f"configured models missing from LANGUAGE_MODEL_INFO: {missing}"


class TestImageSizes:
    """image_sizes' keys are the VisualBrief orientation vocabulary, not free-form labels: the editor
    is offered the keys and the brief's orientation is looked up in the same dict. The old test just
    mirrored the code defaults, so a renamed key stayed green while every brief silently coerced to
    the default orientation."""

    def test_keys_are_exactly_the_orientation_vocabulary(self):
        from shared.constants import VISUAL_ORIENTATIONS

        assert set(PipelineConfig().image_sizes) == set(VISUAL_ORIENTATIONS)

    def test_values_are_pixel_dimensions(self):
        assert all(re.fullmatch(r"\d+x\d+", size) for size in PipelineConfig().image_sizes.values())

    def test_sizes_are_overridable(self):
        cfg = PipelineConfig(image_sizes={"square": "512x512", "landscape": "768x512", "portrait": "512x768"})
        assert cfg.image_sizes["portrait"] == "512x768"

    def test_a_renamed_or_missing_orientation_key_fails_load(self):
        with pytest.raises(ValidationError, match="orientation vocabulary"):
            PipelineConfig(image_sizes={"tall": "1024x1536", "wide": "1536x1024", "square": "1024x1024"})
        with pytest.raises(ValidationError, match="orientation vocabulary"):
            PipelineConfig(image_sizes={"square": "1024x1024", "landscape": "1536x1024"})

    def test_a_non_pixel_size_fails_load(self):
        with pytest.raises(ValidationError, match="1024x1536"):
            PipelineConfig(
                image_sizes={"square": "big", "landscape": "1536x1024", "portrait": "1024x1536"},
            )

    @pytest.mark.parametrize("config_path", CONFIG_FILES, ids=CONFIG_IDS)
    def test_every_config_keeps_the_vocabulary(self, config_path):
        from shared.constants import VISUAL_ORIENTATIONS

        cfg = Config.from_yaml(str(config_path))
        assert set(cfg.pipeline.image_sizes) == set(VISUAL_ORIENTATIONS)


class TestTranscriptLanguage:
    def test_default_is_en(self):
        assert YouTubeCollectorConfig().transcript_language == "en"

    def test_is_configurable(self):
        assert YouTubeCollectorConfig(transcript_language="ko").transcript_language == "ko"


class TestCodeDefaultsMatchTheDeployedConfig:
    """A code default that disagrees with config.yaml is a live trap: it is what a deployment
    without a config.yaml (and every PipelineConfig() in a test) silently gets."""

    @pytest.mark.parametrize("config_path", CONFIG_FILES, ids=CONFIG_IDS)
    def test_countdown_position_default_matches_every_config(self, config_path):
        # The INVARIANT is what matters — a code default that disagrees with the deployed config is
        # what a config-less deployment (and every PipelineConfig() in a test) silently gets. The
        # VALUE is the owner's editorial call: the countdown is the account's signature, so it opens
        # the lead today, but pinning the literal here would break this test on every such decision.
        cfg = Config.from_yaml(str(config_path))
        assert cfg.pipeline.agi_countdown_position == PipelineConfig().agi_countdown_position

    @pytest.mark.parametrize("config_path", CONFIG_FILES, ids=CONFIG_IDS)
    def test_delivery_toggles_are_explicit_in_every_config(self, config_path):
        # Delivery routing must come from the file, never a code default: the visual Lambda is the
        # only Threads publish path, and enable_threads_post defaults to False in code.
        import yaml

        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))["pipeline"]
        assert "enable_threads_post" in raw
        assert "enable_slack_post" in raw


class TestLanguageRules:
    def test_one_form_per_proper_noun_and_particle_agreement(self):
        # Published Korean carried two spellings of the same company in one digest and particles
        # that disagreed with the form as written. One consolidated rule, no per-company name table.
        rules = PipelineConfig().digest_language_rules
        assert "ONE form per proper noun" in rules
        assert "particle" in rules
        assert "transliteration" in rules
