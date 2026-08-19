import re
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from collectors.base import degradation_reason
from shared.config import (
    CollectorsConfig,
    Config,
    PipelineConfig,
    RedditCollectorConfig,
    YouTubeCollectorConfig,
    _utc_offset_hours,
    get_config,
)
from shared.constants import COLLECTOR_NAMES, EMPTY_RATE_CHECK_DISABLED, SourceType
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


class TestVisualImageQuality:
    """Unset, the tier was OpenAI's "auto" — which decides both the on-image text legibility and a
    ~4x per-image price swing, while the render log could only say `quality=auto->unreported`."""

    @pytest.mark.parametrize("config_path", CONFIG_FILES, ids=CONFIG_IDS)
    def test_every_config_pins_a_tier(self, config_path):
        assert Config.from_yaml(str(config_path)).pipeline.visual_image_quality

    def test_a_typo_fails_at_config_load(self):
        # Not as an OpenAI 400 inside the visual Lambda, hours later and only on the day it renders.
        with pytest.raises(ValidationError):
            PipelineConfig(visual_image_quality="hihg")

    def test_empty_stays_a_valid_explicit_opt_out(self):
        assert PipelineConfig(visual_image_quality="").visual_image_quality == ""


class TestSourceSlotVocabulary:
    """source_slots keys are matched against item.source_type.value in the ranker's guaranteed-slot
    pass. A typo'd key matches no item at all: the guarantee silently disappears and the fill pass
    falls back to DEFAULT_SOURCE_SLOT for that source."""

    def test_the_default_keys_are_real_source_types(self):
        assert set(PipelineConfig().source_slots) <= {source.value for source in SourceType}

    def test_an_unknown_source_name_fails_load(self):
        with pytest.raises(ValidationError, match="unknown source type"):
            PipelineConfig(source_slots={"websearch": 2})  # the real value is 'web'

    def test_no_slots_at_all_is_allowed(self):
        # A legitimate config: no per-source guarantees (the origin cap still applies).
        assert PipelineConfig(source_slots={}).source_slots == {}

    @pytest.mark.parametrize("config_path", CONFIG_FILES, ids=CONFIG_IDS)
    def test_every_config_names_real_source_types(self, config_path):
        cfg = Config.from_yaml(str(config_path))
        assert set(cfg.pipeline.source_slots) <= {source.value for source in SourceType}


class TestSlotsVersusTopN:
    """`_apply_source_slots` stops at top_n, so once the guaranteed slots add up to it the relaxation
    passes are unreachable and the score only orders items WITHIN a source. The shipped config is
    exactly at that point (top_n 5, five 1-item slots) and it was recorded nowhere."""

    @staticmethod
    def _warnings(**overrides):
        # The project logger sets propagate=False, so caplog cannot see it; assert on the call.
        with patch("shared.config.logger") as log:
            PipelineConfig(**overrides)
        return [call.args[0] for call in log.warning.call_args_list]

    def test_slots_filling_the_whole_digest_warns(self):
        warnings = self._warnings(top_n=5, source_slots={"web": 1, "x": 1, "rss": 1, "reddit": 1, "youtube": 1})
        assert any("orders items WITHIN a source" in message for message in warnings)

    def test_a_slot_left_over_for_score_is_silent(self):
        assert self._warnings(top_n=6, source_slots={"web": 1, "x": 1, "rss": 1, "reddit": 1, "youtube": 1}) == []

    def test_a_disabled_slot_does_not_count_against_top_n(self):
        assert self._warnings(top_n=2, source_slots={"web": 1, "x": 0}) == []


class TestAlertOnEmptyVocabulary:
    """alert_on_empty is matched against the health report's source names. A name that is not a
    collector alerts on NOTHING, so the dark source it was meant to watch just stays dark."""

    def test_known_collector_names_are_accepted(self):
        assert CollectorsConfig(alert_on_empty=list(COLLECTOR_NAMES)).alert_on_empty == list(COLLECTOR_NAMES)

    def test_an_unknown_collector_name_fails_load(self):
        with pytest.raises(ValidationError, match="unknown collector"):
            CollectorsConfig(alert_on_empty=["websearch"])  # the collector is named 'web_search'

    def test_the_vocabulary_matches_the_collector_registry(self):
        # The names must be exactly what the runner can actually build; otherwise a source could be
        # named here that never appears in a health report (or vice versa).
        from unittest.mock import MagicMock

        from pipeline.runner import collector_registry

        assert set(collector_registry(Config(), MagicMock())) == set(COLLECTOR_NAMES)

    def test_the_vocabulary_matches_the_per_source_config_fields(self):
        assert set(COLLECTOR_NAMES) <= set(CollectorsConfig.model_fields)

    @pytest.mark.parametrize("config_path", CONFIG_FILES, ids=CONFIG_IDS)
    def test_every_config_watches_only_real_collectors(self, config_path):
        cfg = Config.from_yaml(str(config_path))
        assert set(cfg.collectors.alert_on_empty) <= set(COLLECTOR_NAMES)


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


class TestParkedSourceTripwires:
    """The two S3-parked sources (youtube, rsshub) are collected by a DAILY local sync cron, so
    park_max_age_hours has to be set in the file: the code default of 36 is MORE than one sync
    cadence, so a completely skipped sync day still read FRESH/OK — and with rsshub_desired_count=0
    the park file is the only X path there is.

    The empty-rate tripwire is per source, and only rsshub arms it. It targets all-200-with-no-entries
    (expired X cookies making every account feed answer empty), which trips no failure rate at all.
    YouTube has no such shape: a revoked key, an exhausted quota and a datacenter-IP block all RAISE
    and already report FAILED, so arming it there only fired on days when every low-cadence channel
    was quiet — a normal day, and exactly the alert fatigue alert_on_empty's comment argues against.
    """

    # The local sync runs once a day, so a park file older than one cadence has missed a whole run.
    _ONE_SYNC_CADENCE_HOURS = 24
    _PARKED_SOURCES = ("youtube", "rsshub")

    @pytest.mark.parametrize("config_path", CONFIG_FILES, ids=CONFIG_IDS)
    def test_the_empty_rate_tripwire_is_armed_for_the_source_with_a_silent_outage(self, config_path):
        source = Config.from_yaml(str(config_path)).collectors.rsshub
        assert source.empty_rate_threshold < EMPTY_RATE_CHECK_DISABLED

    @pytest.mark.parametrize("config_path", CONFIG_FILES, ids=CONFIG_IDS)
    def test_youtube_does_not_alert_on_a_quiet_day(self, config_path):
        # 9 weekly-cadence channels at max_videos_per_channel 1 all being quiet is 100% empty, which
        # any armed threshold reports as DEGRADED → an SNS 'Source Health' ALERT on a normal day.
        config = Config.from_yaml(str(config_path))
        assert config.collectors.youtube.empty_rate_threshold == EMPTY_RATE_CHECK_DISABLED
        assert "youtube" not in config.collectors.alert_on_empty

    @pytest.mark.parametrize("config_path", CONFIG_FILES, ids=CONFIG_IDS)
    @pytest.mark.parametrize("source_name", _PARKED_SOURCES)
    def test_a_skipped_sync_day_cannot_read_fresh(self, config_path, source_name):
        source = getattr(Config.from_yaml(str(config_path)).collectors, source_name)
        assert source.park_max_age_hours <= self._ONE_SYNC_CADENCE_HOURS

    # How many INPUTS a source declares, per collector name — what the failure RATE is computed over.
    _INPUT_LISTS = {
        "rss": lambda c: c.rss.feeds,
        "reddit": lambda c: c.reddit.subreddits,
        "youtube": lambda c: c.youtube.channels,
        "rsshub": lambda c: c.rsshub.accounts,
        "web_search": lambda c: [q for search in c.web_search.trend_searches for q in search.queries],
    }

    @pytest.mark.parametrize("config_path", CONFIG_FILES, ids=CONFIG_IDS)
    @pytest.mark.parametrize("source_name", sorted(_INPUT_LISTS))
    def test_a_source_too_small_for_a_rate_sets_the_absolute_count(self, config_path, source_name):
        """A source with too few inputs for the RATE to express a partial outage must arm
        max_failed_inputs, or DEGRADED is unreachable for it.

        The bound is derived from the knobs, not picked: the check is `failed/total*100 >
        error_rate_threshold`, so the smallest failure that can trip it needs more than
        100/threshold inputs. At the shipped 50.0 that is 3 — and reddit ships 2 subreddits, where 1
        of 2 is exactly 50.0 (not >, so clean) and 2 of 2 already raises FAILED. The reddit block had
        no max_failed_inputs at all and took the code default of 0: a clean OK with half of Reddit
        missing."""
        config = Config.from_yaml(str(config_path))
        source = getattr(config.collectors, source_name)
        inputs = len(self._INPUT_LISTS[source_name](config.collectors))
        if not inputs:
            pytest.skip(f"{source_name} declares no inputs in {config_path.name}")
        if inputs >= 100 / source.error_rate_threshold + 1:
            return
        assert source.max_failed_inputs > 0, (
            f"{source_name} has {inputs} input(s) and error_rate_threshold "
            f"{source.error_rate_threshold}: no failure count can exceed that rate, so DEGRADED "
            "is unreachable without max_failed_inputs"
        )

    def test_the_code_default_leaves_the_empty_rate_check_unreachable(self):
        # Documents WHY the file must set it: the default equals the sentinel, and the comparison
        # is strict, so no observed empty rate can ever exceed it.
        assert YouTubeCollectorConfig().empty_rate_threshold == EMPTY_RATE_CHECK_DISABLED
        assert (
            degradation_reason(
                total=10,
                failed=0,
                empty=10,
                what="channels",
                threshold=50.0,
                empty_threshold=EMPTY_RATE_CHECK_DISABLED,
                max_failed=0,
            )
            == ""
        )

    def test_an_armed_threshold_reports_an_all_empty_source(self):
        reason = degradation_reason(
            total=10,
            failed=0,
            empty=10,
            what="account feeds",
            threshold=50.0,
            empty_threshold=90.0,
            max_failed=0,
        )
        assert "returned nothing" in reason


class TestLoadNeverSilentlyShipsAnEmptyConfig:
    """`if not config_path.exists(): return cls()` was silent and wrong.

    config/config.yaml is gitignored and both Dockerfiles COPY config/, so a clean-checkout image —
    which is exactly what CI builds and validates — resolved to bare code defaults: empty
    rss/reddit/youtube lists and region us-east-1. Nothing raised; the first sign was a digest with
    no stories."""

    def test_a_missing_config_falls_back_to_the_tracked_template(self, tmp_path, monkeypatch):
        monkeypatch.setattr("shared.config.Path", _ConfigDirRedirect(tmp_path, keep=("config-template.yaml",)))
        config = Config.load()
        assert config.collectors.web_search.trend_searches == (
            Config.from_yaml(str(CONFIG_TEMPLATE)).collectors.web_search.trend_searches
        )

    def test_the_fallback_is_a_real_config_not_code_defaults(self):
        # What the CI image check asserts: the template carries no live source lists (deliberately),
        # but it does carry the trend searches, whose code default is empty. So "a config file backed
        # this load" is checkable, and it is what distinguishes the fallback from bare defaults.
        assert Config().collectors.web_search.trend_searches == []
        assert Config.from_yaml(str(CONFIG_TEMPLATE)).collectors.web_search.trend_searches

    def test_no_config_at_all_raises_instead_of_returning_defaults(self, tmp_path, monkeypatch):
        monkeypatch.setattr("shared.config.Path", _ConfigDirRedirect(tmp_path, keep=()))
        with pytest.raises(FileNotFoundError, match="No config found"):
            Config.load()

    def test_the_ci_image_check_loads_a_config(self):
        # Importing the handlers never loads a config, so the import check alone let a config-less
        # image ship green. The image is built from a clean checkout, so this step is the only place
        # the deployed artifact's config is exercised at all.
        workflow = (Path(__file__).resolve().parent.parent / ".github" / "workflows" / "ci.yml").read_text()
        assert "Config.load()" in workflow
        assert "c.collectors.web_search.trend_searches" in workflow


class _ConfigDirRedirect:
    """Stand-in for `shared.config.Path` that points Config.load()'s config dir at a temp directory,
    optionally pre-seeded with copies of the real config files. Only the `Path(__file__)` call inside
    load() is redirected; every other use of Path stays real."""

    def __init__(self, tmp_path, *, keep: tuple[str, ...]):
        self._config_dir = tmp_path / "config"
        self._config_dir.mkdir()
        for name in keep:
            (self._config_dir / name).write_text((CONFIG_DIR / name).read_text(encoding="utf-8"), encoding="utf-8")

    def __call__(self, *_args, **_kwargs):
        return self

    @property
    def parent(self):
        return self

    def __truediv__(self, other):
        return self._config_dir if other == "config" else self._config_dir / other


class TestCollectionWindowCoversTheGapBetweenRuns:
    """The cutoff is anchored at midnight at the END of the digest date, but the cron collects hours
    before that midnight. All five sources shipped lookback_hours 24 against a 10:00 UTC (19:00 KST)
    cron, so items published 19:00-24:00 KST were unpublished when that day's run collected and
    already before the next run's cutoff: five hours of every day, in no digest ever."""

    @pytest.mark.parametrize("config_path", CONFIG_FILES, ids=CONFIG_IDS)
    @pytest.mark.parametrize("source_name", sorted(COLLECTOR_NAMES))
    def test_every_source_reaches_back_to_the_previous_run(self, config_path, source_name):
        config = Config.from_yaml(str(config_path))
        run_hour_local = (int(config.aws.digest_cron_hour) + _utc_offset_hours(config.aws.timezone)) % 24
        assert getattr(config.collectors, source_name).lookback_hours >= 48 - run_hour_local

    def test_the_code_default_reaches_back_too(self):
        # A bare Config() is what a clean-checkout image resolves to, so the default must be valid
        # on its own — the validator running over it is exactly what proves that.
        assert Config().collectors.rss.lookback_hours >= 48 - 19

    def test_a_config_edit_cannot_reopen_the_hole(self):
        with pytest.raises(ValidationError, match="look back less than"):
            Config(collectors={"rss": {"lookback_hours": 24}})

    def test_a_multi_run_cron_expression_is_not_second_guessed(self):
        # "*/6" has no single gap between runs, so there is no floor to derive; the check steps aside
        # rather than inventing one.
        assert Config(aws={"digest_cron_hour": "*/6"}, collectors={"rss": {"lookback_hours": 1}})

    @pytest.mark.parametrize("source_name", sorted(COLLECTOR_NAMES))
    def test_no_source_accepts_a_window_of_zero_hours(self, source_name):
        # web_search and rsshub used to re-declare `lookback_hours: int = 72`, which replaces the
        # FieldInfo wholesale and dropped the ge=1 with it: those two accepted a NEGATIVE window while
        # the other three failed loudly. The constraint is declared once, so it holds for all five.
        with pytest.raises(ValidationError):
            Config(collectors={source_name: {"lookback_hours": 0}})

    def test_a_wider_window_is_still_configurable_per_source(self):
        config = Config(collectors={"web_search": {"lookback_hours": 72}})
        assert config.collectors.web_search.lookback_hours == 72

    def test_an_earlier_run_needs_a_wider_window(self):
        # A 06:00 KST run is 42h from the previous run's clock time to this run's reference midnight.
        with pytest.raises(ValidationError, match="42h"):
            Config(aws={"digest_cron_hour": "21"})  # 21:00 UTC = 06:00 KST


class TestLanguageRules:
    def test_one_form_per_proper_noun_and_particle_agreement(self):
        # Published Korean carried two spellings of the same company in one digest and particles
        # that disagreed with the form as written. One consolidated rule, no per-company name table.
        rules = PipelineConfig().digest_language_rules
        assert "ONE form per proper noun" in rules
        assert "particle" in rules

    def test_the_glossary_rule_is_stated_positively(self):
        # "never invent a Korean transliteration" is a prohibition that leaves the model to guess what
        # to do instead — and it guessed 홈랍. Stating the remaining option removes the choice.
        rules = PipelineConfig().digest_language_rules
        assert "stays in Latin script" in rules
        assert "transliteration" not in rules
