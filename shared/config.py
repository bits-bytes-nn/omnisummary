import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .constants import RSSHUB_PORT, VISUAL_ORIENTATIONS, LanguageModelId


class _StrictModel(BaseModel):
    """Base for every config model: reject unknown keys so a typo'd config.yaml key (e.g.
    `enable_thread_post`, `min_scor`) fails loudly at Config.load() instead of being silently
    dropped and falling back to a code default — which for the delivery toggles would silently
    mis-route or suppress the digest."""

    model_config = ConfigDict(extra="forbid")


# Korean prose conventions shared by EVERY Korean-output surface (daily digest + deep research,
# Slack + Threads). Kept in one place so the two features can't drift apart on register, anti-
# translationese, or the colon-enumeration ban. Composed into digest_language_rules and the
# research agent's <language> block.
KOREAN_STYLE_RULES: str = (
    "- Write natural, idiomatic Korean — NOT translationese. Avoid stiff translated-English "
    "patterns: drop redundant pronouns (그것은/이것은), avoid overusing passive voice and "
    "'~에 대해/~에 의해/~을 통해', don't calque English connectives. Read each sentence aloud — "
    "if it sounds like a machine translation, rewrite it the way a Korean tech writer would say it.\n"
    "- Use the plain declarative '~다' columnist register consistently (e.g. '~했다', '~이다'); "
    "NEVER the honorific '~입니다/~습니다'. Do not mix the two registers.\n"
    "- Do NOT glue two complete sentences together with punctuation Korean does not use that way. "
    "Not with a colon to introduce an enumeration ('핵심은 세 가지다: ...'), and not with a comma "
    "after a finished predicate ('성립한다, 그 순간이 오지 않으면 ...' / '시대다, 토큰이 곧 ...'). "
    "Once a clause ends in a final verb form the sentence is over: put a period and start a new "
    "one ('핵심은 세 가지다. 첫째는 ...'), or restructure so the first clause connects properly "
    "('성립하는데', '~라면', '~이므로'). A comma is fine INSIDE one sentence. "
    "Keep colons out of mid-prose entirely."
)


class BaseCollectorConfig(_StrictModel):
    enabled: bool = True
    lookback_hours: int = 24
    reference_time: datetime | None = None
    request_timeout: int = Field(default=30, ge=1)
    max_retries: int = Field(default=3, ge=1)
    retry_backoff_sec: int = Field(default=5, ge=0)
    # Age budget for a source's S3 park file (written by the local sync scripts). Beyond it the
    # items are still used — stale beats empty — but the source is reported STALE so a stopped
    # local cron is visible instead of looking like a healthy run.
    park_max_age_hours: int = Field(default=36, ge=1)
    # Share of a source's inputs (feeds / accounts / channels / queries) that may fail before the
    # source is reported DEGRADED. Reporting only: every collected item still reaches the
    # aggregator. One number for every collector — a source that answers from 2 of 40 inputs looks
    # exactly like a healthy one in the item count alone.
    error_rate_threshold: float = Field(default=50.0, ge=0.0, le=100.0)


class YouTubeCollectorConfig(BaseCollectorConfig):
    channels: list[str] = Field(default_factory=list)
    max_videos_per_channel: int = Field(default=3, ge=1)
    resolve_timeout: int = Field(default=15, ge=1)
    transcript_timeout: int = Field(default=15, ge=1)
    transcript_language: str = Field(default="en")
    # How many channels may be collected at once. Each one parks worker threads (page scrape,
    # transcript fetches), so this stays at/below the default asyncio executor width, same bound as
    # rss.max_concurrency / rsshub.max_concurrency.
    max_concurrency: int = Field(default=5, ge=1)

    @property
    def channel_budget_sec(self) -> int:
        """Wall-clock budget for ONE channel's collection: id resolution + the two Data API calls
        (each retried, with backoff) + one transcript fetch per kept video. Derived from the
        existing per-step knobs so there is no second number that can silently drift out of sync
        and start failing healthy channels."""
        api = (self.request_timeout + self.retry_backoff_sec) * self.max_retries * 2
        return self.resolve_timeout + api + self.transcript_timeout * self.max_videos_per_channel


class RedditCollectorConfig(BaseCollectorConfig):
    # Empty like every other source list: a live source list must come from config.yaml, never
    # from a code default that would silently collect from subreddits nobody configured.
    subreddits: list[str] = Field(default_factory=list)
    sort: Literal["hot", "top", "new"] = "hot"
    limit: int = Field(default=20, ge=1)


class RSSCollectorConfig(BaseCollectorConfig):
    feeds: list[str] = Field(default_factory=list)
    # How many feeds may be fetched at once. Each feedparser.parse parks a worker thread, so this
    # stays at/below the default asyncio executor width (min(32, cpu+4) — 6 on a 2-vCPU Lambda);
    # oversubscribing it made a feed's timeout expire while its parse was still queued, turning a
    # healthy feed into a bogus FAILURE. Same bound as rsshub.max_concurrency. Worst-case wall time
    # is ceil(feeds / max_concurrency) * request_timeout, well inside the 15-min Lambda.
    max_concurrency: int = Field(default=5, ge=1)


class TrendSearch(_StrictModel):
    name: str
    queries: list[str]
    domains: list[str] = Field(default_factory=list)
    topic: Literal["news", "general"] = "news"


class WebSearchCollectorConfig(BaseCollectorConfig):
    trend_searches: list[TrendSearch] = Field(default_factory=list)
    max_results_per_query: int = Field(default=10, ge=1)
    lookback_hours: int = 72
    refine_model: LanguageModelId = LanguageModelId.CLAUDE_V5_SONNET
    max_refine_queries: int = Field(default=3, ge=1)
    min_search_score: float = Field(default=0.3, ge=0.0, le=1.0)
    # How many search queries may be in flight against Tavily at once. Unbounded fan-out threw every
    # configured query at the API simultaneously, so a large trend list self-throttled and lost whole
    # queries. Same bound as rss.max_concurrency / rsshub.max_concurrency.
    max_concurrency: int = Field(default=5, ge=1)


class RSSHubAccount(_StrictModel):
    username: str
    platform: str


class RSSHubCollectorConfig(BaseCollectorConfig):
    base_url: str = f"http://localhost:{RSSHUB_PORT}"
    accounts: list[RSSHubAccount] = Field(default_factory=list)
    lookback_hours: int = 72
    # How many account feeds may be fetched at once. Each fetch parks a worker thread, so this
    # stays at/below the default asyncio executor width (min(32, cpu+4) — 6 on a 2-vCPU Lambda);
    # oversubscribing it made a feed's timeout expire while its parse was still queued. Worst-case
    # wall time is ceil(accounts / max_concurrency) * request_timeout, well inside the 15-min Lambda.
    max_concurrency: int = Field(default=5, ge=1)


class CollectorsConfig(_StrictModel):
    youtube: YouTubeCollectorConfig = Field(default_factory=YouTubeCollectorConfig)
    reddit: RedditCollectorConfig = Field(default_factory=RedditCollectorConfig)
    rss: RSSCollectorConfig = Field(default_factory=RSSCollectorConfig)
    web_search: WebSearchCollectorConfig = Field(default_factory=WebSearchCollectorConfig)
    rsshub: RSSHubCollectorConfig = Field(default_factory=RSSHubCollectorConfig)
    # Sources whose EMPTY result is an INCIDENT worth an alert, by collector name (e.g.
    # ["rss", "web_search"]). A dark source produces no items, no exception and no stale park file,
    # so nothing else notices it; but reddit/x are legitimately quiet on many days, which is why
    # this is an explicit opt-in list rather than "alert whenever any source is empty" — that would
    # page daily and be muted within a week. Empty (the default) never alerts on EMPTY.
    alert_on_empty: list[str] = Field(default_factory=list)

    def set_reference_time(self, reference_time: datetime) -> None:
        for cfg in (self.youtube, self.reddit, self.rss, self.web_search, self.rsshub):
            cfg.reference_time = reference_time


# gpt-image sizes are "<width>x<height>"; anything else is rejected at config load rather than
# surfacing as an OpenAI 400 in the visual Lambda, hours later and only on the day it renders.
_IMAGE_SIZE_RE = re.compile(r"\d+x\d+")


class PipelineConfig(_StrictModel):
    top_n: int = Field(default=7, ge=1)
    min_score: float = Field(default=0.6, ge=0.0, le=1.0)
    # Per-source safety net: a source with a guaranteed slot whose BEST item falls just below
    # min_score (within this grace band) still gets that one item considered, so a source the
    # absolute-scoring prompt systematically under-rates (e.g. video/podcast transcripts vs
    # articles) isn't shut out entirely. 0 disables the grace (strict threshold for all).
    source_slot_score_grace: float = Field(default=0.1, ge=0.0, le=0.5)
    # Extra ranked candidates handed to the digest generator beyond top_n, so that when the
    # editor MERGES same-event items (e.g. two takes on one launch) it can still backfill to
    # exactly top_n distinct stories instead of emitting fewer. 0 disables the buffer.
    # The source_slots guarantees are enforced on the top_n CORE, not on top_n + this buffer: the
    # buffer items are handed over flagged as backfill, so a source's guaranteed slot can no longer
    # be "satisfied" by a candidate the editor never publishes.
    digest_candidate_buffer: int = Field(default=3, ge=0)
    # Days a published URL stays in the cross-day dedup ledger; an article seen within this
    # window is skipped so the digest doesn't re-summarize the same story days apart.
    published_url_ttl_days: int = Field(default=6, ge=1)
    # How many recent digest leads to feed back into the prompt as "don't reuse these angles".
    recent_leads_window: int = Field(default=5, ge=0)
    # How many recent visual formats (orientation + style) to track for deliberate variation.
    visual_format_window: int = Field(default=6, ge=0)
    # Target share of recent visuals that should be multi-panel (vs a single frame). The editor
    # leans single-frame on its own, so when the recent window falls below this the prompt nudges
    # toward a multi-panel composition; above it, toward a single frame. A soft steer, not a quota:
    # the story still decides. 0 disables the nudge entirely (pure editor choice).
    visual_multi_panel_target_ratio: float = Field(default=0.34, ge=0.0, le=1.0)
    ranking_model: LanguageModelId = LanguageModelId.CLAUDE_V5_SONNET
    digest_model: LanguageModelId = LanguageModelId.CLAUDE_V5_SONNET
    # Character budget for one item's `body` + `implication` prose, stated to the editor. Derived,
    # not arbitrary: a Threads post caps at 500 characters and the renderer spends the rest on the
    # parts CODE owns — the display title, the "r/LocalLLaMA · 👍 +44" source line, the URL and the
    # blank-line separators (~120 chars in practice). The renderer still enforces the real cap by
    # dropping trailing sentences; 5 of 95 sampled items lost their closing sentence that way (a
    # median of 106 characters, usually the concrete figures) purely because the editor was never
    # told the budget. 0 states no budget, for a deployment whose channel has no post cap.
    digest_item_prose_max_chars: int = Field(default=380, ge=0)
    # A digest whose JSON the editor emits malformed (a stray bracket after the lead string cost
    # the 2026-08-13 and 2026-08-17 digests every story) is RE-ASKED before it is given up on.
    # Regeneration is one Bedrock call; the alternative is a day with no stories at all.
    digest_max_retries: int = Field(default=3, ge=1)
    digest_retry_backoff_sec: float = Field(default=5.0, ge=0)
    # Post-generation faithfulness pass: verify the digest's specific claims against the
    # source items and surgically revise unsupported ones (prompt rules alone couldn't
    # move the faithfulness score). Best-effort; disable to skip the extra LLM call.
    enable_grounding_check: bool = True
    # Language rules injected into the digest prompt's *Language* block. Defaults to the
    # Korean editorial rules + translation glossary; other deployments can override to
    # write the digest in another language without forking the prompt.
    digest_language_rules: str = (
        "- Write in Korean (95%+); English ONLY for proper nouns and untranslatable technical terms. "
        "Use ONE form per proper noun across the whole digest, as the source writes it — never invent "
        "a Korean transliteration for a term outside the glossary below — and make the particle after "
        "it agree with that written form (OpenAI는, GPT-5가).\n"
        "- Translate terms that have established Korean equivalents: architecture → 아키텍처, "
        "benchmark → 벤치마크, inference → 추론, training → 학습, deployment → 배포, "
        "weight → 가중치, parameter → 파라미터, token → 토큰, open-source → 오픈소스, "
        "pipeline → 파이프라인, optimization → 최적화, compression → 압축, memory → 메모리.\n"
        "- General words MUST be Korean: practitioner → 실무자, implication → 시사점, "
        "release → 출시/공개, breakthrough → 돌파구, approach → 접근법, ecosystem → 생태계.\n"
        "- If the original item title is in English, translate it to Korean for the display text.\n"
        + KOREAN_STYLE_RULES
    )
    # Audience/domain the ranking and digest prompts target. Configurable so the pipeline can
    # be reused across domains without forking the prompts.
    ranking_audience_description: str = "a daily digest aimed at practicing ML engineers"
    digest_audience_description: str = "ML engineers"
    # Editorial voice for the digest lead/implications: a recurring narrator persona (a singularity-
    # believing, science-geek/sci-fi technologist) written in a dry Gruber register — opinionated,
    # evidence-obsessed, watching who captures AI's upside; critiques ideas/decisions, never a
    # person. Configurable so the tone can be retuned without forking the prompt.
    digest_voice_guidance: str = (
        "WHO YOU ARE — a character, not a neutral news desk. A technologist who believes the "
        "singularity and AGI are coming — not 'if' but 'when'. A lifelong science geek and sci-fi "
        "reader who grew up imagining this future, now scanning the daily news for real evidence "
        "it's arriving. But an engineer at heart: allergic to noise, and clear-eyed that the real "
        "question isn't 'will the tech work' but 'who does this abundance flow to' — so you cheer "
        "the acceleration AND interrogate who captures its upside. Two instincts at once: you back "
        "acceleration yet watch distribution, so you fit no political camp — it reads as sharp "
        "economic intuition, never a party slogan or manifesto.\n"
        "HOW YOU SOUND — write like John Gruber: dry, well-informed, genuinely funny, committed to a "
        "real opinion. The wit is deadpan and economical, NOT cute or whimsical — keep the charm in "
        "the mascot, keep the prose lean. Let the STORY set intensity: a real acceleration signal "
        "earns conviction; hype or cherry-picked benchmarks or 'this time it's really AGI' earns a "
        "clean, plain-spoken dismissal (that dressed-up-as-science stuff is the one thing you openly "
        "can't stand); hypocrisy or a power grab earns a pointed jab; a routine release earns a wry, "
        "low-key read. Aim skepticism only at things posing as proof — genuine unknowns earn "
        "curiosity, never scorn, and never punch down at a person. A bot hyped every day, angry "
        "every day, or neutral every day is equally boring; the edge is a clear stance, not a fixed "
        "mood — and not every item is a debunking.\n"
        "OPTIONAL SEASONING (each at most once per digest, only if it lands naturally — skip when "
        "forced): a one-line statement of the creed on a genuinely huge day; a wry scientific or "
        "sci-fi aside on an ordinary detail; a brief first-person flash of candor or conviction. "
        "These are rare spices, never the main course; most lines carry none.\n"
        "CRAFT — Spice is in the FRAMING, never the facts: sharpen the angle and contrast, but never "
        "state an inference as a reported fact; mark a sharp reading as judgment ('~로 읽힌다', "
        "'보통 이런 구도는'). Critique ideas, decisions, "
        "incentives — never a person. Plain words over "
        "jargon and short sentences over dense ones, but never drop a rung of the argument to sound "
        "simple — 'easy' is the wording, not a missing step. On a pure-tech story keep the "
        "power/distribution lens light or absent. Concentrate the persona in the LEAD (the standalone "
        "hook); keep item bodies cleaner and let the edge land in each item's closing line. Plain "
        "declarative '~다' register. Ground every take in the supplied facts and trends, never vibes."
    )
    # Tongue-in-cheek "AGI countdown" intro prepended to the digest lead (code computes the day
    # count from agi_countdown_date — never the LLM — so it stays accurate and ticks down daily).
    # The date is a fixed, defensible ~3-years-out target (Jensen Huang's call / tail of Amodei's
    # "1-3 years"); tune it here. Empty date disables the intro.
    agi_countdown_date: str = "2029-01-01"
    # Plain declarative ('~다') to match the columnist body voice — not honorific ('~입니다').
    # Before the D-day it counts down; on/after it, agi_countdown_after counts up (the prediction
    # blew past — a self-aware joke). Empty agi_countdown_date disables the intro entirely.
    agi_countdown_template: str = "AGI 등장 {days}일 전이다. "
    agi_countdown_after: str = "AGI 등장 예정일 D+{days}일째, 아직이다. "
    # Where the gag sits in the lead. "prefix" opens with it — which does spend the ROOT'S FIRST
    # LINE, the one line most feed readers ever see, on the same fixed sentence every day (40
    # consecutive posts opened with it). That is deliberate: the countdown is the account's
    # SIGNATURE, and the owner ranked recognisable branding above the first-line reach argument.
    # "suffix" keeps the gag verbatim but moves it to a closing sign-off, for a deployment that
    # would rather open on the day's angle. Position only: no cadence/skip knobs, which would just
    # be magic numbers.
    agi_countdown_position: Literal["prefix", "suffix"] = "prefix"
    item_text_max_tokens: int = Field(default=8000, ge=1)
    ranking_batch_size: int = Field(default=40, ge=1)
    # Share of the ranking model's context window one batch of items may fill. The rest is the
    # system prompt plus the JSON the model writes back; a batch that overflows fails the Converse
    # call and used to take its whole batch of candidates with it.
    ranking_batch_token_budget_ratio: float = Field(default=0.7, gt=0.0, le=1.0)
    # Context window assumed when the model registry has no entry for the ranking model (so the
    # budget above still bounds the batch instead of falling back to an unbounded one).
    ranking_context_window_fallback: int = Field(default=200_000, ge=1)
    # How many ranking batches may be in flight against Bedrock at once. Unbounded fan-out threw
    # every batch at Converse simultaneously, so a large day self-throttled (ThrottlingException)
    # and dropped whole batches of candidates.
    ranking_max_concurrency: int = Field(default=4, ge=1)
    # A batch that fails is RETRIED before it is given up on: a single throttle used to silently
    # delete ~40 candidates from the day's pool with only a warning.
    ranking_max_retries: int = Field(default=3, ge=1)
    ranking_retry_backoff_sec: float = Field(default=5.0, ge=0)
    # Minimum share of a batch's items the ranker must actually score. A model that quietly omits
    # ids returns a valid response, so those candidates never reach the digest; below this ratio the
    # omitted items get ONE extra re-ask (the shortfall is logged either way). A full-coverage
    # batch — the normal case — makes no extra Bedrock call. 1.0 re-asks on any omission, 0 never.
    ranking_min_coverage_ratio: float = Field(default=0.9, ge=0.0, le=1.0)
    # Per-source guaranteed slots, applied to the top_n stories the READER gets (never to the padded
    # top_n + digest_candidate_buffer candidate list).
    source_slots: dict[str, int] = Field(
        default_factory=lambda: {
            "web": 2,
            "x": 2,
            "rss": 1,
            "reddit": 1,
            "youtube": 1,
        }
    )
    source_cap_multiplier: int = Field(default=2, ge=1)
    max_per_origin: int = Field(default=1, ge=1)
    origin_weights: dict[str, float] = Field(default_factory=dict)
    origin_weight_default: float = Field(default=1.0, ge=0.0)
    origin_weight_nudge: float = Field(default=0.1, ge=0.0, le=1.0)
    # Engagement bonus tiers (views threshold -> score bonus) the ranking prompt applies
    # to items carrying view counts. Tunable instead of baked into the prompt text.
    engagement_tiers: list[tuple[int, float]] = Field(
        default_factory=lambda: [(10000, 0.05), (100000, 0.1), (500000, 0.15)]
    )
    # Taxonomy the ranking prompt assigns to each item. Configurable so non-AI
    # deployments can supply their own categories.
    ranking_categories: list[str] = Field(
        default_factory=lambda: [
            "research",
            "tools",
            "news",
            "release",
            "industry",
            "paper",
            "interview",
            "infrastructure",
            "community",
        ]
    )
    # Score the ranking prompt assigns to duplicate items within a same-topic cluster.
    ranking_duplicate_score_penalty: float = Field(default=0.3, ge=0.0, le=1.0)
    # Score-calibration buckets the ranking prompt applies, injected as template text so
    # ops can retune the distribution without editing the prompt.
    ranking_scoring_rubric: str = (
        "0.9+: field-defining. 0.8-0.89: very important. 0.7-0.79: notable. "
        "0.6-0.69: worth noting (digest bar). <0.6: low value."
    )
    trend_model: LanguageModelId = LanguageModelId.CLAUDE_V5_SONNET
    trend_retention_days: int = Field(default=30, ge=1)
    trend_cooling_days: int = Field(default=7, ge=1)
    trend_max_evidence: int = Field(default=5, ge=1)
    trend_max_active_trends: int = Field(default=10, ge=1)
    trend_momentum_half_life_days: float = Field(default=7.0, gt=0)
    # Delivery channels for the digest, each independently toggleable. Slack on by default;
    # Threads off until its access token / user id are provisioned in SSM.
    enable_slack_post: bool = True
    enable_threads_post: bool = False
    enable_daily_visual: bool = True
    image_model: str = "gpt-image-2"
    # orientation -> gpt-image size. The synopsis brief picks the orientation that fits the
    # visual (wide strip / tall infographic / square meme); not locked to one aspect ratio. The KEYS
    # are not free-form: they are the VisualBrief orientation vocabulary, checked below.
    image_sizes: dict[str, str] = Field(
        default_factory=lambda: {
            "square": "1024x1024",
            "landscape": "1536x1024",
            "portrait": "1024x1536",
        }
    )
    # Bounds on the gpt-image HTTP call. The OpenAI SDK defaults (600s timeout x 2 retries) can
    # exceed the visual Lambda's 15-min budget; one 300s attempt leaves room for the single
    # moderation-softened re-render and still finishes inside the Lambda.
    visual_image_timeout_sec: int = Field(default=300, ge=10)
    # gpt-image quality tier. Empty sends nothing, leaving OpenAI's "auto" — which picks between
    # tiers whose published per-image prices differ ~4x ($0.041-0.053 medium vs $0.165-0.211 high at
    # our sizes), so the monthly bill for one image a day is anywhere from ~$1.3 to ~$5.2 and the
    # code cannot say which. Set it to make the cost deterministic; the render also logs the
    # response's token counts either way, so actual spend is measurable rather than estimated.
    visual_image_quality: str = ""
    visual_image_max_retries: int = Field(default=0, ge=0)
    # Cap on the research steps the visual editor may request (each is a live search call). Matches
    # the "1-3 steps" the editor prompt asks for, so a chatty plan can't fan out into ten searches.
    visual_research_max_steps: int = Field(default=3, ge=1)
    visual_synopsis_source_max_tokens: int = Field(default=2000, ge=1)
    visual_synopsis_context_max_tokens: int = Field(default=1500, ge=1)
    # Emoji prefixed to the Slack caption of a generated visual, for scannability.
    visual_caption_emoji: str = "🎨"
    # Audience/domain the visual prompts target. Configurable so the visual pipeline can
    # be reused across domains without forking the prompts.
    visual_audience_description: str = "a daily AI/ML digest aimed at practicing ML engineers"
    # Language rules for visual output: which language the title/caption use and which
    # language must appear inside the rendered image (image models garble non-Latin glyphs).
    visual_caption_language: str = "Korean"
    visual_on_image_language: str = "SHORT ENGLISH (the image model garbles Korean and other non-Latin glyphs)"
    # Style/humor guidance injected into the visual synopsis prompt. Configurable so the
    # visual pipeline's tone can be retuned (or reused for non-AI domains) without forking.
    visual_synopsis_style_guidance: str = (
        "Multi-panel: same characters and a single consistent, polished art style across panels; "
        "each panel follows from the previous so the sequence reads in order without explanation."
    )
    visual_synopsis_humor_guidance: str = (
        "For comics/cartoons, aim for genuinely funny and shareable — internet-humor sensibility, "
        "a clear setup-and-payoff, expressive characters — in a clean, modern, appealing illustration style."
    )
    # Default aesthetic injected into the image-generation prompt. Configurable so the visual
    # pipeline's look can be retuned (or reused for non-AI domains) without editing the prompt.
    visual_synopsis_style_aesthetic: str = (
        "clean, modern, polished illustration with sound craftsmanship — correct proportions and "
        "perspective, coherent anatomy for any figures, balanced composition, and a deliberate, "
        "harmonious color palette; aim for a professionally art-directed look, never sloppy or distorted"
    )
    # Guardrails appended to every visual instruction. Not style, not a thesis requirement: two
    # things the image must not DO. (1) The digest's angle is handed over as context the art
    # director may ignore, but a 2026-08-18 run turned a lead about circular vendor financing
    # ("who is holding the risk won't be clear until the next downturn") into a triumphal
    # rocket-and-money poster — the opposite register. Not-contradicting is far weaker than the
    # "the image must argue the lead's thesis" rule that was rejected as over-constraining.
    # (2) A 2026-08-15 visual cast the model race as four athletes whose ethnicity stood in for
    # their lab's country. Depicting real, identifiable people IS allowed and wanted — recognising
    # the figures is normal editorial-cartoon practice — but a company or a country is not a race.
    # Empty disables either clause.
    visual_guardrails: str = (
        "Do NOT contradict the editorial angle you were given: if the angle is skeptical or "
        "cautionary, the image must not read as celebratory (and vice versa). You need not argue "
        "the angle — just don't invert it. "
        "Recognisable depictions of real people are fine. Do NOT, however, personify a company or "
        "a country as an ethnically-coded human; use the real people involved, or stylised "
        "figures, objects and mascots instead."
    )
    # Appended to the instruction when the image model's moderation blocks the first render,
    # to soften tone before a single retry. Configurable so ops can retune the safe-for-work
    # guidance without editing code.
    visual_moderation_softening_instruction: str = (
        "IMPORTANT: keep it clearly safe-for-work and good-natured. "
        "Use brand mascots/logos and generic stylized characters rather than realistic "
        "depictions of real named individuals; avoid anything that could read as defamatory."
    )
    # Recurring mascot = the visual embodiment of the digest's narrator persona (the singularity-
    # believing science-geek technologist). Appears only SOME days, when the editor judges it fits
    # the story — like the multi-panel nudge, character presence is a variation axis, not a daily
    # lock. Identity rides on a FEW signature cues (hair cowlick, retro glasses, two-tone cardigan),
    # not an exhaustive prop list, so he reads as the same person while pose/outfit/expression vary
    # day to day instead of converging on one identical look.
    visual_character_enabled: bool = True
    # Roughly what share of visuals should feature the character; the editor still skips it when a
    # story (e.g. a pure architecture explainer) reads better as a concept visual. 0 disables.
    visual_character_target_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    # Kept deliberately LEAN: an over-specified sheet (fixed outfit down to the socks, a repeated
    # "glasses pushed up on the forehead" gag) made every render converge on the same overfit image.
    # We fix only a small identity core and explicitly invite daily variation; framing favors a
    # waist-up shot because full-body figures are where gpt-image proportions/hands break most.
    visual_character_sheet: str = (
        "A recurring Korean man in his late 20s — a cute, earnest tech nerd, the narrator made "
        "visible. Keep him recognizable by just a FEW identity cues, drawn naturally: messy black "
        "hair with an upward cowlick; chunky rounded retro glasses worn normally ON HIS EYES; and a "
        "two-tone color-blocked knit cardigan (mustard + sage green) over a plain tee. Everything "
        "else — exact outfit details, pose, props, expression — should VARY day to day; do not lock "
        "him into one identical look, and do NOT repeat any single gag (e.g. glasses pushed up on "
        "the forehead) across visuals. Warm, bright, charming — never a dark-room hacker. He reacts "
        "to today's story (wonder, a skeptical squint, fired-up conviction) as a witness inside the "
        "scene, not as a replacement for depicting it. Prefer a WAIST-UP or medium shot so his face "
        "and reaction carry the moment; only show a full figure when the composition truly needs it. "
        "Draw him with correct, natural human anatomy and proportions and well-formed hands, as a "
        "polished, professionally-drawn character in whatever art style the day calls for."
    )

    @model_validator(mode="after")
    def _image_sizes_match_the_orientation_vocabulary(self) -> "PipelineConfig":
        """image_sizes is advertised as overridable, but its KEYS are the VisualBrief orientation
        vocabulary: the editor is offered `", ".join(image_sizes)` and the brief's orientation is
        looked up in the same dict. A renamed or dropped key used to make every brief either fail
        validation or coerce to the default orientation, with no signal at all."""
        expected = set(VISUAL_ORIENTATIONS)
        actual = set(self.image_sizes)
        if actual != expected:
            raise ValueError(
                "pipeline.image_sizes keys must be exactly the visual orientation vocabulary "
                f"{sorted(expected)} (got {sorted(actual)}); the keys are what the visual editor is "
                "offered and what VisualBrief.orientation is validated against, so only the SIZES "
                "are tunable here"
            )
        bad_sizes = sorted(size for size in self.image_sizes.values() if not _IMAGE_SIZE_RE.fullmatch(size))
        if bad_sizes:
            raise ValueError(f"pipeline.image_sizes values must look like '1024x1536' (got {bad_sizes})")
        return self


class AgentConfig(_StrictModel):
    model_id: LanguageModelId = LanguageModelId.CLAUDE_V5_SONNET
    community_search_domains: list[str] = Field(
        default_factory=lambda: ["twitter.com", "x.com", "reddit.com", "news.ycombinator.com", "substack.com"]
    )
    search_result_limit: int = Field(default=5, ge=1)
    search_content_preview_chars: int = Field(default=300, ge=1)
    search_request_timeout: int = Field(default=30, ge=1)
    search_max_retries: int = Field(default=3, ge=1)
    search_retry_backoff_sec: int = Field(default=2, ge=0)
    search_paper_max_authors: int = Field(default=3, ge=1)
    search_paper_abstract_max_chars: int = Field(default=200, ge=1)
    recall_memory_top_k: int = Field(default=5, ge=1)
    boto_read_timeout: int = Field(default=300, ge=1)
    boto_connect_timeout: int = Field(default=60, ge=1)
    boto_max_attempts: int = Field(default=3, ge=1)
    # Deep-research soft knobs the agent follows as guidance (interpolated into its prompt, so
    # editing these actually changes behavior — they are not enforced loop bounds).
    research_breadth: int = Field(default=4, ge=1)
    research_max_iterations: int = Field(default=3, ge=1)
    research_slack_target_words: int = Field(default=1500, ge=200)
    # Hard cap on the number of Threads posts (root + replies) a research report may become.
    # Code-enforced so a too-long report can't fan out into dozens of public posts even if the
    # agent ignores the prompt's "write a short Threads version" instruction.
    research_max_threads_posts: int = Field(default=6, ge=1)
    # Hard cap on a single page's extracted text (read_url tool).
    research_content_cap_chars: int = Field(default=50000, ge=1000)
    # OG-image attachment (deep-research delivery only).
    og_image_timeout_sec: int = Field(default=10, ge=1)
    og_image_max_bytes: int = Field(default=8_000_000, ge=10_000)
    # Cap how many images one research run may stage, bounding per-invocation memory.
    research_max_staged_images: int = Field(default=4, ge=1)


class SlackConfig(_StrictModel):
    bot_token: str = ""
    channel_id: str = ""


class AWSConfig(_StrictModel):
    region: str = "us-east-1"
    bedrock_region: str = "us-west-2"
    profile: str = ""
    project_name: str = "omnisummary"
    stage: str = "dev"
    timezone: str = "Asia/Seoul"
    vpc_id: str = ""
    subnet_ids: list[str] = Field(default_factory=list)
    state_bucket_name: str = ""
    s3_prefix: str = ""
    # EventBridge cron is UTC. 10:00 UTC = 19:00 KST (daily 7pm).
    digest_cron_hour: str = "10"
    digest_cron_minute: str = "0"
    # Threads long-lived tokens expire after 60 days; refresh comfortably inside that window.
    threads_token_refresh_days: int = Field(default=50, ge=1, le=59)
    # Running tasks for the RSSHub Fargate service. 0 by default because the digest never reaches
    # it: RSSHubCollector reads the S3 park file FIRST and returns early when it is usable, and the
    # local sync cron refreshes that file before every run (verified in the digest logs — X items
    # are served from rsshub_items.json, and the service was the account's only Fargate task at
    # ~$40/30d). The task definition is still deployed, so raising this to 1 restores the in-AWS
    # fallback for a day when the local sync has stopped (which now reports the source as STALE).
    rsshub_desired_count: int = Field(default=0, ge=0)
    api_throttle_rate_limit: int = Field(default=20, ge=1)
    api_throttle_burst_limit: int = Field(default=10, ge=1)
    waf_rate_limit: int = Field(default=2000, ge=100)


class Config(_StrictModel):
    collectors: CollectorsConfig = Field(default_factory=CollectorsConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    slack: SlackConfig = Field(default_factory=SlackConfig)
    aws: AWSConfig = Field(default_factory=AWSConfig)

    @classmethod
    def from_yaml(cls, file_path: str) -> "Config":
        with open(file_path, encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data if config_data else {})

    @classmethod
    def load(cls) -> "Config":
        load_dotenv()
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        if not config_path.exists():
            return cls()
        return cls.from_yaml(str(config_path))


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Process-wide cached Config for READ-ONLY leaf callers (delivery, research backends, agent
    tools, og-image). Config.load() re-reads and re-validates the whole YAML on every call, which
    a single research run did dozens of times — once per tool invocation.

    Deliberately NOT used where the Config is MUTATED (main/handlers' set_reference_time, the CI
    synth's vpc_id, the infra tests' bucket override): those callers need their own instance, and
    a shared cached object would leak their edits everywhere. Tests clear the cache via an autouse
    fixture in tests/conftest.py."""
    return Config.load()
