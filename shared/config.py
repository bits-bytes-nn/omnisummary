import os
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .constants import (
    COLLECTOR_NAMES,
    RSSHUB_PORT,
    VISUAL_ORIENTATIONS,
    LanguageModelId,
    SourceType,
)
from .logger import is_running_in_aws, logger


class _StrictModel(BaseModel):
    """Base for every config model: reject unknown keys so a typo'd config.yaml key (e.g.
    `enable_thread_post`, `min_scor`) fails loudly at Config.load() instead of being silently
    dropped and falling back to a code default — which for the delivery toggles would silently
    mis-route or suppress the digest."""

    model_config = ConfigDict(extra="forbid")


# Korean prose conventions shared by EVERY Korean-output surface (daily digest + deep research).
# Kept in one place so the two features can't drift apart on register, anti-
# translationese, or the colon-enumeration ban. Composed into digest_language_rules and the
# research agent's <language> block.
KOREAN_PROSE_RULES: str = (
    "- Write it in Korean first, not English thought put into Korean words. A Korean sentence has a "
    "concrete subject that does something: the company, the person, the model, with a verb. "
    "Machine-written Korean instead makes abstract nouns the actors ('관심이 성능에서 청구서로 옮겨 간다', "
    "'승부는 소유권에서 갈린다', '~가 변수가 된다', '~의 비용을 정한다') and ends on a maxim; say the plain "
    "thing a Korean reporter would say out loud. Drop redundant pronouns (그것은/이것은), passive voice "
    "and '~에 대해/~에 의해/~을 통해'.\n"
    "- Say what IS, directly. Do not build a sentence on the 'A가 아니라 B' / 'A보다 B' contrast "
    "unless someone actually claimed A; it is the most worn formula in machine-written commentary.\n"
    "- Do NOT glue two complete sentences together with punctuation Korean does not use that way. "
    "Not with a colon to introduce an enumeration ('핵심은 세 가지다: ...'), and not with a comma "
    "after a finished predicate ('성립한다, 그 순간이 오지 않으면 ...' / '시대다, 토큰이 곧 ...'). "
    "Once a clause ends in a final verb form the sentence is over: put a period and start a new "
    "one ('핵심은 세 가지다. 첫째는 ...'), or restructure so the first clause connects properly. "
    "A comma is fine INSIDE one sentence. "
    "Keep colons out of mid-prose entirely."
)
# The columnist register of the digest and the research report. Kept out of KOREAN_PROSE_RULES
# because a comment reply talks TO a person, where the '~다' column voice reads as a lecture.
KOREAN_STYLE_RULES: str = (
    KOREAN_PROSE_RULES + "\n- Use the plain declarative '~다' columnist register consistently (e.g. '~했다', '~이다'); "
    "NEVER the honorific '~입니다/~습니다'. Do not mix the two registers."
)


class BaseCollectorConfig(_StrictModel):
    enabled: bool = True
    # Must reach back to the PREVIOUS run (floor 48 - run's local hour, checked by
    # Config._lookback_reaches_the_previous_run); 30 is the 19:00 KST floor plus an hour. Declared ONCE:
    # a subclass re-declaration replaces the FieldInfo and drops ge=1. Wider windows go in the YAML.
    lookback_hours: int = Field(default=30, ge=1)
    reference_time: datetime | None = None
    request_timeout: int = Field(default=30, ge=1)
    max_retries: int = Field(default=3, ge=1)
    retry_backoff_sec: int = Field(default=5, ge=0)
    # Age budget for a source's S3 park file (written by the local sync scripts). Beyond it the
    # items are still used — stale beats empty — but the source is reported STALE so a stopped
    # local cron is visible. A healthy park is ~30 min old at the digest; a looser budget let a stray
    # daytime sync hide the next evening's failed one.
    park_max_age_hours: int = Field(default=12, ge=1)
    # Whether the park file is this source's PRIMARY path in AWS. When it is, an ABSENT file is a
    # DEGRADED source rather than a quiet fall-through: YouTube transcripts cannot be fetched from a
    # datacenter IP, so live collection still yields items (metadata only) and a wrong S3_PREFIX, a
    # deleted object or a sync that never ran reported OK while every transcript was dropped. Only
    # checked when running in AWS — locally, collecting live IS the path that writes the file.
    park_required: bool = False
    # Share of a source's inputs (feeds / accounts / channels / queries) that may fail before the
    # source is reported DEGRADED. Reporting only: every collected item still reaches the
    # aggregator. One number for every collector — a source that answers from 2 of 40 inputs looks
    # exactly like a healthy one in the item count alone.
    error_rate_threshold: float = Field(default=50.0, ge=0.0, le=100.0)
    # Share of a source's inputs that may come back EMPTY before the source is reported DEGRADED.
    # All-200-with-no-entries (expired RSSHub cookies, a paywalled 200, a playlist that resolves to
    # nothing) is the same disappearance shape as a failure, but it trips no failure rate — and as
    # long as ONE input still produced an item the source reported OK. Source-dependent: many RSS
    # blogs legitimately publish nothing on a given day, whereas 40 X accounts all going quiet is a
    # broken session. None (the default) disables the check; set it per source in config.yaml for the
    # sources whose silence is never legitimate.
    empty_rate_threshold: float | None = Field(default=None, ge=0.0, le=100.0)
    # Fewest inputs a rate check (failed, empty, a YouTube video without a transcript) judges at all;
    # below it the source is not reported DEGRADED on a rate, since one miss out of two is not a rate.
    min_inputs_for_rate: int = Field(default=1, ge=1)


class YouTubeCollectorConfig(BaseCollectorConfig):
    channels: list[str] = Field(default_factory=list)
    max_videos_per_channel: int = Field(default=3, ge=1)
    resolve_timeout: int = Field(default=15, ge=1)
    transcript_timeout: int = Field(default=15, ge=1)
    transcript_language: str = Field(default="en")
    # Percent of kept videos without a transcript past which the source reports DEGRADED: a sync whose
    # transcript fetches are all blocked still returns every video, with its description only.
    transcript_missing_threshold: float | None = Field(default=50.0, ge=0.0, le=100.0)
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
    # How many feeds may be fetched at once. Worst-case wall time is
    # ceil(feeds / max_concurrency) * (request_timeout + retry_backoff_sec) * max_retries.
    max_concurrency: int = Field(default=5, ge=1)


ResearchCategory = Literal["company", "industry", "invest", "economy", "market", "debenture"]
_DEFAULT_RESEARCH_CATEGORIES: list[ResearchCategory] = ["company", "industry", "invest", "economy"]


class NaverResearchCollectorConfig(BaseCollectorConfig):
    """Korean broker research listed by Naver 증권 (stock.naver.com/research)."""

    enabled: bool = False
    # company, industry, invest (strategy), economy, market (daily wraps), debenture (fixed income).
    categories: list[ResearchCategory] = Field(default_factory=lambda: list(_DEFAULT_RESEARCH_CATEGORIES))
    # Pages of 50 per category, newest first; a peak earnings day had 121 company notes in one day. A
    # category still paging at the cap reports the source DEGRADED.
    max_pages: int = Field(default=10, ge=1)
    page_delay_sec: float = Field(default=1.0, ge=0.0)


class TrendSearch(_StrictModel):
    name: str
    queries: list[str]
    domains: list[str] = Field(default_factory=list)


class WebSearchCollectorConfig(BaseCollectorConfig):
    trend_searches: list[TrendSearch] = Field(default_factory=list)
    max_results_per_query: int = Field(default=10, ge=1)
    refine_model: LanguageModelId = LanguageModelId.CLAUDE_V5_5_SONNET
    max_refine_queries: int = Field(default=3, ge=1)
    # The field the refine step asks for queries about (the edition's pipeline.domain), and its rules.
    domain: str = "AI/ML"
    refine_rules: str = (
        "- Prefer: technical blog posts, paper analyses, benchmarks, tool releases, practitioner discussions\n"
        '- No generic terms ("AI news", "machine learning update")'
    )
    # Hosts never returned by any search, broad or refined (Tavily `exclude_domains`): content farms
    # that rewrite wire stories under SEO titles and kept winning picks over the original report.
    exclude_domains: list[str] = Field(default_factory=list)
    # How many search queries may be in flight against Tavily at once. Unbounded fan-out threw every
    # configured query at the API simultaneously, so a large trend list self-throttled and lost whole
    # queries. Same bound as rss.max_concurrency / rsshub.max_concurrency.
    max_concurrency: int = Field(default=5, ge=1)


class RSSHubCollectorConfig(BaseCollectorConfig):
    base_url: str = f"http://localhost:{RSSHUB_PORT}"
    # X usernames (X only: routes for other platforms were never verified).
    accounts: list[str] = Field(default_factory=list)
    # How many account feeds the local sync fetches at once. Worst-case wall time is
    # ceil(accounts / max_concurrency) * (request_timeout + retry_backoff_sec) * max_retries.
    max_concurrency: int = Field(default=5, ge=1)


class CollectorsConfig(_StrictModel):
    youtube: YouTubeCollectorConfig = Field(default_factory=YouTubeCollectorConfig)
    reddit: RedditCollectorConfig = Field(default_factory=RedditCollectorConfig)
    rss: RSSCollectorConfig = Field(default_factory=RSSCollectorConfig)
    web_search: WebSearchCollectorConfig = Field(default_factory=WebSearchCollectorConfig)
    rsshub: RSSHubCollectorConfig = Field(default_factory=RSSHubCollectorConfig)
    naver_research: NaverResearchCollectorConfig = Field(default_factory=NaverResearchCollectorConfig)
    # Sources whose EMPTY result is an INCIDENT worth an alert, by collector name (e.g.
    # ["rss", "web_search"]). A dark source produces no items, no exception and no stale park file,
    # so nothing else notices it; but reddit/x are legitimately quiet on many days, which is why
    # this is an explicit opt-in list rather than "alert whenever any source is empty" — that would
    # page daily and be muted within a week. Empty (the default) never alerts on EMPTY.
    alert_on_empty: list[str] = Field(default_factory=list)
    # Wall-clock limit on ONE collector's collect() in a digest run. A source past it is cancelled and
    # reported FAILED, the others' items are kept: the per-request timeouts bound each read, not a
    # whole fan-out, and the digest Lambda still has to rank and write after collection.
    collection_deadline_sec: int = Field(default=300, ge=1)

    @model_validator(mode="after")
    def _alert_on_empty_names_known_collectors(self) -> "CollectorsConfig":
        """A name here that is not a collector alerts on NOTHING: the list is matched against the
        health report's source names, so `websearch` (or a renamed source) leaves the very source it
        was meant to watch silently dark. Fail at config load instead."""
        unknown = sorted(set(self.alert_on_empty) - set(COLLECTOR_NAMES))
        if unknown:
            raise ValueError(
                f"collectors.alert_on_empty names unknown collector(s) {unknown}; "
                f"valid names are {sorted(COLLECTOR_NAMES)}"
            )
        return self

    def set_reference_time(self, reference_time: datetime) -> None:
        for name in COLLECTOR_NAMES:
            getattr(self, name).reference_time = reference_time


def _utc_offset_hours(timezone: str) -> int:
    """Whole-hour UTC offset of `timezone` right now, for turning the UTC cron hour into a local one.

    Read from the CURRENT moment rather than a fixed anchor date so a DST-observing deployment is
    judged against the offset it is actually running under."""
    offset = datetime.now(ZoneInfo(timezone)).utcoffset()
    return round(offset.total_seconds() / 3600) if offset else 0


# gpt-image sizes are "<width>x<height>"; anything else is rejected at config load rather than
# surfacing as an OpenAI 400 in the visual Lambda, hours later and only on the day it renders.
_IMAGE_SIZE_RE = re.compile(r"\d+x\d+")


class PipelineConfig(_StrictModel):
    top_n: int = Field(default=5, ge=1)
    min_score: float = Field(default=0.6, ge=0.0, le=1.0)
    # Extra ranked candidates handed to the digest editor beyond top_n: it picks the strongest top_n
    # distinct stories from the pool, so the buffer is both its choice and its room to merge
    # same-event items. 0 hands over exactly top_n. At 3, same-event copies from different publishers
    # (one launch posted by the lab, a newsletter and two X accounts) left exactly top_n distinct
    # stories in the pool on 2 of 3 runs, so the editor had nothing to choose from.
    digest_candidate_buffer: int = Field(default=7, ge=0)
    # Days a published URL stays in the cross-day dedup ledger; an article seen within this
    # window is skipped so the digest doesn't re-summarize the same story days apart. At least 2:
    # a date-only web result falls inside two consecutive windows, and the ledger is what stops it
    # from shipping twice.
    published_url_ttl_days: int = Field(default=6, ge=2)
    # How many recent digest leads (and daily comics) to feed back into the prompts as "don't reuse these".
    recent_leads_window: int = Field(default=5, ge=0)
    # Days of recent story titles the ranker and the editor see: a story returns under another outlet's URL
    # more than a week later, past the URL ledger.
    recent_title_days: int = Field(default=14, ge=0)
    # Days of recent implications fed back with the leads as angles not to repeat.
    recent_implication_days: int = Field(default=2, ge=0)
    # The other editions posting to the same Threads account ("" is the AI/ML edition). Their last day or
    # two of story titles, leads and implications join this edition's own, so one account does not run
    # the same story and thesis twice in a day.
    sibling_editions: list[str] = Field(default_factory=list)
    ranking_model: LanguageModelId = LanguageModelId.CLAUDE_V5_5_SONNET
    digest_model: LanguageModelId = LanguageModelId.CLAUDE_V5_5_SONNET
    # A digest whose JSON the editor emits malformed is RE-ASKED before it is given up on: regeneration
    # is one Bedrock call, the alternative a day with no stories at all.
    digest_max_retries: int = Field(default=3, ge=1)
    digest_retry_backoff_sec: float = Field(default=5.0, ge=0)
    # Post-generation faithfulness pass: verify the digest's specific claims against the
    # source items and surgically revise unsupported ones (prompt rules alone couldn't
    # move the faithfulness score). Best-effort; disable to skip the extra LLM call.
    enable_grounding_check: bool = True
    # Korean editorial rules + translation glossary, injected into the digest prompt's *Language*
    # block so the glossary can be tuned without editing the prompt. NOT a language switch: the
    # digest prompt states the Korean requirement itself, so overriding this alone cannot move the digest to another language.
    # Korean-only is the intended product.
    digest_language_rules: str = (
        "- Write in Korean (95%+); English ONLY for proper nouns and untranslatable technical terms. "
        "Use ONE form per proper noun: company, product, model and person names keep their official "
        "Latin-script form (Anthropic, OpenAI, SoftBank, Simon Willison) even when the source "
        "transliterates them; a term not in the glossary below stays in Latin script. Make the particle after it agree with that "
        "written form (OpenAI는, GPT-5가).\n"
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
    # The field this edition covers, as the prompts name it ("AI/ML", "investing (...)").
    domain: str = "AI/ML"
    # Rules this edition adds to the digest editor's voice and to the research prompt (the investing
    # edition: no buy/sell calls, no targets of its own). Empty for the AI/ML edition, whose prompts
    # are then unchanged.
    domain_rules: str = ""
    # Closes every Threads root and research thread this edition posts (the investing disclaimer).
    # Code-owned, never trimmed: the lead budget and the research post budget make room for it.
    threads_footer: str = ""
    ranking_audience_description: str = "a daily digest aimed at practicing ML engineers"
    digest_audience_description: str = "Korean readers who follow AI news, engineers and curious non-specialists alike"
    # Editorial voice for the digest lead/implications: a recurring narrator persona (a singularity-
    # believing, science-geek/sci-fi technologist) written in a dry Gruber register — opinionated,
    # evidence-obsessed, watching who captures AI's upside; critiques ideas/decisions, never a
    # person. Configurable so the tone can be retuned without forking the prompt.
    digest_voice_guidance: str = (
        "WHO YOU ARE — a character with opinions. A technologist sure that the singularity and AGI "
        "are coming soon. A lifelong science geek and sci-fi "
        "reader who grew up imagining this future, now scanning the daily news for real evidence "
        "it's arriving. But an engineer at heart: allergic to noise, and asking who this "
        "abundance flows to — so you cheer "
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
        "CRAFT — Spice is in the FRAMING, never the facts: sharpen the angle and contrast, but never "
        "state an inference as a reported fact. Critique ideas, decisions, "
        "incentives — never a person. Plain words over "
        "jargon and short sentences over dense ones, but never drop a rung of the argument to sound "
        "simple — 'easy' is the wording, not a missing step. On a pure-tech story keep the "
        "power/distribution lens light or absent. Concentrate the persona in the LEAD (the standalone "
        "hook); keep item bodies cleaner and let the edge land in each item's closing line. Plain "
        "declarative '~다' register. Ground every take in the supplied facts and trends, never vibes."
    )
    # Tongue-in-cheek countdown intro prepended to the digest lead (code computes the day count from
    # countdown_date — never the LLM — so it stays accurate and ticks down daily). The AI default quotes
    # Dario Amodei, Paris AI Action Summit statement (anthropic.com, 2025-02): "possibly by 2026 or 2027
    # (and almost certainly no later than 2030)"; counted to the end of 2027. An edition sets its own
    # date and wording in the same "...까지 N일 남았다" shape; the dates follow the quoted predictions, the text names no one. Empty date disables the intro.
    countdown_date: str = "2027-12-31"
    # Plain declarative ('~다') to match the columnist body voice — not honorific ('~입니다').
    # Before the D-day it counts down; on/after it, countdown_after counts up (the prediction
    # blew past — a self-aware joke). Empty countdown_date disables the intro entirely.
    countdown_template: str = "AGI 등장까지 {days}일 남았다. "
    # The gag OPENS the lead, which spends the root's first line (the one most feed readers see) on
    # the same sentence every day. That is deliberate: the countdown is the account's signature, and
    # the owner ranked recognisable branding above the first-line reach argument.
    countdown_after: str = "AGI 등장 예정일 D+{days}일째, 아직이다. "
    item_text_max_tokens: int = Field(default=10000, ge=1)
    # Each item's text as the RANKER sees it, cut by characters (no CountTokens calls); unset is
    # item_text_max_tokens. Scoring needs the gist, the editor and the grounding check the full text: the
    # investing pool (426 items, many with full article pages or transcripts) sent the ranker 1.23M input
    # tokens a run, eleven times the AI/ML edition's.
    ranking_text_max_chars: int | None = Field(default=None, ge=1)
    ranking_batch_size: int = Field(default=40, ge=1)
    # How many ranking batches may be in flight against Bedrock at once. Unbounded fan-out threw
    # every batch at Converse simultaneously, so a large day self-throttled (ThrottlingException)
    # and dropped whole batches of candidates.
    ranking_max_concurrency: int = Field(default=4, ge=1)
    # A batch that fails is RETRIED before it is given up on: a single throttle used to silently
    # delete ~40 candidates from the day's pool with only a warning.
    ranking_max_retries: int = Field(default=3, ge=1)
    ranking_retry_backoff_sec: float = Field(default=5.0, ge=0)
    # Per-source seats in the candidate pool (top_n + digest_candidate_buffer), so every source's best
    # item above min_score reaches the editor; what ships is the editor's pick of the strongest.
    source_slots: dict[str, int] = Field(
        default_factory=lambda: {
            "web": 1,
            "x": 1,
            "rss": 1,
            "reddit": 1,
            "youtube": 1,
        }
    )
    max_per_origin: int = Field(default=1, ge=1)
    # An edition's extra sentence on how the editor picks its stories as a set (empty: the prompt is
    # unchanged). Only the investing edition sets one; it lost on AI days (docs/decisions.md).
    selection_guidance: str = ""
    # The editor must carry this many stories from these source types, the best-scored ones in the pool:
    # an analyst's or investor's own note or post over the press report of it. Research notes among them
    # get their full text (the note's PDF): the quota alone tied, with the text it won (docs/decisions.md).
    primary_quota: int = Field(default=0, ge=0)
    primary_source_types: list[SourceType] = Field(default_factory=list)
    # Score-calibration buckets the ranking prompt applies, injected as template text so
    # ops can retune the distribution without editing the prompt.
    # The ranker's lenses, one per line; the investing edition swaps in its own.
    ranking_criteria: str = (
        "1. Technical Substance — real depth in content body, not headline. Thin content scores low.\n"
        "2. Practitioner Value — would this change how an engineer works?\n"
        "3. Novelty — first reports, original analysis, new releases over rehashes. A story already in a recent "
        "digest (listed below) is a rehash unless the item reports a real new development on it.\n"
        "4. Industry Impact — competitive landscape shifts, funding, policy.\n"
        "5. Research Significance — notable papers with novel contributions.\n"
        "6. Source Authority — recognized researchers/institutions get a small bonus."
    )
    ranking_scoring_rubric: str = (
        "0.9+: field-defining. 0.8-0.89: very important. 0.7-0.79: notable. "
        "0.6-0.69: worth noting (digest bar). <0.6: low value."
    )
    trend_model: LanguageModelId = LanguageModelId.CLAUDE_V5_5_SONNET
    trend_retention_days: int = Field(default=30, ge=1)
    trend_max_evidence: int = Field(default=5, ge=1)
    trend_max_active_trends: int = Field(default=10, ge=1)
    # Whether the editor's trend context carries each trend's latest note or only its title. The AI/ML
    # edition shows titles only: given the notes, its leads re-told yesterday's figures as the "wider arc".
    # Investing keeps the notes, whose market figures its editor uses (docs/decisions.md).
    trend_context_notes: bool = True
    trend_momentum_half_life_days: float = Field(default=7.0, gt=0)
    # The digest's delivery channel. Off until the Threads access token / user id are provisioned in SSM.
    enable_threads_post: bool = False
    # The ONE topic tag a Threads post may carry (Meta: 1-50 chars, no "." or "&"); it puts the root
    # post into that topic's feed and tag search. Replies carry none. Empty disables it.
    threads_topic_tag: str = Field(default="AI", pattern=r"^[^.&]{0,50}$")
    # Reply to readers' comments on the account's recent posts (the comment-reply Lambda, on its own
    # schedule). Only comments that answer one of the account's posts are considered, and the model
    # may decline one (spam, nothing to answer).
    enable_comment_replies: bool = False
    # Posts and comments older than this are left alone.
    comment_reply_lookback_days: int = Field(default=7, ge=1)
    # Bounds a burst (a spam wave, a first run over a backlog); the rest wait for the next run.
    comment_reply_max_per_run: int = Field(default=10, ge=1)
    # The account answers one commenter at most this many times under one post, so a bot that
    # answers every reply cannot hold it in an endless exchange.
    comment_reply_max_per_commenter: int = Field(default=3, ge=1)
    # Seconds one comment's draft may take; with the Threads reply's publish reserve it is the time a
    # comment needs left before it is started.
    comment_reply_draft_budget_sec: int = Field(default=60, ge=0)
    # Sunburst over flare: the same token price, and it draws brand logos correctly (docs/design.md).
    image_model: str = "gpt-image-2.5-sunburst"
    # orientation -> gpt-image size. The art director picks the frame the editor's form needs (a grid
    # is portrait, a stacked strip tall). The KEYS are not free-form: they are the
    # VisualBrief orientation vocabulary, checked below.
    image_sizes: dict[str, str] = Field(
        default_factory=lambda: {
            "tall": "848x2544",
            "portrait": "1024x1536",
            "square": "1024x1024",
            "landscape": "1536x1024",
        }
    )
    # Bounds on the gpt-image HTTP call. The OpenAI SDK defaults (600s timeout x 2 retries) can
    # exceed the visual Lambda's 15-min budget; one 300s attempt leaves room for the single
    # moderation-softened re-render and still finishes inside the Lambda.
    visual_image_timeout_sec: int = Field(default=300, ge=10)
    # gpt-image quality tier, PINNED rather than left to OpenAI. Empty sends nothing, leaving its
    # "auto", which picks between tiers whose prices differ several-fold, so the bill is unpredictable
    # and the render log could only report `quality=auto->unreported`. It also decides how legible
    # the panels' Korean lines come out, which is not a choice to delegate. Typed, so a typo fails at
    # config load instead of surfacing as an OpenAI 400 inside the visual Lambda hours later.
    visual_image_quality: Literal["", "low", "medium", "high", "xhigh", "max", "auto"] = "xhigh"
    visual_image_max_retries: int = Field(default=0, ge=0)
    visual_synopsis_source_max_tokens: int = Field(default=2000, ge=1)
    # Budget for the HEADLINE story's body handed to the visual EDITOR, which writes the comic: given
    # only the title, the decisions that shape the image were made from a title.
    visual_editor_source_max_tokens: int = Field(default=2000, ge=1)
    # Audience/domain the visual prompts target. Configurable so the visual pipeline can
    # be reused across domains without forking the prompts.
    visual_audience_description: str = "a daily AI news digest read by engineers and curious non-specialists alike"
    # The language of every line rendered INSIDE the image.
    visual_on_image_language: str = "short, natural Korean"
    # Style/humor guidance injected into the visual synopsis prompt. Configurable so the
    # visual pipeline's tone can be retuned (or reused for non-AI domains) without forking.
    visual_synopsis_style_guidance: str = (
        "Multi-panel: the same characters, outfits and art style in every panel; each panel "
        "follows from the previous so the sequence reads in order without explanation, and every speech bubble's "
        "tail points at the character who says it."
    )
    # Default aesthetic injected into the image-generation prompt. Configurable so the visual
    # pipeline's look can be retuned (or reused for non-AI domains) without editing the prompt.
    # The physical half is what rendered comics got wrong (a robot seated on a ledge above the desks,
    # a sign hanging in mid-air, a hand that does not hold its prop), so it is stated as a concrete
    # rule set rather than "correct perspective".
    visual_synopsis_style_aesthetic: str = (
        "a clean, modern, polished Korean webtoon style with confident linework, soft cel shading and a "
        "harmonious palette. Physically plausible staging in every panel: one eye level and floor "
        "plane, every person and object resting on a surface or held in a hand that actually grips it, "
        "sizes consistent with distance, no floating signs or props, no impossible stacking, nothing "
        "passing through anything else, natural anatomy with five-fingered hands, and a drawn face with "
        "eyes and an expression on every person, background figures included"
    )
    # Appended to every art-director instruction. Real people may be drawn (editorial-cartoon
    # practice), but a company or country is never an ethnically-coded human. Empty disables it.
    visual_guardrails: str = (
        "Recognisable depictions of real people are fine. Do NOT, however, personify a company or "
        "a country as an ethnically-coded human; use the real people involved, or the company's "
        "logo or product instead."
    )
    # Appended to the instruction when the image model's moderation blocks the first render,
    # to soften tone before a single retry. Configurable so ops can retune the safe-for-work
    # guidance without editing code.
    visual_moderation_softening_instruction: str = (
        "IMPORTANT: keep it clearly safe-for-work and good-natured. "
        "Keep the real people, drawn as friendly caricatures in neutral, everyday actions; "
        "nothing violent, sexual or humiliating, and nothing that could read as defamatory."
    )
    # Recurring mascot = the visual embodiment of the digest's narrator persona (the singularity-
    # believing science-geek technologist). He appears only when the editor sets use_character.
    # Identity rides on a FEW signature cues (hair cowlick, retro glasses, two-tone cardigan),
    # not an exhaustive prop list, so he reads as the same person while pose/outfit/expression vary
    # day to day instead of converging on one identical look.
    visual_character_enabled: bool = True
    # The character reference sheet (front, three-quarter, side and three expressions), passed to the
    # image model with every render that features the narrator so he looks the same every day. A path
    # relative to the project root; the Dockerfile copies assets/.
    visual_character_reference: str = "assets/persona.jpg"
    # Kept deliberately LEAN: an over-specified sheet (fixed outfit down to the socks, a repeated
    # "glasses pushed up on the forehead" gag) made every render converge on the same overfit image.
    # We fix only a small identity core and explicitly invite daily variation; framing favors a
    # waist-up shot because full-body figures are where gpt-image proportions/hands break most.
    visual_character_sheet: str = (
        "A recurring Korean man in his late 20s — a cute, earnest tech nerd, the narrator made "
        "visible. Keep him recognizable by just a FEW identity cues, drawn naturally: messy black "
        "hair with an upward cowlick; chunky rounded retro glasses worn normally ON HIS EYES; and a "
        "two-tone color-blocked knit cardigan (mustard + sage green) over a plain tee. Warm, bright, "
        "charming — never a dark-room hacker. Prefer a waist-up or medium shot so "
        "his face carries the moment; a full figure only when the composition needs it."
    )

    @model_validator(mode="after")
    def _a_primary_quota_names_its_sources(self) -> "PipelineConfig":
        if self.primary_quota and not self.primary_source_types:
            raise ValueError("pipeline.primary_quota needs pipeline.primary_source_types")
        if self.primary_quota >= self.top_n:
            raise ValueError("pipeline.primary_quota must leave the headline's choice open (below top_n)")
        return self

    @model_validator(mode="after")
    def _source_slots_name_real_source_types(self) -> "PipelineConfig":
        """source_slots keys are matched against `item.source_type.value` in the ranker's guaranteed-
        slot pass, so a typo'd or renamed key matches no item and that source's seat silently vanishes.
        Fail at config load."""
        unknown = sorted(set(self.source_slots) - {source.value for source in SourceType})
        if unknown:
            raise ValueError(
                f"pipeline.source_slots names unknown source type(s) {unknown}; "
                f"valid keys are {sorted(source.value for source in SourceType)}"
            )
        return self

    @model_validator(mode="after")
    def _image_sizes_match_the_orientation_vocabulary(self) -> "PipelineConfig":
        """image_sizes is advertised as overridable, but its KEYS are the VisualBrief orientation
        vocabulary: the brief's orientation (a VisualOrientation) is looked up in this dict. A renamed or dropped key used to make every brief either fail
        validation or coerce to the default orientation, with no signal at all."""
        expected = set(VISUAL_ORIENTATIONS)
        actual = set(self.image_sizes)
        if actual != expected:
            raise ValueError(
                "pipeline.image_sizes keys must be exactly the visual orientation vocabulary "
                f"{sorted(expected)} (got {sorted(actual)}); the keys are what "
                "VisualBrief.orientation is validated against, so only the SIZES "
                "are tunable here"
            )
        bad_sizes = sorted(size for size in self.image_sizes.values() if not _IMAGE_SIZE_RE.fullmatch(size))
        if bad_sizes:
            raise ValueError(f"pipeline.image_sizes values must look like '1024x1536' (got {bad_sizes})")
        return self


class AgentConfig(_StrictModel):
    model_id: LanguageModelId = LanguageModelId.CLAUDE_V5_5_OPUS
    # Publishers of analyst and strategy research; non-empty enables the search_reports tool.
    report_search_domains: list[str] = Field(default_factory=list)
    # Enables the market data tools: market_quote (prices for stocks, indices, FX and crypto) and
    # analyst_ratings (consensus and dated rating and target changes).
    market_quotes: bool = False
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
    # Report length the agent follows as guidance (interpolated into its prompt; not a bound).
    research_target_words: int = Field(default=1500, ge=200)
    # HARD per-invocation bounds handed to the Strands agent loop (strands.types.agent.Limits),
    # checked at every turn boundary. Unlike the guidance knobs above these are enforced by the SDK:
    # the agent re-sends the whole conversation each cycle, so without them a
    # loop that never converges is unbounded in both cost and wall time. Generous enough that a
    # normal run (~breadth x iterations tool rounds) never trips one.
    research_max_turns: int = Field(default=40, ge=1)
    research_max_total_tokens: int = Field(default=2_000_000, ge=10_000)
    research_max_output_tokens: int = Field(default=100_000, ge=1_000)
    # Hard cap on the number of Threads posts (root + replies) a research report may become.
    # Code-enforced so a report can't fan out into dozens of public posts. 10 keeps the thread at
    # about half a full report (~8700 chars); at 6 it could only keep facts by cutting explanation.
    research_max_threads_posts: int = Field(default=10, ge=1)
    # Hard cap on a single page's extracted text (read_url tool).
    research_content_cap_chars: int = Field(default=50000, ge=1000)
    # OG-image attachment (deep-research delivery only).
    og_image_timeout_sec: int = Field(default=10, ge=1)
    og_image_max_bytes: int = Field(default=8_000_000, ge=10_000)
    # Cap how many images one research run may stage, bounding per-invocation memory.
    research_max_staged_images: int = Field(default=4, ge=1)


class AWSConfig(_StrictModel):
    # The templates' region: every resource (state, memory, Lambdas) lives in one region, and the old
    # us-east-1 default sent a config without the key to look for the memory and the state there.
    region: str = "ap-northeast-2"
    bedrock_region: str = "us-west-2"
    profile: str = ""
    project_name: str = "omnisummary"
    stage: str = "dev"
    timezone: str = "Asia/Seoul"
    state_bucket_name: str = ""
    s3_prefix: str = ""
    # EventBridge cron is UTC. 10:00 UTC = 19:00 KST (daily 7pm).
    digest_cron_hour: str = "10"
    digest_cron_minute: str = "0"
    # Threads long-lived tokens expire after 60 days and can be refreshed once a day old. Weekly, so a
    # failed refresh is retried well inside the window: at 50 days one failure left the token to die
    # before the next attempt, and the rate counts from the rule's creation, not the token's issue.
    threads_token_refresh_days: int = Field(default=7, ge=1, le=59)
    # How often the comment-reply Lambda looks for new comments. At least its 10-minute timeout, so
    # runs never queue behind the single concurrency slot.
    comment_reply_interval_minutes: int = Field(default=60, ge=10)
    # Monthly spend budget on everything tagged Project=<project_name> (0: none). The foundation stack
    # emails the alert address at 80% actual and 100% forecast, and on a daily cost anomaly worth a
    # day of this budget.
    monthly_budget_usd: float = Field(default=0.0, ge=0.0)
    # The IAM user the local X/YouTube sync runs as (its keys are the S3_SYNC_* in .env); the stack grants it
    # PutObject on the state prefix. Managed by hand, its policy pointed at the old bucket after the move.
    s3_sync_user: str = ""


EDITION_ENV = "OMNISUMMARY_EDITION"


class Config(_StrictModel):
    # Which edition this config is ("" is the AI/ML digest). One codebase runs several editions, each
    # from its own config/<edition>.yaml with its own state prefix and memory sessions.
    edition: str = Field(default="", pattern=r"^([a-z][a-z0-9]*)?$")
    collectors: CollectorsConfig = Field(default_factory=CollectorsConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    aws: AWSConfig = Field(default_factory=AWSConfig)

    @model_validator(mode="after")
    def _lookback_reaches_the_previous_run(self) -> "Config":
        """Every source's lookback_hours must reach back to the PREVIOUS run, or a slice of every
        single day appears in no digest ever.

        The cutoff is anchored at midnight at the END of the digest date (`resolve_digest_window`),
        but the run happens hours BEFORE that midnight. With `lookback_hours: 24` on all six sources
        (COLLECTOR_NAMES) against a 19:00 KST cron, everything published 19:00-24:00 KST was still
        unpublished when that day's run collected and already before the next run's cutoff. The
        required minimum is therefore `48 - <run hour, local>`: the span from the previous run's
        clock time to this run's reference time.

        Widening is safe rather than duplicative because cross-day repeats are excluded by identity,
        not by the window (`PublishedUrlLedger` plus recent AgentCore snapshots), and it restores the
        park files' 'stale beats empty' contract: with a 24h window a STALE park file's dated items
        were all older than the cutoff, so the source reported STALE carrying zero items.

        Skipped when the cron hour is not a single integer: a multi-run expression has no one gap."""
        if not self.aws.digest_cron_hour.isdigit():
            return self
        run_hour_local = (int(self.aws.digest_cron_hour) + _utc_offset_hours(self.aws.timezone)) % 24
        minimum = 48 - run_hour_local
        short = {
            name: hours
            for name in sorted(COLLECTOR_NAMES)
            if (hours := getattr(self.collectors, name).lookback_hours) < minimum
        }
        if short:
            raise ValueError(
                f"collectors {short} look back less than the {minimum}h between the "
                f"{run_hour_local:02d}:00 {self.aws.timezone} run and the previous one; items published "
                "after a run and before the next cutoff would appear in no digest at all"
            )
        return self

    @classmethod
    def from_yaml(cls, file_path: str) -> "Config":
        with open(file_path, encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data if config_data else {})

    @classmethod
    def load(cls, edition: str | None = None) -> "Config":
        """The deployed config: `config/config.yaml` (`config/<edition>.yaml` for an edition, by default
        the one OMNISUMMARY_EDITION names), falling back to the tracked template.

        Returning bare code defaults for a missing file was silent and wrong. config.yaml is
        gitignored, the Dockerfile COPYs config/, so a clean-checkout image — which is exactly what
        CI builds and validates — resolved to empty rss/reddit/youtube lists and region us-east-1 and
        started up perfectly happily. The first sign was a digest with no stories.

        The template is tracked and always present, so it is a real config rather than nothing: the
        source lists are placeholders, but every knob is the shipped one and the WARNING names what
        happened. With neither file there is nothing to deploy, and load() says so. Inside AWS the fallback
        is refused: an image built from a checkout without config.yaml would otherwise collect from the
        placeholders, rank and publish a thin digest with only a log line to show for it."""
        load_dotenv()
        config_dir = Path(__file__).parent.parent / "config"
        edition = (os.environ.get(EDITION_ENV, "") if edition is None else edition).strip()
        stem = edition or "config"
        config_path = config_dir / f"{stem}.yaml"
        if config_path.exists():
            config = cls.from_yaml(str(config_path))
            if config.edition != edition:
                raise ValueError(f"'{config_path}' declares edition '{config.edition}', but '{edition}' was asked for")
            return config
        if is_running_in_aws():
            raise FileNotFoundError(
                f"'{config_path}' is missing from the deployed image; rebuild it from a checkout that has it."
            )
        template_path = config_dir / f"{stem}-template.yaml"
        if not template_path.exists():
            raise FileNotFoundError(
                f"No config found: neither '{config_path}' nor '{template_path}' exists. "
                f"Copy config/{stem}-template.yaml to config/{stem}.yaml and fill it in."
            )
        logger.warning(
            "'%s' is missing; falling back to the tracked template '%s'. Its source lists are "
            "PLACEHOLDERS, so this config collects nothing.",
            config_path,
            template_path,
        )
        return cls.from_yaml(str(template_path))


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Process-wide cached Config for READ-ONLY leaf callers (delivery, research backends, agent
    tools, og-image). Config.load() re-reads and re-validates the whole YAML on every call, which
    a single research run did dozens of times — once per tool invocation.

    Deliberately NOT used where the Config is MUTATED (main/handlers' set_reference_time, the infra
    tests' bucket override): those callers need their own instance, and
    a shared cached object would leak their edits everywhere. Tests clear the cache via an autouse
    fixture in tests/conftest.py."""
    return Config.load()
