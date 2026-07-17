from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from .constants import RSSHUB_PORT, LanguageModelId


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
    "- Do NOT end a sentence with a colon to introduce an enumeration ('핵심은 세 가지다: ...'); "
    "Korean prose does not terminate sentences this way. Either finish the sentence and start a "
    "new one ('핵심은 세 가지다. 첫째는 ...'), or fold the items into a single flowing sentence "
    "(list them with '·' or commas inside one clause that ends in a normal verb). "
    "Keep colons out of mid-prose entirely."
)


class BaseCollectorConfig(_StrictModel):
    enabled: bool = True
    lookback_hours: int = 24
    reference_time: datetime | None = None
    request_timeout: int = Field(default=30, ge=1)
    max_retries: int = Field(default=3, ge=1)
    retry_backoff_sec: int = Field(default=5, ge=0)


class YouTubeCollectorConfig(BaseCollectorConfig):
    channels: list[str] = Field(default_factory=list)
    max_videos_per_channel: int = Field(default=3, ge=1)
    resolve_timeout: int = Field(default=15, ge=1)
    transcript_timeout: int = Field(default=15, ge=1)
    transcript_language: str = Field(default="en")


class RedditCollectorConfig(BaseCollectorConfig):
    subreddits: list[str] = Field(default_factory=lambda: ["MachineLearning", "artificial", "LocalLLaMA"])
    sort: Literal["hot", "top", "new"] = "hot"
    limit: int = Field(default=20, ge=1)


class RSSCollectorConfig(BaseCollectorConfig):
    feeds: list[str] = Field(default_factory=list)


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


class RSSHubAccount(_StrictModel):
    username: str
    platform: str


class RSSHubCollectorConfig(BaseCollectorConfig):
    base_url: str = f"http://localhost:{RSSHUB_PORT}"
    accounts: list[RSSHubAccount] = Field(default_factory=list)
    lookback_hours: int = 72
    error_rate_threshold: float = Field(default=50.0, ge=0.0, le=100.0)


class CollectorsConfig(_StrictModel):
    youtube: YouTubeCollectorConfig = Field(default_factory=YouTubeCollectorConfig)
    reddit: RedditCollectorConfig = Field(default_factory=RedditCollectorConfig)
    rss: RSSCollectorConfig = Field(default_factory=RSSCollectorConfig)
    web_search: WebSearchCollectorConfig = Field(default_factory=WebSearchCollectorConfig)
    rsshub: RSSHubCollectorConfig = Field(default_factory=RSSHubCollectorConfig)

    def set_reference_time(self, reference_time: datetime) -> None:
        for cfg in (self.youtube, self.reddit, self.rss, self.web_search, self.rsshub):
            cfg.reference_time = reference_time


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
    # Post-generation faithfulness pass: verify the digest's specific claims against the
    # source items and surgically revise unsupported ones (prompt rules alone couldn't
    # move the faithfulness score). Best-effort; disable to skip the extra LLM call.
    enable_grounding_check: bool = True
    # Language rules injected into the digest prompt's *Language* block. Defaults to the
    # Korean editorial rules + translation glossary; other deployments can override to
    # write the digest in another language without forking the prompt.
    digest_language_rules: str = (
        "- Write in Korean (95%+). English ONLY for proper nouns and untranslatable technical terms.\n"
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
        "'보통 이런 구도는'). Critique ideas, decisions, incentives — never a person. Plain words over "
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
    item_text_max_tokens: int = Field(default=8000, ge=1)
    ranking_batch_size: int = Field(default=40, ge=1)
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
    # visual (wide strip / tall infographic / square meme); not locked to one aspect ratio.
    image_sizes: dict[str, str] = Field(
        default_factory=lambda: {
            "square": "1024x1024",
            "landscape": "1536x1024",
            "portrait": "1024x1536",
        }
    )
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
    # lock. Identity rides on signature props (glasses, two-tone cardigan, prediction notebook), not
    # exact facial pixels, so it stays recognizable across the daily-varying art styles.
    visual_character_enabled: bool = True
    # Roughly what share of visuals should feature the character; the editor still skips it when a
    # story (e.g. a pure architecture explainer) reads better as a concept visual. 0 disables.
    visual_character_target_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    visual_character_sheet: str = (
        "A recurring Korean man in his late 20s — a cute, earnest tech nerd, the narrator made "
        "visible. Keep him recognizable in ANY art style via these signatures: messy black hair "
        "with a stubborn upward cowlick; chunky rounded retro glasses (often pushed up on the "
        "forehead); an oversized two-tone color-blocked chunky knit cardigan in mustard and sage "
        "green over a cream tee; loose brown corduroy pants; mismatched cozy socks; and a battered "
        "notebook of scribbled predictions with exactly one circled in red. Warm, bright, charming "
        "— never a dark-room hacker. He reacts to today's story (wonder, a skeptical squint, "
        "fired-up conviction) as a witness inside the scene, not as a replacement for depicting it. "
        "RENDER HIM WELL: correct, natural human anatomy and proportions — a normal head-to-body "
        "ratio, properly shaped hands with five fingers, arms and legs at believable lengths and "
        "angles, a balanced stance with weight that reads as physically real. No distorted limbs, "
        "no oversized head, no melted or extra fingers, no broken perspective. He should look like a "
        "polished, professionally-drawn character — appealing and anatomically sound — in whatever "
        "art style the day calls for."
    )


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
