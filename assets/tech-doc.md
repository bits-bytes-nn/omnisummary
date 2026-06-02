# OmniSummary — Technical Documentation

> Single source of detailed, line-by-line technical reference for OmniSummary.
> Higher-level orientation lives in `README.md` and `.claude/CLAUDE.md`; this document is the deep reference.

## 1. Overview

OmniSummary is a proactive AI/ML daily-digest system. On a daily schedule it collects content from five
source families, aggregates and de-duplicates it, ranks it with an LLM, generates a Korean editorial digest,
delivers it to Slack, and persists state to **Bedrock AgentCore Memory**. A follow-up agent (Strands on
Bedrock AgentCore Runtime) answers questions about digest items and can produce visualizations (comics,
diagrams). Operational health is reported per-source and alerted via SNS email.

```
[EventBridge cron] → [Digest Lambda (Docker)]
   → Collectors (RSS, Reddit, YouTube, WebSearch, X via RSSHub/S3)
   → Aggregator (URL + title dedup)
   → Ranker (Bedrock Claude Opus 4.8, source-slot + per-origin diversity)
   → TrendTracker (trends.md via StateStore)
   → DigestGenerator (Bedrock Claude Sonnet 4.6, Korean Slack mrkdwn)
   → Slack delivery
   → AgentCore Memory (digest snapshot + trend facts)
   → SNS alert if any source FAILED

[Slack mention] → [API Gateway + WAF] → [Slack Lambda]
   → async self-invoke → [Bedrock AgentCore Runtime: Strands agent]
   → tools: get_detail, search_papers, search_community, search_related_news, make_visual
   → reads digest state from AgentCore Memory; posts replies/images to Slack
```

## 2. Repository layout

| Path | Responsibility |
|------|----------------|
| `collectors/` | `BaseCollector` ABC + RSS, Reddit (OAuth), RSSHub (X/Twitter), YouTube, WebSearch (Tavily) |
| `pipeline/` | `ContentAggregator`, `ContentRanker`, `DigestGenerator`, `TrendTracker` |
| `agent/` | Strands agent, tools, `DigestStateManager`, `VisualGenerator` (comic/diagram) |
| `agent_runtime/` | Bedrock AgentCore HTTP server (`BedrockAgentCoreApp`) |
| `shared/` | config, models, constants, utils (Bedrock factory), logger, prompts, state_store, **memory**, proxy |
| `output/` | Slack delivery (text + image upload) |
| `lambda_handlers/` | digest handler, Slack-events handler |
| `infrastructure/` | CDK `foundation_stack` + `application_stack` |
| `scripts/` | `deploy.py`, `ci_synth.py`, `sync_rsshub_to_s3.py` |

## 3. Configuration

`config/config.yaml` → Pydantic models in `shared/config.py` via `Config.load()`. Secrets come from `.env`
(local) or SSM Parameter Store under `/{project}/{stage}/{name}` (AWS).

Key config groups:
- `collectors.*` — each extends `BaseCollectorConfig` (`enabled`, `lookback_hours`, `reference_time`,
  `request_timeout`, `max_retries`, `retry_backoff_sec`).
- `pipeline` — `top_n`, `min_score`, `ranking_model` (Opus 4.8), `digest_model` (Sonnet 4.6),
  `source_slots`, `source_cap_multiplier`, **`max_per_origin`** (per-channel/author/subreddit cap),
  `origin_weights`, `trend_retention_days`.
- `aws` — region, profile, project/stage, `digest_cron_hour/minute`, `api_throttle_*`, `waf_rate_limit`.

Environment variables (`.env`): `SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN`, `SLACK_CHANNEL_ID`, `TAVILY_API_KEY`,
`YOUTUBE_API_KEY`, `REDDIT_CLIENT_ID`/`REDDIT_CLIENT_SECRET`, `OPENAI_API_KEY`, `ALERT_EMAIL`,
`CLOUDFLARE_PROXY_URL`/`CLOUDFLARE_PROXY_TOKEN`. In AWS, `MEMORY_ID`, `ALERT_SNS_TOPIC_ARN`,
`STATE_BUCKET`, `RSSHUB_BASE_URL`, `PROJECT_NAME`, `STAGE` are injected by CDK.

## 4. Collectors

All collectors implement `BaseCollector.collect() -> list[CollectedItem]` and filter by
`cutoff_datetime(lookback_hours, reference_time)` (`collectors/base.py`).

- **RSS** (`rss.py`): feedparser over `config.collectors.rss.feeds`; metadata `feed_url`, `feed_title`.
- **Reddit** (`reddit.py`): **official OAuth API** (public `.json` is IP-blocked). `_resolve_reddit_credentials()`
  reads env then SSM (`reddit-client-id/-secret`); `_fetch_token()` does client-credentials grant; per-subreddit
  GET to `oauth.reddit.com/r/{sub}/{sort}`. Preserves `score`/`num_comments` (ranking engagement signal).
  Missing creds → `[]` (graceful skip, classified EMPTY not FAILED).
- **RSSHub** (`rsshub.py`): X/Twitter feeds via a local/containerized RSSHub; can also load a pre-synced
  snapshot from S3 (`rsshub_items.json`). Self-tracks failed/empty accounts and an `error_rate_threshold`.
- **YouTube** (`youtube.py`): YouTube Data API when `YOUTUBE_API_KEY` set, else RSS fallback via proxy.
  `max_videos_per_channel=1` so a high-volume channel can't flood the candidate pool.
- **WebSearch** (`web_search.py`): Tavily search with LLM query refinement (`RefineQueryPrompt`).

`gather_collector_results()` runs collectors concurrently and swallows per-task exceptions (logs only,
returns flat list). For health reporting, `main.run_collectors_with_health()` runs the same tasks but
returns a `HealthReport` (see §8) — `gather_collector_results` is left unchanged for its other callers.

## 5. Pipeline

1. **Aggregator** (`aggregator.py`): dedup by URL then normalized title; merges metadata of duplicates.
2. **Ranker** (`ranker.py`): formats items (with engagement + origin), calls `RankingPrompt` on Claude Opus
   4.8, parses JSON scores, applies `origin_weights`, filters by `min_score`, then `_apply_source_slots`:
   - fills each source's base slot from `source_slots`,
   - overflow fill up to `source_cap_multiplier × slot`,
   - **`max_per_origin`** caps how many items share one origin key (channel/author/subreddit) — the durable
     fix for single-channel monopoly. Origin resolved by `_resolve_origin_key` (YouTube→channel_url,
     Reddit→subreddit, RSS→feed_url, X→author).
3. **TrendTracker** (`trend_tracker.py`): maintains `trends.md` via a `StateStore`; merges archived history.
4. **DigestGenerator** (`digest_generator.py`): `DigestPrompt` on Claude Sonnet 4.6 → Korean Slack mrkdwn;
   `sanitize_slack_mrkdwn` normalizes output.

## 6. LLM factory (`shared/utils.py`)

`BedrockLanguageModelFactory.get_model(model_id, **kwargs)` returns a `ChatBedrock`/`ChatBedrockConverse`
configured for the model's capabilities (`_LANGUAGE_MODEL_INFO`): thinking, 1M context, performance latency,
prompt caching. `BedrockCrossRegionModelHelper` resolves `global.`/`apac.` inference-profile IDs when
available. Model IDs are enumerated in `shared/constants.py` (`LanguageModelId`), latest = Opus 4.8 / Sonnet 4.6.

**Prompt caching.** Bedrock prompt caching has a ~1024-token minimum cacheable prefix for Claude. It is
applied where it pays off: the follow-up **agent**, whose ~1.7K-token system prompt + tool schemas are
re-sent on every ReAct step and across multi-turn sessions, uses Strands `BedrockModel(cache_config=
CacheConfig(strategy="auto"))` (`agent/agent.py`) to cache that prefix (verified: `cacheWriteInputTokens`
on first call, `cacheReadInputTokens` thereafter). The one-shot pipeline prompts (ranker/digest/trend/visual
synopsis, all ≤~530 tokens and invoked once per run) are below the cache minimum and have no cross-call
reuse, so caching is intentionally not applied there.

## 7. Memory (AgentCore-centric)

`shared/memory.py` defines `MemoryStore` ABC with two implementations:
- **`AgentCoreMemoryStore`** (system of record in AWS): digest snapshots are written as short-term session
  events (`create_event`, session `digest-<date>`); `get_latest_digest()` lists sessions and reads the newest
  digest session's event. Trend summaries are written as events that feed the **semantic** long-term strategy;
  `recall(query)` does `retrieve_memory_records` over namespace `/facts/{actor}/` and is exposed to the
  follow-up agent through the `recall_trends` tool (cross-day memory).
- **`LocalMemoryStore`**: filesystem fallback for offline dev (`digest_*.json`, `trends.jsonl`).

`create_memory_store()` picks AgentCore when `MEMORY_ID` is set, else local. The digest Lambda writes the
snapshot + trend after each run; the AgentCore runtime loads the latest snapshot into `DigestStateManager`
on each invocation and the agent can `recall_trends` to retrieve prior-day context.

Note: the human-readable `trends.md` document (cross-day narrative used to seed digest generation) is a
separate, intentional artifact still persisted via `StateStore` (`TrendTracker`); AgentCore Memory holds the
machine-recallable trend facts used by the agent. The two are complementary, not duplicates.

The Memory resource itself (`AWS::BedrockAgentCore::Memory`) is created in `foundation_stack` with a
semantic strategy and a dedicated `MemoryExecutionRole` (for the extraction model). Cost: long-term
extraction invokes a Bedrock model per event asynchronously; short-term events expire after
`event_expiry_duration` (90 days). For a once-daily digest the event volume is tiny.

## 8. Health check & alerting

`shared/models.py`: `SourceStatus` (`ok`/`empty`/`failed`), `SourceHealth(name, item_count, status, detail)`,
`HealthReport(sources)` with `has_failures` and `summary()`. `run_collectors_with_health` classifies each
source: exception → FAILED (with truncated detail), 0 items → EMPTY (legitimate on quiet days), else OK.
In the digest Lambda, `_maybe_alert` publishes to `ALERT_SNS_TOPIC_ARN` **only** when a source FAILED, before
the empty-items early return (so an outage still alerts even if nothing was collected).

## 9. Agent (Strands on AgentCore Runtime)

`agent/agent.py` builds a Strands `Agent` with a `BedrockModel` (Sonnet 4.6) and tools. The SYSTEM_PROMPT
encodes a strict routing table (Korean), Slack mrkdwn formatting rules, and a response template.

Tools (`agent/agent_tools.py`):
- `get_detail(item_number)` — load item content + ranking metadata from `state_manager`.
- `search_papers(query)` — Semantic Scholar (retry/backoff on 429).
- `search_community(query)` / `search_related_news(query)` — thin wrappers over the shared
  `_tavily_search(query, topic, include_domains)` helper.
- `recall_trends(query)` — cross-day semantic recall via `MemoryStore.recall` (AgentCore long-term memory).
- `make_visual(item_number, mode, panels)` — see §10.

`agent_runtime/app.py` (`BedrockAgentCoreApp`): on invoke, sets a correlation id, loads latest digest state
from Memory, sets `delivery_context` (channel/thread for media tools), runs the agent, and posts the reply to
Slack. The Slack-events Lambda (`slack_event_handler.py`) verifies the Slack signature (HMAC, timing-safe),
de-dupes via DynamoDB conditional writes, and async self-invokes to call the AgentCore runtime.

## 10. Visualization pipeline (synopsis → image)

`agent/visuals.py` generalizes "synopsis → visualization" with a `VisualMode` (a brief prompt + an image-prompt
builder) and a `MODES` registry:
- **comic** (`ComicSynopsisPrompt`): a 1–6 panel narrative cartoon (the agent picks the panel count to fit the
  story); Korean captions, English visual directions; rendered as a single/side-by-side/2x2/2x3 layout.
- **diagram** (`VisualizationBriefPrompt`): one explanatory concept infographic (flow/architecture/comparison).

`VisualGenerator.generate(title, content, mode, panels)`: brief via Claude (Bedrock) → `_parse_json_object`
→ mode-specific image prompt → **OpenAI `gpt-image-1`** (`b64_json`) → PNG bytes. `make_visual` uploads the
image to Slack via `output.slack_handler.send_image_to_slack` (`files_upload_v2`). Disabled gracefully when
`OPENAI_API_KEY` is absent. New modes are added by registering another `VisualMode`.

## 11. Infrastructure (CDK)

**`foundation_stack`**: VPC, ECR repo, DynamoDB dedup table (SSE + PITR-in-prod), S3 state bucket (S3-managed
encryption, versioning, block-public, enforce-SSL when CDK-created), ECS Fargate RSSHub service +
service-discovery, CodeBuild image build, SNS alerts topic (+ optional email subscription), AgentCore
**Memory** resource + execution role, and IAM roles. IAM is least-privilege: scoped `ssm:GetParameter*` on
`/{project}/{stage}/*`, scoped `bedrock:InvokeModel*` on foundation-model/inference-profile ARNs, scoped
`lambda:InvokeFunction` and `bedrock-agentcore:InvokeAgentRuntime`/Memory data-plane actions — no
account-wide managed policies.

**`application_stack`**: digest Lambda (DockerImage), Slack-events Lambda, API Gateway (+ stage throttling),
**WAFv2 WebACL** (rate-limit + AWS managed rule sets: Common, KnownBadInputs, IpReputation) associated to the
stage, EventBridge daily cron (config-driven hour/minute), AgentCore Runtime, SSM parameters for secrets,
CloudWatch alarms (Lambda errors ×2, API 5xx) → SNS. Secrets are plaintext `String` SSM parameters
(CloudFormation cannot create SecureString); the compensating control is the scoped IAM read policy —
promote to Secrets Manager for higher-sensitivity credentials.

## 12. Observability

`shared/logger.py`: structured JSON logs in AWS (`is_running_in_aws()`), human-readable locally; a
`ContextVar`-based correlation id (`set_correlation_id`/`get_correlation_id`) is injected into every record
and seeded from the Lambda request id / AgentCore payload. CloudWatch alarms route to the SNS alerts topic.

## 13. Testing & CI/CD

`tests/` (pytest, `asyncio_mode=auto`): collectors (respx-mocked HTTP/OAuth), aggregator, ranker parsing +
slot/origin-cap logic, health report, logger, memory store (local + AgentCore mocked), digest-handler alert,
agent tools, visuals, and CDK assertions (`aws-cdk.assertions` over both stacks). Coverage gate 45%.

`.github/workflows/ci.yml`: lint (ruff), format (black `--check`), tests + coverage gate, offline `cdk synth`
(`scripts/ci_synth.py`, dummy account — no AWS creds), and a Docker build (amd64, `--provenance=false`).

## 14. Key commands

```bash
uv run python main.py --dry-run --sources rss reddit   # partial dry run
uv run python main.py                                   # full pipeline + Slack
uv run python -m pytest tests/ -v                       # tests
uv run black --check . && uv run ruff check .           # lint/format
uv run python scripts/ci_synth.py                       # offline CDK synth
AWS_PROFILE=research uv run cdk deploy --all -a "uv run python scripts/deploy.py"
```
