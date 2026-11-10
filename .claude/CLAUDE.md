# OmniSummary Development Guidelines

## Project Overview
Proactive AI/ML daily digest: collects from 6 sources (RSS, Reddit, YouTube, X/Twitter via RSSHub, Web Search via Tavily, and for investing Naver 증권 broker research), scores them into a candidate pool, lets the editor LLM pick the strongest 5 stories, writes a Korean editorial digest, and publishes it to Threads with a Korean comic of the headline. A local deep-research CLI (`research_cli.py`, used by the OpenClaw skill at `~/Projects/agent-config/skills/omnisummary-research/SKILL.md`) researches a topic across web/papers/community and returns a persona-voiced, cited Markdown report as JSON on stdout; it publishes to Threads only with `--publish threads`. The same skill reads back past digests and trends for free with `recall_cli.py`. Slack is gone entirely.

## Architecture
```
[EventBridge Cron] → [Digest Lambda (Docker)]
  → Collectors (RSS, Reddit, YouTube, WebSearch, X/S3)
  → Aggregator (URL + title dedup)
  → Ranker (Claude Opus 5.5 for AI, Sonnet 5.5 for investing; parallel batches) → candidate pool (top_n + buffer; source slots shape the POOL)
  → TrendTracker (structured trends.json)
  → DigestGenerator (Claude Sonnet 5.5 picks the strongest top_n and writes DigestContent)
  → AgentCore Memory (digest snapshot) → Daily Visual Lambda (async)
       → Korean comic, form picked per story (gpt-image-2.5-sunburst) + Threads root/reply chain
[EventBridge hourly] → [Comment-reply Lambda] → answers readers' new comments (Sonnet 5.5, or declines)

[OpenClaw skill / terminal] → research_cli.py (local)
  → Deep Research Strands Agent, Claude Opus 5.5 (web_search, community_search, search_papers,
    read_url, recall_trends, recall_digest, attach_image, deliver_report)
  → JSON on stdout + reports/*.md; Threads only with --publish threads
```

## Development Principles
- **Simplicity first**: most straightforward approach wins
- **Async by default**: all I/O uses async/await
- **Pydantic models** at module boundaries
- **LCEL chains** for LLM (prompt | llm | parser)
- **%-formatting** for logger: `logger.info("Processing %s", value)`
- **uv** for packages, update pyproject.toml for deps
- **No comments** in code unless absolutely necessary
- **Prompts in English**, digest output in Korean
- **All error/system messages in English**

## Key Commands
```bash
uv run python main.py --dry-run --sources web_search   # Prints the digest; persists and publishes nothing
uv run python main.py                          # Full pipeline + comic + Threads (per config)
uv run python main.py --pin-url "<url>" [...] --force-republish  # Force URL(s) in; re-collects and writes a NEW digest
# A failed day is re-posted from its STORED snapshot by invoking the visual Lambda with
# {"digest_date": "YYYY-MM-DD"} (+ "force_republish": true only after checking nothing is live); the
# Threads Delivery alert carries that exact command. docs/operations.md is the operator runbook.
# main.py exits 1 when nothing was collected, no digest was produced, or the post did not fully land.
uv run python research_cli.py "<topic>"        # Deep research: one JSON object on stdout, nothing published
uv run python research_cli.py "<topic>" --publish threads   # ...and publish a Threads thread
uv run python recall_cli.py digest 2026-10-11  # Free, read-only: that day's lead + stories (JSON); `recall_cli.py trends` for trends
uv run python scripts/sample_visual_brief.py 2026-10-17 /tmp/vis [--render]  # A stored day's comic plan + brief (+ image); publishes nothing
uv run python scripts/cost_report.py --days 7 [--by stage]  # Spend per day and Lambda/stage from the deployed logs (read-only)
./scripts/sync_all_to_s3.sh                     # Local → S3: X/RSSHub + YouTube (starts the rsshub container if down)
uv run python scripts/sync_to_s3.py rsshub      # One source
uv sync --extra dev                             # once — the dev extra carries pytest, mypy, pre-commit, CDK
uv run python -m pytest tests/ -v               # Tests
uv run lint-imports                             # Ports-and-adapters dependency direction
uv run pre-commit install                       # once — ruff, black, file checks, uv lock --check before commit

# AWS — deploy the pushed, CI-green HEAD of main: build, push (tagged with the commit), deploy pinned to the
# digest. It refuses a dirty tree, an unpushed HEAD or a red CI run. CloudFormation does not redeploy a Lambda
# when the image tag string is unchanged, which is why deploy.py requires DIGEST_IMAGE_REF (the script sets it;
# FOUNDATION_ONLY=1 is only for bootstrap and a fresh account's first foundation deploy). Uses the repo-pinned
# CDK CLI (npx cdk): a global cdk can lag aws-cdk-lib and fail with a cloud-assembly schema mismatch.
scripts/release.sh   # account from aws.profile (not the shell's AWS_PROFILE)
# Secrets are NOT in the template (a CFN template can't hold a SecureString). The stack creates each
# SSM parameter holding a placeholder; write the real values right after:
AWS_PROFILE=research uv run python scripts/put_secrets.py    # --dry-run / --verify / --force
# Tagged APPLICATION inference profiles make Bedrock spend attributable; the resolver prefers them and
# falls back to the system-defined profile. Names come from config (project/stage). Once per account/stage,
# and again after changing a stage model:
AWS_PROFILE=research uv run python scripts/put_inference_profiles.py   # --dry-run / --delete
# put_secrets SKIPS parameters that are already SecureStrings: the Threads token is rotated in place by
# the refresh Lambda, so reasserting the local .env copy would restore an expired token.
# Lockfile: regenerate with `uvx --from uv==0.6.14 uv lock` (CI and the Dockerfile pin uv 0.6.14; a newer
# uv rewrites the lock revision and 0.6.14's `uv lock --check` then fails).
```

## Editions
One codebase, several editions. `OMNISUMMARY_EDITION=invest` (research CLI: `--edition invest`) loads
`config/invest.yaml` (gitignored; `config/invest-template.yaml` tracked): the investing digest at 07:30 KST
(22:30 UTC, 42h window), same narrator, same Threads account, its own S3 prefix (`omnisummary/invest`),
AgentCore sessions (`digest-invest-<date>`), Lambdas (`omnisummary-dev-invest-digest`/`-visual`), EMF
`Edition` dimension and empty-digest alarm. With no extra edition the synthesized templates are
byte-identical to before. `pipeline.domain`, `pipeline.ranking_criteria` and `pipeline.threads_footer`
(code-owned disclaimer, never trimmed) and `pipeline.countdown_*` (the lead's opening gag, paired: "AGI 등장까지 N일 남았다" to 2027-12-31 by default,
"보편적 고소득 시대까지 N일 남았다" to 2035-11-19 for investing) are the edition's knobs; research gets `market_quote` + `analyst_ratings` and
`search_reports` via `agent.market_quotes` / `agent.report_search_domains`. No buy/sell calls or own
price targets (`pipeline.domain_rules`; the comment prompt declines such questions). Korean real estate is one lens of the investing edition (policy, lending, official statistics; 분양/견본주택 promotion is not news). Sources (both editions): technical access controls are never circumvented (paywall, bot check, 403/429, someone else's login wall; not via a UA swap or Tavily extraction either), while policy signals (robots.txt, ToS) are the owner's documented call: Korean press, X and Reddit are used, Telegram is not (docs/design.md, 수집 원칙). Korean broker research comes
from `collectors.naver_research` (Naver 증권 JSON; SourceType.RESEARCH, one origin per broker): the post names the broker and
never links the Naver page (`DigestItem.show_link`); same-stock same-day notes collapse into one consensus item. Two of the investing five are the pool's best research note or X post (`pipeline.primary_quota`), a note with its attached PDF's text (stock.pstatic.net, public; owner's call 11-07): 12-2 blind on 7 stored days. consensus.hankyung.com and nhqv.com are not collected.
Investing posting is ON in the local `config/invest.yaml` since 2026-11-01 (the tracked template keeps it off).

## Project Structure
Ports and adapters (hexagonal). Dependencies point inward only, checked in CI by `uv run lint-imports`
(`[tool.importlinter]` in pyproject): entrypoints → adapters → application → domain → shared.
- `entrypoints/` - `bootstrap.py` (the composition root: builds every adapter and hands it to the use cases) + the four Lambda handlers (digest, daily visual, Threads token refresh, comment replies)
- `adapters/` - the port implementations: `bedrock.py` (LLM factory + model registry), `aws.py` (sessions, SSM secrets), `state_store.py`, `memory.py` (AgentCore digest snapshots), `alerts.py` (SNS), `metrics.py` (EMF), `threads.py` (Threads API + `ThreadsGateway`), `openai_images.py`, `research_backends.py` (Tavily + OpenAlex + `WebResearchSources`), `og_image.py`, `collectors/` (RSS, Reddit, RSSHub, YouTube, WebSearch, NaverResearch + the S3 park loader), `research_agent.py` + `research_tools.py` (Strands; the tools only carry the spec the model reads)
- `application/` - `ports.py` + the use cases: `digest_pipeline.py` (collect/rank/write, `persist_digest`, `publish_inline`), ranker, digest_generator, trend_tracker, daily_visual + visual_generator, digest_publisher, comment_replies, `research.py` (citation guard, fact check, delivery), `history.py` (dedup ledger, rolling logs, Threads ledger, comment-reply ledger), `prompts/`
- `domain/` - models, aggregator (URL normalization + dedup), formatting, renderers (Threads posts), digest_snapshot (`DigestSnapshot` value object)
- `shared/` - the kernel: config, constants, logger, naming (names CDK and code share), utils (retry, LLM-JSON parsing)
- `research_cli.py`, `recall_cli.py`, `main.py` - local CLIs; they get their adapters from `entrypoints/bootstrap.py`
- `infrastructure/` - CDK stacks (foundation + application)
- `scripts/` - deploy, put_secrets, put_inference_profiles, ci_synth, sync_to_s3 + sync_all_to_s3.sh, sample_digest, sample_visual_brief, sample_comment_replies, cost_report, render_excalidraw.mjs
- Tests hand the core fakes (`tests/fakes.py`: FakeThreads, FakeRenderer, FakeSources, MemoryStateStore) instead of patching clients. A new outside dependency goes behind a port in `application/ports.py`, an adapter, and one line in `bootstrap.py`.

## Config
- `config/config.yaml` - main config (gitignored; `config/config-template.yaml` is the tracked one)
- `.env` - secrets (TAVILY_API_KEY, OPENAI_API_KEY, YOUTUBE_API_KEY, THREADS_*, etc.)
- AWS secrets in SSM Parameter Store

## LLM
- AWS Bedrock Claude (and GPT, see below) via langchain-aws ChatBedrockConverse. Only `langchain-core` + `langchain-aws`
  are dependencies — **do NOT add the top-level `langchain` umbrella package**; nothing imports it and
  it drags in advisory-carrying packages this code never reaches.
- Models per stage: the table with reasons is docs/design.md §6 (`#모델-팩토리`); the measurements are in
  docs/decisions.md. Ranking: Sonnet 5.5 code default, Opus 5.5 in the AI edition's config (Sonnet under-filled
  the AI pool). Digest, grounding, visual editor and art director, comment replies, story groups and the research
  thread's fact check read `digest_model` (Sonnet 5.5); trends and query refine Sonnet 5.5; deep research Opus 5.5;
  CountTokens Sonnet 4.6 (the 5.5 models don't expose it, and its tokenizer counts English about 1.7x short).
  Schema-shaped answers use json_schema constrained output (an answer outside the schema raises
  OutputParserException; the ranker turns that into [] so the batch is split, not re-sent). The research fact
  check is logged as `stage=research-fact-check` and is NOT in the research CLI's `usage` JSON.
- `LanguageModelInfo` flags are verified against Converse: only Sonnet 4.6 takes `temperature`;
  Opus 5.5 / Sonnet 5.5 reject forced tool_choice (the factory pins `("auto",)`); Opus 5.5 / Haiku 5.5
  default effort to `medium`, so `effort="high"` is sent via `output_config` (the research agent uses
  the same flag). Fable 5.1 is NOT registered: Bedrock rejects it under the account's default
  data-retention mode.
- GPT-6.1 Sol / Terra (`openai.gpt-6.1-*`) are registered on the same Converse API and IAM (`provider="openai"`):
  effort goes as `reasoning.effort` (they reject `output_config`), the factory states `supports_tool_choice_values`,
  and the research agent sends no Strands cache points (GPT rejects them). They lost the blind digest comparison;
  Sonnet stays the default (docs/decisions.md).
- The digest LLM writes prose only and NO markup; `domain/renderers.py` emits the Threads posts. The root is the lead
  plus the other stories' titles (code-owned): the story replies get only 2-5% of the root's views.
- Image: `gpt-image-2.5-sunburst`, quality xhigh, one render, shipped as drawn. The editor picks the form (a single
  cartoon, a 2-6 panel strip or an illustrated explainer; 2x2 is `portrait`, a stacked strip `tall` 848x2544), casts
  the story's real people and companies, draws ONE idea as literal scenes where the story happens, and the last frame
  SHOWS the story's own twist. The narrator (`assets/persona.jpg`, an `images.edit` input) joins only with
  `use_character`. The editor sees the headline article and `recent_comics.json`, never the lead. The posted brief is
  kept at digest_state/visual_briefs/<date>.json. Each of these rules lost a blind eval when removed or changed, and so
  did the alternatives tried (image models, correction passes, brief restructures): docs/design.md §5.6 and
  docs/decisions.md. Do not add comic rules without `scripts/eval_harness.py` judged both orders, on images.
- Every `get_model()` takes `stage=` and logs `LLM usage stage=... model=... input=... output=... usd=...`:
  the bill is per MODEL and several stages share one. Prices live in `LanguageModelInfo.price` (global-profile
  list price); renders log `Image usage ... usd=`, Tavily `Search usage ... credits=`; in Lambda each also emits an EMF
  `LLMCostUSD`/`ImageCostUSD`/`TavilyCredits` datapoint. The research CLI reports `usage` (with `usd`) in its JSON.

## Gotchas
- **DigestPrompt's JSON key order is load-bearing**: it asks for `items` FIRST and `lead` LAST so the
  lead comments on stories already written (measured overlap with the headline reply: 0.21-0.41 →
  0.03-0.21). The headline is always `items[0]`. **Do NOT reorder the requested
  keys to match `DigestContent`'s field order.**
- The daily-visual Lambda is the ONLY digest publish path (the comment-reply Lambda only answers comments): it loads the snapshot for the date the
  digest Lambda passed it, and both Lambdas re-raise on failure (retry_attempts=0, so a re-raise
  cannot duplicate a post).
- Prompt changes must SIMPLIFY or restructure AND win a blind both-order judge on stored days
  (`scripts/eval_harness.py`) — iteratively adding rules to these prompts has a documented history of
  making the output worse, and simplifications lose too (2026-11-07: the implication spec 3-9, the comic
  frame clause 10-12 on images).
- **Do NOT regex-check the editor's Korean.** A `shared/prose_lint.py` existed for one release and was
  deleted whole. Measured over the stored corpus (14 digests, 44 prose fields) it found ONE true
  positive while an em-dash pattern fired on 3 of 4 digests over idiomatic Korean, a figure check read
  52.2 and 52.9 as the same number, and a lead check flagged exactly the shape `DigestPrompt` mandates.
  The remedy was a byte-identical re-ask, so even a CORRECT firing only re-rolled the dice. Prose rules
  belong in `KOREAN_STYLE_RULES`, which the prompt reads (it now carries one line against the "A가 아니라
  B" contrast formula: 41/155 shipped implications used it). `_verify_grounding` is NOT a precedent: it
  checks containment in the source text, not grammar. Nor is `unsafe_reply_reason` in `application/comment_replies.py`:
  it is a security gate that keeps links, domains, other handles and hashtags out of auto-posted comment replies.
- **Quality-first selection**: source slots and `max_per_origin` shape the candidate POOL; the editor
  picks what ships. There is no per-collector cap past the slot (a cap of 2 kept 0.72-0.80 items out of
  the pool while 0.60 ones got in on 9/16 days); the ranker sees recent digest titles so a story back
  under a new URL scores as a rehash. Do not re-add a per-source guarantee on the shipped stories (the one exception is the owner's
  investing `primary_quota`, measured 12-2 with the notes' full text) or an alert on the
  shipped source mix (it fired on 14/30 runs, nobody could act, and it buried the X outage alert).
- **config.yaml is gitignored and every config model is `extra="forbid"`**: a key deleted from the code
  but left in the operator's `config.yaml` makes `Config.load()` fail. Remove both together.
- **Local runs read the deployed AgentCore Memory**: without `MEMORY_ID` (set only on the Lambdas)
  `create_memory_store` finds the stack's memory by name (`digest_memory_name`, shared with CDK) in
  `config.aws.region`, so a local run sees the same snapshots production writes. The lookup AND the
  `bedrock-agentcore` read client share ONE session on `available_boto_profile(config.aws.profile)` +
  `config.aws.region`: a local shell may inherit `AWS_PROFILE`/`AWS_REGION` for another account or the
  model region, and a read client on them found the memory by name, then answered "Memory not found".
  Pass `--profile`/`--region` to the AWS CLI.
- **`LOG_LEVEL=DEBUG` in `.env` must never reach botocore**: its DEBUG logs the request headers that
  carry the session token. The research CLI pins third-party loggers to WARNING; keep it that way.
- **Never add or change template properties on the SSM secret parameters** (not even a Description):
  CloudFormation would re-put the placeholder over the live secret. `tests/test_infrastructure.py` pins it.
- The Strands `CacheConfig` must use `strategy="anthropic"`, not `"auto"`: an application-profile ARN
  contains neither "claude" nor "anthropic", so `"auto"` silently drops every cache point.
- **The blind judge (`scripts/eval_harness.py judge`) must see the source the editor saw.** At 1,500 characters
  per item it marked real figures from further down as invented and flipped three verdicts (2026-11-06:
  the prompt bundle 3-8 → 14-8). Judge both orders (swap the labels) before trusting a one-run result.
- X dies silently when the local `rsshub` container is stopped (`--restart unless-stopped` does not
  restart an explicitly stopped container; X was dark 54 days in 2026-08/10). `sync_all_to_s3.sh` now
  starts it; a STALE rsshub alert means check the container and the X cookies.
