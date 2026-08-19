<div align="center">

# 🗞️ OmniSummary

**A proactive AI/ML daily digest. It collects from five source families, ranks with an LLM, writes a Korean editorial digest, and delivers it to whichever channels the config enables (Slack, Threads, or both). A Slack-triggered deep-research agent researches any topic across web, papers, and community, then posts a persona-voiced, cited report.**

Runs on AWS · Bedrock AgentCore (Runtime + Memory) · Amazon Bedrock (Claude).

[![CI](https://github.com/bits-bytes-nn/omnisummary/actions/workflows/ci.yml/badge.svg)](https://github.com/bits-bytes-nn/omnisummary/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![AWS CDK](https://img.shields.io/badge/IaC-AWS%20CDK-orange)
![Bedrock](https://img.shields.io/badge/LLM-Amazon%20Bedrock%20(Claude)-green)

🇰🇷 [한국어 README](./README.ko.md) · 📐 [Design doc](./docs/design.md)

![OmniSummary architecture](docs/diagrams/architecture.png)

</div>

---

## What it does

There are two independent paths.

**The daily digest** runs on a cron. It collects from RSS/Substack, Reddit, YouTube, X/Twitter and web search; deduplicates; ranks everything with Claude Opus; writes a Korean editorial digest with Claude Sonnet; then renders that digest per channel and delivers it. An illustration is generated for the headline story and shipped with it.

**The deep-research agent** is triggered by a Slack mention. Given a free-form topic it researches the open web, academic papers, and community discussion on its own, then writes a cited Korean report in the same narrator voice as the digest and posts it to Slack (or Threads, if you ask).

![How the digest works](docs/diagrams/concept-pipeline.png)

## Features

- **Multi-source collection** — Reddit (public `.rss` feed), YouTube, X/Twitter (via RSSHub), RSS/Substack, web search (Tavily)
- **LLM ranking** — Claude Opus 4.8 (Opus 5 / Sonnet 5 also selectable) scoring on multiple axes, with source-slot and per-origin diversity caps so one channel can't take over the day
- **Editorial digest** — Claude Sonnet 5 writes Korean prose, with cross-day trend continuity
- **Multi-channel delivery** — one structured digest, rendered per channel: Slack (Block Kit) and Threads (image root + flat reply chain). Each toggles independently
- **Deep-research agent** — an autonomous Strands agent with eight single-purpose tools, attaching the source article's OG image
- **AgentCore-centric** — digest state lives in Bedrock AgentCore Memory; the agent runs on AgentCore Runtime
- **Operational excellence** — per-source health reporting (`OK` / `EMPTY` / `FAILED` / `STALE` / `DEGRADED`) into SNS email alerts, structured JSON logs with correlation IDs, 12 CloudWatch alarms, AWS WAF on the public API
- **AWS deployment** — Lambda + EventBridge cron + Bedrock AgentCore + ECS (RSSHub), all in CDK

## Quick Start

### Prerequisites

- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- Docker (for RSSHub and for AWS deployment)
- An AWS account with Bedrock access
- A Slack workspace with a bot app

### Installation

```bash
git clone <repo-url> && cd omnisummary
uv sync
cp config/config-template.yaml config/config.yaml
cp .env.template .env
```

### Configuration

`config/config.yaml` holds everything that isn't a secret. The interesting knobs:

```yaml
collectors:
  rss:
    enabled: true
    feeds: ["https://feeds.feedburner.com/geeknews-feed"]
  reddit:
    enabled: true
    subreddits: [LocalLLaMA]
  youtube:
    enabled: true
    channels: ["https://www.youtube.com/@AndrejKarpathy"]
    lookback_hours: 30      # must reach back to the previous run: config rejects less than 48 - the run hour
  web_search:
    enabled: true
    trend_searches:
      - name: frontier_models
        queries: ["frontier AI model release GPT Claude Gemini Llama"]
        topic: news
  rsshub:
    enabled: true
    base_url: "http://localhost:1200"
    accounts:
      - username: "karpathy"
        platform: x

pipeline:
  top_n: 5
  min_score: 0.6
  ranking_model: "anthropic.claude-opus-4-8"
  digest_model: "anthropic.claude-sonnet-5"
  max_per_origin: 1        # cap per channel / author / subreddit / feed / web host
  source_slots: {web: 1, x: 1, rss: 1, reddit: 1, youtube: 1}
```

Every field is documented in [design.md §3](docs/design.md#3-설정).

### Secrets

Secrets go in `.env` locally (see `.env.template`) and in SSM Parameter Store on AWS. Only the first three are needed to run at all; everything else enables one specific feature and degrades gracefully: the feature logs a line and is skipped when its key is absent.

| Variable | Needed for |
|----------|-----------|
| `SLACK_BOT_TOKEN` | Digest delivery + the agent (skippable only with `enable_slack_post: false`) |
| `SLACK_CHANNEL_ID` | Target channel for the digest |
| `TAVILY_API_KEY` | The `web_search` collector + the agent's community/news search |
| `SLACK_SIGNING_SECRET` | The Slack-events API Gateway path (verifies inbound agent events) |
| `YOUTUBE_API_KEY` | YouTube collector (without it: RSS fallback, no transcripts) |
| `OPENAI_API_KEY` | The daily visual's gpt-image render |
| `THREADS_ACCESS_TOKEN` / `THREADS_USER_ID` | Threads delivery (60-day token, auto-refreshed into SSM on AWS) |
| `ALERT_EMAIL` | Source-health SNS email alerts (AWS) |
| `CLOUDFLARE_PROXY_URL` / `CLOUDFLARE_PROXY_TOKEN` | AWS only. Reddit `.rss` + YouTube RSS from datacenter IPs |
| `TWITTER_AUTH_TOKEN` / `TWITTER_CT0` | X/Twitter via RSSHub. Your x.com session cookies |
| `S3_SYNC_ACCESS_KEY_ID` / `S3_SYNC_SECRET_ACCESS_KEY` | Optional dedicated creds for the local→S3 sync (otherwise `AWS_PROFILE`) |

> **Secrets never pass through the CDK stack.** A CloudFormation template can't hold a SecureString, so
> handing values to the stack would publish them in plaintext into `cdk.out`, the staging bucket, and
> every `GetTemplate` response. The stack creates only the parameter *paths* holding a placeholder;
> `scripts/put_secrets.py` writes the real values as SecureStrings after the deploy. See
> [Secrets on AWS](#secrets-on-aws).

### Setup checklist

To produce a digest **locally** (Slack delivery, no X, no visual):

1. `uv sync`, then copy the two template files as above.
2. Fill `SLACK_BOT_TOKEN`, `SLACK_CHANNEL_ID`, `TAVILY_API_KEY` in `.env`.
3. Make sure Bedrock is reachable in `aws.bedrock_region` (default `us-west-2`) and set `AWS_PROFILE` or standard AWS credentials. The ranking and digest LLMs run on Bedrock even for a local run.
4. `uv run python main.py --dry-run --sources rss reddit` prints the digest.

Then add capabilities one at a time:

| Want… | Set | Notes |
|-------|-----|-------|
| YouTube items **with transcripts** | `YOUTUBE_API_KEY` + the local sync | Transcripts only fetch from a residential IP |
| X/Twitter items | RSSHub container + the local sync | See below |
| Daily visual | `OPENAI_API_KEY` | gpt-image-2 |
| Threads delivery | `THREADS_ACCESS_TOKEN`, `THREADS_USER_ID`, `enable_threads_post: true` | **Also needs `enable_daily_visual: true`**. The daily-visual Lambda is what posts to Threads, because the image and the text have to ship as one post set |

### RSSHub container (X/Twitter)

X is read through a local [RSSHub](https://docs.rsshub.app/) container, which needs two cookies from a logged-in x.com session: **`auth_token`** and **`ct0`**.

Grab them from your browser's devtools (on macOS F12 is often remapped, so use ⌥⌘I): **Chrome** → Application → Cookies → `https://x.com`; **Safari** (enable the Develop menu first) or **Firefox** → Storage → Cookies → `x.com`.

```bash
docker run -d --name rsshub --restart unless-stopped -p 1200:1200 \
  -e NODE_ENV=production -e CACHE_TYPE=memory \
  -e TWITTER_AUTH_TOKEN='<auth_token>' \
  -e TWITTER_CT0='<ct0>' \
  diygod/rsshub:latest

curl -s "http://localhost:1200/twitter/user/karpathy" | head   # smoke test
```

Without the cookies the container still starts, but X feeds come back empty. Cookies expire every so often. When the RSSHub failure rate climbs (it's logged as a warning), refresh them and recreate the container. On AWS the same image runs on ECS Fargate.

## Usage

```bash
# Digest pipeline, no delivery
uv run python main.py --dry-run --sources rss reddit

# Full pipeline + delivery
uv run python main.py

# Deep-research agent, locally: research a topic and print the rendered report
uv run python research_cli.py "<topic>" --dry-run
uv run python research_cli.py "<topic>" --channel both --dry-run   # preview Slack + Threads

# Local→S3 sync for the sources that block datacenter IPs (X/RSSHub + YouTube transcripts)
./scripts/sync_all_to_s3.sh                  # both; one failing won't block the other
uv run python scripts/sync_rsshub_to_s3.py   # X only
uv run python scripts/sync_youtube_to_s3.py  # YouTube only
```

| Flag | Description |
|------|-------------|
| `--sources rss reddit youtube` | Select specific sources |
| `--dry-run` | Skip delivery, print to console |
| `--top-n 5` | Override how many items to select |
| `--date 2026-03-28` | Set the digest date (default: today, KST) |
| `--pin-url <url> [<url> ...]` | Force URL(s) into the top stories regardless of score. YouTube URLs resolve via the Data API, others via Tavily. Local CLI only |
| `--force-republish` | Re-post today's digest even if it already went out (bypasses the Threads idempotency guard) |

## How the pipeline works

A summary of each stage. [design.md](docs/design.md) is the line-by-line reference and explains *why* each piece is shaped the way it is.

**1. Collection.** Every collector runs async in parallel with its own lookback window, and reports its own health. Two sources are special: X/Twitter and YouTube transcripts are blocked from datacenter IPs, so a local cron collects them on a residential IP, parks them in S3, and the Lambda reads the parked file. A park file older than its age budget still gets used, since stale beats empty, but the source reports `STALE` so a stopped cron can't look healthy.

| Collector | Source | Method |
|-----------|--------|--------|
| `RedditCollector` | Reddit public `.rss` | Direct first, Cloudflare proxy as fallback (no API app needed) |
| `YouTubeCollector` | YouTube Data API v3 | S3 park file on AWS, live otherwise |
| `RSSCollector` | RSS/Atom | feedparser |
| `RSSHubCollector` | X/Twitter via RSSHub | S3 park file on AWS, local Docker otherwise |
| `WebSearchCollector` | Tavily | Direct, with LLM query refinement |

**2. Aggregation.** `ContentAggregator` deduplicates by URL and by normalized title. When two items collide it keeps the *better* one (pinned > longer body > first seen) rather than whichever arrived first, so a thin Reddit link-post can't displace the full article.

**3. Ranking.** `ContentRanker` scores items with Claude Opus on technical substance, practitioner value, novelty, industry impact, research significance and source authority, with hard filters for promos and thin content. It then selects for diversity: `source_slots` guarantees a minimum per source type, `max_per_origin` caps any single channel/author/feed/host, and a single fill loop relaxes those caps in a fixed order when the digest would otherwise come up short.

**4. Trend tracking.** `TrendTracker` keeps structured trends in `trends.json`. The LLM only classifies today's items into existing or new trends; all bookkeeping is deterministic Python: date stamping, the active/cooling/archived lifecycle, recency-decay momentum, evidence caps. Active and cooling trends feed the next day's digest, which is what gives it cross-day continuity.

**5. Digest generation.** `DigestGenerator` produces a structured `DigestContent` (a `lead`, plus `items[]` each with title/url/body/implication). The LLM writes **only prose**, with no markup and no source tags. Prose budgets are computed in code from the parts code owns, so an item can't overflow the Threads 500-character limit.

> ⚠️ **The JSON key order in the prompt is load-bearing.** It asks for `items` first and `lead` last, so the lead comments on stories that are already written. Measured word overlap with the headline reply dropped from 0.21–0.41 to 0.03–0.21. Do **not** reorder the requested keys to match `DigestContent`'s field order. That tidy-up is a regression.

**6. Channel rendering.** Per-channel renderers in `output/renderers.py` turn one `DigestContent` into Slack Block Kit or a Threads root + flat reply chain. Formatting lives in code, not in prompt rules.

**7. Daily visual.** `DailyVisualMaker` illustrates the **headline** story specifically, so the image, the lead, and the text all point at the same thing. The editor briefs *how* to draw it and picks the orientation; `VisualGenerator` renders it with gpt-image-2. This Lambda also owns the Threads post, and a failed render never swallows the digest. The text still goes out.

**8. Deep-research agent.** An autonomous Strands agent on AgentCore Runtime, triggered by a Slack mention. It composes these eight tools freely. For example, "diffusion LLM 최신 동향" goes `web_search`/`search_papers`/`community_search` → `read_url` → `attach_image` → `deliver_report`:

| Tool | Function |
|------|----------|
| `web_search(query, recency)` | Tavily open web; `recency="news"` for recent news |
| `community_search(query)` | Tavily over Reddit, X, HN, Substack |
| `search_papers(query)` | Semantic Scholar |
| `read_url(url)` | Fetch and extract a primary source's full text |
| `recall_trends(query)` | Keyword match over `trends.json`, momentum-ranked |
| `recall_digest(digest_date)` | What one specific day's digest carried. Never falls back to another date |
| `attach_image(source_url)` | Download a source's OG image and stage it for delivery |
| `deliver_report(report, channel)` | Render and post to Slack (default) or Threads |

Delivery is channel-aware in code, not in prompt rules. If the agent finishes without delivering anything, the runtime posts the report to Slack as a fallback.

## AWS Deployment

### Deploying

Build and push **both** images first (see [Docker images](#docker-images)), then deploy pinning the pushed digest. CloudFormation won't redeploy a Lambda when the image *tag* string is unchanged, so pass the `sha256` digest explicitly.

Use the repo-pinned CDK CLI, not a global `cdk`. The CLI is pinned in `package.json` to a version compatible with the `aws-cdk-lib` in `pyproject.toml`, and a global one can lag the library and fail with a cloud-assembly schema mismatch.

The **order matters on a fresh account**, because the foundation stack owns the only ECR repository and the application stack's Lambdas resolve their image out of it. There is nowhere to push before the foundation exists, and `deploy --all` fails when Lambda cannot resolve an image.

```bash
npm install                                       # once: installs the pinned CDK CLI
export AWS_PROFILE=<profile>

# 1. Once per account+region: create the CDK bootstrap resources (staging bucket, roles).
npx cdk bootstrap -a "uv run python scripts/deploy.py"

# 2. Foundation FIRST — it creates the ECR repo the images are pushed to.
npx cdk deploy '*-foundation' -a "uv run python scripts/deploy.py"

# 3. Log in to that repo and push both images (see Docker images below).
#    The URI is derived, not looked up: <account>.dkr.ecr.<region>.amazonaws.com/<project>-<stage>-agent
ECR_URI="$(aws sts get-caller-identity --query Account --output text).dkr.ecr.<region>.amazonaws.com/omnisummary-<stage>-agent"
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin "${ECR_URI%%/*}"
docker build --platform linux/amd64 --provenance=false -t "$ECR_URI:latest" . && docker push "$ECR_URI:latest"
docker buildx build --platform linux/arm64 --provenance=false -f Dockerfile.agentcore -t "$ECR_URI:arm64" . --push

# 4. Deploy everything, pinning the pushed digest.
export DIGEST_IMAGE_REF=sha256:<pushed-digest>    # AGENTCORE_IMAGE_REF defaults to :arm64
npx cdk deploy --all -a "uv run python scripts/deploy.py"

# 5. Secrets and cost attribution.
uv run python scripts/put_secrets.py             # then write the secrets
uv run python scripts/put_secrets.py --verify    # read-only: any left unset?
uv run python scripts/put_inference_profiles.py  # once per account/stage
```

On every later deploy only steps 3-4 apply: push the new images, then `deploy --all` with the fresh digest.

### What gets created

| Resource | Purpose |
|----------|---------|
| **Lambda** (Docker) | Digest pipeline, 15 min timeout |
| **Lambda** (Docker) | Daily visual, 15 min timeout. Async, off the digest critical path, and the only Threads publish path |
| **Lambda** | Slack event handler, 60 s timeout. The only internet-facing path, on its own least-privilege role |
| **Lambda** (Docker) | Threads token refresh (~50-day schedule, writes the renewed 60-day token back to SSM) |
| **API Gateway** + **WAFv2** | `POST /slack/events` with rate limiting, managed rule sets and stage throttling |
| **EventBridge** | Daily digest cron (config-driven hour/minute, **UTC**) + the token-refresh schedule |
| **Bedrock AgentCore** | Runtime (the agent, arm64) + Memory (digest snapshots) |
| **ECS Fargate** | RSSHub container. `aws.rsshub_desired_count` **defaults to 0**: the digest reads the S3 park file first and never reaches this service, so running it around the clock is ~$40/month of pure cost. The task definition is still deployed, so set it to 1 to restore the AWS fallback |
| **SSM Parameter Store** | All secrets, as SecureStrings written out-of-band |
| **S3** | Trends + park files + Threads image hosting |
| **DynamoDB** | Slack event deduplication |
| **SQS** | Async DLQ. Every Lambda runs `retry_attempts=0`, because Threads has no idempotency key and a retry would double-post. The handlers re-raise, so failures land here for replay |
| **SNS** | Alert topic (email) |
| **CloudWatch** | Structured logs, one-month retention, and 12 alarms (per-Lambda Errors ×4 + Timeout ×4, API 5xx, EmptyDigest, async DLQ, AgentErrors) |
| **ECR** | Docker images (amd64 for Lambda, arm64 for AgentCore) |

### Secrets on AWS

The stack creates each SSM parameter holding a placeholder; `scripts/put_secrets.py` writes the real values from your `.env` as SecureStrings after the deploy. Re-deploys don't clobber them, because CloudFormation only updates a resource whose template properties changed and the placeholder never changes.

Four behaviours worth knowing:

- **Parameters that are already SecureStrings are skipped.** The Threads token is rotated in place by the refresh Lambda, so re-asserting the local `.env` copy would restore an expired token. Use `--force` only when you mean to overwrite the live value.
- **A missing or empty environment variable is skipped, never blanked**, so a partial `.env` can't wipe a working parameter. And `resolve_secret()` treats a parameter still holding the placeholder as unset, so forgetting to run `put_secrets.py` degrades to the normal missing-credential path instead of sending the placeholder to an API as a token.
- **One parameter SSM refuses does not abort the run.** Failures are listed under `FAILED` and the script exits non-zero, but every other secret still gets written.
- **`--verify` is a read-only report** of which parameters are set, which still hold the placeholder, which are plaintext `String`, and which are missing. Safe to run against prod any time.

The X session cookies reach the RSSHub Fargate container through the task definition's `secrets` block. The ARN goes in the template, and the ECS agent fetches the value at task start.

### Bedrock cost attribution

On-demand Bedrock bills against no taggable resource, so `InvokeModel` token spend can't carry a cost-allocation tag. In a shared account the Bedrock line is one unattributable total. An **application inference profile** *is* taggable, and invoking through its ARN attributes the usage.

`scripts/put_inference_profiles.py` creates one per configured model, tagged `Project`/`Stage` and copied from the system-defined cross-region profile so the same global routing is inherited. `BedrockCrossRegionModelHelper` prefers them at resolution time. Since that resolver is the one place both the LangChain factory and the Strands agent go through, the agent's spend is captured too. A missing profile or a denied lookup silently keeps the system-defined id: cost reporting must never stop a generation.

Two things to watch:

- `application-inference-profile` is a **different IAM resource type** from `inference-profile`. The policy grants both; drop the former and every Bedrock call becomes AccessDenied the moment a profile exists.
- Activate the `Project` cost-allocation tag in Billing for this to reach Cost Explorer. It takes up to 24 h and is not retroactive.

Complementing this, every `get_model()` call takes a `stage=` and logs `LLM usage stage=... model=... input=... output=...`, because the bill is per *model* while the digest, grounding pass, trend classifier, visual editor, query refinement and research agent all share Sonnet 5.

### Docker images

Both images install the **exact set `uv.lock` pins** (`uv export` → `uv pip install --system`, the project itself `--no-deps`), so an image can never run a dependency set CI never tested. Dependencies install before the source is copied, so a code-only change reuses that layer. Both run **non-root** (uid 10001), and `.dockerignore` keeps `.env`, `.venv`, `logs/` and `cdk.out` out of the build context.

Both go to the ECR repository the **foundation stack** creates, `<account>.dkr.ecr.<region>.amazonaws.com/<project>-<stage>-agent`, so log in to it first (the Docker credential is short-lived; re-run the login when a push 401s).

```bash
aws ecr get-login-password --region <region> \
  | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com

# Lambda (amd64)
docker build --platform linux/amd64 --provenance=false -t <ecr-uri>:latest .
docker push <ecr-uri>:latest

# AgentCore (arm64)
docker buildx build --platform linux/arm64 --provenance=false \
  -f Dockerfile.agentcore -t <ecr-uri>:arm64 . --push
```

### Cloudflare Workers proxy

Reddit and YouTube are blocked from AWS datacenter IPs, so a Cloudflare Worker fronts them:

```bash
cd cloudflare-proxy
npx wrangler login
npx wrangler secret put PROXY_TOKEN   # a secret, NOT a wrangler.toml [vars] entry
npx wrangler deploy
```

The worker is deliberately not a general-purpose proxy. Only hosts in the `ALLOWED_HOSTS` var are fetched (exact or suffix match), anything else gets a `403`; redirects are followed manually with a bounded hop count and every `Location` re-checked against the same allowlist, so one `302` from an allowed host can't turn it into an open proxy; a caller-supplied `headers` blob is never merged into the outbound request, so a token holder can't forge `Cookie`/`Authorization`/`Host`; the token compare is constant-time; and the token stays in the query string on purpose, because the callers hand the proxied URL straight to a plain GET whose headers they don't control.

### Local cron

X/Twitter and YouTube transcripts both block datacenter IPs, so they must be collected locally and synced to S3 *before* the AWS digest runs. The digest cron is `aws.digest_cron_hour`/`minute` interpreted as **UTC**, so the default `10:00 UTC` is `19:00 KST`. Schedule the sync a bit ahead of it:

```bash
crontab -e
# 18:30 KST daily, 30 min before a 19:00 KST (10:00 UTC) digest
30 18 * * * /path/to/omnisummary/scripts/sync_all_to_s3.sh >> /tmp/omnisummary-sync.log 2>&1
```

`sync_all_to_s3.sh` defaults `AWS_PROFILE=research` and prepends the usual `uv` install dirs to `PATH`, since cron runs with a minimal one. The X sync needs the local RSSHub container up; the YouTube sync needs `YOUTUBE_API_KEY`. The two are independent, so RSSHub being down never blocks YouTube.

### External services

| Service | Purpose | Cost |
|---------|---------|------|
| AWS Bedrock | LLM (Claude Opus/Sonnet) | Usage-based |
| OpenAI | gpt-image-2 daily visual | Usage-based |
| Cloudflare Workers | HTTP proxy for Reddit/YouTube | Free (100K req/day) |
| Tavily | Web search | Free tier |
| Semantic Scholar | Paper search | Free |
| YouTube Data API v3 | Video metadata | Free (10K units/day) |
| Slack | Delivery + agent trigger | Free |
| Threads (Meta) | Delivery | Free |

## Project structure

```
omnisummary/
├── main.py                  # CLI entry point (digest pipeline)
├── research_cli.py          # Deep-research agent local runner
├── Dockerfile               # Lambda (amd64)
├── Dockerfile.agentcore     # AgentCore (arm64)
├── collectors/              # RSS, Reddit, RSSHub (X), YouTube, WebSearch + the S3 park loader
├── pipeline/                # Aggregator, Ranker, DigestGenerator, TrendTracker, DailyVisualMaker, runner (orchestration)
├── agent/                   # Deep-research agent + its 8 tools, VisualGenerator, DigestStateManager
├── agent_runtime/           # Bedrock AgentCore HTTP server
├── shared/                  # Config, models, prompts, state store, AgentCore memory, research, media
├── output/                  # Per-channel renderers + Slack & Threads handlers + delivery routing
├── lambda_handlers/         # digest, slack events, daily visual, threads refresh
├── infrastructure/          # CDK stacks (foundation + application)
├── scripts/                 # deploy, put_secrets, put_inference_profiles, ci_synth, syncs
├── cloudflare-proxy/        # CF Worker proxy
├── config/                  # YAML configuration
├── tests/                   # Unit + CDK assertion tests
└── docs/                    # design.md + diagrams/
```

## Testing & CI

```bash
uv run python -m pytest tests/ -v        # unit + CDK assertion tests (hermetic: no network, no AWS)
uv run black --check . && uv run ruff check . && uv run mypy .
uv lock --check                          # lockfile in sync with pyproject
uv run python scripts/ci_synth.py        # offline CDK synth
uv run pre-commit install                # once: runs CI's gates before you push
```

The suite is hermetic: `tests/conftest.py` clears the ambient secret and infra env vars and disables the SSM client, so results never depend on a developer's `.env` or AWS profile.

CI (`.github/workflows/ci.yml`) runs five jobs: **lint & type-check** (`uv lock --check`, ruff, black `--check`, `mypy .` over the whole repo), **CDK synth** offline through the pinned CLI against the *tracked* `config-template.yaml`, **tests & coverage** (scope and `fail_under` live in `pyproject.toml`), **Docker build & import check**, and **dependency & secret scan**. Every job carries a `timeout-minutes`, and the uv/npm caches are keyed on the lockfiles.

Two of those are worth explaining:

- **The import check** loads each built image and runs it with `--network none` and no credentials to import its real entry modules. Building alone never executes an import, so without this a missing `COPY` or an import-time AWS/HTTP call surfaces at cold start rather than in CI.
- **The dependency scan** audits the *installed* tree (`uv sync --frozen --no-dev --no-install-project`, then `pip-audit --strict --path .venv/...`) because that's the set the images install, already narrowed to the platform that ships. `--no-install-project` matters: pip-audit reports an editable distribution as unauditable and `--strict` turns that into a failure. `gitleaks` runs over **full history**, since a shallow clone only ever sees the tip and would never find a key committed earlier and removed later.

  Heads-up for operators: `pip-audit --strict` demands zero known advisories in the locked set, so a new advisory in a transitive dependency **will red an unrelated push**. The escalation order is `uv lock --upgrade` → remove unused dependencies → relax version caps, with `--ignore-vuln` as a last resort after confirming the code path is unreachable.

## Documentation

[docs/design.md](docs/design.md) is the line-by-line design and technical reference, and the place where every "why is it like this" is answered. Development guidelines and the load-bearing gotchas live in `.claude/CLAUDE.md`, which is local to a checkout rather than tracked.

## License

[MIT License](LICENSE)
