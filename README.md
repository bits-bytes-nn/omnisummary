<div align="center">

# 🗞️ OmniSummary

**A proactive AI/ML daily digest. It collects from five source families, picks the day's strongest stories with an LLM, writes a Korean editorial digest, and publishes it to Threads with a four-panel Korean comic of the headline. A local deep-research CLI (used by an OpenClaw skill) researches any topic across web, papers, and community, then returns a persona-voiced, cited report.**

Runs on AWS · Amazon Bedrock (Claude) · Bedrock AgentCore Memory.

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

**The daily digest** runs on a cron. It collects from RSS/Substack, Reddit, YouTube, X/Twitter and web search; deduplicates; scores everything with Claude Opus 5.5 into a candidate pool; and has Claude Sonnet 5.5 pick the strongest five stories and write a Korean editorial digest. The headline becomes a four-panel Korean comic, and the comic plus the stories go out together as one Threads post set.

**The deep-research CLI** runs locally. Given a free-form topic it researches the open web, academic papers, and community discussion on its own, then writes a cited Korean report in the same narrator voice as the digest. It prints one JSON object on stdout, which is what the OpenClaw skill consumes, and publishes to Threads only when you pass `--publish threads`.

![How the digest works](docs/diagrams/concept-pipeline.png)

## Features

- **Multi-source collection** — Reddit (one multi-subreddit public `.rss` request), YouTube, X/Twitter (via RSSHub), RSS/Substack, web search (Tavily). Feed bodies are HTML-stripped at collection
- **LLM ranking into a candidate pool** — Claude Opus 5.5 scores on multiple axes; source slots and per-origin caps make sure every source's best item reaches the editor, and the editor picks what ships
- **Editorial digest** — Claude Sonnet 5.5 writes Korean prose with cross-day trend continuity, then a grounding pass checks every specific claim against the sources
- **Daily comic** — a 2x2 four-panel comic narrated by the owner-persona character (kept on-model by a reference sheet), with Korean dialogue, rendered by `gpt-image-2.5-sunburst`
- **Threads delivery** — image root + flat reply chain, idempotent per day
- **Comment replies** — every hour Claude Sonnet 5.5 answers readers' new comments on the last week's posts (questions, objections, experiences) in the comment's language, and ignores spam and meaningless ones (bare praise, emoji, bot-style generic remarks)
- **Deep-research CLI + OpenClaw skill** — an autonomous Strands agent with eight single-purpose tools; JSON contract on stdout, Markdown report on disk
- **Operational excellence** — per-source health reporting (`OK` / `EMPTY` / `FAILED` / `STALE` / `DEGRADED`) into SNS email alerts, structured JSON logs with correlation IDs, 10 CloudWatch alarms
- **AWS deployment** — Lambda + EventBridge cron + Bedrock AgentCore Memory, all in CDK

## Quick Start

### Prerequisites

- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- Docker (for RSSHub and for AWS deployment)
- An AWS account with Bedrock access

### Installation

```bash
git clone <repo-url> && cd omnisummary
uv sync
cp config/config-template.yaml config/config.yaml
cp .env.template .env
```

### Configuration

`config/config.yaml` holds everything that isn't a secret. Every config model rejects unknown keys, so a key removed from the code must also be removed from your `config.yaml`. The interesting knobs:

```yaml
collectors:
  rss:
    enabled: true
    feeds: ["https://feeds.feedburner.com/geeknews-feed"]
  reddit:
    enabled: true
    subreddits: [LocalLLaMA, MachineLearning]
  youtube:
    enabled: true
    channels: ["https://www.youtube.com/@AndrejKarpathy"]
    lookback_hours: 30      # must reach back to the previous run: config rejects less than 48 - the run hour
  web_search:
    enabled: true
    exclude_domains: ["finance.biggo.com"]   # never returned by any search, broad or refined
    trend_searches:
      - name: frontier_models
        queries: ["frontier AI model release GPT Claude Gemini Llama"]
  rsshub:
    enabled: true
    base_url: "http://localhost:1200"
    accounts:
      - username: "karpathy"
        platform: x

pipeline:
  top_n: 5
  min_score: 0.6
  ranking_model: "anthropic.claude-opus-5-5"
  digest_model: "anthropic.claude-sonnet-5-5"
  max_per_origin: 1        # cap per channel / author / subreddit / feed / web host
  source_slots: {web: 1, x: 1, rss: 1, reddit: 1, youtube: 1}
```

[design.md §3](docs/design.md#3-설정) explains the fields.

### Secrets

Secrets go in `.env` locally (see `.env.template`) and in SSM Parameter Store on AWS. Only Tavily is needed to run at all; everything else enables one specific feature and degrades gracefully: the feature logs a line and is skipped when its key is absent.

| Variable | Needed for |
|----------|-----------|
| `TAVILY_API_KEY` | The `web_search` collector + the research CLI's web/community/news search |
| `YOUTUBE_API_KEY` | YouTube collector (without it: the channel RSS feeds instead of the Data API) |
| `OPENAI_API_KEY` | The daily comic render |
| `THREADS_ACCESS_TOKEN` / `THREADS_USER_ID` | Threads delivery (60-day token, auto-refreshed into SSM on AWS) |
| `ALERT_EMAIL` | SNS email alerts (AWS). Required to deploy: a deploy without it removes the subscription |
| `S3_SYNC_ACCESS_KEY_ID` / `S3_SYNC_SECRET_ACCESS_KEY` | Optional dedicated creds for the local→S3 sync (otherwise `aws.profile` from the config) |

> **Secrets never pass through the CDK stack.** A CloudFormation template can't hold a SecureString, so
> handing values to the stack would publish them in plaintext into `cdk.out`, the staging bucket, and
> every `GetTemplate` response. The stack creates only the parameter *paths* holding a placeholder;
> `scripts/put_secrets.py` writes the real values as SecureStrings after the deploy. See
> [Secrets on AWS](#secrets-on-aws).

### Setup checklist

To produce a digest **locally** (no X, no comic, nothing published):

1. `uv sync`, then copy the two template files as above.
2. Fill `TAVILY_API_KEY` in `.env`.
3. Make sure Bedrock is reachable in `aws.bedrock_region` (default `us-west-2`) and set `AWS_PROFILE` or standard AWS credentials. The ranking and digest LLMs run on Bedrock even for a local run.
4. `uv run python main.py --dry-run --sources web_search` prints the digest. The template ships no RSS feeds or subreddits, but it does carry the trend searches, so Tavily is the one source that works out of the box.

Then add capabilities one at a time:

| Want… | Set | Notes |
|-------|-----|-------|
| YouTube items **with transcripts** | The local sync (`YOUTUBE_API_KEY` optional) | Transcripts only fetch from a residential IP |
| X/Twitter items | RSSHub container + the local sync | See below |
| Daily comic | `OPENAI_API_KEY` | `gpt-image-2.5-sunburst` |
| Threads delivery | `THREADS_ACCESS_TOKEN`, `THREADS_USER_ID`, `enable_threads_post: true` | The daily-visual Lambda is what posts to Threads, because the image and the text ship as one post set; the post goes out text-only when the comic can't be rendered |
| Comment replies | The Threads credentials + `enable_comment_replies: true` | `uv run python scripts/sample_comment_replies.py` drafts what it would post now, posting nothing |

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

Without the cookies the container still starts, but X feeds come back empty. Cookies expire every so often. When the RSSHub failure rate climbs (it's logged as a warning), refresh them and recreate the container. `--restart unless-stopped` does not bring back a container that was stopped explicitly, so the sync script starts it if `localhost:1200` doesn't answer.

## Usage

```bash
# Digest pipeline, nothing persisted or published
uv run python main.py --dry-run --sources rss reddit

# Full pipeline + comic + Threads (when enabled)
uv run python main.py

# Deep research: one JSON object on stdout, the Markdown report saved under reports/
uv run python research_cli.py "<topic>"
uv run python research_cli.py "<topic>" --publish threads   # also publish a Threads thread

# What the digest already published: free, read-only, one JSON object on stdout
uv run python recall_cli.py digest 2026-10-11   # that day's lead and stories (default: today)
uv run python recall_cli.py trends              # tracked cross-day trends, strongest first

# Local→S3 sync for the sources that block datacenter IPs (X/RSSHub + YouTube transcripts)
./scripts/sync_all_to_s3.sh                  # both; one failing won't block the other
uv run python scripts/sync_to_s3.py rsshub   # X only
uv run python scripts/sync_to_s3.py youtube  # YouTube only
```

| Flag (`main.py`) | Description |
|------|-------------|
| `--sources rss reddit youtube` | Select specific sources |
| `--dry-run` | Print the digest; persist and publish nothing |
| `--top-n 5` | Override how many stories to publish |
| `--date 2026-03-28` | Set the digest date (default: today, KST) |
| `--pin-url <url> [<url> ...]` | Force URL(s) into the top stories regardless of score. YouTube URLs resolve via the Data API, others via Tavily. Local CLI only |
| `--force-republish` | Re-post today's digest even if it already went out (bypasses the Threads idempotency guard) |

## How the pipeline works

A summary of each stage. [design.md](docs/design.md) is the line-by-line reference and explains *why* each piece is shaped the way it is.

**1. Collection.** Every collector runs async in parallel with its own lookback window, and reports its own health. Two sources are special: X/Twitter and YouTube transcripts are blocked from datacenter IPs, so a local cron collects them on a residential IP, parks them in S3, and the Lambda reads the parked file. A park file older than its age budget makes the source report `STALE`, so a stopped cron can't look healthy.

| Collector | Source | Method |
|-----------|--------|--------|
| `RedditCollector` | Reddit public `.rss` | One `r/A+B` request (`t=day` for top), fetched directly with retry |
| `YouTubeCollector` | YouTube Data API v3 | S3 park file on AWS, live otherwise; reads the long-form uploads (`UULF`) so Shorts don't take the channel's slot |
| `RSSCollector` | RSS/Atom | httpx + feedparser, HTML stripped; a short summary-only entry gets its page text via Tavily extract |
| `RSSHubCollector` | X/Twitter via RSSHub | S3 park file on AWS, local Docker otherwise |
| `WebSearchCollector` | Tavily | Direct, with LLM query refinement and `exclude_domains` for content farms |

**2. Aggregation.** `ContentAggregator` deduplicates by URL and by normalized title (a leading `RT name:` is stripped, so a self-retweet matches its original). When two items collide it keeps the *better* one (pinned > the original over its retweet > longer body > first seen) rather than whichever arrived first.

**3. Ranking.** `ContentRanker` scores items with Claude Opus 5.5 on technical substance, practitioner value, novelty, industry impact, research significance and source authority, with hard filters for promos and thin content. It also sees the story titles recent digests ran, so a story that comes back under a new URL scores as a rehash unless it reports a real new development. It then builds a candidate pool of `top_n + digest_candidate_buffer` (5 + 7): `source_slots` seat every source's best item above `min_score`, the rest of the pool goes by score with `max_per_origin` capping any single channel/author/feed/host, and that cap is relaxed only when a quiet day would leave the pool short.

**4. Trend tracking.** `TrendTracker` keeps structured trends in `trends.json`. The LLM only classifies today's items into existing or new trends; all bookkeeping is deterministic Python: date stamping, the active/cooling lifecycle, deletion past retention, recency-decay momentum, evidence caps. Active and cooling trends feed the next day's digest, which is what gives it cross-day continuity. Only evidence dated before the digest date counts, so a same-day re-run never reads its own first run back as history. Memory is compressed by caps, not rewritten: at most five dated evidence lines per trend, the ten highest-momentum trends in the prompt, and a trend unseen for 30 days is dropped.

**5. Digest generation.** `DigestGenerator` has the editor pick the strongest five distinct stories from the pool (merging same-event items) and produce a structured `DigestContent` (a `lead`, plus `items[]` each with title/url/body/implication). It is shown the story titles of every digest in the dedup window and the closing (thesis) sentence of recent leads, so it neither re-tells a story a weekly roundup brought back under a new URL nor repeats an argument. The LLM writes **only prose**, with no markup and no source tags. Prose budgets are computed in code from the parts code owns, so an item can't overflow the Threads 500-character limit.

> ⚠️ **The JSON key order in the prompt is load-bearing.** It asks for `items` first and `lead` last, so the lead comments on stories that are already written. Measured word overlap with the headline reply dropped from 0.21–0.41 to 0.03–0.21. Do **not** reorder the requested keys to match `DigestContent`'s field order. That tidy-up is a regression.

**6. Threads rendering.** `output/renderers.py` turns one `DigestContent` into a Threads root + flat reply chain. Formatting lives in code, not in prompt rules. The root is the lead followed by the other stories' titles, since the story replies are read by only 2-5% of the root's viewers. It carries one topic tag (`pipeline.threads_topic_tag`, default `AI`) so it shows up in that topic's feed and tag search, and the comic carries Korean alt text describing each panel and its line.

**7. Daily comic.** `DailyVisualMaker` illustrates the **headline** story, so the image, the lead, and the text all point at the same thing. The editor, given the headline article and the last few days' comics so today's differs in gag and props, writes a 2x2 four-panel comic that carries one fact, drawn as literal scenes where the story happens (setup, development, turn, payoff: the story's own twist as a sight gag, not a moral) narrated by the owner-persona character, with one short Korean line per panel; `VisualGenerator` turns it into an image brief, renders it with `gpt-image-2.5-sunburst` at xhigh (1024x1536 for the 2x2 grid; another layout, such as a single frame or a tall 848x2544 strip, when the story needs it) (passing the persona reference sheet as an input image whenever the narrator appears, which is the default). The image ships as drawn. This Lambda also owns the Threads post, and a failed render never swallows the digest. The text still goes out.

**8. Deep research.** An autonomous Strands agent run by `research_cli.py`. It composes these eight tools freely. For example, "diffusion LLM 최신 동향" goes `web_search`/`search_papers`/`community_search` → `read_url` → `attach_image` → `deliver_report`:

| Tool | Function |
|------|----------|
| `web_search(query, news, time_range)` | Tavily open web; `news=True` for news coverage, `time_range` day/week/month/year; results show their published date |
| `community_search(query)` | Tavily over Reddit, X, HN, Substack |
| `search_papers(query)` | OpenAlex works search (keyless): date, citation count, abstract |
| `read_url(url)` | Fetch and extract a primary source's full text |
| `recall_trends()` | Every tracked trend in `trends.json`, momentum-ranked, for the agent to judge relevance |
| `recall_digest(digest_date)` | What one specific day's digest carried. Never falls back to another date |
| `attach_image(source_url)` | Download a source's OG image and stage it for delivery |
| `deliver_report(report, channel)` | Hand over the Markdown report (`markdown`), or publish to Threads (`threads`, only with `--publish threads`) |

`deliver_report` refuses a report that cites a URL no tool returned, and names the offending URLs so the agent can fix them.

## Deep research with OpenClaw

The OpenClaw skill lives at `~/.openclaw/workspace/skills/omnisummary-research/SKILL.md` (the workspace is the canonical place for user-defined skills, so the repo holds no copy). It calls the CLI and relays the result:

```bash
uv --directory ~/Projects/omnisummary run python research_cli.py "Mixture-of-Experts 추론 비용 — 최근 반년 동향"
```

```json
{
  "status": "succeeded",
  "topic": "Mixture-of-Experts 추론 비용 — 최근 반년 동향",
  "title": "…",
  "markdown_path": "/…/omnisummary/reports/2026-10-11-180202-f1a02d61.md",
  "markdown": "# …",
  "sources": ["https://…"],
  "images": [],
  "threads": null,
  "model": "anthropic.claude-opus-5-5",
  "stop_reason": "end_turn",
  "usage": {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_write_input_tokens": 0, "cycles": 0, "tool_calls": {}},
  "warnings": [],
  "error": null
}
```

Exit codes: `0` succeeded, `1` failed (no report, a run error, or a requested Threads publish that did not complete), `2` rejected input. The same skill reads back past digests with `recall_cli.py` (`digest [DATE]`, `trends`), which costs nothing: `found: false` means nothing is stored for that date, while `status: failed` means the store could not be read. Logs go to stderr, and third-party loggers stay at `WARNING` regardless of `LOG_LEVEL`, because botocore at DEBUG logs request headers that carry the session token.

## AWS Deployment

### Deploying

Build and push the image first (see [Docker image](#docker-image)), then deploy pinning the pushed digest. CloudFormation won't redeploy a Lambda when the image *tag* string is unchanged, so pass the `sha256` digest explicitly.

Use the repo-pinned CDK CLI, not a global `cdk`. The CLI is pinned in `package.json` to a version compatible with the `aws-cdk-lib` in `pyproject.toml`, and a global one can lag the library and fail with a cloud-assembly schema mismatch.

The **order matters on a fresh account**, because the foundation stack owns the only ECR repository and the application stack's Lambdas resolve their image out of it.

```bash
npm install                                       # once: installs the pinned CDK CLI
export AWS_PROFILE=<profile>

# 1. Once per account+region: create the CDK bootstrap resources (staging bucket, roles).
npx cdk bootstrap -a "uv run python scripts/deploy.py"

# 2. Foundation FIRST — it creates the ECR repo the image is pushed to.
npx cdk deploy '*-foundation' -a "uv run python scripts/deploy.py"

# 3. Log in to that repo and push the image (see Docker image below).
#    The URI is derived, not looked up: <account>.dkr.ecr.<region>.amazonaws.com/<project>-<stage>-agent
ECR_URI="$(aws sts get-caller-identity --query Account --output text).dkr.ecr.<region>.amazonaws.com/omnisummary-<stage>-agent"
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin "${ECR_URI%%/*}"
docker build --platform linux/amd64 --provenance=false -t "$ECR_URI:latest" . && docker push "$ECR_URI:latest"

# 4. Deploy everything, pinning the pushed digest.
export DIGEST_IMAGE_REF=$(aws ecr describe-images --region <region> --repository-name omnisummary-<stage>-agent \
  --image-ids imageTag=latest --query 'imageDetails[0].imageDigest' --output text)
npx cdk deploy --all -a "uv run python scripts/deploy.py"

# 5. Secrets and cost attribution.
uv run python scripts/put_secrets.py             # then write the secrets
uv run python scripts/put_secrets.py --verify    # read-only: any left unset?
uv run python scripts/put_inference_profiles.py  # once per account/stage, and again after changing a stage's model
```

On every later deploy only steps 3-4 apply: push the new image, then `deploy --all` with the fresh digest.

### What gets created

| Resource | Purpose |
|----------|---------|
| **Lambda** (Docker) | Digest pipeline, 15 min timeout. X items come from the S3 park file the local sync writes |
| **Lambda** (Docker) | Daily comic + Threads publish, 15 min timeout. Async, off the digest critical path, and the only digest publish path |
| **Lambda** (Docker) | Threads token refresh (weekly, `threads_token_refresh_days`; writes the renewed 60-day token back to SSM) |
| **Lambda** (Docker) | Comment replies, 10 min timeout, one run at a time, every `aws.comment_reply_interval_minutes` (60). Does nothing unless `pipeline.enable_comment_replies` is on |
| **EventBridge** | Daily digest cron (config-driven hour/minute, **UTC**) + the token-refresh and comment-reply schedules |
| **Bedrock AgentCore** | Memory (digest snapshots, one session per date) |
| **SSM Parameter Store** | All secrets, as SecureStrings written out-of-band |
| **S3** | Trends + park files + Threads image hosting. The role gets `GetObject`/`PutObject` on the project prefix plus `ListBucket`, nothing else |
| **SQS** | Async DLQ for the digest and visual Lambdas. Every Lambda runs `retry_attempts=0`, because Threads has no idempotency key and a retry would double-post. The handlers re-raise, so digest and visual failures land here for replay; the token-refresh and comment-reply Lambdas have no DLQ and raise into their Errors alarms |
| **SNS** | Alert topic (email) |
| **CloudWatch** | Structured logs, one-month retention, and 10 alarms (per-Lambda Errors ×4 + Timeout ×4, EmptyDigest, async DLQ) |
| **ECR** | The Lambda image (amd64) |

### Secrets on AWS

The stack creates each SSM parameter holding a placeholder; `scripts/put_secrets.py` writes the real values from your `.env` as SecureStrings after the deploy. Re-deploys don't clobber them, because CloudFormation only updates a resource whose template properties changed and the placeholder never changes.

Four behaviours worth knowing:

- **Parameters that are already SecureStrings are skipped.** The Threads token is rotated in place by the refresh Lambda, so re-asserting the local `.env` copy would restore an expired token. Use `--force` only when you mean to overwrite the live value.
- **A missing or empty environment variable is skipped, never blanked**, so a partial `.env` can't wipe a working parameter. And `resolve_secret()` treats a parameter still holding the placeholder as unset, so forgetting to run `put_secrets.py` degrades to the normal missing-credential path instead of sending the placeholder to an API as a token.
- **One parameter SSM refuses does not abort the run.** Failures are listed under `FAILED` and the script exits non-zero, but every other secret still gets written.
- **`--verify` is a read-only report** of which parameters are set, which still hold the placeholder, which are plaintext `String`, and which are missing. Safe to run against prod any time.

The X session cookies stay local: only the RSSHub container on your machine uses them, and nothing in AWS runs RSSHub.

### Bedrock models and cost attribution

| Stage | Model |
|-------|-------|
| Ranking | Claude Opus 5.5 |
| Digest, grounding, comic editor and art director, comment replies | Claude Sonnet 5.5 |
| Deep research | Claude Opus 5.5 |
| Trends, query refinement | Claude Sonnet 5.5 |
| Token counting | Claude Sonnet 4.6 (the Opus models don't expose CountTokens) |

Opus 5.5 defaults to effort `medium`, so the factory pins it to `high`. Sonnet 5.5 writes the digest because it keeps the length spec (a 3-4 sentence lead) that Opus 5.5 overran, at under half the cost. Claude Fable 5.1 is not registered: Bedrock rejects it under the account's default data-retention mode. The per-stage choices were measured on real pools; see [design.md §6](docs/design.md#모델-팩토리).

On-demand Bedrock bills against no taggable resource, so `InvokeModel` token spend can't carry a cost-allocation tag. An **application inference profile** *is* taggable, and invoking through its ARN attributes the usage. `scripts/put_inference_profiles.py` creates one per configured model, named and tagged from `config.aws.project_name`/`stage` and copied from the system-defined cross-region profile. `BedrockCrossRegionModelHelper` prefers them at resolution time, for both the LangChain factory and the research agent. A missing profile or a denied lookup silently keeps the system-defined id: cost reporting must never stop a generation.

Two things to watch:

- `application-inference-profile` is a **different IAM resource type** from `inference-profile`. The policy grants both; drop the former and every Bedrock call becomes AccessDenied the moment a profile exists.
- Activate the `Project` cost-allocation tag in Billing for this to reach Cost Explorer. It takes up to 24 h and is not retroactive.

Complementing this, every `get_model()` call takes a `stage=` and logs `LLM usage stage=... model=... input=... output=...`, because the bill is per *model* while several stages share one. The research CLI reports the same token counts in its JSON `usage`.

### Docker image

The image installs the **exact set `uv.lock` pins** (`uv export` → `uv pip install --system`, the project itself `--no-deps`), so it can never run a dependency set CI never tested. Dependencies install before the source is copied, so a code-only change reuses that layer. It runs **non-root** (uid 10001), and `.dockerignore` keeps `.env`, `.venv`, `logs/` and `cdk.out` out of the build context. All four Lambdas use it.

It goes to the ECR repository the **foundation stack** creates, `<account>.dkr.ecr.<region>.amazonaws.com/<project>-<stage>-agent`, so log in to it first (the Docker credential is short-lived; re-run the login when a push 401s).

```bash
aws ecr get-login-password --region <region> \
  | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com

docker build --platform linux/amd64 --provenance=false -t <ecr-uri>:latest .
docker push <ecr-uri>:latest
```

### Local cron

X/Twitter and YouTube transcripts both block datacenter IPs, so they must be collected locally and synced to S3 *before* the AWS digest runs. The digest cron is `aws.digest_cron_hour`/`minute` interpreted as **UTC**, so the default `10:00 UTC` is `19:00 KST`. Schedule the sync a bit ahead of it:

```bash
crontab -e
# 18:30 KST daily, 30 min before a 19:00 KST (10:00 UTC) digest
30 18 * * * /path/to/omnisummary/scripts/sync_all_to_s3.sh >> /tmp/omnisummary-sync.log 2>&1
```

`sync_all_to_s3.sh` defaults `AWS_PROFILE=research`, prepends the usual `uv` install dirs to `PATH` (cron runs with a minimal one), and starts the `rsshub` container if `localhost:1200` doesn't answer. The X sync needs RSSHub. The two are independent, so RSSHub being down never blocks YouTube.

### External services

| Service | Purpose | Cost |
|---------|---------|------|
| AWS Bedrock | LLM (Claude Opus/Sonnet) | Usage-based |
| OpenAI | `gpt-image-2.5-sunburst` daily comic at xhigh: one render a day | Usage-based |
| Tavily | Web search | Free tier |
| OpenAlex | Paper search | Free (keyless, about 100 searches a day) |
| YouTube Data API v3 | Video metadata | Free (10K units/day) |
| Threads (Meta) | Delivery | Free |

## Project structure

```
omnisummary/
├── main.py                  # CLI entry point (digest pipeline)
├── research_cli.py          # Deep-research CLI (JSON on stdout), used by the OpenClaw skill
├── recall_cli.py            # Read back past digests and trends (JSON on stdout), used by the OpenClaw skill
├── Dockerfile               # Lambda image (amd64)
├── collectors/              # RSS, Reddit, RSSHub (X), YouTube, WebSearch + the S3 park loader
├── pipeline/                # Aggregator, Ranker, DigestGenerator, TrendTracker, DailyVisualMaker, CommentResponder, runner (orchestration)
├── agent/                   # Deep-research agent + its 8 tools, VisualGenerator, DigestStateManager
├── shared/                  # Config, models, prompts, state store, AgentCore memory, history, research, media
├── output/                  # Threads renderers + handler, digest publisher, research report delivery
├── lambda_handlers/         # digest, daily visual, threads refresh, comment replies
├── infrastructure/          # CDK stacks (foundation + application)
├── scripts/                 # deploy, put_secrets, put_inference_profiles, ci_synth, sync_to_s3 + sync_all_to_s3.sh, samplers, render_excalidraw.mjs
├── assets/                  # persona.jpg (the narrator's reference sheet)
├── config/                  # YAML configuration
├── tests/                   # Unit + CDK assertion tests
└── docs/                    # design.md + diagrams/
```

## Testing & CI

```bash
uv run python -m pytest tests/ -v        # unit + CDK assertion tests (hermetic: no network, no AWS)
uv run pre-commit run --all-files        # ruff, black, file checks, uv lock --check
uv run mypy .
uv run python scripts/ci_synth.py        # offline CDK synth
uv run pre-commit install                # once: runs the gates before you commit
```

The suite is hermetic: `tests/conftest.py` clears the ambient secret and infra env vars and disables the SSM client, so results never depend on a developer's `.env` or AWS profile.

CI (`.github/workflows/ci.yml`) runs five jobs on push, PR and a weekly schedule: **lint & type-check** (`uv lock --check`, ruff, black `--check`, `mypy .` over the whole repo), **CDK synth** offline through the pinned CLI against the *tracked* `config-template.yaml`, **tests & coverage** (scope and `fail_under` live in `pyproject.toml`), **Docker build & import check**, and **dependency & secret scan**. Every job carries a `timeout-minutes`, the uv/npm caches are keyed on the lockfiles, third-party actions are pinned to commit SHAs, and checkouts don't persist credentials. A separate workflow (`workflow-lint.yml`) runs a checksum-verified actionlint (with shellcheck) over the workflows, and Dependabot updates the pinned actions weekly. uv is pinned to 0.6.14 in CI and the Dockerfile; regenerate the lockfile with `uvx --from uv==0.6.14 uv lock` so its format stays readable by that version.

Two of those are worth explaining:

- **The import check** loads the built image and runs it with `--network none` and no credentials to import its real entry modules and load the config. Building alone never executes an import, so without this a missing `COPY` or an import-time AWS/HTTP call surfaces at cold start rather than in CI.
- **The dependency scan** audits the *installed* tree (`uv sync --frozen --no-dev --no-install-project`, then `pip-audit --strict --path .venv/...`) because that's the set the image installs. `--no-install-project` matters: pip-audit reports an editable distribution as unauditable and `--strict` turns that into a failure. The weekly run is what catches a new advisory against an unchanged lockfile. `gitleaks` scans with full history fetched and gets `GITHUB_TOKEN`, which it needs on pull requests.

  Heads-up for operators: `pip-audit --strict` demands zero known advisories in the locked set, so a new advisory in a transitive dependency **will red the build**. The escalation order is `uv lock --upgrade-package <pkg>` → remove unused dependencies → relax version caps, with `--ignore-vuln` as a last resort after confirming the code path is unreachable.

## Documentation

[docs/design.md](docs/design.md) is the line-by-line design and technical reference, and the place where every "why is it like this" is answered.

## License

[MIT License](LICENSE)
