<div align="center">

# 🗞️ OmniSummary

**An AI/ML daily digest in Korean, published to Threads with a four-panel comic, plus a deep-research CLI that writes cited reports on any topic.**

Runs on AWS · Amazon Bedrock (Claude, GPT) · Bedrock AgentCore Memory

[![CI](https://github.com/bits-bytes-nn/omnisummary/actions/workflows/ci.yml/badge.svg)](https://github.com/bits-bytes-nn/omnisummary/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![AWS CDK](https://img.shields.io/badge/IaC-AWS%20CDK-orange)
![Bedrock](https://img.shields.io/badge/LLM-Amazon%20Bedrock-green)

🇰🇷 [한국어 README](./README.ko.md) · 🛠️ [Operations guide (Korean)](./docs/operations.md) · 📐 [Design doc](./docs/design.md)

![OmniSummary architecture](docs/diagrams/architecture.png)

</div>

---

## What it does

OmniSummary has two independent parts.

- **Daily digest (AWS, on a schedule).** It collects AI news from RSS/Substack, Reddit, YouTube, X and web search, scores every item, and lets an editor model pick the five best stories. It then writes a Korean digest and draws the headline as a four-panel Korean comic. Everything goes out as one Threads thread: the root carries the comic, the lead and the other stories' titles, then one reply per story.
- **Deep research (local CLI).** Give it a topic and an agent searches the web, papers and community discussion, reads the primary sources, and writes a cited Korean report in the digest's voice. It prints one JSON object (the contract an OpenClaw skill reads) and publishes to Threads only with `--publish threads`.

When turned on, an hourly job also answers readers' comments on the last week's posts, and declines spam and empty praise. A reply carrying a link, a domain, another account's handle or a hashtag is never posted, so a comment cannot make the account say what its writer wants.

![How the digest works](docs/diagrams/concept-pipeline.png)

## Quick start

You need Python 3.12+ with [uv](https://docs.astral.sh/uv/), an AWS account with Bedrock access, and Docker (only for X/Twitter and for deploying).

```bash
git clone <repo-url> && cd omnisummary
uv sync --extra dev        # the dev extra adds the test, lint, type-check and CDK tools
cp config/config-template.yaml config/config.yaml
cp .env.template .env      # then set TAVILY_API_KEY
```

Make sure your AWS credentials reach Bedrock in `aws.bedrock_region` (default `us-west-2`). Every LLM stage runs on Bedrock, even locally. Then print a digest without saving or posting anything:

```bash
uv run python main.py --dry-run --sources web_search
```

The template ships no feeds or subreddits but does carry trend searches, so web search is the one source that works out of the box. Add the rest one at a time:

| To get | Set | Notes |
|--------|-----|-------|
| RSS, Reddit, YouTube | Feeds, subreddits, channels in `config.yaml` | `YOUTUBE_API_KEY` is optional (channel RSS otherwise) |
| YouTube transcripts, X posts | The local sync below | Both block datacenter IPs |
| The daily comic | `OPENAI_API_KEY` | Rendered by `gpt-image-2.5-sunburst` |
| Threads posting | `THREADS_ACCESS_TOKEN`, `THREADS_USER_ID`, `pipeline.enable_threads_post: true` | Without a comic the post goes out text-only |
| Comment replies | The Threads credentials + `pipeline.enable_comment_replies: true` | `scripts/sample_comment_replies.py` drafts replies without posting |

## Usage

```bash
# Digest
uv run python main.py --dry-run --sources rss reddit    # print only
uv run python main.py                                   # full run: save, draw, post (when enabled)

# Deep research: one JSON object on stdout, the report under reports/
uv run python research_cli.py "<topic>"
uv run python research_cli.py "<topic>" --publish threads

# What the digest already published (free, read-only, JSON)
uv run python recall_cli.py digest 2026-10-26
uv run python recall_cli.py trends

# Local → S3 sync for X and YouTube transcripts
./scripts/sync_all_to_s3.sh
```

| `main.py` flag | Effect |
|----------------|--------|
| `--sources rss reddit youtube web_search rsshub` | Collect from these sources only (`rsshub` is X) |
| `--dry-run` | Print the digest; save and post nothing |
| `--top-n 5` | Number of stories |
| `--date 2026-03-28` | Digest date (default: today in `aws.timezone`, KST by default) |
| `--pin-url <url> [...]` | Force these URLs into the stories |
| `--force-republish` | Post again for a date that already went out |

With Threads posting on, a run for a date that is already on Threads stops before collecting, unless it is a dry run or `--force-republish`. A future `--date` and an unknown source name are refused. `main.py` exits 1 when it collected nothing, produced no digest, or the Threads post did not fully land (locally there is no alert to say so).

> A full local run writes to production state: the day's snapshot goes to the deployed AgentCore Memory. Its ledgers live in the state bucket: `STATE_BUCKET` when that is set, else `aws.state_bucket_name`; with neither they are local files that can't see what production posted, and Threads may get the day twice. If the stack created its own bucket (`aws.state_bucket_name` was empty at the first deploy), point local runs at it with `STATE_BUCKET=<project>-<stage>-state` in `.env` (plus `S3_PREFIX=<s3_prefix>/digest_state` when `aws.s3_prefix` is set), never by setting `aws.state_bucket_name`: on a non-prod stage the next deploy would delete that bucket.

## Configuration

`config/config.yaml` holds everything that isn't a secret. Every config section rejects unknown keys, so remove a key from your `config.yaml` when the code drops it. The main knobs:

```yaml
pipeline:
  top_n: 5
  min_score: 0.6
  ranking_model: "anthropic.claude-opus-5-5"
  digest_model: "anthropic.claude-sonnet-5-5"   # also the comic editor, grounding, comment replies
  trend_model: "anthropic.claude-sonnet-5-5"
  source_slots: {web: 1, x: 1, rss: 1, reddit: 1, youtube: 1}
  max_per_origin: 1
agent:
  model_id: "anthropic.claude-opus-5-5"         # deep research
```

[design.md §3](docs/design.md#3-설정) describes every field.

### Models

Every LLM stage takes a Claude or a GPT model, all served by Bedrock through the same Converse API, the same IAM role and the same cost-attribution profiles. Set the stage's `*_model` (or `agent.model_id`) to one of the ids in `shared/constants.py`:

| Family | Ids |
|--------|-----|
| Claude | `anthropic.claude-opus-5-5`, `anthropic.claude-sonnet-5-5`, `anthropic.claude-haiku-5-5`, and older |
| GPT | `openai.gpt-6.1-sol`, `openai.gpt-6.1-terra` |

The defaults were measured on real pools:

| Stage | Default | Why |
|-------|---------|-----|
| Ranking | Claude Opus 5.5 | Picks within the usual model-to-model spread of Opus 4.8, slightly cheaper |
| Digest, grounding, comic editor, comment replies, research fact check | Claude Sonnet 5.5 | Keeps the length spec Opus overran, at under half the cost |
| Trends, search-query refinement | Claude Sonnet 5.5 | |
| Deep research | Claude Opus 5.5 | |

GPT-6.1 Sol was compared with Sonnet 5.5 as the digest model on six stored days (2026-10-21 to 10-26), judged blind on context, clarity, accuracy, insight, fit within the Threads limit, and Korean. Sonnet scored 137 against 133, won four days of six, and only GPT added a claim its source did not support. Sonnet stays the default; GPT is a config change away.

The registry in `shared/utils.py` records what each model accepts: no `temperature` (only Sonnet 4.6, used for token counting, takes one), effort pinned to `high` where a model defaults lower (Claude through `output_config`, GPT through `reasoning`), and forced tool choice where it works. The research agent uses Converse prompt caching for Claude only; GPT caches on its own. Claude Fable 5.1 is not registered, because Bedrock rejects it under the account's default data-retention mode.

### Secrets

Secrets live in `.env` locally; on AWS the API keys and Threads credentials live in SSM Parameter Store (`ALERT_EMAIL` and the sync keys stay in `.env`). Only Tavily is required; every other key turns on one feature, which logs a line and is skipped when its key is missing.

| Variable | Turns on |
|----------|----------|
| `TAVILY_API_KEY` | Web search, and the research CLI's searches |
| `YOUTUBE_API_KEY` | The YouTube Data API (channel RSS otherwise) |
| `OPENAI_API_KEY` | The daily comic |
| `THREADS_ACCESS_TOKEN`, `THREADS_USER_ID` | Threads posting (the token is refreshed weekly into SSM on AWS) |
| `ALERT_EMAIL` | Email alerts. Required to deploy: a deploy without it removes the subscription |
| `S3_SYNC_ACCESS_KEY_ID`, `S3_SYNC_SECRET_ACCESS_KEY` | Optional dedicated keys for the local sync |

### X/Twitter through RSSHub

X is read through a local [RSSHub](https://docs.rsshub.app/) container. It needs the `auth_token` and `ct0` cookies of a logged-in x.com session (browser devtools → Application/Storage → Cookies → x.com).

```bash
docker run -d --name rsshub --restart unless-stopped -p 1200:1200 \
  -e NODE_ENV=production -e CACHE_TYPE=memory \
  -e TWITTER_AUTH_TOKEN='<auth_token>' -e TWITTER_CT0='<ct0>' \
  diygod/rsshub:latest
curl -s "http://localhost:1200/twitter/user/karpathy" | head   # smoke test
```

Without the cookies the feeds come back empty. When the RSSHub failure rate climbs, refresh the cookies and recreate the container. The sync script starts the container if nothing answers on port 1200.

## How the digest is made

[design.md](docs/design.md) is the line-by-line reference, including why each piece is shaped the way it is.

1. **Collect.** Each collector runs in parallel with its own time window and reports its health (`OK`, `EMPTY`, `FAILED`, `STALE`, `DEGRADED`). X and YouTube transcripts are collected on your machine and parked in S3, and the Lambda reads the parked file. The sync parks a wider window than the digest reads, and the reader reaches back to the healthy sync whose file the last digest read, so one skipped, late or degraded sync loses nothing (two in a row lose the earlier day).
2. **Deduplicate.** Items are matched by normalized URL (share-link variants such as `youtu.be` or `twitter.com` count as the same item) and by title. The better copy wins: a pinned one, the original over its retweet, the longer body.
3. **Rank.** Claude Opus 5.5 scores each item on technical substance, practitioner value, novelty, impact, research significance and source authority, and filters promos and thin posts. It sees recent digest titles, so a story back under a new URL scores as a rehash. Up to 12 candidates above `min_score` (`top_n` + `digest_candidate_buffer`) go on: each source's best item gets a seat, and any one channel, author or site gets at most one.
4. **Track trends.** The model only sorts today's items into existing or new trends. Dates, lifecycle, momentum and caps are plain Python, kept in `trends.json`, and the active trends give the next day's lead its continuity.
5. **Write.** The editor picks the five strongest distinct stories and writes a lead plus a title, body and takeaway for each, as plain prose. A grounding pass then checks every specific claim against the source text. Character budgets are computed in code, so no post overflows Threads' 500-character limit.
6. **Render.** Code, not the prompt, lays out the posts: the root is the lead plus the other stories' titles (replies get 2-5% of the root's views), one reply per story, and a topic tag.
7. **Draw.** A separate Lambda draws the headline as a 2x2 comic: one fact, literal scenes where the story happens, and the story's own twist as a sight gag in the last panel, narrated by the owner-persona character in Korean. The same Lambda posts to Threads, so a failed render still ships the text.

> ⚠️ **DigestPrompt's JSON key order matters.** It asks for `items` first and `lead` last, so the lead comments on stories already written (word overlap with the headline reply fell from 0.21-0.41 to 0.03-0.21). Do not reorder the keys to match `DigestContent`.

## Deep research

`research_cli.py` runs a Strands agent that combines eight tools as it sees fit:

| Tool | What it does |
|------|--------------|
| `web_search(query, news, time_range)` | Tavily web or news search, with published dates |
| `community_search(query)` | Reddit, X, HN, Substack |
| `search_papers(query)` | OpenAlex: date, citations, abstract |
| `read_url(url)` | A primary source's full text |
| `recall_trends()` / `recall_digest(digest_date)` | What the digest has tracked and published |
| `attach_image(source_url)` | Stage a source's OG image |
| `deliver_report(report, channel)` | Hand over the report, or post it to Threads |

Before anything goes public, `deliver_report` checks the thread:

- every URL it cites must have come back from a tool call;
- it must cite at least one source;
- a separate model reads each cited source and sends back claims the source does not support, and every rewrite is checked again (if the check call itself fails, the thread posts without it);
- it must fit the Threads limits, or it is sent back to be rewritten;
- the image it carries must come from a page the thread cites.

The OpenClaw skill (`~/.openclaw/workspace/skills/omnisummary-research/SKILL.md`) calls the CLI and relays its JSON:

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

Exit codes for both CLIs: `0` succeeded, `1` failed, `2` rejected input (also returned as JSON with `status: "rejected"`). For `recall_cli.py`, `found: false` means nothing is stored for that date, while `status: "failed"` means the store could not be read. Logs go to stderr.

## Running it day to day

[docs/operations.md](docs/operations.md) (Korean) is the operator's guide: the daily schedule, what each alert means and what to run, how to re-post a failed day from its stored snapshot, and what never to do. An alert for a day that posted nothing carries the exact re-post command with the real date.

## Deploying to AWS

### Steps

Before you start: `config/config.yaml` must exist, `.env` must hold `ALERT_EMAIL`, and `<region>` below is your `aws.region`. If you will ever use X or YouTube transcripts, create a bucket yourself and set `aws.state_bucket_name` before the first deploy: the local sync writes only to that bucket. Never switch the key after the first deploy, because on a non-prod stage the bucket the stack created is then deleted.

Use the repo's pinned CDK CLI (`npm install`, then `npx cdk`), not a global `cdk`. On a fresh account the order matters, because the foundation stack owns the ECR repository the Lambdas pull from.

```bash
npm install
export AWS_PROFILE=<profile>

# 1. Once per account and region (no image exists yet, hence FOUNDATION_ONLY)
export FOUNDATION_ONLY=1
npx cdk bootstrap -a "uv run python scripts/deploy.py"

# 2. Foundation first: it creates the ECR repository
npx cdk deploy '*-foundation' -a "uv run python scripts/deploy.py"
unset FOUNDATION_ONLY   # never set it on an account where the application stack is deployed

# 3. Build and push the image
ECR_URI="$(aws sts get-caller-identity --query Account --output text).dkr.ecr.<region>.amazonaws.com/omnisummary-<stage>-agent"
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin "${ECR_URI%%/*}"
docker build --platform linux/amd64 --provenance=false -t "${ECR_URI}:latest" . && docker push "${ECR_URI}:latest"

# 4. Deploy, pinned to the pushed image's sha256 digest (deploy.py refuses to run without it)
export DIGEST_IMAGE_REF=$(aws ecr describe-images --region <region> --repository-name omnisummary-<stage>-agent \
  --image-ids imageTag=latest --query 'imageDetails[0].imageDigest' --output text)
npx cdk deploy --all -a "uv run python scripts/deploy.py"

# 5. Secrets and cost attribution
uv run python scripts/put_secrets.py            # writes the values in .env; --verify for a read-only check
uv run python scripts/put_inference_profiles.py # again after changing a stage's model
```

Later deploys repeat steps 3 and 4. CloudFormation does not redeploy a Lambda whose image tag string is unchanged, which is why the image is pinned by its sha256 digest. `put_inference_profiles.py` uses `aws.profile` from the config, so make it the same account as `AWS_PROFILE`. In zsh, write `"${ECR_URI}:latest"` with braces: `$ECR_URI:l` is a zsh modifier.

### What gets created

| Resource | Purpose |
|----------|---------|
| Lambda: digest | Collects, ranks and writes; 15 min timeout |
| Lambda: visual | Draws the comic and posts to Threads; the only publish path; one run at a time |
| Lambda: Threads token refresh | Weekly; writes the renewed token back to SSM |
| Lambda: comment replies | Hourly; one run at a time |
| EventBridge | The daily digest (`aws.digest_cron_hour` / `digest_cron_minute`, in **UTC**; the default 10:00 UTC is 19:00 KST) and the two schedules above |
| AgentCore Memory | One digest snapshot per date; retained if the stack is deleted |
| S3 | Trends, ledgers, park files, Threads images |
| SSM Parameter Store | The secrets, as SecureStrings |
| SQS | Dead-letter queue for the digest and visual Lambdas |
| SNS + CloudWatch | Email alerts; 10 alarms (errors and near-timeout duration per Lambda, empty digest, DLQ) |
| ECR | The Lambda image; the last 30 images are kept |

Every Lambda runs with `retry_attempts=0`: Threads has no idempotency key, so an automatic retry could post twice. Failures land in the DLQ or raise the Errors alarm.

### Secrets on AWS

A CloudFormation template cannot hold a SecureString, so the stack creates each parameter with a placeholder and `scripts/put_secrets.py` writes the real values after the deploy. Re-deploys never overwrite them. The script:

- skips parameters that are already SecureStrings, because the Threads token is rotated in place (`--force` overwrites);
- never blanks a parameter whose variable is empty;
- keeps going when one parameter fails, and exits non-zero at the end.

Never add or change template properties on the secret parameters, tags included: CloudFormation would put the placeholder back over the live secret. A test pins them.

### Cost attribution

Bedrock on-demand usage has no taggable resource. `scripts/put_inference_profiles.py` creates a tagged application inference profile per configured model, and model resolution prefers it, so Cost Explorer can attribute the spend (activate the `Project` cost-allocation tag in Billing). Each model call also logs `LLM usage stage=... model=... input=... output=...`, because several stages share one model on the bill.

### Local sync for X and YouTube

X and YouTube transcripts block datacenter IPs, so collect them on your machine shortly before the digest runs. The sync writes to `aws.state_bucket_name`, which must have been set before the first deploy (see Deploying to AWS); do not add it to a stack that created its own bucket. For X also set `collectors.rsshub.enabled: true` and its `accounts`. An unattended cron should use the `S3_SYNC_*` keys in `.env`, since SSO sessions expire.

```bash
crontab -e
# 18:30 KST, 30 minutes before a 19:00 KST (10:00 UTC) digest; cron uses the machine's local time zone
30 18 * * * AWS_PROFILE=<profile> /path/to/omnisummary/scripts/sync_all_to_s3.sh >> /path/to/omnisummary/logs/sync.log 2>&1   # not /tmp: a reboot clears it
```

A park file older than `park_max_age_hours` (12h; a healthy one is about 30 minutes old at the digest) makes the source report `STALE`, so a stopped cron can't look healthy.

## Development

```bash
uv run python -m pytest tests/             # hermetic: no network, no AWS, no real secrets
uv run pre-commit run --all-files          # ruff, black, file checks, uv lock --check
uv run mypy .
npx cdk synth -a "uv run python scripts/ci_synth.py"   # offline synth through the pinned CLI
```

CI runs lint and type checks, an offline CDK synth of `config/config-template.yaml`, tests with coverage, a Docker build that imports the entry modules with no network, and a dependency and secret scan. `pip-audit --strict` fails the build on any new advisory in the locked set; fix it with `uv lock --upgrade-package <pkg>`. Regenerate the lockfile with `uvx --from uv==0.6.14 uv lock`, the uv version CI and the Dockerfile pin.

## Project structure

```
omnisummary/
├── main.py            # Digest CLI
├── research_cli.py    # Deep-research CLI (JSON on stdout)
├── recall_cli.py      # Past digests and trends (JSON on stdout)
├── collectors/        # RSS, Reddit, RSSHub (X), YouTube, web search, S3 park loader
├── pipeline/          # Aggregator, ranker, trends, digest, comic, comment replies, runner
├── agent/             # Research agent and tools, image generator, digest state
├── output/            # Threads rendering and posting, research delivery
├── shared/            # Config, models, prompts, model registry, stores, research backends
├── lambda_handlers/   # Digest, visual, token refresh, comment replies
├── infrastructure/    # CDK stacks
├── scripts/           # Deploy, secrets, profiles, sync, samplers
├── assets/            # persona.jpg, the narrator's reference sheet
├── config/            # config-template.yaml
├── tests/
└── docs/              # design.md and diagrams
```

## License

[MIT License](LICENSE)
