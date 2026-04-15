# OmniSummary

Proactive AI/ML daily digest system that collects content from multiple sources, ranks by importance using LLM, generates editorial-style digests, and delivers via Slack. Includes a follow-up agent for deep-dive analysis on specific items.

## Features

- **Multi-source collection**: Reddit, YouTube, X/Twitter (via RSSHub), RSS/Substack, Web Search (Tavily)
- **LLM-powered ranking**: 6-axis evaluation (technical substance, practitioner value, novelty, industry impact, research significance, source authority)
- **Editorial digest**: Korean editorial with trend tracking across days
- **Follow-up agent**: Slack-based interactive agent for detailed analysis, paper search, and community reaction lookup
- **AWS deployment**: Lambda + EventBridge cron + Bedrock AgentCore + ECS (RSSHub) + S3 state storage

## Architecture

![OmniSummary Architecture](assets/architecture.png)

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker (for RSSHub and AWS deployment)
- AWS account with Bedrock access
- Slack workspace with bot app

### Installation

```bash
git clone <repo-url> && cd omnisummary
uv sync
cp config/config-template.yaml config/config.yaml
cp .env.template .env  # Fill in API keys
```

### Configuration

Edit `config/config.yaml`:

```yaml
collectors:
  youtube:
    enabled: true
    channels:
      - "https://www.youtube.com/@AndrejKarpathy"
    lookback_hours: 24
  reddit:
    enabled: true
    subreddits:
      - LocalLLaMA
  rss:
    enabled: true
    feeds:
      - "https://feeds.feedburner.com/geeknews-feed"
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
  ranking_model: "anthropic.claude-opus-4-6-v1"
  digest_model: "anthropic.claude-sonnet-4-6"
  source_slots:
    web: 1
    x: 1
    rss: 1
    reddit: 1
    youtube: 1
```

Required environment variables (`.env`):

```
SLACK_BOT_TOKEN=xoxb-...
SLACK_CHANNEL_ID=C...
TAVILY_API_KEY=tvly-...
YOUTUBE_API_KEY=AIza...            # Optional, falls back to RSS
CLOUDFLARE_PROXY_URL=https://...   # For AWS deployment
CLOUDFLARE_PROXY_TOKEN=...
```

### Local Usage

```bash
# Run digest pipeline (dry-run)
uv run python main.py --dry-run --sources rss reddit

# Run with Slack delivery
uv run python main.py

# Interactive agent mode
uv run python main.py --dry-run --sources rss --interactive

# Slack agent (Socket Mode)
uv run python slack_agent.py

# RSSHub S3 sync (for AWS)
uv run python scripts/sync_rsshub_to_s3.py
```

### CLI Options

| Flag | Description |
|------|-------------|
| `--sources rss reddit youtube` | Select specific sources |
| `--dry-run` | Skip Slack delivery, print to console |
| `--top-n 5` | Override number of items to select |
| `--date 2026-03-28` | Set digest date (default: today KST) |
| `--interactive` | Enter agent chat mode after digest |

## Pipeline Stages

### 1. Collection

Each collector runs async in parallel. Lookback window is configurable per source.

| Collector | Source | Method |
|-----------|--------|--------|
| `RedditCollector` | Reddit JSON API | Cloudflare proxy on AWS |
| `YouTubeCollector` | YouTube Data API v3 / RSS fallback | Direct or proxy |
| `RSSCollector` | RSS/Atom feeds | feedparser |
| `RSSHubCollector` | X/Twitter via RSSHub | Local Docker or S3 sync |
| `WebSearchCollector` | Tavily API | Direct, with LLM query refinement |

### 2. Aggregation

`ContentAggregator` deduplicates by URL and normalized title (case-insensitive, punctuation-stripped, Unicode-normalized).

### 3. Ranking

`ContentRanker` uses Claude Opus for multi-axis evaluation:

- Technical substance, practitioner value, novelty
- Industry impact, research significance, source authority
- Hard filters: promotions, thin content, beginner questions → score ≤ 0.3
- Content bonuses: interviews, paper summaries, major model releases
- `origin_weights`: configurable boost for known AI leaders
- `source_slots`: guaranteed minimum per source type

### 4. Trend Tracking

`TrendTracker` maintains `trends.md` with active/cooling/archived trends. Updated after each digest. Feeds context into digest generation for cross-day narrative continuity.

### 5. Digest Generation

`DigestGenerator` uses Claude Sonnet to produce Korean editorial digest:

- Opening: one editorial angle, not a summary of all items
- Per item: source tag + engagement metrics, core content, technical detail, implications (italic)
- Slack mrkdwn formatting with `sanitize_slack_mrkdwn()` post-processing

### 6. Follow-up Agent

Strands Agent with 4 tools:

| Tool | Function |
|------|----------|
| `get_detail(item_number)` | Full item analysis with ranking metadata |
| `search_papers(query)` | Semantic Scholar API |
| `search_community(query)` | Tavily (Reddit, X, HN, Substack) |
| `search_related_news(query)` | Tavily (general news) |

## AWS Deployment

### Infrastructure (CDK)

```bash
# Deploy all stacks
AWS_PROFILE=<profile> uv run cdk deploy --all -a "uv run python scripts/deploy.py"
```

Resources created:
- **Lambda** (Docker): Digest pipeline, 15min timeout
- **Lambda**: Slack event handler, 60s timeout
- **API Gateway**: `POST /slack/events`
- **EventBridge**: Daily cron (22:00 KST)
- **Bedrock AgentCore**: Follow-up agent runtime (arm64)
- **ECS Fargate**: RSSHub container
- **S3**: Digest state, trends, RSSHub sync data
- **DynamoDB**: Slack event deduplication
- **ECR**: Docker images (amd64 for Lambda, arm64 for AgentCore)

### Docker Images

Two Dockerfiles for different architectures:

```bash
# Lambda (amd64)
docker build --platform linux/amd64 --provenance=false -t <ecr-uri>:latest .
docker push <ecr-uri>:latest

# AgentCore (arm64)
docker buildx build --platform linux/arm64 --provenance=false \
  -f Dockerfile.agentcore -t <ecr-uri>:arm64 . --push
```

### Cloudflare Workers Proxy

Reddit and YouTube are blocked from AWS datacenter IPs. A Cloudflare Worker acts as HTTP proxy:

```bash
cd cloudflare-proxy
npx wrangler login
npx wrangler deploy
```

### Local Cron Setup

RSSHub (X/Twitter) data must be synced from local to S3 before AWS digest runs:

```bash
# Add to crontab
crontab -e
50 21 * * * cd /path/to/omnisummary && .venv/bin/python scripts/sync_rsshub_to_s3.py >> /tmp/rsshub_sync.log 2>&1
```

### External Services

| Service | Purpose | Cost |
|---------|---------|------|
| AWS Bedrock | LLM (Claude Opus/Sonnet) | Usage-based |
| Cloudflare Workers | HTTP proxy for Reddit/YouTube | Free (100K req/day) |
| Tavily | Web search | Free tier |
| Semantic Scholar | Paper search | Free |
| YouTube Data API v3 | Video metadata | Free (10K units/day) |
| Slack | Delivery + agent | Free |

## Project Structure

```
omnisummary/
├── main.py                     # CLI entry point
├── slack_agent.py              # Slack Socket Mode agent
├── Dockerfile                  # Lambda (amd64)
├── Dockerfile.agentcore        # AgentCore (arm64)
├── collectors/                 # Source collectors
├── pipeline/                   # Aggregator, Ranker, DigestGenerator, TrendTracker
├── agent/                      # Strands agent + tools
├── agent_runtime/              # Bedrock AgentCore HTTP server
├── shared/                     # Config, models, utils, prompts, state store
├── output/                     # Slack handler
├── lambda_handlers/            # AWS Lambda handlers
├── infrastructure/             # CDK stacks
├── scripts/                    # Deploy, RSSHub sync
├── cloudflare-proxy/           # CF Worker proxy
├── config/                     # YAML configuration
├── tests/                      # Unit tests (54)
└── docs/                       # Architecture documentation
```

## Testing

```bash
uv run python -m pytest tests/ -v
```

## License

[MIT License](LICENSE)
