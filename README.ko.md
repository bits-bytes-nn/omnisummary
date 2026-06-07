<div align="center">

# 🗞️ OmniSummary

**능동형 AI/ML 데일리 다이제스트 — 멀티 소스 수집, LLM 랭킹, 한국어 에디토리얼 다이제스트를 Slack으로 전달하고, 심화 분석용 Bedrock AgentCore 후속 에이전트까지.**

AWS 위 일간 파이프라인 · Bedrock AgentCore (Runtime + Memory) · Amazon Bedrock (Claude) 기반.

[![CI](https://github.com/bits-bytes-nn/omnisummary/actions/workflows/ci.yml/badge.svg)](https://github.com/bits-bytes-nn/omnisummary/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![AWS CDK](https://img.shields.io/badge/IaC-AWS%20CDK-orange)
![Bedrock](https://img.shields.io/badge/LLM-Amazon%20Bedrock%20(Claude)-green)

🇺🇸 [English README](./README.md)

![OmniSummary 아키텍처](docs/diagrams/architecture.png)

</div>

---

## 주요 기능

- **멀티 소스 수집**: Reddit(프록시 경유 공개 .rss 피드), YouTube, X/Twitter(RSSHub 경유), RSS/Substack, 웹 검색(Tavily)
- **LLM 기반 랭킹**: Claude Opus 4.8, 소스 슬롯 + 출처별 다양성 캡을 적용한 다축(multi-axis) 평가
- **에디토리얼 다이제스트**: Claude Sonnet 4.6 한국어 에디토리얼, 일자 간 트렌드 추적 포함
- **후속 에이전트**: Slack 기반 자율 Strands 에이전트 — 분석, 논문/커뮤니티/뉴스 검색, 일자 간 회상, 자유 형식 이미지 생성(1페이지 슬라이드 / 만화 / 다이어그램 / 인포그래픽, OpenAI gpt-image-2 경유)을 자유롭게 조합
- **AgentCore 중심**: 다이제스트 상태를 Bedrock AgentCore Memory에 보존하고, 에이전트는 AgentCore Runtime에서 실행
- **운영 우수성**: 소스별 헬스 체크 → SNS 이메일 알림, 상관 ID가 붙는 구조화된 JSON 로깅, CloudWatch 알람, API에 AWS WAF
- **AWS 배포**: Lambda + EventBridge cron + Bedrock AgentCore(Runtime + Memory) + ECS(RSSHub)

## 아키텍처

![다이제스트 동작 방식](docs/diagrams/concept-pipeline.png)

## 빠른 시작

### 사전 준비

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) 패키지 매니저
- Docker (RSSHub와 AWS 배포용)
- Bedrock 접근 권한이 있는 AWS 계정
- 봇 앱이 설정된 Slack 워크스페이스

### 설치

```bash
git clone <repo-url> && cd omnisummary
uv sync
cp config/config-template.yaml config/config.yaml
cp .env.template .env  # API 키 입력
```

### 설정

`config/config.yaml`을 편집하세요:

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
  ranking_model: "anthropic.claude-opus-4-8"
  digest_model: "anthropic.claude-sonnet-4-6"
  max_per_origin: 1   # 채널/작성자/서브레딧당 항목 수 상한
  source_slots:
    web: 1
    x: 1
    rss: 1
    reddit: 1
    youtube: 1
```

필수 환경 변수(`.env`):

```
SLACK_BOT_TOKEN=xoxb-...
SLACK_APP_TOKEN=xapp-...           # Socket Mode (slack_agent.py)
SLACK_CHANNEL_ID=C...
TAVILY_API_KEY=tvly-...
YOUTUBE_API_KEY=AIza...            # 선택, 없으면 RSS로 폴백
OPENAI_API_KEY=sk-...              # 선택, make_visual(자유 형식 이미지) 활성화
ALERT_EMAIL=you@example.com        # 선택, 소스 헬스 알림
CLOUDFLARE_PROXY_URL=https://...   # AWS 배포용 (YouTube 폴백)
CLOUDFLARE_PROXY_TOKEN=...
```

### 로컬 사용법

```bash
# 다이제스트 파이프라인 실행 (dry-run)
uv run python main.py --dry-run --sources rss reddit

# Slack 전달과 함께 실행
uv run python main.py

# 대화형 에이전트 모드
uv run python main.py --dry-run --sources rss --interactive

# Slack 에이전트 (Socket Mode)
uv run python slack_agent.py

# RSSHub S3 동기화 (AWS용)
uv run python scripts/sync_rsshub_to_s3.py
```

### CLI 옵션

| 플래그 | 설명 |
|------|-------------|
| `--sources rss reddit youtube` | 특정 소스 선택 |
| `--dry-run` | Slack 전달 생략, 콘솔 출력 |
| `--top-n 5` | 선택 항목 수 재정의 |
| `--date 2026-03-28` | 다이제스트 날짜 지정 (기본: 오늘 KST) |
| `--interactive` | 다이제스트 후 에이전트 대화 모드 진입 |

## 파이프라인 단계

### 1. 수집(Collection)

각 컬렉터는 비동기로 병렬 실행됩니다. 조회 기간(lookback)은 소스별로 설정 가능합니다.

| 컬렉터 | 소스 | 방식 |
|-----------|--------|--------|
| `RedditCollector` | Reddit 공개 `.rss` 피드 | Cloudflare 프록시 경유 (API/앱 불필요) |
| `YouTubeCollector` | YouTube Data API v3 / RSS 폴백 | 직접 또는 프록시 |
| `RSSCollector` | RSS/Atom 피드 | feedparser |
| `RSSHubCollector` | RSSHub 경유 X/Twitter | 로컬 Docker 또는 S3 동기화 |
| `WebSearchCollector` | Tavily API | 직접, LLM 쿼리 정제 포함 |

### 2. 집계(Aggregation)

`ContentAggregator`가 URL과 정규화된 제목(대소문자 무시, 문장부호 제거, 유니코드 정규화)으로 중복을 제거합니다.

### 3. 랭킹(Ranking)

`ContentRanker`가 Claude Opus로 다축 평가를 수행합니다:

- 기술적 실속, 실무자 가치, 신규성
- 산업 영향, 연구 의의, 소스 권위
- 하드 필터: 홍보성, 빈약한 콘텐츠, 초보 질문 → 점수 ≤ 0.3
- 콘텐츠 보너스: 인터뷰, 논문 요약, 주요 모델 출시
- `origin_weights`: 알려진 출처에 대한 가산 점수 보정 — `score + (weight-1.0) * origin_weight_nudge`, [0,1]로 클램프 (배수가 아니라 동점 처리용)
- `source_slots`: 소스 유형별 최소 보장 수

### 4. 트렌드 추적(Trend Tracking)

`TrendTracker`가 `trends.json`에 구조화된 트렌드(날짜가 붙은 근거를 가진 slug-id `Trend` 객체)를 유지합니다. LLM은 오늘의 항목을 기존/신규 트렌드로 분류만 하고, 모든 장부 관리(날짜 기록, active/cooling/archived 생명주기, 최신성 감쇠 모멘텀, 근거/active 캡)는 코드가 담당합니다. active+cooling 트렌드(모멘텀 정렬)는 일자 간 연속성을 위해 다음 다이제스트에 투입됩니다.

### 5. 다이제스트 생성(Digest Generation)

`DigestGenerator`가 Claude Sonnet으로 한국어 에디토리얼 다이제스트를 생성합니다:

- 오프닝: 전체 항목 요약이 아니라 하나의 에디토리얼 앵글
- 항목별: 소스 태그 + 인게이지먼트 지표, 핵심 내용, 기술적 세부, 함의(이탤릭)
- `sanitize_slack_mrkdwn()` 후처리로 Slack mrkdwn 포매팅

### 6. 후속 에이전트(Follow-up Agent)

자율 Strands 에이전트(Bedrock AgentCore Runtime에서 실행, AgentCore Memory에서 다이제스트 상태를 읽음). 요청을 만족시키기 위해 아래 6개의 단일 목적 도구를 자유롭게 조합합니다 — 예: "1번 항목을 1페이지 슬라이드로" → `get_detail` → 근거 확보를 위한 선택적 `search_*` → `make_visual`:

| 도구 | 기능 |
|------|----------|
| `get_detail(item_number)` | 랭킹 메타데이터를 포함한 항목 전체 분석 |
| `search_papers(query)` | Semantic Scholar API |
| `search_community(query)` | Tavily (Reddit, X, HN, Substack) |
| `search_related_news(query)` | Tavily (일반 뉴스) |
| `recall_trends(query)` | 구조화된 `trends.json`(active/cooling)에 대한 키워드 매칭, 모멘텀 순위 |
| `make_visual(instruction, item_number, context)` | 자연어 지시로부터 자유 형식 이미지(1페이지 슬라이드 / 만화 / 다이어그램 / 인포그래픽) 생성 → OpenAI gpt-image-2 경유로 Slack에 게시 |

## AWS 배포

### 인프라 (CDK)

먼저 두 이미지를 모두 빌드·푸시한 뒤(아래 Docker 이미지 참고), 푸시된 digest를 고정해
배포하세요. 이미지 *태그* 문자열이 그대로면 CloudFormation이 Lambda를 재배포하지 않으므로,
푸시된 `sha256` digest를 `DIGEST_IMAGE_REF`로 넘깁니다:

```bash
export DIGEST_IMAGE_REF=sha256:<pushed-digest>   # AGENTCORE_IMAGE_REF은 기본 :arm64
AWS_PROFILE=<profile> uv run cdk deploy --all -a "uv run python scripts/deploy.py"
```

생성되는 리소스:
- **Lambda** (Docker): 다이제스트 파이프라인, 15분 타임아웃
- **Lambda**: Slack 이벤트 핸들러, 60초 타임아웃
- **API Gateway** + **AWS WAFv2**: 레이트 리밋 + 매니지드 룰 + 스로틀링이 적용된 `POST /slack/events`
- **EventBridge**: 일간 cron (설정 기반 시/분)
- **Bedrock AgentCore**: Runtime(후속 에이전트, arm64) + **Memory**(후속 에이전트용 다이제스트 스냅샷)
- **ECS Fargate**: RSSHub 컨테이너
- **S3**: 트렌드 + RSSHub 동기화 데이터
- **DynamoDB**: Slack 이벤트 중복 제거
- **SNS**: 소스 헬스 알림 토픽(이메일)
- **CloudWatch**: 구조화된 로그 + 에러/5xx 알람
- **ECR**: Docker 이미지(Lambda용 amd64, AgentCore용 arm64)

### Docker 이미지

아키텍처별 두 개의 Dockerfile:

```bash
# Lambda (amd64)
docker build --platform linux/amd64 --provenance=false -t <ecr-uri>:latest .
docker push <ecr-uri>:latest

# AgentCore (arm64)
docker buildx build --platform linux/arm64 --provenance=false \
  -f Dockerfile.agentcore -t <ecr-uri>:arm64 . --push
```

### Cloudflare Workers 프록시

Reddit과 YouTube는 AWS 데이터센터 IP에서 차단됩니다. Cloudflare Worker가 HTTP 프록시 역할을 합니다:

```bash
cd cloudflare-proxy
npx wrangler login
npx wrangler deploy
```

### 로컬 Cron 설정

RSSHub(X/Twitter) 데이터는 AWS 다이제스트 실행 전에 로컬에서 S3로 동기화돼야 합니다:

```bash
# crontab에 추가
crontab -e
50 21 * * * cd /path/to/omnisummary && .venv/bin/python scripts/sync_rsshub_to_s3.py >> /tmp/rsshub_sync.log 2>&1
```

### 외부 서비스

| 서비스 | 용도 | 비용 |
|---------|---------|------|
| AWS Bedrock | LLM (Claude Opus/Sonnet) | 사용량 기반 |
| Cloudflare Workers | Reddit/YouTube용 HTTP 프록시 | 무료 (100K req/day) |
| Tavily | 웹 검색 | 무료 티어 |
| Semantic Scholar | 논문 검색 | 무료 |
| YouTube Data API v3 | 영상 메타데이터 | 무료 (10K units/day) |
| Slack | 전달 + 에이전트 | 무료 |

## 프로젝트 구조

```
omnisummary/
├── main.py                     # CLI 진입점
├── slack_agent.py              # Slack Socket Mode 에이전트
├── Dockerfile                  # Lambda (amd64)
├── Dockerfile.agentcore        # AgentCore (arm64)
├── collectors/                 # 소스 컬렉터
├── pipeline/                   # Aggregator, Ranker, DigestGenerator, TrendTracker, DailyVisual
├── agent/                      # Strands 에이전트 + 도구
├── agent_runtime/              # Bedrock AgentCore HTTP 서버
├── shared/                     # 설정, 모델, 포매팅, 프롬프트, 상태 저장소, AgentCore 메모리
├── output/                     # Slack 핸들러
├── lambda_handlers/            # AWS Lambda 핸들러 (digest, slack events, daily visual)
├── infrastructure/             # CDK 스택
├── scripts/                    # 배포, RSSHub 동기화
├── cloudflare-proxy/           # CF Worker 프록시
├── config/                     # YAML 설정
├── tests/                      # 단위 + CDK 어서션 테스트
└── docs/                       # diagrams/ (architecture, concept-pipeline)
```

## 테스트 & CI

```bash
uv run python -m pytest tests/ -v        # 단위 + CDK 어서션 테스트
uv run black --check . && uv run ruff check .
uv run python scripts/ci_synth.py        # 오프라인 CDK synth
```

CI (`.github/workflows/ci.yml`): 린트, 포맷 검사, 테스트 + 커버리지 게이트, CDK synth, Docker 빌드.

## 라이선스

[MIT License](LICENSE)
