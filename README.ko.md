<div align="center">

# 🗞️ OmniSummary

**능동형 AI/ML 데일리 다이제스트. 다섯 계열의 소스에서 콘텐츠를 모아 LLM으로 순위를 매기고, 한국어 에디토리얼 다이제스트를 써서 config에서 켠 채널(Slack, Threads, 또는 둘 다)로 보낸다. Slack 멘션으로 작동하는 딥 리서치 에이전트는 어떤 주제든 웹과 논문과 커뮤니티를 조사해 페르소나 보이스의 인용 기반 리포트를 게시한다.**

AWS 위에서 동작한다. Bedrock AgentCore(Runtime + Memory)와 Amazon Bedrock(Claude)을 쓴다.

[![CI](https://github.com/bits-bytes-nn/omnisummary/actions/workflows/ci.yml/badge.svg)](https://github.com/bits-bytes-nn/omnisummary/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![AWS CDK](https://img.shields.io/badge/IaC-AWS%20CDK-orange)
![Bedrock](https://img.shields.io/badge/LLM-Amazon%20Bedrock%20(Claude)-green)

🇺🇸 [English README](./README.md) · 📐 [설계 문서](./docs/design.md)

![OmniSummary 아키텍처](docs/diagrams/architecture.png)

</div>

---

## 무엇을 하는가

서로 독립적인 두 개의 경로가 있다.

**데일리 다이제스트**는 크론으로 돈다. RSS와 Substack, Reddit, YouTube, X/Twitter, 웹 검색에서 콘텐츠를 모아 중복을 제거하고, Claude Opus로 전부 순위를 매긴 뒤 Claude Sonnet이 한국어 에디토리얼 다이제스트를 쓴다. 그 다이제스트를 채널별로 렌더링해 전달하고, 헤드라인 스토리를 그린 일러스트가 함께 나간다.

**딥 리서치 에이전트**는 Slack 멘션으로 작동한다. 자유형 주제를 받으면 열린 웹과 학술 문헌, 커뮤니티 논의를 스스로 조사한 뒤 다이제스트와 같은 내레이터 보이스로 인용 기반 한국어 리포트를 써서 Slack에 게시한다. 요청하면 Threads에도 올린다.

![다이제스트 동작 방식](docs/diagrams/concept-pipeline.png)

## 주요 기능

- **멀티 소스 수집.** Reddit(공개 `.rss` 피드), YouTube, X/Twitter(RSSHub 경유), RSS/Substack, 웹 검색(Tavily)
- **LLM 랭킹.** Claude Opus 4.8이 다축으로 점수를 매기고(Opus 5와 Sonnet 5도 선택 가능), 소스 슬롯과 출처별 다양성 캡으로 한 채널이 그날을 독점하지 못하게 한다
- **에디토리얼 다이제스트.** Claude Sonnet 5가 한국어 산문을 쓰고 일자 간 트렌드 연속성을 유지한다
- **멀티 채널 전달.** 하나의 구조화된 다이제스트를 채널별로 렌더링한다. Slack은 Block Kit, Threads는 이미지 루트에 평면 답글 체인을 붙이며 각각 독립 토글이다
- **딥 리서치 에이전트.** 단일 목적 도구 8개를 가진 자율 Strands 에이전트이고 출처 기사의 OG 이미지를 첨부한다
- **AgentCore 중심.** 다이제스트 상태는 Bedrock AgentCore Memory에, 에이전트는 AgentCore Runtime에 있다
- **운영 우수성.** 소스별 헬스 보고(`OK`, `EMPTY`, `FAILED`, `STALE`, `DEGRADED`)를 SNS 이메일로 알리고, 상관 ID가 붙는 구조화된 JSON 로그와 CloudWatch 알람 12개, 공개 API에 AWS WAF를 둔다
- **AWS 배포.** Lambda와 EventBridge 크론, Bedrock AgentCore, ECS(RSSHub)를 전부 CDK로 관리한다

## 빠른 시작

### 사전 준비

- Python 3.12 이상과 [uv](https://docs.astral.sh/uv/)
- Docker (RSSHub와 AWS 배포용)
- Bedrock 접근 권한이 있는 AWS 계정
- 봇 앱이 설정된 Slack 워크스페이스

### 설치

```bash
git clone <repo-url> && cd omnisummary
uv sync
cp config/config-template.yaml config/config.yaml
cp .env.template .env
```

### 설정

시크릿이 아닌 모든 것은 `config/config.yaml`에 있다. 눈여겨볼 노브는 다음과 같다.

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
    lookback_hours: 24
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
  max_per_origin: 1        # 채널, 작성자, 서브레딧, 피드, 웹 호스트당 상한
  source_slots: {web: 1, x: 1, rss: 1, reddit: 1, youtube: 1}
```

모든 필드는 [design.md §3](docs/design.md#3-설정)에 문서화되어 있다.

### 시크릿

시크릿은 로컬에서는 `.env`(`.env.template` 참고), AWS에서는 SSM Parameter Store에 둔다. 실행 자체에 필요한 것은 앞의 세 개뿐이다. 나머지는 각각 하나의 기능을 켜는 값이고, 없으면 그 기능이 로그 한 줄을 남기고 스킵된다.

| 변수 | 용도 |
|------|------|
| `SLACK_BOT_TOKEN` | 다이제스트 전달과 에이전트. `enable_slack_post: false`일 때만 생략할 수 있다 |
| `SLACK_CHANNEL_ID` | 다이제스트 대상 채널 |
| `TAVILY_API_KEY` | `web_search` 수집기와 에이전트의 커뮤니티·뉴스 검색 |
| `SLACK_SIGNING_SECRET` | Slack 이벤트 API Gateway 경로. 인바운드 에이전트 이벤트를 검증한다 |
| `YOUTUBE_API_KEY` | YouTube 수집기. 없으면 RSS로 폴백하고 자막은 못 받는다 |
| `OPENAI_API_KEY` | 데일리 비주얼의 gpt-image 렌더 |
| `THREADS_ACCESS_TOKEN`, `THREADS_USER_ID` | Threads 전달. 60일 토큰이고 AWS에서는 SSM으로 자동 갱신된다 |
| `ALERT_EMAIL` | 소스 헬스 SNS 이메일 알림(AWS) |
| `CLOUDFLARE_PROXY_URL`, `CLOUDFLARE_PROXY_TOKEN` | AWS 전용. 데이터센터 IP에서 Reddit `.rss`와 YouTube RSS를 읽는다 |
| `TWITTER_AUTH_TOKEN`, `TWITTER_CT0` | RSSHub 경유 X/Twitter. x.com 세션 쿠키다 |
| `S3_SYNC_ACCESS_KEY_ID`, `S3_SYNC_SECRET_ACCESS_KEY` | 선택. 로컬에서 S3로 올리는 sync 전용 자격증명이고, 없으면 `AWS_PROFILE`을 쓴다 |

> **시크릿은 CDK 스택을 통과하지 않는다.** CloudFormation 템플릿은 SecureString을 담을 수 없어서, 값을
> 스택에 넘기면 `cdk.out`과 스테이징 버킷, 모든 `GetTemplate` 응답에 평문으로 남는다. 스택은 파라미터
> 경로만 플레이스홀더로 만들고, 배포 후 `scripts/put_secrets.py`가 실제 값을 SecureString으로 기록한다.
> [AWS의 시크릿](#aws의-시크릿)을 참고하라.

### 설정 체크리스트

로컬에서 다이제스트를 뽑기까지 필요한 최소 단계다. Slack으로 전달하고 X와 비주얼은 빼는 구성이다.

1. `uv sync`를 돌리고 위의 템플릿 파일 두 개를 복사한다.
2. `.env`에 `SLACK_BOT_TOKEN`과 `SLACK_CHANNEL_ID`, `TAVILY_API_KEY`를 채운다.
3. `aws.bedrock_region`(기본 `us-west-2`)에서 Bedrock에 접근할 수 있는지 확인하고 `AWS_PROFILE`이나 표준 AWS 자격증명을 설정한다. 로컬 실행이어도 랭킹과 다이제스트 LLM은 Bedrock에서 돈다.
4. `uv run python main.py --dry-run --sources rss reddit`을 돌리면 다이제스트가 출력된다.

그다음 기능을 하나씩 붙인다.

| 원하는 것 | 설정 | 비고 |
|-----------|------|------|
| 자막까지 붙은 YouTube 항목 | `YOUTUBE_API_KEY`와 로컬 sync | 자막은 거주용 IP에서만 받아진다 |
| X/Twitter 항목 | RSSHub 컨테이너와 로컬 sync | 아래 참고 |
| 데일리 비주얼 | `OPENAI_API_KEY` | gpt-image-2 |
| Threads 전달 | `THREADS_ACCESS_TOKEN`, `THREADS_USER_ID`, `enable_threads_post: true` | `enable_daily_visual: true`도 필요하다. Threads에 게시하는 주체가 데일리 비주얼 Lambda인데, 이미지와 텍스트가 한 게시물 세트로 나가야 하기 때문이다 |

### RSSHub 컨테이너 (X/Twitter)

X는 로컬 [RSSHub](https://docs.rsshub.app/) 컨테이너로 읽는다. 로그인된 x.com 세션의 쿠키 두 개가 필요하다. `auth_token`과 `ct0`다.

브라우저 개발자 도구에서 가져온다. macOS에서 F12가 리맵돼 있으면 ⌥⌘I를 쓴다. Chrome은 Application 탭에서 Cookies로 들어가 `https://x.com`을 찾고, Safari는 개발자용 메뉴를 먼저 켠 뒤 Storage 탭에서, Firefox도 Storage 탭에서 `x.com`을 찾는다.

```bash
docker run -d --name rsshub --restart unless-stopped -p 1200:1200 \
  -e NODE_ENV=production -e CACHE_TYPE=memory \
  -e TWITTER_AUTH_TOKEN='<auth_token>' \
  -e TWITTER_CT0='<ct0>' \
  diygod/rsshub:latest

curl -s "http://localhost:1200/twitter/user/karpathy" | head   # 스모크 테스트
```

쿠키가 없어도 컨테이너는 뜨지만 X 피드가 빈 채로 돌아온다. 쿠키는 주기적으로 만료되니, RSSHub 실패율이 올라가면(경고로 로깅된다) 쿠키를 갱신하고 컨테이너를 다시 만든다. AWS에서는 같은 이미지가 ECS Fargate에서 돈다.

## 사용법

```bash
# 다이제스트 파이프라인, 전달 없이
uv run python main.py --dry-run --sources rss reddit

# 전체 파이프라인과 전달
uv run python main.py

# 딥 리서치 에이전트를 로컬에서: 주제를 조사하고 렌더링된 리포트를 출력한다
uv run python research_cli.py "<주제>" --dry-run
uv run python research_cli.py "<주제>" --channel both --dry-run   # Slack과 Threads 미리보기

# 데이터센터 IP에서 차단되는 소스의 로컬 → S3 sync (X/RSSHub와 YouTube 자막)
./scripts/sync_all_to_s3.sh                  # 둘 다 돈다. 하나가 실패해도 다른 하나를 막지 않는다
uv run python scripts/sync_rsshub_to_s3.py   # X만
uv run python scripts/sync_youtube_to_s3.py  # YouTube만
```

| 플래그 | 설명 |
|--------|------|
| `--sources rss reddit youtube` | 특정 소스 선택 |
| `--dry-run` | 전달을 생략하고 콘솔에 출력 |
| `--top-n 5` | 선택 항목 수 재정의 |
| `--date 2026-03-28` | 다이제스트 날짜 지정(기본은 오늘, KST) |
| `--pin-url <url> [<url> ...]` | 점수와 무관하게 URL을 상위 스토리에 강제 편입한다. YouTube URL은 Data API로, 나머지는 Tavily로 해석한다. 로컬 CLI 전용 |
| `--force-republish` | 이미 나간 오늘 다이제스트를 다시 게시한다(Threads 멱등 가드를 우회한다) |

## 파이프라인 동작 방식

각 단계의 요약이다. 줄 단위 레퍼런스와 왜 이런 모양인지에 대한 답은 [design.md](docs/design.md)에 있다.

**1. 수집.** 모든 수집기가 각자의 조회 윈도를 갖고 비동기 병렬로 돌며 자기 헬스를 스스로 보고한다. 두 소스는 특별하다. X/Twitter와 YouTube 자막은 데이터센터 IP에서 차단되니, 로컬 크론이 거주용 IP로 수집해 S3에 park해두고 Lambda가 그 파일을 읽는다. 나이 예산을 넘긴 park 파일도 항목은 그대로 쓴다. 오래된 게 빈 것보다 낫다. 대신 소스가 `STALE`로 보고되니 멈춘 크론이 건강한 것처럼 보이지는 않는다.

| 수집기 | 소스 | 방식 |
|--------|------|------|
| `RedditCollector` | Reddit 공개 `.rss` | 직접 요청을 먼저 하고 Cloudflare 프록시로 폴백한다. API 앱은 필요 없다 |
| `YouTubeCollector` | YouTube Data API v3 | AWS에선 S3 park 파일, 그 외에는 라이브 |
| `RSSCollector` | RSS/Atom | feedparser |
| `RSSHubCollector` | RSSHub 경유 X/Twitter | AWS에선 S3 park 파일, 그 외에는 로컬 Docker |
| `WebSearchCollector` | Tavily | 직접 호출하며 LLM 쿼리 정제를 거친다 |

**2. 집계.** `ContentAggregator`가 URL과 정규화된 제목으로 중복을 제거한다. 중복이 걸리면 먼저 도착한 쪽이 아니라 더 좋은 쪽을 남긴다. 핀이 먼저고, 그다음 본문이 더 긴 쪽, 그다음 먼저 온 쪽 순이다. 얇은 Reddit 링크 포스트가 같은 기사의 전문을 밀어내지 못하게 하려는 것이다.

**3. 랭킹.** `ContentRanker`가 Claude Opus로 기술적 실속과 실무자 가치, 신규성, 산업 영향, 연구 의의, 소스 권위를 채점하고 홍보성이나 빈약한 콘텐츠는 하드 필터로 걸러낸다. 그다음 다양성을 고려해 선정한다. `source_slots`가 소스 유형별 최소치를 보장하고, `max_per_origin`이 특정 채널이나 작성자, 피드, 호스트를 상한으로 묶으며, 그대로 두면 다이제스트가 미달일 때만 하나의 fill 루프가 정해진 순서로 캡을 완화한다.

**4. 트렌드 추적.** `TrendTracker`가 구조화된 트렌드를 `trends.json`에 유지한다. LLM은 오늘 항목이 기존 트렌드의 연장인지 신규인지 분류만 하고, 장부 관리는 전부 결정론적 Python이 한다. 날짜 스탬프와 active/cooling/archived 생명주기, 최신성 감쇠 모멘텀, 증거 캡이 그렇다. active와 cooling 트렌드는 다음 날 다이제스트에 투입되며 이것이 일자 간 연속성의 근원이다.

**5. 다이제스트 생성.** `DigestGenerator`가 구조화된 `DigestContent`를 만든다. `lead` 하나와, 각각 title, url, body, implication을 가진 `items[]`다. LLM은 산문만 쓴다. 마크업도 소스 태그도 쓰지 않는다. 산문 예산은 코드가 소유한 고정 파트에서 코드가 계산하니 항목이 Threads의 500자 한도를 넘길 수 없다.

> ⚠️ **프롬프트의 JSON 키 순서가 동작에 영향을 준다.** `items`를 먼저, `lead`를 마지막에 요청하기 때문에 lead가 이미 쓰인 스토리에 대한 논평이 된다. 헤드라인 답글과의 단어 겹침이 0.21–0.41에서 0.03–0.21로 떨어진 것이 측정값이다. `DigestContent`의 필드 순서에 맞춰 키 순서를 되돌리지 말 것. 그 '정리'는 회귀다.

**6. 채널 렌더링.** `output/renderers.py`의 채널별 렌더러가 하나의 `DigestContent`를 Slack Block Kit이나 Threads 루트에 평면 답글 체인을 붙인 형태로 바꾼다. 포매팅은 프롬프트 규칙이 아니라 코드에 있다.

**7. 데일리 비주얼.** `DailyVisualMaker`는 다른 항목이 아니라 헤드라인 스토리를 그린다. 이미지와 lead와 텍스트가 같은 것을 가리키게 하려는 것이다. 에디터는 어떻게 그릴지를 브리핑하고 orientation을 고르며, `VisualGenerator`가 gpt-image-2로 렌더한다. Threads 게시도 이 Lambda가 담당하고, 렌더 실패가 다이제스트를 삼키지 않는다. 텍스트는 그대로 나간다.

**8. 딥 리서치 에이전트.** AgentCore Runtime 위의 자율 Strands 에이전트로 Slack 멘션으로 작동한다. 아래 도구 8개를 자유롭게 조합한다. 예컨대 "diffusion LLM 최신 동향"이면 `web_search`나 `search_papers`, `community_search`를 거쳐 `read_url`, `attach_image`, `deliver_report`로 간다.

| 도구 | 기능 |
|------|------|
| `web_search(query, recency)` | Tavily 공개 웹. 최신 뉴스는 `recency="news"` |
| `community_search(query)` | Reddit과 X, HN, Substack에 대한 Tavily 검색 |
| `search_papers(query)` | Semantic Scholar |
| `read_url(url)` | 1차 출처의 전문 fetch와 추출 |
| `recall_trends(query)` | `trends.json` 키워드 매칭, 모멘텀 순위 |
| `recall_digest(digest_date)` | 특정 날짜의 다이제스트가 담았던 내용. 다른 날짜로 폴백하지 않는다 |
| `attach_image(source_url)` | 출처의 OG 이미지를 내려받아 전달용으로 stage |
| `deliver_report(report, channel)` | 렌더링과 게시. Slack이 기본이고 Threads도 가능 |

전달은 프롬프트 규칙이 아니라 코드에서 채널을 인지해 처리한다. 에이전트가 아무 채널에도 전달하지 못한 채 끝나면 런타임이 리포트를 Slack에 폴백 게시한다.

## AWS 배포

### 배포하기

먼저 두 이미지를 모두 빌드해서 푸시하고([Docker 이미지](#docker-이미지) 참고) 푸시된 digest를 고정해 배포한다. 이미지 태그 문자열이 그대로면 CloudFormation이 Lambda를 재배포하지 않으니 `sha256` digest를 명시적으로 넘겨야 한다.

CDK CLI는 전역 `cdk`가 아니라 저장소에 핀된 것을 쓴다. CLI는 `pyproject.toml`의 `aws-cdk-lib`와 호환되는 버전으로 `package.json`에 핀돼 있고, 전역 CLI는 라이브러리보다 뒤처져 cloud-assembly 스키마 불일치로 실패할 수 있다.

```bash
npm install                                       # 1회. 핀된 CDK CLI 설치
export DIGEST_IMAGE_REF=sha256:<pushed-digest>    # AGENTCORE_IMAGE_REF은 기본 :arm64
AWS_PROFILE=<profile> npx cdk deploy --all -a "uv run python scripts/deploy.py"
AWS_PROFILE=<profile> uv run python scripts/put_secrets.py             # 이어서 시크릿 기록
AWS_PROFILE=<profile> uv run python scripts/put_secrets.py --verify    # 읽기 전용. 미설정이 남았나?
AWS_PROFILE=<profile> uv run python scripts/put_inference_profiles.py  # 계정/스테이지당 1회
```

### 생성되는 리소스

| 리소스 | 용도 |
|--------|------|
| **Lambda** (Docker) | 다이제스트 파이프라인, 15분 타임아웃 |
| **Lambda** (Docker) | 데일리 비주얼, 15분 타임아웃. 비동기이고 다이제스트 크리티컬 패스 밖이며 Threads의 유일한 게시 경로다 |
| **Lambda** | Slack 이벤트 핸들러, 60초 타임아웃. 유일한 인터넷 노출 경로라 전용 최소권한 역할을 쓴다 |
| **Lambda** (Docker) | Threads 토큰 갱신. 약 50일 주기로 돌며 갱신된 60일 토큰을 SSM에 재기록한다 |
| **API Gateway** + **WAFv2** | 레이트 리밋과 매니지드 룰셋, 스테이지 스로틀링이 적용된 `POST /slack/events` |
| **EventBridge** | 일간 다이제스트 크론(설정 기반 시와 분, UTC)과 토큰 갱신 스케줄 |
| **Bedrock AgentCore** | Runtime(에이전트, arm64)과 Memory(다이제스트 스냅샷) |
| **ECS Fargate** | RSSHub 컨테이너. `aws.rsshub_desired_count`는 기본 0이다. 다이제스트가 S3 park 파일을 먼저 읽고 이 서비스에 도달하지 않으니 상시 실행은 월 약 $40의 순수 비용이었다. 태스크 정의는 그대로 배포되니 1로 올리면 AWS 폴백이 복구된다 |
| **SSM Parameter Store** | 모든 시크릿. out-of-band로 기록한 SecureString이다 |
| **S3** | 트렌드와 park 파일, Threads 이미지 호스팅 |
| **DynamoDB** | Slack 이벤트 중복 제거 |
| **SQS** | 비동기 DLQ. 모든 Lambda가 `retry_attempts=0`인데, Threads에는 멱등 키가 없어 재시도가 이중 게시를 만들기 때문이다. 핸들러가 예외를 다시 raise하니 실패 건이 여기 남아 리플레이할 수 있다 |
| **SNS** | 알림 토픽(이메일) |
| **CloudWatch** | 구조화된 로그(보존 1개월)와 알람 12개. Lambda별 Errors 4개와 Timeout 4개, API 5xx, EmptyDigest, 비동기 DLQ, AgentErrors다 |
| **ECR** | Docker 이미지. Lambda용 amd64와 AgentCore용 arm64 |

### AWS의 시크릿

스택은 각 SSM 파라미터를 플레이스홀더 값으로 만들어두고, 배포 후 `scripts/put_secrets.py`가 `.env`의 실제 값을 SecureString으로 기록한다. 재배포가 값을 되돌리지 않는데, CloudFormation은 템플릿 속성이 바뀐 리소스만 갱신하고 플레이스홀더는 바뀌지 않기 때문이다.

알아둘 동작이 네 가지 있다.

- **이미 SecureString인 파라미터는 건너뛴다.** Threads 토큰은 갱신 Lambda가 제자리에서 회전시키니, 로컬 `.env` 사본을 다시 쓰면 만료된 토큰으로 되돌아간다. `--force`는 살아 있는 값을 정말 덮어쓸 때만 쓴다.
- **비어 있거나 없는 환경 변수는 건너뛰고 절대 지우지 않는다.** 그래서 반쪽짜리 `.env`가 잘 동작하는 파라미터를 날릴 수 없다. 여기에 `resolve_secret()`이 플레이스홀더를 미설정으로 취급하니, `put_secrets.py`를 잊고 배포해도 플레이스홀더를 API 토큰으로 보내는 대신 정상적인 '자격증명 없음' 경로로 degrade한다.
- **한 파라미터가 거절돼도 실행 전체가 죽지 않는다.** 실패한 것은 `FAILED` 목록에 담기고 스크립트는 non-zero로 끝나지만, 나머지 시크릿은 모두 기록된다.
- **`--verify`는 읽기 전용 리포트다.** 어떤 파라미터가 설정됐고, 어떤 것이 아직 플레이스홀더이고, 어떤 것이 평문 `String`이며, 어떤 것이 아예 없는지 알려준다. 언제든 프로덕션에 대고 돌려도 안전하다.

X 세션 쿠키는 Fargate 태스크 정의의 `secrets` 블록으로 컨테이너에 도달한다. 템플릿에는 ARN만 들어가고 값은 태스크 시작 시 ECS 에이전트가 가져온다.

### Bedrock 비용 귀속

온디맨드 Bedrock은 과금 대상 리소스가 없어서 `InvokeModel` 토큰 지출에 비용 할당 태그를 붙일 수 없다. 여러 워크로드가 같은 계정을 쓰면 Bedrock 청구는 하나의 귀속 불가 총액이 된다. application inference profile은 태그가 붙고, 그 ARN으로 호출하면 사용량이 귀속된다.

`scripts/put_inference_profiles.py`가 설정된 모델마다 하나씩 만든다. `Project`와 `Stage` 태그를 달고, 같은 글로벌 라우팅을 물려받도록 시스템 정의 크로스리전 프로필을 copy한다. `BedrockCrossRegionModelHelper`가 해석 시점에 이 프로필을 우선하는데, 그 리졸버는 LangChain 팩토리와 Strands 에이전트가 둘 다 거치는 한 곳이라 에이전트 지출까지 함께 잡힌다. 프로필이 없거나 조회가 거부되면 조용히 시스템 정의 id로 폴백한다. 비용 리포팅이 생성을 막아서는 안 된다.

주의할 것이 두 가지 있다.

- `application-inference-profile`은 `inference-profile`과 다른 IAM 리소스 타입이다. 정책은 둘 다 부여하며, 앞의 것을 빠뜨리면 프로필이 존재하는 순간 모든 Bedrock 호출이 AccessDenied가 된다.
- Cost Explorer까지 도달하려면 Billing에서 `Project`를 비용 할당 태그로 활성화해야 한다. 최대 24시간이 걸리고 소급 적용되지 않는다.

이를 보완해, 모든 `get_model()` 호출은 `stage=`를 받아 `LLM usage stage=... model=... input=... output=...`을 로깅한다. 청구는 모델 단위인데 다이제스트와 그라운딩, 트렌드 분류, 비주얼 에디터, 쿼리 정제, 리서치 에이전트가 Sonnet 5 하나를 공유하기 때문이다.

### Docker 이미지

두 이미지 모두 `uv.lock`이 핀한 정확한 집합을 설치한다(`uv export`로 뽑아 `uv pip install --system`하고 프로젝트 자신은 `--no-deps`). 그래서 CI가 테스트한 적 없는 의존성 집합이 이미지에서 도는 일이 있을 수 없다. 의존성은 소스 COPY보다 먼저 설치되니 코드만 바뀐 변경은 그 레이어를 재사용한다. 둘 다 non-root(uid 10001)로 돌고, `.dockerignore`가 `.env`와 `.venv`, `logs/`, `cdk.out`을 빌드 컨텍스트에서 제외한다.

```bash
# Lambda (amd64)
docker build --platform linux/amd64 --provenance=false -t <ecr-uri>:latest .
docker push <ecr-uri>:latest

# AgentCore (arm64)
docker buildx build --platform linux/arm64 --provenance=false \
  -f Dockerfile.agentcore -t <ecr-uri>:arm64 . --push
```

### Cloudflare Workers 프록시

Reddit과 YouTube는 AWS 데이터센터 IP에서 차단되니 Cloudflare Worker가 앞에 선다.

```bash
cd cloudflare-proxy
npx wrangler login
npx wrangler secret put PROXY_TOKEN   # 시크릿으로 넣는다. wrangler.toml의 [vars]가 아니다
npx wrangler deploy
```

이 워커는 의도적으로 범용 프록시가 아니다. `ALLOWED_HOSTS` 변수에 있는 호스트만 fetch하고(정확 일치 또는 suffix 일치) 나머지는 `403`이다. 호출자가 보낸 `headers` 블롭은 아웃바운드 요청에 병합되지 않으니 토큰 소지자가 `Cookie`나 `Authorization`, `Host`를 위조할 수 없다. 토큰이 쿼리 문자열에 있는 것은 의도적인데, 두 호출자 모두 프록시 URL을 `feedparser.parse`에 그대로 넘기고 그쪽은 헤더를 붙일 수 없기 때문이다.

### 로컬 크론

X/Twitter와 YouTube 자막은 둘 다 데이터센터 IP에서 차단되니, AWS 다이제스트가 돌기 전에 로컬에서 수집해 S3로 동기화해야 한다. 다이제스트 크론은 `aws.digest_cron_hour`와 `minute`을 UTC로 해석하니 기본값 `10:00 UTC`는 `19:00 KST`다. sync를 그보다 조금 앞에 둔다.

```bash
crontab -e
# 매일 18:30 KST. 19:00 KST(10:00 UTC) 다이제스트보다 30분 앞이다
30 18 * * * /path/to/omnisummary/scripts/sync_all_to_s3.sh >> /tmp/omnisummary-sync.log 2>&1
```

`sync_all_to_s3.sh`는 `AWS_PROFILE=research`를 기본값으로 두고, 크론이 최소한의 PATH로 도니 흔한 `uv` 설치 경로를 `PATH` 앞에 붙인다. X sync는 로컬 RSSHub 컨테이너가 떠 있어야 하고 YouTube sync는 `YOUTUBE_API_KEY`가 필요하다. 두 sync는 독립적이라 RSSHub가 죽어도 YouTube를 막지 않는다.

### 외부 서비스

| 서비스 | 용도 | 비용 |
|--------|------|------|
| AWS Bedrock | LLM (Claude Opus/Sonnet) | 사용량 기반 |
| OpenAI | gpt-image-2 데일리 비주얼 | 사용량 기반 |
| Cloudflare Workers | Reddit과 YouTube용 HTTP 프록시 | 무료 (100K req/day) |
| Tavily | 웹 검색 | 무료 티어 |
| Semantic Scholar | 논문 검색 | 무료 |
| YouTube Data API v3 | 영상 메타데이터 | 무료 (10K units/day) |
| Slack | 전달과 에이전트 트리거 | 무료 |
| Threads (Meta) | 전달 | 무료 |

## 프로젝트 구조

```
omnisummary/
├── main.py                  # CLI 진입점 (다이제스트 파이프라인)
├── research_cli.py          # 딥 리서치 에이전트 로컬 실행
├── Dockerfile               # Lambda (amd64)
├── Dockerfile.agentcore     # AgentCore (arm64)
├── collectors/              # RSS, Reddit, RSSHub(X), YouTube, WebSearch, S3 park 로더
├── pipeline/                # Aggregator, Ranker, DigestGenerator, TrendTracker, DailyVisualMaker
├── agent/                   # 딥 리서치 에이전트와 도구 8개, VisualGenerator, DigestStateManager
├── agent_runtime/           # Bedrock AgentCore HTTP 서버
├── shared/                  # 설정, 모델, 프롬프트, 상태 저장소, AgentCore 메모리, research, media
├── output/                  # 채널별 렌더러, Slack과 Threads 핸들러, 전달 라우팅
├── lambda_handlers/         # digest, slack events, daily visual, threads refresh
├── infrastructure/          # CDK 스택 (foundation, application)
├── scripts/                 # deploy, put_secrets, put_inference_profiles, ci_synth, sync
├── cloudflare-proxy/        # CF Worker 프록시
├── config/                  # YAML 설정
├── tests/                   # 단위 테스트와 CDK 어서션 테스트
└── docs/                    # design.md와 diagrams/
```

## 테스트와 CI

```bash
uv run python -m pytest tests/ -v        # 단위 테스트와 CDK 어서션 테스트 (hermetic: 네트워크와 AWS 없이)
uv run black --check . && uv run ruff check . && uv run mypy .
uv lock --check                          # 락파일이 pyproject와 일치하는지
uv run python scripts/ci_synth.py        # 오프라인 CDK synth
uv run pre-commit install                # 1회. 푸시 전에 CI의 게이트를 돌린다
```

테스트 스위트는 hermetic하다. `tests/conftest.py`가 앰비언트 시크릿과 인프라 env를 비우고 SSM 클라이언트를 막아서, 결과가 개발자의 `.env`나 AWS 프로파일에 좌우되지 않는다.

CI(`.github/workflows/ci.yml`)는 다섯 개 잡을 돈다. lint & type-check(`uv lock --check`와 ruff, black `--check`, 레포 전체 `mypy .`), CDK synth(핀된 CLI로 오프라인, 추적되는 `config-template.yaml` 대상), tests & coverage(범위와 `fail_under`는 `pyproject.toml`에 있다), Docker build & import check, dependency & secret scan이다. 모든 잡에 `timeout-minutes`가 붙어 있고 uv와 npm 캐시는 락파일로 키를 잡는다.

이 중 둘은 설명이 필요하다.

- **import 체크**는 빌드된 각 이미지를 올려 `--network none`과 자격증명 없이 실제 엔트리 모듈을 import한다. 빌드만으로는 import가 한 번도 실행되지 않으니, 예전에는 COPY 누락이나 import 시점의 AWS/HTTP 호출이 CI가 아니라 콜드 스타트에서 깨졌다.
- **의존성 스캔**은 설치된 트리를 감사한다(`uv sync --frozen --no-dev --no-install-project` 후 `pip-audit --strict --path .venv/...`). 그것이 이미지가 설치하는 집합이고, 실제 배포되는 플랫폼으로 이미 좁혀진 집합이기 때문이다. `--no-install-project`가 중요하다. pip-audit은 editable 배포를 감사 불가로 보고하고 `--strict`가 그것을 실패로 바꾼다. `gitleaks`는 전체 히스토리에 돈다. shallow clone은 tip만 보니 과거에 커밋되고 나중에 지워진 키를 절대 찾지 못한다.

  운영자를 위한 주의사항이 있다. `pip-audit --strict`는 잠긴 집합에 알려진 권고가 0건이기를 요구하니, 전이 의존성에 새 권고가 공개되면 무관한 다음 푸시가 빨개진다. 해소 순서는 `uv lock --upgrade`, 미사용 의존성 제거, 상한 완화이며 `--ignore-vuln`은 코드 경로가 도달 불가임을 확인한 뒤의 마지막 수단이다.

## 문서

[docs/design.md](docs/design.md)가 줄 단위 설계·기술 레퍼런스이고 "왜 이런 모양인가"에 대한 답이 있는 곳이다. 개발 가이드라인과 동작에 영향을 주는 함정들은 `.claude/CLAUDE.md`에 있는데, 이 파일은 추적되지 않고 체크아웃에만 존재한다.

## 라이선스

[MIT License](LICENSE)
