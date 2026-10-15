<div align="center">

# 🗞️ OmniSummary

**능동형 AI/ML 데일리 다이제스트. 다섯 계열의 소스에서 콘텐츠를 모아 LLM으로 그날 가장 강한 이야기를 고르고, 한국어 에디토리얼 다이제스트를 써서 헤드라인을 그린 한국어 4컷 만화와 함께 Threads에 게시한다. 로컬 딥 리서치 CLI(OpenClaw 스킬이 쓴다)는 어떤 주제든 웹과 논문과 커뮤니티를 조사해 페르소나 보이스의 인용 기반 리포트를 돌려준다.**

AWS 위에서 동작한다. Amazon Bedrock(Claude)과 Bedrock AgentCore Memory를 쓴다.

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

**데일리 다이제스트**는 크론으로 돈다. RSS와 Substack, Reddit, YouTube, X/Twitter, 웹 검색에서 콘텐츠를 모아 중복을 제거하고, Claude Opus 5.5로 전부 채점해 후보 풀을 만든 뒤, Claude Sonnet 5.5가 가장 강한 다섯 이야기를 골라 한국어 에디토리얼 다이제스트를 쓴다. 헤드라인은 한국어 4컷 만화가 되고, 만화와 이야기들이 한 세트로 Threads에 나간다.

**딥 리서치 CLI**는 로컬에서 돈다. 자유형 주제를 받으면 열린 웹과 학술 문헌, 커뮤니티 논의를 스스로 조사한 뒤 다이제스트와 같은 내레이터 보이스로 인용 기반 한국어 리포트를 쓴다. 결과는 stdout에 JSON 한 개로 나오고 OpenClaw 스킬이 그것을 읽는다. Threads 게시는 `--publish threads`를 줄 때만 한다.

![다이제스트 동작 방식](docs/diagrams/concept-pipeline.png)

## 주요 기능

- **멀티 소스 수집.** Reddit(공개 `.rss` 피드를 다중 서브레딧 요청 한 번으로), YouTube, X/Twitter(RSSHub 경유), RSS/Substack, 웹 검색(Tavily). 피드 본문의 HTML은 수집 시점에 벗긴다
- **후보 풀을 만드는 LLM 랭킹.** Claude Opus 5.5가 다축으로 채점하고, 소스 슬롯과 출처별 상한이 모든 소스의 최선 항목을 에디터 앞에 놓는다. 무엇을 내보낼지는 에디터가 고른다
- **에디토리얼 다이제스트.** Claude Sonnet 5.5가 일자 간 트렌드 연속성을 살려 한국어 산문을 쓰고, 그라운딩 패스가 구체적 주장을 원문과 대조한다
- **데일리 만화.** 오너 페르소나 캐릭터(기준 시트 이미지로 모습을 고정)가 화자인 2x2 4컷 만화를 `gpt-image-2.5-sunburst`가 한국어 대사로 그린다
- **Threads 전달.** 이미지 루트에 평면 답글 체인을 붙이고, 하루 한 번만 게시한다
- **딥 리서치 CLI와 OpenClaw 스킬.** 단일 목적 도구 8개를 가진 자율 Strands 에이전트다. stdout에 JSON 계약, 디스크에 Markdown 리포트를 남긴다
- **운영 안정성.** 소스별 헬스 보고(`OK`, `EMPTY`, `FAILED`, `STALE`, `DEGRADED`)를 SNS 이메일로 알리고, 상관 ID가 붙는 구조화된 JSON 로그와 CloudWatch 알람 8개를 둔다
- **AWS 배포.** Lambda와 EventBridge 크론, Bedrock AgentCore Memory, ECS(RSSHub)를 모두 CDK로 정의한다

## 빠른 시작

### 사전 준비

- Python 3.12 이상과 [uv](https://docs.astral.sh/uv/)
- Docker (RSSHub와 AWS 배포용)
- Bedrock 접근 권한이 있는 AWS 계정

### 설치

```bash
git clone <repo-url> && cd omnisummary
uv sync
cp config/config-template.yaml config/config.yaml
cp .env.template .env
```

### 설정

`config/config.yaml`에는 시크릿이 아닌 모든 것이 들어간다. 모든 설정 모델이 모르는 키를 거부하니, 코드에서 지운 키는 내 `config.yaml`에서도 지워야 한다. 눈여겨볼 노브는 이렇다.

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
    lookback_hours: 30      # 직전 실행까지 닿아야 한다. config가 "48 - 실행 시각"보다 작은 값을 거부한다
  web_search:
    enabled: true
    exclude_domains: ["finance.biggo.com"]   # broad·정제 검색 어디서도 돌려받지 않는 호스트
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
  ranking_model: "anthropic.claude-opus-5-5"
  digest_model: "anthropic.claude-sonnet-5-5"
  max_per_origin: 1        # 채널, 작성자, 서브레딧, 피드, 웹 호스트당 상한
  source_slots: {web: 1, x: 1, rss: 1, reddit: 1, youtube: 1}
```

모든 필드는 [design.md §3](docs/design.md#3-설정)에 설명되어 있다.

### 시크릿

시크릿은 로컬에서는 `.env`(`.env.template` 참고)에, AWS에서는 SSM Parameter Store에 둔다. 돌리는 데 꼭 필요한 것은 Tavily 키 하나다. 나머지는 각각 특정 기능 하나를 켜며, 키가 없으면 그 기능이 로그 한 줄을 남기고 건너뛴다.

| 변수 | 필요한 곳 |
|----------|-----------|
| `TAVILY_API_KEY` | `web_search` 수집기와 리서치 CLI의 웹·커뮤니티·뉴스 검색 |
| `YOUTUBE_API_KEY` | YouTube 수집기(없으면 RSS 폴백, 자막 없음) |
| `OPENAI_API_KEY` | 데일리 만화 렌더 |
| `THREADS_ACCESS_TOKEN` / `THREADS_USER_ID` | Threads 전달(60일 토큰, AWS에서는 SSM으로 자동 갱신) |
| `ALERT_EMAIL` | 소스 헬스 SNS 이메일 알림(AWS) |
| `TWITTER_AUTH_TOKEN` / `TWITTER_CT0` | RSSHub 경유 X/Twitter. 본인 x.com 세션 쿠키 |
| `S3_SYNC_ACCESS_KEY_ID` / `S3_SYNC_SECRET_ACCESS_KEY` | 로컬 → S3 sync 전용 자격증명(선택. 없으면 `AWS_PROFILE`) |

> **시크릿은 CDK 스택을 통과하지 않는다.** CloudFormation 템플릿은 SecureString을 담을 수 없어서,
> 값을 스택에 넘기면 `cdk.out`과 스테이징 버킷, 모든 `GetTemplate` 응답에 평문으로 남는다. 스택은
> 플레이스홀더를 담은 파라미터 경로만 만들고, 실제 값은 배포 후 `scripts/put_secrets.py`가
> SecureString으로 기록한다. [AWS의 시크릿](#aws의-시크릿) 참고.

### 설정 체크리스트

다이제스트를 **로컬에서** 만들려면(X 없음, 만화 없음, 게시 없음)

1. `uv sync`를 돌리고 위처럼 템플릿 파일 두 개를 복사한다.
2. `.env`에 `TAVILY_API_KEY`를 채운다.
3. `aws.bedrock_region`(기본 `us-west-2`)에서 Bedrock에 닿는지 확인하고 `AWS_PROFILE`이나 표준 AWS 자격증명을 설정한다. 로컬 실행이어도 랭킹과 다이제스트 LLM은 Bedrock에서 돈다.
4. `uv run python main.py --dry-run --sources web_search`가 다이제스트를 출력한다. 템플릿에는 RSS 피드와 서브레딧이 없지만 트렌드 검색은 들어 있어서, 바로 동작하는 소스는 Tavily 하나다.

그다음 기능을 하나씩 붙인다.

| 원하는 것 | 설정 | 비고 |
|-------|-----|-------|
| 자막이 붙은 YouTube 항목 | `YOUTUBE_API_KEY`와 로컬 sync | 자막은 거주용 IP에서만 받아진다 |
| X/Twitter 항목 | RSSHub 컨테이너와 로컬 sync | 아래 참고 |
| 데일리 만화 | `OPENAI_API_KEY` | `gpt-image-2.5-sunburst` |
| Threads 전달 | `THREADS_ACCESS_TOKEN`, `THREADS_USER_ID`, `enable_threads_post: true` | 이미지와 텍스트가 한 게시 세트로 나가야 해서 Threads 게시는 데일리 비주얼 Lambda가 한다. 만화를 렌더하지 못하면 텍스트만 나간다 |

### RSSHub 컨테이너 (X/Twitter)

X는 로컬 [RSSHub](https://docs.rsshub.app/) 컨테이너로 읽는다. 로그인된 x.com 세션의 쿠키 두 개, **`auth_token`**과 **`ct0`**이 필요하다.

브라우저 개발자 도구에서 가져온다(macOS에서는 F12가 다른 기능에 묶여 있는 경우가 많으니 ⌥⌘I를 쓴다). **Chrome**은 Application → Cookies → `https://x.com`, **Safari**(먼저 개발자용 메뉴를 켠다)와 **Firefox**는 Storage → Cookies → `x.com`이다.

```bash
docker run -d --name rsshub --restart unless-stopped -p 1200:1200 \
  -e NODE_ENV=production -e CACHE_TYPE=memory \
  -e TWITTER_AUTH_TOKEN='<auth_token>' \
  -e TWITTER_CT0='<ct0>' \
  diygod/rsshub:latest

curl -s "http://localhost:1200/twitter/user/karpathy" | head   # 스모크 테스트
```

쿠키 없이도 컨테이너는 뜨지만 X 피드가 비어서 돌아온다. 쿠키는 때때로 만료된다. RSSHub 실패율이 올라가면(경고로 로깅된다) 쿠키를 갱신하고 컨테이너를 다시 만든다. `--restart unless-stopped`는 명시적으로 멈춘 컨테이너를 다시 띄우지 않으니, `localhost:1200`이 응답하지 않으면 sync 스크립트가 컨테이너를 켠다. AWS에서는 같은 이미지를 ECS Fargate로 돌릴 수 있다.

## 사용법

```bash
# 다이제스트 파이프라인. 아무것도 저장하거나 게시하지 않는다
uv run python main.py --dry-run --sources rss reddit

# 전체 파이프라인과 만화, Threads(켜져 있을 때)
uv run python main.py

# 딥 리서치. stdout에 JSON 한 개, Markdown 리포트는 reports/에 저장
uv run python research_cli.py "<주제>"
uv run python research_cli.py "<주제>" --publish threads   # Threads 스레드로도 게시

# 다이제스트가 이미 게시한 것. 무료 읽기 전용이고 stdout에 JSON 한 개
uv run python recall_cli.py digest 2026-10-11   # 그날 lead와 스토리(기본값은 오늘)
uv run python recall_cli.py trends              # 추적 중인 일자 간 트렌드, 강한 순

# 데이터센터 IP에서 차단되는 소스의 로컬 → S3 sync (X/RSSHub와 YouTube 자막)
./scripts/sync_all_to_s3.sh                  # 둘 다. 한쪽이 실패해도 다른 쪽은 막지 않는다
uv run python scripts/sync_to_s3.py rsshub   # X만
uv run python scripts/sync_to_s3.py youtube  # YouTube만
```

| 플래그(`main.py`) | 설명 |
|------|-------------|
| `--sources rss reddit youtube` | 특정 소스만 선택 |
| `--dry-run` | 다이제스트를 출력만 하고 저장·게시하지 않는다 |
| `--top-n 5` | 게시할 이야기 수를 재정의 |
| `--date 2026-03-28` | 다이제스트 날짜 지정(기본: 오늘, KST) |
| `--pin-url <url> [<url> ...]` | 점수와 무관하게 URL을 상위 스토리에 강제로 넣는다. YouTube URL은 Data API로, 나머지는 Tavily로 해석한다. 로컬 CLI 전용 |
| `--force-republish` | 오늘 다이제스트가 이미 나갔어도 다시 게시한다(Threads 멱등 가드를 우회) |

## 파이프라인 동작 방식

단계별 요약이다. [design.md](docs/design.md)가 줄 단위 레퍼런스이며 각 부분이 *왜* 그런 모양인지 설명한다.

**1. 수집.** 모든 수집기가 각자의 조회 윈도를 갖고 비동기 병렬로 돌며 자기 헬스를 스스로 보고한다. 두 소스는 특별하다. X/Twitter와 YouTube 자막은 데이터센터 IP에서 차단되니, 로컬 크론이 거주용 IP로 수집해 S3에 park해두고 Lambda가 그 파일을 읽는다. 나이 예산을 넘긴 park 파일이면 소스가 `STALE`로 보고되니 멈춘 크론이 건강한 것처럼 보이지 않는다.

| 수집기 | 소스 | 방식 |
|-----------|--------|--------|
| `RedditCollector` | Reddit 공개 `.rss` | `r/A+B` 요청 한 번(top이면 `t=day`). 재시도를 붙여 직접 가져온다 |
| `YouTubeCollector` | YouTube Data API v3 | AWS에서는 S3 park 파일, 그 외에는 라이브. 정규 영상 목록(`UULF`)을 읽어 Shorts가 채널 자리를 차지하지 않는다 |
| `RSSCollector` | RSS/Atom | httpx와 feedparser, HTML 제거. 요약만 있는 엔트리는 Tavily extract로 페이지 본문을 받는다 |
| `RSSHubCollector` | RSSHub 경유 X/Twitter | AWS에서는 S3 park 파일, 그 외에는 로컬 Docker |
| `WebSearchCollector` | Tavily | 직접 호출, LLM 쿼리 정제, 콘텐츠 팜은 `exclude_domains`로 제외 |

**2. 집계.** `ContentAggregator`가 URL과 정규화된 제목으로 중복을 제거한다(앞머리의 `RT 이름:`을 떼어 셀프 리트윗이 원문과 겹치게 한다). 중복이 걸리면 먼저 도착한 쪽이 아니라 더 좋은 쪽을 남긴다. 핀이 먼저고, 그다음 본문이 더 긴 쪽, 그다음 먼저 온 쪽 순이다.

**3. 랭킹.** `ContentRanker`가 Claude Opus 5.5로 기술적 실속과 실무자 가치, 신규성, 산업 영향, 연구 의의, 소스 권위를 채점하고 홍보성이나 빈약한 콘텐츠는 하드 필터로 걸러낸다. 랭커는 최근 다이제스트가 실은 스토리 제목도 받는다. 그래서 새 URL로 돌아온 이야기는 실질적 새 진전이 없으면 재탕으로 채점된다. 그다음 `top_n + digest_candidate_buffer`(5 + 7)개의 후보 풀을 만든다. `source_slots`가 소스마다 `min_score` 이상 최선 항목의 자리를 주고, 나머지 풀은 점수 순으로 채우되 `max_per_origin`이 특정 채널이나 작성자, 피드, 호스트를 상한으로 묶는다. 조용한 날 풀이 미달일 때만 그 상한을 푼다.

**4. 트렌드 추적.** `TrendTracker`가 구조화된 트렌드를 `trends.json`에 유지한다. LLM은 오늘 항목이 기존 트렌드의 연장인지 신규인지 분류만 하고, 장부 관리는 전부 결정론적 Python이 한다. 날짜 스탬프와 active/cooling 생명주기, 보존 기간 지난 트렌드 삭제, 최신성 감쇠 모멘텀, 증거 캡이 그렇다. active와 cooling 트렌드는 다음 날 다이제스트에 투입되며 이것이 일자 간 연속성의 근원이다. 다이제스트 날짜 이전 증거만 쓰므로 같은 날 재실행이 첫 실행을 과거로 다시 읽지 않는다. 기억은 요약을 다시 쓰는 방식이 아니라 상한으로 압축한다. 트렌드당 날짜가 붙은 증거 최대 5줄, 프롬프트에는 모멘텀 상위 10개, 30일 동안 보이지 않은 트렌드는 삭제한다.

**5. 다이제스트 생성.** `DigestGenerator`에서 에디터가 후보 풀에서 가장 강한 다섯 이야기를 고르고(같은 사건은 병합한다) 구조화된 `DigestContent`를 만든다. `lead` 하나와, 각각 title, url, body, implication을 가진 `items[]`다. 에디터는 중복 제거 창 안의 모든 다이제스트 제목과 최근 리드의 마지막(논지) 문장을 받으니, 주간 요약이 새 URL로 다시 가져온 이야기를 또 쓰지도, 같은 논지를 되풀이하지도 않는다. LLM은 산문만 쓴다. 마크업도 소스 태그도 쓰지 않는다. 산문 예산은 코드가 소유한 고정 파트에서 코드가 계산하니 항목이 Threads의 500자 한도를 넘길 수 없다.

> ⚠️ **프롬프트의 JSON 키 순서는 load-bearing이다.** `items`를 먼저, `lead`를 마지막에 요청해서 lead가 이미 쓰인 스토리에 대한 논평이 되게 한다. 헤드라인 답글과의 단어 겹침이 0.21–0.41에서 0.03–0.21로 떨어졌다. `DigestContent`의 필드 순서에 맞춘다고 요청 키를 재배열하지 **말 것.** 그 정리는 회귀다.

**6. Threads 렌더링.** `output/renderers.py`가 하나의 `DigestContent`를 Threads 루트에 평면 답글 체인을 붙인 형태로 바꾼다. 포매팅은 프롬프트 규칙이 아니라 코드에 있다. 루트는 lead 아래에 나머지 스토리 제목을 적는다. 스토리 답글은 루트 조회의 2~5%만 읽히기 때문이다. 루트는 토픽 태그 하나(`pipeline.threads_topic_tag`, 기본 `AI`)를 달아 그 토픽 피드와 태그 검색에 나오고, 만화에는 컷마다 장면과 대사를 적은 한국어 대체 텍스트가 붙는다.

**7. 데일리 만화.** `DailyVisualMaker`는 헤드라인 스토리를 그린다. 이미지와 lead와 텍스트가 같은 것을 가리키게 하려는 것이다. 에디터가 오너 페르소나 캐릭터가 이끄는 2x2 기승전결 4컷(결말은 교훈이 아니라 사실에서 나온 시각 개그)과 컷마다 짧은 한국어 대사 한 줄을 쓰고, `VisualGenerator`가 이를 이미지 브리프로 바꿔 `gpt-image-2.5-sunburst`(xhigh)로 렌더한다(2x2는 1024x1536. 이야기에 맞으면 에디터가 848x2544 세로 스트립이나 다른 구도를 고른다). 화자가 등장하는 날(기본)에는 페르소나 기준 시트를 입력 이미지로 함께 보내 모습을 고정한다. 이미지는 그린 그대로 게시한다. Threads 게시도 이 Lambda가 담당하고, 렌더 실패가 다이제스트를 삼키지 않는다. 텍스트는 그대로 나간다.

**8. 딥 리서치.** `research_cli.py`가 실행하는 자율 Strands 에이전트다. 아래 도구 8개를 자유롭게 조합한다. 예컨대 "diffusion LLM 최신 동향"이면 `web_search`나 `search_papers`, `community_search`를 거쳐 `read_url`, `attach_image`, `deliver_report`로 간다.

| 도구 | 기능 |
|------|----------|
| `web_search(query, news, time_range)` | Tavily 열린 웹 검색. 뉴스면 `news=True`, 기간은 `time_range`(day/week/month/year). 결과에 게시일을 보여준다 |
| `community_search(query)` | Reddit, X, HN, Substack 대상 Tavily 검색 |
| `search_papers(query)` | OpenAlex 논문 검색(키 없음). 게시일, 피인용 수, 초록 |
| `read_url(url)` | 1차 출처의 전문을 가져와 추출 |
| `recall_trends()` | `trends.json`의 추적 트렌드 전부를 모멘텀 순으로. 관련성은 에이전트가 판단한다 |
| `recall_digest(digest_date)` | 특정 날짜 다이제스트가 실었던 것. 다른 날짜로 폴백하지 않는다 |
| `attach_image(source_url)` | 출처의 OG 이미지를 내려받아 전달용으로 stage |
| `deliver_report(report, channel)` | Markdown 리포트를 넘기거나(`markdown`) Threads에 게시(`threads`, `--publish threads`일 때만) |

`deliver_report`는 어떤 도구도 돌려준 적 없는 URL을 인용한 리포트를 받지 않고, 문제의 URL을 열거해 에이전트가 고치게 한다.

## OpenClaw로 딥 리서치

OpenClaw 스킬은 `~/.openclaw/workspace/skills/omnisummary-research/SKILL.md`에 있다(사용자 정의 스킬의 정본은 워크스페이스이고 저장소에는 사본을 두지 않는다). 스킬이 CLI를 호출하고 결과를 전달한다.

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

종료 코드는 `0` 성공, `1` 실패(리포트 없음, 실행 오류, 요청한 Threads 게시 미완료), `2` 입력 거부다. 같은 스킬이 `recall_cli.py`(`digest [DATE]`, `trends`)로 지난 다이제스트를 비용 없이 되읽는다. `found: false`는 그 날짜에 저장된 것이 없다는 뜻이고, `status: failed`는 저장소를 읽지 못했다는 뜻이다. 로그는 stderr로 가고, `LOG_LEVEL`과 무관하게 서드파티 로거는 `WARNING`으로 고정한다. botocore DEBUG는 세션 토큰이 담긴 요청 헤더를 남기기 때문이다.

## AWS 배포

### 배포하기

이미지를 먼저 빌드해 푸시하고([Docker 이미지](#docker-이미지) 참고) 푸시된 digest를 고정해 배포한다. 이미지 *태그* 문자열이 그대로면 CloudFormation이 Lambda를 재배포하지 않으니 `sha256` digest를 명시해서 넘긴다.

글로벌 `cdk`가 아니라 레포에 고정된 CDK CLI를 쓴다. CLI는 `package.json`에 `pyproject.toml`의 `aws-cdk-lib`와 호환되는 버전으로 고정되어 있고, 글로벌 CLI는 라이브러리보다 뒤처져 cloud-assembly 스키마 미스매치로 실패할 수 있다.

**빈 계정에서는 순서가 중요하다.** 유일한 ECR 리포지토리를 foundation 스택이 만들고, application 스택의 Lambda가 거기서 이미지를 해석하기 때문이다.

```bash
npm install                                       # 1회. 고정된 CDK CLI 설치
export AWS_PROFILE=<profile>

# 1. 계정+리전당 1회. CDK 부트스트랩 리소스(스테이징 버킷, 역할) 생성
npx cdk bootstrap -a "uv run python scripts/deploy.py"

# 2. foundation을 먼저. 이미지를 푸시할 ECR 리포를 여기서 만든다
npx cdk deploy '*-foundation' -a "uv run python scripts/deploy.py"

# 3. 그 리포에 로그인하고 이미지를 푸시한다(아래 Docker 이미지 참고).
#    URI는 조회하는 게 아니라 규칙으로 만든다: <account>.dkr.ecr.<region>.amazonaws.com/<project>-<stage>-agent
ECR_URI="$(aws sts get-caller-identity --query Account --output text).dkr.ecr.<region>.amazonaws.com/omnisummary-<stage>-agent"
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin "${ECR_URI%%/*}"
docker build --platform linux/amd64 --provenance=false -t "$ECR_URI:latest" . && docker push "$ECR_URI:latest"

# 4. 푸시된 digest를 고정해 전체 배포
export DIGEST_IMAGE_REF=$(aws ecr describe-images --region <region> --repository-name omnisummary-<stage>-agent \
  --image-ids imageTag=latest --query 'imageDetails[0].imageDigest' --output text)
npx cdk deploy --all -a "uv run python scripts/deploy.py"

# 5. 시크릿과 비용 귀속
uv run python scripts/put_secrets.py             # 시크릿 기록
uv run python scripts/put_secrets.py --verify    # 읽기 전용. 비어 있는 게 남았나?
uv run python scripts/put_inference_profiles.py  # 계정/스테이지당 1회, 단계 모델을 바꾼 뒤에도 다시
```

이후 배포에서는 3-4단계만 반복한다. 새 이미지를 푸시하고 새 digest로 `deploy --all`을 돌린다.

### 생성되는 리소스

| 리소스 | 용도 |
|----------|---------|
| **Lambda** (Docker) | 다이제스트 파이프라인, 15분 타임아웃, VPC 안(RSSHub 서비스에 닿는다) |
| **Lambda** (Docker) | 데일리 만화와 Threads 게시, 15분 타임아웃. 비동기로 다이제스트 크리티컬 패스에서 벗어나 있고 VPC 밖이며, 유일한 Threads 게시 경로다 |
| **Lambda** (Docker) | Threads 토큰 갱신(약 50일 주기, 갱신된 60일 토큰을 SSM에 다시 기록) |
| **EventBridge** | 데일리 다이제스트 크론(config의 시와 분, **UTC**)과 토큰 갱신 스케줄 |
| **Bedrock AgentCore** | Memory(다이제스트 스냅샷, 날짜당 세션 하나) |
| **ECS Fargate** | RSSHub 컨테이너. `aws.rsshub_desired_count`는 **기본 0**이다. 다이제스트가 S3 park 파일을 먼저 읽고 이 서비스에 닿지 않으니 상시 가동은 월 약 $40의 순수 비용이다. 태스크 정의는 그대로 배포되니 1로 올리면 AWS 폴백이 복구된다 |
| **SSM Parameter Store** | 모든 시크릿, out-of-band로 기록한 SecureString |
| **S3** | 트렌드, park 파일, Threads 이미지 호스팅. 역할에는 프로젝트 prefix의 `GetObject`/`PutObject`와 `ListBucket`만 준다 |
| **SQS** | 다이제스트와 비주얼 Lambda의 비동기 DLQ. 모든 Lambda가 `retry_attempts=0`이다. Threads에는 멱등 키가 없어서 재시도가 이중 게시가 되기 때문이다. 핸들러가 예외를 다시 던지니 다이제스트와 비주얼의 실패 건이 여기 쌓여 리플레이를 기다린다. 토큰 갱신 Lambda는 DLQ 없이 Errors 알람으로 올라간다 |
| **SNS** | 알림 토픽(이메일) |
| **CloudWatch** | 구조화된 로그(1개월 보존)와 알람 8개(Lambda별 Errors ×3, Timeout ×3, EmptyDigest, 비동기 DLQ) |
| **ECR** | Lambda 이미지(amd64) |

### AWS의 시크릿

스택은 각 SSM 파라미터를 플레이스홀더 값으로 만들어두고, 배포 후 `scripts/put_secrets.py`가 `.env`의 실제 값을 SecureString으로 기록한다. 재배포가 값을 되돌리지 않는데, CloudFormation은 템플릿 속성이 바뀐 리소스만 갱신하고 플레이스홀더는 바뀌지 않기 때문이다.

알아둘 동작이 네 가지 있다.

- **이미 SecureString인 파라미터는 건너뛴다.** Threads 토큰은 갱신 Lambda가 제자리에서 회전시키니, 로컬 `.env` 사본을 다시 쓰면 만료된 토큰으로 되돌아간다. `--force`는 살아 있는 값을 정말 덮어쓸 때만 쓴다.
- **비어 있거나 없는 환경 변수는 건너뛰고 절대 지우지 않는다.** 그래서 반쪽짜리 `.env`가 잘 동작하는 파라미터를 날릴 수 없다. 여기에 `resolve_secret()`이 플레이스홀더를 미설정으로 취급하니, `put_secrets.py`를 잊고 배포해도 플레이스홀더를 API 토큰으로 보내는 대신 정상적인 '자격증명 없음' 경로로 degrade한다.
- **한 파라미터가 거절돼도 실행 전체가 죽지 않는다.** 실패한 것은 `FAILED` 목록에 담기고 스크립트는 non-zero로 끝나지만, 나머지 시크릿은 모두 기록된다.
- **`--verify`는 읽기 전용 리포트다.** 어떤 파라미터가 설정됐고, 어떤 것이 아직 플레이스홀더이고, 어떤 것이 평문 `String`이며, 어떤 것이 아예 없는지 알려준다. 언제든 프로덕션에 대고 돌려도 안전하다.

X 세션 쿠키는 Fargate 태스크 정의의 `secrets` 블록으로 컨테이너에 도달한다. 템플릿에는 ARN만 들어가고 값은 태스크 시작 시 ECS 에이전트가 가져온다.

### Bedrock 모델과 비용 귀속

| 단계 | 모델 |
|-------|-------|
| 랭킹 | Claude Opus 5.5 |
| 다이제스트, 그라운딩, 만화 에디터와 아트 디렉터 | Claude Sonnet 5.5 |
| 딥 리서치 | Claude Opus 5.5 |
| 트렌드, 쿼리 정제 | Claude Sonnet 5.5 |
| 토큰 카운트 | Claude Sonnet 4.6(Opus 계열은 CountTokens를 지원하지 않는다) |

Opus 5.5는 effort 기본값이 `medium`이라 팩토리가 `high`로 고정한다. 다이제스트는 Sonnet 5.5가 쓴다. Opus 5.5가 넘긴 길이 규격(리드 3~4문장)을 지키고 비용은 절반 아래다. Claude Fable 5.1은 등록하지 않았다. Bedrock이 계정의 기본 데이터 보존 모드에서 이 모델을 거부한다. 단계별 선택은 실제 후보 풀로 측정했다. [design.md §6](docs/design.md#모델-팩토리) 참고.

온디맨드 Bedrock은 과금 대상 리소스가 없어서 `InvokeModel` 토큰 지출에 비용 할당 태그를 붙일 수 없다. application inference profile은 태그가 붙고, 그 ARN으로 호출하면 사용량이 귀속된다. `scripts/put_inference_profiles.py`가 설정된 모델마다 하나씩 만든다. 이름과 `Project`·`Stage` 태그는 `config.aws.project_name`/`stage`에서 오고, 시스템 정의 크로스리전 프로필을 copy한다. `BedrockCrossRegionModelHelper`가 해석 시점에 이 프로필을 우선하며, LangChain 팩토리와 리서치 에이전트가 둘 다 이 경로를 거친다. 프로필이 없거나 조회가 거부되면 조용히 시스템 정의 id로 폴백한다. 비용 리포팅이 생성을 막아서는 안 된다.

주의할 것이 두 가지 있다.

- `application-inference-profile`은 `inference-profile`과 다른 IAM 리소스 타입이다. 정책은 둘 다 부여하며, 앞의 것을 빠뜨리면 프로필이 존재하는 순간 모든 Bedrock 호출이 AccessDenied가 된다.
- Cost Explorer까지 도달하려면 Billing에서 `Project`를 비용 할당 태그로 활성화해야 한다. 최대 24시간이 걸리고 소급 적용되지 않는다.

이를 보완해, 모든 `get_model()` 호출은 `stage=`를 받아 `LLM usage stage=... model=... input=... output=...`을 로깅한다. 청구는 모델 단위인데 여러 단계가 한 모델을 공유하기 때문이다. 리서치 CLI는 같은 토큰 수를 JSON의 `usage`로 보고한다.

### Docker 이미지

이미지는 `uv.lock`이 핀한 정확한 집합을 설치한다(`uv export`로 뽑아 `uv pip install --system`하고 프로젝트 자신은 `--no-deps`). 그래서 CI가 테스트한 적 없는 의존성 집합이 이미지에서 도는 일이 없다. 의존성은 소스 COPY보다 먼저 설치되니 코드만 바뀐 변경은 그 레이어를 재사용한다. non-root(uid 10001)로 돌고, `.dockerignore`가 `.env`와 `.venv`, `logs/`, `cdk.out`을 빌드 컨텍스트에서 제외한다. Lambda 세 개가 모두 이 이미지를 쓴다.

foundation 스택이 만든 ECR 리포지토리 `<account>.dkr.ecr.<region>.amazonaws.com/<project>-<stage>-agent`로 간다. 먼저 로그인해야 한다(Docker 크리덴셜은 만료되니 푸시가 401이면 다시 로그인).

```bash
aws ecr get-login-password --region <region> \
  | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com

docker build --platform linux/amd64 --provenance=false -t <ecr-uri>:latest .
docker push <ecr-uri>:latest
```

### 로컬 크론

X/Twitter와 YouTube 자막은 둘 다 데이터센터 IP에서 차단되니, AWS 다이제스트가 돌기 전에 로컬에서 수집해 S3로 동기화해야 한다. 다이제스트 크론은 `aws.digest_cron_hour`와 `minute`을 UTC로 해석하니 기본값 `10:00 UTC`는 `19:00 KST`다. sync를 그보다 조금 앞에 둔다.

```bash
crontab -e
# 매일 18:30 KST. 19:00 KST(10:00 UTC) 다이제스트보다 30분 앞이다
30 18 * * * /path/to/omnisummary/scripts/sync_all_to_s3.sh >> /tmp/omnisummary-sync.log 2>&1
```

`sync_all_to_s3.sh`는 `AWS_PROFILE=research`를 기본값으로 두고, 크론이 최소한의 PATH로 도니 흔한 `uv` 설치 경로를 `PATH` 앞에 붙이며, `localhost:1200`이 응답하지 않으면 `rsshub` 컨테이너를 켠다. X sync는 RSSHub가 필요하고 YouTube sync는 `YOUTUBE_API_KEY`가 필요하다. 두 sync는 독립적이라 RSSHub가 죽어도 YouTube를 막지 않는다.

### 외부 서비스

| 서비스 | 용도 | 비용 |
|--------|------|------|
| AWS Bedrock | LLM (Claude Opus/Sonnet) | 사용량 기반 |
| OpenAI | `gpt-image-2.5-sunburst` 데일리 만화(xhigh, 하루 렌더 1회) | 사용량 기반 |
| Tavily | 웹 검색 | 무료 티어 |
| OpenAlex | 논문 검색 | 무료(키 없음, 하루 약 100회) |
| YouTube Data API v3 | 영상 메타데이터 | 무료 (10K units/day) |
| Threads (Meta) | 전달 | 무료 |

## 프로젝트 구조

```
omnisummary/
├── main.py                  # CLI 진입점(다이제스트 파이프라인)
├── research_cli.py          # 딥 리서치 CLI(stdout에 JSON). OpenClaw 스킬이 쓴다
├── recall_cli.py            # 지난 다이제스트와 트렌드 조회 CLI(stdout에 JSON). OpenClaw 스킬이 쓴다
├── Dockerfile               # Lambda 이미지(amd64)
├── collectors/              # RSS, Reddit, RSSHub(X), YouTube, WebSearch와 S3 park 로더
├── pipeline/                # Aggregator, Ranker, DigestGenerator, TrendTracker, DailyVisualMaker, runner(오케스트레이션)
├── agent/                   # 딥 리서치 에이전트와 도구 8개, VisualGenerator, DigestStateManager
├── shared/                  # 설정, 모델, 프롬프트, 상태 저장소, AgentCore 메모리, 히스토리, 리서치, 미디어
├── output/                  # Threads 렌더러와 핸들러, 다이제스트 퍼블리셔, 리서치 리포트 전달
├── lambda_handlers/         # digest, daily visual, threads refresh
├── infrastructure/          # CDK 스택(foundation과 application)
├── scripts/                 # deploy, put_secrets, put_inference_profiles, ci_synth, sync_to_s3, 샘플러
├── assets/                  # persona.jpg (화자 캐릭터 기준 시트)
├── config/                  # YAML 설정
├── tests/                   # 단위 테스트와 CDK assertion 테스트
└── docs/                    # design.md와 diagrams/
```

## 테스트와 CI

```bash
uv run python -m pytest tests/ -v        # 단위와 CDK assertion 테스트(hermetic. 네트워크도 AWS도 없다)
uv run pre-commit run --all-files        # ruff, black, 파일 검사, uv lock --check
uv run mypy .
uv run python scripts/ci_synth.py        # 오프라인 CDK synth
uv run pre-commit install                # 1회. 커밋 전에 게이트를 돌린다
```

테스트는 hermetic하다. `tests/conftest.py`가 셸에 있던 시크릿과 인프라 환경 변수를 비우고 SSM 클라이언트를 막아서, 결과가 개발자의 `.env`나 AWS 프로필에 좌우되지 않는다.

CI(`.github/workflows/ci.yml`)는 push와 PR, 그리고 주간 스케줄로 다섯 개 잡을 돈다. lint & type-check(`uv lock --check`와 ruff, black `--check`, 레포 전체 `mypy .`), CDK synth(핀된 CLI로 오프라인, 추적되는 `config-template.yaml` 대상), tests & coverage(범위와 `fail_under`는 `pyproject.toml`에 있다), Docker build & import check, dependency & secret scan이다. 모든 잡에 `timeout-minutes`가 붙어 있고, uv와 npm 캐시는 락파일로 키를 잡으며, 서드파티 액션은 커밋 SHA로 고정하고 checkout은 자격증명을 남기지 않는다. 별도 워크플로(`workflow-lint.yml`)가 체크섬을 검증한 actionlint(shellcheck 포함)로 워크플로를 검사하고, Dependabot이 고정된 액션을 매주 갱신한다. uv는 CI와 Dockerfile 모두 0.6.14로 고정이다. 잠금 파일은 `uvx --from uv==0.6.14 uv lock`으로 다시 만들어 그 버전이 읽을 수 있는 형식을 유지한다.

이 중 두 가지는 따로 설명한다.

- **import 체크**는 빌드된 이미지를 `--network none`, 자격증명 없음으로 띄워 실제 엔트리 모듈을 import하고 config를 로드한다. 빌드만으로는 import가 한 번도 실행되지 않으니, 이 체크가 없으면 `COPY` 누락이나 import 시점의 AWS/HTTP 호출이 CI가 아니라 콜드스타트에서 드러난다.
- **의존성 스캔**은 설치된 트리를 감사한다(`uv sync --frozen --no-dev --no-install-project` 후 `pip-audit --strict --path .venv/...`). 그게 이미지가 설치하는 집합이기 때문이다. `--no-install-project`가 중요하다. pip-audit은 editable 배포를 감사 불가로 보고하고 `--strict`가 그것을 실패로 바꾼다. 잠금 파일이 그대로인데 새 권고가 나온 경우는 주간 실행이 잡는다. `gitleaks`는 전체 히스토리를 받아 스캔하고, PR에서 필요한 `GITHUB_TOKEN`을 받는다.

  운영자 참고. `pip-audit --strict`는 잠긴 집합의 알려진 권고가 0건이기를 요구하니, 전이 의존성에 새 권고가 나오면 빌드가 빨개진다. 대응 순서는 `uv lock --upgrade-package <pkg>`, 미사용 의존성 제거, 버전 상한 완화 순이고, `--ignore-vuln`은 코드 경로가 도달 불가임을 확인한 뒤의 마지막 수단이다.

## 문서

[docs/design.md](docs/design.md)가 줄 단위 설계와 기술 레퍼런스이고, "왜 이렇게 생겼는가"에 대한 답이 모두 거기 있다.

## 라이선스

[MIT License](LICENSE)
