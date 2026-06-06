# OmniSummary — 기술 문서

> OmniSummary의 상세한 line-by-line 기술 레퍼런스를 담은 단일 문서입니다.
> 상위 수준 개요는 `README.md`와 `.claude/CLAUDE.md`에 있고, 이 문서는 심화 레퍼런스입니다.

## 1. 개요

OmniSummary는 능동형(proactive) AI/ML 일일 다이제스트 시스템입니다.

- **수집:** 매일 정해진 스케줄에 5개 소스 계열에서 콘텐츠를 수집.
- **처리:** 집계·중복 제거 후 LLM으로 순위를 매김.
- **생성/전달:** 한국어 에디토리얼 다이제스트를 생성해 Slack으로 전달.
- **상태 저장:** 상태를 Bedrock AgentCore Memory에 저장.
- **후속 에이전트:** AgentCore Runtime 위의 Strands가 다이제스트 항목 질문에 답하고 시각화(만화, 다이어그램)를 생성.
- **운영 헬스:** 소스별로 리포팅되며 SNS 이메일로 알림.

```
[EventBridge 크론] → [다이제스트 Lambda (Docker)]
   → 수집기 (RSS, Reddit, YouTube, WebSearch, X via RSSHub/S3)
   → 집계기 (URL + 제목 중복 제거)
   → 랭커 (Bedrock Claude Opus 4.8, 소스 슬롯 + origin 다양성)
   → 트렌드 트래커 (구조화 trends.json, StateStore)
   → 다이제스트 생성기 (Bedrock Claude Sonnet 4.6, 한국어 Slack mrkdwn)
   → Slack 전달
   → AgentCore Memory (다이제스트 스냅샷)
   → 데일리 비주얼 Lambda 비동기 트리거 (gpt-image-2)
   → 실패한 소스가 있으면 SNS 알림

[Slack 멘션] → [API Gateway + WAF] → [Slack Lambda]
   → 비동기 self-invoke → [Bedrock AgentCore Runtime: Strands 에이전트]
   → 도구: get_detail, search_papers, search_community, search_related_news, recall_trends, make_visual
   → AgentCore Memory에서 다이제스트 상태를 읽고, Slack에 답변/이미지를 게시
```

파이프라인 개념도(수집 → 랭킹 → 다이제스트 → 전달, 후속 에이전트 루프):

![How the digest works](concept-pipeline.png)

AWS 아키텍처(두 경로 — 스케줄 다이제스트 / 인터랙티브 후속):

![AWS architecture](architecture.png)

## 2. 저장소 구조

| 경로 | 책임 |
|------|------|
| `collectors/` | `BaseCollector` ABC + RSS, Reddit(.rss 피드), RSSHub(X/Twitter), YouTube, WebSearch(Tavily) |
| `pipeline/` | `ContentAggregator`, `ContentRanker`, `DigestGenerator`, `TrendTracker` |
| `agent/` | Strands 에이전트, 도구, `DigestStateManager`, `VisualGenerator`(자유형 이미지) |
| `agent_runtime/` | Bedrock AgentCore HTTP 서버(`BedrockAgentCoreApp`) |
| `shared/` | config, models, constants, utils(Bedrock 팩토리), logger, prompts, state_store, **memory**, proxy |
| `output/` | Slack 전달(텍스트 + 이미지 업로드) |
| `lambda_handlers/` | 다이제스트 핸들러, Slack 이벤트 핸들러, 일일 시각화 핸들러(`visual_handler`, 다이제스트 Lambda가 비동기 호출) |
| `infrastructure/` | CDK `foundation_stack` + `application_stack` |
| `scripts/` | `deploy.py`, `ci_synth.py`, `sync_rsshub_to_s3.py` |

## 3. 설정(Configuration)

`config/config.yaml` → `shared/config.py`의 Pydantic 모델로 `Config.load()`를 통해 로드됩니다. 시크릿은
`.env`(로컬) 또는 SSM Parameter Store의 `/{project}/{stage}/{name}` 경로(AWS)에서 옵니다.

**우선순위.** `config.yaml`의 값이 Pydantic 필드 기본값을 재정의합니다. 모델 ID는 코드에 하드코딩되어 있지
않습니다 — 예컨대 `PipelineConfig`는 `ranking_model`/`digest_model` 둘 다 Sonnet 4.6을 기본값으로 두지만,
`config.yaml`이 `ranking_model`을 Opus 4.8로 올려 잡고 있어 실제 배포에서 랭킹은 Opus 4.8로 돕니다.
아래 표기는 `config.yaml` 기준 실효값입니다.

### 3.1 `collectors.*`

각 수집기는 `BaseCollectorConfig`를 상속하며, 더해 개별 필드를 둡니다.

| 그룹 | 필드 | 설명 |
|------|------|------|
| 공통(상속) | `enabled`, `lookback_hours`, `reference_time`, `request_timeout`, `max_retries`, `retry_backoff_sec` | 활성화/조회 윈도/타임아웃/재시도 |
| `rss` | `feeds` | RSS 피드 URL 목록 |
| `reddit` | `subreddits`, `sort`, `limit` | 서브레딧·정렬·개수 |
| `youtube` | `channels`, `max_videos_per_channel`, `resolve_timeout`, `transcript_timeout`, `transcript_language` | 채널·영상 수·자막 |
| `web_search` | `trend_searches`, `max_results_per_query`, `max_refine_queries`, `min_search_score`, `refine_model` | Tavily 검색·관련도 필터 |
| `rsshub` | `base_url`, `accounts`, `error_rate_threshold` | X 계정(로컬 컨테이너/S3) |

### 3.2 `pipeline`

| 영역 | 필드 | 설명 |
|------|------|------|
| 모델 | `ranking_model`(실효 Opus 4.8), `digest_model`(Sonnet 4.6), `trend_model` | 단계별 모델 |
| 랭킹 | `ranking_batch_size`, `engagement_tiers`, `ranking_categories`, `ranking_duplicate_score_penalty`, `ranking_scoring_rubric`, `item_text_max_tokens` | 병렬 배치·참여도 보정·카테고리·점수 루브릭 |
| 선정/다양성 | `top_n`, `min_score`, `source_slots`, `source_cap_multiplier`, `max_per_origin`, `origin_weights`, `origin_weight_default`, `origin_weight_nudge` | 상위 N·소스 슬롯·origin 상한·가산 보정 |
| 트렌드 | `trend_retention_days`, `trend_cooling_days`, `trend_max_evidence`, `trend_max_active_trends`, `trend_momentum_half_life_days` | 보존/냉각/증거·active 캡·momentum 반감기 |
| 시각화 | `enable_daily_visual`, `image_model`, `image_size`, `visual_synopsis_source_max_tokens`, `visual_synopsis_context_max_tokens`, `visual_context_max_results`, `visual_context_preview_chars`, `visual_caption_emoji` | 데일리 비주얼 on/off·gpt-image 모델·입력 상한·캡션 이모지 |
| 프롬프트 주입(하드코딩 대신 템플릿 변수) | `digest_language_rules`, `ranking_audience_description`, `digest_audience_description`, `visual_audience_description`, `visual_caption_language`, `visual_on_image_language`, `visual_synopsis_style_guidance`, `visual_synopsis_humor_guidance`, `visual_synopsis_style_aesthetic`, `visual_moderation_softening_instruction` | 언어/대상독자/톤·유머/미감/모더레이션 완화 문구 |

캡션 언어와 이미지 내부 텍스트 언어를 분리(`visual_caption_language` vs `visual_on_image_language`)한 것은
이미지 모델이 비라틴 글리프를 깨뜨리기 때문입니다(캡션=한국어, 이미지 내부=영어).

### 3.3 `agent`

| 필드 | 설명 |
|------|------|
| `model_id`, `enable_interactive` | 에이전트 모델·인터랙티브 on/off |
| `community_search_domains` | community 검색 도메인 허용 목록 |
| `search_result_limit`, `search_content_preview_chars`, `search_request_timeout`, `search_max_retries`, `search_retry_backoff_sec` | 검색 결과 수·미리보기·타임아웃·재시도 |
| `search_paper_max_authors`, `search_paper_abstract_max_chars` | Semantic Scholar 결과 포맷 |
| `detail_max_tokens`, `recall_memory_top_k` | get_detail 본문 토큰 상한·recall 상위 K |
| `boto_read_timeout`, `boto_connect_timeout`, `boto_max_attempts` | AgentCore Bedrock 클라이언트 boto 설정 |

### 3.4 `aws`

| 필드 | 설명 |
|------|------|
| `region`, `bedrock_region`, `profile`, `project_name`, `stage` | 리전·프로파일·프로젝트/스테이지 |
| `timezone` | 다이제스트 날짜 기준 TZ(예: `Asia/Seoul`) |
| `digest_cron_hour`/`digest_cron_minute` | EventBridge 크론(**UTC** 기준) |
| `vpc_id`, `subnet_ids`, `state_bucket_name`, `s3_prefix` | 네트워킹·상태 버킷 |
| `api_throttle_rate_limit`/`api_throttle_burst_limit`, `waf_rate_limit` | API GW 스로틀·WAF 레이트리밋 |

### 3.5 시크릿 & 환경 변수

| 변수 | 출처 | 용도 |
|------|------|------|
| `SLACK_BOT_TOKEN` | `.env` → SSM | Slack 메시지/이미지 전송 |
| `SLACK_SIGNING_SECRET` | `.env` → SSM | Slack 이벤트 서명 검증 |
| `SLACK_CHANNEL_ID` | `.env` → SSM | 다이제스트/비주얼 대상 채널 |
| `TAVILY_API_KEY` | `.env` → SSM | 웹/커뮤니티/뉴스 검색 |
| `OPENAI_API_KEY` | `.env` → SSM | gpt-image 이미지 생성 |
| `YOUTUBE_API_KEY` | `.env` → SSM | YouTube Data API |
| `ALERT_EMAIL` | `.env` → 배포 시 SNS 구독 | 소스 실패 알림 |
| `CLOUDFLARE_PROXY_URL`/`CLOUDFLARE_PROXY_TOKEN` | `.env` | Reddit/YouTube 프록시(데이터센터 IP 우회) |
| `MEMORY_ID`, `STATE_BUCKET`, `S3_PREFIX`, `ALERT_SNS_TOPIC_ARN`, `RSSHUB_BASE_URL`, `PROJECT_NAME`, `STAGE` | CDK 주입(AWS) | 런타임 리소스 식별자 |

`.env`의 시크릿은 `scripts/deploy.py`가 배포 시 SSM `/{project}/{stage}/{name}`에 적재하고, Lambda/AgentCore는
env→SSM 순으로 `resolve_secret`이 해소합니다. `RSSHUB_BASE_URL`은 `rsshub_base_url` CDK context로 재정의
가능하며, 로컬 개발에선 RSSHub Docker 컨테이너가 `localhost:RSSHUB_PORT`(기본 `1200`)에서 동작해야 X 수집이 됩니다.

## 4. 수집기(Collectors)

**공통 계약.** 모든 수집기는 `BaseCollector.collect() -> list[CollectedItem]`을 구현하고
`cutoff_datetime(lookback_hours, reference_time)`(`collectors/base.py`)로 필터링합니다.

**RSS** (`rss.py`)
- **소스:** `config.collectors.rss.feeds`에 대해 feedparser 사용.
- **메타데이터:** `feed_url`, `feed_title`.

**Reddit** (`reddit.py`)
- **방식:** 공개 `.rss` 피드 사용 — `https://www.reddit.com/r/{sub}/{sort}/.rss`.
- **이유:** Reddit이 셀프서비스 OAuth 앱 생성을 동결(Responsible Builder Policy, 2025-11)했고 `.json` API는
  데이터센터 IP를 차단했지만, `.rss` 피드는 열려 있음.
- **경로:** Cloudflare 프록시(`get_proxied_url`) 경유로 가져와 AWS Lambda IP에서도 동작. 자격증명·앱 등록 불필요.
- **트레이드오프:** RSS엔 `score`/`num_comments`(engagement)가 없어 랭킹은 LLM 품질 판단에 의존.

**RSSHub** (`rsshub.py`)
- **소스:** 로컬/컨테이너 RSSHub를 통한 X/Twitter 피드; S3에 사전 동기화된 스냅샷(`rsshub_items.json`)도 로드 가능.
- **헬스:** 실패/빈 계정을 자체 추적하며 `error_rate_threshold` 보유.

**YouTube** (`youtube.py`)
- **소스:** `YOUTUBE_API_KEY`가 있으면 YouTube Data API, 없으면 프록시 경유 RSS 폴백.
- **다양성:** `max_videos_per_channel=1`로 고빈도 채널이 후보 풀을 독점하지 못하게 함.

**WebSearch** (`web_search.py`)
- **소스:** LLM 쿼리 정제(`RefineQueryPrompt`)를 곁들인 Tavily 검색.

**동시 실행 & 헬스.**
- `gather_collector_results()` — 수집기를 동시 실행하고 작업별 예외를 삼킴(로깅만, 평탄한 리스트 반환).
- `main.run_collectors_with_health()` — 헬스 리포팅용으로 동일 작업을 실행하되 `HealthReport`(§8 참조)를 반환.
  `gather_collector_results`는 다른 호출자들을 위해 그대로 유지.

## 5. 파이프라인(Pipeline)

### 1. 집계기 (`aggregator.py`)
- **처리:** URL → 정규화 제목 순으로 중복 제거.
- **출력:** 중복의 메타데이터 병합.

### 2. 랭커 (`ranker.py`)
- **입력:** 항목 포맷팅(engagement + origin 포함).
- **점수 산출:** Claude Opus 4.8로 `RankingPrompt` 호출 → JSON 점수 파싱.
- **origin 가산 보정:** `origin_weights`를 가산 보정으로 적용 — `score + (weight-1.0)*origin_weight_nudge`를
  [0,1]로 클램프(곱셈 배수가 아님). 미등록 origin엔 `origin_weight_default`.
- **필터:** `min_score` 필터 적용.
- **선정/다양성 (`_apply_source_slots`):**
  - `source_slots`로 소스별 기본 슬롯 채우기.
  - `source_cap_multiplier × slot`까지 오버플로 채우기.
  - `max_per_origin`으로 하나의 origin 키(채널/작성자/서브레딧)가 차지하는 항목 수 제한 — 단일 채널
    독점에 대한 근본 해결책.
  - origin은 `_resolve_origin_key`로 해석: YouTube→channel_url, Reddit→subreddit, RSS→feed_url, X→author.

### 3. 트렌드 트래커 (`trend_tracker.py`)
- **상태:** 구조화 `trends.json` 유지 — slug id, 증거 리스트.
- **생명주기:** 날짜 기반 상태(active/cooling/archived), momentum 감쇠 랭킹, active 캡 아카이브 (§7 참조).

### 4. 다이제스트 생성기 (`digest_generator.py`)
- **처리:** Claude Sonnet 4.6로 `DigestPrompt` → 한국어 Slack mrkdwn.
- **정규화:** `sanitize_slack_mrkdwn`이 출력 정규화.

### 5. 데일리 비주얼 (`daily_visual.py`, `enable_daily_visual`)
- **트리거:** 다이제스트 전송 후 실행.
- **스토리 선택:** `VisualEditorPrompt`로 스토리 1건을 고름(주로 뉴스 선호, 적합한 게 없으면 `skip`).
- **맥락 보강:** 선택 시 Tavily로 추가 맥락 검색.
- **생성:** `VisualGenerator`(시놉시스 → gpt-image)로 1컷 밈/패러디/일러스트 또는 N컷 카툰 생성 → Slack 게시.
- **best-effort:** OpenAI 키 없음/부적합/오류 시 조용히 건너뛰며 파이프라인을 막지 않음.

## 6. LLM 팩토리 (`shared/utils.py`)

**모델 팩토리.** `BedrockLanguageModelFactory.get_model(model_id, **kwargs)`
- **반환:** 모델 역량(`_LANGUAGE_MODEL_INFO`)에 맞게 구성된 `ChatBedrock`/`ChatBedrockConverse`.
- **구성 역량:** thinking, 1M 컨텍스트, 성능 레이턴시, 프롬프트 캐싱.
- **리전:** `BedrockCrossRegionModelHelper`가 가능 시 `global.`/`apac.` inference-profile ID를 해석.
- **모델 ID:** `shared/constants.py`(`LanguageModelId`)에 열거; 최신은 Opus 4.8 / Sonnet 4.6.

**시크릿 헬퍼.** `resolve_secret(env_var, ssm_suffix)`
- **해석 순서:** env 우선, 그다음 SSM(`/{project}/{stage}/{suffix}`, SecureString 복호화).
- **사용처:** OpenAI 키(`make_visual`).

**프롬프트 캐싱.** Bedrock 프롬프트 캐싱은 Claude 기준 캐시 가능 프리픽스 최소치가 약 1024 토큰. 효과가 있는 곳에만 적용:
- **에이전트(적용):** 약 1.7K 토큰 시스템 프롬프트 + 도구 스키마가 매 ReAct 스텝마다, 그리고 멀티턴 세션 내내
  재전송되므로 Strands `BedrockModel(cache_config=CacheConfig(strategy="auto"))`(`agent/agent.py`)로 해당
  프리픽스를 캐싱. 검증: 첫 호출에 `cacheWriteInputTokens`, 이후 `cacheReadInputTokens` 발생.
- **파이프라인(미적용):** 단발성 프롬프트(랭커/다이제스트/트렌드/시각화 시놉시스, 모두 약 530 토큰이며 실행당
  1회 호출)는 캐시 최소치 미만이고 호출 간 재사용도 없어 의도적으로 캐싱을 적용하지 않음.

## 7. 메모리: 두 개의 분리된 저장소

트렌드 기억과 다이제스트 스냅샷은 **성격이 달라 서로 다른 저장소**에 둡니다.

**(a) 트렌드 — 구조화 `trends.json` (`StateStore`, 시스템 오브 레코드)**

- **관리 주체:** `pipeline/trend_tracker.py`의 `TrendTracker`.
- **LLM 역할:** `TrendClassifyPrompt`는 오늘 아이템이 기존 트렌드(id) 확장인지 신규인지 분류만 함. 부기는 전부 결정론적 Python.
- **결정론적 부기:**
  - 증거 날짜는 코드가 스탬프(LLM 아님).
  - 상태(active/cooling/archived)는 `last_seen` vs `trend_cooling_days`/`trend_retention_days`로 계산.
  - momentum은 recency 감쇠(`0.5^(age/half_life)`, `trend_momentum_half_life_days` 기본 7일).
  - 트렌드당 증거 `trend_max_evidence` 캡.
  - active 트렌드 수 `trend_max_active_trends` 캡(최저 momentum 아카이브).
  - 동일 날짜 재실행은 멱등(그날 증거 교체).
- **진실의 원천:** `trends.json`(`TrendMemory`)이 원천이고 마크다운은 렌더된 뷰.
- **마이그레이션:** 첫 실행 시 레거시 `trends.md`가 있으면 `from_markdown`으로 1회 마이그레이션.
- **주입:** 다이제스트 생성 시 active/cooling 트렌드를 momentum 순 마크다운으로 렌더해 `DigestPrompt`에 주입.

**(b) 다이제스트 스냅샷 — AgentCore Memory (`shared/memory.py`)**
- **`AgentCoreMemoryStore`:**
  - **기록:** 오늘의 ranked 아이템 스냅샷을 단기 세션 이벤트로 기록(`create_event`, 세션 `digest-<date>`,
    `_fit_to_limit`로 100k 한도 보장).
  - **읽기:** `get_latest_digest()`가 최신 세션을 읽음.
  - **목적:** 후속 에이전트(`get_detail`)와 데일리 비주얼 Lambda가 cross-Lambda로 이 스냅샷을 공유하는 수단.
  - **제거됨:** 시맨틱 recall/장기 전략 제거(관리형 추출이 트렌드 흐름이 아닌 안정적 사용자-사실만 뽑아 부적합).
- **`LocalMemoryStore`:** 오프라인 폴백(`digest_*.json`만).

**`recall_trends` 도구.** AgentCore가 아니라 `trends.json`을 직접 쿼리(키워드 매칭 + momentum 정렬,
`TrendMemory.search`). 메모리 리소스(`AWS::BedrockAgentCore::Memory`)는 이제 이벤트 전용(단기,
`event_expiry_duration` 90일)이며 시맨틱 전략/`RetrieveMemoryRecords` 권한은 없음.

## 8. 헬스 체크 & 알림

**모델 (`shared/models.py`):**
- `SourceStatus` — `ok`/`empty`/`failed`.
- `SourceHealth(name, item_count, status, detail)`.
- `HealthReport(sources)` — `has_failures`, `summary()` 보유.

**소스 분류 (`run_collectors_with_health`):**
- 예외 → FAILED(잘린 detail 포함).
- 0 항목 → EMPTY(조용한 날엔 정상).
- 그 외 → OK.

**알림 (`_maybe_alert`, 다이제스트 Lambda):** 소스가 FAILED일 때만, 그리고 빈 항목 조기 반환 이전에
`ALERT_SNS_TOPIC_ARN`으로 게시(아무것도 수집 못 해도 장애는 알림되도록).

## 9. 에이전트(AgentCore Runtime 위의 Strands)

**구성 (`agent/agent.py`).** `BedrockModel`(Sonnet 4.6)과 도구로 Strands `Agent`를 구성.

**SYSTEM_PROMPT.** 자율 에이전트 철학을 따름.
- 고정 라우팅 없이 작은 단일 목적 도구들을 자유롭게 조합하도록 안내.
- Slack mrkdwn 포맷 규칙과 응답 템플릿을 포함.

**도구 (`agent/agent_tools.py`) — 모두 독립적이며 에이전트가 자유롭게 조합:**
- `get_detail(item_number)` — `state_manager`에서 항목 본문 + 랭킹 메타데이터 로드.
- `search_papers(query)` — Semantic Scholar(429 시 retry/backoff).
- `search_community(query)` / `search_related_news(query)` — 공유 `_tavily_search(query, topic,
  include_domains)` 헬퍼를 감싼 얇은 래퍼.
- `recall_trends(query)` — 구조화 `trends.json`에 대한 키워드 매칭 + momentum 정렬(active/cooling 트렌드).
  시맨틱 recall이나 AgentCore 장기 메모리가 아님.
- `make_visual(instruction, item_number, context)` — 자유형 이미지 생성, §10 참조.

**전형적 조합 예** ("1번을 1페이지 슬라이드로"): `get_detail`(+필요시 `search_papers`/`search_related_news`로
보강) → `make_visual(instruction="...설명하는 1페이지 프리젠테이션 슬라이드...", item_number=1,
context=<수집한 리서치>)`. 고정 워크플로가 아니라 에이전트가 매번 계획을 세움.

**런타임 (`agent_runtime/app.py`, `BedrockAgentCoreApp`).** invoke 시 순서대로:
- correlation id 설정.
- Memory에서 최신 다이제스트 상태 로드.
- `delivery_context`(미디어 도구용 채널/스레드) 설정.
- 에이전트 실행.
- Slack에 답변 게시.

**진입 Lambda (`slack_event_handler.py`).**
- Slack 서명을 검증(HMAC, 타이밍 안전).
- DynamoDB 조건부 쓰기로 중복 제거.
- 비동기 self-invoke로 AgentCore 런타임을 호출.

## 10. 시각화 파이프라인(자유형 시놉시스 → 이미지)

**설계 (`agent/visuals.py`의 `VisualGenerator`).** 모드 없는 자유형 생성기.
- 고정된 comic/diagram 모드나 컷 수 파라미터가 없음.
- 에이전트가 자연어 `instruction`으로 원하는 형식(1페이지 프리젠테이션 슬라이드, N컷 만화, 개념 다이어그램,
  인포그래픽, 포스터 등)을 묘사.
- source(다이제스트 항목)와 직접 수집한 `context`(논문/기사 리서치)를 넘김.

**생성 흐름 (`VisualGenerator.generate(instruction, source, context)`):**
- **브리프:** `VisualSynopsisPrompt`로 Claude(Bedrock)가 단일 이미지 브리프 생성(JSON: title·caption·prompt).
- **파싱:** `_parse_brief`(`extract_json_from_llm_output` + `VisualBrief.model_validate`).
- **이미지:** 브리프의 `prompt`로 OpenAI `gpt-image-2`(1024x1536 portrait by default, `b64_json`) → PNG 바이트.
- **업로드:** `make_visual` 도구가 `output.slack_handler.send_image_to_slack`(`files_upload_v2`)로 Slack에 이미지 업로드.

**기타.**
- OpenAI 키(`resolve_secret`로 env→SSM 해석)가 없으면 우아하게 비활성화.
- 새 출력 형식은 코드 변경 없이 instruction 문구만 바꾸면 됨(에이전틱).

## 11. 인프라(CDK)

### `foundation_stack`

- **리소스:** VPC, ECR 리포, DynamoDB 중복 제거 테이블(SSE + prod에서 PITR), S3 상태 버킷(CDK 생성 시
  S3-managed 암호화, 버저닝, 퍼블릭 차단, SSL 강제), ECS Fargate RSSHub 서비스 + service-discovery,
  CodeBuild 이미지 빌드, SNS 알림 토픽(+ 선택적 이메일 구독), AgentCore Memory 리소스 + 실행 역할, IAM 역할들.
- **IAM(최소 권한):**
  - `/{project}/{stage}/*`로 스코프된 `ssm:GetParameter*`.
  - foundation-model/inference-profile ARN으로 스코프된 `bedrock:InvokeModel*`.
  - 스코프된 `lambda:InvokeFunction` 및 `bedrock-agentcore:InvokeAgentRuntime`/Memory 데이터플레인 액션.
  - 프로젝트 로그 그룹 ARN으로 스코프된 CloudWatch Logs.
  - 계정 전역 관리형 정책 없음.

### `application_stack`

- **리소스:** 다이제스트 Lambda(DockerImage), Slack 이벤트 Lambda, API Gateway(+ 스테이지 스로틀링),
  스테이지에 연결된 WAFv2 WebACL(rate-limit + AWS 관리형 규칙셋: Common, KnownBadInputs, IpReputation),
  EventBridge 일일 크론(설정 기반 시/분), AgentCore Runtime(설정 가능한 `agentcore_image_ref`로 이미지 바인딩),
  시크릿용 SSM 파라미터, SNS로 향하는 CloudWatch 알람(Lambda 에러 ×2, API 5xx).
- **시크릿 처리:** 평문 `String` SSM 파라미터(CloudFormation은 SecureString 생성 불가). 보완 통제는 스코프된
  IAM 읽기 정책이며, 더 민감한 자격증명은 Secrets Manager로 승격 권장.

## 12. 관측성(Observability)

**로깅 (`shared/logger.py`).**
- **포맷:** AWS에서는 구조화 JSON 로그(`is_running_in_aws()`), 로컬에서는 사람이 읽는 형식.
- **correlation id:** `ContextVar` 기반(`set_correlation_id`/`get_correlation_id`)이 모든 레코드에 주입되고
  Lambda 요청 id / AgentCore 페이로드에서 시드됨.

**알람.** CloudWatch 알람은 SNS 알림 토픽으로 라우팅됨.

## 13. 테스트 & CI/CD

**테스트 (`tests/`, pytest, `asyncio_mode=auto`).** 300+ 테스트, 커버리지 게이트 55%. 커버 영역:
- 수집기(모킹한 HTTP/feedparser).
- Slack 이벤트 핸들러(서명 검증/중복 제거).
- 집계기, 랭커 파싱 + 슬롯/origin-cap 로직.
- 헬스 리포트, logger.
- 메모리 스토어(로컬 + AgentCore 모킹).
- 다이제스트 핸들러 알림.
- 에이전트 도구, visuals.
- AgentCore 엔트리포인트(`agent_runtime/app.py` — 상태 로드·Slack 토큰 env/SSM 해석·invoke 해피패스/예외 처리·correlation ID).
- trend_tracker(trim/evidence-cap/archived-merge).
- CDK assertion(`aws-cdk.assertions`로 두 스택 검증).

**CI (`.github/workflows/ci.yml`).**
- lint(ruff), 포맷 체크(black `--check`), mypy 타입 체크.
- 테스트 + 커버리지 게이트.
- 오프라인 `cdk synth`(`scripts/ci_synth.py`, 더미 계정 — AWS 자격증명 불필요).
- Docker 빌드(amd64, `--provenance=false`).

## 14. 주요 명령어

```bash
uv run python main.py --dry-run --sources rss reddit   # 부분 dry run
uv run python main.py                                   # 전체 파이프라인 + Slack
uv run python -m pytest tests/ -v                       # 테스트
uv run black --check . && uv run ruff check .           # lint/format
uv run mypy shared/ collectors/ pipeline/ agent/ output/ lambda_handlers/ main.py
uv run python scripts/ci_synth.py                       # 오프라인 CDK synth
# 프로파일은 config.aws.profile에서 오며, 환경 변수로 재정의할 수 있다 (기본값 research)
AWS_PROFILE=${AWS_PROFILE:-research} uv run cdk deploy --all -a "uv run python scripts/deploy.py"
```
