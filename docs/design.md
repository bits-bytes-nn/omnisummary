# OmniSummary — 설계 문서

> OmniSummary의 상세한 line-by-line 설계·기술 레퍼런스를 담은 단일 문서입니다.
> 상위 수준 개요는 `README.md`와 `.claude/CLAUDE.md`에 있고, 이 문서는 심화 레퍼런스입니다.

## 1. 개요

OmniSummary는 능동형(proactive) AI/ML 일일 다이제스트 시스템입니다.

- **수집:** 매일 정해진 스케줄에 5개 소스 계열에서 콘텐츠를 수집.
- **처리:** 집계·중복 제거 후 LLM으로 순위를 매김.
- **생성/전달:** 한국어 에디토리얼 다이제스트를 구조화된 `DigestContent`로 생성하고, 채널별 렌더러로 Slack(Block Kit)과 Threads에 전달. Slack은 다이제스트 Lambda가, **Threads는 데일리 비주얼 Lambda가** 게시한다(이미지와 텍스트가 한 게시물 세트로 나가야 하므로).
- **상태 저장:** 상태를 Bedrock AgentCore Memory에 저장.
- **딥 리서치 에이전트:** Slack 멘션으로 트리거되는 AgentCore Runtime 위의 Strands가 자유형 토픽을 받아 웹/논문/커뮤니티를 독립 리서치하고, 한국어로 합성한 출처 표기 리포트를 채널(Slack/Threads)에 전달. 다이제스트와 분리된 독립 웹 리서치임.
- **운영 헬스:** 소스별로 리포팅되며 SNS 이메일로 알림.

```
[EventBridge 크론] → [다이제스트 Lambda (Docker)]
   → 수집기 (RSS, Reddit, YouTube, WebSearch, X via RSSHub/S3)
   → 집계기 (URL + 제목 중복 제거)
   → 랭커 (Bedrock Claude Opus 4.8, 소스 슬롯 + origin 다양성)
   → 트렌드 트래커 (구조화 trends.json, StateStore)
   → 다이제스트 생성기 (Bedrock Claude Sonnet 5, 한국어 구조화 DigestContent)
   → 채널별 렌더링 → Slack(Block Kit) 전달 (enable_slack_post)
   → AgentCore Memory (다이제스트 스냅샷)
   → 데일리 비주얼 Lambda 비동기 트리거 (gpt-image-2)
        └→ 이 Lambda가 이미지와 함께 Threads(root + reply chain)를 게시 (enable_threads_post)
   → FAILED 또는 STALE 소스가 있으면 SNS 알림

[Slack 멘션] → [API Gateway + WAF] → [Slack 이벤트 Lambda]
   → 서명 검증 → DynamoDB 중복 제거 → 즉시 ack 게시 → 비동기 self-invoke
   → [Bedrock AgentCore Runtime: Strands 딥 리서치 에이전트]
   → 도구: web_search, community_search, search_papers, read_url, recall_trends, recall_digest,
     attach_image, deliver_report
   → 다중 소스 리서치 후 채널별(Slack Block Kit / Threads) 리포트를 deliver_report로 게시
```

파이프라인 개념도(수집 → 랭킹 → 다이제스트 → 전달):

![How the digest works](diagrams/concept-pipeline.png)

AWS 아키텍처(두 경로 — 스케줄 다이제스트 / Slack 트리거 딥 리서치):

![AWS architecture](diagrams/architecture.png)

## 2. 저장소 구조

| 경로 | 책임 |
|------|------|
| `collectors/` | `BaseCollector` ABC + 공유 `load_items_from_s3`(로컬→S3 park 파일 로더) + RSS, Reddit(.rss 피드), RSSHub(X/Twitter), YouTube, WebSearch(Tavily) |
| `pipeline/` | `ContentAggregator`, `ContentRanker`, `DigestGenerator`, `TrendTracker`, `DailyVisualMaker` |
| `agent/` | 딥 리서치 Strands 에이전트(`research_agent.py`), 그 7개 도구(`research_tools.py`), `DigestStateManager`(`tool_state.py`, 다이제스트/비주얼 파이프라인용 인메모리 상태), `VisualGenerator`(`visuals.py`, 데일리 비주얼이 사용하는 자유형 이미지) |
| `agent_runtime/` | Bedrock AgentCore HTTP 서버(`BedrockAgentCoreApp`) — 딥 리서치 에이전트 invoke 엔트리포인트 |
| `shared/` | config(공유 `KOREAN_STYLE_RULES` 포함), models, constants(`TRENDS_KEY` 포함), utils(Bedrock 팩토리), logger, prompts, state_store, **memory**, **history_store**(cross-day dedup 원장 + 롤링 로그), **research**(`research_backends.py`: Tavily/Semantic Scholar 백엔드), **media**(`og_image.py`: OG 이미지 fetch), proxy |
| `output/` | 채널별 렌더러(`renderers.py`: 다이제스트용 Slack Block Kit / Threads + 리서치용 `render_research_blocks`/`render_threads_research` + 폴백 `render_agent_blocks`) + 리서치 전달 오케스트레이션(`delivery.py`) + Slack 전달(`slack_handler.py`) + Threads 전달(`threads_handler.py`) |
| `lambda_handlers/` | 다이제스트 핸들러, Slack 이벤트 핸들러, 일일 시각화 핸들러(`visual_handler`, 다이제스트 Lambda가 비동기 호출), Threads 토큰 갱신 핸들러(`threads_refresh_handler`) |
| `infrastructure/` | CDK `foundation_stack` + `application_stack` |
| `scripts/` | `deploy.py`, `ci_synth.py`, `sync_rsshub_to_s3.py`, `sync_youtube_to_s3.py`, `sync_all_to_s3.sh`(두 sync를 함께 실행) |

## 3. 설정(Configuration)

`config/config.yaml` → `shared/config.py`의 Pydantic 모델로 `Config.load()`를 통해 로드됩니다. 시크릿은
`.env`(로컬) 또는 SSM Parameter Store의 `/{project}/{stage}/{name}` 경로(AWS)에서 옵니다.
`config/config.yaml`은 gitignore 대상이고 **`config/config-template.yaml`만 추적**되므로, CI synth와
인프라 테스트는 이 템플릿을 로드합니다(`Config.load()`는 CI에서 조용히 코드 기본값으로 떨어져 아무것도
증명하지 못했음).

**캐싱(`get_config()`).** `Config.load()`는 호출마다 YAML을 다시 읽고 전체를 재검증하므로, 리서치 1회
실행이 도구 호출마다 이를 반복했습니다. **읽기 전용 리프 호출자**(`output/delivery.py`,
`shared/research/research_backends.py`, `agent/research_tools.py`, `shared/media/og_image.py`)는
`lru_cache(maxsize=1)`로 감싼 `get_config()`를 씁니다. Config를 **변경하는** 호출자
(`main`/핸들러의 `set_reference_time`, `ci_synth`의 `vpc_id`, 인프라 테스트의 버킷 오버라이드)는
자기 인스턴스가 필요하므로 계속 `Config.load()`/`from_yaml`을 씁니다. 테스트는 conftest의 autouse
픽스처가 매 테스트 전후로 캐시를 비웁니다.

**우선순위.** `config.yaml`의 값이 Pydantic 필드 기본값을 재정의합니다. 모델 ID는 코드에 하드코딩되어 있지
않습니다 — 예컨대 `PipelineConfig`는 `ranking_model`/`digest_model` 둘 다 Sonnet 5를 기본값으로 두지만,
`config.yaml`이 `ranking_model`을 Opus 4.8로 올려 잡고 있어 실제 배포에서 랭킹은 Opus 4.8로 돕니다.
아래 표기는 `config.yaml` 기준 실효값입니다.

### 3.1 `collectors.*`

각 수집기는 `BaseCollectorConfig`를 상속하며, 더해 개별 필드를 둡니다.

| 그룹 | 필드 | 설명 |
|------|------|------|
| 공통(상속) | `enabled`, `lookback_hours`, `reference_time`, `request_timeout`, `max_retries`, `retry_backoff_sec`, `park_max_age_hours`(기본 36), `error_rate_threshold`(기본 50.0) | 활성화/조회 윈도/타임아웃/재시도/S3 park 파일 나이 예산(초과 시 항목은 쓰되 헬스 STALE)/입력(피드·계정·채널·쿼리) 실패율 임계 — 넘으면 소스를 DEGRADED로 **보고만** 함(항목은 그대로 전달) |
| `rss` | `feeds`, `max_concurrency`(기본 5) | RSS 피드 URL 목록·동시 fetch 상한 |
| `reddit` | `subreddits`, `sort`, `limit` | 서브레딧·정렬·개수 |
| `youtube` | `channels`, `max_videos_per_channel`, `resolve_timeout`, `transcript_timeout`, `transcript_language` | 채널·영상 수·자막 |
| `web_search` | `trend_searches`, `max_results_per_query`, `max_refine_queries`, `min_search_score`, `refine_model` | Tavily 검색·관련도 필터 |
| `rsshub` | `base_url`, `accounts`, `max_concurrency` | X 계정(로컬 컨테이너/S3)·동시 fetch 상한 |

`error_rate_threshold`는 RSSHub 전용이 아니라 **`BaseCollectorConfig` 공통 노브**입니다(두 번째 숫자를 만들지 않음). RSS·YouTube·web_search도 같은 임계로 DEGRADED를 보고합니다.

`collectors.alert_on_empty`(기본 `[]`)는 **EMPTY가 사건인 소스 이름 목록**입니다(예: `["rss", "web_search"]`). 어두워진 소스는 예외도, stale park 파일도, 실패율도 남기지 않아 다른 어떤 신호에도 걸리지 않습니다 — 반면 reddit·x는 조용한 날이 정상이므로 "빈 소스면 무조건 알림"은 매일 페이징하다 곧 무시됩니다. 그래서 명시적 opt-in 목록이고, 비면 EMPTY로는 절대 알리지 않습니다.

### 3.2 `pipeline`

| 영역 | 필드 | 설명 |
|------|------|------|
| 모델 | `ranking_model`(실효 Opus 4.8), `digest_model`(Sonnet 5), `trend_model` | 단계별 모델 |
| 랭킹 | `ranking_batch_size`, `ranking_max_concurrency`(기본 4), `ranking_max_retries`(기본 3), `ranking_retry_backoff_sec`(기본 5), `engagement_tiers`, `ranking_categories`, `ranking_duplicate_score_penalty`, `ranking_scoring_rubric`, `item_text_max_tokens` | 병렬 배치·Bedrock fan-out 상한·배치 재시도·참여도 보정·카테고리·점수 루브릭 |
| 선정/다양성 | `top_n`, `min_score`, `source_slot_score_grace`(기본 0.1), `source_slots`, `source_cap_multiplier`, `max_per_origin`, `origin_weights`, `origin_weight_default`, `origin_weight_nudge` | 상위 N·소스 슬롯·grace 밴드(슬롯 보유 소스가 min_score 위 항목이 전무하면 grace 밴드 내 최선 1건 구제)·origin 상한·가산 보정 |
| 다이제스트 버퍼/중복 | `digest_candidate_buffer`(기본 3), `published_url_ttl_days`(기본 6), `recent_leads_window`(기본 5) | 랭커 오버선정 버퍼(소스 슬롯은 **top_n 코어**에만 적용하고, 버퍼분은 `backfill` 플래그로 넘겨 병합 보충용임을 항목별로 알림)·cross-day dedup 원장 TTL·반복 방지용 최근 lead 윈도 |
| 트렌드 | `trend_retention_days`, `trend_cooling_days`, `trend_max_evidence`, `trend_max_active_trends`, `trend_momentum_half_life_days` | 보존/냉각/증거·active 캡·momentum 반감기 |
| 전달 | `enable_slack_post`, `enable_threads_post` | 채널별 전달 on/off(각각 독립 토글; 코드 기본값은 Slack on / Threads off, 실제 상태는 배포 환경 설정에 따름). Slack은 다이제스트 Lambda가, **Threads는 데일리 비주얼 Lambda가** 게시 |
| AGI 카운트다운 | `agi_countdown_date`(기본 `2029-01-01`), `agi_countdown_template`, `agi_countdown_after`, `agi_countdown_position`(`prefix`\|`suffix`, 기본·배포 설정 모두 `suffix`) | 다이제스트 lead에 코드가 붙이는 "AGI N일 전" 인트로(D-day 전엔 카운트다운) + lead의 어느 쪽 끝에 붙일지(§5.2 참조) |
| 시각화 | `enable_daily_visual`, `image_model`, `image_sizes`, `visual_format_window`(기본 6), `visual_synopsis_source_max_tokens`, `visual_synopsis_context_max_tokens`, `visual_caption_emoji`, `visual_image_timeout_sec`(기본 300)·`visual_image_max_retries`(기본 0), `visual_multi_panel_target_ratio`(기본 0.34), `visual_character_enabled`·`visual_character_sheet`·`visual_character_target_ratio` | 데일리 비주얼 on/off·gpt-image 모델·orientation→size 딕셔너리·포맷 변주 추적 윈도(orientation+style)·입력 상한·캡션 이모지·gpt-image HTTP 호출 상한(SDK 기본 600s×2회는 비주얼 Lambda 15분 예산을 넘길 수 있어 config로 고정) |
| 프롬프트 주입(하드코딩 대신 템플릿 변수) | `digest_language_rules`, `digest_voice_guidance`(Gruber 톤; 단일 냉소 프레임으로 기본 고정하지 말고 그날 사실이 정당화할 때만 각을 선택), `ranking_audience_description`, `digest_audience_description`, `visual_audience_description`, `visual_caption_language`, `visual_on_image_language`, `visual_synopsis_style_guidance`, `visual_synopsis_humor_guidance`, `visual_synopsis_style_aesthetic`, `visual_moderation_softening_instruction` | 언어/대상독자/톤·유머/미감/모더레이션 완화 문구 |

캡션 언어와 이미지 내부 텍스트 언어를 분리(`visual_caption_language` vs `visual_on_image_language`)한 것은
이미지 모델이 비라틴 글리프를 깨뜨리기 때문입니다(캡션=한국어, 이미지 내부=영어).

### 3.3 `agent`

딥 리서치 에이전트 설정. `research_*`/`og_image_*` "소프트 놉"은 강제 루프 한계가 아니라 에이전트가 가이던스로
따르도록 **시스템 프롬프트에 보간**되는 값이고(이 값을 바꾸면 실제 동작이 바뀜), `research_max_threads_posts`·
`research_content_cap_chars`·`research_max_staged_images`는 **코드가 강제하는 하드 캡**입니다.

| 필드 | 설명 |
|------|------|
| `model_id` | 에이전트 모델(기본 Sonnet 5) |
| `research_breadth`, `research_max_iterations` | 프롬프트에 주입되는 검색 폭(쿼리 수)·깊이(라운드 수) 가이던스 |
| `research_slack_target_words` | Slack 리포트 목표 분량(단어) 가이던스 |
| `research_max_threads_posts` | Threads 게시물(root+reply) 총수 **하드 캡**(기본 6) — 너무 긴 리포트가 공개 게시물 수십 개로 퍼지지 않게 코드가 트림 |
| `research_content_cap_chars` | `read_url` 한 페이지 추출 텍스트 상한(기본 50000) |
| `research_max_staged_images` | 한 리서치 실행이 stage할 수 있는 OG 이미지 수 캡(기본 4, per-invocation 메모리 bound) |
| `og_image_timeout_sec`, `og_image_max_bytes` | OG 이미지 fetch 타임아웃·최대 바이트(스트리밍 중 초과 시 중단) |
| `community_search_domains` | `community_search` 도메인 허용 목록(reddit/x/HN/substack 등) |
| `search_result_limit`, `search_content_preview_chars`, `search_request_timeout`, `search_max_retries`, `search_retry_backoff_sec` | 검색 결과 수·미리보기·타임아웃·재시도 |
| `search_paper_max_authors`, `search_paper_abstract_max_chars` | Semantic Scholar 결과 포맷 |
| `recall_memory_top_k` | `recall_trends`가 반환할 상위 K 트렌드 |
| `boto_read_timeout`, `boto_connect_timeout`, `boto_max_attempts` | AgentCore Bedrock 클라이언트 boto 설정 |

### 3.4 `aws`

| 필드 | 설명 |
|------|------|
| `region`, `bedrock_region`, `profile`, `project_name`, `stage` | 리전·프로파일·프로젝트/스테이지 |
| `timezone` | 다이제스트 날짜 기준 TZ(예: `Asia/Seoul`) |
| `digest_cron_hour`/`digest_cron_minute` | EventBridge 크론(**UTC** 기준) |
| `threads_token_refresh_days` | Threads 장기 토큰(60일 만료) 갱신 주기(기본 50일, ≤59) |
| `vpc_id`, `subnet_ids`, `state_bucket_name`, `s3_prefix` | 네트워킹·상태 버킷 |
| `api_throttle_rate_limit`/`api_throttle_burst_limit`, `waf_rate_limit` | API GW 스로틀·WAF 레이트리밋 |

### 3.5 시크릿 & 환경 변수

| 변수 | 출처 | 용도 |
|------|------|------|
| `SLACK_BOT_TOKEN` | `.env` → SSM | Slack 메시지/이미지 전송 |
| `SLACK_SIGNING_SECRET` | `.env` → SSM | Slack 이벤트 서명 검증 |
| `SLACK_CHANNEL_ID` | `.env` → SSM | 다이제스트/비주얼 대상 채널 |
| `TAVILY_API_KEY` | `.env` → SSM | 웹/커뮤니티/뉴스 검색. **web_search 수집기만** `resolve_secret(strict=True)`로 해석 — SSM **읽기 실패**(거부·스로틀)는 `""`(=키 미설정)와 구분되어야 하고, 예전엔 둘 다 `[]`가 되어 그날 웹 소스 전체가 경고 한 줄과 함께 사라졌다. 이제 raise → 헬스 FAILED → 알림. 파라미터가 정말 없으면 `""`로 조용히 스킵. `shared/research` 백엔드와 에이전트 경로는 기존 ''-degrade 계약 유지 |
| `OPENAI_API_KEY` | `.env` → SSM | gpt-image 이미지 생성 |
| `THREADS_ACCESS_TOKEN` | `.env` → SSM | Threads 게시(장기 토큰, 50일 주기 자동 갱신 후 SSM에 재기록) |
| `THREADS_USER_ID` | `.env` → SSM | Threads 게시 대상 사용자 ID |
| `YOUTUBE_API_KEY` | `.env` → SSM | YouTube Data API |
| `ALERT_EMAIL` | `.env` → 배포 시 SNS 구독 | 소스 실패 알림 |
| `CLOUDFLARE_PROXY_URL`/`CLOUDFLARE_PROXY_TOKEN` | `.env` | Reddit/YouTube 프록시(데이터센터 IP 우회). 워커 쪽 값은 `wrangler secret put PROXY_TOKEN`으로 넣습니다 — `wrangler.toml`의 `[vars]`는 평문으로 버전 관리에 들어가므로 토큰을 두지 않습니다 |
| `MEMORY_ID`, `STATE_BUCKET`, `S3_PREFIX`, `ALERT_SNS_TOPIC_ARN`, `RSSHUB_BASE_URL`, `PROJECT_NAME`, `STAGE` | CDK 주입(AWS) | 런타임 리소스 식별자 |

`.env`의 시크릿은 **CDK 스택을 통과하지 않습니다.** CloudFormation 템플릿은 SecureString을 담을 수 없어서, 값을
스택에 넘기면 `cdk.out/*.template.json`·CDK 스테이징 버킷·`cloudformation:GetTemplate` 응답에 **평문**으로 남습니다
(실제로 Slack 봇 토큰·Tavily/OpenAI/YouTube 키·Threads 토큰·X 세션 쿠키가 그렇게 있었습니다). 스택은 파라미터
**경로만** `SSM_PLACEHOLDER`로 만들고, 실제 값은 `scripts/put_secrets.py`가 **SecureString**으로 기록합니다.
재배포가 값을 되돌리지 않습니다 — CFN은 템플릿 속성이 바뀐 리소스만 갱신하고 플레이스홀더는 불변이기 때문입니다.
스크립트는 **이미 SecureString인 파라미터를 건너뜁니다**(Threads 토큰은 갱신 Lambda가 제자리에서 회전시키므로
로컬 `.env` 사본을 다시 쓰면 만료 토큰으로 되돌아갑니다 — `--force`로만 덮어씀), 비어 있는 env var도 지우지 않습니다.
`resolve_secret`은 플레이스홀더를 **미설정으로 취급**해, put_secrets를 건너뛴 배포가 플레이스홀더를 API 토큰으로
보내는 대신 정상적인 '자격증명 없음' 경로로 degrade합니다. Lambda/AgentCore는 env→SSM 순으로 해소합니다.
X 세션 쿠키는 Fargate 태스크 정의의 `secrets`(ARN만 템플릿에 들어가고 값은 ECS 에이전트가 시작 시 가져옴)로 주입됩니다. `RSSHUB_BASE_URL`은 `rsshub_base_url` CDK context로 재정의
가능하며, 로컬 개발에선 RSSHub Docker 컨테이너가 `localhost:RSSHUB_PORT`(기본 `1200`)에서 동작해야 X 수집이 됩니다.

## 4. 수집기(Collectors)

**공통 계약.** 모든 수집기는 `BaseCollector.collect() -> list[CollectedItem]`을 구현하고
`cutoff_datetime(lookback_hours, reference_time)`(`collectors/base.py`)로 필터링합니다.

**S3 park 파일 로더 (`collectors/base.py` `load_items_from_s3(filename, max_age_hours)`).** 데이터센터(Lambda) IP에서
차단되는 소스(X·YouTube)는 로컬 sync 스크립트가 거주용 IP로 항목을 수집해 S3에 미리 적재하고, AWS에선
수집기가 라이브 fetch 대신 이 파일을 읽습니다. S3 키는 trends.json과 동일 규칙(`S3_PREFIX`의 부모 디렉터리 + 파일명).
원래 RSSHub의 `_load_from_s3`였던 것을 이 공유 헬퍼로 일반화했습니다.
- **반환 타입 `ParkedItems(outcome, items, age_hours, detail, meta)`.** park 파일의 **나이가 데이터와 함께** 흐르도록 명시적 모델로 반환합니다(모듈 전역/threadlocal staleness 플래그 없음). `outcome`은 `absent`(버킷 미설정·객체 없음) / `fresh` / `stale` / `error`(읽기 불가)이고, 파생 프로퍼티 `usable`(fresh·stale → park 항목 사용)과 `degraded`(stale·error → 헬스 STALE)로 호출자가 분기합니다. 수집기는 결과를 `self.park_status`에 남겨 `run_collectors_with_health`가 읽습니다. `meta`는 park 파일을 쓴 sync가 남긴 **수집 방식 기록**(선택)입니다 — RSSHub sync는 `accounts_total`/`accounts_failed`/`accounts_empty`를 적고, 수집기는 되읽어 실패율이 `error_rate_threshold`(라이브 경고와 **같은** 노브)를 넘으면 `degraded_detail`을 세워 헬스가 DEGRADED가 되게 합니다. 신선한 park 파일만으로는 40개 계정 중 3개만 모은 sync를 구분할 수 없었습니다.
- **신선도 봉투(staleness guard).** sync 스크립트는 `{generated_at, items, meta?}` 봉투(`dump_items_envelope`)로 적재하고(`meta`는 비어 있으면 아예 쓰지 않아 구버전 리더와 바이트 호환), 로더는 봉투(meta 유무 무관)/레거시 bare-list를 모두 읽되 `generated_at`이 `park_max_age_hours`(수집기별 config, 기본 36h — 모듈 기본은 `S3_ITEMS_MAX_AGE_HOURS`)보다 오래되면 항목은 그대로 반환하되(stale이 빈 것보다 낫다) `stale`로 표시합니다. 로컬 cron이 조용히 멈춰 며칠 지난 항목을 "오늘 것"으로 재수집하는 사고가 정상 실행(OK)처럼 보이지 않고 **헬스 STALE + SNS 알림**으로 표면화됩니다.
- **빈 park 파일 처리.** 항목이 0건이고 **동시에** 나이 예산을 넘긴 봉투는 '부재'(`absent`)로 취급합니다(→ 라이브 수집으로 폴백, 그쪽에서 전면 장애면 FAILED로 알림). 로컬 sync가 멈춰 빈 파일만 남은 상태를 '오늘은 조용했다'로 오해하지 않기 위함. 반대로 **신선한** 0건 봉투는 정말 조용한 sync 날이므로 그대로 반환해 거짓 FAILED 알림을 만들지 않음.
- **읽기 오류 분류.** 손상 JSON·검증 실패·예기치 않은 S3 오류(AccessDenied, 스로틀 등)는 `error`로 분류해 **경고 로그 + 헬스 STALE**로 올립니다. `NoSuchKey`/`NoSuchBucket`/404만 조용한 `absent`입니다. 분류는 로그 레벨과 보고 상태만 바꾸며 **어떤 ClientError도 raise하지 않고** 항상 라이브 수집으로 폴백합니다(예전엔 권한 오류가 "파일 없음"과 똑같이 info 로그로 묻혔음).
- **sync 스크립트의 빈 봉투 기록.** `sync_*_to_s3.py`는 항목이 0건이어도 봉투를 **항상 업로드**합니다 — `generated_at` 스탬프가 "sync가 돌았다"의 유일한 증거이기 때문. 단 이는 `collect()`가 정상 반환한 경우뿐이며, 수집기 예외는 그대로 전파되어 **직전의 좋은 park 파일을 덮어쓰지 않습니다**.

**RSS** (`rss.py`)
- **소스:** `config.collectors.rss.feeds`에 대해 feedparser 사용.
- **메타데이터:** `feed_url`, `feed_title`.
- **fan-out 상한(`max_concurrency`, 기본 5):** 피드마다 `feedparser.parse`가 워커 스레드를 점유하므로, 수십 개 피드를 한꺼번에 던지면 기본 asyncio executor(2 vCPU Lambda에서 6)가 초과 구독되어 **파싱이 시작되기도 전에 per-feed 타임아웃이 만료**됩니다(멀쩡한 피드가 FAILED로 집계). 세마포어는 `collect()` 안(실행 중인 루프)에서 만들고 **per-feed 타임아웃보다 먼저 획득**해 타임아웃이 큐 대기가 아니라 fetch 자체를 재게 함 — RSSHub와 동일한 패턴.
- **일시적 실패 재시도:** 타임아웃과 **일시적 상태 코드**(429/5xx — YouTube 수집기의 `_RETRIABLE_STATUS_CODES`를 그대로 재사용)는 `retry_async`로 `max_retries`(기본 3)까지 재시도한다. 재시도는 **타임아웃을 감싸므로** 매 시도가 자기 `request_timeout`을 온전히 갖는다(예전엔 한 번의 blip이 그 피드의 하루치 항목을 통째로 잃었다). 403/404·파싱 불가 본문은 재시도해도 결론이 안 바뀌므로 첫 응답에서 즉시 실패.
- **최악 wall time:** 피드당 `max_retries * request_timeout + 선형 backoff` = 기본값에서 `3*30s + (5s+10s)` = 105초, 피드는 `max_concurrency`개씩 도므로 수집기 전체는 `ceil(feeds / max_concurrency) * 105s`(운영 설정 22 피드/5 동시 ≈ 8.8분) — 다른 수집기와 병렬로 도는 수집 단계 전체가 다이제스트 Lambda의 15분 예산 안.
- **실패 신호:** 죽은 피드(HTTP 4xx/5xx, entries 없는 bozo)와 재시도를 소진한 타임아웃은 빈 결과가 아니라 **예외**로 올림. `gather_collector_results(raise_if_all_failed=True)`가 **전 피드 실패일 때만** 승격시키므로, 일부 피드 장애는 로깅 후 건너뛰고(부분 허용) 전면 장애는 FAILED로 알림.

**Reddit** (`reddit.py`)
- **방식:** 공개 `.rss` 피드 사용 — `https://www.reddit.com/r/{sub}/{sort}/.rss`.
- **이유:** Reddit이 셀프서비스 OAuth 앱 생성을 동결(Responsible Builder Policy, 2025-11)했고 `.json` API는
  데이터센터 IP를 차단했지만, `.rss` 피드는 열려 있음.
- **경로:** `parse_feed_with_fallback`로 직접-우선/Cloudflare 프록시-폴백(`.rss`는 프록시에서 403, 직접은 200이라 직접이 먼저) — AWS Lambda IP에서도 동작. 자격증명·앱 등록 불필요.
- **레이트리밋 대응:** 서브레딧을 `asyncio.gather`로 동시 요청하면 단일 IP의 버스트가 429를 유발(관측상 매 실행 한 서브레딧 유실)하므로, **순차 수집 + 요청 간 간격**을 두고 각 fetch는 **지터를 곁들인 재시도**(429/5xx만 재시도, 서브레딧명 시드 기반 결정적 지터; 404 등 영구 오류는 즉시 실패). `feedparser.parse`엔 타임아웃이 없어 `asyncio.wait_for`로 감싸 매달린 fetch를 막음(타임아웃은 재시도 대상). 전 서브레딧 실패 시에만 RuntimeError로 올려 헬스체크가 FAILED로 알림.
- **트레이드오프:** RSS엔 `score`/`num_comments`(engagement)가 없어 랭킹은 LLM 품질 판단에 의존.

**RSSHub** (`rsshub.py`)
- **소스:** 로컬/컨테이너 RSSHub를 통한 X/Twitter 피드; S3에 사전 동기화된 스냅샷(`rsshub_items.json`, `scripts/sync_rsshub_to_s3.py`가 적재)을 공유 `load_items_from_s3`로 로드 가능.
- **타임아웃:** 계정별 `feedparser.parse`를 `asyncio.wait_for(request_timeout)`로 감싸(RSS와 동일) 매달린 피드 호스트가 워커 스레드를 무한 점유하지 못하게 함. 타임아웃은 빈 결과가 아니라 **실패**로 집계.
- **팬아웃 상한:** 계정 수가 40+라 모든 `parse`를 한 번에 띄우면 기본 asyncio executor가 과가입되어 아직 시작도 못 한 fetch의 `wait_for`가 먼저 만료됨. `collect()` 안에서(임포트/`__init__`이 아니라 **실행 중인 루프**에서) `asyncio.Semaphore(max_concurrency)`를 만들고 **타임아웃보다 먼저 획득**해, 타임아웃이 큐 대기가 아닌 실제 fetch를 재도록 함. 최악 벽시계 = `ceil(accounts / max_concurrency) × request_timeout`으로 Lambda 예산 안.
- **헬스:** 실패/빈 계정을 자체 추적하며 `error_rate_threshold` 보유. 서비스 도달성(`_check_reachable`)이 OK인데 **모든 계정이 실패**하면 RuntimeError로 올려 FAILED로 알림(조용한 날과 구분). 일부 실패는 허용.

**YouTube** (`youtube.py`)
- **소스:** AWS에선 `scripts/sync_youtube_to_s3.py`가 거주용 IP로 자막까지 수집해 적재한 `youtube_items.json`을 공유 `load_items_from_s3`로 먼저 로드(강력 선호). 없으면 `YOUTUBE_API_KEY`로 라이브 수집, 키도 없으면 프록시 경유 RSS 폴백.
- **API 키 해석은 1회, 루프 밖에서:** `collect()`가 park 파일 확인 **후** `asyncio.to_thread`로 `resolve_secret`(env → SSM)을 **한 번** 호출하고 결과를 `self.api_key`에 둡니다. 예전엔 lazy 프로퍼티가 채널 fan-out 안에서 처음 접근될 때 블로킹 SSM 호출을 이벤트 루프 스레드에서 돌렸습니다. park 파일로 단축되는 경로는 SSM을 아예 건드리지 않습니다.
- **채널 ID 해석:** Data API의 `forHandle` 룩업으로 @handle → canonical UC id를 해석(Lambda IP에서도 동작). 워치 페이지 HTML 스크레이프는 데이터센터 IP에서 차단되므로 API가 해석 실패할 때(예: @handle 없는 URL)만 폴백.
- **자막 언어 폴백:** 설정 언어(`transcript_language`)를 먼저 시도하고, 없으면 영상에 존재하는 임의 자막(비영어 채널·자동 생성 트랙)으로 폴백 — 'en' 트랙 부재가 빈 본문으로 떨어지지 않게 함. 라이브(데이터센터 IP) fetch는 차단되어 본문이 description으로 떨어지므로 S3 park 파일이 선호됨.
- **실패 신호:** API 거부(쿼터 소진·키 폐기 등 non-200), 깨진 JSON, 채널 ID 해석 실패는 빈 결과가 아니라 예외. 채널 하나의 실패는 허용되고, **모든 채널 실패**일 때만 FAILED로 승격.
- **다양성:** `max_videos_per_channel=1`로 고빈도 채널이 후보 풀을 독점하지 못하게 함.

**WebSearch** (`web_search.py`)
- **소스:** LLM 쿼리 정제(`RefineQueryPrompt`)를 곁들인 Tavily 검색.
- **날짜 파싱:** `_parse_date`는 Tavily의 date-only(`2026-07-10`)·tz 없는 ISO 문자열을 UTC로 정규화 — naive datetime을 tz-aware cutoff와 비교하다 TypeError로 결과가 조용히 드롭되지 않게 함.

**동시 실행 & 헬스.**
- `gather_collector_results(tasks, labels, raise_if_all_failed=False)` — 작업을 동시 실행하고 작업별 예외를 로깅 후 건너뜀. 반환은 평탄한 리스트가 아니라 `CollectorRunResult(items, total, failed, empty)`로, **몇 개의 입력이 응답했는지**가 항목과 함께 흐른다(항목 수만으로는 40개 피드 중 2개만 답한 실행이 건강한 실행과 구분되지 않았다). `raise_if_all_failed=True`(RSS·YouTube·Reddit 수집기가 사용)면 **모든 작업이 실패했을 때만** RuntimeError를 올려 소스가 EMPTY가 아니라 FAILED로 분류되게 함 — 부분 실패 허용은 그대로.
- `BaseCollector.record_run_health(total, failed, empty, threshold, what, hint)` / `flag_degraded_park(parked, ...)` — 실패율이 `error_rate_threshold`를 넘으면 `degraded_detail`을 세우고, 같은 카운트를 `run_meta`(park-meta 키)에 남겨 sync 스크립트가 항목과 함께 park한다. RSSHub 전용 코드였던 것을 **모든 수집기가 쓰는 한 구현**으로 올렸다 — 한 소스만 반쪽 상태를 보고하고 나머지는 침묵하는 상태를 없애기 위함. **보고 전용**: 어떤 항목도 필터링하지 않는다.
- `main.run_collectors_with_health()` — 헬스 리포팅용으로 동일 작업을 실행하되 `HealthReport`(§8 참조)를 반환.
  `gather_collector_results`는 다른 호출자들을 위해 그대로 유지.

## 5. 파이프라인(Pipeline)

### 1. 집계기 (`aggregator.py`)
- **처리:** URL → 정규화 제목 순으로 중복 제거.
- **URL 정규화 (모듈 레벨 `normalize_url`):** scheme/host case/trailing slash/추적 파라미터(`utm_*`, `fbclid`, `ref` 등)/fragment를 접어 같은 기사가 https로 일치하게 함. cross-day 원장과 같은 정규형을 공유.
- **cross-day dedup:** `aggregate(items, exclude_urls=...)` — 호출자(`main.run_pipeline`)가 넘긴 정규화 URL 집합(최근 발행 기사)을 랭킹 이전에 제외해, 같은 스토리가 며칠 간격으로 재요약되지 않게 하고 랭커 토큰도 절약(§7-c 참조). **핀 항목(`--pin-url`)은 URL·제목 dedup을 모두 우회** — 사용자가 오늘 명시 요청한 URL이라, 최근 발행됐거나 제목이 겹쳐도 살아남아 랭커의 핀 복구까지 도달.
- **survivor 선택:** 중복이 걸리면 먼저 온 항목을 무조건 유지하지 않고 **품질 우선으로 승자 선택**(`_pick_survivor`: 핀 > 더 긴 본문 > 먼저 온 것). 얇은 Reddit `.rss` 링크-포스트가 같은 기사의 전문 RSS/웹 항목을 수집 순서만으로 밀어내지 않게 함(랭커·다이제스트가 승자의 `text`를 읽음). 동점은 먼저 온 것 유지(결정성). 패자의 메타데이터는 승자에 없는 키만 채움(origin/engagement 미덮어쓰기).

### 2. 랭커 (`ranker.py`)
- **입력:** 항목 포맷팅(engagement + origin 포함). `Origin` 줄은 `format_origin_label`이 만들고 **web-search 항목도 포함**한다(URL 호스트, `resolve_origin_key`와 동일하게 `netloc`에서 `www.`만 제거 — 도메인/권위 표도 PSL 로직도 없다). 프롬프트가 "Source Authority"를 채점하는데 web 항목만 매체명이 빠져 있어서 콘텐츠 팜과 통신사 기사가 구별되지 않았다. Tavily의 relevance score는 **의도적으로 넣지 않는다**(검색 적합도는 소스 권위가 아니고, 검증되지 않은 신호를 랭킹 입력에 더하게 된다).
- **점수 산출:** Claude Opus 4.8로 `RankingPrompt` 병렬 배치 호출 → JSON 점수 파싱.
- **배치 재시도 & fan-out 상한:** 각 배치의 Converse 호출은 `retry_async`로 재시도합니다(`ranking_max_retries` 기본 3, `ranking_retry_backoff_sec` 기본 5초 선형 백오프). 예전엔 한 번의 스로틀/일시적 5xx가 `[]`로 삼켜져 경고 한 줄만 남기고 **후보 40건이 그날 풀에서 조용히 사라졌습니다**. 동시에 in-flight 배치 수를 `ranking_max_concurrency`(기본 4)로 묶어 큰 날에 스스로 ThrottlingException을 유발하지 않게 합니다(세마포어는 `rank()` 안, 실행 중인 루프에서 생성).
- **전면 실패만 승격:** 재시도까지 실패한 배치가 있으면 **ERROR로 남기고**(사라진 후보 수까지 로그) 나머지 배치 결과로 계속 진행하고, **모든 배치가 실패**했을 때만 RuntimeError로 올려 실행이 FAILED로 잡히게 합니다(`gather_collector_results(raise_if_all_failed=True)`와 같은 규칙). 같은 판정을 `ContentRanker.health`(`RankingHealth`)에 남겨 `run_pipeline`이 `DigestResult.ranking_health`로 실어 보내고, 다이제스트 Lambda가 **파이프라인 이후 별도 SNS 알림**으로 게시합니다 — 배치 하나가 사라진 다이제스트도 겉보기엔 완전히 정상이기 때문입니다. 파싱 실패(모델이 JSON이 아닌 문자열 반환)는 예전처럼 빈 결과로 degrade — 핀 복구 경로가 그 배치의 핀을 min_score로 되살립니다.
- **오버선정 + 코어/백필 구분:** `rank(items, select_count, core_count)`은 `top_n + digest_candidate_buffer`(기본 3)만큼 넘기되, **소스 슬롯 보장은 `core_count`(=top_n) 코어에만** 적용한다. 슬롯을 `top_n + buffer` 전체에 적용하면 어떤 소스의 보장 슬롯이 에디터가 끝내 쓰지 않는 후보로 충족될 수 있어, 독자가 받는 다이제스트에는 그 보장이 존재하지 않았다. 버퍼분은 그대로 전부 넘기고 `RankedItem.backfill=True`로 표시하며, 프롬프트에는 문장을 추가하지 않고 `_format_ranked_items`가 항목별 `BACKFILL:` 필드로 알린다(`MUST INCLUDE`와 같은 방식). 백필 후보도 완전히 사용 가능하므로 "병합 후 보충" 동작은 그대로다.
- **origin 가산 보정:** `origin_weights`를 가산 보정으로 적용 — `score + (weight-1.0)*origin_weight_nudge`를
  [0,1]로 클램프(곱셈 배수가 아님). 미등록 origin엔 `origin_weight_default`.
- **필터:** `min_score` 필터 적용.
- **grace 구제 (`_grace_candidates`, `source_slot_score_grace` 기본 0.1):** 슬롯을 보유한 소스가 min_score 위에 단 하나도 없으면, grace 밴드(`min_score - grace`) 안의 최선 항목 1건을 후보로 admit. 절대 점수 프롬프트가 체계적으로 저평가하는 대화체 소스(영상/팟캐스트 transcript vs 짧은 기사)가 전부 차단되지 않게 함. grace 항목은 **자기 소스의 보장 슬롯만** 채울 수 있고 완화된 fallback fill에선 제외(조용한 날 약한 항목으로 패딩 방지).
- **선정/다양성 (`_apply_source_slots`):**
  - `source_slots`로 소스별 기본 슬롯 채우기.
  - `source_cap_multiplier × slot`까지 오버플로 채우기.
  - `max_per_origin`으로 하나의 origin 키(채널/작성자/서브레딧)가 차지하는 항목 수 제한 — 단일 채널
    독점에 대한 근본 해결책.
  - fill 패스는 하나의 `fill(respect_origin, respect_source)` 루프로 통일되어 **어떤 캡을 지키는지만** 다름:
    ① 캡 둘 다 → ② per-origin 캡만 완화(source 캡 유지) → ③ **최후 수단**으로 source 캡까지 완화(먼저
    max_per_origin을 만족하는 후보부터). ③은 `len(selected) < limit`일 때만 들어가고 발동 시 INFO 1줄을
    남긴다(수집기 부분 장애가 보이도록) — 남은 후보가 전부 한 소스에 몰린 날 다양성 캡이 읽을 스토리 수를
    깎지 않게 하기 위함. grace 항목은 ②③에서 모두 제외.
  - origin은 `resolve_origin_key`로 해석: YouTube→channel_url, Reddit→subreddit, RSS→feed_url, X→author, **Web→URL 호스트**(`urlparse().netloc`에서 `www.` 제거 — PSL/등록가능도메인 휴리스틱이 아니라 서브도메인은 별개 origin). 호스트 키가 없던 시절 web 항목은 origin 캡을 전부 우회해 한 매체가 여러 슬롯을 차지할 수 있었음.
  - **핀 항목도 캡에 계수:** 핀은 `rank()`가 앞에 붙이고 이 fill을 통과하지 않아 origin/source가 계수되지 않았음 → 핀과 같은 origin 항목이 나란히 실렸다. 이제 카운터를 핀으로 **선(先)채운** 뒤 채우며, 캡 때문에 미달이면 마지막 완화 패스가 top_n까지 메움.

### 3. 트렌드 트래커 (`trend_tracker.py`)
- **상태:** 구조화 `trends.json` 유지 — slug id, 증거 리스트.
- **생명주기:** 날짜 기반 상태(active/cooling/archived), momentum 감쇠 랭킹, active 캡 아카이브 (§7 참조).

### 4. 다이제스트 생성기 (`digest_generator.py`)
- **처리:** Claude Sonnet 5로 `DigestPrompt` → **구조화 `DigestContent`**(Pydantic: `lead`, 코드가 항상 1로 고정하는 `headline_index`, `items[]` 각각 title/url/source_tag/metrics/body/implication). LLM은 산문(lead·body·implication)만 작성하고, source tag·metrics는 코드(`_fill_source_metadata`)가 URL로 매칭해 채움 — 매칭 키는 집계기의 `normalize_url`이라 에디터가 URL을 되쓸 때 생긴 trailing slash·http→https·utm 파라미터 차이로 소스 줄이 통째로 사라지지 않는다(URL이 이미 동일하면 동작 변화 없음). 랭킹 소스와 끝내 매칭되지 않는 항목은 최후 수단으로 `urlsplit(url).netloc`을 태그로 쓴다(도메인 매핑 표는 두지 않음).
- **파싱 견고성(`_parse_content`):** LLM JSON은 `parse_json_from_llm_output`(`strict=False` — 문자열 값 안의 raw 제어문자 허용)로 파싱하고, **items를 개별 검증**해 한 항목이 malformed여도(예: url/body 누락) 그 항목만 스킵하고 나머지는 유지(전체를 0-item으로 무너뜨리지 않음). 단 **items[0](헤드라인)이 검증 실패하거나** lead가 없거나 JSON이 통째로 깨지면 `DigestContentError`를 **raise**한다 — 예전의 minimal 폴백(`lead=raw[:1000], items=[]`)이 2026-08-13·08-17에 다섯 스토리를 전부 잃은 채로 게시된 경로였다. 호출자는 `digest_max_retries`만큼 재질의하고, 계속 실패하면 실행 자체가 실패로 남아 깨진 게시물이 나가지 않는다.
- **JSON 키 순서가 load-bearing:** 프롬프트는 `items`를 **먼저**, `lead`를 **마지막에** 요청한다. 이미 쓴
  스토리에 대한 논평으로 lead를 쓰게 만드는 장치이고, 측정된 효과가 있다(헤드라인 reply와의 단어 겹침
  0.21–0.41 → 0.03–0.21). `headline_index`는 프롬프트에서 아예 빼고 코드가 1로 고정한다(에디터가 lead와
  비주얼을 서로 다른 스토리로 가리킬 수 없게). **`DigestContent`의 필드 선언 순서(lead, headline_index,
  items)에 맞춰 프롬프트 키 순서를 되돌리지 말 것** — 모델은 쓰는 순서대로 사고하므로 그 '정리'는 겹침
  회귀다.
- **예산을 코드가 계산해 전달:** 항목 산문 예산은 추정치가 아니라 **코드가 소유한 고정 파트**에서 파생한다
  (`_item_prose_budget` → 500 − 후보 중 최악의 `URL + 소스 줄 + 빈 줄 구분자`, `threads_item_overhead_chars`).
  이 숫자에는 **에디터가 쓰는 title도 포함**된다(예전엔 body+implication만 세어 한국어 제목이 예산 밖에서
  소비되었고, 표본 95건 중 5건이 마지막 문장을 잃었다). `digest_item_prose_max_chars`(기본 380)는 상한
  ceiling일 뿐이고 0이면 채널 캡 없음. lead도 예산을 받는다(`_lead_budget` → 500 − 코드가 붙이는 카운트다운
  개그와 그 앞 빈 줄).
- **target_count + recent_leads + recent_titles:** `generate(..., recent_leads=..., recent_titles=...)`. 프롬프트에 `target_count`와 `recent_leads`(최근 며칠 lead — "이 오프닝 각은 피하라", 특정 문구를 금지하지 않고 일반화. **각 lead의 첫 문장만** 보여준다: 달라야 하는 건 오프닝 각이고 그것이 첫 문장이다. 잘라내기는 저장 포맷이 아니라 **포맷 시점**(`_format_recent_leads`)에 일어나므로 전문(全文)으로 저장된 기존 이력도 그대로 동작하고, 마이그레이션이 없다. `RECENT_LEAD_PREVIEW_CHARS`는 문장 경계가 없는 산문용 백스톱), `recent_titles`(직전 다이제스트가 실은 **스토리 제목 목록** — 오늘이 그것의 재방송이 되지 않게. 프레이밍은 한 줄이고 임계·유사도 휴리스틱은 없다. 실제로 재발행을 막는 건 여전히 URL 원장이며 여기선 정보로만 준다. `main.run_pipeline`이 cross-day dedup이 이미 가져온 스냅샷에서 뽑으므로 추가 호출도 없다)를 함께 넣음. `target_count`는 기본 `min(top_n, 후보수)`이되, 사용자가 top_n보다 많은 URL을 핀하면 **헤드라인 1 + 전체 핀**을 담도록 상향(핀도 헤드라인도 트림에 안 밀리게). 에디터는 오버선정 후보를 병합해 정확히 target_count개의 distinct 스토리를 내되, **모델이 초과 emit하면 코드가 트림**(`_trim_keeping_pinned`: 결정론적 상한; items[0] 헤드라인 우선 보존 후 나머지 슬롯에 핀 보존).
- **Slack 마크업 없음:** 다이제스트 경로는 `sanitize_slack_mrkdwn`을 호출하지 **않음**(그 정규화는 이제 딥 리서치 경로 전용 — `output/delivery.py`의 `_deliver_slack`이 모델이 흘린 마크업을 1차로 보정하고, `agent_runtime/app.py` 폴백이 동일 정규화를 적용). 채널별 마크업은 각 렌더러가 붙임.
- **시스템 오브 레코드:** `render_digest_text`가 구조화 콘텐츠를 평문 산문으로 렌더해 `digest_text`를 만들고, 이는 트렌드 분류기·AgentCore 스냅샷이 사용.
- **그라운딩(옵션, `enable_grounding_check`):** 산문 필드의 구체적 주장을 소스 항목(+코드 산출 트렌드 사실)에 대조해 근거 없는 부분만 외과적으로 수정.

### 5. 채널별 렌더링 (`output/renderers.py`)
구조화 `DigestContent`를 채널 포맷으로 변환(다이제스트 경로는 채널마다 다른 렌더러를 통과):
- **`render_slack_blocks`:** Slack Block Kit — header / lead section / (이미지) / 항목별 divider·title 링크·source·metrics context·body는 `rich_text_quote`·implication. 메시지당 블록 상한으로 청크 분할. `output/slack_handler.send_digest_to_slack`가 사용.
- **`render_threads_posts`:** Threads용 — root는 lead, 항목마다 평탄한 reply 하나(≤500자, 문장 경계로 트림, title·소스 줄·URL 유지, Slack 마크업 없음). **implication은 body와 한 단락으로 이어 붙이지 않고 자기 블록**(빈 줄로 분리)으로 나간다 — 이어 붙이면 목소리 줄이 그냥 본문의 마지막 문장처럼 읽혀 항목이 착지하는 지점이 사라졌다. 이 추가 구분자는 `threads_item_overhead_chars`(에디터에게 알려주는 파생 예산)와 `_item_post_overflows`(트림 카운트)에도 **똑같이** 반영돼 예산이 정확히 유지된다. 산문이 캡에 걸려 잘린 항목이 있으면 **개수만** WARNING으로 남긴다(본문 텍스트는 로깅하지 않음) — 에디터가 산문 예산을 넘겼다는 신호.
- **`render_research_blocks`:** 딥 리서치 리포트(Slack mrkdwn)를 다이제스트와 같은 룩의 Block Kit로 렌더 — header 블록(`:satellite: OmniSummary Deep Research`) 뒤로, 번호 매긴 섹션 제목(`*N. ...*`)마다 그 앞에 divider를 넣어 한 덩어리 텍스트가 아니라 깔끔히 구획된 형태로 보이게 함(header 바로 아래 divider는 빈 띠로 보이므로 억제). 산문은 `SLACK_MAX_SECTION_CHARS`(2900) 단위로 단락 패킹하고 메시지당 블록 캡으로 청크 분할. 리서치 Slack 경로(`output/delivery.py` `_deliver_slack`)의 기본 렌더러.
- **`render_threads_research`:** 딥 리서치 리포트를 Threads용 root + 평탄한 reply chain(각 ≤500자)으로 렌더. 에이전트가 `---`만 있는 줄로 자기 게시물 경계를 표시하므로(번호+제목+본문이 한 게시물에 묶임), 렌더러는 그 경계를 존중하고 500자 초과 게시물만 문장 경계로 재분할(인용 URL 보존). 구분자가 없는 구버전 출력은 문장 패킹으로 폴백. `max_posts`(>0)로 총 게시물 수를 **하드 캡**해 초과분을 드롭. Slack 마크업은 `_strip_slack_mrkdwn`으로 제거(`<url|label>`→`label (url)`, `*bold*`/`_italic_`/`` `code` `` 마커 제거, 단 URL은 보호).
- **`render_agent_blocks`:** 구조 없는 자유형 에이전트 텍스트를 Block Kit section으로 단순 단락 패킹/래핑하는 **폴백 전용** 래퍼 — 이제 `agent_runtime/app.py`의 Slack 폴백(`_send_slack_message`, 에이전트가 `deliver_report`를 끝내 호출하지 않았거나 Slack 전달이 실패한 경우)에서만 쓰임. 정상 리서치 Slack 경로는 `render_research_blocks`를 사용.

### 5.1 데일리 비주얼 (`daily_visual.py`, `enable_daily_visual`)
- **트리거:** 다이제스트 전송 후 실행.
- **스토리 선택:** **헤드라인(`items[0]`)을 그림 — lead·이미지·텍스트가 한 스토리로 일치**하도록 강제. 에디터는 무엇을 그릴지(HOW)만 브리핑한다(프롬프트에서 `item_number`를 아예 요구하지 않는다 — 헤드라인은 상류에서 마킹되고 코드는 그 값을 읽지 않았다). 다이제스트 프롬프트의 헤드라인 선정은 **중요도 우선이고, 시각화 용이성은 동등하게 중요한 스토리 사이의 tie-break만** 한다(그 이상으로 몰면 deep-tech 뉴스가 헤드라인에서 밀리고 에디터의 드문 `skip` 경로가 흔해진다). 적합하지 않으면 `skip`.
- **포맷 변주 (`visual_formats.json`, `RollingLog`, `visual_format_window` 기본 6):** 최근 비주얼의 orientation+format을 추적하고, 가장 오래 안 쓴(least-recently-used) orientation을 에디터 프롬프트(`format_guidance`)와 생성 instruction에 주입해 연속된 비주얼이 모양/구성에서 실제로 달라지게 함. LRU는 orientation별 **마지막 사용 인덱스**로 계산(윈도의 첫 항목을 그대로 집으면 나중에 다시 쓴 orientation—즉 가장 최근 것—을 고르게 됨). 게시 후 선택한 포맷을 `date`로 dedup해 기록(같은 날 재실행은 교체, 변주 윈도 잠식 방지). 상태 스토어 초기화 실패 시 히스토리 없이 degrade(크래시 없음).
- **멀티패널 비율 유도 (`visual_multi_panel_target_ratio` 기본 0.34):** 에디터는 방치하면 단컷 구성으로 기울기 때문에, 최근 윈도의 멀티패널 비중이 목표보다 낮으면 "시퀀스·반전·설정과 응수가 있으면 멀티패널 만화로" 쪽으로, 높으면 단컷 쪽으로 프롬프트를 **soft-steer**함. 쿼터가 아니라 유도이며 스토리가 최종 결정. 0이면 유도 없음(순수 에디터 판단), 히스토리에 해당 키를 기록한 항목이 없으면 근거가 없으니 유도를 건너뜀.
- **재등장 캐릭터 (`visual_character_enabled`, `visual_character_sheet`, `visual_character_target_ratio`):** 에디터가 이 스토리에 맞다고 판단한 날(`use_character`)에만 등장하며, 캐릭터 시트를 instruction에 주입해 이미지 모델이 **같은 인물**을 그리게 함. 정체성은 시그니처 소품에 실려 매일 바뀌는 화풍을 견딤. 시트는 의도적으로 얇게 유지 — 참조 과적합과 해부 붕괴를 유발했기 때문(`0d79b33`). 등장 빈도도 최근 윈도 기준으로 목표 비율을 향해 soft-steer(0이면 유도 없음)하되, 스토리에 안 맞으면 에디터가 여전히 건너뜀.
- **편집 관점 전달:** 다이제스트의 리드(카운트다운 접두 제거)와 헤드라인 항목의 `implication`을 instruction에 **정보로** 넘김. 아트 디렉터가 원본 기사만 보던 탓에 표면 사실만 그리는 문제(2026-08-15: 논지는 "출시 주기가 격차의 원인"인데 그림은 4자 동시 골인 = "다 비슷하다")를 막기 위한 것. 일치를 **강제하지 않음** — 이미지가 리드의 논지를 논증해야 한다는 제약은 과하다고 판단.
- **가드레일 (`visual_guardrails`, 비우면 미적용):** 스타일도 논지 요구도 아니라 이미지가 **하지 말아야** 할 두
  가지다. (1) 받은 편집 관점의 정서를 **뒤집지 말 것** — 2026-08-18 실행이 순환 벤더 파이낸싱("누가 위험을
  지는지는 다음 다운턴에야 드러난다")에 대한 리드를 로켓과 지폐가 쏟아지는 승리 포스터로 그렸다. "논지를
  논증하라"보다 훨씬 약한 요구이며, 그 강한 규칙은 과한 제약으로 기각됐다. (2) 기업·국가를 **인종으로 코딩된
  인물로 의인화하지 말 것**(2026-08-15 비주얼이 모델 경쟁을 각 랩 국적의 육상 선수로 그렸다). **실존 인물을
  알아보게 그리는 것은 허용·권장**된다 — 시사만평의 표준 관행이고 계정 주인의 편집 판단이다.
  ⚠️ 이 문구의 **인과 효과는 미증명**이다. 문제가 났던 케이스로 A/B를 시도했지만 에디터가 합성 content가 아닌
  `ranked_items` 헤드라인을 그려 실험이 무효였고, 이미지 생성이 확률적이라 팔당 1샘플로는 노이즈와 구분되지
  않는다. 비용 0·config로 즉시 해제 가능·지시문 포함 여부는 테스트로 고정 — 효과는 주장하지 않는다.
- **맥락 보강:** 에디터가 고른 리서치 스텝(papers/community/news)을 실행해 맥락 수집.
- **생성:** `VisualGenerator`(시놉시스 → gpt-image)로 1컷 밈/패러디/일러스트 또는 N컷 카툰 생성 → Slack 게시(+`enable_threads_post` 시 Threads에도 게시).
- **플랜 파싱 실패:** 에디터 JSON을 못 읽으면 `{"skip": True}`로 취급해 그대로 건너뜀 — 재질의(추가 LLM 호출)도, 일반 폴백 instruction으로 gpt-image를 태우는 낭비 렌더도 하지 않음.
- **비주얼 실패가 다이제스트를 삼키지 않음:** 이미지는 **첨부물**이고 이 함수가 Threads의 유일한 게시 경로다. OpenAI 키 없음·에디터 호출 실패·에디터 skip·렌더 실패는 모두 `_make_visual` 안에서 흡수되어 `(None, None)`으로 떨어지고, `run()`은 그대로 **텍스트 전용**으로 Threads(lead + 스토리별 reply)를 게시한다. 예전엔 이 세 경우가 게시 이전에 `return False`였기 때문에 비주얼만의 문제로 그날 다이제스트가 조용히 사라졌다. OpenAI 키는 `strict=True`로 읽어 **"미설정"과 "SSM을 읽지 못함"을 구분**하되(느슨한 읽기는 둘 다 `""`여서 파라미터 스토어 장애가 의도된 설정처럼 보였다), `SecretUnavailableError`는 **`_make_visual` 안에서 잡는다** — 엄격한 시크릿 읽기가 텍스트 다이제스트를 비용으로 삼는 일은 없어야 한다.
- **instruction 빌더가 하나(`_build_instruction`):** 편집 관점·guardrails·포맷 유도·캐릭터 시트를 붙여 최종 아트 디렉터 instruction을 만드는 부분은 I/O 없는 **순수 함수**로 분리돼 있다. `scripts/sample_visual_brief.py`가 **프로덕션이 실제로 보내는 문자열**을 채점할 수 있어야 하기 때문이다 — 예전 샘플러는 맨 `plan["instruction"]`만 브리핑해서, 편집 관점도 guardrails도 포맷 유도도 캐릭터도 없는(=배포되지 않는) 프롬프트를 평가했다. 샘플러는 다이제스트를 먼저 생성해 **실제 `DigestContent`**를 넘기고(없는 편집 관점을 지어내지 않음), 테스트가 두 경로의 출력이 바이트 단위로 같음을 고정한다.
- **이미 게시된 날 조기 종료:** `run()` 맨 앞에서 "게시할 것이 남아 있지 않다"(Threads 원장에 오늘이 있고 `enable_slack_post`가 꺼져 있으며 force가 아님)를 확인하면 에디터 호출·gpt-image 렌더 비용을 아예 쓰지 않는다. 게이트는 의도적으로 좁다 — Slack 전달이 켜져 있으면 이미지에는 Threads 마커와 무관한 별도 목적지가 있으므로 그대로 진행한다.
- **스토리 없는 날은 렌더를 사지 않는다:** `_render_would_be_wasted(content)` — 스토리가 0건이면 Threads는 (의도적으로) 게시하지 않고, `enable_slack_post`가 꺼져 있으면 이미지에 남은 목적지가 없다. 두 조건이 **동시에** 참일 때만 렌더 이전에 종료한다(Slack이 켜져 있으면 업로드가 실제 목적지이므로 그대로 진행). 판정은 순수 predicate로 두고, 로깅과 `threads_outcome = ThreadsDelivery(0, 1)` 기록은 **`run()`이** 한다 — predicate 안에서 상태를 바꾸지 않으며, 이 기록 덕분에 그날의 전달 알림이 no-op이 되지 않는다.
- **헤드라인 매핑:** `content.headline_index`(큐레이션 items 기준)를 `normalize_url`로 랭킹 항목에 되매핑한다. 끝내 매칭되지 않으면 예전의 `or 1`(= 랭킹 1위, lead와 **다른** 스토리)이 아니라 **큐레이션 헤드라인 자신의 title/body/implication**을 소스로 브리핑해 이미지와 텍스트의 동기화를 지킨다(에디터에게 넘기는 헤드라인 마커는 0 = 없음).
- **성공 판정:** `run()`은 **활성화된 채널 중 하나라도 게시 성공**하면 True(Slack만 보던 시절엔 `enable_slack_post: false` 구성에서 Threads가 성공해도 'skipped'로 기록됐다). 게시 결과(`ThreadsDelivery`)는 `maker.threads_outcome`으로 노출되어 비주얼 Lambda가 부분 전달을 알림으로 올린다.
- **best-effort:** 파이프라인을 막지 않으며, 실패는 항상 로깅된다.

### 5.2 AGI 카운트다운 인트로 (`shared/formatting.py` `agi_countdown_intro`)
- **동작:** "AGI 등장 N일 전이다"식 인트로를 **LLM이 아니라 코드**가 계산(`agi_countdown_date` 기본 `2029-01-01`, `agi_countdown_template`). D-day **이전엔 카운트다운**, D-day 당일/이후엔 `agi_countdown_after`로 **카운트업**("AGI 등장 예정일 D+N일째, 아직이다"). 빈 `agi_countdown_date`면 비활성. 템플릿은 운영자 편집 config 문자열이므로 `.format()`을 try/except로 감싸 오타(잘못된 placeholder·괄호)면 인트로를 비워 **생성 도중 크래시하지 않음**(수집·랭킹·LLM 비용을 다 쓴 뒤 죽는 걸 방지).
- **적용 시점:** 다이제스트 **생성 시점**에 `content.lead`에 붙임(`digest_generator.generate` → `place_countdown_intro`), 그 날 실행의 KST `digest_date`로 계산. 인트로가 저장 콘텐츠의 일부가 되어 **모든 채널**(Slack Block Kit · Threads root)에 함께 나가며, 트렌드 재등장 수치와 같은 시계(날짜)를 씀.
- **위치 노브(`agi_countdown_position`, 기본 `suffix`, 배포 설정도 `suffix`):** 접두로 두면 Threads root의 **첫 줄**—피드 독자가 유일하게 보는 줄—을 매일 같은 고정 문장이 차지한다(연속 40개 게시물이 동일 문장으로 시작). `suffix`는 개그를 **문구 그대로** 두고 lead의 마지막 줄(맺음말)로 옮겨 첫 줄이 그날의 각이 되게 한다. 위치만 노브로 두고 cadence·N일마다 생략·랜덤은 두지 않는다(매직 넘버).
- **양 끝 제거(`editorial_lead`):** 최근 lead 신선도 비교와 비주얼의 편집 관점 전달은 개그를 뺀 각만 봐야 하므로, 접두/접미 **어느 쪽에 붙어 있어도** 제거한다(저장된 lead가 설정 변경 이전 것일 수 있음).
- **넘치면 개그가 먼저 나간다:** Threads root가 500자를 넘으면 `_fit_lead`가 **코드가 소유한 카운트다운 줄을 먼저 버리고** 에디터의 산문(그날의 논지)을 지킨다. 개그를 버리는 조건은 **마지막 줄이 그 개그임을 식별할 수 있을 때뿐**이다 — 호출자(`daily_visual`)가 계산한 인트로 문자열을 `render_threads_posts(content, countdown)`로 넘겨 비교한다. 마지막 줄을 무조건 버리는 방식은 `prefix` 위치(또는 개그 비활성)에서 **진짜 산문**을 삭제하므로 쓰지 않는다. 식별되지 않으면 앞에서부터 온전한 문장만 남기므로 접두 개그는 살아남는다. 트림이 발생하면 WARNING으로 남긴다(에디터가 산문 예산을 넘겼다는 신호).

### 5.3 Threads 전달 (`output/threads_handler.py`, `enable_threads_post`)
- **호출자:** 다이제스트 Lambda가 아니라 **데일리 비주얼 Lambda**(`DailyVisualMaker.run`)가 게시한다 — Threads 게시물은 이미지 root + reply chain이 한 세트라 이미지를 만든 쪽이 함께 보내야 한다. 따라서 Threads 전달에는 `enable_threads_post`와 `enable_daily_visual`이 **둘 다** 필요하다.
- **흐름(`post_to_threads`):** 이미지 root 게시 → 스토리당 reply 하나의 평탄한 chain. reply는 서로가 아니라 **모두 root에 매닮**(reply-of-reply로 중첩되면 첫 개만 보임).
- **이미지 호스팅:** Threads는 바이트 업로드가 불가하고 **공개 URL만** fetch하므로, PNG를 S3에 올리고 단기 presigned URL을 Meta에 한 번 넘김(`_upload_image_for_hosting`).
- **인덱싱 지연 폴링:** 방금 게시된 이미지 root는 곧바로 reply 대상이 되지 못해 Meta가 "media not found"(code 24 / subcode 4279009)를 반환할 수 있음. reply의 create-container 쓰기를 blind하게 재시도하는 대신(각 시도가 낭비 쓰기 + sleep), **값싼 GET으로 root가 addressable해질 때까지 한 번 폴링**(`_wait_until_addressable`)한 뒤 reply chain을 시작. 준비 여부는 root의 속성이라 chain 전체가 하나의 예산(`THREADS_INDEXING_BUDGET_SEC`≈270초)을 공유하며, 비주얼 Lambda 타임아웃 15분이 총량을 bound. TEXT-only root(이미지 없음)는 거의 즉시 인덱싱되므로 폴링 생략. reply에는 GET이 200을 준 뒤에도 드물게 나는 eventual-consistency 경계용 **짧은 안전망 재시도**(`_publish_reply_with_retry`, 기본 3회)만 남김.
- **per-reply best-effort + 전달량 회계(`ThreadsDelivery`):** reply 게시는 건별로 try/except — 한 reply가 실패해도 나머지를 포기하지 않아 댓글 chain이 중간에 끊기지 않음. 반환값은 bool이 아니라 `(posted, expected)` NamedTuple(root 포함)로, `published`(root + reply chain이면 최소 1건)와 `partial`(게시됐지만 일부 누락)을 구분한다. 예전엔 5개 중 4개만 붙어도 그냥 성공이라 truncated chain을 아무도 알 수 없었다. **호출자는 값 자체의 truthiness로 분기하면 안 된다**(NamedTuple은 `(0, 5)`도 truthy) — `daily_visual`/`delivery` 모두 `.published`를 명시적으로 읽으며, `published`가 아니면 ledger 마커를 롤백해 그날을 재시도 가능하게 둔다(이미지만 있고 스토리 없는 다이제스트를 "게시됨"으로 굳히지 않음). 부분 전달·전면 실패는 ERROR로 로깅되고, 비주얼 Lambda가 `ALERT_SNS_TOPIC_ARN`이 있을 때 SNS 알림을 올린다(없으면 no-op).
- **best-effort:** API 오류는 로깅 후 건너뜀(절대 raise 안 함 — `output/delivery.py`의 리서치 경로가 이 계약에 의존). 단 자격증명(`THREADS_ACCESS_TOKEN`/`THREADS_USER_ID`) 부재는 **ERROR**로 올린다: `enable_threads_post`가 켜진 구성에서 그날 다이제스트가 어디에도 전달되지 않는 상태인데 예전엔 평범한 INFO "skipping" 한 줄이었다.
- **토큰 갱신:** `lambda_handlers/threads_refresh_handler.py` + ~50일 주기 EventBridge 스케줄이 60일 만료 장기 토큰을 갱신해 SSM에 재기록(§11 참조).

## 6. LLM 팩토리 (`shared/utils.py`)

**모델 팩토리.** `BedrockLanguageModelFactory.get_model(model_id, **kwargs)`
- **반환:** 모델 역량(`_LANGUAGE_MODEL_INFO`)에 맞게 구성된 `ChatBedrock`/`ChatBedrockConverse`.
- **구성 역량:** thinking, 1M 컨텍스트, 성능 레이턴시, 프롬프트 캐싱.
- **리전:** `BedrockCrossRegionModelHelper`가 가능 시 `global.`/`apac.` inference-profile ID를 해석.
- **모델 ID:** `shared/constants.py`(`LanguageModelId`)에 열거; 최신은 **Opus 5 / Sonnet 5**(Opus 4.8도 유지).
  Opus 5의 역량 플래그는 버전 번호로 추정하지 않고 Converse로 **직접 검증**했다 — `temperature`와 레거시
  `thinking.type="enabled"`/`budget_tokens`는 둘 다 ValidationException이고, `adaptive` + `output_config.effort`만
  통과한다(Opus 4.7/4.8·Sonnet 5와 동일). 단가도 Opus 4.8과 같으므로 **비용 옵션이 아니라 품질 옵션**이다.
- **샘플링 파라미터 게이팅:** Sonnet 5·Opus 4.7/4.8은 비기본 `temperature`/`top_k`/`top_p`를 400으로 거부하므로, 해당 모델은 `LanguageModelInfo.supports_temperature=False`로 표시하고 팩토리가 `temperature`와 `top_k`를 함께 생략한다.

- **단계 태깅 & 사용량 로깅:** `get_model(model_id, stage="ranking"|"digest"|"trends"|"visual-editor"|
  "visual-synopsis"|"query-refine")`. 붙은 콜백이 호출마다
  `LLM usage stage=... model=... input=... output=... cache_read=... cache_write=...`를 남긴다. 청구는 **모델**
  단위인데 Sonnet 5 하나를 다이제스트·그라운딩·트렌드·비주얼·쿼리정제·리서치 에이전트가 공유하므로, 이 태그
  없이는 토큰 총량을 쓴 주체로 되짚을 수 없다(실측: 실행당 Sonnet 입력 157k > 랭커 82k). 텔레메트리는 설계상
  best-effort — 어떤 읽기 실패도 생성을 막지 못한다.
- **비용 귀속(application inference profile):** 온디맨드 `InvokeModel`은 과금 대상 리소스가 없어 비용 할당
  태그가 붙지 않는다. `BedrockCrossRegionModelHelper`가 시스템 프로필을 해석한 뒤 이 프로젝트의
  **application inference profile**(`{project}-{stage}-{model-slug}`, `scripts/put_inference_profiles.py`가
  `Project`/`Stage` 태그와 함께 생성)을 찾아 그 ARN을 반환한다. 해석이 이 한 곳이라 LangChain 팩토리와
  **Strands 리서치 에이전트가 동시에** 커버된다(에이전트는 `get_model`을 우회하므로 단계 로깅으로는 안 잡힘).
  프로필이 없거나 조회가 거부되면 시스템 프로필로 조용히 폴백한다 — 리포팅이 생성을 막아선 안 된다.
  `ChatBedrockConverse`는 ARN에 `provider`를 요구하므로 config 빌더가 `provider="anthropic"`을 붙인다.
  IAM 주의: `application-inference-profile`은 `inference-profile`과 **다른 ARN 리소스 타입**이라 정책에
  따로 넣어야 한다(누락 시 프로필이 존재하는 순간 모든 Bedrock 호출이 AccessDenied).

**토큰 카운트.** `count_tokens(text)` / `truncate_to_tokens(text, max_tokens)`
- Bedrock CountTokens API로 권위 있는 카운트(로컬 휴리스틱 아님). 일부 베이스 모델만 CountTokens를 노출(Sonnet 4.6은 지원, Opus 4.8은 미지원 — AccessDenied/'doesn't support counting tokens')하므로, **호출자 모델과 무관하게 항상 `TOKEN_COUNT_MODEL`(Sonnet 4.6)로 카운트**. `model_id` 파라미터는 두 함수에서 제거됨.
- **토크나이저 주의:** `TOKEN_COUNT_MODEL`은 안정적으로 CountTokens를 지원하는 Sonnet 4.6으로 고정돼 있다. Sonnet 5는 토크나이저가 달라 같은 텍스트를 더 많은 토큰으로 세므로, 이 카운트는 Sonnet 5 실제 사용량을 약간 과소평가한다 — `item_text_max_tokens` 컷은 (더 넉넉한) 상한이 되므로 컨텍스트 초과 위험은 없고 보수적이다.
- cross-region `global.`/`us.` 등 프리픽스는 베이스 id로 스트립. 오류 시 char/4 추정으로 폴백. `truncate_to_tokens`는 문자 컷 지점을 이진 탐색.
- **메모이제이션:** 결과를 팩토리 인스턴스에 텍스트 해시로 캐시. 프롬프트 빌드가 같은 항목 텍스트를 랭커/다이제스트/그라운딩 단계에서 반복 카운트하고 `truncate_to_tokens`의 이진 탐색이 겹치는 prefix를 여러 번 재는데, 각각 별도 API 과금될 것을 캐시가 흡수(팩토리는 Lambda invoke당 1회 생성이라 캐시도 그 범위로 bounded).

**시크릿 헬퍼.** `resolve_secret(env_var, ssm_suffix)`
- **해석 순서:** env 우선, 그다음 SSM(`/{project}/{stage}/{suffix}`, SecureString 복호화).
- **사용처:** OpenAI 키는 이제 **데일리 비주얼의 gpt-image 렌더에서만** 사용(에이전트 측 이미지 생성 도구는 제거됨); Tavily 키는 리서치 백엔드/웹서치 수집기에서 env→SSM로 해소.

**프롬프트 캐싱.** Bedrock 프롬프트 캐싱은 Claude 기준 캐시 가능 프리픽스 최소치가 약 1024 토큰. 효과가 있는 곳에만 적용:
- **에이전트(적용):** 약 1.7K 토큰 시스템 프롬프트 + 도구 스키마가 매 ReAct 스텝마다, 그리고 멀티턴 세션 내내
  재전송되므로 Strands `BedrockModel(cache_config=CacheConfig(strategy="auto"))`(`agent/research_agent.py`)로 해당
  프리픽스를 캐싱. 검증: 첫 호출에 `cacheWriteInputTokens`, 이후 `cacheReadInputTokens` 발생.
- **파이프라인(미적용):** 단발성 프롬프트(랭커/다이제스트/트렌드/시각화 시놉시스, 모두 약 530 토큰이며 실행당
  1회 호출)는 캐시 최소치 미만이고 호출 간 재사용도 없어 의도적으로 캐싱을 적용하지 않음.

## 7. 메모리: 두 개의 분리된 저장소

트렌드 기억과 다이제스트 스냅샷은 **성격이 달라 서로 다른 저장소**에 둡니다.

**(a) 트렌드 — 구조화 `trends.json` (`StateStore`, 시스템 오브 레코드)**

- **스토어 선택(`create_state_store`):** `STATE_BUCKET`(env) → `config.aws.state_bucket_name` → 로컬 파일
  폴백 순으로, **버킷 유무만** 보고 결정합니다. 예전의 `is_running_in_aws()` 플랫폼 감지는 Lambda가 아닌
  호출자(AgentCore 런타임, 컨테이너, 실 버킷을 향한 로컬 실행)가 `STATE_BUCKET`을 들고 있어도 trends.json을
  로컬 파일시스템에 써서 트렌드 히스토리를 통째로 잃게 만들었습니다. AWS 밖에서는 세션을
  `config.aws.profile`/`region`으로 만들어 `.env`에 `STATE_BUCKET`을 둔 개발자가 자격증명을 잃지 않게 하고,
  AWS 안에서는 실행 역할(기본 세션)을 씁니다. prefix 규약은 env 경로(`S3_PREFIX`가 곧 digest-state prefix)와
  config 경로(`s3_prefix` + `/digest_state`)가 서로 다르므로 그대로 유지합니다.
- **읽기 실패 ≠ 히스토리 없음(`StateReadError`):** 예전엔 스로틀/거부된 S3 GET이 `None`을 반환해 "키가 없다"와 구분되지 않았고, 다음 read-modify-write가 그 공백을 **영구화**했다(발행 URL 원장·최근 lead·비주얼 포맷 윈도·Threads 멱등 마커가 한 번의 실패 읽기로 비워짐). 이제 `read`/`exists`는 `NoSuchKey`/404만 조용히 없음으로 처리하고 그 외 `ClientError`(및 로컬 `OSError`)는 `StateReadError`를 raise한다. 소비자는 **전부 동일하게** 처리한다: ERROR 로깅 → 히스토리는 '모름' → **쓰기 생략**. 게시 경로로는 절대 전파되지 않는다(`RollingLog.entries()`는 `[]`, `ThreadsPostLedger.already_posted()`는 `False`를 반환 — 중복 게시는 복구 가능하지만 미게시는 아니므로), 그리고 `TrendTracker`는 그 실행의 trends.json 쓰기를 건너뛴다.
- **관리 주체:** `pipeline/trend_tracker.py`의 `TrendTracker`.
- **LLM 역할:** `TrendClassifyPrompt`는 오늘 아이템이 기존 트렌드(id) 확장인지 신규인지 분류만 함. 부기는 전부 결정론적 Python.
- **결정론적 부기:**
  - 증거 날짜는 코드가 스탬프(LLM 아님).
  - 상태(active/cooling/archived)는 `last_seen` vs `trend_cooling_days`/`trend_retention_days`로 계산.
  - momentum은 recency 감쇠(`0.5^(age/half_life)`, `trend_momentum_half_life_days` 기본 7일).
  - 트렌드당 증거 `trend_max_evidence` 캡.
  - active 트렌드 수 `trend_max_active_trends` 캡(최저 momentum 아카이브).
  - **아카이브 purge:** 아카이브는 status만 바꾸고 증거를 유지하므로 "증거 없는 트렌드 제거" 규칙에 안 걸려 영구 잔존 → `trends.json`이 무한 성장. `last_seen`이 retention의 2배를 넘긴 아카이브 트렌드는 완전 제거(짧게 아카이브됐다 되살아날 여지를 남기는 grace).
  - 동일 날짜 재실행은 멱등(그날 증거 교체).
- **로드 견고성:** 전체 `TrendMemory.model_validate_json` 실패 시(스키마 드리프트·제거된 enum 값 등) 모든 history를 버리지 않고 **트렌드별로 관대하게 복구**(`_recover_trends`: 개별 검증해 살아남는 것만 유지) — 레코드 하나가 나빠도 누적 history가 통째로 날아가지 않음.
- **진실의 원천:** `trends.json`(`TrendMemory`)이 원천이고 렌더된 텍스트는 뷰.
- **주입(recurrence "ammunition"):** 다이제스트 생성 시 active/cooling 트렌드를 momentum 순으로 렌더해 `DigestPrompt`에 주입하되, 각 트렌드에 **코드가 증거에서 산출한 재등장 사실**(추적 N일째 / 서로 다른 N일 재등장 / 이번 달 N회)을 붙임(`_render_ammunition`). 이 수치는 lead의 날카로운 근거로 쓰이며 LLM이 지어내지 않음. **다이제스트용 블록만 `trend_max_active_trends`로 캡**한다(새 노브를 만들지 않고 기존 active 캡을 재사용): `visible`에는 cooling도 들어가서 20줄 넘게 넘어갈 수 있고 대부분은 에디터가 쓰지 않는 식은 실이다. **분류기용 `_render_existing`은 캡하지 않는다** — 거기서 cooling 트렌드를 숨기면 그 실이 고아가 되고 모델이 같은 주제에 중복 id를 새로 만든다.

**(b) 다이제스트 스냅샷 — AgentCore Memory (`shared/memory.py`)**
- **`AgentCoreMemoryStore`:**
  - **기록:** 오늘의 ranked 아이템 스냅샷을 단기 세션 이벤트로 기록(`create_event`, 세션 `digest-<date>`,
    `_fit_to_limit`로 100k 한도 보장).
  - **읽기:** `get_digest(date)`가 **그 날짜의 세션**(`digest-YYYY-MM-DD`)을 직접 읽는다 — 비주얼 Lambda는 자기가 트리거된 날짜의 콘텐츠를 게시해야 하므로 '최신을 읽고 날짜를 비교'는 쓸 수 없다(`digest_result.generated_at`은 UTC라서 09:00 KST 이전 실행에서는 KST 다이제스트 날짜와 항상 어긋난다). 없으면 `None`이며 **최신으로 폴백하지 않는다**(어제 스토리를 오늘 게시하는 것을 막음). 읽기 자체가 실패하면(스로틀·거부) `None`이 아니라 **`MemoryReadError`를 raise**한다 — 예전엔 '그 날 다이제스트가 없음'과 구분되지 않아 비주얼 Lambda가 게시를 건너뛰고 200을 반환했다. 게시 경로는 그대로 터뜨리고(Errors 알람 + DLQ, `retry_attempts=0`), 보강용 읽기(`main.py`의 cross-day dedup 시드)만 catch해 degrade한다. `get_latest_digest()`도 남아 있고, `_digest_session_ids`는 `list_sessions`를 **NextToken으로 페이지네이션**(세션은 삭제되지 않아 100개/페이지를 넘기면 단일 페이지가 최신 세션을 놓칠 수 있음; `MAX_SESSION_PAGES` 안전 캡).
  - **세션 안에서도 최신 이벤트를 고른다:** 한 세션은 보통 이벤트 1건이지만 같은 날 재실행이 두 번째를 append하고, `list_events`는 순서를 보장하지 않는다 — `maxResults=1`은 그날의 **첫(폐기된) 시도**를 서빙할 수 있었다. 이제 작은 페이지(`EVENTS_PER_SESSION`)를 읽고 `eventTimestamp`로 최신을 고르며, 동률일 때만 페이로드의 `digest_result.generated_at`으로 **선택적** tie-break한다(그 필드 없이 저장된 기존 스냅샷도 그대로 로드된다). 페이지 크기를 작게 두는 이유는 `get_recent_digests`가 **세션마다** 이 페이지를 읽기 때문 — 이력이 늘어도 세션당 비용이 커지지 않아야 한다. 파싱 불가 이벤트는 건너뛰되, 그래서 아무것도 남지 않으면 예외를 올려 '읽기 실패'가 '빈 날'로 읽히지 않게 한다.
  - **목적:** 데일리 비주얼 Lambda가 cross-Lambda로 이 스냅샷을 읽어 맥락을 공유하는 수단. (트렌드 회상은 별개 — 딥 리서치 에이전트의 `recall_trends`는 이 스냅샷이 아니라 `shared/constants.py`의 `TRENDS_KEY`(`trends.json`)를 직접 쿼리.)
  - **제거됨:** 시맨틱 recall/장기 전략 제거(관리형 추출이 트렌드 흐름이 아닌 안정적 사용자-사실만 뽑아 부적합).
- **`LocalMemoryStore`:** 오프라인 폴백(`digest_*.json`만).
- **윈도 조회:** `get_recent_digests(n, exclude_date, after_date)`가 cross-day dedup 시드를 제공(아래 (c)). `exclude_date`는 오늘 자기 스냅샷을 빼고(같은 날 재실행이 자기 스토리를 살림), `after_date`는 원장과 같은 TTL 윈도로 하한을 둠.

**(c) cross-day dedup 히스토리 — `StateStore` (`shared/history_store.py`)**

- **`PublishedUrlLedger` (`published_urls.json`):** 정규화 URL → 마지막 발행 ISO 날짜의 롤링 맵. TTL = `published_url_ttl_days`(기본 6). `recent_urls(today)`는 **엄격히 더 이른 날(`0 < age < ttl`)**만 반환 — 같은 날(age 0)은 제외해 같은 날 재실행이 자기 다이제스트를 재현(within-run 중복은 집계기가 처리). `record()`가 발행 URL을 오늘로 스탬프해 병합하고 TTL 밖 항목을 prune.
- **`RollingLog`:** 한 JSON blob에 담는 capped FIFO. 반복 방지용 최근 lead(`recent_leads.json`)와 비주얼 포맷 변주(`visual_formats.json`)에 사용. `append(record, dedup_key=...)`로 같은 키 값의 기존 항목을 교체 가능 — leads는 `date`로 dedup해 `--force-republish` 재실행이 같은 날 lead를 중복 추가(반복 방지 윈도 잠식)하지 않게 함.
- **`ThreadsPostLedger` (`threads_posted.json`):** 데일리 Threads 게시 멱등 마커. `{date: owner_run_id}` 맵으로 저장(레거시 bare-list도 읽음). 호출자(비주얼 Lambda)가 다중-분 게시 **전에** 날짜를 마크하고 실패 시 롤백하되, `unmark`은 **소유권 스코프**(자기 `run_id`=correlation id일 때만 해제) — 동시 invocation의 실패 롤백이 성공한 게시의 마커를 지워 다음 실행이 중복 게시하는 것을 막음. read-modify-write는 여전히 원자적이 아니지만(락 없음) '남의 성공을 내 실패가 지우는' 구멍은 닫힘.
- **`published_urls_from_snapshots`:** 과거 다이제스트 스냅샷의 `content.items[].url`을 뽑아, dedup이 원장뿐 아니라 AgentCore Memory 히스토리로도 self-heal(원장이 비어도 작동).
- **시드 & 기록 (`main.run_pipeline`):** exclude 집합을 **원장 AND 최근 AgentCore Memory 스냅샷**(`get_recent_digests(ttl, exclude_date=today, after_date=today-ttl)` — 같은 TTL 윈도로 날짜 한정) 양쪽에서 시드. 생성 후 발행된 `content.items` URL을 원장에 기록하고, lead를 `recent_leads.json`에 append하되 **AGI 카운트다운 프리픽스를 제거**(`_editorial_lead`)해 novelty 신호가 고정 보일러플레이트가 아닌 편집 각이 되게 함.

**`recall_trends` 도구.** AgentCore가 아니라 `trends.json`을 직접 쿼리(키워드 매칭 + momentum 정렬,
`TrendMemory.search`). 메모리 리소스(`AWS::BedrockAgentCore::Memory`)는 이제 이벤트 전용(단기,
`event_expiry_duration` 90일)이며 시맨틱 전략/`RetrieveMemoryRecords` 권한은 없음.

## 8. 헬스 체크 & 알림

**모델 (`shared/models.py`):**
- `SourceStatus` — `ok`/`empty`/`failed`/`stale`/`degraded`.
- `SourceHealth(name, item_count, status, detail)`.
- `HealthReport(sources)` — `has_failures`, `stale_sources`, `degraded_sources`, `empty_sources`, `summary()` 보유.
- `RankingHealth(batches_total, batches_failed, items_total, items_scored, items_lost)` — `degraded`(후보가 실제로 사라졌는지)와 `summary()`. `DigestResult.ranking_health`로 실려 나간다.
  **STALE·DEGRADED는 실패가 아니므로** `has_failures`를 켜지 않습니다(FAILED 승격 경로와 분리).

**소스 분류 (`run_collectors_with_health`):**
- 예외 → FAILED(잘린 detail 포함).
- park 파일이 `degraded`(stale = 나이 예산 초과 / error = 읽기 불가) → **STALE**(항목 수 + park detail 포함).
  항목은 나왔지만 그 sync가 멈춰 있다는 뜻이라 OK도 FAILED도 아닌 별도 상태. `_build_collector_tasks`가
  코루틴과 함께 **수집기 인스턴스**를 반환하고, 이 함수가 각 인스턴스의 `park_status`를 읽어 판정합니다.
- 수집기가 `degraded_detail`을 남겼으면 → **DEGRADED**(항목 수 + detail). 항목은 제때 나왔지만 그 소스의
  **입력 중 일부만** 응답한 경우(예: RSSHub 계정 피드 대부분 실패)로, 40개 계정이 3개로 줄어도 OK로 보였던
  구멍을 메웁니다. STALE 판정이 더 조치 가능하므로 park가 degraded면 STALE이 우선합니다. **보고/알림만
  바꾸며 집계기에 도달하는 항목은 그대로입니다.**
- 0 항목 → EMPTY(조용한 날엔 정상). 단 `collectors.alert_on_empty`가 이름을 지목한 소스는 어두워진 것이 **사건**이므로 로컬 실행에서도 ERROR 한 줄을 남기고 Lambda는 알림을 올린다.
- 그 외 → OK.

**알림 (`_maybe_alert`, 다이제스트 Lambda):** 소스가 FAILED **또는 STALE 또는 DEGRADED**일 때, 그리고 빈 항목 조기 반환
이전에 `ALERT_SNS_TOPIC_ARN`으로 게시(아무것도 수집 못 해도 장애는 알림되도록). 메시지는 실패/stale 소스
목록(실패/stale/degraded/empty)을 각각 분리해 담습니다 — `has_failures`만 보던 게이트에서는 죽은 로컬 cron이
며칠간 무음이었습니다. **EMPTY는 `collectors.alert_on_empty`가 지목한 소스만** 포함합니다(reddit·x 조용한 날이
매일 페이징하지 않도록 config 게이트).

**랭킹 헬스 알림 (`_maybe_alert_ranking`, 파이프라인 이후):** 위 수집기 알림은 **파이프라인 이전** 호출을 그대로
두고, 랭킹 판정은 **별도 게시**로 올립니다 — 파이프라인 예외가 수집기 알림을 삼킬 수 없게 하기 위함입니다.
재시도까지 실패한 배치가 있으면(≈후보 40건 소실) 겉보기 정상인 다이제스트에도 알림이 갑니다.

**핸들러 예외 전파.** 다이제스트·비주얼·Threads 갱신 핸들러는 실패를 로깅(correlation id 포함)한 뒤
**다시 raise**한다. 500 body를 반환하면 Lambda 입장에선 정상 종료라 Errors 알람도, 비동기 DLQ도 절대
울리지 않았다. 세 함수 모두 `retry_attempts=0`이라 재시도로 인한 이중 게시 위험은 없다.

**게시량 메트릭 & 날짜 전달 (다이제스트 → 비주얼 Lambda).**
- `DigestItemsPublished`(EMF)는 **큐레이션된 스토리 수**(`digest.content.items`)를 센다. 랭커 후보 수를
  세던 탓에 2026-08-13·08-17에 스토리 0건으로 게시된 날에도 만점처럼 보고되어 `EmptyDigestAlarm`이 울리지
  않았다. 타임스탬프는 UTC(위 참조).
- **짧은 다이제스트는 성공으로 로깅되지 않는다:** `_parse_content`에서 emit된 항목이 **드롭**되면 ERROR
  (그 뒤 다이제스트는 겉보기 정상이라 이 줄이 유일한 흔적이다). 반면 동일 사건 **병합**으로 target보다 적어진
  것은 정당하므로 `run_pipeline`이 WARNING으로만 남긴다(코드가 명시적으로 허용하는 경우).
- `_trigger_visual(digest_date)`는 날짜를 **명시적으로** 페이로드에 담아 비동기 invoke한다. 비주얼 Lambda는
  `_requested_date`로 그 값과 **"날짜가 명시됐는지" 플래그**를 읽고(DLQ 재생 시에는 봉투의 `requestPayload`
  아래 값도 인정), 해당 날짜의 스냅샷만 로드한다. 스냅샷이 없을 때 **날짜가 명시된 invoke면 raise**한다
  (방금 persist한 실행이 부른 것이므로 = 그날 무출력), 오늘로 폴백한 invoke(로컬·수동)는 조용히 종료한다.
- `_trigger_visual`은 best-effort가 **아니다**: AWS(`is_running_in_aws()`)에서 `VISUAL_FUNCTION_NAME`이
  비었거나 invoke가 실패하면 raise한다. 스냅샷은 이미 persist된 시점이라 잃는 것이 없고, 이것이 Errors 알람과
  DLQ 재생을 켜는 유일한 신호다. 로컬에서는 `main.py`가 비주얼을 인라인 실행하므로 env 미설정이 정상이고 조용히 no-op.
- **스냅샷 persist 실패 시 비주얼을 트리거하지 않고 시끄럽게 실패한다.** 비주얼 Lambda가 이 스냅샷으로
  게시하는 유일한 Threads 경로이므로, persist가 실패한 날 트리거하면 **다른 날짜**의 스토리를 게시하게
  된다. 트리거를 건너뛰고 예외를 다시 raise해 Errors 알람 + DLQ(재생 가능)로 남긴다 — 조용히 넘기면
  '어제 콘텐츠 게시' 대신 '완전 무출력인데 아무 신호 없음'이 된다.
- **Threads 부분 전달 알림 + 게시 결과 메트릭:** 비주얼 Lambda는 `ThreadsDelivery`(posted/expected)를 보고
  누락이 있으면 `ALERT_SNS_TOPIC_ARN`으로 SNS 알림을 올리고(env가 없으면 no-op이라 로컬 실행은 조용함),
  `ThreadsPostsPublished`와 `ThreadsImagePublished`(root가 그날 이미지를 실었는지 0/1)를 **하나의 EMF 레코드**로,
  결과가 없는 실행(=0건)에도 **무조건** 남긴다(EMF stdout 전용 — 아직 알람/CDK 연결 없음). 데이터포인트를 아예
  안 찍으면 CloudWatch에서는 0이 아니라 "데이터 없음"으로 읽히는데, 그 경우가 바로 측정 가치가 가장 큰 날이다.
  타임스탬프는 `datetime.now(UTC)` — naive 로컬 시계로 찍으면 UTC epoch ms로 해석돼 엉뚱한 시각에 기록된다
  (`digest_handler._emit_digest_items_metric`도 같은 버그였고 같은 방식으로 고쳤다). `_post_threads`는
  **콘텐츠가 있었는데 아무것도 게시되지 않은 경로**(스토리 0건, 게시 예외)에서도 `threads_outcome`을 남기므로
  (`expected>=1`) 알림/메트릭이 조용히 넘어가지 않는다. 반면 **이미 게시된 날의 스킵과 채널 비활성 스킵은
  실패가 아니므로 outcome을 남기지 않는다**(무음).
- **남은 실행 시간으로 게시 경로를 bound:** 비주얼 Lambda는 자기 `context.get_remaining_time_in_millis()`를
  **평범한 monotonic float 하나**로 바꿔(`_remaining_deadline`; context 객체는 파이프라인에 넘기지 않는다)
  `run(deadline=...)` → `_post_threads` → `post_to_threads`, 그리고 이미지 생성기로 흘린다. `deadline=None`
  (로컬 실행·`main.py`·research_cli)이면 동작은 이전과 **완전히 동일**하다. 딜라인이 있을 때만 인덱싱 예산이
  `min(270초, 남은 시간 − 게시 예비분)`으로 줄고(예비분 `THREADS_PUBLISH_RESERVE_SEC`은 기존
  `THREADS_MEDIA_PROCESS_WAIT_SEC` + reply 재시도 상수에서 산출), 남은 시간이 충분하면 **270초를 절대 깎지
  않는다**(인덱싱 인내심 부족이 애초에 스토리를 잃은 원인). 이미지 쪽에서는 moderation 재렌더(추가
  `visual_image_timeout_sec` 한 판)를 시간이 없으면 포기한다.

## 9. 딥 리서치 에이전트(AgentCore Runtime 위의 Strands)

Slack 멘션으로 트리거되는 **자율 딥 리서치** 에이전트. 자유형 토픽을 받아 열린 웹·학술 문헌·커뮤니티를 독립적으로
리서치한 뒤 한국어로 합성한 출처 표기 리포트를 채널에 전달한다. **다이제스트와 분리된 독립 웹 리서치**이며,
다이제스트 항목에 묶이지 않는다(예전 "후속 에이전트"는 제거됨).

**구성 (`agent/research_agent.py`의 `create_research_agent`).**
- `BedrockModel`(기본 Sonnet 5, `config.agent.model_id`)을 streaming + `CacheConfig(strategy="auto")`(§6 캐싱 참조)로 구성하고 7개 도구로 Strands `Agent`를 만든다. `max_tokens`는 `_LANGUAGE_MODEL_INFO`에서 모델 역량으로, 미등록 모델이면 `_DEFAULT_MAX_OUTPUT_TOKENS`(64000)로 폴백. cross-region inference-profile id는 `BedrockCrossRegionModelHelper`로 해석.
- AWS에선 env 리전, 로컬에선 `config.aws.bedrock_region`/`profile`로 boto 세션을 만들고 `boto_read_timeout`/`boto_connect_timeout`/`boto_max_attempts`를 적용.

**`SYSTEM_PROMPT_TEMPLATE`.** 자율 에이전트 철학을 따르되 리서치 리포트에 특화된 구획을 가진다:
- `<role>`: Slack 트리거 딥 리서치 에이전트, 다이제스트와 무관한 독립 리서치, 토픽/각을 메시지에서 추론(되묻지 않음).
- `<voice>`: 데일리 다이제스트와 **동일한 반복 내레이터** 페르소나(`config.pipeline.digest_voice_guidance` 주입) — 단 리포트 길이로 적응. 반복 금지·선형 전개·산문체·섹션 번호 매김 규율.
- `<tools>`/`<flow>`: 7개 도구와 권장(강제 아님) 흐름(이해/재작성 → 다중 소스 리서치 → 아웃라인 → 작성 → 이미지 첨부 → 전달). 검색 폭 `research_breadth`·깊이 `research_max_iterations`를 가이던스로 주입.
- `<delivery>`: 기본 채널 Slack, 사용자가 "쓰레드/스레드/threads"를 명시할 때만 Threads(추가 요청이면 둘 다, 대체 요청이면 Threads만). Slack(`research_slack_target_words` 분량)과 Threads(게시물당 ≤500자, `---` 구분, `research_max_threads_posts` 캡)는 **별개 아티팩트**로 각자 작성.
- `<language>`/`<formatting>`/`<citations>`: 한국어 규칙(공유 `KOREAN_STYLE_RULES` 주입), Slack mrkdwn vs Threads 평문, 출처 구분(검증/주장/추론)과 날조 금지.

**도구 (`agent/research_tools.py`) — 모두 `@tool` 비동기, 에이전트가 자유롭게 조합:**
- `web_search(query, recency)` — 열린 웹 검색. `recency="news"`면 Tavily `topic="news"`. 공유 `_tavily_search`(`shared/research/research_backends.py`) 위임. 결과 포맷(`_format_search_results`)은 Tavily가 일부 페이지에 주는 명시적 null title/url/content를 `or ''`로 흡수(`None[:n]`이 쿼리 전체를 실패시켜 에이전트를 배경지식 폴백=환각으로 밀지 않게).
- `community_search(query)` — Reddit/X/HN/Substack 반응·여론. `community_search_domains`를 `include_domains`로 `_tavily_search`에 전달.
- `search_papers(query)` — Semantic Scholar(`_search_papers`, 429 시 retry/backoff).
- `read_url(url)` — 특정 페이지 전문 fetch(`extract_url` → Tavily extract, `research_content_cap_chars`로 캡).
- `recall_trends(query)` — `shared/constants.py`의 `TRENDS_KEY`(`trends.json`)를 직접 쿼리(키워드 매칭 + momentum 정렬, active/cooling 트렌드, 상위 `recall_memory_top_k`). 시맨틱 recall이나 AgentCore 장기 메모리가 아님 — cross-day 트렌드 메모리의 "이전 동향" 각을 위함.
- `recall_digest(digest_date)` — **그 날짜의** 다이제스트 스냅샷(AgentCore Memory)에서 lead와 스토리 제목을 되읽음("X일에 뭘 다뤘나"). 단일 목적 도구이고 모드/파라미터가 없다. 출력은 bounded(`top_n`개 스토리, 줄당 `search_content_preview_chars`), 없는 날은 **다른 날짜로 폴백하지 않고** `No digest stored for <date>.` 문자열을 돌려준다(엉뚱한 날을 그날의 커버리지로 인용하는 것보다 recall 실패가 낫다). **읽기 실패는 '없는 날'과 다른 문장**으로 degrade한다 — 스로틀·거부·설정 오류를 "그날은 아무것도 안 다뤘다"로 보고하면 리포트가 실제로 다룬 주제를 안 다뤘다고 주장하게 된다. 잘못된 날짜 형식도 평문 한 줄로 degrade.
- `attach_image(source_url)` — 소스 페이지의 대표 이미지(og:image)를 받아 전달 컨텍스트에 stage(`fetch_og_image`). `research_max_staged_images` 캡 도달 시 거부.
- `deliver_report(report, channel)` — 완성 리포트를 채널("slack" 기본/"threads")에 게시. `output.delivery.deliver_research_report` 위임. 알 수 없는 채널이면 에이전트가 스스로 고치도록 오류 문자열 반환(조용한 강등 없음). 반환 문자열은 **실제 전달량**(`DeliveryStats`: rendered/delivered/dropped/trimmed)을 담는다 — 캡을 넘겨 드롭된 게시물, 500자 컷으로 잘린 게시물, 안 붙은 reply가 모두 "Delivered the report"로 보고돼 에이전트가 최종 답변에서 완전한 전달을 단정했다. 불완전하면 그렇게 말하고, **재전송은 하지 않는다**(`delivered_channels` 가드로 두 번째 호출은 no-op이므로 재전송 경로 자체를 만들지 않음).

`DeliveryContext`/`current_delivery_context`/`request_context`는 전달 계약을 소유한 `output/delivery.py`에 살고, 에이전트 엔트리포인트와 도구가 바인딩하도록 여기서 re-export된다.

### 9.1 채널 인지 전달 (`output/delivery.py`)
- **`DeliveryContext`(dataclass):** invoke별 전달 타깃 + staging. `channel_id`/`thread_ts`, `staged_images`(attach_image가 쌓은 OG 이미지), `delivered_channels`(성공 게시된 채널 — 채널별 폴백 판단용), `last_report`(deliver_report에 넘긴 마지막 리포트 — 런타임 폴백이 한 줄 확인 메시지가 아니라 실제 리포트를 재게시하도록), `dry_run`(로컬 CLI에서 stdout으로 단락). `request_context`는 contextvar로 동시 invoke가 글로벌을 공유하지 않게 바인딩하고, `current_delivery_context`는 바인딩이 없으면 새 인스턴스를 반환(warm 컨테이너에서 모듈 싱글톤이 staged_images/채널을 누적하지 않게).
- **`DeliveryStats`:** 마지막 전달 시도의 결과(`channel`, `rendered`, `delivered`, `dropped`, `trimmed`, `complete`). `DeliveryContext.last_stats`로 실려 `deliver_report`의 반환 문자열과 부분 전달 알림의 근거가 된다.
- **`deliver_research_report`:** 채널별 디스패치. **채널별 멱등** — `channel in delivered_channels`면 재게시 스킵(재시도/중복 도구 호출이 이중 게시하지 않게). 성공 시 `delivered_channels`에 기록(런타임의 마지막 폴백이 필요한지 판단하는 신호도 이 집합이다). 성공했지만 `last_stats.complete`가 아니면 `_notify_incomplete_delivery`가 요청자의 **같은 스레드**(`thread_ts`)에 한 줄 안내를 남긴다.
- **`_deliver_slack`:** staged OG 이미지를 먼저 각각 파일 업로드(소스 크레딧 캡션, `extension_for(content_type)`로 파일 확장자 결정) → 리포트에 `sanitize_slack_mrkdwn`을 적용해 모델이 흘린 마크업(## 헤딩/**bold**/`[text](url)`/이모지)을 코드로 보정(폴백 경로와 일치; `[text](url)`→`<url|text>` 변환은 **URL 내 균형 괄호를 보존**해 위키피디아/arXiv/DOI 인용이 첫 `)`에서 잘리지 않음) → `render_research_blocks(header=":satellite: OmniSummary Deep Research")`로 Block Kit 청크를 게시. 알림/프리뷰 텍스트는 `_strip_slack_mrkdwn`으로 평문화. best-effort(실패 시 False 반환).
- **`_deliver_threads`:** `render_threads_research(report, max_posts=research_max_threads_posts)`로 root + 평탄한 reply chain(각 ≤500자) + **드롭/트림 수**(`ThreadsResearchRender`). staged 이미지가 있으면 **첫 1장만** root에 태움(Threads 미디어 인덱싱이 느려 나머지는 Slack 전용) — PNG/원본 content_type 바이트를 S3 키(`{prefix}threads/research_<sha>.<ext>`)로 host하고 `post_to_threads`에 `image_content_type`을 함께 넘김. 상태 버킷이 없으면 텍스트 전용으로 게시.
- **`_dry_run_print`:** 실제 게시 대신 렌더 결과를 stdout으로(Threads는 root/reply, Slack은 sanitize 후 header+섹션 블록). Threads는 첫 이미지만 첨부됨을 명시.

### 9.2 진입 Lambda (`lambda_handlers/slack_event_handler.py`)
**스탠드얼론 zip 제약.** 이 핸들러는 **`lambda_handlers/`만 담긴 독립 zip**으로 패키징되므로 `shared`(또는 어떤 형제 패키지)도 import해선 안 된다 — zip에 없어 cold start에서 `ImportModuleError`로 깨진다. 그래서 의존성 없는 **stdlib `logging` 로거**를 자체적으로 둔다. `tests/test_slack_event_handler.py::test_handler_has_no_sibling_package_imports`가 이 규약을 가드한다.

ingress 흐름:
- **서명 검증:** Slack 서명을 HMAC-SHA256으로 타이밍 안전 비교(`x-slack-signature`/`x-slack-request-timestamp`, `SIGNATURE_EXPIRATION_SEC` 윈도). 비숫자 timestamp는 `float()` ValueError로 502가 되지 않게 try/except로 감싸 깨끗이 401 반환. `url_verification` 챌린지는 즉시 echo.
- **중복 제거:** `app_mention` 이벤트의 `event_id`(그리고 비동기 단계에선 `event_id:text` 해시)를 DynamoDB 조건부 쓰기(`attribute_not_exists` + TTL)로 멱등 처리. dedup 마커는 **디스패치 성공을 전제로** 다루며, self-invoke가 throw하면 `_release_event_marker`로 마커를 해제하고 500을 반환해 Slack 재시도가 깨끗한 상태로 들어오게 함(마커가 먼저 굳어 재시도가 영구 드롭되는 것 방지). dedup 스토어 자체 오류는 fail-open(진짜 이벤트를 막지 않음).
- **즉시 ack:** 비동기 단계에서 AgentCore를 호출하기 전에 `_post_ack`가 원 스레드에 "딥 리서치를 시작합니다" + 모래시계 힌트를 게시(딥 리서치가 수 분 걸려 스레드가 침묵하지 않게). best-effort.
- **비동기 self-invoke:** 첫 호출은 200을 즉시 반환하고 `action=invoke_agentcore`로 자기 Lambda를 `InvocationType="Event"`로 재호출. 비동기 단계에서 멘션(`<@...>`)을 스트립하고 `invoke_agent_runtime`(`AGENTCORE_RUNTIME_ARN`, `qualifier="DEFAULT"`)으로 `prompt`/`channel_id`/`thread_ts` 페이로드를 전달.
- **폴백:** 런타임 invoke 자체가 throw(스로틀/cold-start 타임아웃)하면 외부 Slack 요청은 이미 200을 받았으므로, `_post_fallback`이 원 스레드에 가시적 오류 메시지를 게시한다.

### 9.3 런타임 (`agent_runtime/app.py`, `BedrockAgentCoreApp`)
`@app.entrypoint invoke(payload)` 순서:
- payload의 `correlation_id`로 correlation id 시드.
- `DeliveryContext(channel_id, thread_ts)`를 만들고 `create_research_agent()`로 에이전트 생성.
- `request_context(delivery)`로 contextvar 스코프(동시 invoke가 한 요청의 채널을 다른 요청으로 누출하지 않게) 안에서 에이전트 실행. 응답은 `sanitize_slack_mrkdwn`. 예외 시 `_emit_agent_error_metric`(EMF `OmniSummary/AgentErrors`)을 찍고 **raw 예외 문자열이 아닌 일반 안내 메시지**로 응답(모델 ID·ARN·백엔드 오류 바디가 Slack에 새지 않게).
- **Slack 폴백:** 에이전트가 **어떤 채널에도 전달하지 못했을 때만**(`channel_id and not delivery.delivered_channels` — `deliver_report`를 끝내 호출 안 했거나 모든 전달이 실패) `_send_slack_message`로 게시해 사용자가 최소한 무언가는 받게 한다. Slack이 타깃이 아니었다는 이유만으로는 폴백하지 않는다 — Threads 전용 요청이 Threads에 성공했으면 (Threads 포맷) 리포트를 Slack에 중복 투척하면 안 되기 때문. 이때 한 줄 확인 메시지가 아니라 `delivery.last_report`(실제 리포트)를 우선 사용하고, `sanitize_slack_mrkdwn`으로 게시(`_send_slack_message`가 `render_agent_blocks` 폴백 래퍼 사용). 이 마지막 폴백 게시도 try/except로 감싸 여기서의 raise가 invocation을 하드 에러로 만들지 않게 하되, 그 경우 어떤 채널에도 아무것도 도달하지 못했으므로 `_emit_agent_error_metric()`을 찍어 알람이 울게 한다(엔트리포인트는 여전히 텍스트를 반환하므로 그 외엔 무증상).

### 9.4 OG 이미지 첨부 (`shared/media/og_image.py`)
- **`fetch_og_image(url)`:** 페이지를 브라우저 UA로 fetch해 og:image/twitter:image 메타(`og:image`→`og:image:url`→`twitter:image`→`twitter:image:src` 우선순위)를 파싱하고 상대 URL은 페이지 URL로 절대화. 이미지는 **스트리밍**으로 받아 oversize 바디를 다 버퍼링하지 않고 중간에 중단(Content-Length 선검사 + 스트림 누적 검사, `og_image_max_bytes`/`og_image_timeout_sec`). 렌더 가능한 래스터 타입(jpeg/png/webp/gif)만 통과 — SVG 등 벡터/이색 타입은 Slack 프리뷰/Threads fetcher가 못 다뤄 제외. 어떤 오류·미존재·비이미지·oversize도 `None` 반환(절대 raise 안 함). 반환 `ImageAsset(data, source_url, image_url, content_type, alt)`.
- **`extension_for(content_type)`:** 이미지 MIME → 파일 확장자(기본 `png`). Slack 파일명과 Threads S3 키에 쓰여 content_type이 다운스트림까지 일관되게 전달됨.

### 9.5 공유 한국어 스타일 (`shared/config.py` `KOREAN_STYLE_RULES`)
모든 한국어 출력 표면(데일리 다이제스트 + 딥 리서치, Slack + Threads)이 공유하는 산문 규약 상수 — 번역투 회피, `~다` 평서체(존댓말 금지), 콜론-나열 금지. 다이제스트의 `digest_language_rules`와 리서치 에이전트의 `<language>` 블록 양쪽에 합성되어, 두 기능이 register/어조에서 갈라지지 않게(같은 작성자, 같은 규칙) 한다.

## 10. 시각화 생성기(자유형 시놉시스 → 이미지)

`agent/visuals.py`의 `VisualGenerator`는 모드 없는 자유형 이미지 생성기로, 이제 **데일리 비주얼 파이프라인**
(`pipeline/daily_visual.py` `DailyVisualMaker`, §5.1)이 구동한다(예전의 에이전트 측 이미지 생성 도구는 제거됨).

**설계.**
- 고정된 comic/diagram 모드나 컷 수 파라미터가 없음.
- 자연어 `instruction`으로 원하는 형식(1페이지 프리젠테이션 슬라이드, N컷 만화, 개념 다이어그램, 인포그래픽, 포스터 등)을 묘사 — 데일리 비주얼에선 에디터(`VisualEditorPrompt`)가 헤드라인 스토리를 어떻게 그릴지 브리핑.
- source(다이제스트 헤드라인 항목)와 수집한 `context`(논문/기사 리서치)를 넘김.

**생성 흐름 (`VisualGenerator.generate(instruction, source, context)`):**
- **브리프:** `VisualSynopsisPrompt`로 Claude(Bedrock)가 단일 이미지 브리프 생성(JSON: title·caption·prompt).
- **파싱:** Bedrock 구조화 출력(`with_structured_output(VisualBrief)`)으로 검증된 객체를 받음 — 손으로 JSON을 파싱하지 않는다(브리프의 `prompt`가 최대 4000자 자유 문구라 escape 안 된 인용부호/개행에 파서가 깨졌음).
- **필드 유출 방어(`VisualBrief` 검증기):** 구조화 출력이 다음 필드 값을 앞 문자열에 흘리는 슬립이 반복됐다 — 2026-08-17엔 태그 형태(`</caption>\n<parameter name="orientation">landscape`)가 Threads에 그대로 게시되고, 08-18 로컬 실행에선 태그 없는 형태(캡션이 맨 끝에 `\nportrait`)가 나왔다. `title`/`caption`은 (1) 태그 유사 마크업을 제거하고(`<2%` 같은 산문은 보존), (2) **마지막 줄 전체**가 orientation 필드의 **허용값 중 하나**면 그 줄을 떨어뜨린다. 후보값은 하드코딩 단어 목록이 아니라 `typing.get_args`로 Literal에서 파생하며, 비교 대상을 파싱된 `orientation` 하나로 두면 08-17처럼 값이 어긋난 유출(캡션은 `landscape`, 필드는 기본값 `portrait`)을 놓친다. 산문 중간에 그 단어가 들어간 경우는 건드리지 않고, 실제로 값을 떨어뜨린 경우에만 WARNING을 남긴다.
- **이미지:** 브리프의 `prompt`로 OpenAI `gpt-image` 호출(`b64_json`) → PNG 바이트. 블로킹 호출(30-120초)이라 `asyncio.to_thread`로 이벤트 루프에서 분리(동시 Slack/Threads I/O가 렌더 동안 멈추지 않게). orientation(square/landscape/portrait)은 브리프가 시각에 맞게 고르고, `image_sizes` 딕셔너리로 gpt-image size에 매핑. 모더레이션 차단(intermittent) 시 완화된 브리프로 1회 재생성.
- **게시:** `DailyVisualMaker`가 `output.slack_handler.send_image_to_slack`(`files_upload_v2`)로 Slack에 업로드(+`enable_threads_post` 시 Threads에도).

**기타.**
- **기본값 중복 없음:** 열 개 넘는 비주얼 놉을 모두 **필수 키워드 인자**로 받는다. 예전엔 같은 기본값을 여기와 `PipelineConfig`에 두 벌 뒀다가 드리프트했고(`style_aesthetic`이 "clean modern style"로 썩음), 일부 인자만 넘기는 호출자가 그 낡은 사본을 조용히 받았다. 이제 `PipelineConfig`가 단일 원천이다.
- **호출 상한:** OpenAI 클라이언트를 `visual_image_timeout_sec`/`visual_image_max_retries`로 생성(SDK 기본 600s×2회는 15분 Lambda를 넘길 수 있음).
- **비용 가시성 & 결정성:** `quality`는 `visual_image_quality`가 설정될 때만 보낸다. 비우면 OpenAI의 `auto`가
  티어를 고르는데 **장당 단가가 약 4배 차이**(우리 사이즈 기준 medium $0.041–0.053 vs high $0.165–0.211)라
  월 청구가 하루 1장에 ~$1.3~5.2 사이로 불확정이고 코드가 어느 티어를 산 건지 말할 수 없다. 값을 고정하면
  결정적이 된다. 어느 쪽이든 렌더가 응답의 **실제 과금 토큰 수**를 로그에 남긴다(`_usage_summary`,
  usage가 없거나 필드명이 바뀌면 `"unknown"`으로 degrade — SDK 변경이 그날 이미지를 날려선 안 된다).
- OpenAI 키(`resolve_secret`로 env→SSM 해석)가 없으면 우아하게 비활성화.
- 새 출력 형식은 코드 변경 없이 instruction 문구만 바꾸면 됨.

## 11. 인프라(CDK)

### `foundation_stack`

- **리소스:** VPC, ECR 리포, DynamoDB 중복 제거 테이블(SSE + prod에서 PITR), S3 상태 버킷(CDK 생성 시
  S3-managed 암호화, 버저닝, 퍼블릭 차단, SSL 강제), ECS Fargate RSSHub 서비스 + service-discovery,
  CodeBuild 이미지 빌드, SNS 알림 토픽(+ 선택적 이메일 구독), AgentCore Memory 리소스 + 실행 역할, IAM 역할들.
- **RSSHub 서비스는 `aws.rsshub_desired_count`(기본 0)로 스케일된다.** 다이제스트는 이 서비스에 **도달하지**
  않는다 — `RSSHubCollector`는 S3 park 파일을 **먼저** 읽고 쓸 수 있으면 도달성 확인조차 하기 전에 리턴하며,
  로컬 sync cron이 매 실행 직전에 그 파일을 갱신한다(2026-08-17 파일은 19:00 실행을 위해 18:30에 올라왔고,
  다이제스트 로그의 X 항목은 그 파일에서 나왔다). 계정의 유일한 상시 Fargate 태스크로 월 약 $40이었다.
  **태스크 정의는 그대로 배포되므로** 1로 올리면 로컬 sync가 멈춘 날의 AWS 폴백이 복구된다(그 상태는 이미
  헬스 STALE로 표면화된다).
- **IAM(최소 권한):**
  - `/{project}/{stage}/*`로 스코프된 `ssm:GetParameter*`.
  - foundation-model / inference-profile / **application-inference-profile** ARN으로 스코프된
    `bedrock:InvokeModel*`. 마지막 것은 별도 리소스 타입이며 **필수**다 — 모델 리졸버가 비용 귀속용
    application inference profile을 선호하므로, 빠뜨리면 프로필이 존재하는 순간 모든 Bedrock 호출이
    AccessDenied가 된다(= 그날 다이제스트 전체).
  - `lambda:InvokeFunction`은 **데일리 비주얼 함수 하나**(`{project}-{stage}-visual`)로 스코프 — 파이프라인
    역할의 유일한 cross-function 호출이 그 비동기 fan-out이다. 예전의 `{project}-{stage}-*`는 공개
    API Gateway 뒤의 Slack 핸들러와 토큰 갱신 Lambda까지 포함했다. ARN은 application_stack이 붙이는
    **리터럴 이름**으로 만든다(함수 객체를 참조하면 `fnd → app` 순환 의존이 됨).
  - S3 객체 접근은 프로젝트 **루트 prefix**(`config.aws.s3_prefix` + `/*`)로 스코프 — 상태 버킷은 기존 공유
    버킷일 수 있고, 이 prefix가 프로젝트가 만지는 모든 키(state_store의 `digest_state`, 수집기 park 파일,
    데일리 비주얼의 `threads/*.png`)를 덮는다. 버킷 레벨 List(CDK가 붙임)는 그대로 둔다.
  - `bedrock-agentcore:InvokeAgentRuntime`/Memory 데이터플레인 액션.
  - 프로젝트 로그 그룹 ARN으로 스코프된 CloudWatch Logs.
  - 계정 전역 관리형 정책 없음.

### `application_stack`

- **리소스:** 다이제스트 Lambda(DockerImage), 데일리 비주얼 Lambda(DockerImage, 비동기), Slack 이벤트 Lambda,
  Threads 토큰 갱신 Lambda(DockerImage), API Gateway(+ 스테이지 스로틀링),
  스테이지에 연결된 WAFv2 WebACL(rate-limit + AWS 관리형 규칙셋: Common, KnownBadInputs, IpReputation),
  EventBridge 일일 다이제스트 크론(설정 기반 시/분) + EventBridge Threads 토큰 갱신 스케줄(`threads_token_refresh_days`,
  기본 ~50일 주기 — 60일 만료 안쪽에서 토큰을 갱신해 SSM에 재기록), AgentCore Runtime(설정 가능한
  `agentcore_image_ref`로 이미지 바인딩), 시크릿용 SSM 파라미터, SNS로 향하는 CloudWatch 알람(§12 참조).
- **재시도/DLQ:** 다이제스트·비주얼·Slack 이벤트 Lambda는 `retry_attempts=0` + SQS DLQ(`foundation.async_dlq`로
  `on_failure`)로, Threads 갱신 Lambda도 `retry_attempts=0`(명시값이 없으면 비동기 재시도 2회가 기본이라 갱신
  엔드포인트를 재호출)으로 구성. 파이프라인이 멱등이 아니고 **Threads는 idempotency key가 없어 재시도가 이중
  게시**를 일으키므로, 자동 재시도 대신 실패 건을 DLQ에 남겨 점검/수동 리플레이(핸들러는 §8처럼 raise해
  Errors 알람이 뜨게 한다). 모든 Lambda는 `log_retention=ONE_MONTH`.
- **RSSHub 보안그룹 ingress:** 다이제스트 Lambda SG → RSSHub Fargate 서비스 SG(`RSSHUB_PORT`) 인그레스를
  **이 스택에서** `CfnSecurityGroupIngress`로 추가. 규칙이 없어 AWS에서 X 피드 fetch가 전부 타임아웃했다(실제
  X 항목은 S3 park 파일이 공급). `connections.allow_from()`은 규칙을 foundation 쪽에 붙여 `fnd → app` 순환
  참조가 되므로 명시적 인그레스 리소스를 쓴다. Lambda SG는 기본 전체 egress라 ingress만 빠진 반쪽이었다.
- **시크릿 처리:** 스택은 파라미터 **경로만** 만들고 값은 `SSM_PLACEHOLDER`로 둔다(CloudFormation은 SecureString
  생성 불가 → 값을 스택에 넘기면 템플릿·CDK staging 버킷·`GetTemplate` 응답에 평문으로 박힌다). 실제 값은
  배포 후 `scripts/put_secrets.py`가 **SecureString**으로 out-of-band 기록한다(`--dry-run` 미리보기,
  `--verify` 읽기 전용 점검, `--force`로만 기존 SecureString 덮어씀). 재배포는 값을 건드리지 않는다(placeholder가
  안 바뀌므로 CloudFormation이 리소스를 업데이트하지 않음) — 그래서 **이 리소스의 템플릿 속성은 무엇도 추가·변경하면
  안 된다**(Description 하나만 붙여도 CloudFormation이 PutParameter를 다시 실행해 살아 있는 시크릿 위에 placeholder를
  쓴다). `tests/test_infrastructure.py`가 렌더된 속성 집합을 고정해 그 편집을 배포 전에 잡는다.
  `String` → `SecureString` 타입 변경은 SSM이 `ValidationException`으로 거절하므로, 스크립트는 **CloudFormation이
  남긴 정확한 상태**(`Type == String` **그리고** `Value == SSM_PLACEHOLDER`)일 때만 파라미터를 **삭제하고 SecureString으로
  재생성**한다 — 이것이 "시크릿은 SecureString"이라는 주장을 실제로 참으로 만든다. 그 밖의 `String`은 **실제 값을 담고
  있으므로 절대 삭제하지 않고**(살아 있는 자격증명을 잃는 건 암호화되지 않은 것보다 나쁘다) 값만 제자리에 쓰고 시끄럽게
  알린다. 삭제는 됐는데 재기록이 실패한 경우는 **가장 나쁜 상태**(파라미터 자체가 없음)이므로 `FAILED` 목록 + non-zero
  종료 + **복구 명령 전문**(`aws ssm put-parameter --name … --type SecureString …`)을 출력한다. 한 파라미터의 실패로
  루프가 죽지 않는다 — 예전엔 첫 거절에서 루프가 죽어 그 뒤 시크릿이 전부 placeholder(=런타임에선 미설정)로 남았다.
  `--verify`는 이제 값이 있어도 `String`이면 **PLAINTEXT**로 따로 보고하고 non-zero로 끝낸다(설정됨 ≠ 암호화됨).
  보완 통제는 스코프된 IAM 읽기 정책. Threads 갱신 Lambda는 갱신된 토큰을
  `put_parameter(Overwrite=True)`로 **`Type`을 지정하지 않고** 덮어쓴다 — `Type=SecureString`을 얹는 것은 타입 변경이라
  파라미터가 아직 `String`이면 SSM이 `ValidationException`으로 거절하고, 그러면 토큰이 갱신되지 않은 채 60일 뒤 Threads
  전달이 끊긴다. `Type`을 생략하면 기존 타입(마이그레이션 후에는 SecureString)을 유지하고 값만 갱신한다. 다만 그 생략은
  암호화되지 않은 파라미터도 **조용히 보존**하므로, 기록 후 타입을 확인해 `SecureString`이 아니면 ERROR로 남긴다
  (best-effort — 확인 실패가 성공한 갱신을 에러로 바꾸지는 않는다).

## 12. 관측성(Observability)

**로깅 (`shared/logger.py`).**
- **포맷:** AWS에서는 구조화 JSON 로그(`is_running_in_aws()`), 로컬에서는 사람이 읽는 형식.
- **correlation id:** `ContextVar` 기반(`set_correlation_id`/`get_correlation_id`)이 모든 레코드에 주입되고
  Lambda 요청 id / AgentCore 페이로드에서 시드됨.

**알람.** CloudWatch 알람 12개가 모두 SNS 알림 토픽(→ 이메일)으로 라우팅됨(`_add_alarms`):
- **Lambda별 Errors ×4 + Timeout ×4** — digest / slack-events / visual / threads-refresh 각각에 대해 예외(Errors)와
  타임아웃 임박(max Duration ≥ 설정 타임아웃의 90%; 타임아웃은 Errors로 집계되지 않으므로 별도) 알람.
- **API 5xx** — API Gateway server-error.
- **EmptyDigestAlarm** — EMF `OmniSummary/DigestItemsPublished`(실행당 1회) **24h 윈도**(CloudWatch가 `evaluation_periods × period ≤ 86400s`로 제한하므로 그 이상은 배포 시 거부), 0건 게시 또는 그날 미실행을 모두 포착(missing-data=BREACHING).
- **AsyncDLQAlarm** — async DLQ에 메시지가 쌓이면(실패한 digest/visual 실행 대기) 알림.
- **AgentErrorsAlarm** — EMF `OmniSummary/AgentErrors`(AgentCore 런타임이 자체 예외를 잡아 에러 메시지로 응답하므로, 체계적 장애가 EMF 메트릭으로만 보임).

## 13. 테스트 & CI/CD

**테스트 (`tests/`, pytest, `asyncio_mode=auto`).** 1000+ 테스트, 커버리지 게이트 80%(측정 ~90%). `tests/conftest.py`의 autouse 픽스처가 앰비언트 시크릿/인프라 env를 monkeypatch로 비우고 SSM 클라이언트를 막아 **hermetic**하게 만든다(개발자 `.env`/AWS 프로파일에 결과가 좌우되지 않고, 실 SSM 왕복으로 낭비하던 수십 초도 사라짐). 커버 영역:
- 수집기(모킹한 HTTP/feedparser).
- Slack 이벤트 핸들러(서명 검증/중복 제거 + **형제 패키지 import 금지 가드** `test_handler_has_no_sibling_package_imports`).
- 집계기, 랭커 파싱 + 슬롯/origin-cap 로직 + **배치 재시도/전면 실패 승격/fan-out 상한**.
- **`main.run_pipeline` 오케스트레이션(`test_run_pipeline.py`)**: 집계 후 빈 입력·임계 미달 조기 반환, 원장/leads 기록과 트렌드 갱신, 원장·AgentCore 스냅샷 양쪽에서 시드하는 cross-day dedup(URL 정규화 포함), dry-run이 상태를 쓰지 않고 아무 채널에도 보내지 않음, 로컬 인라인 비주얼 실행과 그 실패의 non-fatal성. LLM/네트워크 협력자만 스텁하고 원장·롤링 로그·집계는 임시 디렉터리 StateStore로 실제 실행.
- **`StateStore`(`test_state_store.py`)**: `S3StateStore` 키 prefix·UTF-8 인코딩·NoSuchKey vs 기타 ClientError, `create_state_store`의 버킷 기반 선택(AWS 밖에서도 `STATE_BUCKET`이면 S3 + 프로파일 세션).
- **WebSearch `collect()`(`test_web_search.py`)**: 쿼리별 fan-out·URL dedup·per-trend domains/topic 전달, 전면 실패 승격, 부분 실패 허용, LLM 정제 2단계와 그 실패의 non-fatal성.
- **Slack 이벤트 중복 제거(`test_slack_event_handler.py`)**: 조건부 PutItem + TTL 마커, 중복 판정, dedup 스토어 장애 시 fail-open, 마커 릴리스, Slack 재전송이 실제로 1회만 dispatch되는 end-to-end.
- 헬스 리포트, logger.
- 메모리 스토어(로컬 + AgentCore 모킹).
- 다이제스트 핸들러 알림.
- 딥 리서치 에이전트(`test_research_agent.py` 구성·프롬프트 보간), 7개 도구(`test_research_tools.py`), 리서치 백엔드(`test_research_backends.py`), 채널별 전달(`test_delivery.py`), 렌더러(`test_renderers.py` — research/threads 블록), OG 이미지(`test_og_image.py`), 리서치 CLI(`test_research_cli.py`), `VisualGenerator`(`test_visuals.py`).
- AgentCore 엔트리포인트(`agent_runtime/app.py` — 에이전트 생성·Slack 토큰 env/SSM 해석·invoke 해피패스/예외 처리·correlation ID·Slack 폴백).
- trend_tracker(trim/evidence-cap/archived-merge).
- CDK assertion(`aws-cdk.assertions`로 두 스택 검증).

**CI (`.github/workflows/ci.yml`).**
- **락파일 고정:** `uv lock --check` + 모든 잡의 `uv sync --frozen`으로, 리뷰/테스트된 것과 다른 버전이 조용히 해석되지 않게 함.
- lint(ruff), 포맷 체크(black `--check`), **`mypy .`**(경로 열거 대신 레포 전체 — 제외는 `[tool.mypy] exclude`. 예전 열거식은 새 최상위 모듈·`scripts/`를 조용히 게이트 밖에 뒀음).
- 테스트 + 커버리지 게이트: 범위와 `fail_under`가 `pyproject.toml`(`[tool.coverage.*]`)에 있어 커맨드라인 수정으로 좁혀지지 않음.
- **잡 상한 & 캐시:** 모든 잡에 `timeout-minutes`(기본 6시간 러너 타임아웃으로 멈춘 빌드가 방치되지 않게), uv 휠 캐시는 `uv.lock` 키(의존성이 바뀌면 재설치되므로 깨진 의존성 집합을 캐시가 가릴 수 없음), Node는 npm 캐시.
- **레포 고정 CDK CLI로** 오프라인 `cdk synth`(Node 22 + `npm ci`로 `package.json`에 핀된 `aws-cdk` 설치 — 예전 `npm ci || npm install` 폴백은 lock 부재/불일치를 삼키고 다른 CLI를 깔아 이 잡의 의미를 없앴음 → `npx cdk synth -a "uv run python scripts/ci_synth.py"`). 인프로세스 `app.synth()`가 아닌 실제 CLI를 태워 **CLI↔`aws-cdk-lib` cloud-assembly 스키마 핸드셰이크**를 검증(글로벌 CLI가 라이브러리보다 뒤처져 배포가 스키마 미스매치로 깨지던 클래스를 PR에서 잡음). `ci_synth`는 `vpc_id`를 비우고 env-agnostic 계정을 써 자격증명 없이 완전 오프라인.
- Docker 빌드 + **이미지 import 체크**: 두 이미지를 단일 플랫폼 `load: true`(네이티브 amd64)로 빌드해 로컬 데몬에 올린 뒤 `docker run --rm --network none --entrypoint python`으로 실제 엔트리 모듈을 import한다(digest: `lambda_handlers.*` + `main`, agentcore: `agent_runtime.app`). 빌드만으로는 import가 한 번도 실행되지 않아 COPY 누락이나 개발자 머신에서만 해석되는 모듈이 그대로 통과했다. 자격증명 없이 `--network none`이므로 **import 시점 AWS 호출/HTTP fetch가 콜드스타트가 아니라 CI에서** 깨진다. import 체크는 빌드가 얼마나 캐시됐든 로드된 이미지에 대해 항상 실행되므로 레이어 캐시가 실패를 건너뛸 수 없다. 캐시는 `type=gha`(이미지별 scope).
  - agentcore는 배포는 arm64지만 CI는 amd64로 빌드한다(베이스·의존성 모두 멀티아치이고, QEMU 에뮬레이션 없이 import를 실행하려면 네이티브여야 한다 — QEMU 하 `pip install`은 잡 예산을 넘긴다). 이 잡이 잡는 것(COPY 누락·미해결 의존성)은 아키텍처 무관.
- **의존성·시크릿 스캔 (`security` 잡):** `uv export --frozen`으로 **`uv.lock`이 핀한 정확한 집합**(= 이미지가
  설치하는 그 버전들)을 뽑아 `pip-audit --strict`로 감사한다 — pyproject 범위를 재해석해 감사하면 배포되지 않는
  버전을 검사하게 된다. 그리고 `gitleaks`를 **전체 히스토리**에 돌린다(shallow clone은 tip만 보므로 과거에
  커밋되고 나중에 지워진 키를 절대 못 찾는다). CFN 템플릿이 오늘까지 실제로 평문 토큰을 담고 있었고,
  `config/config.yaml`은 gitignore인데 실제 값을 갖고 있으므로 잘못 add된 파일 하나가 곧 유출이다.
- **이미지 하드닝:** 두 Dockerfile 모두 `uv.lock`이 핀한 집합을 설치한다(`uv export` → `uv pip install --system`,
  프로젝트 자신은 `--no-deps`). 예전 `pip install .`은 빌드 시점에 pyproject 범위를 다시 해석했으므로 **CI가
  테스트한 적 없는 의존성 집합이 Lambda에서 돌 수 있었다**. 의존성 레이어를 소스 COPY보다 먼저 두어 코드 변경이
  레이어를 재사용한다. 런타임은 **non-root**(uid 10001) — 이 전환이 실제 취약점을 하나 드러냈다: `shared/logger.py`가
  모듈 스코프에서 로그 디렉터리를 무가드 `mkdir`해 쓰기 권한이 없으면 **import 자체가 PermissionError**로 죽었다
  (프로덕션은 `is_running_in_aws()` 가드로 무사했지만 CI 임포트 검사·읽기 전용 체크아웃·샌드박스는 모두 해당).
  이제 mkdir과 FileHandler 열기 모두 콘솔 전용으로 degrade한다. `.dockerignore`도 추가 — 빌드 컨텍스트가
  `.env`·`.venv`·`logs/`·`cdk.out`을 데몬으로 보내고 있었다.
- **pre-commit (`.pre-commit-config.yaml`, `uv run pre-commit install`):** CI와 같은 게이트(ruff, black, YAML/JSON,
  private-key 탐지, `uv lock --check`)를 푸시 전에 돌린다. mypy는 의도적으로 훅에서 제외 — 전체 의존성 해석이
  필요하고 느려서 사람들이 `--no-verify`를 쓰기 시작하면 얻는 것보다 잃는 게 크다.
- **CI는 추적되는 config로 synth:** `config/config.yaml`이 gitignore이므로 `scripts/ci_synth.py`는 `config/config-template.yaml`을 로드한다(`Config.load()`는 CI에서 코드 기본값으로 조용히 떨어져 아무도 배포하지 않는 스택을 synth했다). 인프라 assertion 테스트도 같은 템플릿을 쓴다.

## 14. 주요 명령어

```bash
uv run python main.py --dry-run --sources rss reddit   # 부분 dry run
uv run python main.py                                   # 전체 파이프라인 + 전달(현재 설정: Threads)
uv run python -m pytest tests/ -v                       # 테스트
uv run black --check . && uv run ruff check .           # lint/format
uv run mypy .                                           # 레포 전체 타입 체크
uv run python scripts/ci_synth.py                       # 오프라인 CDK synth(인프로세스)
# 배포: 두 이미지(digest amd64 + agentcore arm64)를 먼저 빌드/푸시하고, 푸시된 sha256 digest를
# DIGEST_IMAGE_REF로 넘겨 배포(태그 문자열이 안 바뀌면 CFN이 Lambda를 재배포 안 함). CDK CLI는
# npm install 후 npx로 — package.json에 aws-cdk-lib와 호환되게 핀돼 있어 글로벌 cdk의 스키마 미스매치를 피함.
npm install                                             # 1회 — 핀된 CDK CLI 설치
export DIGEST_IMAGE_REF=sha256:<pushed>                 # AGENTCORE_IMAGE_REF 기본 :arm64
AWS_PROFILE=${AWS_PROFILE:-research} npx cdk deploy --all -a "uv run python scripts/deploy.py"
# 시크릿은 템플릿에 없다 — 배포 직후 실제 값을 SecureString으로 기록(§3.5).
AWS_PROFILE=${AWS_PROFILE:-research} uv run python scripts/put_secrets.py            # --dry-run / --verify / --force
# 온디맨드 Bedrock은 과금 대상 리소스가 없어 비용 할당 태그가 안 붙는다. 계정/스테이지당 1회:
AWS_PROFILE=${AWS_PROFILE:-research} uv run python scripts/put_inference_profiles.py  # --dry-run / --delete
uv run pre-commit install                               # 1회 — CI 게이트를 푸시 전에
```
