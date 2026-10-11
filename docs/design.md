# OmniSummary — 설계 문서

> OmniSummary의 설계와 구현을 줄 단위까지 설명하는 단일 레퍼런스다. 상위 수준 개요는 `README.md`에 있고,
> "왜 이런 모양인가"에 대한 답은 여기에 있다.
>
> 읽는 방식에 대한 안내가 하나 필요하다. 단순한 코드가 기본값인 곳에서 복잡한 쪽을 골랐다면 그 이유를 함께
> 적었고, 근거로 쓴 수치는 실제 출력에서 재본 값이다. 배경 없이 코드만 보면 과한 장치로 읽히고, 그러면 다음
> 사람이 선의로 되돌려놓는다. 반대로 변경 이력은 여기 없다. 그건 git이 갖고 있다.

## 목차

| 절 | 내용 |
|----|------|
| [1. 개요](#1-개요) | 두 개의 실행 경로와 전체 흐름 |
| [2. 저장소 구조](#2-저장소-구조) | 디렉터리별 책임 |
| [3. 설정](#3-설정) | `config.yaml` 전체 필드와 시크릿 |
| [4. 수집기](#4-수집기) | 다섯 소스, S3 park 파일, 실패 신호 |
| [5. 파이프라인](#5-파이프라인) | 집계에서 랭킹, 다이제스트, 렌더링, 비주얼, Threads까지 |
| [6. LLM 팩토리](#6-llm-팩토리-sharedutilspy) | 모델 구성, 비용 귀속, 토큰 카운트, 프롬프트 캐싱 |
| [7. 메모리](#7-메모리-세-개의-분리된-저장소) | 트렌드, 다이제스트 스냅샷, cross-day dedup |
| [8. 헬스 체크와 알림](#8-헬스-체크와-알림) | 소스 분류, SNS 알림, 메트릭 |
| [9. 딥 리서치 에이전트](#9-딥-리서치-에이전트) | Strands 에이전트, 도구 8개, 로컬 CLI와 OpenClaw 스킬 |
| [10. 시각화 생성기](#10-시각화-생성기) | 자유형 시놉시스에서 이미지까지 |
| [11. 인프라](#11-인프라cdk) | CDK 두 스택 |
| [12. 관측성](#12-관측성) | 로깅과 알람 |
| [13. 테스트와 CI/CD](#13-테스트와-cicd) | 게이트, 그리고 그 게이트가 잡는 것 |
| [14. 주요 명령어](#14-주요-명령어) | 로컬과 배포 커맨드 |

## 1. 개요

OmniSummary는 능동형(proactive) AI/ML 일일 다이제스트 시스템이다. 사용자가 물어봐야 답하는 도구가 아니라
매일 정해진 시각에 스스로 읽을 거리를 만들어 보내는 쪽이다. 그래서 설계의 상당 부분이 아무도 보고 있지 않을
때 조용히 실패하지 않는 일에 쓰여 있다.

시스템에는 서로 독립적인 두 개의 실행 경로가 있다.

**첫째는 스케줄 다이제스트다.** EventBridge 크론이 다이제스트 Lambda를 깨우면 다섯 계열의 소스에서 콘텐츠를
수집해 중복을 제거하고, LLM으로 후보 풀을 고른 다음 한국어 에디토리얼 다이제스트를 구조화된 `DigestContent`로
생성한다. 상태는 Bedrock AgentCore Memory에 저장하고, 소스별 헬스는 SNS 이메일로 보고한다. 게시는 **데일리
비주얼 Lambda**가 한다. Threads 게시물은 4컷 만화 이미지 루트와 스토리 답글 체인이 한 세트로 나가야 하니,
이미지를 만든 쪽이 텍스트까지 함께 보낸다.

**둘째는 로컬 딥 리서치다.** `research_cli.py`가 Strands 에이전트를 로컬에서 실행한다. 에이전트는 자유형 토픽을
받아 웹과 논문과 커뮤니티를 스스로 조사하고, 한국어로 합성한 출처 표기 리포트를 JSON으로 stdout에 돌려준다.
OpenClaw 스킬(`~/.openclaw/workspace/skills/omnisummary-research/SKILL.md`)이 이 CLI를 호출하고, Threads 게시는
`--publish threads`를 줄 때만 한다. 다이제스트와 분리된 독립 리서치라서 다이제스트가 다룬 항목에 묶이지 않는다. 같은 스킬이
`recall_cli.py`로 그날 다이제스트가 다룬 것과 추적 중인 트렌드를 모델 호출 없이 되읽는다.

```
[EventBridge 크론] → [다이제스트 Lambda (Docker)]
   → 수집기 (RSS, Reddit, YouTube, WebSearch, X via RSSHub/S3)
   → 집계기 (URL + 제목 중복 제거)
   → 랭커 (Claude Opus 5.5, 소스 슬롯과 origin 상한으로 후보 풀 구성)
   → 트렌드 트래커 (구조화 trends.json, StateStore)
   → 다이제스트 생성기 (Claude Sonnet 5.5, 후보 풀에서 가장 강한 5개를 골라 한국어 DigestContent 작성)
   → AgentCore Memory (다이제스트 스냅샷)
   → 데일리 비주얼 Lambda 비동기 트리거
        └→ 4컷 한글 만화 (gpt-image-2.5-sunburst) + Threads root/reply 체인 게시 (enable_threads_post)
   → FAILED, STALE, DEGRADED 소스(그리고 `alert_on_empty` 소스의 EMPTY)가 있으면 SNS 알림

[OpenClaw 스킬 / 터미널] → research_cli.py (로컬)
   → Strands 딥 리서치 에이전트 (Claude Opus 5.5)
   → 도구: web_search, community_search, search_papers, read_url, recall_trends, recall_digest,
     attach_image, deliver_report
   → Markdown 리포트를 JSON으로 stdout에 반환, --publish threads일 때만 Threads 게시
```

파이프라인 개념도(수집에서 랭킹, 다이제스트, 전달까지):

![How the digest works](diagrams/concept-pipeline.png)

아키텍처(스케줄 다이제스트와 로컬 딥 리서치, 두 경로):

![AWS architecture](diagrams/architecture.png)

## 2. 저장소 구조

| 경로 | 책임 |
|------|------|
| `collectors/` | `BaseCollector` ABC와 공유 `load_items_from_s3`(로컬에서 S3로 올린 park 파일 로더), 그리고 RSS, Reddit(`.rss` 피드), RSSHub(X/Twitter), YouTube, WebSearch(Tavily) 구현 |
| `pipeline/` | `ContentAggregator`, `ContentRanker`, `DigestGenerator`, `TrendTracker`, `DailyVisualMaker`, 그리고 오케스트레이션(`runner.py`: `run_collectors_with_health`, `run_pipeline`, `persist_digest`, `resolve_digest_window`) — CLI와 다이제스트 Lambda가 함께 타는 단일 경로 |
| `agent/` | 딥 리서치 Strands 에이전트(`research_agent.py`)와 도구 8개(`research_tools.py`), 실행별 인메모리 상태 `DigestStateManager`(`tool_state.py`), 데일리 비주얼이 쓰는 이미지 생성기 `VisualGenerator`(`visuals.py`) |
| `research_cli.py` | 딥 리서치 로컬 CLI. JSON 한 개를 stdout에 쓰고 로그는 stderr로 보낸다 |
| `recall_cli.py` | 다이제스트 메모리 조회 CLI(`digest [DATE]`, `trends`). 모델 호출 없는 무료 읽기 전용이고 JSON 한 개를 쓴다. 종료 코드 0/1/2이며, `failed`는 저장소를 못 읽었다는 뜻이고 `found: false`는 저장된 것이 없다는 뜻이다 |
| `shared/` | config(공유 `KOREAN_STYLE_RULES` 포함), models, constants, utils(Bedrock 팩토리와 모델 레지스트리), logger, prompts, state_store, memory, history_store(cross-day dedup 원장과 롤링 로그), formatting, alerts, metrics, research(Tavily와 OpenAlex), media(OG 이미지 fetch) |
| `output/` | 채널별 렌더러(`renderers.py`), 다이제스트 게시(`digest_delivery.py`: Threads 체인, 멱등 원장, 이미지 S3 호스팅), 리서치 리포트 전달(`delivery.py`), Threads API(`threads_handler.py`) |
| `lambda_handlers/` | 다이제스트 핸들러, 데일리 비주얼 핸들러(`visual_handler`, 다이제스트 Lambda가 비동기로 호출한다), Threads 토큰 갱신 핸들러(`threads_refresh_handler`) |
| `infrastructure/` | CDK `foundation_stack`과 `application_stack` |
| `scripts/` | `deploy.py`, `put_secrets.py`, `put_inference_profiles.py`, `ci_synth.py`, `sync_to_s3.py <rsshub\|youtube>`, `sync_all_to_s3.sh`(두 소스를 차례로 sync한다), `sample_digest.py`, `sample_visual_brief.py`, `render_excalidraw.mjs` |

## 3. 설정

`Config.load()`가 `config/config.yaml`을 읽어 `shared/config.py`의 Pydantic 모델로 만든다.
시크릿은 로컬에서는 `.env`, AWS에서는 SSM Parameter Store의 `/{project}/{stage}/{name}` 경로에서 온다.

`config/config.yaml`은 gitignore 대상이고 **`config/config-template.yaml`만 추적된다.** 그래서 CI synth와
인프라 테스트는 `Config.load()`에 기대지 않고 이 템플릿을 명시적으로 로드한다. CI에는 `config.yaml`이 없어서
`Config.load()`는 경고와 함께 템플릿으로 폴백하고(둘 다 없을 때만 실패한다), 그 폴백으로 synth해서는 운영자가
실제로 배포하는 설정을 증명하지 못한다. 같은 이유로 config 불변식 테스트(모델 레지스트리 동기화, 코드 기본값과 배포 설정의 일치,
orientation 어휘, 전달 토글의 명시성)는 `config/*.yaml` **전체**를 parametrize한다. 추적되는 파일만이 모든
체크아웃에 존재하므로, 한 파일에만 걸린 검사는 노트북 한 대에서만 도는 검사다.

값의 우선순위는 `config.yaml`이 Pydantic 필드 기본값을 재정의하는 쪽이다. 단계별 모델의 코드 기본값은
템플릿과 같게 맞춘다([§6](#모델-팩토리) 표). 그 밖의 필드는 아래 표기가 코드 기본값이며, 운영 config가 다르게
두는 값(`park_max_age_hours`, `item_text_max_tokens`, `lookback_hours` 등)은 따로 적는다. 모든 설정 모델은
`extra="forbid"`라서, 코드에서 지운 키가 gitignore된 `config.yaml`에 남아 있으면 `Config.load()`가 실패한다.
필드를 지우는 변경은 운영자의 `config.yaml`도 함께 고쳐야 한다.

`Config.load()`는 호출마다 YAML을 다시 읽고 전체를 재검증한다. 리서치 한 번이면 도구 호출마다 그 작업이
반복되므로, 값을 읽기만 하는 리프 호출자는 `lru_cache(maxsize=1)`로 감싼 `get_config()`를 쓴다
(`output/delivery.py`, `shared/research/research_backends.py`, `agent/research_tools.py`,
`shared/media/og_image.py`). 반면 Config를 변경하는 쪽은 자기 인스턴스가 필요하니 계속
`Config.load()`나 `from_yaml`을 쓴다. `main`과 핸들러의 `set_reference_time`, `ci_synth`의 `vpc_id`, 인프라
테스트의 버킷 오버라이드가 여기 해당한다. 테스트에서는 conftest의 autouse 픽스처가 매 테스트 전후로 캐시를
비운다.

### 3.1 `collectors.*`

각 수집기는 `BaseCollectorConfig`를 상속하고 그 위에 자기 필드를 더한다.

| 그룹 | 필드 | 설명 |
|------|------|------|
| 공통(상속) | `enabled`, `lookback_hours`, `reference_time`, `request_timeout`, `max_retries`, `retry_backoff_sec`, `park_max_age_hours`(기본 36), `error_rate_threshold`(기본 50.0), `empty_rate_threshold`(기본 100.0=비활성) | 활성화 여부, 조회 윈도, 타임아웃, 재시도, S3 park 파일의 나이 예산(초과하면 항목은 쓰되 헬스를 STALE로 보고한다), 그리고 DEGRADED 판정 두 갈래: 입력(피드·계정·채널·쿼리) 실패율과 입력 빈 응답률. 어느 하나만 걸려도 DEGRADED이고 보고만 할 뿐 항목은 그대로 전달한다 |
| `rss` | `feeds`, `max_concurrency`(기본 5) | RSS 피드 URL 목록과 동시 fetch 상한 |
| `reddit` | `subreddits`, `sort`, `limit` | 서브레딧, 정렬, 개수 |
| `youtube` | `channels`, `max_videos_per_channel`, `resolve_timeout`, `transcript_timeout`, `transcript_language`, `max_concurrency`, `park_required` | 채널, 영상 수, 자막 관련 설정, 동시 채널 상한, AWS에서 park 파일 부재를 DEGRADED로 볼지([§4.2](#42-s3-park-파일-로더)) |
| `web_search` | `trend_searches`, `max_results_per_query`, `max_refine_queries`, `min_search_score`, `refine_model`(Sonnet 5.5), `max_concurrency` | Tavily 검색과 관련도 필터, 쿼리 정제 모델 |
| `rsshub` | `base_url`, `accounts`, `max_concurrency` | X 계정(로컬 컨테이너 또는 S3)과 동시 fetch 상한 |

`lookback_hours`는 하루가 아니라 **직전 실행까지** 닿아야 한다. cutoff는 다이제스트 날짜의 **끝** 자정에
앵커되는데(`resolve_digest_window`) cron은 그 자정보다 몇 시간 앞서 수집한다. 그래서 24는 윈도를 정확히
다이제스트 날짜와 같게 만드는데, 그러면 19:00 KST에 수집이 끝난 뒤 발행된 글은 그날 실행에는 아직 없고 다음
실행의 cutoff보다는 이미 과거다. 매일 다섯 시간이 어느 다이제스트에도 실리지 않는다. 최소값은
`48 - 실행 시각(로컬)`이고 `Config._lookback_reaches_the_previous_run`이 config 로드에서 검증한다. 넓혀도 같은
기사가 두 번 나가지 않는다. cross-day 중복은 윈도가 아니라 URL 동일성(`PublishedUrlLedger` + 최근 AgentCore
스냅샷)으로 배제된다. 그리고 이 폭이 park 파일의 'stale beats empty' 계약을 실제로 성립시킨다. 24에서는 STALE
파일의 날짜 있는 항목이 전부 cutoff보다 앞이라, 소스가 STALE을 항목 0개로 보고한다.

이 값은 `BaseCollectorConfig`에 `Field(default=30, ge=1)`로 **한 번만** 선언된다. 하위 모델이
`lookback_hours: int = 72`로 다시 선언하면 그 재선언이 FieldInfo를 통째로 갈아치워 `ge=1`까지 함께 지우고, 그
소스만 음수 윈도를 받아들인다. 더 넓은 윈도가 필요한 소스는 나머지 값들과 같은 자리인 YAML에 쓴다.
config.yaml은 YAML 앵커로 다섯 소스가 같은 값을 쓰고, 템플릿은 youtube에 정의한 앵커를 reddit과 rss가
alias하며 web_search와 rsshub는 72를 따로 적는다.

`error_rate_threshold`는 RSSHub 전용이 아니라 `BaseCollectorConfig`의 공통 노브다. 같은 뜻의 숫자를 두 벌
만들지 않으려는 것이고, RSS와 YouTube와 web_search도 같은 임계로 DEGRADED를 보고한다.

실패율이 못 보는 모양이 하나 있어서 노브가 하나 더 있다. 입력 전부가 HTTP 200에 entries 0으로 답하는
경우다(만료된 RSSHub 쿠키, 페이월 200, 아무것도 없는 플레이리스트). 실패율은 0이고, 항목을 낸 입력이 하나만
있어도 소스는 OK로 읽힌다. `empty_rate_threshold`가 빈 응답률로 이것을 잡는다. 소스마다 성격이 달라서(RSS
블로그는 하루 글이 없는 게 정상이고, X 계정 40개가 모두 조용한 건 세션이 깨진 것이다) 기본값은 비활성인
100.0이고 config.yaml에서 소스별로 켠다.

**절대 개수로 판정하는 노브는 두지 않는다.** 비율은 입력이 적은 소스에서 판정을 표현하지 못하지만, 입력이 적은
유일한 소스인 reddit은 서브레딧 전체를 요청 한 번으로 받으니([§4.4](#44-reddit-redditpy)) 실패가 소스
단위로만 일어나고 FAILED로 충분하다.

`collectors.alert_on_empty`(기본 `[]`)는 EMPTY가 사건인 소스의 이름 목록이다(예: `["rss", "web_search"]`).
목록이 필요한 이유는 이렇다. 어두워진 소스는 예외도 남기지 않고 stale park 파일도 남기지 않고 실패율도
남기지 않아서 다른 어떤 신호에도 걸리지 않는다. 그렇다고 빈 소스면 무조건 알리게 두면 reddit이나 X처럼
조용한 날이 정상인 소스가 매일 페이징하고, 그러면 사람은 곧 알림을 무시한다. 그래서 명시적 opt-in 목록으로
두었고, 비어 있으면 EMPTY로는 절대 알리지 않는다. 이름은 `shared/constants.py`의 `COLLECTOR_NAMES`(=
`pipeline/runner.py`의 수집기 레지스트리 키)로 검증한다. 오타 난 이름은 어떤 소스와도 매칭되지 않아서, 감시하려던
어두운 소스가 그대로 어두운 채 남는다. 그래서 config 로드에서 실패시킨다. 같은 이유로 `pipeline.source_slots`의
키는 `SourceType` 값이어야 한다. 랭커는 `item.source_type.value`로 매칭하니 오타 난 키는 그 소스의 보장 슬롯을
조용히 없앤다.

### 3.2 `pipeline`

| 영역 | 필드 | 설명 |
|------|------|------|
| 모델 | `ranking_model`(Opus 5.5), `digest_model`(Sonnet 5.5, 그라운딩과 비주얼 편집기·아트 디렉터도 이 모델을 쓴다), `trend_model`(Sonnet 5.5) | 단계별 모델([§6](#모델-팩토리)) |
| 랭킹 | `ranking_batch_size`(기본 40), `ranking_max_concurrency`(기본 4), `ranking_max_retries`(기본 3), `ranking_retry_backoff_sec`(기본 5), `ranking_scoring_rubric`, `item_text_max_tokens` | 병렬 배치, Bedrock fan-out 상한, 배치 재시도, 점수 루브릭 |
| 후보 풀 | `top_n`(기본 5), `min_score`(기본 0.6), `digest_candidate_buffer`(기본 3), `source_slots`(소스마다 1), `max_per_origin`(기본 1) | 에디터에게 넘기는 후보는 `top_n + digest_candidate_buffer`개이고, 소스 슬롯과 origin 상한은 이 풀을 채우는 데만 쓴다. 실제로 나가는 `top_n`개는 에디터가 고른다([§5.2](#52-랭커-rankerpy)) |
| 중복 | `published_url_ttl_days`(기본 6, 최소 2), `recent_leads_window`(기본 5) | cross-day dedup 원장 TTL과 최근 다이제스트 제목을 읽는 창, 반복 방지용 최근 lead 윈도. TTL이 최소 2인 것은 날짜만 있는 웹 결과가 이틀 연속 윈도에 들어가고([§4.7](#47-websearch-web_searchpy)) 두 번째 게시를 원장이 막기 때문이다 |
| 트렌드 | `trend_retention_days`, `trend_cooling_days`, `trend_max_evidence`, `trend_max_active_trends`, `trend_momentum_half_life_days` | 보존과 냉각 기간, 증거와 active 캡, momentum 반감기 |
| 전달 | `enable_threads_post`, `threads_topic_tag`(기본 `AI`) | Threads 게시 on/off. 코드 기본값은 off이고 배포 설정은 on이다. 데일리 비주얼 Lambda가 게시한다. 토픽 태그는 root 게시물에만 붙는다([§5.8](#58-threads-전달-outputthreads_handlerpy-enable_threads_post)) |
| AGI 카운트다운 | `agi_countdown_date`(기본 `2029-01-01`), `agi_countdown_template`, `agi_countdown_after`, `agi_countdown_position`(`prefix`\|`suffix`, 코드 기본값과 배포 설정 모두 `prefix`) | 다이제스트 lead에 코드가 붙이는 "AGI N일 전" 인트로와, lead의 어느 쪽 끝에 붙일지([§5.7](#57-agi-카운트다운-인트로-sharedformattingpy-agi_countdown_intro) 참조) |
| 시각화 | `image_model`(`gpt-image-2.5-sunburst`), `image_sizes`, `visual_image_quality`(`xhigh`), `visual_editor_source_max_tokens`, `visual_synopsis_source_max_tokens`, `visual_image_timeout_sec`(기본 300)과 `visual_image_max_retries`(기본 0), `visual_character_enabled`, `visual_character_sheet`, `visual_character_reference`(`assets/persona.jpg`) | 이미지 모델과 품질 등급, orientation에서 size로 가는 딕셔너리(키는 `VisualOrientation` 어휘로 고정이고 값만 튜닝한다), 입력 상한, 이미지 HTTP 호출 상한. SDK 기본값 600초씩 2회는 비주얼 Lambda의 15분 예산을 넘길 수 있어서 config로 고정했다 |
| 프롬프트 주입 | `digest_language_rules`, `digest_voice_guidance`, `ranking_audience_description`, `digest_audience_description`, `visual_audience_description`, `visual_on_image_language`, `visual_synopsis_style_guidance`, `visual_synopsis_style_aesthetic`, `visual_guardrails`, `visual_moderation_softening_instruction` | 언어 규칙과 대상 독자, 톤, 미감, 가드레일, 모더레이션 완화 문구. 하드코딩 대신 템플릿 변수로 주입한다. `digest_language_rules`는 한국어 규칙과 번역 용어집을 튜닝하는 자리이고 언어 스위치가 아니다. 한국어 전용이 의도한 제품이다. `digest_voice_guidance`는 Gruber 톤이며, 단일 냉소 프레임으로 기본 고정하지 말고 그날 사실이 정당화할 때만 각을 세우도록 쓰여 있다 |

`visual_on_image_language`는 "short, natural Korean"이다. 이미지 안의 말풍선과 라벨을 짧은 한국어로 쓴다.
`gpt-image-2`, `gpt-image-2.5-flare`, `gpt-image-2.5-sunburst` 세 모델 모두 같은 브리프의 한국어 문자열을
정확히 렌더했다([§5.6](#56-데일리-비주얼-daily_visualpy)).

### 3.3 `agent`

딥 리서치 에이전트 설정이다. 이 절의 값은 두 종류로 나뉘고 그 구분이 중요하다. `research_breadth`,
`research_max_iterations`, `research_target_words`는 시스템 프롬프트에 보간되어 에이전트가 가이던스로 따르는 값이다
(그래도 바꾸면 실제 동작이 바뀐다). 나머지는 코드가 강제한다. `research_max_turns`, `research_max_total_tokens`,
`research_max_output_tokens`는 실행을 멈추는 Strands `Limits`이고, `research_content_cap_chars`,
`research_max_staged_images`, `og_image_timeout_sec`, `og_image_max_bytes`는 도구가 적용하는 하드 캡이다.
`research_max_threads_posts`는 둘 다다. 프롬프트가 그 게시물 수 안에서 서사를 짜라고 알리고, 렌더러가 넘치는 게시물을 잘라낸다.

| 필드 | 설명 |
|------|------|
| `model_id` | 에이전트 모델(기본 Opus 5.5) |
| `research_breadth`, `research_max_iterations` | 프롬프트에 주입되는 검색 폭(쿼리 수)과 깊이(라운드 수) 가이던스. 루프 상한이 아니다 |
| `research_max_turns`, `research_max_total_tokens`, `research_max_output_tokens` | 툴 루프의 하드 상한(`strands.types.agent.Limits`로 실행마다 전달한다). SDK가 매 턴 경계에서 검사하고 `stop_reason="limit_*"`으로 멈춘다. 사이클마다 대화 전체를 다시 보내므로, 이 값들이 없으면 비용과 벽시계 시간에 상한이 없다 |
| `research_target_words` | Markdown 리포트의 목표 분량(단어) 가이던스(기본 1500) |
| `research_max_threads_posts` | Threads 게시물(root와 reply) 총수 하드 캡(기본 10). 너무 긴 리포트가 공개 게시물 수십 개로 퍼지지 않게 코드가 트림한다. 이 값이 Threads 리포트의 산문 예산을 정한다([§9의 `<delivery>`](#system_prompt_template) 참조) |
| `research_content_cap_chars` | `read_url` 한 페이지의 추출 텍스트 상한(기본 50000) |
| `research_max_staged_images` | 한 리서치 실행이 stage할 수 있는 OG 이미지 수 캡(기본 4, invoke당 메모리 bound) |
| `og_image_timeout_sec`, `og_image_max_bytes` | OG 이미지 fetch 타임아웃과 최대 바이트(스트리밍 중 초과하면 중단한다) |
| `community_search_domains` | `community_search`의 도메인 허용 목록(reddit, x, HN, substack 등) |
| `search_result_limit`, `search_content_preview_chars`, `search_request_timeout`, `search_max_retries`, `search_retry_backoff_sec` | 검색 결과 수, 미리보기, 타임아웃, 재시도 |
| `search_paper_max_authors`, `search_paper_abstract_max_chars` | OpenAlex 논문 결과 포맷 |
| `boto_read_timeout`, `boto_connect_timeout`, `boto_max_attempts` | 에이전트 Bedrock 클라이언트의 boto 설정 |

### 3.4 `aws`

| 필드 | 설명 |
|------|------|
| `region`, `bedrock_region`, `profile`, `project_name`, `stage` | 리전, 프로파일, 프로젝트와 스테이지 |
| `timezone` | 다이제스트 날짜의 기준 TZ(예: `Asia/Seoul`) |
| `digest_cron_hour` / `digest_cron_minute` | EventBridge 크론(UTC 기준) |
| `threads_token_refresh_days` | Threads 장기 토큰(60일 만료)의 갱신 주기(기본 50일, 59 이하) |
| `rsshub_desired_count` | RSSHub Fargate 서비스의 태스크 수(기본 0, [§11](#foundation_stack) 참조) |
| `vpc_id`, `subnet_ids`, `state_bucket_name`, `s3_prefix` | 네트워킹과 상태 버킷 |

### 3.5 시크릿과 환경 변수

| 변수 | 출처 | 용도 |
|------|------|------|
| `TAVILY_API_KEY` | `.env` → SSM | 웹, 커뮤니티, 뉴스 검색 (아래 단서 참고) |
| `OPENAI_API_KEY` | `.env` → SSM | gpt-image 이미지 생성 |
| `THREADS_ACCESS_TOKEN` | `.env` → SSM | Threads 게시. 장기 토큰이며 50일 주기로 자동 갱신한 뒤 SSM에 재기록한다 |
| `THREADS_USER_ID` | `.env` → SSM | Threads 게시 대상 사용자 ID |
| `YOUTUBE_API_KEY` | `.env` → SSM | YouTube Data API |
| `ALERT_EMAIL` | `.env` → 배포 시 SNS 구독 | 소스 실패 알림 |
| `MEMORY_ID`, `STATE_BUCKET`, `S3_PREFIX`, `ALERT_SNS_TOPIC_ARN`, `RSSHUB_BASE_URL`, `PROJECT_NAME`, `STAGE` | CDK 주입(AWS) | 런타임 리소스 식별자 |

`TAVILY_API_KEY`에는 단서가 하나 붙는다. web_search 수집기만 이 키를 `resolve_secret(strict=True)`로
해석한다. SSM 읽기 실패(거부나 스로틀)와 키 부재를 구분해야 하기 때문이다. 둘을 같이 `[]`로 떨어뜨리면 그날 웹
소스 전체가 경고 한 줄과 함께 사라진다. 읽기 실패는 raise해서 헬스를 FAILED로, 그리고 알림으로 잇고,
파라미터가 정말 없을 때만 `""`로 조용히 스킵한다. `shared/research` 백엔드와 에이전트 경로는 빈 문자열 degrade
계약을 그대로 쓴다. 그쪽은 기능 하나가 비활성인 것이고, 수집기 쪽은 그날 다이제스트의 소스 하나가 빠지는
것이다.

**`.env`의 시크릿은 CDK 스택을 통과하지 않는다.** CloudFormation 템플릿은 SecureString을 담을 수 없어서, 값을
스택에 넘기면 `cdk.out/*.template.json`과 CDK 스테이징 버킷, `cloudformation:GetTemplate` 응답에 평문으로
남는다. 그 경로에 노출되는 것은 Tavily·OpenAI·YouTube 키, Threads 토큰, X 세션 쿠키, 곧 이 프로젝트의 모든
자격증명이다.

그래서 스택은 파라미터 경로만 `SSM_PLACEHOLDER` 값으로 만들고, 실제 값은 `scripts/put_secrets.py`가
SecureString으로 기록한다. 재배포가 값을 되돌리지 않는데, CloudFormation은 템플릿 속성이 바뀐 리소스만
갱신하고 플레이스홀더는 변하지 않기 때문이다.

스크립트에는 안전장치가 두 개 있다. 첫째, 이미 SecureString인 파라미터는 건너뛴다. Threads 토큰은 갱신
Lambda가 제자리에서 회전시키니, 로컬 `.env` 사본을 다시 쓰면 만료된 토큰으로 되돌아간다(`--force`로만
덮어쓴다). 둘째, 비어 있는 환경 변수로 기존 값을 지우지 않는다. 여기에 `resolve_secret`이 플레이스홀더를
미설정으로 취급하니, put_secrets를 건너뛴 배포는 플레이스홀더를 API 토큰으로 보내는 대신 정상적인 '자격증명
없음' 경로로 degrade한다. Lambda와 로컬 CLI는 env를 먼저 보고 그다음 SSM을 본다.

X 세션 쿠키만 경로가 다르다. Fargate 태스크 정의의 `secrets` 블록으로 주입되며, 템플릿에는 ARN만 들어가고 값은
태스크 시작 시 ECS 에이전트가 가져온다. `RSSHUB_BASE_URL`은 `rsshub_base_url` CDK context로 재정의할 수 있고,
로컬 개발에서는 RSSHub Docker 컨테이너가 `localhost:RSSHUB_PORT`(기본 `1200`)에서 돌아야 X 수집이 된다.

## 4. 수집기

### 4.1 공통 계약

모든 수집기는 `BaseCollector.collect() -> list[CollectedItem]`을 구현하고
`in_collection_window(published_at, lookback_hours, reference_time)`(`collectors/base.py`)으로 시간 범위를
필터링한다. 윈도는 `[reference_time - lookback_hours, reference_time]`이고 **양끝이 닫혀 있다**. 라이브 경로와
park 경로가 이 한 정의를 쓰므로, `--date`로 과거 날짜를 다시 돌려도 어느 소스도 **오늘** 항목을 싣지 않는다.
`published_at`이 없는 항목은 윈도 **안**으로 본다. 날짜가 없다는 것이 범위 밖이라는 증거는
아니고, 그것으로 떨어뜨리면 메타데이터 결손 때문에 소스가 조용히 줄어든다.

수집기 하나의 `collect()` 전체는 `collectors.collector_budget_sec`(기본 600초)로 묶인다. 입력별 타임아웃은
소스를 묶지 못한다. RSSHub 계정 피드 41개를 `max_concurrency` 5로, 각각 `(request_timeout +
retry_backoff_sec) * max_retries = 105초`까지 허용하면 직렬 라운드로 약 16분이 필요하고, 이는 다이제스트
Lambda의 15분 예산 전체보다 길다. 소스 하나가 막히면 실행 전체가 죽는다는 뜻이다. 이 예산의 타임아웃은
raise되어 러너의 예외 분기로 들어가고, 그 소스는 **부분 결과 없이** FAILED로 보고된다. 0이면 비활성이고,
그때의 유일한 상한은 다시 Lambda 예산이다.

### 4.2 S3 park 파일 로더

`collectors/base.py`의 `load_items_from_s3(filename, max_age_hours)`가 담당한다. X와 YouTube는 데이터센터
(Lambda) IP에서 차단되니, 로컬 sync 스크립트가 거주용 IP로 항목을 수집해 S3에 미리 적재하고 AWS에서는
수집기가 라이브 fetch 대신 이 파일을 읽는다. S3 키 규칙은 trends.json과 같다(`S3_PREFIX`의 부모 디렉터리에
파일명을 붙인다). 두 수집기가 같은 로더를 공유하니 park 계약이 소스마다 갈라지지 않는다.

로더는 `ParkedItems(outcome, items, detail, meta)`를 돌려준다. park 파일의 나이가 데이터와 함께
흐르게 하려고 명시적 모델로 만든 것이고, 그래서 모듈 전역이나 threadlocal에 staleness 플래그를 두지 않는다.
`outcome`은 네 값 중 하나다. `absent`는 버킷이 설정되지 않았거나 객체가 없는 경우, `fresh`와 `stale`은 이름
그대로이고, `error`는 읽지 못한 경우다. 호출자는 파생 프로퍼티로 분기한다. `usable`이면 park 항목을 쓰고
(fresh와 stale), `degraded`면 헬스를 STALE로 올린다(stale과 error). 수집기는 결과를 `self.park_status`에
남기고 `run_collectors_with_health`가 그것을 읽는다.

park 항목도 라이브 경로와 **같은** 수집 윈도를 통과해야 한다. `parked_items_in_window`가
`in_collection_window`를 그대로 적용한다. 파일 항목을 그대로 넘기면 윈도 두 개가 조용히 무시된다. STALE이지만
쓸 수 있는 파일은 `park_max_age + lookback`만큼 오래된 항목까지 랭킹에 흘려보내고, 실행의 기준 시각(`--date`,
다이제스트 핸들러의 `set_reference_time`)은 어디에도 적용되지 않는다. 윈도 밖으로 떨어진 개수는 WARNING으로
남는다. 그 수가 올라가는 것 자체가 멈춘 sync의 신호다.

이 필터는 `lookback_hours`가 이전 실행까지 닿는다는 전제에 의존한다(§3.1). 전제가 깨지면 sync 시각과 다음
cutoff 사이의 슬라이스가 어느 park 파일에도 없으면서 다음 실행에서는 이미 과거가 되어, 필터가 정상 경로에
구멍을 낸다. 앵커 30과 `_lookback_reaches_the_previous_run` 검증이 그 전제를 강제한다.

`meta`는 park 파일을 쓴 sync가 남긴 수집 방식 기록이며 선택적이다. RSSHub sync는 `accounts_total`,
`accounts_failed`, `accounts_empty`를 적고, 수집기는 그것을 되읽어 라이브 경로와 **같은** 임계
(`error_rate_threshold`, `empty_rate_threshold`)로 판정한 뒤 `degraded_detail`을 세워 헬스를 DEGRADED로 만든다.
파일의 신선도만으로는 40개 계정 중 3개만 모은 sync를 건강한 sync와 구분할 수 없다. 빈 응답
개수(`accounts_empty`)까지 되읽는 이유도 같다. 그 값을 무시하면 모든 계정이 200에 entries 0으로 답한 sync가 쓴
신선한 park 파일이 완전히 건강하게 읽힌다.

**신선도 봉투.** sync 스크립트는 `{generated_at, items, meta?}` 봉투(`dump_items_envelope`)로 적재한다.
`meta`는 비어 있으면 아예 쓰지 않아서 구버전 리더와 바이트 호환이다. 로더는 봉투와 레거시 bare-list를 모두
읽되, `generated_at`이 `park_max_age_hours`(수집기별 config, 기본 36시간이고 호출자가 항상 넘긴다)보다
오래되면 항목은 그대로 반환하면서 `stale`로 표시한다. 오래된 게 빈 것보다
낫다는 판단이다. 대신 이렇게 해두면 로컬 cron이 조용히 멈춰 며칠 지난 항목을 오늘 것으로 재수집하는 사고가
정상 실행처럼 보이지 않고 헬스 STALE과 SNS 알림으로 표면화된다.

**park 파일이 그 소스의 주 경로면 부재도 DEGRADED다(`park_required`).** `absent`는 설계상
`ParkedItems.degraded`에서 빠져 있다. 로컬에서는 '라이브로 수집하라'는 정상 신호이기 때문이다. 그런데 AWS의
youtube는 그렇지 않다. 자막은 데이터센터 IP로 못 받으니 park 파일이 자막을 실어오는 유일한 경로인데, 라이브
수집은 메타데이터만으로 항목을 계속 만들어낸다. 그대로 두면 S3_PREFIX 오타나 삭제된 오브젝트나 한 번도 돌지
않은 sync가 항목을 내놓고 OK로 보고하면서 자막을 전부 조용히 버린다. `park_required`가 켜진
소스는 **AWS에서** 부재를 DEGRADED로 올리고 detail에 S3 키를 담는다(`flag_missing_park`). 로컬에서는 라이브
수집이 바로 그 파일을 쓰는 경로라서 검사하지 않는다. 여기도 보고만 하고 항목은 그대로 전달한다.

**빈 park 파일에는 두 가지 의미가 있다.** 항목이 0건이면서 동시에 나이 예산을 넘긴 봉투는 부재(`absent`)로
취급해 라이브 수집으로 폴백한다. 그쪽에서 전면 장애면 FAILED로 알림이 간다. 로컬 sync가 멈춰 빈 파일만 남은
상태를 '오늘은 조용했다'로 오해하면 안 되기 때문이다. 반대로 신선한 0건 봉투는 정말 조용한 sync 날이니 그대로
반환해 거짓 FAILED 알림을 만들지 않는다.

**읽기 오류는 분류한다.** 손상된 JSON, 검증 실패, 예기치 않은 S3 오류(AccessDenied, 스로틀 등), 그리고
자격증명 부재나 연결·읽기 타임아웃 같은 `BotoCoreError`는 모두 `error`로 분류해 경고 로그를 남기고 헬스를
STALE로 올린다. 조용한 `absent`는 `NoSuchKey`와 `NoSuchBucket`, 404뿐이다. 분류는 로그 레벨과 보고 상태만
바꾸며, 어떤 S3 실패도 raise하지 않고 항상 라이브 수집으로 폴백한다. `ClientError`만 잡으면
`NoCredentialsError`와 `ReadTimeoutError`가 수집기를 뚫고 나가 소스 전체를 실패시켜, 이 함수가 문서로 약속한
계약을 정면으로 어긴다. 권한 오류를 파일 없음과 같은 info 로그로 묻는 것도 같은 종류의 실수다.

**sync 스크립트는 빈 봉투도 올린다.** `scripts/sync_to_s3.py <rsshub|youtube>`는 항목이 0건이어도 봉투를 항상
업로드한다.
`generated_at` 스탬프가 sync가 돌았다는 사실의 유일한 증거이기 때문이다. 단 이는 `collect()`가 정상 반환한
경우뿐이고, 수집기 예외는 그대로 전파되어 직전의 좋은 park 파일을 덮어쓰지 않는다.

**sync는 로컬 크론이 돌린다.** `scripts/sync_all_to_s3.sh`가 두 소스를 차례로 sync하고, 한쪽 실패가 다른 쪽을
멈추지 않는다. X는 로컬 RSSHub 컨테이너(`rsshub`)가 떠 있어야 수집되는데, 명시적으로 멈춘 컨테이너는
`--restart unless-stopped`로도 다시 뜨지 않는다. 그래서 스크립트가 `localhost:1200` 응답이 없으면
`docker start rsshub`을 실행하고 2초 간격으로 30번까지 응답을 기다린다. 컨테이너가 멈춘 채로 크론만 돌면 X park 파일은 계속
STALE이 되고, park 항목은 수집 윈도 밖이라 소스가 0건을 낸다(2026-08-17부터 54일간 그랬다).

### 4.3 RSS (`rss.py`)

`config.collectors.rss.feeds`의 피드를 읽고 `feed_url`과 `feed_title`을 메타데이터로 남긴다. fetch와 파싱과
엔트리 루프는 모두 `collectors/base.py`의 공용 헬퍼가 소유한다(`fetch_feed`, `fetch_feed_with_retry`,
`parse_feed_entries`). RSS와 Reddit, RSSHub의 엔트리 루프는 글자 그대로 같으므로, 사본을 셋 두면 한쪽을 고칠 때
나머지 둘이 조용히 뒤처진다.

**피드 구성은 1차 출처를 먼저 둔다.** 운영 config는 연구소와 플랫폼의 자체 피드(OpenAI, Google DeepMind, Google
Research, Hugging Face)와 1차 분석 뉴스레터(Latent Space, Import AI 등)를 두고, 21일 동안 한 건도 내지 않은 피드는
뺀다. 웹 검색이 찾는 재가공 기사보다 연구소의 원문이 낫고, 그 원문이 후보 풀에 없으면 랭커가 고를 수도 없다.
Tavily의 X 프로필 쿼리처럼 결과가 대부분 프로필 페이지인 검색도 두지 않는다. X는 RSSHub가 공급한다.

**본문은 httpx로 받아서 feedparser에 넘긴다.** `feedparser.parse(url)`은 타임아웃 없는 urllib으로 직접
가져오니 매달린 호스트를 끊을 수 없다. 코루틴 안의 httpx 타임아웃은 진짜로 취소되고 아무 비용도 남기지 않는다.

**본문의 HTML은 수집 시점에 벗긴다.** `parse_feed_entries`가 `<`를 포함한 본문을 BeautifulSoup
`get_text(" ", strip=True)`로 텍스트로 바꾼다. Substack 본문은 65~74%가 마크업이라(예: 122,215자 중 텍스트
31,424자) 그대로 두면 `item_text_max_tokens` 절단이 산문 대신 태그를 남기고, 같은 이야기의 중복 쌍에서 마크업이
많은 쪽이 "더 긴" 본문으로 살아남는다. 링크를 `item.text`에서 읽는 코드는 없다.

**fan-out 상한(`max_concurrency`, 기본 5)이 필요한 이유.** 한 번에 도는 fetch 수를 묶어서 수집기의 최악
wall time을 계산 가능하게 유지한다. 세마포어는 `collect()` 안에서, 다시 말해 실행 중인 루프에서 만든다
(임포트나 `__init__`이 아니다). RSSHub와 같은 패턴이다.

**일시적 실패는 재시도한다.** 타임아웃과 전송 계층 실패, 일시적 상태 코드(429와 5xx)는
`fetch_feed_with_retry`가 `retry_async`로 `max_retries`(기본 3)까지 재시도하고, 매 시도가 자기
`request_timeout`을 온전히 갖는다. 재시도가 없으면 한 번의 blip이 그 피드의 하루치 항목을 통째로 가져간다.
backoff는 URL을 시드로 한 지터를 얹은 선형이다. 지터가 없으면 동시에 재시도하는 수십 개 피드가 상류가
레이트리밋한 바로 그 버스트로 다시 뭉친다. 반면 403이나 404, 파싱 불가 본문은 재시도해도 결론이 바뀌지 않으니
첫 응답에서 즉시 실패시킨다.

분류는 `collectors/base.py`의 `feed_status_failure(description, status)`와
`feed_parse_failure(description, bozo_exception)` 한 쌍이 소유하고 RSS와 Reddit, RSSHub가 함께 쓴다. 상태
코드는 `RETRIABLE_STATUS_CODES`(429와 5xx)로 판정한다. 전송 계층 실패도 여기서 갈린다. httpx의
`TimeoutException`과 나머지 `HTTPError`(DNS, reset, TLS)는 transient로 올리고, 진짜 malformed 문서만 영구
실패다. 이 구분이 없으면 DNS 한 번의 흔들림이 영구 파싱 실패로 분류되어 그 피드의 하루를 날린다. 같은 피드의
HTTP 503은 세 번 시도하는데 그보다 회복 가능성이 높은 실패를 한 번에 포기하는 셈이다.

**최악 wall time.** 피드당 `max_retries * request_timeout + 선형 backoff`이므로 기본값에서
`3*30s + (5s+10s)` = 105초다. 피드는 `max_concurrency`개씩 도니 수집기 전체는
`ceil(feeds / max_concurrency) * 105s`다. 피드 수는 config가 정하니 여기에 숫자를 박지 않는다. 실제 상한은
`collectors.collector_budget_sec`(기본 600초)이고 Lambda의 15분 예산이 아니다([§4.1](#41-공통-계약)). 이 곱이
600초를 넘기면 소스는 **부분 결과 없이** FAILED로 보고되니, `max_concurrency`를 올리거나 피드를 줄여야
한다는 뜻이다.

**실패 신호.** 죽은 피드(HTTP 4xx/5xx, entries 없는 bozo)와 재시도를 소진한 타임아웃은 빈 결과가 아니라
예외로 올린다. `gather_collector_results(raise_if_all_failed=True)`가 전 피드 실패일 때만 이를 승격시키니,
일부 피드 장애는 로깅 후 건너뛰고 전면 장애만 FAILED로 알린다.

### 4.4 Reddit (`reddit.py`)

공개 `.rss` 피드를 쓴다. Reddit이 셀프서비스 OAuth 앱 생성을 동결했고(Responsible Builder Policy, 2025-11)
`.json` API는 데이터센터 IP를 차단했지만 `.rss` 피드는 열려 있기 때문이다. 그래서 자격증명도 앱 등록도 필요하지
않다.

**서브레딧 전부를 요청 한 번으로 받는다.** `https://www.reddit.com/r/A+B/{sort}/.rss?limit=…`다. 몇 초 간격으로
서브레딧을 하나씩 요청하면 두 번째 요청이 429를 받는다. 운영 30회 실행 모두 r/MachineLearning이 첫 두 시도에서
429였고 17회는 수집을 놓쳤다. 다중 서브레딧 피드는 하루치 전체를 돌려주며(실측 52개, 48 + 4), 각 엔트리는 링크의
`/r/<sub>/comments/`(없으면 category 태그)로 서브레딧을 되찾는다. `limit`은 서브레딧당 상한이고 피드 순서(업스트림
랭킹)대로 그만큼만 남긴다. `limit × 서브레딧 수`가 Reddit 리스팅 상한 100을 넘지 않게 요청한다.

**top 정렬은 `t=day`다.** Reddit의 창은 fetch 시각 기준이고 실행은 하루 간격이다. 더 넓은 창(`t=week`)은 다른
랭킹이다. 지난주에 뜬 글이 오늘 글을 밀어내고, 실제 30시간 윈도는 `in_collection_window`가 어차피 다시 거른다.

경로는 공용 `fetch_feed_with_retry`이고 Lambda에서도 직접 요청한다. `.rss`는 데이터센터 IP에서도 200이고, 429는
재시도 가능한 오류로 분류되어 지터가 붙은 선형 백오프로 다시 시도한다. 프록시 경로는 없다. Lambda에 연결된 적이 한 번도
없었고, 로컬에서는 재시도 가능한 429를 프록시의 403(즉시 실패)으로 바꿔 Reddit을 통째로 잃게 했다.

**헬스.** 요청이 하나라서 실패는 소스 단위다. 재시도를 소진하면 예외가 올라가 FAILED가 된다. 피드에 한 건도
없는 서브레딧은 빈 입력으로 세고 `empty_rate_threshold`로 판정한다.

**트레이드오프.** RSS에는 `score`나 `num_comments` 같은 engagement 값이 없어서, Reddit 항목의 랭킹은 LLM의
품질 판단에 더 의존한다.

### 4.5 RSSHub (`rsshub.py`)

로컬이나 컨테이너 RSSHub를 통해 X/Twitter 피드를 읽고, S3에 사전 동기화된 스냅샷(`rsshub_items.json`,
`scripts/sync_to_s3.py rsshub`가 적재한다)이 있으면 공유 `load_items_from_s3`로 그것을 쓴다.

계정별 fetch도 RSS와 같은 공용 `fetch_feed_with_retry`를 쓴다. httpx 타임아웃이 매달린 피드 호스트를 끊고,
타임아웃은 빈 결과가 아니라 실패로 집계한다. 매 시도가 자기 `request_timeout`을 온전히 갖고, RSSHub 자신의
HTTP 상태도 분류한다. 429와 5xx는 재시도하고 그 밖의 4xx와 파싱 불가 본문은 즉시 실패시킨다. 계정이 가장 많은
소스라 시도가 한 번뿐이면 blip 몇 개가 겹치는 것만으로 `error_rate_threshold`까지 밀린다.

degraded 힌트는 실제로 실패한 계정의 platform에서 뽑는다. Twitter 쿠키 만료를 무조건 단정하면 mastodon 같은
다른 라우트가 죽은 날에 운영자를 엉뚱한 컨테이너 설정으로 보낸다.

**팬아웃 상한.** 계정 수가 40개를 넘으니 한 번에 모두 띄우면 상류에 대한 버스트가 된다. 그래서 `collect()`
안에서 `asyncio.Semaphore(max_concurrency)`를 만들고, 소스 전체는 `collectors.collector_budget_sec`가
묶는다([§4.1](#41-공통-계약)).

**헬스.** 실패한 계정과 빈 계정을 자체 추적하며 `error_rate_threshold`를 갖는다. 서비스
도달성(`_check_reachable`)은 OK인데 모든 계정이 실패하면 RuntimeError로 올려 FAILED로 알린다. 조용한 날과
구분하기 위함이다. 일부 실패는 허용한다.

### 4.6 YouTube (`youtube.py`)

AWS에서는 `scripts/sync_to_s3.py youtube`가 거주용 IP로 자막까지 수집해 적재한 `youtube_items.json`을 공유
`load_items_from_s3`로 먼저 읽는다. 이 경로를 강하게 선호한다. 없으면 `YOUTUBE_API_KEY`로 라이브 수집하고,
키도 없으면 채널 RSS로 폴백한다.

**정규 영상 목록을 읽는다.** 채널의 `UULF…` 재생목록(정규 영상만, Shorts 제외)을 먼저 읽고, 그 목록이 없는
채널(`404 playlistNotFound`)만 전체 업로드 `UU…`로 폴백한다. `max_videos_per_channel`이 1이면 전체 업로드
목록에서는 40초짜리 Short가 그 채널의 자리를 가져갔다. Lenny's와 No Priors의 최근 업로드 8개 중 6개가 Shorts였다.

**API 키 해석은 루프 밖에서 한 번만 한다.** `collect()`가 park 파일을 확인한 뒤 `asyncio.to_thread`로
`resolve_secret`(env를 먼저, 그다음 SSM)을 한 번 호출하고 결과를 `self.api_key`에 둔다. lazy 프로퍼티로 두면
채널 fan-out 안에서 처음 접근될 때 블로킹 SSM 호출이 이벤트 루프 스레드에서 돈다. park 파일로 단축되는 경로는
SSM을 아예 건드리지 않는다.

**채널 ID 해석.** Data API의 `forHandle` 룩업으로 @handle을 canonical UC id로 해석한다. Lambda IP에서도
동작한다. 워치 페이지 HTML 스크레이프는 데이터센터 IP에서 차단되니, API가 해석에 실패할 때만(예: @handle이
없는 URL) 폴백으로 쓴다.

**자막 언어 폴백.** 설정 언어(`transcript_language`)를 먼저 시도하고, 없으면 영상에 존재하는 임의 자막으로
폴백한다. 비영어 채널이나 자동 생성 트랙이 그렇다. 'en' 트랙이 없다는 이유만으로 본문이 빈 채로 떨어져서는 안
된다. 라이브(데이터센터 IP) fetch는 차단되어 본문이 description으로 떨어지니 S3 park 파일을 선호한다.

**실패 신호.** API 거부(쿼터 소진, 키 폐기 등 non-200), 깨진 JSON, 채널 ID 해석 실패는 빈 결과가 아니라
예외다. 채널 하나의 실패는 허용하고, 모든 채널이 실패할 때만 FAILED로 승격한다.

**RSS 폴백도 공용 피드 경로를 탄다.** `fetch_feed_with_retry` +
`parse_feed_entries(limit=_CHANNEL_FETCH_DEPTH)`다. 타임아웃과 재시도, bozo/빈 문서 검사가 RSS와 같다. video id는 `yt_videoid`(없으면 watch 링크)에서 읽어 item id로 쓰고, 읽히지 않는 항목은 버린다. 자막 조회와
canonical watch URL이 모두 그 id에서 나오므로 해시 폴백 id는 하류에서 쓸모없다.

**헬스 임계는 수집기 config가 소유한다.** `record_run_health`와 `flag_degraded_park`는 임계를 인자로 받지 않고
`self.config`(`BaseCollectorConfig`)의 `error_rate_threshold`와 `empty_rate_threshold`를 읽는다. 호출 지점이
임계를 빠뜨려 config 값이 조용히 무시되는 모양 자체가 없다.

**다양성.** `max_videos_per_channel`(코드 기본값과 템플릿은 3, 운영 config는 1)로 고빈도 채널이 후보 풀을 독점하지 못하게 한다.

### 4.7 WebSearch (`web_search.py`)

LLM 쿼리 정제(`RefineQueryPrompt`)를 곁들인 Tavily 검색이다. `exclude_domains`(웹 검색 config에 하나)는 broad와
정제 검색 모두에 Tavily의 `exclude_domains`로 넘어가 콘텐츠 팜을 결과에서 뺀다. 정제 쿼리는 어떤 트렌드
검색에도 속하지 않아서 트렌드별 필드로는 그 검색에 닿지 않는다. 운영 config는 최근 게시 URL 33개 중 6개를
차지한 `finance.biggo.com`과 `tech-insider.org`를 제외한다.

날짜 파싱에 주의점이 있다. `_parse_date`는 tz 없는 ISO 문자열을 UTC로 정규화한다. naive datetime을 tz-aware
cutoff와 비교하면 TypeError가 나고, 그 예외는 결과를 조용히 드롭한다. 날짜만 있는 값(`2026-07-10`)은
**다이제스트 시간대(`aws.timezone`) 기준 그날 23:59:59**로 읽는다. 그 순간은 그날 윈도(그날 자정 KST에서 끝난다)와
다음 날 윈도 양쪽에 들어가므로, 그날 실행 뒤에 발행된 결과도 다음 실행에서 실리고 재수록은 published-URL 원장이 막는다.
00:00으로 읽으면 다음 날 윈도보다 앞이 되어 어느 실행에도 실리지 않고, 23:59:59 UTC로 읽으면 그날 윈도 밖이라
당일 뉴스가 하루 늦게 실린다. RFC 2822 폴백에서 `-0000`처럼 시간대가 없는 값은 호스트 로컬 시간이 아니라
UTC로 읽는다.

`_deduplicate`는 aggregator와 중복처럼 보이지만 역할이 다르다. 쿼리 정제 LLM 호출 전에 broad 결과를 줄인다.

정제 프롬프트는 오늘 날짜(`reference_time` 기준)를 받고, broad 결과 제목이 아직 다루지 않은 지난 하루의 주요
AI/ML 소식을 찾는 쿼리를 만든다. "이미 찾은 기사를 보완하라"와 "고유명사를 쓰라"였던 이전 지시는 받은 제목을 그대로
되풀이하거나("Reflection AI …"가 사흘 연속) 사전지식의 철 지난 버전과 연도("OpenAI new model release December
2025", "best open source local LLM 2024")를 박아, 10-11에는 정제 결과 12개 중 7개가 이미 풀에 있던 한 이야기였다.
`industry` 트렌드 쿼리도 금융 기사를 부르던 "AI startup funding acquisition strategy" 대신 "AI lab acquisition or
major funding news"다.

쿼리 재시도는 재시도로 결과가 바뀔 수 있는 실패만 대상으로 한다. 타임아웃(`asyncio`와 Tavily 양쪽)과 전송
계층 오류(`httpx.HTTPError`)다. 폐기된 키나 401, 사용량 초과는 판정이라 첫 응답에서 끝낸다. `retry_on`을
`(Exception,)`으로 두면 그런 판정에도 쿼리마다 세 번씩 시도해 약 15초를 태우고, 헬스가 DEGRADED를 보고하는
시점이 그만큼 늦어진다. RSS와 YouTube의 재시도 조건이 좁게 잡혀 있는 것도 같은 이유다.

### 4.8 동시 실행과 헬스

`gather_collector_results(tasks, labels, raise_if_all_failed=False)`가 작업을 동시 실행하고 작업별 예외를
로깅한 뒤 건너뛴다. 반환값이 평탄한 리스트가 아니라 `CollectorRunResult(items, total, failed, empty)`인 것이
핵심이다. 몇 개의 입력이 응답했는지가 항목과 함께 흘러야 한다. 항목 수만으로는 40개 피드 중 2개만 답한 실행과
건강한 실행을 구분할 수 없다. `raise_if_all_failed=True`면(RSS와 YouTube, web_search 수집기가 쓴다. Reddit은 요청이
하나라 그 요청의 실패가 곧 소스 실패다) 모든 작업이 실패했을 때만 RuntimeError를 올려 소스가 EMPTY가 아니라
FAILED로 분류되게 한다. 부분 실패는 그대로 허용한다.

`BaseCollector.record_run_health(total, failed, empty, what, hint)`와
`flag_degraded_park(parked, ...)`는 임계 중 하나라도 걸리면 `degraded_detail`을 세우고, 같은 카운트를
`run_meta`(park-meta 키)에 남겨 sync 스크립트가 항목과 함께 park하게 한다. 모든 수집기가 이 한 구현을 쓴다.
소스별 사본을 두면 한 소스만 반쪽 상태를 보고하고 나머지는 침묵한다. 이 메서드들은 보고만 하고 어떤 항목도
필터링하지 않는다.

`pipeline.runner.run_collectors_with_health()`는 수집기를 동시에 돌리고 `HealthReport`([§8](#8-헬스-체크와-알림)
참조)를 반환한다.

**공유 스레드 풀은 하나다.** 이 함수가 시작할 때 `collectors.thread_pool_max_workers`(기본 16) 크기의
`ThreadPoolExecutor`를 루프의 기본 executor로 설치한다. 수집기별 `max_concurrency`는 자기 fan-out만 묶는데
`asyncio.to_thread` 호출은 전부 같은 기본 executor로 들어가니, 그 풀을 명시하지 않으면 실효 상한이 stdlib 기본값
(2 vCPU Lambda에서 6개)이 된다. 그러면 한 소스의 작업이 다른 소스 뒤에 큐잉된 채로 자기 타임아웃을 소진한다.
`asyncio.run`이 루프와 함께 풀을 정리한다.

**수집 윈도는 순수 함수가 소유한다.** `pipeline.runner.resolve_digest_window(config, date_arg)`가 다이제스트
날짜와 reference time을 함께 돌려준다. reference time은 `config.aws.timezone` 기준으로 다이제스트 날짜
**다음날** 자정이고, 모든 수집기의 `lookback_hours`가 거기서부터 거슬러 센다. 그래서 19:00 KST 실행도 그날
전체를 보고, `--date`로 지난 날을 재실행하면 지금 끝나는 윈도가 아니라 그날이 가졌던 윈도를 본다. CLI와
다이제스트 Lambda가 이 한 함수를 공유한다. 그날 무엇이 보이는지를 정하는 계산이니 사본이 갈라지면 두 경로가
서로 다른 다이제스트를 만든다.

앵커가 다음날 자정이라는 사실이 `lookback_hours`의 하한을 정한다. cron은 그 자정보다 앞서 수집하니, 수집이
끝난 뒤부터 자정까지 발행된 글을 다음 실행이 보려면 윈도가 직전 실행 시각까지 닿아야 한다. 최소값은
`48 - 실행 시각(로컬)`이고([§3.1](#31-collectors)), `Config`가 config 로드에서 검증한다. 19:00 실행에서 24는
매일 19:00–24:00 KST의 다섯 시간을 어느 다이제스트에도 싣지 못한다.

## 5. 파이프라인

### 5.1 집계기 (`aggregator.py`)

URL로 먼저, 그다음 정규화된 제목으로 중복을 제거한다. 제목 정규화(`normalize_title`)는 앞머리의
`RT 이름:`을 떼므로 셀프 리트윗이 원문과 겹친다.

**URL 정규화.** 모듈 레벨 `normalize_url`이 scheme과 host case, trailing slash, 추적 파라미터(`utm_*`,
`fbclid`, `ref` 등), fragment를 접어서 같은 기사가 https 기준으로 일치하게 만든다. cross-day 원장도 같은
정규형을 공유한다.

**cross-day dedup.** `aggregate(items, exclude_urls=...)`에서 호출자(`pipeline.runner.run_pipeline`)가 넘긴 정규화 URL
집합(최근 발행 기사들)을 랭킹 이전에 제외한다. 같은 스토리가 며칠 간격으로 재요약되지 않게 하는 것이
목적이고, 부수적으로 랭커 토큰도 절약된다([§7.3](#73-cross-day-dedup-히스토리-sharedhistory_storepy) 참조). 다만 핀
항목(`--pin-url`)은 URL과 제목 dedup을 모두 우회한다. 사용자가 오늘 명시적으로 요청한 URL이니, 최근
발행됐거나 제목이 겹쳐도 살아남아 랭커의 핀 복구 단계까지 도달해야 한다.

**중복이 걸리면 어느 쪽을 남기는가.** 먼저 온 항목을 무조건 유지하지 않고 품질을 기준으로 승자를 고른다
(`_pick_survivor`: 핀이 먼저고, 그다음 본문이 더 긴 쪽, 그다음 먼저 온 쪽). 얇은 Reddit `.rss` 링크 포스트가
같은 기사의 전문 RSS나 웹 항목을 수집 순서만으로 밀어내면 안 된다. 랭커와 다이제스트가 읽는 것은
승자의 `text`다. 동점이면 먼저 온 것을 유지해 결정성을 지킨다. 패자의 메타데이터는 승자에 없는 키만 채우니
origin이나 engagement 값이 덮어써지지 않는다.

### 5.2 랭커 (`ranker.py`)

**입력 포맷.** 항목을 engagement와 origin을 포함해 포맷팅한다. `Origin` 줄은 `format_origin_label`이 만들고
web-search 항목도 포함한다. URL 호스트를 쓰며 `resolve_origin_key`와 똑같이 `netloc`에서 `www.`만 제거한다.
도메인 권위 표도, PSL 로직도 없다. 프롬프트가 "Source Authority"를 채점하는데 web 항목만 매체명이 빠져 있어서
콘텐츠 팜과 통신사 기사가 구별되지 않았기 때문이다. 반면 Tavily의 relevance score는 의도적으로 넣지 않는다.
검색 적합도는 소스 권위가 아니고, 검증되지 않은 신호를 랭킹 입력에 더하는 일이 된다.

**점수 산출.** Claude Opus 5.5로 `RankingPrompt`를 병렬 배치 호출하고 JSON 점수를 파싱한다. 프롬프트는 점수와
근거만 묻는다. 카테고리나 동일 사건 중복 페널티는 묻지 않는다. 같은 사건은 에디터가 병합하고, 배치 경계에 따라
달라지는 페널티는 배치가 다른 두 항목을 둘 다 통과시키곤 했다.

**배치 재시도와 fan-out 상한.** 각 배치의 Converse 호출은 `retry_async`로 재시도한다(`ranking_max_retries`
기본 3, `ranking_retry_backoff_sec` 기본 5초 선형 백오프). 재시도 없이 `[]`로 삼키면 한 번의 스로틀이나 일시적
5xx가 경고 한 줄만 남기고 후보 40건을 그날 풀에서 지운다. 동시에 in-flight 배치 수를
`ranking_max_concurrency`(기본 4)로 묶어 큰 날에 스스로 ThrottlingException을 유발하지 않게 한다. 세마포어는
`rank()` 안, 실행 중인 루프에서 만든다.

**전면 실패만 승격한다.** 재시도까지 실패한 배치가 있으면 ERROR로 남기고 사라진 후보 수까지 로그에 적은 뒤
나머지 배치 결과로 계속 진행한다. 모든 배치가 실패했을 때만 RuntimeError로 올려 실행이 FAILED로 잡히게 한다.
`gather_collector_results(raise_if_all_failed=True)`와 같은 규칙이다. 같은 판정을
`ContentRanker.health`(`RankingHealth`)에 남겨 `run_pipeline`이 `DigestResult.ranking_health`로 실어 보내고,
다이제스트 Lambda가 파이프라인 이후에 별도 SNS 알림으로 게시한다. 배치 하나가 사라진 다이제스트도 겉보기엔
완전히 정상이기 때문이다. 파싱 실패, 다시 말해 모델이 JSON이 아닌 문자열을 반환한 경우는 빈 결과로 모두 실패한
것으로 집계한다. 첫 응답과 커버리지 재질의가 둘 다 파싱되지 않으면 그 배치를 RuntimeError로 올린다. 여기서
`[]`를 반환하면 `failures`가 비고 `items_lost`가 0이라 `RankingHealth`가 깨끗해 보이고, 후보 한 배치가 사라진
날에도 알림이 전부 침묵한다. 핀 복구 경로는 그대로 돌아 그 배치의 핀을 `min_score`로 되살린다.

**응답의 item_id는 유일하게 접는다(`_parse_rankings`).** 입력 id당 `RankedItem`은 최대 하나이고 첫 항목이
이긴다. 모델이 한 id를 반복하고 다른 하나를 빠뜨리는 일이 실무에서 같이 오는데, 접지 않으면
`len(ranked) == len(items)`가 되어 커버리지가 1.0으로 읽힌다. 재질의가 뜨지 않고, `items_scored`가 실제 풀보다
커지고, 에디터가 같은 스토리를 두 번 받는다.

**빠진 id는 한 번 다시 묻는다.** 모델이 배치의 일부 id를 빠뜨리면 그 id만 모아 한 번 재질의한다. 비율 임계를
두지 않는다. 40개 중 1개가 빠지는 일이 운영 30회 중 5회 있었고, 그 1건은 재질의가 없으면 경고 한 줄과 함께
사라진다. 재질의가 실패하거나 여전히 모자라도 첫 결과는 그대로 남는다.

**배치 구성(`_make_batches`).** 항목 수(`ranking_batch_size`)만 상한으로 쓴다. `ranking_batch_size` x
`item_text_max_tokens`(40 x 10k)는 Opus 5.5의 1M 창에 한참 못 미치므로 토큰 예산 상한은 걸릴 일이 없었고,
항목마다 CountTokens 호출만 늘렸다. 수 상한은 배치마다 고르게 나눈다(`ceil(n / ceil(n / ranking_batch_size))`,
41건이면 20/21). 꽉 채워 자르면 1건짜리 꼬리 배치가 비교 대상 없이 혼자 채점된다. 항목은 `item_id`의 **해시**
순으로 정렬해 배치에 넣는다. 배치 구성이 결정론적이면서 소스와 문자 체계가 섞인다. `item_id` 자체는 피드
GUID(RSS와 X는 기사 URL, 그 밖은 글이나 영상 id)라서 그대로 정렬하면 GeekNews 33건이 한 배치에, X 전부가
다른 배치에 몰렸다.

**최근 다이제스트 제목을 랭커도 본다.** `rank(..., recent_titles=...)`는 러너가 에디터에게 주는 것과 같은
`_recent_story_titles(snapshots)`(`published_url_ttl_days` 창의 출고 스토리 제목)를 받아 RankingPrompt의
*Already in recent digests* 블록에 넣는다(`format_recent_titles`). Novelty 기준은 목록에 있는 스토리를 실질적
새 진전이 없으면 재탕으로 본다. URL 원장은 정확한 URL만 맞추므로, 다른 매체나 주간 정리로 돌아온 같은 사건은
랭커가 볼 근거가 없어 0.67–0.85를 받고 풀에 들어왔다. 2026-10-11 풀에서 기존 흐름은 1–4일 전에 나간 스토리 3개를
다시 내보냈고, 제목을 랭커에 주자 그 3개가 풀에서 빠졌다. 정보일 뿐 필터가 아니다.

**후보 풀과 품질 우선 선정.** `rank(items, select_count)`는 `top_n + digest_candidate_buffer`(기본 5 + 3 = 8)개의
후보 풀을 고른다. 소스 슬롯과 origin 상한은 이 **풀**을 채우는 데 쓰여서, 각 소스의 `min_score` 이상 최선
항목이 에디터 앞에 놓인다. 실제로 나가는 5개는 에디터가 풀에서 가장 강한 서로 다른 이야기를 골라 정한다.
보장 자리를 출고분에 강제하지 않는 이유는 두 가지다. 운영 30회 중 14회 에디터가 약한 보장 항목을 더 강한
후보로 바꿨고, 강제하면 0.62짜리 링크뿐인 Reddit 글이 0.70짜리 Anthropic 보고서를 밀어냈다. 다양성은 풀이
보장하고 품질은 에디터가 정한다.

임계 아래 항목을 구제하는 장치는 두지 않는다(운영에서 발동한 3회 모두 루브릭이 저가치라 부르는 0.6 미만
항목을 넣었다). 슬롯 보유 소스에 임계 이상 항목이 없으면 그 자리는 fill 패스가 다음 항목으로 채운다.

**선정과 다양성 (`_apply_source_slots`).** 먼저 `source_slots`로 소스별 기본 슬롯을 config 순서대로 채운다.
나머지 풀은 점수 순으로 채우되 `max_per_origin`으로 하나의 origin 키가 차지하는 항목 수를 제한한다(①). 그래도
모자라면 origin 캡을 풀고 채운다(②). 서로 다른 origin이 너무 적은 조용한 날에도 풀이 짧아지지 않게 하려는 것이다.
수집기(소스) 단위 상한은 없다. "rss"는 13개 매체를 묶고, 매체 단위 다양성은 `max_per_origin`이 이미 지킨다.
수집기당 2개 상한을 두었을 때 16일 중 9일, 0.72–0.80 항목이 풀 밖에 남고 0.60–0.68 항목이 들어갔다.

origin은 `resolve_origin_key`로 해석한다. YouTube는 channel_url, Reddit은 subreddit, RSS는 feed_url, X는
author, Web은 URL 호스트(`urlparse().netloc`에서 `www.` 제거)다. PSL이나 등록가능도메인 휴리스틱을 쓰지 않으니
서브도메인은 별개 origin이다. 다섯 소스 전부에 키가 있어야 한다. 한 소스만 키가 비면 그 소스의 항목이 origin
캡을 전부 우회해서 한 매체가 여러 슬롯을 차지한다.

**`source_slots: {}`도 origin 캡을 잃지 않는다.** 슬롯이 비었을 때 `above_threshold[:limit]`로 조기 반환하면
`max_per_origin`이 슬롯과 함께 사라져서 한 피드가 그날 전부를 차지할 수 있다. origin 캡은 바로 그것을 막으려고
있다. 그래서 보장 슬롯 패스만 (반복할 것이 없어서) 비고, fill 패스는 그대로 돈다.

**저자별 가산 보정은 없다.** origin별 가중 nudge를 두지 않는다. nudge 0.1에서도 최대 변화폭이 0.02인데 실제
Opus 점수는 0.02–0.05 단위로 양자화되어 나온다(관측된 한 실행의 0.82/0.78/0.75/0.72/0.70). 거의 정확한 동점만
가를 수 있는 노브이면서, 대가로 손으로 관리하는 핸들 allowlist와 그 목록을 유지하기 위해서만 존재하는
unmatched-key 경고 경로를 요구한다. 권위는 프롬프트의 *Source Authority* 기준이 이미 판단한다. 같은 신호를 두 곳에서 판단하지 않는다.

**핀 항목도 캡에 계수한다.** 핀은 `rank()`가 앞에 붙이고 이 fill을 통과하지 않으니, 카운터에 반영하지 않으면
핀과 같은 origin의 항목이 나란히 실린다. 그래서 카운터를 핀으로 먼저 채운 뒤 나머지를 채우고, 캡 때문에 미달이
되면 마지막 완화 패스가 풀 크기까지 메운다.

### 5.3 트렌드 트래커 (`trend_tracker.py`)

구조화된 `trends.json`을 유지한다. slug id와 증거 리스트를 가진 `Trend` 객체들이다. 날짜 기반 상태
(active, cooling), 보존 기간이 지나면 삭제, momentum 감쇠 랭킹, active 캡이 전부 여기 있고, 자세한 규칙은
[§7.1](#71-트렌드--구조화-trendsjson)에 정리했다.

### 5.4 다이제스트 생성기 (`digest_generator.py`)

Claude Sonnet 5.5로 `DigestPrompt`를 호출해 구조화 `DigestContent`를 얻는다(모델 선택의 측정은 [§6](#모델-팩토리)). Pydantic 모델이며 `lead`, 코드가
항상 1로 고정하는 `headline_index`, 그리고 각각 title, url, source_tag, metrics, body, implication을 가진
`items[]`로 이루어진다.

**항목 스펙은 메뉴가 아니다.** `implication`의 요구는 하나다. 업계 일반론이 아니라 **이 스토리에** 독자가 반대할 수
있는 입장을 취하는 한국어 한 문장이고, 조건절과 `~수 있다`는 금지다(32일 160건 중 34건이 "시대·결국·언제나" 류의
업계 경구로 흘렀다). 형태를 여러 개 제시하면(단정, 질문, 반증 가능한 예측, 결정적 수치 하나,
반박 초대 같은 목록) 모델은 그중 가장 안전한 것을 고른다. 그런 목록을 준 상태로 보관된 20건에서 10건이
'~다면/~라면 ... ~수 있다' 형태의 조건부 헤지였다. `body`도 같은 이유로 '2-3문장'이 아니라 '스토리에 필요한
만큼(대개 두 문장)'이다. 고정 문장 수를 요구하면 20건 중 17건이 정확히 세 문장이 되고 가운데 문장이 매번
'핵심은...'으로 시작해서, 다섯 항목이 한 템플릿을 다시 채운 것처럼 읽힌다. 두 요구 모두 스펙을 **줄이는**
방향이다. 이 프롬프트에 규칙을 더하는 방향은 출력을 나쁘게 만들고, 짧아진 항목은 Threads 500자 캡에도 유리하다.

**LLM은 산문만 쓴다.** lead와 body, implication만 모델이 작성하고 source tag와 metrics는
코드(`_fill_source_metadata`)가 URL로 매칭해 채운다. 매칭 키는 집계기의 `normalize_url`이다. 에디터가 URL을
되쓸 때 생긴 trailing slash나 http에서 https로의 차이, utm 파라미터 때문에 소스 줄이 통째로 사라지면 안 되기
때문이다. URL이 이미 동일하면 동작에 변화가 없다.

**매칭되지 않는 항목은 독자에게 가지 않는다.** 어떤 후보와도 매칭되지 않는 항목은 에디터가 만들어낸
스토리이거나 URL을 정규화로도 복구 못 할 만큼 망친 것이다. 드롭하고 ERROR로 남긴다. 통과시키면 자기 host가
태그로 붙어 독자에게 실리고, 게시된 URL 원장에도 기록되어 TTL 윈도 내내 **진짜** 기사를 억제한다. 그것이
헤드라인이면 `DigestContentError`로 emission 자체를 거부한다. lead와 데일리 비주얼이 둘 다 `items[0]`을 두고
쓰이므로 건질 것이 없고, 호출자의 재질의가 곧 해결책이다(그래서 트림과 매칭이 재시도 클로저 안에 있다).
이 함수는 살아남은 스토리별 소스 항목을 반환해서 그라운딩 검사가 실제로 실린 것만 보게 한다.

**파싱 견고성 (`_parse_content`).** LLM JSON은 `parse_json_from_llm_output`으로 파싱한다(`strict=False`로
두어 문자열 값 안의 raw 제어문자를 허용한다). 그리고 items를 개별 검증해, 한 항목이 malformed여도 그 항목만
스킵하고 나머지는 유지한다. url이나 body가 빠진 경우가 그렇다. 전체를 0-item으로 무너뜨리지 않는 것이다.

단 `items[0]`(헤드라인)이 검증에 실패하거나 lead가 없거나 JSON이 통째로 깨지면 `DigestContentError`를
raise한다. 스토리 없는 minimal 폴백을 두지 않는다. 그런 다이제스트는 정상 게시로 통과해 버린다. 호출자가 `digest_max_retries`만큼 재질의하고, 계속 실패하면 실행 자체가 실패로
남아 깨진 게시물이 나가지 않는다.

**JSON 키 순서가 load-bearing이다.** 프롬프트는 `items`를 먼저, `lead`를 마지막에 요청한다. 이미 쓴 스토리에
대한 논평으로 lead를 쓰게 만드는 장치이고 측정된 효과가 있다. 헤드라인 reply와의 단어 겹침이 0.21–0.41에서
0.03–0.21로 떨어졌다. lead 스펙은 같은 이유로, `items[0]`의 implication이 이미 한 말이 아닌 논지(트렌드 재료에서 끌어온
더 넓은 흐름)를 세우라고 한다. "그것이 무엇을 뜻하는가"만 요구하던 때는 lead와 바로 아래 reply 1이 같은 논지를 연달아
말했다(렌더한 14일 중 약 10일). `headline_index`는 프롬프트에서 아예 빼고 코드가 1로 고정한다. 그래야 에디터가 lead와
비주얼을 서로 다른 스토리로 가리킬 수 없다.

⚠️ `DigestContent`의 필드 선언 순서(lead, headline_index, items)에 맞춰 프롬프트 키 순서를 되돌리지 말 것.
모델은 쓰는 순서대로 사고하니 그 '정리'는 겹침 회귀다.

**예산은 코드가 계산해서 넘긴다.** 항목 산문 예산은 추정치가 아니라 코드가 소유한 고정 파트에서 파생한다
(`_item_prose_budget`은 `THREADS_MAX_POST_CHARS`에서 그 항목의 `URL + 소스 줄 + 빈 줄 구분자`, 즉
`threads_item_overhead_chars`를 뺀 값이다). 예산은 **후보마다 따로** 계산해서 `PROSE BUDGET`이라는 코드 소유
필드로 붙여 보낸다. 후보 전체의 최악 오버헤드로 하나의 숫자를 뽑아 프롬프트
문장에 박으면 URL이 짧은 항목도 풀에서 가장 긴 URL 기준 예산을 받아, 닿지도 않을 상한 때문에 마지막 문장을 잃는다.

이 숫자에는 에디터가 쓰는 title도 들어간다. body와 implication만 세면 한국어 제목이 예산 밖에서 소비되고, 표본
95건 중 5건이 마지막 문장을 잃는다. 이 계산 위에 덧씌우는 고정 상한은 없다(URL이 짧은 항목은 약 440자를
받는다). lead도 예산을 받는다(`_lead_budget`은 500에서 코드가 붙이는 카운트다운 개그와 그 앞의 빈 줄을 뺀
값이다).

캡과 계산에 쓰는 채널 무관 프리미티브는 `shared`에 둔다. `shared/constants.py`의
`THREADS_MAX_POST_CHARS`(500)와 `THREADS_POST_SEPARATOR`, `shared/formatting.py`의 `split_sentences`,
`truncate_at_word`, `strip_markup`, `threads_item_overhead_chars`다. 예산 계산이 `output/renderers.py`의
private 헬퍼를 부르면 파이프라인이 출력 채널의 내부 이름에 의존하게 되고, 500이라는 같은 숫자가 렌더러와
게시기에 따로 박힌다.

**target_count와 recent_leads, recent_titles.** `generate(..., recent_leads=..., recent_titles=...)`으로 세
가지를 함께 넣는다.

- `target_count`는 기본적으로 `min(top_n, 후보수)`다. 다만 사용자가 `top_n`보다 많은 URL을 핀하면 헤드라인
  1개와 전체 핀을 담도록 상향해서 핀도 헤드라인도 트림에 밀리지 않게 한다. 에디터는 후보 풀에서 가장 강한
  `target_count`개의 distinct 스토리를 고르고 같은 사건은 병합하되, 모델이 초과 emit하면 코드가 트림한다
  (`_trim_keeping_pinned`는 결정론적 상한이고, items[0] 헤드라인을 우선 보존한 뒤 나머지 슬롯에 핀을 보존한다).
- `recent_leads`는 최근 며칠의 lead를 "이 논지는 되풀이하지 말라"는 뜻으로 보여준다. 특정 문구를 금지하지 않고
  일반화하는 방식이다. 각 lead의 **마지막** 문장만 보여준다. lead는 첫 문장에서 스토리 이름을 대고 마지막에
  논지를 놓으니, 되풀이하면 안 되는 논지가 마지막 문장에 있다(스토리 반복은 `recent_titles`가 다룬다). 첫 문장을
  보여주던 때는 "누가 그 이득을 가져가나"라는 같은 논지가 lead 32개 중 11개를 닫았다. 잘라내기는 포맷
  시점(`_format_recent_leads`)에 일어나니 전문으로 저장된 기존 이력도 그대로 동작한다.
  `RECENT_LEAD_PREVIEW_CHARS`는 문장 경계가 없는 산문용 백스톱이다.
- `recent_titles`는 TTL 창(`published_url_ttl_days`) 안 **모든** 스냅샷의 스토리 제목을 최신순으로 중복 없이 모은
  목록이다. 주간 요약 글이 이미 실린 이야기를 새 URL로 다시 가져오면 URL 원장은 그것을 알아보지 못하고, 직전
  하루치 제목만 보여주던 때는 2026-10-11 다섯 이야기 중 셋이 1~4일 전에 실린 것이었다. 프레이밍은 한 줄이다.
  실질적 새 진전이 있을 때만 다시 싣고, 그때는 지난번 이후 무엇이 새로운지 본문에 말하라고 한다. 같은 목록을
  랭커도 받는다([§5.2](#52-랭커-rankerpy)). 임계나 유사도 휴리스틱은 없다. 실제로 재발행을 막는 것은 여전히 URL 원장이다. `pipeline.runner.run_pipeline`이
  cross-day dedup이 이미 가져온 스냅샷에서 뽑으니 추가 호출도 없다.

**시스템 오브 레코드.** `render_digest_text`가 구조화 콘텐츠를 평문 산문으로 렌더해 `digest_text`를 만들고,
트렌드 분류기와 AgentCore 스냅샷이 그것을 쓴다.

**그라운딩(옵션, `enable_grounding_check`).** 산문 필드의 구체적 주장을 소스 항목과 코드가 산출한 트렌드
사실에 대조해, 근거 없는 부분만 외과적으로 수정한다. 대조 대상은 실제로 실린 스토리의 소스뿐이다
(`_fill_source_metadata`가 돌려주는 매칭 결과). 랭킹 후보 전체를 이어 붙이면 에디터가 고르지 않은 후보에만 있는
숫자나 제품명이 근거 있음으로 통과한다. 만들어낸 구체 사실을 잡으려고 둔 한 번뿐인 검사가 그것을 놓치고,
동시에 풀의 잉여 입력 토큰(항목당 `item_text_max_tokens`)을 크리티컬 패스에 싣는다. 수정본은 라벨 줄
(`LEAD:`, `ITEM N BODY:`, `ITEM N IMPLICATION:`)로 되돌려 붙이는데, 마커가 없거나 **비어 있는** 필드는 원래 값을
유지한다. 빈 마커를 그대로 적용하면 빈 필드가 게시된다.

**에디터의 한국어를 정규식으로 검사하지 않는다.** 산문 규칙은 프롬프트가 읽는 `KOREAN_STYLE_RULES`에만 있고,
그것을 코드로 다시 판정하는 린트 계층은 두지 않는다. 이유는 두 겹이다.

첫째, 한국어 문법은 정규식으로 표현되지 않는다. `[가-힣]다,`는 '다'를 종결어미로 가정하지만 '소다'와 '바다'는
명사이고, 인용 조사는 손으로 열거해야 한다. 실제로 재보면 그 부정확이 통계로 나타난다. 보관된 코퍼스(다이제스트
14개, 산문 44필드)에 규칙 넷을 대고 잡은 참 양성은 **1건**이다. lead의 구체성 검사는 `DigestPrompt`가 헤드라인을
이름으로 부르라고 지시한 바로 그 형태를 물었고, 숫자 반복 검사는 소수점 아래를 잘라 52.2와 52.9를 같은 값으로
읽었고, em-dash 패턴은 다이제스트 4개 중 3개에서 자연스러운 한국어를 물었다(다이제스트 보이스가 Gruber 계열이고
그 서명이 em-dash다).

둘째, 걸렸을 때 할 수 있는 일이 없다. 재질의는 같은 `prompt_vars`를 바이트 단위로 동일하게 다시 보내므로 무엇이
걸렸는지 모델에 전달되지 않는다. 옳게 발화해도 약 50k 토큰짜리 다이제스트 호출로 주사위를 다시 굴리는 것뿐이다.
품질 문제가 재발한다면 필요한 것은 정규식이 아니라 무엇이 틀렸는지 실어보내는 재질의다.

운영 30회 중 18회 한 건 이상을 수정했고(원문 '수십만 번'을 '수만 번'으로 쓴 경우, 최고치를 평균으로 쓴 경우
등), 수정안은 '근거 없는 구체값 빼기' 하나다. '추론으로 낮추기'는 두지 않는다. 시사점 규칙이 금지하는
'~수 있다' 류 헤지를 그라운딩이 다시 써 넣게 되기 때문이다. 근거 없는 주장에 "보도에 따르면"을 붙이는 선택지도
두지 않는다. 원문 어디에도 없는 주장이니 출처를 붙이면 지어낸 사실이 거짓 출처를 얻는다.

**`KOREAN_STYLE_RULES`는 "A가 아니라 B"와 "A보다 B" 대비 공식을 피하라고 한다.** 게시된 31개 다이제스트
(09-09~10-10)의 시사점 155개 중 41개(26%)가 그 틀이었다. 보이스 가이던스는 그 틀을 유도하는 문장도, 그대로
복사되는 예시 문구('~로 읽힌다', '보통 이런 구도는')도 담지 않는다. 예시를 주면 모델이 그 문구를 그대로 가져와
lead와 리서치 리포트에 반복했다. "날카로운 해석은 판단으로 표시하라"는 문장도 지웠다. 시사점은 단정하라는 스펙과
충돌해 '~로 읽힌다 / 셈이다 / 모양새다' 헤지를 불렀다(시사점 160건 중 12건, lead 32건 중 5건). 추론을 보도된 사실처럼
쓰지 말라는 원칙은 남는다. "누가 그 이득을 가져가나" 렌즈도 '항상'을 뺐다. 순수 기술 스토리에서는 그 렌즈를
가볍게 두라는 문장과 모순됐고 lead 32건 중 12건이 그 공식이었다. 고유명사는 소스가 음역해도 공식 라틴 표기를
지킨다(Anthropic, OpenAI, SoftBank). "소스가 쓴 대로"였을 때는 한국어 소스가 음역한 날마다 표기가 바뀌어(Anthropic
영문 15일·한글 8일) 같은 이름의 검색이 갈라졌다. lead 스펙은 "3-4 concrete
sentences"다.

`_verify_grounding`은 이 규칙의 예외가 아니다. 그쪽은 주장이 원문에 있는지를 보는 포함 관계 검사여서 코드로
판정되고, 문법 판단이 아니다.

### 5.5 Threads 렌더링 (`output/renderers.py`)

구조화 `DigestContent`와 리서치 리포트를 Threads 게시물로 바꾼다. 마크업은 없다(Threads는 렌더링하지 않는다).

**`render_threads_posts`.** root는 lead이고 항목마다 평탄한 reply 하나를 만든다. 500자 이하로 문장 경계에서
트림하며 title과 소스 줄, URL은 유지한다.

**소스 줄은 자기 블록이 아니라 제목 줄의 괄호 안에 온다**(`제목 (r/LocalLLaMA · 조회 775)`). 조회수는 이모지가
아니라 `조회` 텍스트다(`YOUTUBE_VIEWS_LABEL`). Threads는 이모지를 UTF-8 바이트 수로 500자 한도에 세는데(`▶️`는
6) 렌더러는 `len()`으로 예산을 잡으므로, 꽉 찬 YouTube reply가 넘쳐 떨어질 수 있었다. 매체명
(`Simon Willison's Weblog`)만 한 줄을 차지하면 흘러나온 조각처럼 읽힌다. 괄호는 한국어에서 그 자체로 부가정보
표식이고, 다섯 소스 형태 전부에 라벨 단어 없이 적용된다.

**YouTube 태그는 플랫폼이 아니라 채널명이다.** 다른 네 소스의 태그는 구체적 출처를 지목하는데(서브레딧, 계정,
매체, 호스트) 여기서만 리터럴 `YouTube`를 쓰면 독자가 얻는 정보가 없고, 에디터가 화자를 **제목에** 덧붙여
벌충한다. 수집기가 `author`에 `channelTitle`을 이미 담고 있다. RSS 폴백 경로는 author를 남기지 않으니 그때만
플랫폼 이름이 바닥값으로 남는다.

**implication은 body와 한 단락으로 잇지 않는다.** 빈 줄로 분리해 자기 블록으로 내보낸다. 이어 붙이면 목소리 줄이
본문의 마지막 문장처럼 읽혀서 항목이 착지하는 지점이 사라진다. 이 추가 구분자는
`threads_item_overhead_chars`(에디터에게 알려주는 파생 예산)와 `_item_post_overflows`(트림 카운트)에도 똑같이
반영되어 예산이 정확히 유지된다. 산문이 캡에 걸려 잘린 항목이 있으면 개수만 WARNING으로 남긴다.

**`render_threads_research`.** 딥 리서치 리포트(Markdown)를 Threads용 root와 평탄한 reply chain(각 500자
이하)으로 렌더한다. 에이전트가 `---`만 있는 줄로 자기 게시물 경계를 표시하니 렌더러는 그 경계를 존중하고, 500자를
넘긴 게시물만 문장 경계로 재분할하면서 인용 URL을 보존한다. 구분자가 없으면 문장 패킹으로 폴백한다.
`max_posts`(0보다 큰 값)로 총 게시물 수를 하드 캡하고 초과분은 드롭한다. 게시물 번호는 퍼블리셔가 다시 매기고,
Markdown 마크업은 `strip_markup`으로 제거한다(`[label](url)`은 `label (url)`로 바꾼다).

### 5.6 데일리 비주얼 (`daily_visual.py`)

다이제스트 Lambda가 비동기로 호출하는 비주얼 Lambda에서 돌며, 그날 Threads 게시물의 루트 이미지인 4컷 한글
만화를 만든다.

**무엇을 그리는가.** 헤드라인(`items[0]`)을 그린다. lead와 이미지와 텍스트가 한 스토리로 일치하게 하려는
것이다. 에디터는 무엇을 그릴지가 아니라 어떻게 그릴지만 브리핑한다. 다이제스트 프롬프트의 헤드라인 선정은 중요도
우선이고, 시각화 용이성은 동등하게 중요한 스토리 사이의 tie-break만 한다.

**형식은 화자가 이끄는 2x2 4컷이 기본이다.** `VisualEditorPrompt`는 왼쪽에서 오른쪽, 위에서 아래로 읽는 네 컷
(setup, development, turn, punchline)과 컷마다 짧은 한국어 대사 또는 내레이션 한 줄을 요구한다. 이야기가 분명히
다른 구성(세로 스트립, 단일 컷, 가로 파노라마)으로 더 잘 전달될 때만 바꾸고, 그 구성을 instruction에 적는다. 편집기
출력은 `{skip, use_character, instruction}`이고, `instruction`은 시놉시스 틀을 따른다. LOGLINE(독자가 가져갈 주장 하나, 기사나 다이제스트 관점의 나머지 주장은 넣지 않는다), COMPOSITION, 그리고 컷마다 PANEL n(기승전결): 장소, 인물(실존 인물은 알아볼 특징과 함께, 화자는 외모를 쓰지 않는다. 기준 시트가 정한다), 행동, 대사 한 줄이다. 목표는 기사를 읽지 않은 독자가 한눈에 이해한 뒤 웃는 것이다. 1컷 내레이션이 무슨 일이 있었는지(누가, 무엇을, 핵심 숫자) 그대로 말하고, 2-3컷은 실제 인물·제품·숫자를 비유가 아닌 장면으로 보여 주고, 4컷이 펀치라인이다. 대사와 간판·라벨은 비전문가가 아는 한국어로 쓰고(인명·회사·제품명은 그대로), 벤치마크명·지표·전문용어는 넣지 않는다. 자유 서술 하나였을 때는 다이제스트 관점의 주장 두세 개가 모두 소품이 되어 3-4컷이 붐볐고, 비유(가중치 상자, GPU 마스코트)와 용어(acqui-hire, Pareto)가 많아 기사를 모르면 읽히지 않았다. 같은 헤드라인 세 개를 바꾼 뒤 다시 렌더하자 세 편 모두 한 줄 주장과 한국어 라벨로 정리됐다. 인물은 배경 인물까지 모두 눈과 표정이 있는 얼굴로 그린다(스타일 문구. 세 장 중 두 장에서 이름 없는 인물이 얼굴 없이 나왔다). 회사를 표현할 때는 실존 인물이나 그 회사의 로고·제품을 쓰고 마스코트로 의인화하지 않는다. 아트 디렉터(`VisualSynopsisPrompt`)는 세로 스트립을 `tall`
(848x2544, 4:3 컷 네 개를 한 열로 쌓은 1:3. 2560을 넘는 해상도는 실험적이라 긴 변을 그 아래로 둔다)로, 2x2를
`portrait`로 브리프한다. 2x2(1024x1536)의 컷 하나는 512x768로 세로 스트립(컷당 848x636)보다 작지만, 피드에서 네
컷이 한 화면에 들어오는 쪽을 기본으로 골랐다(세로 스트립은 피드 미리보기에서 펀치라인 컷이 가려질 수 있다). Threads는 종횡비 10:1과 8MB까지 받는다. 날마다 형식과 방향을 돌리던 장치(형식 이력,
최근 방향 넛지, 멀티패널·캐릭터 비율 넛지)는 두지 않는다(측정상 형식을 바꾸지 못했다. 요청한 방향과 실제
방향이 28회 중 8회 일치). `VisualBrief.orientation`은 필수 필드라 브리프가 빠뜨린 값이 조용히 기본값이 되지 않는다.
`VisualBrief.prompt`의 상한은 GPT 이미지 API 한도인 32000자이고 `title`에는 상한이 없다. 4컷 브리프는 4000자를
넘기기도 해서, 그보다 낮은 상한은 좋은 브리프를 검증에서 떨어뜨려 그날 이미지를 잃게 한다(title은 로그 한 줄에만
쓰인다).

아트 디렉터 프롬프트는 이야기에 필요한 실제 단서(인물, 조직 로고와 색, 제품, 장소)로 대상을 알아보게 하라고만
한다. 브랜드를 간판이나 배지 글자로 반드시 적으라는 지시는 두지 않는다. 그 지시는 이야기와 무관한 로고와 간판
문구를 끌어들였다. 아트 디렉터에게 따로 주는 유머 지시도 없다. 농담은 편집기가 쓰고 아트 디렉터는 그것을 옮긴다.

**이미지 문구는 한국어다.** 같은 4컷 브리프를 `gpt-image-2`, `gpt-image-2.5-flare`, `gpt-image-2.5-sunburst`
(모두 high)로 렌더해 비교했다. 세 모델 모두 한국어 문자열을 정확히 그렸다. sunburst를 고른 이유는 로고가
정확하고(flare는 GLM 자리에 OpenAI 로고를 그렸다) 완성도가 가장 높으며 렌더 한 장의 출력 토큰이 약 1.4k(약
$0.05)이기 때문이다. sunburst는 빈 벽을 지시하지 않은 영어 슬로건으로 채우는 경향이 있다. 그래서 아트 디렉터는
포스터와 슬로건, 캡션을 지어내지 않을 뿐 아니라 이미지 프롬프트에도 "벽과 소품에는 다른 글자가 없다"고 적는다.
금지를 아트 디렉터에게만 두었던 렌더는 네 컷 모두 벽에 영어 슬로건이 붙었고, 프롬프트에 적자 사라졌다. 대사 외의
글자(화면, 표지, 라벨)는 크게 몇 단어를 한 번만 쓰고 여러 사본에 반복하지 않는다. 남은 문구 오류는 전부 작은
소품 글자였다(로봇 40대에 붙은 모델명 태그, 서류철 여섯 장의 라벨). 큰 글자 한 곳은 정확히 렌더되고, 작은 글자를
많은 사본에 쓰면 깨지며, 수정 편집도 그것은 고치지 못했다.

**오너 페르소나가 화자다 (`visual_character_enabled`, `visual_character_sheet`,
`visual_character_reference`).** 다이제스트 화자(특이점을 믿는 과학 덕후 기술자)를 그린 재등장 캐릭터가 만화의
시점 인물이다. 나오는 컷에서는 구경만 하지 않고 반응하거나 끼어들거나 펀치라인을 친다. 편집기는 그가 농담을
방해할 때만 `use_character`를 끈다. 플랜이 그 값을 빠뜨리면 화자가 나오는 쪽이 기본이다. 캐릭터 시트(문장)는 instruction에 들어가고, 정면·3/4·측면과 표정 세 가지를
담은 기준 시트 이미지(`assets/persona.jpg`)가 입력 이미지로 함께 간다. 그래서 그 날의 렌더는 `images.generate`가
아니라 `images.edit`이고, 프롬프트 앞에 "첨부 이미지가 화자의 기준 시트이니 그대로 그려라"는 고정 문구
(`REFERENCE_NOTE`)가 붙는다. 문장만으로는 날마다 얼굴과 옷이 달랐다(테스트 렌더 대부분의 청년이 시트의 안경과
카디건을 갖추지 않았다). 시트 이미지를 넣자 세로 4컷 네 편 모두에서 머리 모양, 안경, 투톤 카디건이 시트와
같았다. 시트 파일을 못 읽으면 경고하고 참조 없이 그린다. 시트를 바꾸려면 이 파일만 교체하면 된다.

**실존 인물은 알아보게 그린다.** 아트 디렉터는 실존 인물마다 이름과 한눈에 알아볼 특징(얼굴형, 헤어라인,
안경, 체형, 상징적인 복장)을 적고, 모든 컷에서 같게 유지한다. 시사만평의 관행대로 실존 인물 묘사는 허용한다
(소유자 결정).

**물리적으로 그럴듯해야 한다.** 렌더된 만화가 실제로 틀린 것은 화풍이 아니라 배치였다. 칸막이 책상 줄 위
선반에 의자째 올라앉은 로봇, 허공에 걸린 표지판, 쥐지 않은 채 손에 붙은 소품, 거리와 맞지 않는 크기다. 그래서
`visual_synopsis_style_aesthetic`은 화풍(깔끔한 한국 웹툰체) 다음에 구체적인 규칙을 둔다. 컷마다 눈높이와 바닥면은
하나이고, 모든 인물과 물체는 표면 위에 있거나 실제로 쥐고 있으며, 크기는 거리와 맞고, 떠 있는 표지판이나 불가능한
적층, 관통이 없고, 손가락은 다섯이며, 컷당 인물은 셋 이하에 동작은 하나다. 품질 등급은 `xhigh`다. 같은 브리프를
high, xhigh, 고해상도(1536x2304)로 렌더해 비교하자 xhigh가 손과 소품, 원근이 가장 일관됐다.

**렌더 뒤에 검수하고 로그로 남긴다 (`VisualReviewPrompt`).** 브리프를 쓴 모델(Sonnet 5.5)이 이미지를 보고
휴대폰 화면에서 독자가 알아챌 결함을 눈에 띄는 순서로, 컷 번호와 함께 모두 적는다. 물리·공간, 크기·원근, 해부(빈
얼굴 포함), 브리프와 다른 문구나 지어낸 문구(이야기 속 회사의 로고와 워드마크는 제외), 컷 수와 순서다. 브리프는 각
컷의 뜻과 대사를 알려주는 맥락일 뿐이어서, 연출이 브리프와 다른 것(자세, 몸짓, 소품 위치)은 뜻을 깨지 않는 한
결함이 아니다. 검수에는 원본 해상도 그대로 JPEG로 다시 인코딩한 사본을 보낸다. PNG가 2.5~3.6MB라 Bedrock 이미지
한도 3.75MB에 붙어 있고, 넘으면 검수가 매번 건너뛰어진다(실측 3.6MB → 0.7MB). 이미지는 그린 그대로 게시한다. 찾은
결함만 고치라는 `images.edit`을 한 번 하고 결함이 더 적은 쪽을 쓰던 수정 단계는 뺐다. 12번 중 4번만 결함 수가
줄었고(그것도 1~3개), 편집 한 번이 렌더 한 번만큼 비쌌으며, 검수 자체의 흔들림(한국어 오독, 연출 지적)이 비교에
섞였다. 이해하기 쉬운 만화를 만드는 것은 수정이 아니라 시놉시스였다. 검수는 품질 신호로 2주(2026-10-26까지) 로그만
남기고, 그 뒤 유지 여부를 정한다. 검수 실패와 마감 부족은 검수 없이 게시하는 쪽으로 떨어진다.

**편집 관점을 함께 넘긴다.** 다이제스트의 리드에서 카운트다운 접두를 제거한 것과 헤드라인 항목의
`implication`을 instruction에 정보로 넘긴다. 아트 디렉터가 원본 기사만 보면 표면 사실만 그린다. 다만 일치를
강제하지는 않는다. 이미지가 리드의 논지를 논증해야 한다는 제약은 과하다.

**가드레일 (`visual_guardrails`, 비우면 미적용).** 스타일 지시도 논지 요구도 아니라, 이미지가 하지 말아야 할
두 가지다.

1. 받은 편집 관점의 정서를 뒤집지 말 것. 회의적이거나 경고하는 관점을 축하하는 그림으로 그리지 않는다. "논지를
   논증하라"보다 훨씬 약한 요구다.
2. 기업이나 국가를 인종으로 코딩된 인물로 의인화하지 말 것. 반면 실존 인물을 알아보게 그리는 것은 허용한다.
   시사만평의 표준 관행이다.

이 문구의 인과 효과는 측정되지 않았다. 이미지 생성이 확률적이라 사례당 한 샘플로는 노이즈와 구분되지 않는다.
비용이 0이고 config로 즉시 해제할 수 있다는 근거로 둔다.

렌더 전의 조사 단계는 없다. 에디터가 고른 검색을 돌리던 단계는 논문 검색 API의 429를 30일에 19회 냈고 실패
문자열이 그대로 맥락으로 들어갔다. 에디터는 헤드라인 전문과 다이제스트의 관점을 이미 받는다.

**브리프는 다이제스트 모델이 쓴다.** 편집기와 아트 디렉터 모두 `digest_model`(Sonnet 5.5)을 쓴다. 10-07
헤드라인("OpenAI, AI가 쓴 수학 연구원고 722편 공개")으로 Sonnet 5.5 브리프를 렌더하자 기승전결과 한국어
대사가 정확한 4컷이 나왔다.

**플랜 파싱이 실패하면 그냥 건너뛴다.** 에디터 JSON을 못 읽으면 `{"skip": True}`로 취급한다. 재질의도, 일반 폴백
instruction으로 렌더하는 낭비도 없다.

**플랜의 불리언은 `coerce_bool`로 강제한다.** `skip`과 `use_character`는 모델이 쓴 JSON 값이라
문자열일 수 있고, 맨 truthiness는 `"false"`를 True로 읽는다.

**비주얼 실패가 다이제스트를 삼키지 않는다.** 이미지는 첨부물이고, 이 함수가 Threads의 유일한 게시 경로다.
OpenAI 키가 없거나 에디터 호출이 실패하거나 에디터가 skip하거나 렌더가 실패하는 경우는 모두 `_make_visual`
안에서 흡수되어 `(None, None)`으로 떨어지고, `run()`은 그대로 텍스트만으로 Threads(lead와 스토리별 reply)를
게시한다. OpenAI 키는 `strict=True`로 읽어서 미설정과 SSM 읽기 실패를 구분하되, `SecretUnavailableError`는
`_make_visual` 안에서 잡는다. 엄격한 시크릿 읽기가 텍스트 다이제스트를 비용으로 삼는 일은 없어야 한다.

**instruction 빌더는 하나뿐이다(`_build_instruction`).** 편집 관점과 가드레일, 캐릭터 시트를 붙여 최종 아트
디렉터 instruction을 만드는 부분은 I/O 없는 순수 함수다. `scripts/sample_visual_brief.py`가 프로덕션이 실제로
보내는 문자열을 채점할 수 있어야 하기 때문이고, 테스트가 두 경로의 출력이 같음을 고정한다.

**게시는 `output/digest_delivery.py`의 `DigestPublisher`가 한다.** Threads root와 reply 체인, 그 체인의 멱등
원장, 이미지의 S3 호스팅이 모두 여기 있다. 게시기가 이미지를 **옵셔널 인자**로 받으니 렌더 실패는 구조적으로
텍스트 전용 게시일 뿐이고, 게시 경로를 OpenAI 키 없이 테스트할 수 있다.

**게시할 곳이 없으면 렌더하지 않는다.** 렌더 전에 `DigestPublisher.has_a_destination`에 묻는다. Threads가
꺼져 있거나, 원장에 오늘이 이미 있거나(force가 아닐 때), 스토리가 0건이면 에디터 호출과 렌더 비용을 쓰지 않는다.
스토리 없는 날은 `threads_outcome = ThreadsDelivery(0, 1)`을 남겨 그날의 전달 알림이 no-op이 되지 않게 한다.

**텍스트만 나간 날도 알린다.** `outcome.published and not outcome.with_image`면 `shared/alerts.publish_alert`
경로로 ALERT를 보낸다. 렌더 실패와 키 부재는 그 밖에는 maker 안의 로그 한 줄만 남긴다.

**헤드라인 매핑.** `content.headline_index`(큐레이션 items 기준)를 `normalize_url`로 랭킹 항목에 되매핑한다.
끝내 매칭되지 않으면 랭킹 1위로 떨어뜨리지 않고 큐레이션 헤드라인 자신의 title과 body, implication을 소스로
브리핑해 이미지와 텍스트의 동기화를 지킨다.

전체적으로 best-effort다. 파이프라인을 막지 않으며 실패는 항상 로깅된다.

### 5.7 AGI 카운트다운 인트로 (`shared/formatting.py` `agi_countdown_intro`)

"AGI 등장 N일 전이다" 식의 인트로를 LLM이 아니라 코드가 계산한다(`agi_countdown_date` 기본 `2029-01-01`,
`agi_countdown_template`). D-day 이전에는 카운트다운, D-day 당일과 이후에는 `agi_countdown_after`로
카운트업한다("AGI 등장 예정일 D+N일째, 아직이다"). `agi_countdown_date`가 비면 비활성이다. 템플릿은 운영자가
편집하는 config 문자열이니 `.format()`을 try/except로 감싼다. 잘못된 placeholder나 괄호 같은 오타가 있으면
인트로를 비우고 생성 도중 크래시하지 않는다. 수집과 랭킹과 LLM 비용을 다 쓴 뒤에 죽는 것이 최악이다.

**언제 붙는가.** 다이제스트 생성 시점에 `content.lead`에 붙이며(`digest_generator.generate` →
`place_countdown_intro`) 그 실행의 KST `digest_date`로 계산한다. 인트로가 저장 콘텐츠의 일부가 되니 Threads
root에 함께 나가고, 트렌드 재등장 수치와 같은 시계(같은 날짜)를 쓴다.

**위치 노브(`agi_countdown_position`, 코드 기본값과 배포 설정 모두 `prefix`).** 접두로 두면 Threads root의 첫
줄, 곧 피드 독자가 유일하게 보는 줄을 매일 같은 고정 문장이 차지한다. 실제로 연속 40개 게시물이 동일 문장으로
시작했다. 그럼에도 `prefix`가 기본이고 배포 설정도 그대로다. 카운트다운은 이 계정의 서명이고, 소유자가 첫 줄
도달률 논리보다 알아볼 수 있는 브랜딩을 위에 뒀다. 첫 줄을 그날의 각으로 열고 싶은 배포는 `suffix`로 두면
개그를 문구 그대로 lead의 맺음말로 옮긴다. 위치만 노브로 두고 cadence나 "N일마다 생략", 랜덤은 두지 않는다.
매직 넘버가 된다.

**양 끝에서 제거한다(`editorial_lead`).** 최근 lead 신선도 비교와 비주얼의 편집 관점 전달은 개그를 뺀 각만
봐야 하니, 접두든 접미든 어느 쪽에 붙어 있어도 제거한다. 저장된 lead가 설정 변경 이전 것일 수 있다.

**넘치면 개그가 먼저 나간다.** Threads root가 500자를 넘으면 `_fit_lead`가 코드 소유의 카운트다운 줄을 먼저
버리고 에디터의 산문, 곧 그날의 논지를 지킨다. 개그를 버리는 조건은 마지막 줄이 그 개그임을 식별할 수 있을
때뿐이다. 호출자(`output/digest_delivery.py`)가 계산한 인트로 문자열을 `render_threads_posts(content, countdown)`로 넘겨
비교한다. 마지막 줄을 무조건 버리는 방식은 `prefix` 위치나 개그 비활성 상태에서 진짜 산문을 삭제하니 쓰지
않는다. 식별되지 않으면 앞에서부터 온전한 문장만 남기니 접두 개그는 살아남는다. 트림이 발생하면 WARNING을
남긴다. 에디터가 산문 예산을 넘겼다는 신호다.

### 5.8 Threads 전달 (`output/threads_handler.py`, `enable_threads_post`)

**누가 부르는가.** 다이제스트 Lambda가 아니라 데일리 비주얼 Lambda가 `DigestPublisher.publish`로 게시한다.
Threads 게시물은 이미지 root와 reply chain이 한 세트라 이미지를 만든 쪽이 함께 보내야 한다. 비주얼 Lambda는
항상 돌고 Threads 게시만 `enable_threads_post`가 정한다. 이미지를 끄는 토글은 따로 두지 않는다. 그런 토글은 이
Lambda가 유일한 게시 경로라서 Threads 전달을 알림 없이 통째로 멈췄다.

**흐름(`post_to_threads`).** 이미지 root를 게시하고 스토리당 reply 하나로 평탄한 chain을 만든다. reply는
서로가 아니라 모두 root에 매단다. reply-of-reply로 중첩되면 첫 개만 보인다.

**토픽 태그와 대체 텍스트는 root에만 붙는다.** Threads 게시물 하나는 토픽 태그를 하나만 가질 수 있고(1–50자,
`.`과 `&` 불가), 붙은 게시물은 그 토픽 피드와 태그 검색에 들어간다. root가 `pipeline.threads_topic_tag`(기본 `AI`,
config가 형식을 검증한다)를 싣고 reply에는 붙이지 않는다. 다섯 reply에 같은 태그를 달면 스팸처럼 읽힌다. 리서치
리포트의 root도 같은 태그를 쓴다. 만화 root는 `alt_text`(최대 1000자)로 `VisualBrief.alt_text`를 보낸다. 아트
디렉터가 쓰는 한국어 설명으로 컷마다 한 문장과 대사를 담는다. 그러지 않으면 만화의 한국어 대사가 픽셀로만 남아
스크린 리더가 읽을 것이 없다. 검색 효과는 Meta가 문서화하지 않았으니 접근성 개선이다.

**이미지 호스팅.** Threads는 바이트 업로드를 받지 않고 공개 URL만 fetch하니, PNG를 S3에 올리고 단기 presigned
URL을 Meta에 한 번 넘긴다(`_upload_image_for_hosting`).

**인덱싱 지연.** 방금 게시된 이미지 root는 곧바로 reply 대상이 되지 못해서 Meta가 "media not found"(code 24 /
subcode 4279009)를 반환할 수 있다. reply마다 create-container를 하나의 공유 예산
(`THREADS_INDEXING_BUDGET_SEC`, 약 270초) 안에서 재시도한다. 비주얼 Lambda의 15분 타임아웃이 총량을 bound한다.

**root 준비 상태를 먼저 폴링하지는 않는다.** 값싼 GET으로 root가 addressable해질 때까지 기다리는 단계는 이
지연을 구조적으로 막지 못한다. 82건의 code-24 실패에서 `error_user_msg`를 디코딩해보면 Meta가 지목하는 media
id는 root가 아니라 **reply 자신의 create-container**다. 실측도 같은 방향이다. 30회 실행에서 "root가 예산 안에
addressable해지지 않았다"는 **0번**인 반면 reply 재시도는 25일간 **77번** 발생했고, 그 GET은 매번 첫 시도에 200을
돌려준다. 77건을 실제로 회복시키는 것은 reply 쪽 재시도(`_publish_reply_with_retry`, 기본 3회)와 위의 공유
예산이다.

**per-reply best-effort와 전달량 회계(`ThreadsDelivery`).** reply 게시는 건별로 try/except다. 한 reply가
실패해도 나머지를 포기하지 않아 댓글 chain이 중간에 끊기지 않는다. 반환값은 bool이 아니라 root를 포함한
`(posted, expected)` NamedTuple이며, `published`(root와 reply chain이면 최소 1건)와 `partial`(게시됐지만 일부
누락)을 구분한다. bool이면 5개 중 4개만 붙은 실행도 그냥 성공이어서 truncated chain을 아무도 알 수 없다.

⚠️ 호출자는 값 자체의 truthiness로 분기하면 안 된다. NamedTuple은 `(0, 5)`도 truthy다. `daily_visual`과
`delivery` 모두 `.published`를 명시적으로 읽으며, `published`가 아니면 ledger 마커를 롤백해 그날을 재시도
가능하게 둔다. 이미지만 있고 스토리가 없는 다이제스트를 게시됨으로 굳혀서는 안 된다. 부분 전달과 전면
실패는 ERROR로 로깅되고, 비주얼 Lambda가 `ALERT_SNS_TOPIC_ARN`이 있을 때 SNS 알림을 올린다. 없으면 no-op이다.

**best-effort 계약.** API 오류는 로깅 후 건너뛰고 절대 raise하지 않는다. `output/delivery.py`의 리서치 경로가
이 계약에 의존한다. 단 자격증명(`THREADS_ACCESS_TOKEN`, `THREADS_USER_ID`) 부재는 ERROR로 올린다.
`enable_threads_post`가 켜진 구성에서 자격증명이 없으면 그날 다이제스트가 어디에도 전달되지 않으니, INFO
"skipping" 한 줄로 넘길 사건이 아니다.

**토큰 갱신.** `lambda_handlers/threads_refresh_handler.py`와 약 50일 주기 EventBridge 스케줄이 60일 만료
장기 토큰을 갱신해 SSM에 재기록한다([§11](#application_stack) 참조).

## 6. LLM 팩토리 (`shared/utils.py`)

### 의존성 경계

이 프로젝트는 `langchain-core`(프롬프트, 출력 파서, 러너블, 콜백)와 `langchain-aws`(모델 클래스)만 쓴다.
**최상위 `langchain` umbrella 패키지는 의존성이 아니며 다시 추가하지 말 것.** 아무것도 import하지 않으면서
`langchain-text-splitters`와 `sqlalchemy`를 끌고 오고, 그 둘은 이 코드가 도달하지 않는 경로의 권고로 pip-audit
게이트를 실패시킨다.

### 모델 팩토리

`BedrockLanguageModelFactory.get_model(model_id, stage=..., **kwargs)`가 `ChatBedrockConverse`를 반환한다.
리전은 `BedrockCrossRegionModelHelper`가 `global.` inference-profile ID를 먼저, 없으면 리전 프리픽스(us-west-2면
`us.`)를 붙인 ID로 해석하고,
이 프로젝트의 application inference profile이 있으면 그 ARN을 우선한다.

**모델 클래스는 하나다.** `use_converse` 같은 분기로 `ChatBedrock`과 갈리지 않고, thinking이나 1M 컨텍스트,
성능 레이턴시를 구성하는 역량 레이어도 없다. 해석기가 항상 profile ARN이나 `global.*` id를 돌려주니 다른 분기는
도달 불가이고, 레지스트리의 플래그는 **Converse** 기준의 값이다. 해석이 맨 모델 id로 폴백해도 Converse가 그것을
받으니 잃는 폴백도 없다.

**생성자 표면은 테스트로 고정한다.** `ChatBedrockConverse`가 pydantic `extra="forbid"`이니 프레임워크 메이저
업그레이드가 kwarg 하나를 renaming하면 즉시 깨진다. 그런데 나머지 테스트는 전부 config 딕셔너리만 검증하니,
그 딕셔너리를 실제 클래스에 넣어보지 않으면 그린으로 배포된다. 그래서 `tests/test_model_factory.py`의
`TestModelClassConstructorSurface`가 배포 환경에서만 도는 profile ARN 경로까지 포함해 실제 인스턴스를 생성한다.

**모델 ID와 단계별 선택.** `shared/constants.py`의 `LanguageModelId`는 선택 가능한 모델만 담는다(Sonnet 4.6, Sonnet 5,
Opus 4.8, Opus 5, 그리고 5.5 계열). 단계별 모델은 다음과 같다.

| 단계 | 모델 | 근거 |
|------|------|------|
| 랭킹 | Opus 5.5 | 135건 실제 풀에서 Opus 4.8과 상위 5개 중 3개, 8개 중 5개가 겹쳤다(Sonnet과 Opus를 비교했을 때와 같은 수준). 한 번에 Opus 4.8 $0.49, Opus 5.5 $0.45로 더 싸다. 점수가 전반적으로 약 0.04 낮게 나오니 `min_score` 근처 항목 수를 지켜본다 |
| 다이제스트, 그라운딩, 비주얼 편집기와 아트 디렉터 | Sonnet 5.5 | 같은 후보 풀(2026-10-11)로 비교했다. Opus 5.5는 리드가 6문장(스펙 3-4), 본문이 5·4·4·3·3문장, 산문 1426자였고(다이제스트와 그라운딩 한 번에 약 $0.27), Sonnet 5.5는 리드 4문장, 본문 모두 3문장, 1186자로 스펙을 지켰다(약 $0.11). Opus 5.5의 비유는 더 생생했지만 리드에서 같은 논지를 되풀이했다. 네 단계가 모두 `digest_model`을 읽으니 함께 바뀌고, Sonnet 5.5가 쓴 4컷 브리프도 깔끔했다 |
| 딥 리서치 | Opus 5.5 | 온디맨드 실행이고 품질이 결과물의 전부다. 실측 한 번에 약 2만 출력 토큰, 10 사이클 |
| 트렌드, 쿼리 정제 | Sonnet 5.5 | 짧은 구조화 출력 |
| CountTokens | Sonnet 4.6 | Opus 계열은 CountTokens를 지원하지 않는다(아래) |

Fable 5.1은 등록하지 않았다. Bedrock이 계정의 기본 데이터 보존 모드에서 이 모델을 거부한다("data retention mode
'default' is not available for this model"). 계정 단위의 설정 변경이 필요하다.

**레지스트리 플래그(`LanguageModelInfo`)는 Converse로 직접 확인한 값이다.** 재확인에 실제 호출 비용이 드니
docstring에도 같은 사실을 적어두었다.

- `supports_temperature`: Sonnet 4.6만 True다. 나머지는 비기본 `temperature`를 ValidationException으로 거부하고,
  레거시 `thinking.type="enabled"`/`budget_tokens`도 거부한다. 팩토리는 False인 모델에 `temperature`를 보내지
  않는다.
- `supports_forced_tool_choice`: Opus 5.5와 Sonnet 5.5는 `tool_choice`의 `any`와 `tool`을 거부한다(Haiku 5.5는
  받는다). langchain-aws는 id에 "claude"가 있으면 강제 선택을 지원한다고, application inference profile ARN이면
  아무것도 지원하지 않는다고 추론한다. 그러면 비주얼 브리프의 `with_structured_output`이 global id에서는 도구를
  강제해 400이 나고 ARN에서는 강제하지 않는다. 그래서 False인 모델은 팩토리가 `supports_tool_choice_values=("auto",)`를
  명시하고, 모델이 도구를 부르지 않고 답하면 `VisualGenerator.brief()`가 명시적인 오류를 낸다.
- `effort`: Opus 5.5와 Haiku 5.5는 effort 기본값이 `medium`이다(다른 모델은 `high`). 이 필드가 `"high"`인 모델은
  팩토리가 `additional_model_request_fields={"output_config": {"effort": "high"}}`를 보내고, 리서치 에이전트도 같은
  필드로 Strands `additional_request_fields`를 채운다. 단계의 모델을 바꿨다고 추론 깊이가 조용히 낮아지지 않게
  하려는 것이다. 5.5 계열은 thinking을 끌 수 없고 생략하면 adaptive로 돈다.

### 단계 태깅과 사용량 로깅

`get_model(model_id, stage="ranking"|"digest"|"grounding"|"trends"|"visual-editor"|"visual-synopsis"|"query-refine")`처럼
단계를 함께 받는다. 붙은 콜백이 호출마다
`LLM usage stage=... model=... input=... output=... cache_read=... cache_write=...`를 남긴다.

이유는 단순하다. 청구는 모델 단위인데 여러 단계가 한 모델을 공유한다(랭킹과 리서치는 Opus 5.5, 다이제스트와
그라운딩, 비주얼, 트렌드, 쿼리 정제는 Sonnet 5.5). 이 태그 없이는 토큰 총량을 쓴 주체로 되짚을 수 없다. 리서치 에이전트는 `get_model`을 우회하니 대신
CLI 결과 JSON의 `usage`가 같은 단위(입력, 출력, 캐시 읽기·쓰기 토큰)로 보고한다. 텔레메트리는 설계상 best-effort이며 어떤 읽기 실패도 생성을 막지 못한다.

### 비용 귀속 (application inference profile)

온디맨드 `InvokeModel`은 과금 대상 리소스가 없어서 비용 할당 태그가 붙지 않는다. 그래서
`BedrockCrossRegionModelHelper`가 시스템 프로필을 해석한 뒤, 이 프로젝트의 application inference
profile(`{project}-{stage}-{model-slug}`)을 찾아 그 ARN을 반환한다. 이름은
`application_profile_name(model_id, project, stage)` 하나가 만든다. `scripts/put_inference_profiles.py`는
`config.aws.project_name`/`stage`로 이름과 `Project`·`Stage` 태그를 함께 정하고, 런타임 해석기는 스택이 같은
config에서 Lambda에 넣는 `PROJECT_NAME`/`STAGE`를 쓴다. 그래서 dev가 아닌 스테이지에서도 두 경로가 같은 이름을 본다.

해석이 이 한 곳뿐이라는 점이 중요하다. LangChain 팩토리와 Strands 리서치 에이전트가 동시에 커버된다.
프로필이 없거나 조회가 거부되면 시스템
프로필로 조용히 폴백한다. 리포팅이 생성을 막아서는 안 된다. `ChatBedrockConverse`는 ARN에 `provider`를
요구하니 config 빌더가 `provider="anthropic"`을 붙인다.

단계의 모델을 바꾸면 그 모델의 프로필을 `put_inference_profiles.py`로 다시 만든다. 없으면 시스템 프로필로 돌고
비용이 태그되지 않는다.

⚠️ IAM 주의. `application-inference-profile`은 `inference-profile`과 다른 ARN 리소스 타입이라 정책에 따로
넣어야 한다. 누락하면 프로필이 존재하는 순간 모든 Bedrock 호출이 AccessDenied가 된다.

### 토큰 카운트

`count_tokens(text)`와 `truncate_to_tokens(text, max_tokens)`는 로컬 휴리스틱이 아니라 Bedrock CountTokens
API로 권위 있는 카운트를 얻는다. 일부 베이스 모델만 CountTokens를 노출하니(Sonnet 4.6은 지원하고 Opus 4.8은
AccessDenied 또는 'doesn't support counting tokens'로 거부한다) 호출자 모델과 무관하게 항상
`TOKEN_COUNT_MODEL`(Sonnet 4.6)로 카운트한다. 그래서 두 함수는 `model_id`를 받지 않는다.

**토크나이저 주의.** Opus 4.7 이후 모델과 Sonnet 5 계열은 토크나이저가 달라서 같은 텍스트를 더 많은 토큰으로
센다(최대 약 1.35배). 따라서 이 카운트는 실제 사용량을 약간 과소평가한다. 다만 `item_text_max_tokens` 컷이 더 넉넉한 상한이 되니 컨텍스트
초과 위험은 없고 보수적인 방향이다.

cross-region `global.`이나 `us.` 같은 프리픽스는 베이스 id로 스트립하고, 오류가 나면 char/4 추정으로
폴백한다. `truncate_to_tokens`는 문자 컷 지점을 이진 탐색으로 찾는다.

**메모이제이션.** 결과를 팩토리 인스턴스에 텍스트 해시로 캐시한다. 프롬프트 빌드가 같은 항목 텍스트를
랭커와 다이제스트, 그라운딩 단계에서 반복 카운트하고 `truncate_to_tokens`의 이진 탐색이 겹치는 prefix를 여러
번 재는데, 각각 별도로 API 과금될 것을 캐시가 흡수한다. 팩토리는 Lambda invoke당 한 번 생성되니 캐시도 그
범위로 bounded다.

### 시크릿 헬퍼

`resolve_secret(env_var, ssm_suffix)`는 env를 먼저 보고 그다음 SSM(`/{project}/{stage}/{suffix}`,
SecureString 복호화)을 본다. OpenAI 키는 데일리 비주얼의 gpt-image 렌더에서만 쓰이고, Tavily 키는 리서치
백엔드와 웹서치 수집기에서 같은 순서로 해소된다. 사다리는 이 하나뿐이다. 사본을 두면 플레이스홀더 처리와 리전
해석이 빠진 채로 갈라진다.

**리전은 `aws_region()`이 정한다.** env(`AWS_REGION` → `AWS_DEFAULT_REGION`), 없으면 `config.aws.region`,
그것도 비면 `None`을 돌려 boto3가 프로파일에서 풀게 한다. 모듈마다
`os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "ap-northeast-2"))` 리터럴을 들고 있으면 개발자 한 명의
리전이 코드에 박혀 `config.aws.region`과 갈라지고, 다른 배포에서 SSM이나 AgentCore 호출이 엉뚱한 리전으로 간다.

env가 config보다 앞이라는 점은 로컬에서 함정이 된다. 모델 호출용으로 `AWS_REGION=us-west-2`를 물려받은 셸에서
돌리면 AgentCore Memory(ap-northeast-2) 조회가 `ResourceNotFoundException`으로 실패하고 cross-day dedup이 경고만
남긴다. Memory를 건드리는 로컬 실행에는 `AWS_REGION=ap-northeast-2`를 준다. Lambda에서는 `AWS_REGION`이 함수
리전이라 문제가 없다.

### 프롬프트 캐싱

Bedrock 프롬프트 캐싱은 Claude 기준으로 캐시 가능 프리픽스 최소치가 약 1024 토큰이다. 그래서 효과가 있는
곳에만 적용한다.

**에이전트에는 적용한다.** 약 1.7K 토큰의 시스템 프롬프트와 도구 스키마, 누적된 도구 결과가 매 ReAct 스텝마다
재전송되니 그 프리픽스를 캐싱한다(실측 한 실행에서 캐시 읽기 30만 토큰).
`BedrockModel(cache_config=CacheConfig(strategy="anthropic"))`(`agent/research_agent.py`)이다.

여기서 `"auto"`가 아니라 `"anthropic"`인 것이 중요하다. Strands는 `"auto"`를 model id 문자열에 `claude`나
`anthropic`이 있는지로 판단하는데, 배포되는 id는 application inference profile
ARN(`arn:...:application-inference-profile/<opaque-id>`)이고 그 안에는 둘 다 없다. 그러면 `"auto"`는 이 모델이
캐싱을 지원하지 않는다고 결론 내려 cache point를 전부 조용히 버린다. 비용 귀속 프로필을 켜는 것만으로 캐싱이
꺼지는 셈이고, 리서치 턴은 시스템 프롬프트와 누적된 도구 결과 전체를 매번 재전송하니 모르고 잃으면 가장 비싼
기능이다. 이 레지스트리의 모든 모델이 Anthropic이니(Bedrock 팩토리도 ARN id에 `provider="anthropic"`을 붙이며 같은
사실을 단언한다) 문자열 추측에 맡기지 않고 명시한다. 검증은 `AgentResult.metrics.accumulated_usage`로 한다. 첫
호출에 `cacheWriteInputTokens`, 이후 `cacheReadInputTokens`가 발생한다.

**파이프라인에는 적용하지 않는다.** 단발성 프롬프트(랭커와 다이제스트, 트렌드, 시각화 시놉시스로 모두 약 530
토큰이며 실행당 한 번 호출한다)는 캐시 최소치에 미달하고 호출 간 재사용도 없으니 의도적으로 캐싱하지 않는다.
그래서 모델 레지스트리에도 `supports_prompt_caching` 같은 플래그를 두지 않는다. 어차피 모든 엔트리가 True가 되고
읽는 곳도 없어서, 레지스트리에 쓰이지 않는 플래그만 남는다. 파이프라인 스테이지의
`cache_read`/`cache_write` 로그 필드가 항상 0인 것도 그래서 정상이다.

## 7. 메모리: 세 개의 분리된 저장소

기억해야 할 것이 세 종류이고 성격이 서로 달라서 저장소도 셋으로 나뉘어 있다. 트렌드는 여러 날에 걸쳐 누적되는
구조화된 상태이고, 다이제스트 스냅샷은 Lambda 사이에 넘기는 그날의 사진이며, cross-day dedup 히스토리는 짧은
TTL의 원장이다.

- **기억하는 것.** 매일 다이제스트를 분류해 `trends.json`에 트렌드별 한 줄 근거를 날짜와 함께 쌓는다. 날짜별
  다이제스트 스냅샷(90일), 최근 lead, 발행 URL과 Threads 게시 날짜도 남긴다.
- **압축하는 방식.** 요약을 다시 쓰지 않고 상한으로 자른다. 트렌드당 근거 최근 `trend_max_evidence`(5)개,
  30일(`trend_retention_days`) 동안 다시 보이지 않으면 삭제, 다이제스트에는 모멘텀 상위
  `trend_max_active_trends`(10)개만 넣는다. LLM이 매일 요약을 다시 쓰는 방식은 이력을 잃고 지어내서 쓰지 않는다.
- **기억하지 않는 것.** 근거 5개를 넘는 오래된 흐름의 서사는 사라지고 첫 등장일과 추적 일수만 남는다. 읽는
  쪽(분류기와 다이제스트는 최신 근거, `recall_trends`는 최근 3개)이 그 이상을 쓰지 않는다.

### 7.1 트렌드 — 구조화 `trends.json`

`StateStore`에 저장되며 시스템 오브 레코드다. 관리 주체는 `pipeline/trend_tracker.py`의 `TrendTracker`다.

**스토어 선택(`create_state_store`).** `STATE_BUCKET`(env), `config.aws.state_bucket_name`, 로컬 파일 폴백
순으로 **버킷 유무만** 보고 결정한다. `is_running_in_aws()` 같은 플랫폼 감지로 갈라서는 안 된다. Lambda가 아닌
호출자(로컬 리서치 CLI, 실 버킷을 향한 로컬 실행)가 `STATE_BUCKET`을 들고 있어도 trends.json을 로컬
파일시스템에 쓰게 되고, 그러면 트렌드 히스토리가 통째로 사라진다. AWS 밖에서는 세션을 `config.aws.profile`과
`region`으로 만들어 `.env`에 `STATE_BUCKET`을 둔 개발자가 자격증명을 잃지 않게 하고, AWS 안에서는 실행 역할, 곧
기본 세션을 쓴다. prefix 규약은 env 경로(`S3_PREFIX`가 곧 digest-state prefix)와 config 경로(`s3_prefix`에
`/digest_state`를 붙인다)가 서로 다르니 그대로 유지한다.

**읽기 실패는 히스토리 없음과 다르다(`StateReadError`).** 스로틀되거나 거부된 S3 GET이 `None`을 반환하면 키가
없는 것과 구분되지 않고, 다음 read-modify-write가 그 공백을 영구화한다. 발행 URL 원장과 최근 lead, Threads 멱등
마커가 한 번의 실패 읽기로 비워질 수 있다는 뜻이다.

그래서 `read`는 `NoSuchKey`와 404만 조용히 없음으로 처리하고, 그 외 `ClientError`(및 로컬
`OSError`)는 `StateReadError`를 raise한다. 소비자는 전부 동일하게 처리한다. ERROR를 로깅하고, 히스토리는
모른다고 보고, 쓰기를 생략한다. 이 예외는 게시 경로로는 절대 전파되지 않는다. `RollingLog.entries()`는 `[]`,
`ThreadsPostLedger.already_posted()`는 `False`를 반환한다. 중복 게시는 복구할 수 있지만 미게시는 그렇지 않기
때문이다. 그리고 `TrendTracker`는 그 실행의 trends.json 쓰기를 건너뛴다.

**LLM의 역할은 분류뿐이다.** `TrendClassifyPrompt`는 오늘 아이템이 기존 트렌드(id)의 확장인지 신규인지만
분류한다. 부기는 전부 결정론적 Python이다.

- 증거 날짜는 코드가 스탬프한다. LLM이 아니다.
- 상태(active, cooling)는 `last_seen`을 `trend_cooling_days`에 비교해 계산하고, `last_seen`은 저장된 증거의
  가장 늦은 날짜에서 도출한다. 그래서 같은 날 재실행이 오늘 증거를 걷어내면 `last_seen`도 함께 뒤로 간다.
- `last_seen`이 `trend_retention_days`를 넘긴 트렌드는 보관 단계 없이 삭제한다(보관된 트렌드는 어느 독자도 읽지
  않아 되살아날 수 없었다).
- momentum은 recency 감쇠다. `0.5^(age/half_life)`이고 `trend_momentum_half_life_days` 기본값은 7일이다.
- 트렌드당 증거는 `trend_max_evidence`로 캡한다.
- `trend_max_active_trends`는 **읽는 쪽에만** 캡을 건다(`get_trends_context`). 저장된 status를 이 캡으로
  바꾸지는 않는다. 캡 때문에 숨긴 트렌드는 분류기도 못 보게 되어 연장될 수 없고 새 id로 다시 생성된다.
- 동일 날짜 재실행은 멱등이다. 그날 증거를 교체한다. 오늘 증거는 분류기용 목록을 렌더하기 **전에** 걷어내서,
  분류기가 첫 실행의 스토리를 기존 트렌드로 보지 않는다. 다이제스트용 블록도 다이제스트 날짜 **이전** 증거만
  쓴다. 그렇지 않으면 19:00 실행의 스토리가 19:25 재실행에 "최근 며칠의 트렌드"로 되돌아갔다.

**로드 견고성.** 전체 `TrendMemory.model_validate_json`이 실패하면(스키마 드리프트나 제거된 enum 값 등) 모든
history를 버리지 않고 트렌드별로 관대하게 복구한다(`_recover_trends`가 개별 검증해 살아남는 것만 유지한다).
레코드 하나가 나빠도 누적 history가 통째로 날아가지 않아야 한다.

**진실의 원천.** `trends.json`(`TrendMemory`)이 원천이고 렌더된 텍스트는 뷰다.

**주입, 이른바 recurrence ammunition.** 다이제스트 생성 시 active와 cooling 트렌드를 momentum 순으로 렌더해
`DigestPrompt`에 주입하되, 각 트렌드에 코드가 증거에서 산출한 재등장 사실을 붙인다(`_render_ammunition`).
추적 N일째, 서로 다른 N일 재등장 같은 값이다. 이 수치가 lead의 날카로운 근거로 쓰이고 LLM이 지어내지 않는다.
"이번 달 N회"는 두지 않는다. 근거 캡(5개)에 걸린 트렌드에서는 실제보다 적게 셌다.

다이제스트용 블록만 `trend_max_active_trends`로 캡한다. 새 노브를 만들지 않고 기존 active 캡을 재사용했다.
`visible`에는 cooling도 들어가서 20줄을 넘길 수 있고 대부분은 에디터가 쓰지 않는 끊긴 흐름이다. 반면 분류기용
`_render_existing`은 캡하지 않는다. 거기서 cooling 트렌드를 숨기면 그 실이 고아가 되고 모델이 같은 주제에
중복 id를 새로 만든다.

### 7.2 다이제스트 스냅샷 — AgentCore Memory (`shared/memory.py`)

`AgentCoreMemoryStore`가 그날의 스냅샷(`ranked_items`와 `digest_result`)을 단기 세션 이벤트로 기록한다
(`create_event`, 세션 `digest-<date>`). 각 `RankedItem`이 자기 `CollectedItem`을 품고 있으니 수집 원본은 따로
저장하지 않는다. `_fit_to_limit`이 100k자 한도를 지킨다. 넘치면 ranked 항목 본문을 자르고, 그래도 넘치면 본문
없는 메타데이터만 남긴다. 목적은 데일리 비주얼 Lambda와 리서치 CLI의 `recall_digest`가 이 스냅샷을 읽는 것이다.

**읽기는 그 날짜의 세션을 직접 읽는다.** `get_digest(date)`가 `digest-YYYY-MM-DD` 세션을 직접 읽는다. 비주얼
Lambda는 자기가 트리거된 날짜의 콘텐츠를 게시해야 하니 최신을 읽고 날짜를 비교하는 방식은 쓸 수 없다.
`digest_result.generated_at`이 UTC라서 09:00 KST 이전 실행에서는 KST 다이제스트 날짜와 항상 어긋난다. 없으면
`None`이며 최신으로 폴백하지 않는다. 어제 스토리를 오늘 게시하는 것을 막는다.

읽기 자체가 실패하면(스로틀이나 거부) `None`이 아니라 `MemoryReadError`를 raise한다. `None`으로 떨어뜨리면 그날
다이제스트가 없는 것과 구분되지 않아 비주얼 Lambda가 게시를 건너뛰고 200을 반환한다. 게시 경로는 그대로 터뜨려
Errors 알람과 DLQ로 보내고(`retry_attempts=0`이다), 보강용 읽기인 cross-day dedup 시드만 날짜별로 건너뛴다.

**세션 안에서도 최신 이벤트를 고른다.** 한 세션은 보통 이벤트 1건이지만 같은 날 재실행이 두 번째를
append하고, `list_events`는 순서를 보장하지 않는다. `maxResults=1`은 그날의 첫 시도, 즉 폐기된 쪽을 서빙할 수
있다. 그래서 작은 페이지(`EVENTS_PER_SESSION`)를 읽고 `eventTimestamp`로 최신을 고르며, 동률일 때만 페이로드의
`digest_result.generated_at`으로 선택적 tie-break를 한다. 그 필드 없이 저장된 기존 스냅샷도 그대로 로드된다.
페이지 크기를 작게 둔 이유는 `get_recent_digests`가 날짜마다 이 페이지를 읽기 때문이다. 이력이 늘어도 세션당
비용이 커지지 않아야 한다. 파싱 불가 이벤트는 건너뛰되, 그래서 아무것도 남지 않으면 예외를 올려 읽기 실패가 빈
날로 읽히지 않게 한다.

**시맨틱 recall과 장기 전략은 쓰지 않는다.** 관리형 추출은 트렌드의 흐름이 아니라 안정적인 사용자-사실을 뽑도록
설계돼 있어서 이 용도에 맞지 않는다. cross-day 연속성은 `trends.json`이 담당한다([§7.1](#71-트렌드--구조화-trendsjson)).

**스토어 선택(`create_memory_store`).** `MEMORY_ID`(CDK가 Lambda에만 넣는다)가 있으면 그 id를, 없으면 스택이
만든 메모리를 이름(`digest_memory_name(project, stage)`, CDK와 같은 함수)으로 찾아 `config.aws.region`에서 연다.
로컬 실행(`main.py`, `research_cli.py`, `recall_cli.py`)이 운영 스냅샷을 읽게 하려는 것이다. 예전처럼 `MEMORY_ID`가
없다고 로컬 파일로 떨어지면 cross-day dedup은 어제를 못 보고 `recall_digest`는 거의 모든 날짜에 답하지 못했다. 이름
조회와 이벤트를 읽는 `bedrock-agentcore` 클라이언트는 하나의 세션을 쓴다. 프로필은
`available_boto_profile(config.aws.profile)`, 리전은 스택 리전(`config.aws.region`)이다. 셸의 `AWS_PROFILE`과
`AWS_REGION`은 다른 계정이나 모델 호출 리전을 가리킬 수 있고, 읽기 클라이언트만 기본 세션을 쓰던 때는 이름으로
찾은 메모리를 "Memory not found"로 답했다. 메모리를 찾지 못할 때만
`LocalMemoryStore`(`digest_*.json`)로 폴백한다.

**윈도 조회.** `get_recent_digests(before, days)`가 cross-day dedup 시드를
제공한다([§7.3](#73-cross-day-dedup-히스토리-sharedhistory_storepy)). 세션 id가 날짜로 정해지니 `before` 이전 `days`일의
세션을 직접 읽고, 세션 목록을 페이지네이션하지 않는다(그래서 `ListSessions` 권한도 필요 없다). `before` 당일은
빼서 같은 날 재실행이 자기 스토리를 살리게 하고, 한 날짜의 읽기 실패는 경고하고 건너뛴다.

### 7.3 cross-day dedup 히스토리 (`shared/history_store.py`)

`StateStore` 위에 작은 구조 세 개가 올라간다.

**`PublishedUrlLedger` (`published_urls.json`).** 정규화 URL에서 마지막 발행 ISO 날짜로 가는 롤링 맵이고 TTL은
`published_url_ttl_days`(기본 6)다. `recent_urls(today)`는 엄격히 더 이른 날(`0 < age < ttl`)만 반환한다. 같은
날(age 0)을 제외하니 같은 날 재실행이 자기 다이제스트를 재현할 수 있다. within-run 중복은
집계기가 처리한다. `record()`가 발행 URL을 오늘로 스탬프해 병합하고 TTL 밖 항목을 prune한다.

**`RollingLog`.** 한 JSON blob에 담는 capped FIFO다. 반복 방지용 최근 lead(`recent_leads.json`)에 쓴다. `append(record, dedup_key=...)`로 같은 키 값의 기존 항목을 교체할 수 있고,
leads는 `date`로 dedup해서 `--force-republish` 재실행이 같은 날 lead를 중복 추가해 반복 방지 윈도를 잠식하지
않게 한다.

**`ThreadsPostLedger` (`threads_posted.json`).** 데일리 Threads 게시의 멱등 마커다. `{date: owner_run_id}`
맵으로 저장하고 레거시 bare-list도 읽는다. 호출자(비주얼 Lambda)가 여러 분에 걸친 게시 전에 날짜를 마크하고
실패하면 롤백하는데, `unmark`은 소유권 스코프다. 자기 `run_id`(correlation id)일 때만 해제한다. 그러지 않으면
동시 invocation의 실패 롤백이 성공한 게시의 마커를 지우고 다음 실행이 중복 게시한다. read-modify-write 자체는
원자적이 아니지만(락이 없다), 남의 성공을 내 실패가 지우는 경로는 이 스코프가 막는다.

**`published_urls_from_snapshots`.** 과거 다이제스트 스냅샷의 `content.items[].url`을 뽑아, dedup이 원장뿐
아니라 AgentCore Memory 히스토리로도 self-heal되게 한다. 원장이 비어도 작동한다.

**시드와 기록(`pipeline.runner.run_pipeline`).** exclude 집합을 원장과 최근 AgentCore Memory 스냅샷 양쪽에서
시드한다(`get_recent_digests(today, published_url_ttl_days)`로 원장과 같은 TTL 윈도를 읽는다). 생성 후에는 발행된 `content.items` URL을 원장에 기록하고, lead를 `recent_leads.json`에 append하되
AGI 카운트다운 프리픽스를 제거한다(`_editorial_lead`). novelty 신호가 고정 보일러플레이트가 아니라 편집 각이
되어야 한다.

### `recall_trends`는 어디를 보는가

혼동하기 쉬운 지점이라 따로 적어둔다. `recall_trends` 도구는 AgentCore가 아니라 `trends.json`을 직접
읽고, 증거가 있는 트렌드 전부를 momentum 순으로 돌려준다. 메모리 리소스(`AWS::BedrockAgentCore::Memory`)는
이벤트 전용이고(단기, `event_expiry_duration` 90일) 시맨틱 전략도 `RetrieveMemoryRecords` 권한도 없다.

## 8. 헬스 체크와 알림

### 모델 (`shared/models.py`)

- `SourceStatus`는 `ok`, `empty`, `failed`, `stale`, `degraded` 중 하나다.
- `SourceHealth(name, item_count, status, detail)`.
- `HealthReport(sources)`는 `failed_sources`, `stale_sources`, `degraded_sources`, `empty_sources`, `summary()`를
  갖는다. 넷 다 **이름 목록**을 돌려준다. 하나만 bool로 두면 알림 쪽이 그것을 쓸 수 없어 같은 판정을 인라인
  컴프리헨션으로 다시 구현하게 되고, 판정이 두 벌로 갈라진다.
- `RankingHealth(batches_total, batches_failed, items_total, items_scored, items_lost)`는
  `coverage`와 `degraded`, `summary()`를 갖고 `DigestResult.ranking_health`로 실려 나간다. `degraded`의 조건은
  재시도까지 실패한 배치가 있는 것 하나다. 커버리지 비율로 degrade하는 조건은 두지 않는다. 실측 커버리지가 30회
  실행에서 0.93~1.0 또는 0.0으로 이분되어, 비율이 `batches_failed`와 다른 것을 가려낼 여지가 없다. 빠진 id는
  랭커가 재질의로 메운다([§5.2](#52-랭커-rankerpy)).

STALE과 DEGRADED는 실패가 아니니 `failed_sources`에 들어가지 않는다. FAILED 승격 경로와 분리되어 있다.

### 소스 분류 (`run_collectors_with_health`)

- 예외가 나면 **FAILED**이고 잘린 detail을 담는다.
- park 파일이 `degraded`(stale은 나이 예산 초과, error는 읽기 불가)면 **STALE**이고 항목 수와 park detail을
  담는다. 항목은 나왔지만 그 sync가 멈춰 있다는 뜻이라 OK도 FAILED도 아닌 별도 상태다.
  `_build_collector_tasks`가 코루틴과 함께 수집기 인스턴스를 반환하고, 이 함수가 각 인스턴스의
  `park_status`를 읽어 판정한다.
- 수집기가 `degraded_detail`을 남겼으면 **DEGRADED**이고 항목 수와 detail을 담는다. 항목은 제때 나왔지만 그
  소스의 입력 중 일부만 응답한 경우로, RSSHub 계정 피드가 대부분 실패한 상황이 그렇다. 이 상태가 없으면 40개
  계정이 3개로 줄어든 날도 OK로 읽힌다. STALE 판정이 더 조치 가능하니 park가 degraded면 STALE이 우선한다. 이
  판정은 보고와 알림만 바꾸며 집계기에 도달하는 항목은 그대로다.
- 0 항목이면 **EMPTY**이고, 조용한 날엔 정상이다. 단 `collectors.alert_on_empty`가 이름을 지목한 소스는
  어두워진 것이 사건이니 로컬 실행에서도 ERROR 한 줄을 남기고 Lambda는 알림을 올린다.
- 그 외에는 **OK**다.

### 알림

**소스 알림(`_maybe_alert`, 다이제스트 Lambda).** 소스가 FAILED거나 STALE, DEGRADED일 때, 그리고 빈 항목 조기
반환 이전에 `ALERT_SNS_TOPIC_ARN`으로 게시한다. 아무것도 수집하지 못한 날에도 장애가 알림되어야 한다.
메시지는 실패와 stale, degraded, empty 소스 목록을 각각 분리해 담는다. FAILED만 보는 게이트라면 죽은 로컬 cron이
며칠간 무음으로 지나간다. EMPTY는 `collectors.alert_on_empty`가 지목한 소스만 포함한다. reddit이나 X의 조용한
날이 매일 페이징하지 않게 하는 config 게이트다.

**랭킹 헬스 알림(`_maybe_alert_ranking`, 파이프라인 이후).** 위 수집기 알림은 파이프라인 이전 호출을 그대로
두고, 랭킹 판정은 별도로 게시한다. 파이프라인 예외가 수집기 알림을 삼켜서는 안 된다. 재시도까지 실패한 배치가
있으면(약 후보 40건 소실) 겉보기 정상인 다이제스트에도 알림이 간다.

출고된 다이제스트의 소스 구성은 알리지 않는다. 무엇을 내보낼지는 에디터의 판단이고([§5.2](#52-랭커-rankerpy)),
사후에 할 수 있는 조치가 없다. 그런 알림이 운영 30회 중 14회 울리자 진짜 소스 장애 알림(X 수집 중단 54일)이 그
소음에 묻혔다. 알림은 사람이 조치할 수 있는 상태에만 쓴다.

**알림 경로는 한 곳이 소유한다(`shared/alerts.publish_alert`).** `ALERT_SNS_TOPIC_ARN` 조회와 `format_alarm`
조립, publish, 경고 로그, 예외 삼키기가 모두 여기 있고, 핸들러는 `fields` 조립만 갖는다. 두 Lambda 핸들러가 같은
루틴을 각자 들면 그 다섯 단계가 바이트 단위로 같은 사본이 되어, 알림 경로 수정이 한쪽에만 적용된다. 계약은 "절대
raise하지 않는다"다. 알림 실패가 자기가 보고하려던 실행을 실패시켜서는 안 된다.

**알림은 귀속 가능해야 한다(`format_alarm`).** subject는 `[{project}/{stage}] {event} — {STATUS}`이고
project와 stage를 각 함수의 env(`PROJECT_NAME`, `STAGE`)에서 읽는다. 하드코딩된 기본값에 의존하면 dev 스테이지
알림과 prod 스테이지 알림이 바이트 단위로 같아져서, 두 번째 배포가 엉뚱한 이름으로 알린다. 본문 마지막 필드는
correlation id다. 매 invoke마다 세팅되는 값이니 알림에 실어야 운영자가 메일에서 대응되는 JSON 로그 줄로 건너갈 수
있다.

### 핸들러 예외 전파

다이제스트와 비주얼, Threads 갱신 핸들러는 실패를 correlation id와 함께 로깅한 뒤 다시 raise한다. 500 body를
반환하면 Lambda 입장에서는 정상 종료라서 Errors 알람도 비동기 DLQ도 울리지 않는다. 세 함수 모두
`retry_attempts=0`이니 재시도로 인한 이중 게시 위험은 없다.

### 게시량 메트릭과 날짜 전달 (다이제스트에서 비주얼 Lambda로)

**EMF 봉투는 `shared/metrics.emit_emf`가 소유한다.** `_aws`/Timestamp 봉투를 호출 지점마다 손으로 만들면
`"Dimensions": [[]]`로 렌더되기 쉽고, 차원이 없으면 dev와 prod가 **같은 데이터포인트**에 쓴다. prod가 아무것도
내지 못한 날 dev의 5건짜리 실행이 prod의 `Maximum<1` EmptyDigestAlarm을 초록으로 유지한다. 그래서 이 함수가 `PROJECT_NAME`/`STAGE`에서 읽은 `Project`/`Stage` 차원을 붙이고(값이
비면 그 차원은 생략한다. CloudWatch가 빈 차원 값을 가진 레코드를 거부하기 때문이다), CDK 알람도 같은
`metric_dimensions` 맵으로 만든다. 네임스페이스는 `shared/constants.METRIC_NAMESPACE` 하나다. EMF는 로그 한
줄이니 새 AWS 리소스는 필요 없다.

**`DigestItemsPublished`(EMF)는 큐레이션된 스토리 수(`digest.content.items`)를 센다.** 랭커 후보 수를 세면
스토리 0건으로 게시된 날도 만점처럼 보고되어 `EmptyDigestAlarm`이 울리지 않는다. 타임스탬프는 UTC다.

**짧은 다이제스트는 성공으로 로깅되지 않는다.** `_parse_content`에서 emit된 항목이 드롭되면 ERROR다. 그 뒤
다이제스트는 겉보기 정상이라 이 줄이 유일한 흔적이다. 반면 동일 사건 병합으로 target보다 적어진 것은 정당하니
`run_pipeline`이 WARNING으로만 남긴다. 코드가 명시적으로 허용하는 경우다.

**날짜를 명시적으로 넘긴다.** `_trigger_visual(digest_date)`는 날짜를 페이로드에 담아 비동기로 invoke한다.
비주얼 Lambda는 `_requested_date`로 그 값과 날짜가 명시됐는지를 나타내는 플래그를 읽고(DLQ 재생 시에는 봉투의
`requestPayload` 아래 값도 인정한다) 해당 날짜의 스냅샷만 로드한다. 스냅샷이 없을 때 날짜가 명시된 invoke면
raise한다. 방금 persist한 실행이 부른 것이니 그날 무출력이라는 뜻이다. 오늘로 폴백한 invoke, 곧 로컬이나
수동 실행은 조용히 종료한다.

**`_trigger_visual`은 best-effort가 아니다.** AWS(`is_running_in_aws()`)에서 `VISUAL_FUNCTION_NAME`이 비었거나
invoke가 실패하면 raise한다. 스냅샷은 이미 persist된 시점이라 잃는 것이 없고, 이것이 Errors 알람과 DLQ 재생을
켜는 유일한 신호다. 로컬에서는 `main.py`가 비주얼을 인라인으로 실행하니 env 미설정이 정상이고 조용히
no-op이다.

**스냅샷 persist가 실패하면 비주얼을 트리거하지 않고 시끄럽게 실패한다.** 비주얼 Lambda가 이 스냅샷으로
게시하는 유일한 Threads 경로이니, persist가 실패한 날 트리거하면 다른 날짜의 스토리를 게시하게 된다. 트리거를
건너뛰고 예외를 다시 raise해 Errors 알람과 DLQ(재생 가능)로 남긴다. 조용히 넘기면 '어제 콘텐츠 게시' 대신
'완전 무출력인데 아무 신호 없음'이 된다.

**Threads 부분 전달 알림.** 비주얼 Lambda는 `ThreadsDelivery`(posted/expected)를 보고 누락이 있거나, 텍스트만
나갔거나, 게시가 실패하면 `ALERT_SNS_TOPIC_ARN`으로 SNS 알림을 올린다. env가 없으면 no-op이라 로컬 실행은
조용하다. 게시 결과를 EMF 메트릭으로 따로 남기지 않는다. 그 메트릭을 읽는 알람이나 대시보드가 없었고, 같은
판정을 SNS 알림이 이미 낸다.

타임스탬프는 어느 emit 지점에서든 `datetime.now(UTC)`를 쓴다. naive 로컬 시계로 찍으면 UTC epoch ms로 해석돼
엉뚱한 시각에 기록된다.

`_post_threads`는 콘텐츠가 있었는데 아무것도 게시되지 않은 경로(스토리 0건, 게시 예외)에서도
`threads_outcome`을 남기니(`expected>=1`) 알림과 메트릭이 조용히 넘어가지 않는다. 반면 이미 게시된 날의 스킵과
채널 비활성 스킵은 실패가 아니니 outcome을 남기지 않는다. 무음이다.

**남은 실행 시간으로 게시 경로를 bound한다.** 비주얼 Lambda는 자기 `context.get_remaining_time_in_millis()`를
평범한 monotonic float 하나로 바꿔(`_remaining_deadline`이고, context 객체는 파이프라인에 넘기지 않는다)
`run(deadline=...)`에서 `_post_threads`, `post_to_threads`로, 그리고 이미지 생성기로 흘린다.
`deadline=None`이면(로컬 실행과 `main.py`, `research_cli.py`) 아무것도 깎지 않는다. 딜라인이 있을 때만 인덱싱 예산이
`min(270초, 남은 시간 − 게시 예비분)`으로 줄고(예비분 `THREADS_PUBLISH_RESERVE_SEC`은
`THREADS_MEDIA_PROCESS_WAIT_SEC`과 reply 재시도 상수에서 산출한다), 남은 시간이 충분하면 270초를 절대 깎지
않는다. 스토리를 잃는 쪽은 인덱싱을 오래 기다리는 것이 아니라 일찍 포기하는 것이다. 이미지 쪽에서는 moderation
재렌더(추가 `visual_image_timeout_sec` 한 판)를 시간이 없으면 포기한다.

## 9. 딥 리서치 에이전트

자유형 토픽 하나를 열린 웹과 학술 문헌, 커뮤니티로 스스로 리서치해 한국어로 합성한 출처 표기 리포트를 쓰는
자율 에이전트다. 다이제스트와 분리된 독립 리서치이며 다이제스트 항목에 묶이지 않는다. 로컬 CLI가 Strands로
실행한다. AWS에 상시 떠 있는 런타임은 없다.

### 9.1 CLI 계약 (`research_cli.py`)과 OpenClaw 스킬

```bash
uv --directory ~/Projects/omnisummary run python research_cli.py "<토픽>" [--publish threads] [--output-dir PATH]
```

**stdout에는 JSON 객체 하나만 쓴다.** 로그는 전부 stderr로 간다. 필드는 `status`(`succeeded`, `failed`,
`rejected`), `topic`, `title`, `markdown_path`, `markdown`, `sources`(리포트가 인용한 URL), `images`(stage된 OG
이미지), `threads`, `model`, `stop_reason`, `usage`(입력·출력·캐시 읽기·쓰기 토큰, 사이클 수, 도구별 호출 수),
`warnings`, `error`다. `--publish threads`를 준 실행만 `threads`가
`{published, partial, posts_delivered, posts_rendered, dropped, trimmed}`이고, 나머지는 `null`이다. 리포트는
`reports/`(gitignore 대상, `--output-dir`로 변경)에 `YYYY-MM-DD-HHMMSS-<토픽 해시>.md`로 저장된다.

**종료 코드.** `0` 성공, `1` 실패(리포트 없음, 실행 오류, 요청한 Threads 게시가 완료되지 않음), `2` 입력
거부(빈 토픽. 알 수 없는 옵션이면 argparse가 JSON 대신 사용법을 stderr에 쓴다).

**게시는 플래그로만 한다.** Threads는 공개 계정이다. `--publish threads`가 없으면 에이전트가 무엇을 쓰든 게시되지
않는다. 플래그는 `DeliveryContext.publish_threads`라는 데이터로 흐르고, 프롬프트에는 `[DELIVERY] Threads
publishing: ALLOWED|NOT allowed.` 한 줄로 알린다. 토픽의 문구로 게시 여부를 추론하게 두면 'Meta Threads 추천
알고리즘'처럼 주제가 Threads인 요청까지 게시 요청이 된다.

**경고(`warnings`).** 에이전트가 `deliver_report`로 리포트를 제출하지 않고 답만 했으면 인용이 검사되지 않았다는
경고와, 어떤 도구도 돌려주지 않은 URL 목록을 싣는다. `stop_reason`이 `limit_*`이면 예산에서 멈췄다고 적고,
Threads에서 드롭되거나 잘린 게시물이 있으면 그 수를 적는다.

**로그 보안.** `.env`가 `LOG_LEVEL=DEBUG`여도 서드파티 로거는 WARNING으로 고정한다. botocore DEBUG는 세션
토큰이 담긴 요청 헤더를 남기고, 호출자(OpenClaw)는 stderr를 보관할 수 있다. `LOG_LEVEL`은 앱 로거만 움직인다.

**OpenClaw 스킬.** 정본은 `~/.openclaw/workspace/skills/omnisummary-research/SKILL.md`이고 저장소에는 두지 않는다
(사용자 정의 스킬은 워크스페이스 한 곳에만 둔다). scholar-lens 스킬과 같은 구성으로 외부 효과와 경계, 사전 점검,
명령, 출력, 검증과 보고, 실패와 재시도 정책을 적는다. 핵심 규칙은 둘이다. `--publish threads`는 사용자가 그
요청에서 Threads 게시를 명시했을 때만 붙이고, Threads에 게시한 실행은 다시 돌리지 않는다(`--publish` 없이 실패한
실행은 한 번 재시도할 수 있다).

### 구성 (`agent/research_agent.py`의 `create_research_agent`)

`BedrockModel`(기본 Opus 5.5, `config.agent.model_id`)을 streaming과
`CacheConfig(strategy="anthropic")`([§6의 프롬프트 캐싱](#프롬프트-캐싱) 참조)으로 구성하고 도구 8개로 Strands
`Agent`를 만든다. `max_tokens`는 `LANGUAGE_MODEL_INFO`에서 가져오고, `temperature`와 effort는 Bedrock 팩토리와
같은 레지스트리 플래그로 정한다(Opus 5.5는 `additional_request_fields`로 effort `high`). cross-region
inference-profile id는 `BedrockCrossRegionModelHelper`로 해석한다. 에이전트는 stdout에 아무것도 쓰지 않는다
(`callback_handler=None`). stdout은 JSON 한 개의 자리다.

boto 세션은 `AWS_BEDROCK_REGION` 또는 `config.aws.bedrock_region`과, 해석되는 경우의 `config.aws.profile`로
만들고 `boto_read_timeout`, `boto_connect_timeout`, `boto_max_attempts`(재시도 모드 `adaptive`, 파이프라인 팩토리와 같다)를
적용한다. 실행마다
`research_run_limits(config)`가 `Limits`(`research_max_turns`, `research_max_total_tokens`,
`research_max_output_tokens`)를 넘긴다. 상한에 걸린 실행도 텍스트를 반환하니, CLI가 `stop_reason`으로 경고를 단다.

**도구 목록은 프롬프트에서 파생된다.** `_render_tools_block(tools)`이 실제로 바인딩된 도구의 Strands tool
spec에서 이름과 인자, 한 줄 요약을 뽑아 `<tools>` 블록을 만든다. 손으로 쓴 목록을 두면 도구 이름을 바꾸는 순간
모델이 호출할 수 없는 메뉴를 읽는다.

### `SYSTEM_PROMPT_TEMPLATE`

- **`<role>`** — 독립 딥 리서치 에이전트임을 밝히고, 토픽과 각을 메시지에서 추론하게 한다. 되묻지 않는다.
- **`<context>`** — 오늘 날짜와 타임존을 코드가 시계에서 계산해 주입한다. 이것이 없으면 모델은 "최신"을 학습
  시점 기준으로 판단하고 `recall_digest`에 넘길 절대 날짜를 추측해야 한다.
- **`<voice>`** — 데일리 다이제스트와 같은 내레이터 페르소나(`config.pipeline.digest_voice_guidance` 주입)를
  리포트 길이에 맞게 쓴다. 반복 금지는 한 번만 적는다.
- **`<tools>`와 `<flow>`** — 도구 8개와 권장하되 강제하지 않는 흐름이다. 이해와 재작성, 다중 소스 리서치,
  아웃라인, 작성, 이미지 첨부(기본은 생략), 전달 순이다. 검색 폭 `research_breadth`와 깊이
  `research_max_iterations`를 가이던스로 주입한다.
- **`<delivery>`** — 항상 전체 리포트(약 `research_target_words` 단어)를 Markdown으로
  `deliver_report(channel="markdown")`에 넘긴다. `[DELIVERY]`가 허용할 때만 Threads 스레드를 추가로 넘긴다.
  Threads는 깊이가 아니라 **범위**로 갈린다. `research_max_threads_posts` 캡 안에서(게시물당
  `THREADS_MAX_POST_CHARS` 이하, `---` 구분) 논점을 더 적게 가져가되 하나하나를 제대로 설명한다. 같은 사실을 다
  담은 채 길이만 줄이라고 하면 사실을 의미 있게 만드는 설명이 먼저 버려져 주장 목록처럼 읽힌다.
- **`<language>`, `<formatting>`, `<citations>`** — 한국어 규칙(공유 `KOREAN_STYLE_RULES` 주입), Markdown
  리포트(`# 제목`, `## N. 섹션`)와 Threads 평문의 차이, 출처 구분(검증과 주장, 추론)과 날조 금지.

### 도구 (`agent/research_tools.py`)

모두 `@tool` 비동기 함수이고 에이전트가 자유롭게 조합한다.

**`web_search(query, news, time_range)`** — 열린 웹 검색이다. `news=True`면 Tavily `topic="news"`를 쓰고,
`time_range`(`day`|`week`|`month`|`year`)는 기간을 좁힌다. 둘 다 스키마의 boolean과 enum으로 나간다. 공유
`tavily_search`(`shared/research/research_backends.py`)에 위임한다. 결과에는 Tavily가 날짜를 줄 때
`Published:` 줄이 붙는다. "최신 동향" 질문이 이 도구의 주 용도인데 날짜를 버리면 모든 결과가 날짜 없이 읽혔다.
결과 포맷은 Tavily가 일부 페이지에 주는 명시적 null title/url/content를 흡수한다.

**`community_search(query)`** — Reddit과 X, HN, Substack의 반응을 본다. `community_search_domains`를
`include_domains`로 넘긴다.

**`search_papers(query)`** — OpenAlex `works?search=`다(`paper_search`, 키 없음, 하루 약 100회 무료). 자연어
쿼리를 받고 `filter=has_abstract:true`로 초록이 있는 논문만 받는다. 게시일과 피인용 수를 붙이고, 역색인으로 오는
초록을 문장으로 되살린다. 429와 5xx는 retry와 backoff를 거친다. 키 없는 Semantic Scholar는 이 환경에서 7회 호출
모두 429였고, 그래서 리포트에 논문이 한 편도 실리지 않았다.

**`read_url(url)`** — 특정 페이지의 전문을 가져온다(Tavily extract, `research_content_cap_chars`로 캡).

**`recall_trends()`** — 인자가 없다. `trends.json`에서 증거가 있는 트렌드 전부를 momentum(KST 날짜 기준) 순으로,
처음·마지막 관측일과 관측 일수, 최신 요지(`search_content_preview_chars`로 캡)와 함께 준다. 관련성은 모델이 판단한다.
키워드 매칭은 "diffusion LLM"에 "AI Doom Discourse"를 1위로 주었고, 한국어 질의는 영어 트렌드와 하나도 맞지 않았다.

**`recall_digest(digest_date)`** — 그 날짜의 다이제스트 스냅샷(AgentCore Memory)에서 lead와 스토리 제목을
되읽는다. 출력은 bounded다(`top_n`개 스토리, 줄당 `search_content_preview_chars`). 없는 날은 다른 날짜로
폴백하지 않고 `No digest stored for <date>.`를 돌려주고, 읽기 실패는 그와 다른 문장으로 degrade한다. 스로틀이나
거부를 "그날은 아무것도 안 다뤘다"로 보고하면 리포트가 사실과 다른 주장을 한다.

**`attach_image(source_url)`** — 소스 페이지의 대표 이미지(og:image)를 받아 stage한다(`fetch_og_image`).
`research_max_staged_images` 캡에 도달하면 거부한다. Threads 게시물에는 첫 1장만 실린다.

**`deliver_report(report, channel)`** — `channel`은 `"markdown"`|`"threads"` enum이다. `"markdown"`(기본)은 리포트를 CLI에 넘기고, 다시 부르면 수정본으로
교체한다. `"threads"`는 이 실행이 허용할 때만 게시한다. 허용되지 않은 채널과 알 수 없는 채널은 아무것도 하지
않고 스스로 고칠 수 있는 거부 문자열을 돌려준다. Threads 반환 문자열은 실제 전달량(`DeliveryStats`의 rendered,
delivered, dropped, trimmed)을 담는다. 이미 게시된 채널에 두 번째 호출이 오면 `NOT_POSTED`다. 두 번째 호출의
가장 그럴듯한 이유는 수정된 리포트이니, "Delivered"로 답하면 나가지도 않은 정정이 나갔다고 말하는 셈이다.

**인용 가드.** 검색·읽기 도구는 결과에 실려 나간 URL을 모두 `DeliveryContext.seen_urls`에
`normalize_citation_url` 형태로 적어두고, `deliver_report`는 리포트가 인용한 URL 중 그 집합에 없는 것이 하나라도
있으면 받지 않는다. 반환 문자열이 문제의 URL을 열거하니 에이전트가 그것만 빼거나 바꿔 다시 호출한다. 비교는
양쪽 모두 `shared.formatting.extract_urls`를 거치며, Markdown 링크 `[label](url)`과 괄호 안 URL, http→https
재작성을 통과시킨다. 가드는 조작된 인용을 막으려고 있으니 실제 인용을 떨어뜨리는 쪽으로는 엄격해지지 않는다.

**AgentCore 내장 도구는 쓰지 않는다.** AgentCore Gateway의 Web Search 커넥터는 결과마다 게시일과 날짜 필터를 주지만
us-east-1, eu-west-1, ap-northeast-1에만 있고(us-west-2와 ap-northeast-2는 없다) 1천 건당 약 $7로 Tavily와 비슷하다.
쓰려면 세 번째 리전에 Gateway와 서비스 역할, MCP 클라이언트 수명 관리가 붙고, `read_url`과 웹 검색 수집기 때문에
Tavily도 그대로 남아 검색 벤더가 둘이 된다. 고유 이점인 날짜는 `web_search`의 `Published:`와 `time_range`로 얻었다.
AgentCore Browser도 쓰지 않는다. 다이제스트에 실린 실제 URL 10개로 Tavily extract를 재면 8개는 전문을, 페이월
1개는 요약만 읽었고, 브라우저도 페이월은 못 넘는다. Claude의 웹 검색·fetch 서버 도구는 Bedrock에 없다.

### 9.2 리포트 전달 (`output/delivery.py`)

**`DeliveryContext`(dataclass)** — 실행별 상태다. `publish_threads`, `staged_images`, `delivered_channels`,
`partial_channels`(root만 게시되고 reply가 모두 실패한 채널. 재전송하면 공개 계정에 중복이 생긴다), `seen_urls`,
`report`(제출된 최신 Markdown), `threads_stats`(Threads 전달이 실제로 낸 렌더·게시·드롭·트림 수)가 있다. `request_context`는 contextvar로 바인딩해
상태가 실행 사이에 쌓이지 않게 하고, `current_delivery_context`는 바인딩이 없으면 새 인스턴스를 반환한다.

**`deliver_research_report`** — `MARKDOWN`은 리포트를 기록하고 `POSTED`를 돌려준다. `THREADS`는 멱등이고
`DeliveryOutcome`(`POSTED`, `NOT_POSTED`, `FAILED`)을 돌려준다. bool이 아닌 이유는 "이미 게시돼 이번 텍스트는
안 나갔다"와 "게시 실패"가 다른 답이기 때문이다.

**`_deliver_threads`** — `render_threads_research(report, max_posts=research_max_threads_posts)`로 root와
평탄한 reply chain, 그리고 드롭·트림 수를 얻는다. stage된 이미지가 있으면 첫 1장을 S3
(`{prefix}threads/research_<sha>.<ext>`)에 올려 root에 태운다. 상태 버킷이 없으면 텍스트만 게시한다.
`post_to_threads`의 결과는 NamedTuple이라 늘 truthy이니 카운트를 명시적으로 읽는다.

### 9.3 OG 이미지 첨부 (`shared/media/og_image.py`)

**`fetch_og_image(url)`** — 페이지를 브라우저 UA로 fetch해 메타 태그를 파싱한다. 우선순위는 `og:image`,
`og:image:url`, `twitter:image`, `twitter:image:src` 순이고 상대 URL은 페이지 URL로 절대화한다. 이미지는
스트리밍으로 받아 oversize 바디를 다 버퍼링하지 않고 중간에 중단한다(Content-Length 선검사와 스트림 누적 검사,
`og_image_max_bytes`와 `og_image_timeout_sec`). 렌더 가능한 래스터 타입(jpeg, png, webp, gif)만 통과시킨다.
SVG 같은 벡터나 이색 타입은 Threads fetcher가 다루지 못한다. 어떤 오류든, 미존재든, 비이미지든,
oversize든 `None`을 반환하고 절대 raise하지 않는다. 반환 타입은
`ImageAsset(data, source_url, image_url, content_type, alt)`다.

**`extension_for(content_type)`** — 이미지 MIME을 파일 확장자로 바꾼다(기본 `png`). Threads S3 키에 쓰이며,
content_type이 다운스트림까지 일관되게 전달되게 한다.

### 9.4 공유 한국어 스타일 (`shared/config.py` `KOREAN_STYLE_RULES`)

데일리 다이제스트와 딥 리서치가 함께 쓰는 산문 규약 상수다. 번역투 회피, `~다` 평서체(존댓말 금지), 콜론
나열과 쉼표로 두 문장을 잇는 것 금지, "A가 아니라 B" 같은 대비 공식 피하기가 들어 있다. 다이제스트의 `digest_language_rules`와 리서치
에이전트의 `<language>` 블록 양쪽에 합성되어 두 기능이 register와 어조에서 갈라지지 않게 한다. 같은 작성자,
같은 규칙이다.

## 10. 시각화 생성기

`agent/visuals.py`의 `VisualGenerator`는 모드 없는 자유형 이미지 생성기다. 호출자는 데일리 비주얼
파이프라인(`pipeline/daily_visual.py`의 `DailyVisualMaker`,
[§5.6](#56-데일리-비주얼-daily_visualpy)) 하나다.

**설계.** 고정된 comic이나 diagram 모드, 컷 수 파라미터가 없다. 자연어 `instruction`이 원하는 형식을 묘사하고,
데일리 비주얼에서는 그것이 2x2 4컷 만화다(에디터가 다른 구성을 고른 날만 예외다).

**생성 흐름 (`VisualGenerator.generate(instruction, source, references=..., deadline=...)`).**

1. **브리프** — `VisualSynopsisPrompt`로 Claude(Bedrock)가 단일 이미지 브리프를 만든다(`title`, `prompt`,
   `orientation`). 구조화 출력(`with_structured_output(VisualBrief)`)으로 검증된 객체를 받는다. 강제 tool choice를
   거부하는 모델(Opus 5.5, Sonnet 5.5)은 `auto`로 스키마 도구만 붙으니, 도구 호출 없이 답하면 명시적인 오류를
   낸다.
2. **이미지** — 브리프의 `prompt`로 OpenAI 이미지 API를 호출해(`b64_json`) PNG 바이트를 얻는다. 참조 이미지가
   있으면 `images.edit`, 없으면 `images.generate`다. 수십 초가 걸리는 블로킹 호출이라 `asyncio.to_thread`로 이벤트
   루프에서 분리한다. orientation(tall, portrait, square, landscape)은 `image_sizes` 딕셔너리로 size에 매핑한다. 어휘는 한 곳에만 있다. `VisualBrief.orientation`이
   `shared/constants.py`의 `VisualOrientation`으로 타이핑된 **필수** 필드이고, `PipelineConfig`가 `image_sizes`의
   키 집합이 정확히 그 어휘인지 검증한다. 매핑에 없는 orientation이면 raise한다.
3. **모더레이션** — 차단은 간헐적으로 나니 완화 문구(`visual_moderation_softening_instruction`)를 붙인 브리프로 한
   번 재생성한다. 비주얼 Lambda가 넘긴 `deadline`에 한 판을 더 돌 시간이 없으면 포기한다.
4. **검수** — `_reviewed`가 위의 검수를 하고 결함을 로그에 남긴다. 비주얼 Lambda가 넘기는 마감에서 Threads 게시
   몫(인덱싱 예산과 게시 예약)을 뺀 시각이 이미지 단계의 마감이고, 그 안에 검수 예산이 없으면 검수를 건너뛴다.
5. **게시** — 반환된 `(image_bytes, VisualBrief)`를 `DigestPublisher`가 Threads root에 싣는다. 이미지는 옵셔널이라
   렌더가 실패한 날도 텍스트로 나간다.

**기본값을 두 벌 두지 않는다.** 비주얼 노브를 모두 필수 키워드 인자로 받고, 단일 원천은 `PipelineConfig`다.

**호출 상한.** OpenAI 클라이언트를 `visual_image_timeout_sec`과 `visual_image_max_retries`로 생성한다. SDK
기본값 600초씩 2회는 15분 Lambda를 넘길 수 있다.

**품질 등급은 config가 못박는다.** `quality`를 비우면 OpenAI의 `auto`가 장당 단가가 몇 배 다른 등급 중 하나를
골라 청구가 불확정이고, 렌더 로그도 무엇을 샀는지 말하지 못한다. 등급은 컷 안 한국어 문장의 가독성도 정한다.
그래서 `visual_image_quality`를 `"xhigh"`로 고정하고, 타입을 low/medium/high/xhigh/max/auto와 빈 문자열로 좁혀 오타가 config
로드에서 걸리게 한다. 렌더는 응답이 보고하는 실제 등급과 과금 토큰 수를 로그에 남긴다(`_usage_summary`. usage가
없거나 필드명이 바뀌면 `"unknown"`으로 degrade한다).

OpenAI 키(`resolve_secret`으로 env를 먼저, 그다음 SSM에서 해석한다)가 없으면 우아하게 비활성화된다.

## 11. 인프라(CDK)

### `foundation_stack`

VPC와 ECR 리포(`{project}-{stage}-agent`, 다이제스트 이미지가 들어간다), S3 상태 버킷(CDK가 생성하는 경우
S3-managed 암호화와 버저닝, 퍼블릭 차단, SSL 강제), ECS Fargate RSSHub 서비스와 service-discovery, SNS 알림
토픽과 선택적 이메일 구독, 비동기 DLQ, AgentCore Memory 리소스(전략 없음, 실행 역할 없음), 그리고 Lambda 실행
역할을 만든다.

**RSSHub 서비스는 `aws.rsshub_desired_count`(기본 0)로 스케일된다.** 다이제스트는 이 서비스에 도달하지 않는다.
`RSSHubCollector`는 S3 park 파일을 먼저 읽고, 쓸 수 있으면 도달성 확인조차 하기 전에 리턴하며, 로컬 sync cron이
매 실행 30분 전에 그 파일을 갱신한다. 그래서 다이제스트 로그의 X 항목은 항상 park 파일에서 나온다. 반면 이
서비스는 계정의 유일한 상시 Fargate 태스크로 월 약 $40이다. 태스크 정의는 그대로 배포되니 1로 올리면 로컬 sync가
멈춘 날의 AWS 폴백이 복구된다. 그 상태는 이미 헬스 STALE로 표면화된다.

**IAM은 최소 권한으로 잡는다.**

- `ssm:GetParameter`는 `/{project}/{stage}/*`로 스코프한다. 코드는 `get_parameter`만 부른다. `ssm:PutParameter`는
  Threads 토큰 파라미터 하나로 스코프한다.
- `bedrock:InvokeModel`과 `bedrock:CountTokens`는 foundation-model과 inference-profile,
  application-inference-profile ARN으로 스코프한다. Lambda는 스트리밍하지 않으니 스트리밍 호출 액션은 주지 않는다. 마지막 것은 별도 리소스 타입이며 필수다. 모델 리졸버가 비용 귀속용 application inference
  profile을 선호하니, 빠뜨리면 프로필이 존재하는 순간 모든 Bedrock 호출이 AccessDenied가 된다. 그날
  다이제스트 전체가 죽는다는 뜻이다.
- `lambda:InvokeFunction`은 데일리 비주얼 함수 하나(`{project}-{stage}-visual`)로 스코프한다. 파이프라인
  역할의 유일한 cross-function 호출이 그 비동기 fan-out이다. ARN은 application_stack이 붙이는 리터럴 이름으로 만든다. 함수
  객체를 참조하면 `fnd → app` 순환 의존이 된다.
- S3는 프로젝트 루트 prefix(`config.aws.s3_prefix`에 `/*`를 붙인 것)의 `s3:GetObject`와 `s3:PutObject`, 그리고
  버킷의 `s3:ListBucket`만 준다. 코드가 부르는 것은 get, put, 그리고 같은 역할로 서명하는 presigned GET뿐이다.
  상태 버킷은 기존 공유 버킷일 수 있어서 삭제, 보존, 법적 보존 액션과 버킷 단위 Get은 주지 않는다. state_store의
  `digest_state`와 수집기 park 파일, 데일리 비주얼의 `threads/*.png`가 그 prefix 안에 있다. `ListBucket`에
  `s3:prefix` 조건을 걸지 않는 이유가 있다. 없는 키를 GetObject할 때 S3는 `ListBucket`으로 NoSuchKey와
  AccessDenied를 가르는데, 그 검사에는 `s3:prefix`가 실리지 않아서 조건을 걸면 없는 키 읽기가 전부 AccessDenied가
  된다. 그러면 처음 쓰는 상태 파일이 "히스토리 없음"이 아니라 읽기 실패가 된다.
- Memory 데이터플레인 액션(`CreateEvent`, `ListEvents`)은 `memory/*`가 아니라 **이 스택이 만든 memory ARN**(과 그
  `/*` 하위 리소스)으로 스코프한다. 세션 id가 날짜로 정해지니 `ListSessions`는 필요 없다. 와일드카드로 두면 공유
  계정의 다른 프로젝트 대화 상태까지 읽고 쓸 수 있게 된다.
- 로그 쓰기는 Lambda VPC 관리형 정책(`AWSLambdaVPCAccessExecutionRole`)이 덮는다. 다이제스트 Lambda가 VPC 안에 있어
  어차피 필요한 정책이다.

### `application_stack`

다이제스트 Lambda(DockerImage, VPC 안), 데일리 비주얼 Lambda(DockerImage, 비동기, VPC 밖), Threads 토큰 갱신
Lambda(DockerImage), EventBridge 일일 다이제스트 크론(설정 기반 시와 분), EventBridge Threads 토큰 갱신
스케줄(`threads_token_refresh_days` 기본 약 50일 주기로, 60일 만료 안쪽에서 토큰을 갱신해 SSM에 재기록한다),
시크릿용 SSM 파라미터, 그리고 SNS로 향하는 CloudWatch 알람([§12](#12-관측성) 참조)을 만든다. 세 Lambda가 같은
이미지를 쓰고 핸들러만 다르다.

비주얼 Lambda가 VPC 밖인 이유는 부르는 것이 전부 공개 엔드포인트이기 때문이다(Bedrock, S3, AgentCore Memory,
SNS, OpenAI, Threads). VPC는 Threads 게시의 유일한 경로에 ENI와 NAT 의존만 더했다. 다이제스트 Lambda는 RSSHub
Fargate 서비스에 닿아야 하니 VPC 안에 남는다.

**재시도와 DLQ.** 다이제스트와 비주얼 Lambda는 `retry_attempts=0`에 SQS
DLQ(`foundation.async_dlq`로 `on_failure`)를 붙인다. Threads 갱신 Lambda도 `retry_attempts=0`인데, 명시값이
없으면 비동기 재시도 2회가 기본이라 갱신 엔드포인트를 재호출한다. 파이프라인이 멱등이 아니고 Threads는
idempotency key가 없어서 재시도가 이중 게시를 일으키니, 자동 재시도 대신 실패 건을 DLQ에 남겨 점검과 수동
리플레이를 한다. 핸들러는 [§8](#핸들러-예외-전파)처럼 raise해서 Errors 알람이 뜨게 한다.

**로그 그룹은 선언한다.** 세 Lambda 모두 `log_group=logs.LogGroup(...)`(retention 30일, removal DESTROY)을
직접 받는다. deprecated된 `log_retention=` prop은 속성을 세우는 것이 아니라 함수마다 `Custom::LogRetention`
Lambda를 하나씩 렌더하고, 그 각각이 자기 실행 역할에 `logs:PutRetentionPolicy`와 `logs:DeleteRetentionPolicy`를
Resource `"*"`로 든다. 공유 계정에서는 아무 로그 그룹의 보존 기간이나 줄이거나 없앨 수 있는 권한이다.
선언 방식은 그런 Lambda와 와일드카드 정책을 아예 만들지 않고,
로그 그룹이 템플릿 리소스가 되므로 `test_sensitive_actions_not_wildcard_resource`와 retention 단정이 두 스택
전체를 보게 된다. RSSHub 컨테이너의 그룹도 `ecs.LogDrivers.aws_logs(log_retention=...)`으로 보존 기간을 받는다.
그러지 않으면 `RetentionInDays` 없이 `DeletionPolicy: Retain`이 붙어 두 스택에서 유일하게 영구 보존되는 그룹이
된다.

⚠️ CloudFormation은 **이미 존재하는 로그 그룹을 adopt하지 않는다.** 이 선언 없이 배포된 적이 있는 스택이라면
Lambda가 암묵적으로 만들어둔 `/aws/lambda/<function>`을 먼저 삭제하거나 import해야 한다.

**RSSHub 보안그룹 ingress.** 다이제스트 Lambda SG에서 RSSHub Fargate 서비스 SG(`RSSHUB_PORT`)로 가는 인그레스를
이 스택에서 `CfnSecurityGroupIngress`로 추가한다. Lambda SG는 기본이 전체 egress라 ingress만 빠지기 쉽고, 그
상태에서는 AWS의 X 피드 fetch가 전부 타임아웃한다(park 파일이 항목을 공급하므로 다이제스트만 보면 정상으로
읽힌다). `connections.allow_from()`은 규칙을 foundation 쪽에 붙여 `fnd → app` 순환 참조가 되니 명시적 인그레스
리소스를 쓴다.

**시크릿 처리.** 스택은 파라미터 경로만 만들고 값은 `SSM_PLACEHOLDER`로 둔다. CloudFormation은 SecureString을
만들 수 없고, 값을 스택에 넘기면 템플릿과 CDK staging 버킷, `GetTemplate` 응답에 평문으로 박힌다. 실제 값은
배포 후 `scripts/put_secrets.py`가 SecureString으로 out-of-band 기록한다(`--dry-run` 미리보기, `--verify` 읽기
전용 점검, `--force`로만 기존 SecureString 덮어쓰기).

재배포는 값을 건드리지 않는다. 플레이스홀더가 바뀌지 않으니 CloudFormation이 리소스를 업데이트하지 않는다.
⚠️ 그래서 이 리소스의 템플릿 속성은 무엇도 추가하거나 변경하면 안 된다. Description 하나만 붙여도
CloudFormation이 PutParameter를 다시 실행해 살아 있는 시크릿 위에 플레이스홀더를 쓴다.
`tests/test_infrastructure.py`가 렌더된 속성 집합을 고정해 그 편집을 배포 전에 잡는다.

`String`에서 `SecureString`으로 가는 타입 변경은 SSM이 `ValidationException`으로 거절한다. 그래서 스크립트는
CloudFormation이 남긴 정확한 상태, 곧 `Type == String`이면서 `Value == SSM_PLACEHOLDER`일 때만 파라미터를
삭제하고 SecureString으로 재생성한다. 이것이 "시크릿은 SecureString"이라는 주장을 실제로 참으로 만드는
단계다. 그 밖의 `String`은 실제 값을 담고 있으니 절대 삭제하지 않고 값만 제자리에 쓰고 시끄럽게 알린다. 살아
있는 자격증명을 잃는 것은 암호화되지 않은 것보다 나쁘다. 삭제는 됐는데 재기록이 실패한 경우는 파라미터 자체가
없는 가장 나쁜 상태이니 `FAILED` 목록과 non-zero 종료, 그리고 복구 명령 전문
(`aws ssm put-parameter --name … --type SecureString …`)을 출력한다. 한 파라미터의 실패로 루프가 죽지 않는다.
첫 거절에서 루프가 죽으면 그 뒤의 시크릿이 전부 플레이스홀더로, 런타임에서는 미설정인 상태로 남는다. `--verify`는
값이 있어도 `String`이면 PLAINTEXT로 따로 보고하고 non-zero로 끝낸다. 설정된 것과 암호화된 것은 다르다. 보완
통제는 스코프된 IAM 읽기 정책이다.

Threads 갱신 Lambda는 갱신된 토큰을 `put_parameter(Overwrite=True)`로, `Type`을 지정하지 않고 덮어쓴다.
`Type=SecureString`을 얹는 것은 타입 변경이라 파라미터가 아직 `String`이면 SSM이 `ValidationException`으로
거절하고, 그러면 토큰이 갱신되지 않은 채 60일 뒤 Threads 전달이 끊긴다. `Type`을 생략하면 기존 타입(마이그레이션
후에는 SecureString)을 유지하고 값만 갱신한다. 그 생략은 암호화되지 않은 파라미터도 그대로 두니, 암호화 여부는
갱신 Lambda가 매번 확인하지 않고 `put_secrets.py --verify`가 보고한다.

## 12. 관측성

### 로깅 (`shared/logger.py`)

AWS에서는 구조화 JSON 로그를 쓰고(`is_running_in_aws()`) 로컬에서는 사람이 읽는 형식을 쓴다. correlation id는
`ContextVar` 기반(`set_correlation_id`와 `get_correlation_id`)으로 모든 레코드에 주입되며, Lambda 요청 id에서
시드된다.

id는 서비스 경계를 넘어 전달된다. 다이제스트 Lambda는 비주얼 Lambda 호출 payload에 자기 `correlation_id`를
넣고, 비주얼은 DLQ 봉투의 `requestPayload`까지 포함해 그 값을 자기 요청 id보다 먼저 쓴다. 보내는 쪽이 빠지면 매
홉이 새 id를 만들어 한 번의 다이제스트를 두 Lambda에 걸쳐 추적할 수 없다.

### 알람

CloudWatch 알람 8개가 모두 SNS 알림 토픽으로, 그리고 이메일로 라우팅된다(`_add_alarms`).

- **Lambda별 Errors 3개와 Timeout 3개** — digest와 visual, threads-refresh 각각에 대해 예외(Errors)와 타임아웃
  임박(max Duration이 설정 타임아웃의 90% 이상) 알람을 둔다. 타임아웃은 Errors로 집계되지 않으니 별도가 필요하다.
- **EmptyDigestAlarm** — EMF `OmniSummary/DigestItemsPublished`(실행당 한 번)에 대한 24시간 윈도 알람이다.
  CloudWatch가 `evaluation_periods × period ≤ 86400s`로 제한하니 그 이상은 배포 시 거부된다. 0건 게시와 그날
  미실행을 모두 포착한다(missing-data=BREACHING).
- **AsyncDLQAlarm** — async DLQ에 메시지가 쌓이면 알린다. 실패한 digest나 visual 실행이 대기 중이라는 뜻이다.

딥 리서치는 로컬 CLI라 알람이 없다. 실패는 CLI의 종료 코드와 `error` 필드로 호출자에게 바로 돌아간다.

## 13. 테스트와 CI/CD

### 테스트 (`tests/`, pytest, `asyncio_mode=auto`)

약 1,100개의 테스트가 있고 커버리지 게이트는 90%다. 게이트는 측정값을 따라 움직여야 한다.
10포인트 이상 낮게 방치된 게이트는 진짜 회귀를 오타만큼이나 편하게 통과시킨다. `tests/conftest.py`의 autouse
픽스처가 앰비언트 시크릿과 인프라 env를 monkeypatch로 비우고 SSM 클라이언트를 막아 hermetic하게 만든다. 개발자의
`.env`나 AWS 프로파일에 결과가 좌우되지 않고, 실 SSM 왕복에 드는 수십 초도 들지 않는다. 같은 파일이
`BedrockCrossRegionModelHelper._resolution_cache`(클래스 변수)도 매 테스트마다 비운다. 비우지 않으면 해석 경로를
처음 건드린 테스트가 이후 모든 테스트의 해석 결과를 결정한다.

커버하는 영역은 다음과 같다.

- 수집기(공용 `collectors.base.fetch_feed`를 모킹해 재시도와 분류, 헬스 판정을 돈다).
- 집계기와 랭커 파싱, 슬롯과 origin-cap 로직, 그리고 배치 재시도와 전면 실패 승격, fan-out 상한.
- **`pipeline.runner.run_pipeline` 오케스트레이션(`test_run_pipeline.py`)** — 집계 후 빈 입력과 임계 미달 조기 반환,
  원장과 leads 기록, 트렌드 갱신, 원장과 AgentCore 스냅샷 양쪽에서 시드하는 cross-day dedup(URL 정규화 포함),
  dry-run이 상태를 쓰지 않고 아무 채널에도 보내지 않음, 로컬 인라인 비주얼 실행과 그 실패의 non-fatal성. LLM과
  네트워크 협력자만 스텁하고 원장과 롤링 로그, 집계는 임시 디렉터리 StateStore로 실제 실행한다.
- **`StateStore`(`test_state_store.py`)** — `S3StateStore`의 키 prefix와 UTF-8 인코딩, NoSuchKey와 기타
  ClientError의 구분, `create_state_store`의 버킷 기반 선택(AWS 밖에서도 `STATE_BUCKET`이면 S3와 프로파일 세션).
- **WebSearch `collect()`(`test_web_search.py`)** — 쿼리별 fan-out과 URL dedup, per-trend domains와 topic 전달,
  전면 실패 승격, 부분 실패 허용, LLM 정제 2단계와 그 실패의 non-fatal성.
- 헬스 리포트와 logger.
- 메모리 스토어(로컬과 AgentCore 모킹).
- 다이제스트 핸들러 알림.
- 딥 리서치 — 구성과 프롬프트 보간(`test_research_agent.py`), 도구 8개(`test_research_tools.py`), 리서치
  백엔드(`test_research_backends.py`), 리포트 전달(`test_delivery.py`), 렌더러(`test_renderers.py`), OG
  이미지(`test_og_image.py`), CLI의 JSON 계약과 종료 코드, `Limits` 전달(`test_research_cli.py`).
- `VisualGenerator`(`test_visuals.py`)와 데일리 비주얼(`test_daily_visual.py`).
- trend_tracker(보존 기간 삭제, 재실행 시 `last_seen` 도출, 같은 날 재실행이 오늘 증거를 문맥에서 빼는지, evidence 캡).
- CDK assertion(`aws-cdk.assertions`로 두 스택을 검증한다).

핵심 로직의 회귀 테스트는 **변이로 검증했다.** 코드를 일부러 망가뜨렸을 때 실패하는지 확인한 테스트들이다.
그라운딩 수정이 올바른 필드에 붙고 빈 마커는 원래 값을 지키는지, 렌더 도중 다른 실행이 날짜를 차지하면 게시하지
않는지, Threads 원장이 가장 최근 날짜를 남기고 다른 실행의 마커를 넘겨받지 않는지, URL 원장의 TTL 경계, 하드 컷된
본문 뒤에도 URL이 온전히 남는지, fill 패스 순서(소스 캡 경계, origin 캡을 소스 캡보다 먼저 완화, 점수순 반환),
배치가 소스를 섞는지가 그렇다. 반대로 결함을 고정하던 테스트(5000자 브리프를 거부하라는 테스트 등)는 고쳤다.

### CI (`.github/workflows/ci.yml`)

**락파일 고정.** `uv lock --check`와 모든 잡의 `uv sync --frozen`으로, 리뷰되고 테스트된 것과 다른 버전이
조용히 해석되지 않게 한다.

**lint와 타입 체크.** ruff와 black `--check`, 그리고 `mypy .`로 레포 전체를 본다. 검사 경로를 열거하지 않고
제외만 `[tool.mypy] exclude`에 둔다. 열거식은 새로 생긴 최상위 모듈을 조용히 게이트 밖에 남긴다.

**테스트와 커버리지 게이트.** 범위와 `fail_under`가 `pyproject.toml`(`[tool.coverage.*]`)에 있어서 커맨드라인
수정으로 좁혀지지 않는다.

**잡 상한과 캐시.** 모든 잡에 `timeout-minutes`가 있다. 기본 6시간 러너 타임아웃에 걸린 채 멈춘 빌드를
방치하지 않기 위해서다. uv 휠 캐시는 `uv.lock`으로 키를 잡아 의존성이 바뀌면 재설치되니, 깨진 의존성 집합을
캐시가 가릴 수 없다. Node는 npm 캐시를 쓴다.

**레포 고정 CDK CLI로 오프라인 synth.** Node 22와 `npm ci`로 `package.json`에 핀된 `aws-cdk`를 설치하고
`npx cdk synth -a "uv run python scripts/ci_synth.py"`를 돈다. `npm ci || npm install` 같은 폴백을 두면 lock
부재나 불일치를 삼키고 다른 CLI를 깔아서 이 잡의 의미가 사라진다. 인프로세스 `app.synth()`가 아니라 실제 CLI를
태우는 이유는 CLI와 `aws-cdk-lib`의 cloud-assembly 스키마 핸드셰이크를 검증하는 것이다. 글로벌 CLI가 라이브러리보다
뒤처져 배포가 스키마 미스매치로 깨지는 부류를 PR 단계에서 잡는다. `ci_synth`는 `vpc_id`를 비우고 env-agnostic
계정을 써서 자격증명 없이 완전 오프라인으로 돈다.

**Docker 빌드와 이미지 import 체크.** 다이제스트 이미지를 `load: true`(네이티브 amd64)로 빌드해 로컬 데몬에
올린 뒤, `docker run --rm --network none --entrypoint python`으로 세 `lambda_handlers.*`를 import하고
`Config.load()`가 템플릿을 읽는지 확인한다. 빌드만으로는 import가 한 번도 실행되지 않으니
COPY 누락이나 개발자 머신에서만 해석되는 모듈이 그대로 통과한다. 자격증명 없이 `--network none`이니 import
시점의 AWS 호출이나 HTTP fetch가 콜드스타트가 아니라 CI에서 깨진다. import 체크는 빌드가 얼마나 캐시됐든
로드된 이미지에 대해 항상 실행되니 레이어 캐시가 실패를 건너뛸 수 없다. 캐시는 `type=gha`이며 이미지별 scope를
쓴다.

**의존성과 시크릿 스캔(`security` 잡).** `uv.lock`이 핀한 정확한 집합, 곧 이미지가 설치하는 그 버전들을
`pip-audit --strict`로 감사한다. pyproject 범위를 재해석해 감사하면 배포되지 않는 버전을 검사하게 된다. 감사
대상은 requirements 파일이 아니라 설치된 트리다. `uv sync --frozen --no-dev --no-install-project`로 잠긴 집합을
디스크에 올린 뒤 `pip-audit --path <purelib>`로 `dist-info`만 읽는다. 아무것도 빌드하지
않고, 실제 배포되는 플랫폼으로 이미 좁혀진 집합이다. 이 `<purelib>`는 `.venv/bin/python`의 `sysconfig`에서
뽑아내고 디렉터리가 비어 있지 않은지까지 확인한다. 존재하지 않는 경로를 넘기면 pip-audit가 "No known
vulnerabilities found"를 찍고 0으로 끝나기 때문이다. 경로에 `python3.12`를 문자열로 박아두면 그 게이트가 인터프리터
버전 하나에 매달린다. 그래서 `.python-version`을 추적 파일로 두고 CI의 `uv python install`도 인수 없이 그 핀을
읽는다.

**pip-audit은 `uvx --from pip-audit==2.9.0 pip-audit --strict --path <purelib>`로만 부른다.** `-r` 모드는
`ensurepip`로 자체 venv를 만드는데 uv 관리 파이썬에는 ensurepip가 없다. `--no-install-project`도 필수다. editable
배포는 감사 불가로 보고되고 `--strict`가 이를 실패로 바꾼다.

그리고 `gitleaks`를 돌린다(`fetch-depth: 0`). PR 실행은 `GITHUB_TOKEN`이 있어야 하니 step env로 넘긴다.
`config/config.yaml`은 gitignore인데 실제 값을 갖고 있으니 잘못 add된 파일 하나가 곧 유출이다.

**스케줄 실행.** `pip-audit --strict`는 잠긴 집합에 알려진 권고가 0건이기를 요구하니, 전이 의존성에 새 권고가
공개되면 잠금 파일이 그대로여도 게이트가 빨개져야 한다. 그래서 CI는 push와 PR 외에 매주 월요일 00:00 UTC에도
돈다. 해소 순서는 `uv lock --upgrade-package <pkg>`, 미사용 의존성 제거, 상한 완화이며 `--ignore-vuln`은 도달
불가를 코드로 확인한 뒤의 마지막 수단이다.

**uv 버전은 하나다.** CI의 `UV_VERSION`과 Dockerfile이 0.6.14로 같다. 더 새 uv로 `uv lock`을 돌리면 잠금 파일의
`revision`이 바뀌어 0.6.14의 `uv lock --check`가 실패하니, 잠금 파일은 `uvx --from uv==0.6.14 uv lock`으로 갱신한다.

**액션 고정과 워크플로 검사.** 서드파티 액션은 커밋 SHA로 고정하고(주석에 메이저 태그), 모든 checkout은
`persist-credentials: false`다. `.github/dependabot.yml`이 GitHub Actions를 매주 갱신한다(minor와 patch는 한 PR로
묶는다). `.github/workflows/workflow-lint.yml`은 체크섬을 검증한 actionlint로 워크플로와 내장 셸 스크립트를
shellcheck까지 검사한다. Python과 npm 의존성은 dependabot 대상이 아니다. 잠금 파일 갱신은 위의 주간 pip-audit가
신호를 준다.

**이미지 하드닝.** Dockerfile은 `uv.lock`이 핀한 집합을 설치한다(`uv export`로 뽑아
`uv pip install --system`하고 프로젝트 자신은 `--no-deps`). `pip install .`은 빌드 시점에 pyproject 범위를 다시
해석하므로 CI가 테스트한 적 없는 의존성 집합이 Lambda에서 돌 수 있다. 의존성 레이어를 소스 COPY보다 먼저 두어
코드 변경이 레이어를 재사용한다. 이미지에는 Lambda가 쓰는 패키지만 들어간다(로컬 CLI인 `main.py`는 넣지 않고
포트도 열지 않는다). `agent/__init__.py`는 비어 있어서 Lambda가 `agent.visuals`나 `agent.tool_state`를 import해도
리서치 에이전트(Strands)는 로드되지 않는다.

런타임은 non-root(uid 10001)다. 그래서 `shared/logger.py`는 로그 디렉터리를 무가드로 `mkdir`하지 않는다. 모듈
스코프에서 그것을 하면 쓰기 권한이 없는 환경에서 import 자체가 PermissionError로 죽는다. 프로덕션은
`is_running_in_aws()` 가드 안이라 무사하지만 CI 임포트 검사와 읽기 전용 체크아웃, 샌드박스가 모두 해당된다.
mkdir과 FileHandler 열기 둘 다 실패하면 콘솔 전용으로 degrade한다. `.dockerignore`는 `.env`와 `.venv`, `logs/`,
`cdk.out`을 빌드 컨텍스트에서 뺀다.

**pre-commit (`.pre-commit-config.yaml`, `uv run pre-commit install`).** pre-commit은 dev extra에 들어 있다. ruff와
black, YAML/JSON, 파일 끝·공백, private-key 탐지, `uv lock --check`를 커밋 전에 돌린다. mypy는 의도적으로 훅에서 제외했다. 전체
의존성 해석이 필요하고 느려서, 사람들이 `--no-verify`를 쓰기 시작하면 얻는 것보다 잃는 게 크다.

**CI는 추적되는 config로 synth한다.** `scripts/ci_synth.py`와 인프라 assertion 테스트 모두
`config/config-template.yaml`을 명시적으로 로드한다. 이유는 [§3](#3-설정)에 있다.

## 14. 주요 명령어

```bash
uv run python main.py --dry-run --sources web_search   # 부분 dry run(상태 쓰기와 게시 없음)
uv run python main.py                                   # 전체 파이프라인 + 비주얼 + Threads(config에서 켰을 때)
uv run python research_cli.py "<토픽>"                  # 딥 리서치, JSON을 stdout에(게시 없음)
uv run python research_cli.py "<토픽>" --publish threads # 리포트를 Threads에도 게시
./scripts/sync_all_to_s3.sh                             # 로컬 → S3: X(RSSHub, 꺼져 있으면 컨테이너를 켠다) + YouTube
uv run python scripts/sync_to_s3.py rsshub              # 소스 하나만
uv run python -m pytest tests/ -v                       # 테스트
uv run pre-commit run --all-files                       # ruff, black, 파일 검사, uv lock --check
uv run mypy .                                           # 레포 전체 타입 체크
uv run python scripts/ci_synth.py                       # 오프라인 CDK synth(인프로세스)

# 배포: 다이제스트 이미지(amd64)를 빌드/푸시하고, 푸시된 sha256 digest를 DIGEST_IMAGE_REF로 넘긴다(태그 문자열이
# 안 바뀌면 CFN이 Lambda를 재배포하지 않고, 빠뜨리면 Lambda가 :latest로 고정된다). CDK CLI는 package.json에 핀된
# 것을 npx로 쓰고, deploy.py는 계정을 CLI가 넣는 CDK_DEFAULT_ACCOUNT에서 읽는다.
npm install                                             # 1회 — 핀된 CDK CLI 설치
npx cdk bootstrap -a "uv run python scripts/deploy.py"  # 계정+리전당 1회
# 빈 계정에서는 foundation부터(ECR 리포가 거기 있다): npx cdk deploy '*-foundation' -a "uv run python scripts/deploy.py"
aws ecr get-login-password --profile research --region ap-northeast-2 | docker login --username AWS --password-stdin <account>.dkr.ecr.ap-northeast-2.amazonaws.com
docker build --platform linux/amd64 --provenance=false -t <ecr>:latest . && docker push <ecr>:latest
export DIGEST_IMAGE_REF=sha256:<pushed>
AWS_PROFILE=research npx cdk deploy --all -a "uv run python scripts/deploy.py"
# 시크릿은 템플릿에 없다. 배포 직후 실제 값을 SecureString으로 기록한다(§3.5).
AWS_PROFILE=research uv run python scripts/put_secrets.py              # --dry-run / --verify / --force
# 비용 귀속용 application inference profile(이름은 config의 project/stage). 계정/스테이지당 1회, 단계 모델을 바꿨을 때도 다시:
AWS_PROFILE=research uv run python scripts/put_inference_profiles.py   # --dry-run / --delete
uv run pre-commit install                               # 1회 — 커밋 전 게이트
```

**다이어그램.** README가 싣는 두 그림은 관리 방식이 다르다.

- `docs/diagrams/architecture.drawio`는 손으로 편집하고
  `drawio -x -f png -s 2 -b 10 -o docs/diagrams/architecture.png docs/diagrams/architecture.drawio`로 내보낸다.
  `-e`는 쓰지 않는다(하류 이미지 처리가 깨진다).
- `docs/diagrams/concept-pipeline.excalidraw`는 제자리에서 편집하고
  `node scripts/render_excalidraw.mjs docs/diagrams/concept-pipeline.excalidraw docs/diagrams/concept-pipeline.png`로
  다시 그린다. 이 스크립트는 폰트를 임베드한 SVG를 헤드리스 브라우저로 캡처해 손글씨 폰트를 살린다. 일반
  `excalidraw-to-png` 경로는 resvg로 래스터화하면서 임베드 폰트를 버리고 sans-serif로 떨어진다.
