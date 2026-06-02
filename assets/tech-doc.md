# OmniSummary — 기술 문서

> OmniSummary의 상세한 line-by-line 기술 레퍼런스를 담은 단일 문서입니다.
> 상위 수준 개요는 `README.md`와 `.claude/CLAUDE.md`에 있고, 이 문서는 심화 레퍼런스입니다.

## 1. 개요

OmniSummary는 능동형(proactive) AI/ML 일일 다이제스트 시스템입니다. 매일 정해진 스케줄에 5개 소스
계열에서 콘텐츠를 수집하고, 집계·중복 제거 후 LLM으로 순위를 매기고, 한국어 에디토리얼 다이제스트를
생성해 Slack으로 전달하며, 상태를 **Bedrock AgentCore Memory**에 저장합니다. 후속 에이전트(AgentCore
Runtime 위의 Strands)는 다이제스트 항목에 대한 질문에 답하고 시각화(만화, 다이어그램)를 생성할 수
있습니다. 운영 헬스는 소스별로 리포팅되며 SNS 이메일로 알림됩니다.

```
[EventBridge 크론] → [다이제스트 Lambda (Docker)]
   → 수집기 (RSS, Reddit, YouTube, WebSearch, X via RSSHub/S3)
   → 집계기 (URL + 제목 중복 제거)
   → 랭커 (Bedrock Claude Opus 4.8, 소스 슬롯 + origin 다양성)
   → 트렌드 트래커 (StateStore의 trends.md)
   → 다이제스트 생성기 (Bedrock Claude Sonnet 4.6, 한국어 Slack mrkdwn)
   → Slack 전달
   → AgentCore Memory (다이제스트 스냅샷 + 트렌드 사실)
   → 실패한 소스가 있으면 SNS 알림

[Slack 멘션] → [API Gateway + WAF] → [Slack Lambda]
   → 비동기 self-invoke → [Bedrock AgentCore Runtime: Strands 에이전트]
   → 도구: get_detail, search_papers, search_community, search_related_news, recall_trends, make_visual
   → AgentCore Memory에서 다이제스트 상태를 읽고, Slack에 답변/이미지를 게시
```

## 2. 저장소 구조

| 경로 | 책임 |
|------|------|
| `collectors/` | `BaseCollector` ABC + RSS, Reddit(.rss 피드), RSSHub(X/Twitter), YouTube, WebSearch(Tavily) |
| `pipeline/` | `ContentAggregator`, `ContentRanker`, `DigestGenerator`, `TrendTracker` |
| `agent/` | Strands 에이전트, 도구, `DigestStateManager`, `VisualGenerator`(만화/다이어그램) |
| `agent_runtime/` | Bedrock AgentCore HTTP 서버(`BedrockAgentCoreApp`) |
| `shared/` | config, models, constants, utils(Bedrock 팩토리), logger, prompts, state_store, **memory**, proxy |
| `output/` | Slack 전달(텍스트 + 이미지 업로드) |
| `lambda_handlers/` | 다이제스트 핸들러, Slack 이벤트 핸들러 |
| `infrastructure/` | CDK `foundation_stack` + `application_stack` |
| `scripts/` | `deploy.py`, `ci_synth.py`, `sync_rsshub_to_s3.py` |

## 3. 설정(Configuration)

`config/config.yaml` → `shared/config.py`의 Pydantic 모델로 `Config.load()`를 통해 로드됩니다. 시크릿은
`.env`(로컬) 또는 SSM Parameter Store의 `/{project}/{stage}/{name}` 경로(AWS)에서 옵니다.

주요 설정 그룹:
- `collectors.*` — 각각 `BaseCollectorConfig`를 상속(`enabled`, `lookback_hours`, `reference_time`,
  `request_timeout`, `max_retries`, `retry_backoff_sec`).
- `pipeline` — `top_n`, `min_score`, `ranking_model`(Opus 4.8), `digest_model`(Sonnet 4.6),
  `source_slots`, `source_cap_multiplier`, **`max_per_origin`**(채널/작성자/서브레딧당 상한),
  `origin_weights`, `origin_weight_default`, `trend_retention_days`.
- `aws` — region, profile, project/stage, `digest_cron_hour/minute`, `api_throttle_*`, `waf_rate_limit`.

환경 변수(`.env`): `SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN`, `SLACK_CHANNEL_ID`, `TAVILY_API_KEY`,
`YOUTUBE_API_KEY`, `OPENAI_API_KEY`, `ALERT_EMAIL`,
`CLOUDFLARE_PROXY_URL`/`CLOUDFLARE_PROXY_TOKEN`. AWS에서는 `MEMORY_ID`, `ALERT_SNS_TOPIC_ARN`,
`STATE_BUCKET`, `RSSHUB_BASE_URL`, `PROJECT_NAME`, `STAGE`를 CDK가 주입합니다.

## 4. 수집기(Collectors)

모든 수집기는 `BaseCollector.collect() -> list[CollectedItem]`을 구현하고
`cutoff_datetime(lookback_hours, reference_time)`(`collectors/base.py`)로 필터링합니다.

- **RSS** (`rss.py`): `config.collectors.rss.feeds`에 대해 feedparser 사용; 메타데이터 `feed_url`, `feed_title`.
- **Reddit** (`reddit.py`): **공개 `.rss` 피드** 사용. Reddit이 셀프서비스 OAuth 앱 생성을 동결하고
  (Responsible Builder Policy, 2025-11) `.json` API는 데이터센터 IP를 차단했지만 `.rss` 피드는 열려 있음.
  `https://www.reddit.com/r/{sub}/{sort}/.rss`를 Cloudflare 프록시(`get_proxied_url`) 경유로 가져와 AWS
  Lambda IP에서도 동작. 자격증명·앱 등록 불필요. 트레이드오프: RSS엔 `score`/`num_comments`(engagement)가
  없어 랭킹은 LLM 품질 판단에 의존.
- **RSSHub** (`rsshub.py`): 로컬/컨테이너 RSSHub를 통한 X/Twitter 피드; S3에 사전 동기화된 스냅샷
  (`rsshub_items.json`)도 로드 가능. 실패/빈 계정을 자체 추적하며 `error_rate_threshold` 보유.
- **YouTube** (`youtube.py`): `YOUTUBE_API_KEY`가 있으면 YouTube Data API, 없으면 프록시 경유 RSS 폴백.
  `max_videos_per_channel=1`로 고빈도 채널이 후보 풀을 독점하지 못하게 함.
- **WebSearch** (`web_search.py`): LLM 쿼리 정제(`RefineQueryPrompt`)를 곁들인 Tavily 검색.

`gather_collector_results()`는 수집기를 동시 실행하고 작업별 예외를 삼킴(로깅만, 평탄한 리스트 반환).
헬스 리포팅용으로 `main.run_collectors_with_health()`가 동일 작업을 실행하되 `HealthReport`(§8 참조)를
반환 — `gather_collector_results`는 다른 호출자들을 위해 그대로 유지.

## 5. 파이프라인(Pipeline)

1. **집계기** (`aggregator.py`): URL → 정규화 제목 순으로 중복 제거; 중복의 메타데이터 병합.
2. **랭커** (`ranker.py`): 항목 포맷팅(engagement + origin 포함), Claude Opus 4.8로 `RankingPrompt` 호출,
   JSON 점수 파싱, `origin_weights`(미등록 origin엔 `origin_weight_default`) 적용, `min_score` 필터,
   이후 `_apply_source_slots`:
   - `source_slots`로 소스별 기본 슬롯 채우기,
   - `source_cap_multiplier × slot`까지 오버플로 채우기,
   - **`max_per_origin`**으로 하나의 origin 키(채널/작성자/서브레딧)가 차지하는 항목 수 제한 — 단일 채널
     독점에 대한 근본 해결책. origin은 `_resolve_origin_key`로 해석(YouTube→channel_url, Reddit→subreddit,
     RSS→feed_url, X→author).
3. **트렌드 트래커** (`trend_tracker.py`): `StateStore`를 통해 `trends.md` 유지; 아카이브 이력 병합.
4. **다이제스트 생성기** (`digest_generator.py`): Claude Sonnet 4.6로 `DigestPrompt` → 한국어 Slack mrkdwn;
   `sanitize_slack_mrkdwn`이 출력 정규화.

## 6. LLM 팩토리 (`shared/utils.py`)

`BedrockLanguageModelFactory.get_model(model_id, **kwargs)`는 모델 역량(`_LANGUAGE_MODEL_INFO`)에 맞게
구성된 `ChatBedrock`/`ChatBedrockConverse`를 반환합니다: thinking, 1M 컨텍스트, 성능 레이턴시, 프롬프트
캐싱. `BedrockCrossRegionModelHelper`가 가능 시 `global.`/`apac.` inference-profile ID를 해석합니다. 모델
ID는 `shared/constants.py`(`LanguageModelId`)에 열거되며, 최신은 Opus 4.8 / Sonnet 4.6입니다.

`resolve_secret(env_var, ssm_suffix)`는 env 우선, 그다음 SSM(`/{project}/{stage}/{suffix}`,
SecureString 복호화) 순으로 시크릿을 해석하는 공유 헬퍼입니다. OpenAI 키(`make_visual`)가 이를 사용합니다.

**프롬프트 캐싱.** Bedrock 프롬프트 캐싱은 Claude 기준 캐시 가능 프리픽스 최소치가 ~1024 토큰입니다.
효과가 있는 곳에만 적용했습니다: 후속 **에이전트**는 ~1.7K 토큰 시스템 프롬프트 + 도구 스키마가 매 ReAct
스텝마다, 그리고 멀티턴 세션 내내 재전송되므로 Strands `BedrockModel(cache_config=
CacheConfig(strategy="auto"))`(`agent/agent.py`)로 해당 프리픽스를 캐싱합니다(검증: 첫 호출에
`cacheWriteInputTokens`, 이후 `cacheReadInputTokens` 발생). 단발성 파이프라인 프롬프트(랭커/다이제스트/트렌드/
시각화 시놉시스, 모두 ≤~530 토큰이며 실행당 1회 호출)는 캐시 최소치 미만이고 호출 간 재사용도 없어
의도적으로 캐싱을 적용하지 않았습니다.

## 7. 메모리(AgentCore 중심)

`shared/memory.py`는 `MemoryStore` ABC와 두 구현을 정의합니다:
- **`AgentCoreMemoryStore`**(AWS에서의 시스템 오브 레코드): 다이제스트 스냅샷을 단기 세션 이벤트로 기록
  (`create_event`, 세션 `digest-<date>`); `get_latest_digest()`가 세션을 나열해 가장 최근 다이제스트 세션의
  이벤트를 읽음. 트렌드 요약은 **시맨틱** 장기 전략에 공급되는 이벤트로 기록되며, `recall(query)`가
  네임스페이스 `/facts/{actor}/`에 대해 `retrieve_memory_records`를 수행하고 `recall_trends` 도구를 통해
  후속 에이전트에 노출됨(교차일 메모리).
- **`LocalMemoryStore`**: 오프라인 개발용 파일시스템 폴백(`digest_*.json`, `trends.jsonl`).

`create_memory_store()`는 `MEMORY_ID`가 설정되면 AgentCore를, 아니면 로컬을 선택합니다. 다이제스트
Lambda는 매 실행 후 스냅샷 + 트렌드를 기록하고, AgentCore 런타임은 매 호출 시 최신 스냅샷을
`DigestStateManager`에 로드하며 에이전트는 `recall_trends`로 이전 날짜의 맥락을 조회할 수 있습니다.

참고: 사람이 읽는 `trends.md` 문서(다이제스트 생성 시드로 쓰이는 교차일 내러티브)는 별도의 의도적
산출물로 여전히 `StateStore`(`TrendTracker`)를 통해 저장됩니다. AgentCore Memory는 에이전트가 기계적으로
recall하는 트렌드 사실을 보관합니다. 둘은 중복이 아니라 상호 보완적입니다.

메모리 리소스 자체(`AWS::BedrockAgentCore::Memory`)는 시맨틱 전략과 전용 `MemoryExecutionRole`(추출
모델용)과 함께 `foundation_stack`에서 생성됩니다. 비용: 장기 추출이 이벤트당 Bedrock 모델을 비동기로
호출하며, 단기 이벤트는 `event_expiry_duration`(90일) 후 만료됩니다. 하루 1회 다이제스트라 이벤트 볼륨은
매우 작습니다.

## 8. 헬스 체크 & 알림

`shared/models.py`: `SourceStatus`(`ok`/`empty`/`failed`), `SourceHealth(name, item_count, status, detail)`,
`HealthReport(sources)`(`has_failures`, `summary()` 보유). `run_collectors_with_health`가 각 소스를 분류:
예외 → FAILED(잘린 detail 포함), 0 항목 → EMPTY(조용한 날엔 정상), 그 외 → OK. 다이제스트 Lambda에서
`_maybe_alert`가 소스가 FAILED일 때만, 그리고 빈 항목 조기 반환 이전에 `ALERT_SNS_TOPIC_ARN`으로
게시(아무것도 수집 못 해도 장애는 알림되도록).

## 9. 에이전트(AgentCore Runtime 위의 Strands)

`agent/agent.py`는 `BedrockModel`(Sonnet 4.6)과 도구로 Strands `Agent`를 구성합니다. SYSTEM_PROMPT에
엄격한 라우팅 테이블(한국어), Slack mrkdwn 포맷 규칙, 응답 템플릿이 인코딩되어 있습니다.

도구(`agent/agent_tools.py`):
- `get_detail(item_number)` — `state_manager`에서 항목 본문 + 랭킹 메타데이터 로드.
- `search_papers(query)` — Semantic Scholar(429 시 retry/backoff).
- `search_community(query)` / `search_related_news(query)` — 공유 `_tavily_search(query, topic,
  include_domains)` 헬퍼를 감싼 얇은 래퍼.
- `recall_trends(query)` — `MemoryStore.recall`을 통한 교차일 시맨틱 recall(AgentCore 장기 메모리).
- `make_visual(item_number, mode, panels)` — §10 참조.

`agent_runtime/app.py`(`BedrockAgentCoreApp`): invoke 시 correlation id 설정, Memory에서 최신 다이제스트
상태 로드, `delivery_context`(미디어 도구용 채널/스레드) 설정, 에이전트 실행, Slack에 답변 게시. Slack
이벤트 Lambda(`slack_event_handler.py`)는 Slack 서명을 검증(HMAC, 타이밍 안전)하고 DynamoDB 조건부
쓰기로 중복 제거하며 비동기 self-invoke로 AgentCore 런타임을 호출합니다.

## 10. 시각화 파이프라인(시놉시스 → 이미지)

`agent/visuals.py`는 "시놉시스 → 시각화"를 `VisualMode`(브리프 프롬프트 + 이미지 프롬프트 빌더)와 `MODES`
레지스트리로 일반화합니다:
- **comic**(`ComicSynopsisPrompt`): 1~6컷 내러티브 만화(에이전트가 스토리에 맞게 컷 수 선택); 한국어
  캡션, 영어 비주얼 지시; 단일/나란히/2x2/2x3 레이아웃으로 렌더링.
- **diagram**(`VisualizationBriefPrompt`): 핵심 개념을 설명하는 인포그래픽 한 장(흐름/아키텍처/비교).

`VisualGenerator.generate(title, content, mode, panels)`: Claude(Bedrock)로 브리프 생성 →
`_parse_json_object` → 모드별 이미지 프롬프트 → **OpenAI `gpt-image-1`**(`b64_json`) → PNG 바이트.
`make_visual`이 `output.slack_handler.send_image_to_slack`(`files_upload_v2`)로 Slack에 이미지 업로드. OpenAI
키(`resolve_secret`로 env→SSM 해석)가 없으면 우아하게 비활성화. 새 모드는 또 다른 `VisualMode`를 등록하면
추가됩니다.

## 11. 인프라(CDK)

**`foundation_stack`**: VPC, ECR 리포, DynamoDB 중복 제거 테이블(SSE + prod에서 PITR), S3 상태 버킷
(CDK 생성 시 S3-managed 암호화, 버저닝, 퍼블릭 차단, SSL 강제), ECS Fargate RSSHub 서비스 +
service-discovery, CodeBuild 이미지 빌드, SNS 알림 토픽(+ 선택적 이메일 구독), AgentCore **Memory** 리소스
+ 실행 역할, IAM 역할들. IAM은 최소 권한: `/{project}/{stage}/*`로 스코프된 `ssm:GetParameter*`,
foundation-model/inference-profile ARN으로 스코프된 `bedrock:InvokeModel*`, 스코프된 `lambda:InvokeFunction`
및 `bedrock-agentcore:InvokeAgentRuntime`/Memory 데이터플레인 액션, 프로젝트 로그 그룹 ARN으로 스코프된
CloudWatch Logs — 계정 전역 관리형 정책 없음.

**`application_stack`**: 다이제스트 Lambda(DockerImage), Slack 이벤트 Lambda, API Gateway(+ 스테이지
스로틀링), 스테이지에 연결된 **WAFv2 WebACL**(rate-limit + AWS 관리형 규칙셋: Common, KnownBadInputs,
IpReputation), EventBridge 일일 크론(설정 기반 시/분), AgentCore Runtime(설정 가능한
`agentcore_image_ref`로 이미지 바인딩), 시크릿용 SSM 파라미터, SNS로 향하는 CloudWatch 알람(Lambda 에러
×2, API 5xx). 시크릿은 평문 `String` SSM 파라미터입니다(CloudFormation은 SecureString 생성 불가) —
보완 통제는 스코프된 IAM 읽기 정책이며, 더 민감한 자격증명은 Secrets Manager로 승격 권장.

## 12. 관측성(Observability)

`shared/logger.py`: AWS에서는 구조화 JSON 로그(`is_running_in_aws()`), 로컬에서는 사람이 읽는 형식;
`ContextVar` 기반 correlation id(`set_correlation_id`/`get_correlation_id`)가 모든 레코드에 주입되고 Lambda
요청 id / AgentCore 페이로드에서 시드됩니다. CloudWatch 알람은 SNS 알림 토픽으로 라우팅됩니다.

## 13. 테스트 & CI/CD

`tests/`(pytest, `asyncio_mode=auto`): 수집기(모킹한 HTTP/feedparser), Slack 이벤트 핸들러(서명 검증/중복 제거),
집계기, 랭커 파싱 + 슬롯/origin-cap
로직, 헬스 리포트, logger, 메모리 스토어(로컬 + AgentCore 모킹), 다이제스트 핸들러 알림, 에이전트 도구,
visuals, trend_tracker(trim/evidence-cap/archived-merge), 그리고 CDK assertion(`aws-cdk.assertions`로 두
스택 검증). 커버리지 게이트 55%.

`.github/workflows/ci.yml`: lint(ruff), 포맷 체크(black `--check`), mypy 타입 체크, 테스트 + 커버리지 게이트,
오프라인 `cdk synth`(`scripts/ci_synth.py`, 더미 계정 — AWS 자격증명 불필요), Docker 빌드(amd64,
`--provenance=false`).

## 14. 주요 명령어

```bash
uv run python main.py --dry-run --sources rss reddit   # 부분 dry run
uv run python main.py                                   # 전체 파이프라인 + Slack
uv run python -m pytest tests/ -v                       # 테스트
uv run black --check . && uv run ruff check .           # lint/format
uv run mypy shared/ collectors/ pipeline/ agent/ output/ lambda_handlers/ main.py
uv run python scripts/ci_synth.py                       # 오프라인 CDK synth
AWS_PROFILE=research uv run cdk deploy --all -a "uv run python scripts/deploy.py"
```
