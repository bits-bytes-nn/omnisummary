# 운영 가이드

혼자 OmniSummary를 운영할 때 매일 확인할 것, 알림을 받았을 때 할 일, 문제가 생긴 날을 되살리는 방법을 모았다. 설계 이유는
[design.md](./design.md), 설치와 배포는 [README](../README.ko.md)를 본다.

아래 명령은 `research` 프로필 계정과 ap-northeast-2 기준이다. 다른 계정이면 `--profile`과 `--region`을 바꾼다. 셸이 다른 계정의
`AWS_PROFILE`·`AWS_REGION`을 물려받을 수 있으니 AWS CLI에는 항상 둘을 직접 적는다(zsh는 `P="--profile ..."`처럼 변수에 담은
옵션을 단어로 나누지 않는다).

## 하루의 흐름

| 시각(KST) | 일 | 어디서 |
|-----------|----|--------|
| 18:30 | X와 YouTube 자막을 모아 S3에 올린다 | 로컬 cron(`scripts/sync_all_to_s3.sh`) |
| 19:00 | 수집, 랭킹, 다이제스트 작성, 스냅샷 저장 | digest Lambda |
| 19:0x | 만화를 그리고 Threads에 첫 글과 답글 5개를 올린다 | visual Lambda |
| 07:00 | 투자판의 X와 YouTube 자막을 모아 S3(`omnisummary/invest/`)에 올린다 | 로컬 cron(`OMNISUMMARY_EDITION=invest scripts/sync_all_to_s3.sh`) |
| 07:30 | 투자판: 수집, 랭킹, 다이제스트 작성, 스냅샷 저장, 만화와 게시 | invest-digest, invest-visual Lambda |
| 매시간 | 최근 일주일 게시물의 새 댓글에 답한다(켜 두었을 때) | comment-replies Lambda |
| 매주 | Threads 토큰을 갱신해 SSM에 쓴다 | threads-refresh Lambda |

정상인 날에는 알림이 오지 않는다. 확인하고 싶으면 그날 게시물을 읽는다. 로컬에서 `main.py`를 돌렸다면 종료 코드를
본다. 수집 0건, 다이제스트 없음, 게시 실패나 미완료면 1이다(로컬에는 알림이 없다).

```bash
uv run python recall_cli.py digest            # 오늘 저장된 리드와 기사(JSON). 게시 여부는 Threads에서 본다
uv run python recall_cli.py digest 2026-10-28 # 특정 날짜
OMNISUMMARY_EDITION=invest uv run python recall_cli.py digest   # 투자판
```

투자판 게시는 `config/invest.yaml`의 `pipeline.enable_threads_post`로 켠다. 처음 며칠은 꺼 두고 저장된 다이제스트를 위
명령으로 읽은 뒤 켠다. 켜고 나서는 이미지를 다시 빌드하고 배포해야 반영된다. 투자판 알림 제목에는 `dev/invest`가 붙고, 재게시
명령은 `omnisummary-dev-invest-visual`을 가리킨다.

켠 뒤 확인한다.

1. 배포한 이미지가 켠 설정으로 빌드됐는지 본다. `grep -n enable_threads_post config/invest.yaml`이 `true`이고, 그 체크아웃에서 빌드한 이미지 digest로 배포했어야 한다.
2. 다음 07:30 실행 뒤 비주얼 Lambda 로그에 게시 줄이 있는지 본다. `Threads posting is disabled; nothing to publish for <날짜>`가 보이면 꺼진 설정이 배포된 것이다.

```bash
aws logs filter-log-events --profile research --region ap-northeast-2 --log-group-name /aws/lambda/omnisummary-dev-invest-visual \
  --filter-pattern '"Daily visual posted for <날짜>"' --query 'events[].message' --output text
```

3. Threads 앱에서 그날 투자판 첫 글(만화, 리드, 나머지 기사 제목, 고지 문구)과 답글이 모두 올라왔는지 본다.

## 알림을 받았을 때

알림 메일 제목에는 프로젝트, 스테이지, 이벤트, 상태가 들어가고, 본문 끝의 correlation id로 그 실행의 로그를 찾을 수 있다.

```bash
aws logs filter-log-events --profile research --region ap-northeast-2 --log-group-name /aws/lambda/omnisummary-dev-visual \
  --filter-pattern '"<correlation id>"' --query 'events[].message' --output text
```

### Source Health

| 상태 | 뜻 | 할 일 |
|------|----|-------|
| `FAILED` | 소스가 아예 실패했다 | 보고서의 오류를 본다. 대개 일시적이라 다음 날 복구된다 |
| `STALE` | X나 YouTube 적재 파일이 12시간보다 오래됐다(그날 18:30 동기화, 투자판은 07:00 동기화가 실패했거나 돌지 않았다). 그 동기화의 일부 실패도 상세에 함께 적힌다 | Mac이 그 시각에 깨어 있었는지(잠든 사이의 cron은 실행되지 않는다), Docker Desktop과 `rsshub` 컨테이너가 떠 있었는지, cron 로그(`logs/sync.log`)를 본다. 하루 건너뛴 것은 다음 정상 동기화가 다시 읽어 와 복구된다(이틀 연속이면 앞날은 잃는다) |
| `DEGRADED` | 입력의 일부만 응답했다 | 보고서의 상세를 본다. X라면 쿠키 만료가 흔한 원인이다. README의 'RSSHub로 X 읽기' 절에 따라 쿠키를 새로 받아 컨테이너를 다시 만들고, 아래 'X 쿠키를 갱신한 뒤'로 확인한다 |
| `EMPTY` | 꼭 있어야 하는 소스가 0건이다 | 피드 주소나 API 키를 확인한다 |

### X 쿠키를 갱신한 뒤

1. 설정의 계정 하나로 RSSHub 경로가 항목을 돌려주는지 본다. 빈 채널(`<item>`이 0개)이면 쿠키가 아직 맞지 않는다.

```bash
curl -s http://localhost:1200/twitter/user/<collectors.rsshub.accounts의 계정> | grep -c '<item>'   # 0보다 커야 한다
```

2. 동기화를 한 번 돌리고 적재 파일이 새로 올라갔는지 본다. 키는 `aws.s3_prefix` 아래의 `omnisummary/rsshub_items.json`(투자판은 `omnisummary/invest/rsshub_items.json`)이고 버킷은 `aws.state_bucket_name`(비어 있으면 `omnisummary-dev-state`)이다.

```bash
./scripts/sync_all_to_s3.sh
aws s3api head-object --profile research --region ap-northeast-2 --bucket <상태 버킷> --key omnisummary/rsshub_items.json \
  --query LastModified --output text   # 방금 실행한 시각이어야 한다
tail -1 digest_state/last_park_sync_rsshub.txt   # 정상 동기화로 기록된 시작·업로드 시각(투자판은 digest_state/invest/)
```

`last_park_sync_rsshub.txt`에 새 줄이 없으면 그 동기화는 실패율이나 빈 응답률 임계를 넘은 비정상 동기화다. 쿠키와 계정 목록을 다시 본다.

### Ranking Health

랭킹 배치 하나가 재시도까지 실패해 후보 일부가 빠진 채 다이제스트가 나갔다는 뜻이다. 게시물 자체는 정상이다. 같은
경고가 며칠 이어지면 Bedrock 스로틀이나 권한을 본다.

### Threads Delivery

`Detail`에 할 일이 적혀 있고, 아무것도 게시되지 않은 날에는 날짜를 채운 실행 명령도 함께 온다. `Root post`는 첫 글 id, `Standalone posts`는 답글인데 별도 게시물로 올라간 글의 id다.

| Detail | 뜻 | 할 일 |
|--------|----|-------|
| `published TEXT-ONLY` | 만화 없이 글만 올라갔다 | 로그에서 렌더 실패 이유를 본다(OpenAI 키, 모더레이션, 시간 부족). 다시 올리지 않는다 |
| `reply chain incomplete` | 첫 글은 있고 답글 일부가 빠졌다 | Threads 앱에서 그날 첫 글과 그 답글을 지운 뒤 알림의 `force_republish: true` 명령으로 저장된 다이제스트를 다시 올린다. 지우지 않고 다시 올리면 그날 글이 둘이 된다 |
| `Standalone posts` 항목이 있다 | 답글이 첫 글 아래가 아니라 별도 게시물로 올라갔다 | 첫 글을 지워도 이 게시물은 남으니 함께 지운 뒤 다시 올린다 |
| `the root is live but no story reply landed under it` | 첫 글만 올라갔다(답글이 실패했거나 별도 게시물로 올라갔다) | 첫 글(과 별도 게시물)을 지운 뒤 알림의 `force_republish: true` 명령을 실행한다. 지우지 않고 다시 올리면 첫 글이 둘이 된다 |
| `the root MAY be live` | 첫 글 게시 요청이 오류로 끝났지만 실제로 올라갔는지 알 수 없다 | Threads 앱에서 그날 첫 글이 있는지 본다. **없을 때만** 알림에 적힌 명령(`force_republish: true`)을 실행한다 |
| `the stored digest has NO stories` | 저장된 다이제스트에 기사가 없어 아무것도 올리지 않았다 | 다시 게시해도 같으니, 알림에 적힌 `main.py --date … --force-republish`로 새로 만든다 |
| `could NOT be confirmed as recorded` | 게시는 됐는데 게시 기록(원장)에 그날이 남았는지 확인하지 못했다 | 올라간 글이 남아 있는 동안은 그날을 다시 올리지도, 다시 돌리지도 않는다. 기록이 없으면 막을 장치가 없어 같은 글이 한 번 더 올라간다. 같은 실행의 다른 알림이 지우고 다시 올리라고 하면, 지운 뒤에는 다시 올려도 된다. 다음 날부터는 영향이 없다 |
| `NOT published` | 아무것도 올라가지 않았다 | 알림에 적힌 명령을 실행한다. 게시 기록이 남아 있는 날이면 `force_republish: true`가 든 명령이 오니, 앱에서 첫 글이 없음을 확인한 뒤 실행한다 |

### 비용 알림

`aws.monthly_budget_usd`가 0보다 크면 스택이 `Project` 태그 기준 월 예산(실제 80%, 예측 100%)과 비용 이상 탐지(하루 예산만큼의 이상)를 만들고 알림 주소로 메일을 보낸다. 로컬 실행(리서치, 평가, 판정)은 `<stage>-local` 추론 프로파일로 청구돼 Cost Explorer에서 `Stage` 태그로 운영과 나뉜다. 모델을 바꾸거나 프로파일을 새로 만들 때는 `scripts/put_inference_profiles.py`를 다시 실행한다. 확인: Cost Explorer에서 태그 `Stage`로 묶어 `dev`와 `dev-local`이 따로 보이는지 본다.

### CloudWatch 알람

| 알람 | 뜻 | 할 일 |
|------|----|-------|
| `DigestErrorsAlarm`, `VisualErrorsAlarm`, `InvestDigestErrorsAlarm`, `InvestVisualErrorsAlarm`, `CommentReplyErrorsAlarm`, `ThreadsRefreshErrorsAlarm` | 그 Lambda가 예외로 끝났다 | 로그 그룹 `/aws/lambda/omnisummary-dev-digest`, `-visual`, `-invest-digest`, `-invest-visual`, `-comment-replies`, `-threads-refresh`를 본다 |
| 같은 이름의 `…TimeoutAlarm` | 실행 시간이 타임아웃의 90%에 닿았다 | 어느 단계가 느렸는지 로그를 본다 |
| `EmptyDigestAlarm` / `InvestEmptyDigestAlarm` | 하루 동안 게시할 기사가 0건이었거나 다이제스트가 돌지 않았다 | 해당 판의 digest Lambda 로그(`omnisummary-dev-digest` / `omnisummary-dev-invest-digest`)를 본다 |
| `AsyncDLQAlarm` | 실패한 digest·visual 실행이 DLQ에 쌓였다 | 아래 "그날을 다시 올리기"를 따른다 |

댓글 답글이 실패하면 `CommentReplyErrorsAlarm`이 울린다. 일시적일 수 있는 게시 실패와 검증·파싱 실패는 세 번까지만 다시
시도하고, 게시 요청이 오류로 끝나 실제로 올라갔는지 알 수 없는 답글(게시 단계 5xx나 타임아웃)은 다시 시도하지 않는다.
스로틀이나 Bedrock 장애처럼 모든 댓글이 함께 실패하는 경우는 횟수를 세지 않으므로, 장애가 이어지는 동안은 알람도 매시간 울린다.

## 그날을 다시 올리기

먼저 Threads 앱에서 그날 첫 글이 정말 없는지 확인한다. 일부만 올라간 첫 글이 있으면 먼저 지운다. 지우지 않고 다시 올리면 첫 글이 둘이 된다.

**저장된 스냅샷으로 다시 게시한다(권장).** 다이제스트는 이미 만들어져 저장돼 있으니 visual Lambda만 다시 부른다. 같은
글과 같은 기사가 나가고, 만화는 새로 그린다. Threads Delivery 알림에 날짜가 채워진 이 명령이 오니 `--profile research`를 붙여 실행한다.

```bash
aws lambda invoke --profile research --region ap-northeast-2 --function-name omnisummary-dev-visual --invocation-type Event \
  --cli-binary-format raw-in-base64-out --payload '{"digest_date": "2026-10-28"}' /dev/null
```

게시 기록에 그날이 남아 있으면(`the root MAY be live`였던 날) 이 호출은 건너뛴다. 앱에서 첫 글이 없음을 확인한 뒤에만
`"force_republish": true`를 더해 다시 부른다.

```bash
aws lambda invoke --profile research --region ap-northeast-2 --function-name omnisummary-dev-visual --invocation-type Event \
  --cli-binary-format raw-in-base64-out --payload '{"digest_date": "2026-10-28", "force_republish": true}' /dev/null
```

다시 올린 뒤 확인한다. 둘 다 맞아야 끝난 것이다.

1. 비주얼 Lambda 로그에 그날의 게시 줄이 있다. `Daily visual skipped for <날짜>`가 보이면 게시 기록 때문에 건너뛴 것이다.

```bash
aws logs filter-log-events --profile research --region ap-northeast-2 --log-group-name /aws/lambda/omnisummary-dev-visual \
  --filter-pattern '"Daily visual posted for 2026-10-28"' --query 'events[].message' --output text
```

2. Threads 앱에서 그날 첫 글 하나와 그 아래 기사 답글이 모두(보통 5개) 달려 있고, 별도 게시물로 올라간 답글이 없다. Threads Delivery 알림이 다시 오지 않았는지도 본다.

`main.py --date <날짜> --force-republish`는 다시 수집하고 새로 쓰므로 그날 게시된 것과 다른 다이제스트가 나가고, 저장된
기록도 그것으로 바뀐다. 저장된 스냅샷이 없거나 기사가 0건일 때만 쓴다.

DLQ에 쌓인 메시지는 다시 올리고 위 확인을 마친 뒤에만 지운다. digest Lambda 메시지를 다시 넣으면 그 메시지가 실행됐던 날짜의 다이제스트를 다시 만든다(이벤트의 `time`을 읽는다). 그날 비주얼 Lambda가 아직 게시 중이면 새로 만든 스냅샷이 그날의 기록을 덮어써서, 게시된 글과 회상·재게시 기준이 달라진다. 저장된 다이제스트가 있고 게시만 실패했다면 digest 메시지를 다시 넣지 말고 비주얼 Lambda에 `{"digest_date": "YYYY-MM-DD"}`를 넘겨 저장된 것을 게시한다.

DLQ는 두 판이 함께 쓴다. 큐를 통째로 비우지 말고, 메시지마다 `functionArn`과 `requestPayload`로 어느 Lambda의 어느 날짜인지
확인한 뒤 다시 올리고 확인까지 마친 메시지만 수신 핸들로 지운다. 수신한 메시지는 `--visibility-timeout` 동안만 핸들이 유효하다.

```bash
Q=$(aws sqs get-queue-url --profile research --region ap-northeast-2 --queue-name omnisummary-dev-async-dlq --query QueueUrl --output text)
aws sqs receive-message --profile research --region ap-northeast-2 --queue-url "$Q" --max-number-of-messages 10 \
  --visibility-timeout 900 --output json \
  | jq -r '.Messages[] | (.Body | fromjson) as $b | [.ReceiptHandle, $b.requestContext.functionArn, ($b.requestPayload | tostring)] | @tsv'
aws sqs delete-message --profile research --region ap-northeast-2 --queue-url "$Q" --receipt-handle '<위에서 확인한 핸들>'
```

## 배포

`scripts/release.sh`는 origin/main에 올라가 CI를 통과한 HEAD만 배포한다. 이미지를 빌드한 뒤 올리기 전에 CI의 docker 작업과 같은
가져오기·설정 검사를 그 이미지로 다시 돌린다(`config/invest.yaml`이 있으면 투자판도). CI가 보는 이미지에는 운영자 설정이 없기
때문이다. CDK 배포는 승인을 묻지 않는다(`--require-approval never`, 운영자가 TTY 없는 셸에서 배포한다). 배포가 끝나면 스테이지의 이미지 Lambda가 모두 방금 올린
digest로 돌고(`ResolvedImageUri`), `State`가 `Active`, `LastUpdateStatus`가 `Successful`인지 읽기만 해서 확인하고, 하나라도 다르면
실패로 끝난다.

## 자주 하는 점검

| 무엇을 | 명령 |
|--------|------|
| 비밀값이 모두 들어 있는지 | `uv run python scripts/put_secrets.py --verify` |
| X 수집기가 살아 있는지 | `curl -s http://localhost:1200/twitter/user/<설정의 X 계정> \| head` |
| 최근 정상 동기화 시각(시작, 업로드) | `cat digest_state/last_park_sync_rsshub.txt digest_state/last_park_sync_youtube.txt`(투자판은 `digest_state/invest/`) |
| 동기화 로그 | `tail -50 logs/sync.log`(cron이 그 경로에 쓸 때) |
| 추적 중인 트렌드 | `uv run python recall_cli.py trends` |
| 지금 답할 댓글 초안(게시하지 않는다) | `uv run python scripts/sample_comment_replies.py` |
| 저장된 날의 만화 계획(그리지 않는다) | `uv run python scripts/sample_visual_brief.py 2026-10-28 /tmp/vis` |
| 최근 비용(일별, Lambda별 또는 단계별) | `uv run python scripts/cost_report.py --days 7 [--by stage]` |

## 하지 말 것

- 첫 글이 이미 올라간 날에 `--force-republish`를 쓰지 않는다.
- 배포된 계정에서 `FOUNDATION_ONLY=1`을 쓰지 않는다.
- `aws.state_bucket_name`은 스택이 만들고 보존하는 상태 버킷이다(버전 관리, 이전 버전 30일 보관). 이름을 바꾸면 새 버킷이 생기고 기존 버킷은 남으니, 바꿀 때는 `config.yaml`과 `invest.yaml`을 함께 바꾸고(deploy.py가 불일치를 거부한다) 배포한 뒤 `aws s3 sync`로 `omnisummary/` 접두사를 옮기고 객체 수를 확인한다. 로컬 동기화 사용자(`aws.s3_sync_user`)의 쓰기 권한은 스택이 새 버킷으로 옮긴다.
- SSM 비밀값 파라미터의 템플릿 속성(태그 포함)을 바꾸지 않는다. CloudFormation이 실제 비밀값을 자리표시자로 덮는다.
- `put_secrets.py --force`로 Threads 토큰을 덮어쓰지 않는다. Lambda가 갱신한 토큰이 오래된 로컬 값으로 돌아간다.
