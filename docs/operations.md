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
| 매시간 | 최근 일주일 게시물의 새 댓글에 답한다(켜 두었을 때) | comment-replies Lambda |
| 매주 | Threads 토큰을 갱신해 SSM에 쓴다 | threads-refresh Lambda |

정상인 날에는 알림이 오지 않는다. 확인하고 싶으면 그날 게시물을 읽는다. 로컬에서 `main.py`를 돌렸다면 종료 코드를
본다. 수집 0건, 다이제스트 없음, 게시 실패나 미완료면 1이다(로컬에는 알림이 없다).

```bash
uv run python recall_cli.py digest            # 오늘 저장된 리드와 기사(JSON). 게시 여부는 Threads에서 본다
uv run python recall_cli.py digest 2026-10-28 # 특정 날짜
```

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
| `STALE` | X나 YouTube 적재 파일이 12시간보다 오래됐다(그날 18:30 동기화가 실패했거나 돌지 않았다) | Mac이 18:30에 깨어 있었는지, Docker Desktop과 `rsshub` 컨테이너가 떠 있었는지, cron 로그를 본다(README의 cron 줄은 `logs/sync.log`에 쓴다. 예전 줄은 재부팅하면 지워지는 `/tmp/omnisummary-sync.log`에 쓴다). 하루 건너뛴 것은 다음 정상 동기화가 다시 읽어 와 복구된다(이틀 연속이면 앞날은 잃는다) |
| `DEGRADED` | 입력의 일부만 응답했다 | X라면 쿠키 만료가 흔한 원인이다. README의 'RSSHub로 X 읽기' 절에 따라 쿠키를 새로 받아 컨테이너를 다시 만든다 |
| `EMPTY` | 꼭 있어야 하는 소스가 0건이다 | 피드 주소나 API 키를 확인한다 |

### Ranking Health

랭킹 배치 하나가 재시도까지 실패해 후보 일부가 빠진 채 다이제스트가 나갔다는 뜻이다. 게시물 자체는 정상이다. 같은
경고가 며칠 이어지면 Bedrock 스로틀이나 권한을 본다.

### Threads Delivery

`Detail`에 할 일이 적혀 있고, 아무것도 게시되지 않은 날에는 날짜를 채운 실행 명령도 함께 온다. `Root post`는 첫 글 id, `Standalone posts`는 답글인데 별도 게시물로 올라간 글의 id다.

| Detail | 뜻 | 할 일 |
|--------|----|-------|
| `published TEXT-ONLY` | 만화 없이 글만 올라갔다 | 로그에서 렌더 실패 이유를 본다(OpenAI 키, 모더레이션, 시간 부족). 다시 올리지 않는다 |
| `reply chain incomplete` | 첫 글은 있고 답글 일부가 빠졌다 | 그날을 다시 올리지 않는다(첫 글이 하나 더 생긴다). 빠진 기사는 Threads 앱에서 첫 글에 직접 답글로 단다 |
| `Standalone posts` 항목이 있다 | 답글이 첫 글 아래가 아니라 별도 게시물로 올라갔다(2026-10-27에 있었다) | 그 id의 게시물을 Threads 앱에서 지우고, 같은 기사를 첫 글에 직접 답글로 단다 |
| `the root is live but no story reply landed under it` | 첫 글만 올라갔다(답글이 실패했거나 별도 게시물로 올라갔다) | `--force-republish`를 하지 않는다(첫 글이 하나 더 생긴다) |
| `the root MAY be live` | 첫 글 게시 요청이 오류로 끝났지만 실제로 올라갔는지 알 수 없다 | Threads 앱에서 그날 첫 글이 있는지 본다. **없을 때만** 알림에 적힌 명령(`force_republish: true`)을 실행한다 |
| `the stored digest has NO stories` | 저장된 다이제스트에 기사가 없어 아무것도 올리지 않았다 | 다시 게시해도 같으니, 알림에 적힌 `main.py --date … --force-republish`로 새로 만든다 |
| `could NOT be confirmed as recorded` | 게시는 됐는데 게시 기록(원장)에 그날이 남았는지 확인하지 못했다 | 그날은 다시 올리지도, 다시 돌리지도 않는다. 기록이 없으면 막을 장치가 없어 같은 글이 한 번 더 올라간다. 다음 날부터는 영향이 없다 |
| `NOT published` | 아무것도 올라가지 않았다 | 알림에 적힌 명령을 실행한다. 게시 기록이 남아 있는 날이면 `force_republish: true`가 든 명령이 오니, 앱에서 첫 글이 없음을 확인한 뒤 실행한다 |

### CloudWatch 알람

| 알람 | 뜻 | 할 일 |
|------|----|-------|
| `DigestErrorsAlarm`, `VisualErrorsAlarm`, `CommentReplyErrorsAlarm`, `ThreadsRefreshErrorsAlarm` | 그 Lambda가 예외로 끝났다 | 로그 그룹 `/aws/lambda/omnisummary-dev-digest`, `-visual`, `-comment-replies`, `-threads-refresh`를 본다 |
| 같은 이름의 `…TimeoutAlarm` | 실행 시간이 타임아웃의 90%에 닿았다 | 어느 단계가 느렸는지 로그를 본다 |
| `EmptyDigestAlarm` | 하루 동안 게시할 기사가 0건이었거나 다이제스트가 돌지 않았다 | digest Lambda 로그를 본다 |
| `AsyncDLQAlarm` | 실패한 digest·visual 실행이 DLQ에 쌓였다 | 아래 "그날을 다시 올리기"를 따른다 |

댓글 답글이 실패하면 `CommentReplyErrorsAlarm`이 울린다. 일시적일 수 있는 게시 실패와 검증·파싱 실패는 세 번까지만 다시
시도하고, 게시 요청이 오류로 끝나 실제로 올라갔는지 알 수 없는 답글(게시 단계 5xx나 타임아웃)은 다시 시도하지 않는다.
스로틀이나 Bedrock 장애처럼 모든 댓글이 함께 실패하는 경우는 횟수를 세지 않으므로, 장애가 이어지는 동안은 알람도 매시간 울린다.

## 그날을 다시 올리기

먼저 Threads 앱에서 그날 첫 글이 정말 없는지 확인한다. 하나라도 올라가 있으면 다시 올리지 않는다(첫 글이 둘이 된다).

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

`main.py --date <날짜> --force-republish`는 다시 수집하고 새로 쓰므로 그날 게시된 것과 다른 다이제스트가 나가고, 저장된
기록도 그것으로 바뀐다. 저장된 스냅샷이 없거나 기사가 0건일 때만 쓴다.

DLQ에 쌓인 메시지는 확인한 뒤 지운다. digest Lambda 메시지를 다시 넣으면 그 메시지가 실행됐던 날짜의 다이제스트를 다시 만든다(이벤트의 `time`을 읽는다).

```bash
Q=$(aws sqs get-queue-url --profile research --region ap-northeast-2 --queue-name omnisummary-dev-async-dlq --query QueueUrl --output text)
aws sqs receive-message --profile research --region ap-northeast-2 --queue-url "$Q" --max-number-of-messages 10
aws sqs purge-queue --profile research --region ap-northeast-2 --queue-url "$Q"   # 다시 올린 뒤에만
```

## 자주 하는 점검

| 무엇을 | 명령 |
|--------|------|
| 비밀값이 모두 들어 있는지 | `uv run python scripts/put_secrets.py --verify` |
| X 수집기가 살아 있는지 | `curl -s http://localhost:1200/twitter/user/karpathy \| head` |
| 최근 정상 동기화 시각(시작, 업로드) | `cat digest_state/last_park_sync_rsshub.txt digest_state/last_park_sync_youtube.txt` |
| 동기화 로그 | `tail -50 logs/sync.log`(cron이 그 경로에 쓸 때) |
| 추적 중인 트렌드 | `uv run python recall_cli.py trends` |
| 지금 답할 댓글 초안(게시하지 않는다) | `uv run python scripts/sample_comment_replies.py` |
| 저장된 날의 만화 계획(그리지 않는다) | `uv run python scripts/sample_visual_brief.py 2026-10-28 /tmp/vis` |

## 하지 말 것

- 첫 글이 이미 올라간 날에 `--force-republish`를 쓰지 않는다.
- 배포된 계정에서 `FOUNDATION_ONLY=1`을 쓰지 않는다.
- 첫 배포 뒤에 `aws.state_bucket_name`을 바꾸지 않는다. prod가 아닌 스테이지에서는 스택이 만든 버킷이 지워진다.
- SSM 비밀값 파라미터의 템플릿 속성(태그 포함)을 바꾸지 않는다. CloudFormation이 실제 비밀값을 자리표시자로 덮는다.
- `put_secrets.py --force`로 Threads 토큰을 덮어쓰지 않는다. Lambda가 갱신한 토큰이 오래된 로컬 값으로 돌아간다.
