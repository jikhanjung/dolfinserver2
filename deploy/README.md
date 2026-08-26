# deploy — 검토 화면을 컨테이너로 띄우기

**이 저장소가 컨테이너로 내는 것은 검토·수집 화면 하나다.** 학습도 추론도
아니다 — 그것은 2080ti 에서 management command 로 돌고, 산출물(`fin.db` ·
`reid/<판>` · `crops/`)만 이쪽으로 건너온다. 그래서 이미지에 torch 가 없고
157MB 로 끝난다.

```
     2080ti (작업 자리)                    m710q (서비스 자리)
  segment · train · reid_cls        →     /srv/dolfinserver2/{db,reid,crops}
  db/fin.db · reid/v3 · crops/            ↑ sync_data.sh          ↓ 마운트
                                          nginx :8085 → 컨테이너 127.0.0.1:8104
```

## 세 자리

| | 무엇 | 역할 | DB |
|---|---|---|---|
| `/srv/dolfinserver2` · nginx **8085** | **운영** (m710q) | `work` | **진짜 `fin.db`** |
| GCP `dolfinid2` | **운영** | `reid` | 거기 `fin.db` (GCP 가 주인) |
| `/srv/dolfinserver2-test` · nginx **8086** | 시험 | 골라 띄운다 | NAS 백업 사본 |
| `manage.py runserver` | 개발 | 골라 띄운다 | 사본 (`FIN_DB`) |

**운영 자리는 사본을 안 든다.** `/srv/dolfinserver2/db/fin.db` 가 곧 파이프라인
명령이 여는 파일이다(`settings` 의 `FIN_DB` 기본값). 사본이면 검토는 컨테이너
쪽에 쌓이고 파이프라인은 그것을 영영 못 본다 — 같은 파일을 열면 맞출 일이 없다.

**크롭·조각은 저장소 것을 읽기전용(`:ro`)으로 건다.** 파생물이고 파이프라인이
계속 다시 쓰는 것이라 사본을 두면 갈아 끼울 때마다 낡는다. `.gitignore` 가
`crops/`·`reid/` 를 무시해서 `git` 이 그 디렉토리를 안 건드린다.

**시험 DB 는 NAS 백업에서 뜬다** (`host/test_db.sh`). 사본이라 무엇을 해도
사람의 판정이 안 다치고, **덤으로 백업이 성한지를 잰다** — 백업에서 복원해
화면이 도는 것을 본 적이 없으면 그것은 백업이 아니라 파일일 뿐이다.

## 파일

| | |
|---|---|
| `Dockerfile` · `requirements-web.txt` | 코드만. 자료는 전부 호스트 마운트 |
| `entrypoint.sh` | collectstatic · migrate · gunicorn. **자료를 나르지 않는다** |
| `docker-compose.yml` · `nginx.conf` | 운영(8085 → 8104)의 저장소본 |
| `docker-compose.test.yml` · `nginx.test.conf` | 시험(8086 → 8105)의 저장소본 |
| `env.example` | `.env` 의 본 |
| `build.sh` | 테스트 → 버전 → build (→ push) |
| `host/bootstrap.sh` | `/srv` 자리 + nginx. **root 필요**, `--test` 로 시험 자리 |
| `host/test_db.sh` | 시험 DB 를 NAS 백업에서 뜬다. root 불요 |

## 처음 한 번

```bash
sudo deploy/host/bootstrap.sh            # 운영 자리 · nginx 8085
sudo deploy/host/bootstrap.sh --test     # 시험 자리 · nginx 8086
deploy/host/test_db.sh                   # 시험 DB (NAS 백업 → 사본)
deploy/build.sh 0.2.0
cd /srv/dolfinserver2      && docker compose up -d
cd /srv/dolfinserver2-test && docker compose up -d
```

## 그 다음부터

```bash
deploy/build.sh 0.4.2 --push                 # 테스트 → 버전 → build → Docker Hub
sed -i 's/^IMAGE_TAG=.*/IMAGE_TAG=0.4.2/' /srv/dolfinserver2/.env
cd /srv/dolfinserver2 && docker compose up -d
ssh dolfinid /srv/dolfinserver2/deploy.sh 0.4.2   # GCP 는 당겨서 뜬다
```

이미지는 `honestjung/dolfinserver2`(공개)다. `deploy/gcp/ship.sh` 는
레지스트리가 막혔을 때의 뒷문으로 남겨 뒀다.

시험 자리에서 두 역할을 다 돌려 본 뒤에 운영으로 올린다:

```bash
sed -i 's/^FIN_ROLE=.*/FIN_ROLE=reid/' /srv/dolfinserver2-test/.env   # GCP 갈래
cd /srv/dolfinserver2-test && docker compose up -d
curl -s http://m710q:8086/healthz | python3 -m json.tool
```

## 역할 — `FIN_ROLE`

**개체를 만들고 지느러미를 개체에 넣는 일은 한 자리에서만 한다.**

| | 걸리는 길 |
|---|---|
| `work` (기본) | `/` `/review` `/edit` `/compare` `/detect` `/photo` `/healthz` |
| `reid` | `/reid` `/catalog` `/api/reid/*` `/reid/chip/…` `/healthz` |

**막는 자리가 앱이다** (`review/urls.py` 의 `patterns_for`). nginx 로만 막으면
정작 새는 자리가 안 막힌다 — `runserver` 앞에는 nginx 가 없는데, 구멍이 바로
거기다: **`/reid` 는 열기만 해도 보류함 `Individual` 을 하나 만든다**
(`review/views.py` 의 `get_or_create`). 경로를 URLconf 에서 빼면 그 길이 없다.

`/healthz` 가 역할을 낸다 — **배포가 뒤바뀐 것을 여기서 잡는다.** 개체 분류를
받을 자리가 `work` 로 떠 있으면 화면은 멀쩡히 200 을 내면서 그날 판정을 한 건도
못 받는다. 차림표에도 `작업 자리`·`re-ID 자리` 로 늘 보인다.

```bash
curl -s http://m710q:8085/healthz | python3 -m json.tool
```

## 문 — 접속 코드 (`FIN_ACCESS_CODE`)

**이 앱에는 사람마다의 인증이 없다.** `login_required` 도 `LoginRequiredMiddleware`
도 없어서 `/api/reid/assign`·`/api/review` 같은 **쓰기 경로가 그대로 열려 있고**,
그것이 쓰는 것은 다시 만들 수 없는 사람의 판정이다. 코드 하나로 문을 막는다.

처음 들어오면 `/enter` 로 가고, 맞히면 세션에 표가 남는다(30일). 막는 자리는
**미들웨어**다(`review/gate.py`) — 뷰마다 데코레이터를 붙이면 새로 만든 뷰
하나가 빠지는 날이 온다. **기본이 막힘이고 예외만 적는다.**

| 자리 | |
|---|---|
| m710q 운영 · GCP | **코드 있음** |
| m710q 시험 · `runserver` | 없음 (DB 가 사본이다) |

문 밖에 두는 것은 `/enter`·`/healthz`·`/static` 뿐이다. `/healthz` 가 밖에
있는 것은 smoke 와 배포가 읽어야 해서인데, **밖에서는 수를 안 낸다** —
상자·개체가 몇인지까지 알려 주면 문을 세운 뜻이 절반 없어진다. 안에서 부르면
그대로 다 낸다.

**인증의 대신이지 인증이 아니다.** 셋을 알고 쓴다:

- **누가 했는지가 안 남는다** — `Review.reviewer` 가 계속 `NULL` 이다
- **한 사람만 뺄 수 없다** — 코드를 바꾸면 다 같이 나간다
- **TLS 없이 공개 주소로 열면 코드도 세션 쿠키도 평문으로 간다.** 지금 GCP 는
  tailnet 에만 열려 있고 그 구간은 암호화된다. 공개로 옮기려면 https 와
  `FIN_COOKIE_SECURE=1` 이 먼저다

긁어 보는 것은 10번에서 15분 잠긴다(주소 단위). 코드 비교는 `compare_digest` 라
맞은 글자 수가 시간으로 새지 않고, 문을 지날 때 **세션 키를 새로 뽑는다**
(남이 미리 심어 둔 키로 안까지 들어오는 것을 막는다).

시험은 `review/tests.py` 의 `AccessCodeTests` 8종 + `HealthzOutsideTheDoorTests` 2종.

## 백업 — 시간마다 (`.guides/web/data-safety.md`)

`fin.db` 는 사람의 판정이라 **다시 만들 수 없다.** 가이드가 말하는
"operator-entered data" 가 정확히 그것이다.

```
10 * * * * python3 /srv/dolfinserver2/scripts/backup_db.py >> …/logs/backup.log 2>&1
```

| | |
|---|---|
| 어디에 | `/srv/dolfinserver2/backup/hourly/fin_<기계>_<날짜>_<시>.sqlite3` |
| 몇 벌 | **24** — `RETAIN_COUNT × 주기 ≥ 오프사이트 간격` (§6). 튜닝값이 아니다 |
| NAS | **04시 회차만** `dolfinserver2_backup/daily/` 로. 90일 + 매달 1일 영구 |
| 크기 | 29MB (VACUUM 뒤) × 24 = **700MB** |

**계약에서 지키는 것 넷.**

- **§2 스냅샷을 검사한다** (라이브가 아니라). 걸리면 **정리를 건너뛰고**
  증거본 하나(`*_INTEGRITY_FAIL.corrupt`)와 센티넬을 남긴다. 이 규칙을 어긴
  형제가 실제로 **성한 스냅샷 0개**가 된 적이 있다 — 손상된 것이 매시 들어오며
  성한 것을 밀어냈다.
- **§4 하위 트랙은 검증된 스냅샷을 소비한다.** NAS 도, 시험 자리도
  (`test_db.sh`) 라이브 DB 를 복사하지 않는다.
- **§5 `journal_mode=DELETE`** — WAL 형제가 딸려 오면 옮길 때 커밋 꼬리를 놓친다.
- **§7 반출 위생** — `django_session` 삭제 + `VACUUM`, **검증 이후에.**
  세션 키가 곧 쿠키 값이라 사본을 읽은 사람이 그대로 재제출하면 로그인된다
  (`SECRET_KEY` 가 달라도 안 막힌다).

**이름에 기계가 들어간다** (`fin_m710q_…`). 곧 GCP 에도 진짜 `fin.db` 가 생겨
같은 NAS 로 오는데, 이름이 같으면 나중에 뜬 기계가 먼저 뜬 것을 갈아 치운다 —
이 저장소가 한 번 겪은 일이다. **정리도 제 기계 갈래만 본다.**

막히면 `/healthz` 가 **`degraded`(200)** 로 말한다. 503 이 아닌 것은 —
503 은 "트래픽 보내지 말라" 는 뜻이라 배포 스크립트의 liveness 대기를 멈춰
세워, 게이트가 아니라 배포 장애가 된다.

```bash
python3 /srv/dolfinserver2/scripts/backup_db.py     # 손으로 한 번
tail -20 /srv/dolfinserver2/logs/backup.log
deploy/host/test_db.sh                              # 그 스냅샷으로 되돌려 본다
```

**`manage.py backup` 은 다른 일을 한다** — 기계 사이 인계용이다(가중치와
`--derived` 로 크롭·조각까지 NAS 로). 시간별 트랙이 "한 시간 전으로 되돌리기"
라면 그쪽은 "다른 기계에서 이어받기" 다. 자리도 갈라 뒀다
(`dolfinserver2_backup/db/` 대 `…/daily/`).

시험은 `finseg/tests_backup.py` 8종이 잰다 — **잘 도는 백업이 아니라 실패했을
때의 행동을** 잰다. 계약이 지켜지는지는 손상됐을 때에만 드러나고, 그때는 이미
늦기 때문이다.

## 걸리는 자리

- **저장소 `db/` 를 다시 쓰지 말 것.** 2026-08-26 에 `/srv/dolfinserver2/db/`
  로 옮겼고, 남아 있는 `db/fin.db.repo-retired-2026-08-26` 은 옮기던 날의
  사본이다. 그것을 `FIN_DB` 로 물리면 **운영과 갈라진 판정을 쌓게 된다.**
  (한동안 사본을 두고 `sync_data.sh` 로 맞추다가, 맞출 일 자체를 없앴다.)
- **`FIN_REID` 를 빠뜨리면 격자가 조용히 빈다.** 개체 판정은 새 조각을
  가리키는데 settings 의 기본값은 지운 옛 판이라, 화면이 텅 빈 채로 200 을
  낸다. `.env` 에 `FIN_REID=/app/reid/v3` 이 있어야 한다. `reid` 자리에서는
  `/healthz` 가 **503** 으로 그것을 말한다.
- **DEBUG=0 이면 Django 가 크롭·사진을 안 낸다** (`finweb/urls.py`). nginx 의
  `/crops`·`/photos` 가 받지 않으면 이미지가 통째로 깨진다.
- **조각(`/reid/chip/…`)은 nginx 가 아니라 Django 가 낸다.** 같은 번호로
  `look/`(사람이 보는 큰 그림)과 `chips/`(모델이 먹는 것) 중 있는 것을 고르는
  판단이 view 에 있다.
- **`/detect` 는 이미지에 없다.** `static/models/*.onnx`(37MB)와
  `static/vendor/ort`(26MB)는 내려받는 것이라 `.dockerignore` 에 있다. 화면이
  왜 없는지 스스로 말한다.

## 앞으로 — dolfinid2 (GCP)

지금 이 자리는 **시험**이다. 운영은 GCP 의 `dolfinid2` 로 올린다. 같은
이미지를 쓰되 다음이 다르다.

- **사진 마운트가 없다.** compose 의 `/nas/JikhanJung/dolfinimage` 줄을 지운다.
  `/photo` 화면은 "사진이 없다" 고 말하고 나머지는 그대로 돈다. 원본이
  필요해지면 그때 무엇을 올릴지 따로 정한다 (사진 4,181장).
- **자료를 어떻게 보내나.** `crops/`(703MB) + `reid/v3`(230MB) + `fin.db`(34MB).
  `sync_data.sh` 의 rsync 를 원격으로 돌리는 것이 가장 가깝다.
- **DB 의 주인이 바뀐다.** 지금은 2080ti 가 주인이고 여기가 사본이지만,
  수집을 GCP 에서 받기 시작하면 **거기가 주인이다.** 그때부터 2080ti 는
  받아 오는 쪽이 되고, `sync_data.sh` 의 방향이 뒤집힌다 — 그 전에
  `.guides/web/` 의 백업 레인(hourly + 무결성 게이트)을 먼저 세울 것.
  사람의 판정은 다시 만들 수 없다.
- **공개 리스너가 nginx 하나** 인 것은 같다. TLS 와 도메인은 GCP 쪽에서
  붙이고, `FIN_ALLOWED_HOSTS` · `FIN_TRUSTED_ORIGINS` 에 그 이름을 적는다
  (안 적으면 POST 가 CSRF 에서 막힌다).
