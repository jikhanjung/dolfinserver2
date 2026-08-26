# dolfinserver2

> **세션 시작 시 `HANDOFF.md` 를 읽어 현재 상황을 파악할 것.**
> 앞으로 할 일은 `TODOs.md`, 지나간 것은 `devlog/` 에 있다.

남방큰돌고래 등지느러미 **instance segmentation** 과 그 학습 반복.

## 무엇을 하려는 프로젝트인가

기존 [dolfinserver](../dolfinserver) 는 YOLOv5 시절의 **bbox 검출**에서 멈춰 있고
게시판·경험치·접근로그가 함께 얹혀 있다. 여기서는 그것을 걷어내고
**학습 → 검토 → 재학습 한 바퀴**만 남긴다.

```
기존 bbox ─[SAM2.1 박스 프롬프트]→ 후보 마스크 ─[사람이 고름]→ 확정 마스크
                                                                    │
                        새 사진 ←[추론]─ YOLO11-seg ←[학습]─────────┘
```

**최종 목표는 개체 식별(re-ID)이다.** 분할은 거기로 가는 길이지 끝이 아니다.
그래서 지금 판정에 남기는 축들(`edges`·`base_partial`·`boxname`)이 분할에는
필요 없어 보여도 적어 둔다 — 나중에 붙이려면 전부 다시 검토해야 한다.

**이 저장소가 하지 않는 것**: 시민과학 웹 UI, 게시판, 사용자 관리, 이미지 서빙.
그것은 `dolfinserver` 의 일이다. 여기 검토 UI 는 **한 사람이 쓰는 도구**다.

## 계획

| 단계 | 무엇 | 상태 |
|---|---|---|
| 1 | 자료 경로 (표본 → 크롭 → SAM2.1 → 검토 → 내보내기) | **완료** |
| 2 | 첫 바퀴 검토 (상자 2,315개) | **완료** (2026-08-16) |
| 3 | YOLO11-seg 학습 · SAM2.1 과 비교하기 | **두 바퀴 돌았다** ← 지금 |
| 4 | 삽입점 키포인트 모델로 제안 자동화 | **완료** — 3단계보다 앞당겼다 |
| 5 | 검출기 교체 (지금은 옛 YOLOv5 상자에 의존) | **착수** — 자료·명령은 섰다 |
| 6 | **re-ID** — 뒷날 결각으로 개체 매칭 | 구상 |

**4단계를 3단계보다 먼저 했다.** 검토 중에 `fix`(밑동을 사람이 고친 비율)가
85%에서 안 움직였고, 그 제안을 고칠 수 있는 것은 키포인트 모델뿐이었다.
**남은 검토가 있을 때 붙여야 값이 있다** — 다 끝난 뒤에 붙이면 그 바퀴에는
아무 도움이 안 된다. 붙인 뒤 `fix` 가 50%까지 내려갔다.

## 구조

Django 5.2 + SQLite. **파이프라인은 전부 management command 다** — 스키마의
주인을 ORM 하나로 두려고 `sqlite3` 직접 + FastAPI 에서 옮겼다 (`devlog/…_001` 5절).

```
finseg/models.py   스키마. 판정이 왜 네 축인지 여기 적혀 있다
finseg/rules.py    **판정 규칙 한 곳.** 검토 화면과 내보내기가 같은 함수를 부른다
finseg/evaluate.py **비교 계산 한 곳.** eval_masks 와 /compare 가 같은 함수를 부른다
finseg/geometry.py 폴리곤 표기와 **원본↔크롭 사상 — 이 식은 여기 둘뿐이다**
finseg/baseline.py 아래 경계(밑동 현). **자동 탐지 실패 기록이 여기 있다**
finseg/management/commands/  import_boxes · crops · segment · export_yolo ·
                             train · infer · eval_masks
finweb/            Django 프로젝트 (settings · urls). 자료 경로는 settings 에 있다
review/            격자 검토 앱 — 기본값으로 두고 예외만 누른다
                   **`FIN_ROLE` 이 어느 길을 거는지 정한다** (`urls.patterns_for`)
                   `work` 검출·분할·밑동·검토 / `reid` 개체 분류만 —
                   개체의 주인을 한 자리로 두려는 것이다 (`HANDOFF`)
                   `/` 검토 · `/photo/<box>` 원본 · `/edit/<box>` 윤곽·밑동
                   `/compare?runs=5,16` 엔진 비교
                   대기열 여섯: 검토할 것 · 검토한 것 · 교정 대기 · 엔진 바뀜 ·
                   **새 검출**(우리 검출기가 새로 찾은 것 — 마스크가 없다) ·
                   **re-ID 후보**(격자에 실렸는데 안 본 것 — `p(fin)` 낮은 것부터).
                   `/reid` 는 판정을 안 쓴다: 거르는 자리가 여기 하나다
```

## 명령

```bash
pip install -r requirements.txt                            # 검토·자료 준비
pip install -r requirements.txt -r requirements-gpu.txt    # 2080ti

python manage.py migrate
python manage.py import_boxes --dry-run     # 표본을 먼저 눈으로 본다
#   DB 는 `/srv/dolfinserver2/db/fin.db` 다. 시험·개발은 사본을 물린다:
#   FIN_DB=<사본> python manage.py ...
python manage.py import_boxes
python manage.py crops
python manage.py segment                                   # [GPU]
python manage.py runserver 0.0.0.0:8900                    # 검토 — **시험용이다**
#   정식 검토 화면은 컨테이너다 (m710q `work` :8085 · GCP `reid`).
#   `deploy/README.md` · HANDOFF 의 `## 운영·시험·개발 세 자리`
python manage.py export_yolo --out datasets/seg-v1 --group coarse --val-date <날짜>
python manage.py train --data datasets/seg-v1                  # [GPU] 갈래는 MANIFEST 가 정한다
python manage.py infer --weights runs/<name>/weights/best.pt --compare-only   # [GPU]
python manage.py eval_masks --runs <sam2> <yolo> --date <val_date>
python manage.py promote --run <yolo run>                  # 엔진 교체 (되돌리려면 옛 run)

python manage.py export_detect --out datasets/detect-merged   # 검출 (5단계)
python manage.py train --data datasets/detect-merged --imgsz 1280 --batch 8   # [GPU]
python manage.py eval_detect --date <val_date> --weights runs/<name>/weights/best.pt
python manage.py infer_boxes --weights runs/<name>/weights/best.pt --date <날짜>  # [GPU]
python manage.py crops                                     # 새 상자를 자른다
#   → 검토 화면 `새 검출` 대기열에서 본다
python manage.py eval_detect                               # 그 판정으로 문턱별 정밀도

# 두 자리 사이 (작업 자리 → re-ID 자리). **이름에 방향이 들어 있다**
deploy/gcp/from_work_to_reid.sh          # 레인 한 번 (행 + 조각·크롭)
python manage.py export_from_work_to_reid --out <파일>   # 작업 자리가 주인인 것만
python manage.py import_from_work_to_reid --from <파일>  # re-ID 자리에서. upsert
deploy/gcp/from_reid_to_work.sh          # 되받기 (개체·개체판정을 갈아 끼운다)

python manage.py export_pose --out datasets/pose-v1        # 밑동 두 점 (4단계)
python manage.py train --data datasets/pose-v1             # [GPU]
python manage.py infer_base --weights runs/pose-v1/weights/best.pt

python manage.py test          # 판정 규칙·좌표 사상·규칙 표시 (fin.db 를 안 건드린다)
```

## 자주 빠지는 함정

- **옛 DB 는 읽기 전용으로만 연다.** 운영 웹서버가 같은 파일을 쓰고 있고
  SQLite 는 쓰기가 하나다
- **`fin.db` 를 두 기계에서 함께 열지 않는다.** 형제 프로젝트가 그것으로
  프레임 229장을 잃었다. 운영 자리는 **m710q** 한 곳이다 (한 기계 안에서
  컨테이너와 명령이 같은 파일을 여는 것은 괜찮다 — WAL 이 그것을 받는다)
- **개체 판정은 GCP 에서만 만든다** (`FIN_ROLE`). m710q 에서는 `/reid` 가 아예
  안 걸린다 — **열기만 해도 보류함 `Individual` 이 생겨서**, 링크를 숨기는
  것으로는 안 막힌다 (HANDOFF 의 `## 서버를 둘로 나눈다`)
- **2080ti 는 Turing(sm_75) 이라 bf16 이 없다.** AMP 는 fp16 이어야 한다
  (`finseg/management/commands/segment.py:autocast_dtype()`)
- **성적은 `val` 이 아니라 `val_date` 로 읽는다.** `val` 만 좋으면 그 날의
  바다 상태를 외운 것이다
- **모델을 고치면 `makemigrations` 를 함께 커밋한다.** 마이그레이션은 사람의
  판정이 든 `fin.db` 위에서 돌고, 그것은 다시 만들 수 없다 — 지우거나 좁히는
  변경은 백업을 먼저 뜬다
- **성적을 내기 전에 "이 자가 무엇을 재나" 를 물을 것.** 하루에 세 번 걸렸다 —
  SAM2 가 자기 출력을 정답으로, YOLO v1 이 자기 출력을 정답으로, 옛 검출기가
  자기 학습 날에서. **한쪽이 만든 정답으로 그쪽을 채점하면 안 된다**
  (`eval_masks` 의 `독립` 열, `eval_detect` 의 `val_date` 경고)
- **재현율에는 늘 한정이 붙는다** — "옛 YOLOv5 상자 범위 안에서". 그 검출기가
  못 본 지느러미는 이 루프 안에서 표시할 방법이 없다 (`infer_boxes` 가 그
  천장을 여는 자리다)
- **`fin.db` 는 표본이지 전부가 아니다.** 사진 92장에 옛 DB 는 상자 189개를
  들고 있는데 우리는 109개만 들여왔다. "이미 아는 상자" 를 물을 때 `fin.db`
  하고만 견주면 빠진 80개가 **"새 검출" 로 둔갑한다** (`infer_boxes --src-db`)
- **학습 모델은 검출기가 아니다.** 크롭 640 을 받아 "가운데 것" 을 분할할 뿐,
  사진 전체에서 지느러미를 찾지 않는다. SAM2 는 상자를 프롬프트로 받지만
  YOLO 는 못 받아 틀 안에서 알아서 찾는다 — 그래서 `mosaic=0` 이고, 추론 때
  여럿이 나오면 프롬프트 상자와 가장 많이 겹치는 것을 고른다
- **`CLASSES` 순서가 곧 YOLO 클래스 번호다.** 한 번 내보낸 뒤에는 `none` 앞에
  덧붙이기만 할 것. 학습 라벨을 묶는 것은 `--group` 이 따로 한다

## 형제 프로젝트 공통 표준

웹 배포·데이터 안전·운영 규약은 `.guides/web/README.md` 에 있다 (형제 프로젝트들이
같은 사고를 각각 겪고 도달한 표준). **없으면 devdocs 클론이 안 걸린 것이다** —
이 저장소에는 커밋하지 않고 형제 클론으로 가는 심볼릭 링크만 둔다.

```bash
cd .. && git clone --filter=blob:none --sparse git@github.com:jikhanjung/devdocs.git
cd devdocs && git sparse-checkout set --no-cone '/guides/'
cd ../dolfinserver2 && ln -s ../devdocs/guides .guides
```

끊어진 심볼릭 링크는 조용히 빈 디렉토리처럼 보인다 — 안 보이면 위를 먼저 볼 것.

여기서 특히 걸리는 것은 **데이터 안전**이다. `fin.db` 는 사람의 판정이라
다시 만들 수 없고, 가이드가 말하는 "operator-entered data" 가 정확히 그것이다.
배포 다섯 동사(preflight·deploy·seed·smoke·rollback)는 검토 UI 가 한 사람이
쓰는 로컬 도구인 동안에는 과하다 — 여럿이 쓰게 되면 그때 채택한다 (`TODOs.md`).

## 규약

- 커밋 메시지: 영문 첫 줄 요약 + 한국어 본문 (왜 그렇게 했는지)
- `devlog/YYYYMMDD_nnn_제목.md` 작업 기록 · `_Pnn_` 계획 · `_Rnn_` 리뷰
- **본문은 무엇을 했는지보다 왜 그렇게 했고 무엇을 버렸는지를 적는다**
- `HANDOFF.md`·`TODOs.md` 는 완료분을 며칠만 두고 지워 현행화한다
  (기록은 `devlog/` 에 영구히 남는다)
- 사진·가중치는 저장소 밖에 둔다. **DB 는 `/srv/dolfinserver2/db/` 에**
  (2026-08-26 에 저장소 `db/` 에서 옮겼다) — 검토 화면이 컨테이너로 도는데
  그것이 보는 파일과 파이프라인이 쓰는 파일이 다르면 **판정이 한쪽에만 쌓인다.**
  `settings` 의 `FIN_DB` 기본값이 그 자리이고, 다른 기계·시험·개발은 `FIN_DB`
  로 대 준다. 없는 자리를 가리키면 시스템 검사가 말한다 (`finseg/checks.py`)

### 말투

문서·커밋·화면에 쓰는 한국어. **읽는 사람이 걸리지 않는 말을 쓴다.**

- **`견주다` 말고 `비교하다`.** 뜻은 통하지만 덜 쓰는 말이라 눈에 걸린다
- **`값` 과 `가치` 를 가른다.** 영어 *value* 를 그대로 옮겨 `값` 이라 쓰지 말 것 —
  "알아 둘 **값**이 있다"(어색) / "알아 둘 **가치**가 있다"(맞다). `값` 은
  숫자·수치를 가리킬 때만 쓴다 (`base_line` 의 값, 문턱 값). 다만 `제 값을
  한다`(제 몫을 한다)처럼 굳어진 표현은 그대로 둔다
- **DB 의 table 은 `테이블`.** `표` 라고 하지 말 것 — 이 저장소의 문서는
  마크다운 표를 잔뜩 쓰는데, 그 둘이 같은 말이면 "표 셋을 나른다" 가 문서의
  표를 말하는지 DB 의 테이블을 말하는지 읽는 자리에서 갈리지 않는다.
  `finseg_review 테이블` · `테이블 셋만 주고받는다`
- 영어를 직역한 티가 나면 다시 쓴다. **번역이 아니라 한국어로 쓰는 것이다**
