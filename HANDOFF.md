# Handoff — 현재 진행 상황

**2026-08-15.** 저장소를 열고 자료 경로를 끝까지 놓았다. **사람이 검토한 마스크는
아직 하나도 없다** — 다음 손이 할 일이 그것이다.

지나간 것은 `devlog/20260815_001_저장소_구성과_자료_경로.md`,
앞으로 할 일은 `TODOs.md`.

## 지금 있는 것

| | |
|---|---|
| 표본 | 관찰일 20일 (2016-03-15 ~ 2019-11-05) · 상자 2,315 · 사진 1,909 |
| 크롭 | 640×640 · 154MB · `crops/` |
| 옛 DB 에서 함께 온 것 | 개체명 18건 · `not_fin` 3 · `not_identifiable` 2 |
| 마스크 | **GPU 에서 아직 안 돌렸다.** CPU 로 52장만 확인 |
| 검토 | **없음** |

표본은 씨앗 `20260815` 로 다시 뽑을 수 있다. `Run` 표에 그때 쓴 인자가 남아 있다.

## 여기까지 됐다

- `manage.py migrate` · `import_boxes` · `crops` — 돈다
- `manage.py segment` — **CPU 로만** 52장 확인. 마스크 품질은 좋다 (위 윤곽이
  깨끗하고, 겹친 개체에서도 앞쪽 지느러미를 가림 경계에서 정확히 잘라 낸다)
- `export_yolo` — 사본 DB 에 가짜 판정을 넣어 경로만 확인했다. 진짜 검토는 없다
- `train` · `infer` · `eval_masks` — **한 번도 안 돌렸다**
- 검토 화면 — 뜬다. 격자·분류·날·밑동 두 점 끌기까지. **아래 경계 규칙을 늘
  띄운다** (문구는 `finseg/baseline.py` 의 `RULE`·`RULE_POINTS`)
- `python manage.py test` — 26개. 판정 규칙·좌표 사상·규칙 표시

## 2080ti 에 자리를 잡았다 (JikhanDesktop)

저장소를 옮겼고 **환경은 준비됐다. 없는 것은 자료뿐이다.**

| | |
|---|---|
| GPU | RTX 2080 Ti · torch 2.13+cu130 이 잡는다. 연산능력 **(7,5) = sm_75** 확인 |
| 꾸러미 | `.venv/` 에 torch·ultralytics·SAM-2 까지 들어갔다 |
| `fin.db` · `crops/` | **없다.** 만든 기계에서 옮기거나 여기서 다시 만든다 |
| 사진 `/srv/dolfinserver/uploads` | **없다.** NAS 가 이 기계에 안 붙어 있다 |
| 옛 DB `../dolfinserver/db.sqlite3` | **0바이트 껍데기다.** 다시 만들려면 진짜가 필요하다 |

`torch.cuda.is_bf16_supported()` 는 **2080ti 에서도 True 를 낸다.** 그것을 믿으면
안 된다 — `autocast_dtype()` 이 연산능력을 직접 보는 이유가 이것이다.

## 다음에 할 것

1. **자료를 여기로 가져온다.** `fin.db` 와 `crops/`(154MB) 를 통째로 옮기거나,
   NAS 와 진짜 옛 DB 를 붙이고 `migrate → import_boxes → crops` 를 다시 돌린다
   (씨앗 `20260815` 로 같은 표본이 나온다). **어느 쪽이든 한 곳에서만 쓴다**
2. `python manage.py segment` — 2,315개, 몇 분
3. 첫 바퀴 검토
4. `export_yolo` → `train` → `eval_masks`

## 알아 둘 것

- **재현율의 천장이 옛 YOLOv5 다.** 그 검출기가 못 본 지느러미는 이 루프 안에서
  표시할 방법이 없다. `eval_masks` 가 내는 숫자에 늘 이 한정이 붙는다
- **2080ti 는 Turing(sm_75) 이라 bf16 이 없다.** AMP 는 fp16
  (`finseg/management/commands/segment.py:autocast_dtype()`)
- **`fin.db` 를 두 기계에서 함께 열지 않는다.** SQLite 는 쓰기가 하나다
- 옛 DB(`../dolfinserver/db.sqlite3`)는 **읽기 전용으로만** 연다 — 운영 웹서버가
  같은 파일을 쓰고 있다
- **삽입점 자동 탐지는 두 번 실패했다.** 다시 시도하기 전에 `finseg/baseline.py`
  의 실패 기록을 읽을 것
