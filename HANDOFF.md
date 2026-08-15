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
| 마스크 | **2,315개 — 상자마다 하나씩, 빠짐없이.** SAM2.1 · 2080ti · run 5 |
| 검토 | **없음** ← 다음 손이 할 일이 이것이다 |

마스크 점수는 중앙값 0.952 · 하위 5% 가 0.863 이고, **0.8 아래가 64개(2.8%)** 다.
검토를 그 64개부터 시작하면 무엇이 틀리는지 빨리 보인다.

표본은 씨앗 `20260815` 로 다시 뽑을 수 있다. `Run` 표에 그때 쓴 인자가 남아 있다.

## 여기까지 됐다

- `manage.py migrate` · `import_boxes` · `crops` — 돈다
- `manage.py segment` — **2080ti 에서 전량 돌렸다.** 2,304개에 실패 0 (run 5).
  앞서 CPU 로 본 품질 그대로다 (위 윤곽이 깨끗하고, 겹친 개체에서도 앞쪽
  지느러미를 가림 경계에서 정확히 잘라 낸다)
- `export_yolo` — 사본 DB 에 가짜 판정을 넣어 경로만 확인했다. 진짜 검토는 없다
- `train` · `infer` · `eval_masks` — **한 번도 안 돌렸다**
- 검토 화면 — 뜬다. 격자·분류·날·밑동 두 점 끌기까지. **아래 경계 규칙을 늘
  띄운다** (문구는 `finseg/baseline.py` 의 `RULE`·`RULE_POINTS`)
- `python manage.py test` — 26개. 판정 규칙·좌표 사상·규칙 표시

## 운영 자리는 2080ti 다 (JikhanDesktop)

자료가 왔고 환경도 섰다. **`fin.db` 는 이제 여기 것이다** — 검토가 시작되면
다른 기계에서 열지 않는다.

| | |
|---|---|
| GPU | RTX 2080 Ti · torch 2.13+cu130. 연산능력 **(7,5) = sm_75** 확인 |
| 꾸러미 | `.venv/` — torch·ultralytics·SAM-2·huggingface_hub |
| `fin.db` · `crops/` | 있다. **git 으로 옮겼고 옮긴 뒤 추적에서 뺐다** (5d63a5c) |
| 사진 `/srv/dolfinserver/uploads` | **없다.** NAS 가 안 붙어 있다 — 표본을 다시 뽑으려면 필요하다 |
| 옛 DB `../dolfinserver/db.sqlite3` | **0바이트 껍데기다.** 위와 같다 |

`torch.cuda.is_bf16_supported()` 는 **2080ti 에서도 True 를 낸다.** 그것을 믿으면
안 된다 — `autocast_dtype()` 이 연산능력을 직접 보는 이유가 이것이다.

`sam2` 는 `huggingface_hub` 를 선언하지 않는다. 없으면 **모델을 내려받는 첫
순간에** 터진다 (`requirements-gpu.txt` 에 적어 두었다).

## 다음에 할 것 — **첫 바퀴 검토다**

준비는 끝났다. 남은 것은 사람의 눈뿐이다.

```bash
python manage.py runserver 0.0.0.0:8900
```

1. **아래 경계 규칙을 먼저 읽는다.** 화면 위에 떠 있다 — 밑동 현이고 수면선이
   아니다. 첫 100장과 마지막 100장의 기준이 달라지면 되살릴 수 없다
2. 검토하면서 **세 숫자를 잰다** (`TODOs.md`) — `fix` 비율 · `edges != both`
   비율 · `cls != fin` 비율. 각각 다음 결정을 좌우한다
3. `export_yolo` → `train` → `eval_masks`. **성적은 `val_date` 로 읽는다**

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
