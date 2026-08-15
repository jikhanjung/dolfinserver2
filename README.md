# dolfinserver2

남방큰돌고래 등지느러미 **instance segmentation** 과 그 학습 반복.

```
기존 bbox ─[SAM2.1 박스 프롬프트]→ 후보 마스크 ─[사람이 고름]→ 확정 마스크
                                                                    │
                        새 사진 ←[추론]─ YOLO11-seg ←[학습]─────────┘
```

기존 [dolfinserver](../dolfinserver) 는 YOLOv5 시절의 **bbox 검출**에서 멈춰 있고
게시판·경험치·접근로그가 함께 얹혀 있다. 여기서는 **학습 → 검토 → 재학습 한 바퀴**
만 남긴다. 시민과학 UI 와 개체 식별(JTA###)은 이 저장소의 일이 아니다.

## 시작점이 다르다

DiaRUGA 는 SAM2 AMG(프롬프트 없는 자동 분할)로 시작해 재현율 50~60% 에서 막혔다.
여기는 **이미 사람이 쓸 수 있는 bbox 가 999,528 개 있다** — SAM2.1 에 박스를
프롬프트로 넣으면 분할은 처음부터 거의 풀린 문제다. 병목은 사람의 검토뿐이다.

대신 **재현율의 천장이 YOLOv5 다.** 옛 검출기가 못 본 지느러미는 이 루프 안에서
표시할 방법이 없다. 첫 바퀴에서는 감수하고, 숫자를 밖에 낼 때 한정을 함께 적는다.

## 실측 — 왜 크롭인가

| | |
|---|---|
| 원본 | 5472 × 3648 |
| bbox 폭 | 10% 53px · 50% **103px** · 90% 203px · 최대 1654px |
| bbox 높이 | 10% 48px · 50% 87px · 90% 166px |
| 면적 > 15000 | 28.4% (40만 표본) |
| 지느러미 없는 사진 | 68,365 장 (12.5%) |

전체 사진을 `imgsz=1280` 으로 넣으면 축소비가 0.23 이라 **중앙값 지느러미가
24px** 로 뭉개진다(YOLO 의 작은 물체 기준 32px 아래). DiaRUGA 가 타일링을 버릴 수
있었던 것은 원본이 2752 이고 대상이 169px 이라 79px 로 들어갔기 때문이고,
여기는 그 조건이 아니다.

그래서 **크롭 단위로 분할한다** — bbox 둘레를 여유 있게 잘라 한 장에 지느러미
하나. SAM2.1 의 인터페이스(상자 넣으면 마스크가 나온다)와 같은 모양이라, 학습한
모델을 SAM2.1 자리에 그대로 끼울 수 있다. 검출기 자체의 교체는 별개의 일로 둔다.

## 규칙

DiaRUGA 가 값을 치르고 얻은 것을 그대로 가져온다.

- **마스크는 덮어쓰지 않고 쌓는다** (`is_current`). 엔진을 갈아도 같은 자리에서
  나란히 견줄 수 있어야 한다
- **거부한 것도 기하째 남긴다.** 어려운 음성 표본이고, 재현율의 분모다
- **판정 규칙은 함수 하나에만 둔다.** 뷰어와 내보내기가 같은 것을 부른다 —
  갈라지면 화면과 학습 자료가 어긋나고, 어긋난 것은 눈에 안 띈다
- **검토하지 않은 사진은 학습에서 통째로 뺀다.** 넣으면 SAM2 의 오검출까지 배운다
- **`MANIFEST.json` 에 커밋 해시와 쓴 사진 목록을 적는다.** 어떤 자료로 학습한
  모델인지 반드시 나중에 묻게 된다
- **검증을 둘로 둔다** — 무작위 `val`(학습이 되나) 과 통째로 뺀 관찰일
  `val_date`(다른 날·다른 무리에 듣나)

## 구성

```
import_boxes → crops → segment(SAM2.1) → review → export_yolo → train → infer
                 ↑                                                        │
                 └────────────────── 다음 바퀴 ───────────────────────────┘
```

| | |
|---|---|
| `finseg/db.py` | `fin.db` 스키마. **판정을 상자와 마스크로 나눈 이유**가 여기 적혀 있다 |
| `finseg/rules.py` | **판정 규칙 한 곳.** 검토 UI 와 내보내기가 같은 함수를 부른다 |
| `finseg/import_boxes.py` | 옛 `db.sqlite3` → 관찰일 층화 표본 (읽기 전용으로만 연다) |
| `finseg/crops.py` | 상자마다 정사각형 640 크롭. **원본↔크롭 사상도 여기 둘뿐이다** |
| `finseg/segment.py` | SAM2.1 박스 프롬프트 → 후보 마스크 **[GPU]** |
| `finseg/review/` | 격자 검토 — 기본 통과, 누르는 것이 예외 |
| `finseg/export_yolo.py` | 확정 마스크 → YOLO-seg 꾸러미 + `MANIFEST.json` |
| `finseg/train.py` | 학습. **증강 선택의 근거가 표로 적혀 있다** **[GPU]** |
| `finseg/infer.py` | 학습한 모델 → 후보 마스크. SAM2.1 자리에 그대로 낀다 **[GPU]** |
| `finseg/eval.py` | 두 엔진에 같은 자를 댄다 |

```bash
pip install -r requirements.txt                    # 검토·자료 준비
pip install -r requirements.txt -r requirements-gpu.txt   # 2080ti

python -m finseg.import_boxes --dry-run
python -m finseg.import_boxes
python -m finseg.crops
python -m finseg.segment                           # [GPU]
uvicorn finseg.review.app:app --host 0.0.0.0 --port 8900
python -m finseg.export_yolo --out datasets/v1
python -m finseg.train --data datasets/v1          # [GPU]
python -m finseg.eval --runs <sam2> <yolo> --date <val_date>
```

## 자료

원본은 `fin.db` (SQLite) 다. 사진은 저장소 밖에 있다.

| | |
|---|---|
| 사진 | `/srv/dolfinserver/uploads/nas/YYYY/MM/DD/` (2012~2020) |
| 기존 DB | `../dolfinserver/db.sqlite3` — 사진 544,585 · 박스 999,528 · 관찰일 253 |
| GPU | 이 기계에는 없다. SAM2.1 추론과 학습은 2080ti 에서 돈다 |

2080ti 는 Turing(sm_75)이라 **bf16 이 없다.** AMP 는 fp16 이어야 한다.
