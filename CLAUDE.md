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
| 1 | 자료 경로 (표본 → 크롭 → SAM2.1 → 검토 → 내보내기) | 뼈대 완료 |
| 2 | 첫 바퀴 검토 (상자 2,315개) | 미착수 |
| 3 | YOLO11-seg 학습 · SAM2.1 과 견주기 | 미착수 |
| 4 | 삽입점 키포인트 모델로 제안 자동화 | 미착수 |
| 5 | 검출기 교체 (지금은 옛 YOLOv5 상자에 의존) | 미착수 |
| 6 | **re-ID** — 뒷날 결각으로 개체 매칭 | 구상 |

## 구조

```
finseg/db.py       fin.db 스키마. 판정이 왜 여러 축인지 여기 적혀 있다
finseg/rules.py    **판정 규칙 한 곳.** 검토 UI 와 내보내기가 같은 함수를 부른다
finseg/baseline.py 아래 경계(밑동 현). **자동 탐지 실패 기록이 여기 있다**
finseg/import_boxes.py · crops.py · segment.py · export_yolo.py · train.py · infer.py · eval.py
finseg/review/     격자 검토 — 기본값으로 두고 예외만 누른다
```

## 명령

```bash
pip install -r requirements.txt                            # 검토·자료 준비
pip install -r requirements.txt -r requirements-gpu.txt    # 2080ti

python -m finseg.import_boxes --dry-run     # 표본을 먼저 눈으로 본다
python -m finseg.import_boxes
python -m finseg.crops
python -m finseg.segment                                   # [GPU]
uvicorn finseg.review.app:app --host 0.0.0.0 --port 8900
python -m finseg.export_yolo --out datasets/v1
python -m finseg.train --data datasets/v1                  # [GPU]
python -m finseg.eval --runs <sam2> <yolo> --date <val_date>
```

## 자주 빠지는 함정

- **옛 DB 는 읽기 전용으로만 연다.** 운영 웹서버가 같은 파일을 쓰고 있고
  SQLite 는 쓰기가 하나다
- **`fin.db` 를 두 기계에서 함께 열지 않는다.** 형제 프로젝트가 그것으로
  프레임 229장을 잃었다. 운영 자리는 2080ti 한 곳이다
- **2080ti 는 Turing(sm_75) 이라 bf16 이 없다.** AMP 는 fp16 이어야 한다
  (`segment.py:autocast_dtype()`)
- **성적은 `val` 이 아니라 `val_date` 로 읽는다.** `val` 만 좋으면 그 날의
  바다 상태를 외운 것이다
- **`CREATE TABLE IF NOT EXISTS` 는 있는 표를 고치지 않는다.** 칸을 더할 때는
  `db.migrate()` 에도 적는다
- **재현율에는 늘 한정이 붙는다** — "옛 YOLOv5 상자 범위 안에서". 그 검출기가
  못 본 지느러미는 이 루프 안에서 표시할 방법이 없다

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
- 사진·DB·가중치는 저장소 밖에 둔다
