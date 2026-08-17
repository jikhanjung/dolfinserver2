"""YOLO11 을 크롭 자료로 학습시킨다 — 분할과 밑동 키포인트 둘 다. **[GPU]**

    python manage.py train --data datasets/seg-v1                    # seg  (export_yolo)
    python manage.py train --data datasets/pose-v1               # pose (export_pose)
    python manage.py train --data datasets/detect-human --imgsz 1280   # detect (export_detect)

**검출은 크롭이 아니라 사진 한 장이 이미지 하나다.** 그래서 `imgsz` 를 크게
줘야 한다 — 4928 폭을 640 으로 줄이면 지느러미가 16px 이 되고, 중앙값 상자
(80×80)는 21px 로 살아나려면 1280 이 필요하다. 화소가 4배라 `--batch` 는 줄인다.

**갈래는 꾸러미가 말한다** (`MANIFEST.json` 의 `task`). 사람이 `--model` 로
맞춰 주게 하면 언젠가 seg 자료에 pose 가중치를 물리게 된다. 증강은 둘이 같다 —
크롭이 "가운데 것 하나" 라는 약속도 같기 때문이다. pose 에서는 `fliplr` 이
`data.yaml` 의 `flip_idx` 와 함께 돌아 두 점의 자리가 함께 바뀐다.

**꾸러미 안에서 돈다.** ultralytics 는 `data.yaml` 의 `path:` 를 yaml 위치가
아니라 실행 디렉토리 기준으로 푼다 — 상대경로로 적어 두고 여기서 `cd` 한다.

## 증강 — 배 위에서 찍은 바다 사진에 맞는 것만

기본값은 자연 사진 기준이라 그대로 쓰면 안 되는 것이 있고, **형제 프로젝트
(현미경 슬라이드)와 반대인 것이 둘 있다.**

| | 여기 | 왜 |
|---|---|---|
| `flipud` | **0** | 등지느러미는 위를 향한다. 위아래를 뒤집으면 **있을 수 없는 사진**이 된다 (슬라이드 위 규조각에는 방향이 없어 0.5 였다) |
| `fliplr` | 0.5 | 돌고래는 어느 쪽으로도 헤엄친다. **다만 re-ID 임베딩에는 쓰면 안 된다** — 지느러미는 좌현·우현에서 다르게 보인다 |
| `degrees` | 10 | 배가 흔들리는 만큼이지 그 이상은 아니다 |
| `scale` | 0.5 | **넉넉히.** 겉보기 크기가 거리에 따라 크게 변한다 (규조류는 크기가 곧 종의 단서라 좁게 잡았다) |
| `hsv_*` | 기본값 | 맑은 날·흐린 날·역광에서 바다 색이 크게 다르다 |
| `mosaic` | **0** (크롭) / **1.0** (검출) | 크롭은 "가운데 것 하나" 라는 약속이라 넷을 이어 붙이면 깨진다. **검출은 정반대** — 사진에서 여럿을 찾는 일이라 모자이크가 그것을 가르친다 |

`mosaic` 을 끈 것이 이 자료의 핵심 전제다. 이 모델은 **SAM2.1 자리에 들어가는
것**이지 사진 전체에서 지느러미를 찾는 검출기가 아니다. 검출기를 갈아 끼우는
것은 별개의 일이고, 그때는 타일링과 함께 다시 생각해야 한다.
"""
import json
import os
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from finseg import runs

# **증강은 갈래마다 다르다.** 아래 표의 근거는 자료의 성질에서 나온다.
#
# 크롭(seg·pose)은 "가운데 것 하나" 라는 약속 위에 서 있다. `mosaic` 은 넷을
# 이어 붙여 그 약속을 깨므로 0 이다. 추론도 늘 가운데 있는 크롭에 한다.
#
# **검출은 정반대다.** 사진 전체에서 여러 마리를 찾는 일이라 모자이크가 바로
# 그것을 가르치고, 작은 물체와 배경 다양성을 크게 늘린다 — 자료가 적을수록
# 효과가 크다. 처음에 크롭 값을 그대로 써서 `mosaic=0` 으로 돌렸다가 정밀도가
# 47.9%까지 무너졌다.
#
# `degrees` 도 갈린다. 크롭에서는 배가 흔들리는 만큼(10)이 뜻이 있지만, 원본
# 사진은 수평선이 거의 고정이라 옛 YOLOv5 도 0 이었다.
AUG = {
    "segment": dict(flipud=0.0, fliplr=0.5, degrees=10.0, scale=0.5, mosaic=0.0),
    "pose":    dict(flipud=0.0, fliplr=0.5, degrees=10.0, scale=0.5, mosaic=0.0),
    # 옛 `dolfin_1280_s_100.pt` 체크포인트에서 꺼낸 값과 맞춘다
    "detect":  dict(flipud=0.0, fliplr=0.5, degrees=0.0, scale=0.5, mosaic=1.0,
                    translate=0.1),
}
# 갈래마다 밑바탕 가중치가 다르다. **꾸러미의 MANIFEST 가 갈래를 말한다** —
# 사람이 `--model` 로 맞춰 주게 하면 언젠가 seg 자료에 pose 가중치를 물린다.
DEFAULT_MODEL = {"segment": "yolo11m-seg.pt", "pose": "yolo11m-pose.pt",
                 "detect": "yolo11s.pt"}


class Command(BaseCommand):
    help = "YOLO11-seg 를 학습시킨다 [GPU]"

    def add_arguments(self, p):
        p.add_argument("--data", required=True, help="자료 꾸러미")
        p.add_argument("--model", help="기본값은 꾸러미의 갈래에서 고른다")
        p.add_argument("--epochs", type=int, default=120)
        p.add_argument("--imgsz", type=int, default=640, help="크롭 한 변과 같게")
        p.add_argument("--batch", type=int, default=16,
                       help="자료가 작을 때는 키우지 않는다 — 한 에폭의 갱신"
                            " 횟수가 줄어드는 쪽이 손해다")
        p.add_argument("--name",
                       help="기본값은 자료 이름 + 갈래 — 이름만 보고 무엇으로"
                            " 무엇을 했는지 알 수 있어야 한다")
        p.add_argument("--device", default="0")

    def handle(self, **o):
        data = Path(o["data"]).resolve()
        man = data / "MANIFEST.json"
        if not man.exists():
            raise CommandError(f"MANIFEST.json 이 없다: {man}"
                               f"  (export_yolo 로 만든 꾸러미인가)")
        m = json.loads(man.read_text())
        task = m.get("task", "segment")
        if task not in DEFAULT_MODEL:
            raise CommandError(f"모르는 갈래: {task}")
        model = o["model"] or DEFAULT_MODEL[task]
        aug = AUG[task]
        # **이름은 자료에서 딴다.** `fin-det-v2` 가 `det-v1` 로 돌린 두 번째
        # 실험이었는데 이름만 보면 `det-v2` 로 돌린 것처럼 읽혔다 — 자료 판과
        # 학습 판이 각각 번호를 갖는 순간 헷갈린다.
        name = o["name"] or f"{data.name}-{task[:3]}"
        w = self.stdout.write
        w(f"자료   {data}  ({task})")
        w(f"  만든 코드 {m['git_sha']} · 마스크 run {m.get('mask_runs', '-')}")
        w(f"  {m['counts']}")
        w(f"  val_date = {m['val_date']} (학습에 넣지 않는다)")

        from ultralytics import YOLO
        run = runs.start("train", model=model, params={
            "data": str(data), "task": task,
            "epochs": o["epochs"], "imgsz": o["imgsz"],
            "batch": o["batch"], "manifest_git_sha": m["git_sha"],
            "mask_runs": m.get("mask_runs"), "val_date": m["val_date"], **aug})
        cwd = os.getcwd()
        os.chdir(data)          # ultralytics 가 path: 를 여기서 푼다
        try:
            YOLO(model).train(
                data="data.yaml", epochs=o["epochs"], imgsz=o["imgsz"],
                batch=o["batch"], device=o["device"], name=name,
                # `close_mosaic` 은 마지막 N 에폭에서 모자이크를 끄는 것이다.
                # 크롭은 애초에 안 쓰니 0 이고, 검출은 끝에서 실제 배치에
                # 맞추도록 기본값(10)을 쓴다
                project=str(Path(cwd) / "runs"),
                close_mosaic=10 if task == "detect" else 0, **aug)
        finally:
            os.chdir(cwd)
        runs.finish(run)
        w(f"\nrun {run.id} · 가중치는 runs/{name}/weights/best.pt")
        w("성적은 val 이 아니라 **val_date** 로 읽을 것 — manage.py eval_masks")
