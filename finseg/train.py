#!/usr/bin/env python3
"""YOLO11-seg 를 크롭 자료로 학습시킨다. **[GPU]**

    python -m finseg.train --data datasets/v1 --model yolo11m-seg.pt

**꾸러미 안에서 돈다.** ultralytics 는 `data.yaml` 의 `path:` 를 yaml 위치가
아니라 실행 디렉토리 기준으로 푼다 — 상대경로로 적어 두고 여기서 `cd` 한다.

## 증강 — 배 위에서 찍은 바다 사진에 맞는 것만

기본값은 자연 사진 기준이라 그대로 쓰면 안 되는 것이 있고, **DiaRUGA 와 반대인
것이 둘 있다.** 현미경 슬라이드와 바다는 다르다.

| | 여기 | 왜 |
|---|---|---|
| `flipud` | **0** | 등지느러미는 위를 향한다. 위아래를 뒤집으면 **있을 수 없는 사진**이 된다 (DiaRUGA 는 0.5 였다 — 슬라이드 위 규조각에는 방향이 없다) |
| `fliplr` | 0.5 | 돌고래는 어느 쪽으로도 헤엄친다 |
| `degrees` | 10 | 배가 흔들리는 만큼이지 그 이상은 아니다 |
| `scale` | 0.5 | **넉넉히.** 겉보기 크기가 거리에 따라 크게 변한다 (DiaRUGA 는 크기가 곧 종의 단서라 좁게 잡았다) |
| `hsv_*` | 기본값 | 맑은 날·흐린 날·역광에서 바다 색이 크게 다르다 |
| `mosaic` | **0** | 크롭은 "가운데 것 하나" 라는 약속이다. 넷을 이어 붙이면 그 약속이 깨지고, 추론은 늘 가운데 있는 크롭에 한다 |

`mosaic` 을 끈 것이 이 자료의 핵심 전제다. 이 모델은 **SAM2.1 자리에 들어가는
것**이지 사진 전체에서 지느러미를 찾는 검출기가 아니다. 검출기를 갈아 끼우는
것은 별개의 일이고, 그때는 타일링과 함께 다시 생각해야 한다.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from finseg import db as fdb  # noqa: E402

AUG = dict(flipud=0.0, fliplr=0.5, degrees=10.0, scale=0.5, mosaic=0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=None)
    ap.add_argument("--data", type=Path, required=True, help="자료 꾸러미")
    ap.add_argument("--model", default="yolo11m-seg.pt")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--imgsz", type=int, default=640, help="크롭 한 변과 같게")
    ap.add_argument("--batch", type=int, default=16,
                    help="자료가 작을 때는 키우지 않는다 — 한 에폭의 갱신 횟수가"
                         " 줄어드는 쪽이 손해다")
    ap.add_argument("--name", default="fin-seg")
    ap.add_argument("--device", default=0)
    args = ap.parse_args()

    data = args.data.resolve()
    man = data / "MANIFEST.json"
    if not man.exists():
        sys.exit(f"MANIFEST.json 이 없다: {man}  (export_yolo 로 만든 꾸러미인가)")
    manifest = json.loads(man.read_text())
    print(f"자료   {data}")
    print(f"  만든 코드 {manifest['git_sha']} · 마스크 run {manifest['mask_runs']}")
    print(f"  {manifest['counts']}")
    print(f"  val_date = {manifest['val_date']} (학습에 넣지 않는다)")

    from ultralytics import YOLO

    conn = fdb.init(args.db)
    run_id = fdb.start_run(conn, "train", model=args.model, params={
        "data": str(data), "epochs": args.epochs, "imgsz": args.imgsz,
        "batch": args.batch, "manifest_git_sha": manifest["git_sha"],
        "mask_runs": manifest["mask_runs"], "val_date": manifest["val_date"],
        **AUG})

    cwd = os.getcwd()
    os.chdir(data)                     # ultralytics 가 path: 를 여기서 푼다
    try:
        model = YOLO(args.model)
        model.train(data="data.yaml", epochs=args.epochs, imgsz=args.imgsz,
                    batch=args.batch, device=args.device, name=args.name,
                    project=str(Path(cwd) / "runs"),
                    # 마지막 에폭에서 배경 분포를 진짜로 되돌린다
                    close_mosaic=0, **AUG)
    finally:
        os.chdir(cwd)
    fdb.finish_run(conn, run_id)
    print(f"\nrun {run_id} · 가중치는 runs/{args.name}/weights/best.pt")
    print("성적은 val 이 아니라 **val_date** 로 읽을 것 — "
          "python -m finseg.eval --weights ... --data ...")


if __name__ == "__main__":
    main()
