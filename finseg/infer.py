#!/usr/bin/env python3
"""학습한 YOLO-seg 로 후보 마스크를 만든다 — SAM2.1 자리에 그대로 끼운다. **[GPU]**

    python -m finseg.infer --weights runs/fin-seg/weights/best.pt --redo

`segment.py` 와 **같은 표에 같은 모양으로** 넣는다 (`mask`, 원본 좌표 폴리곤).
그래야 검토 화면도 내보내기도 엔진을 모른 채로 돌고, 두 엔진의 숫자에 같은 자를
댈 수 있다. 마스크는 덮어쓰지 않고 쌓이므로 (`is_current`) 같은 상자 위에서
SAM2 와 나란히 견줄 수 있다.

## 크롭 하나에 지느러미 하나

크롭은 "가운데 것" 이라는 약속으로 만들어졌고 학습도 그렇게 시켰다(`mosaic=0`).
YOLO 가 여럿을 내면 **프롬프트 상자와 가장 많이 겹치는 것 하나**를 고른다 —
가운데에서 가장 가까운 것이 아니라 상자와 겹치는 것이다. 이웃 지느러미가
크롭 가운데로 밀려 들어온 경우에 그 둘이 갈린다.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from finseg import crops as fcrops  # noqa: E402
from finseg import db as fdb        # noqa: E402
from finseg.segment import mask_to_polygon, SIMPLIFY  # noqa: E402

CROPS = Path(os.environ.get("FIN_CROPS", "crops"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=None)
    ap.add_argument("--crops", default=CROPS, type=Path)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--conf", type=float, default=0.10,
                    help="낮게 뽑아 둔다 — 운영점은 DB 에서 다시 고를 수 있다")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--note")
    args = ap.parse_args()

    import numpy as np
    from ultralytics import YOLO

    conn = fdb.init(args.db)
    q = ("SELECT c.box_id, c.path, c.x0, c.y0, c.x1, c.y1, c.w, c.h,"
         "       b.x1 bx1, b.y1 by1, b.x2 bx2, b.y2 by2"
         "  FROM crop c JOIN box b ON b.id = c.box_id")
    if not args.redo:
        q += " WHERE c.box_id NOT IN (SELECT box_id FROM mask WHERE is_current=1)"
    q += " ORDER BY c.box_id"
    rows = conn.execute(q).fetchall()
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        print("할 것이 없다. (--redo 로 다시 할 수 있다)")
        return
    print(f"상자 {len(rows):,} 개 · {args.weights}")

    model = YOLO(args.weights)
    run_id = fdb.start_run(conn, "yolo", model=args.weights, note=args.note,
                           params={"conf": args.conf, "imgsz": args.imgsz})
    n_ok = n_none = 0
    for i, r in enumerate(rows, 1):
        res = model.predict(str(args.crops / r["path"]), imgsz=args.imgsz,
                            conf=args.conf, device=args.device, verbose=False,
                            retina_masks=True)[0]
        if res.masks is None or len(res.masks.data) == 0:
            n_none += 1
            continue
        (px1, py1), (px2, py2) = fcrops.to_crop(
            [(r["bx1"], r["by1"]), (r["bx2"], r["by2"])], r)
        data = res.masks.data.cpu().numpy() > 0.5
        confs = res.boxes.conf.cpu().numpy()

        # 프롬프트 상자와 가장 많이 겹치는 것 하나
        prompt = np.zeros(data.shape[1:], bool)
        prompt[max(0, int(py1)):int(py2) + 1, max(0, int(px1)):int(px2) + 1] = True
        overlaps = [(m & prompt).sum() for m in data]
        k = int(np.argmax(overlaps))
        if overlaps[k] == 0:
            n_none += 1
            continue

        poly_c, area_c = mask_to_polygon(data[k], SIMPLIFY)
        if poly_c is None:
            n_none += 1
            continue
        s = fcrops.scale_of(r)
        conn.execute("UPDATE mask SET is_current=0 WHERE box_id=?", (r["box_id"],))
        conn.execute(
            "INSERT INTO mask (box_id, run_id, polygon, area, conf, is_current)"
            " VALUES (?,?,?,?,?,1)",
            (r["box_id"], run_id, fdb.dumps_polygon(fcrops.to_orig(poly_c, r)),
             int(area_c / (s * s)), round(float(confs[k]), 4)))
        n_ok += 1
        if i % 200 == 0:
            conn.commit()
            print(f"  {i:,} / {len(rows):,}")
    conn.commit()
    fdb.finish_run(conn, run_id)
    print(f"마스크 {n_ok:,} 개" + (f" · 아무것도 못 낸 것 {n_none:,}" if n_none else "")
          + f"  (run {run_id})")
    if n_none:
        print("  ↑ 이것이 이 엔진의 재현율 손실이다. SAM2 는 상자를 받으면"
              " 늘 무언가를 냈다 — 견줄 때 함께 적을 것.")


if __name__ == "__main__":
    main()
