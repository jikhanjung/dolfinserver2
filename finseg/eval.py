#!/usr/bin/env python3
"""엔진 둘을 같은 자로 잰다 — SAM2.1 과 학습한 YOLO-seg.

    python -m finseg.eval --runs 3 7
    python -m finseg.eval --runs 3 7 --date 2019-06-17     # val_date 에서만

**정답은 사람이 채택한 마스크다.** `ok` 는 그 마스크 그대로, `fix` 는 교정본이다.
`not_fin` 은 정답에서 빠지고, 검토 안 된 상자도 빠진다.

## 이 숫자에는 한정이 붙는다

정답이 **옛 YOLOv5 의 상자 위에서** 만들어졌다. 그 검출기가 아예 못 본 지느러미는
정답에도 없고 여기 계산에도 안 들어간다. 그러니 이것은 "지느러미를 얼마나 찾나"
가 아니라 **"주어진 상자 안의 윤곽을 얼마나 잘 따나"** 다.

**두 엔진에 같은 자를 대므로 견주기는 유효하다.** 절대값을 밖에 낼 때만 조심한다.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from finseg import crops as fcrops  # noqa: E402
from finseg import db as fdb        # noqa: E402
from finseg import rules            # noqa: E402


def rasterize(points, size):
    import cv2
    import numpy as np
    m = np.zeros((size, size), np.uint8)
    pts = np.array([[int(round(x)), int(round(y))] for x, y in points], np.int32)
    if len(pts) >= 3:
        cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=None)
    ap.add_argument("--runs", type=int, nargs="+", required=True,
                    help="견줄 mask.run_id 들")
    ap.add_argument("--date", help="이 관찰일에서만 (보통 MANIFEST 의 val_date)")
    args = ap.parse_args()

    import numpy as np
    conn = fdb.init(args.db)

    # 정답 — 사람이 채택한 것만
    truth = {}
    where = "WHERE r.verdict IS NOT NULL"
    params = ()
    if args.date:
        where += (" AND b.image_id IN (SELECT id FROM image WHERE obsdate=?)")
        params = (args.date,)
    for d in rules.box_states(conn, where, params):
        if d["label"] != rules.POSITIVE:
            continue
        truth[d["box_id"]] = d["polygon"]
    if not truth:
        sys.exit("정답이 없다 — 먼저 검토할 것"
                 + (f" ({args.date})" if args.date else ""))

    crops = {r["box_id"]: dict(r) for r in conn.execute("SELECT * FROM crop")}
    print(f"정답 {len(truth):,} 개"
          + (f" · 관찰일 {args.date}" if args.date else "") + "\n")

    hdr = f"{'run':>5} {'kind':<6} {'낸 것':>7} {'평균 IoU':>9} " \
          f"{'≥0.5':>7} {'≥0.7':>7} {'≥0.9':>7}"
    print(hdr)
    print("-" * len(hdr))
    for run_id in args.runs:
        run = conn.execute("SELECT * FROM run WHERE id=?", (run_id,)).fetchone()
        if run is None:
            print(f"{run_id:>5} — 그런 run 이 없다")
            continue
        ious = []
        produced = 0
        for box_id, tpoly in truth.items():
            m = conn.execute("SELECT polygon FROM mask WHERE box_id=? AND run_id=?"
                             " ORDER BY id DESC LIMIT 1",
                             (box_id, run_id)).fetchone()
            if m is None:
                ious.append(0.0)          # 못 낸 것은 IoU 0 이다, 빼는 것이 아니라
                continue
            produced += 1
            c = crops[box_id]
            a = rasterize(fcrops.to_crop(fdb.loads_polygon(m["polygon"]), c), c["w"])
            b = rasterize(fcrops.to_crop(fdb.loads_polygon(tpoly), c), c["w"])
            u = (a | b).sum()
            ious.append(float((a & b).sum() / u) if u else 0.0)
        v = np.array(ious)
        print(f"{run_id:>5} {run['kind']:<6} {produced:>7,} {v.mean():>9.3f} "
              f"{(v >= .5).mean():>7.1%} {(v >= .7).mean():>7.1%} "
              f"{(v >= .9).mean():>7.1%}")
        if produced < len(truth):
            print(f"      ↑ {len(truth) - produced:,} 개는 아무것도 못 냈다"
                  f" — IoU 0 으로 셌다")
    print("\n※ 정답이 옛 YOLOv5 상자 위에서 만들어졌다. 이 숫자는 '지느러미를"
          " 얼마나 찾나' 가 아니라 '상자 안의 윤곽을 얼마나 잘 따나' 다.")


if __name__ == "__main__":
    main()
