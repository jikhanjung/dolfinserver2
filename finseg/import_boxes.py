#!/usr/bin/env python3
"""기존 `dolfinserver` 의 상자를 표본 추출해 `fin.db` 로 들인다.

    python -m finseg.import_boxes --dry-run
    python -m finseg.import_boxes --boxes 2000

**옛 DB 는 읽기 전용으로만 연다.** 운영 중인 웹서버가 같은 파일을 쓰고 있고,
SQLite 는 쓰기가 하나다.

## 무엇을 고르나

- **면적 15000 초과분에서만.** 옛 웹 UI 가 쓰던 문턱(`MINIMUM_BOX_SIZE`)과 같다.
  전체의 28.4% 다. 그보다 작은 것은 폭이 100px 아래라 윤곽을 딸 것이 별로 없고,
  사람이 검토해도 판단이 흔들린다
- **관찰일별 층화.** 253일에 걸쳐 고르게 뽑는다. 자료가 한 날짜에 쏠리면 그 날의
  바다 상태·역광·배의 위치를 외운다 — DiaRUGA 가 슬라이드 한 장에 56% 쏠려
  겪은 것과 같은 문제이고, **여기서는 처음부터 피할 수 있다**
- **파일이 실제로 있는 것만.** NAS 에 없는 사진이 DB 에 남아 있다
- **크롭 안에 들어오는 이웃 상자도 함께 들인다** (아래)

나누기(train/val/val_date)는 여기서 하지 않는다. **내보낼 때 정한다** — 검토가
얼마나 진행됐는지에 따라 달라지고, 여기서 굳히면 다시 못 고친다.

## 이웃을 함께 들이는 이유

크롭은 상자의 2배라 옆 개체의 지느러미가 함께 들어온다 — 표본에서 **9.2%** 다.
그 지느러미에 라벨이 없으면 **YOLO 는 그것을 배경으로 배운다.** 같은 모양을
어떤 크롭에서는 지느러미라 하고 어떤 크롭에서는 아니라고 하는 자료가 된다.

그냥 버릴 수도 있지만(9.2%), 그러면 **무리 지어 있는 개체만 골라 빠진다** —
정작 어려운 쪽이다. 그래서 크롭에 들어오는 상자는 전부 들여 함께 검토한다.
비용은 11% 늘어나는 것뿐이다 (사진의 상자를 통째로 들이면 2.2배가 된다).

이웃의 크롭에 또 이웃이 있을 수 있어 **폐포를 잡는다.** 사진 안의 상자 수로
막혀 있어 반드시 끝난다.
"""
import argparse
import os
import random
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from finseg import db as fdb  # noqa: E402

SRC_DB = Path(os.environ.get("FIN_SRC_DB",
                             Path(__file__).resolve().parent.parent.parent
                             / "dolfinserver" / "db.sqlite3"))
PHOTOS = Path(os.environ.get("FIN_PHOTOS", "/srv/dolfinserver/uploads"))
MIN_AREA = 15000


def bbox_of(coords_str):
    """`coords_str` 에서 상자를 뽑는다. 폴리곤이면 그 외접 상자다."""
    try:
        v = [int(round(float(x))) for x in coords_str.split(",")]
    except (ValueError, AttributeError):
        return None
    if len(v) < 4 or len(v) % 2:
        return None
    xs, ys = v[0::2], v[1::2]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def eligible(src, min_area):
    """쓸 만한 상자를 관찰일별로 모은다.

    면적은 `boxsize` 칼럼을 믿지 않고 좌표에서 다시 센다 — 그 칼럼은
    `set_finbox_size` 관리명령이 나중에 채운 것이라 좌표와 어긋날 수 있다.
    """
    by_date = defaultdict(list)
    q = ("SELECT b.id, b.dolfin_image_id, b.coords_str, b.obsdate,"
         "       i.filename, i.imagefile, i.exifdatetime"
         "  FROM dolfinrest_dolfinbox b"
         "  JOIN dolfinrest_dolfinimage i ON i.id = b.dolfin_image_id"
         " WHERE b.coords_str != '' AND b.obsdate IS NOT NULL")
    for r in src.execute(q):
        bb = bbox_of(r["coords_str"])
        if bb is None:
            continue
        area = (bb[2] - bb[0]) * (bb[3] - bb[1])
        if area < min_area:
            continue
        by_date[r["obsdate"]].append((r, bb, area))
    return by_date


def close_neighbors(src, chosen, pad, margin, min_overlap=0.3):
    """고른 상자의 크롭에 들어오는 이웃을 폐포로 끌어온다.

    `margin` 은 여유의 여유다. 나중에 `--pad` 를 조금 키워도 이웃 목록이
    어긋나지 않게 넉넉히 잡아 둔다 — 어긋나면 라벨 없는 지느러미가 조용히
    학습 자료에 들어간다. 내보낼 때 한 번 더 막지만(검토 안 된 상자가 든
    크롭은 뺀다), 여기서 막는 편이 자료를 덜 버린다.
    """
    from finseg.crops import crop_rect

    by_img = {}          # src image id -> [(box row, bbox, area)]
    for r, bb, area in chosen:
        by_img.setdefault(r["dolfin_image_id"], []).append((r, bb, area))

    # 사진마다 그 사진의 상자를 한 번에 읽는다
    pool = {}
    for sid in by_img:
        rows = []
        for r in src.execute(
                "SELECT b.id, b.dolfin_image_id, b.coords_str, b.obsdate,"
                "       i.filename, i.imagefile, i.exifdatetime"
                "  FROM dolfinrest_dolfinbox b"
                "  JOIN dolfinrest_dolfinimage i ON i.id = b.dolfin_image_id"
                " WHERE b.dolfin_image_id = ? AND b.coords_str != ''", (sid,)):
            bb = bbox_of(r["coords_str"])
            if bb:
                rows.append((r, bb, (bb[2] - bb[0]) * (bb[3] - bb[1])))
        pool[sid] = rows

    added = 0
    for sid, seeds in by_img.items():
        have = {r["id"] for r, _, _ in seeds}
        queue = list(seeds)
        while queue:
            r, bb, _ = queue.pop()
            # 사진 크기를 모르면 상자를 다 담을 만큼 크게 잡는다 — crop_rect 는
            # 사진 안으로 밀어 넣을 뿐이라 여기서는 자르지 않는 쪽이 안전하다
            big = 10 ** 6
            cx0, cy0, cx1, cy1 = crop_rect(bb, big, big, pad * margin)
            for cand, cbb, carea in pool[sid]:
                if cand["id"] in have:
                    continue
                ox = max(0, min(cbb[2], cx1) - max(cbb[0], cx0))
                oy = max(0, min(cbb[3], cy1) - max(cbb[1], cy0))
                if ox <= 0 or oy <= 0:
                    continue
                if ox * oy < min_overlap * (cbb[2] - cbb[0]) * (cbb[3] - cbb[1]):
                    continue
                have.add(cand["id"])
                item = (cand, cbb, carea)
                queue.append(item)
                by_img[sid].append(item)
                added += 1
    out = [it for items in by_img.values() for it in items]
    return out, added


def dedup(chosen, iou_min=0.9):
    """같은 사진 안에서 거의 겹치는 상자를 하나로 줄인다.

    옛 DB 에 IoU 1.0 짜리 쌍이 남아 있다 (표본 2,318 개에 3쌍). 그냥 두면
    **같은 크롭이 두 벌 생기고** 내보낼 때 한 지느러미에 폴리곤이 겹쳐 붙는다.
    나란히 헤엄치는 개체는 IoU 가 0.3~0.5 라 문턱을 높게 잡아 살려 둔다.
    """
    by_img = {}
    for item in chosen:
        by_img.setdefault(item[0]["dolfin_image_id"], []).append(item)
    out, removed = [], 0
    for items in by_img.values():
        keep = []
        for r, bb, area in sorted(items, key=lambda t: t[0]["id"]):
            hit = False
            for kr, kbb, karea in keep:
                ox = min(bb[2], kbb[2]) - max(bb[0], kbb[0])
                oy = min(bb[3], kbb[3]) - max(bb[1], kbb[1])
                if ox <= 0 or oy <= 0:
                    continue
                inter = ox * oy
                if inter / (area + karea - inter) >= iou_min:
                    hit = True
                    break
            if hit:
                removed += 1
            else:
                keep.append((r, bb, area))
        out.extend(keep)
    return out, removed


def spread(items, n):
    """정렬된 것에서 n 개를 고르게 집는다 (앞뒤 끝을 포함한다)."""
    if n >= len(items):
        return list(items)
    if n == 1:
        return [items[len(items) // 2]]
    step = (len(items) - 1) / (n - 1)
    return [items[int(round(i * step))] for i in range(n)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=None, help="fin.db 경로")
    ap.add_argument("--src", default=SRC_DB, type=Path, help="옛 db.sqlite3")
    ap.add_argument("--photos", default=PHOTOS, type=Path, help="사진 뿌리")
    ap.add_argument("--boxes", type=int, default=2000, help="들일 상자 수")
    ap.add_argument("--dates", type=int, default=20, help="쓸 관찰일 수")
    ap.add_argument("--min-area", type=int, default=MIN_AREA)
    ap.add_argument("--seed", type=int, default=20260815,
                    help="표본을 다시 뽑을 수 있게 씨앗을 적어 둔다")
    ap.add_argument("--pad", type=float, default=2.0,
                    help="크롭 여유. crops.py 의 --pad 와 같아야 한다")
    ap.add_argument("--margin", type=float, default=1.25,
                    help="이웃을 찾을 때 크롭보다 더 넓게 보는 배수")
    ap.add_argument("--no-neighbors", action="store_true",
                    help="크롭에 들어오는 이웃 상자를 들이지 않는다 (권하지 않는다)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.src.exists():
        sys.exit(f"옛 DB 가 없다: {args.src}")
    src = sqlite3.connect(f"file:{args.src}?mode=ro", uri=True)
    src.row_factory = sqlite3.Row

    print(f"옛 DB   {args.src}")
    print(f"사진    {args.photos}")
    by_date = eligible(src, args.min_area)
    total = sum(len(v) for v in by_date.values())
    print(f"면적 > {args.min_area:,} 인 상자 {total:,} 개 / 관찰일 {len(by_date)} 일")

    # 상자가 너무 적은 날은 뺀다 — 한 날에서 한둘만 뽑으면 그 날의 조건을
    # 대표하지 못하면서 자리만 차지한다.
    per_date = max(1, args.boxes // args.dates)
    usable = sorted(d for d, v in by_date.items() if len(v) >= per_date)
    if not usable:
        sys.exit("쓸 만한 관찰일이 없다 — --boxes 를 줄이거나 --dates 를 줄일 것")
    picked_dates = spread(usable, args.dates)
    print(f"관찰일 {len(picked_dates)} 일 선택 "
          f"({picked_dates[0]} ~ {picked_dates[-1]}), 날마다 {per_date} 개")

    rng = random.Random(args.seed)
    chosen = []
    for d in picked_dates:
        pool = sorted(by_date[d], key=lambda t: t[0]["id"])
        chosen.extend(rng.sample(pool, min(per_date, len(pool))))
    print(f"뽑힌 상자 {len(chosen):,} 개")

    if not args.no_neighbors:
        chosen, added = close_neighbors(src, chosen, args.pad, args.margin)
        print(f"크롭에 들어오는 이웃 {added:,} 개를 함께 들인다 "
              f"→ 모두 {len(chosen):,} 개")

    chosen, dropped = dedup(chosen)
    if dropped:
        print(f"거의 겹치는 중복 상자 {dropped:,} 개를 걸렀다 → {len(chosen):,} 개")

    if args.dry_run:
        print("\n관찰일별:")
        cnt = defaultdict(int)
        for r, _, _ in chosen:
            cnt[r["obsdate"]] += 1
        for d in picked_dates:
            print(f"  {d}  {cnt[d]:>4}  (후보 {len(by_date[d]):,})")
        print("\n--dry-run 이라 아무것도 쓰지 않았다.")
        return

    conn = fdb.init(args.db)
    run_id = fdb.start_run(conn, "import", model=str(args.src), params={
        "boxes": args.boxes, "dates": args.dates, "min_area": args.min_area,
        "seed": args.seed, "photos": str(args.photos),
    })

    from PIL import Image  # 헤더만 읽는다 — 화소를 풀지 않는다

    n_img = n_box = n_missing = 0
    seen_img = {}
    for r, bb, area in chosen:
        rel = r["imagefile"]
        if not rel:
            n_missing += 1
            continue
        full = args.photos / rel
        img_id = seen_img.get(rel)
        if img_id is None:
            if not full.exists():
                n_missing += 1
                continue
            try:
                with Image.open(full) as im:
                    w, h = im.size
            except Exception as e:
                print(f"  못 읽음 {rel}: {e}")
                n_missing += 1
                continue
            cur = conn.execute(
                "INSERT OR IGNORE INTO image (src_id, path, obsdate, exifdatetime,"
                " width, height) VALUES (?,?,?,?,?,?)",
                (r["dolfin_image_id"], rel, r["obsdate"], r["exifdatetime"], w, h))
            if cur.rowcount:
                n_img += 1
            img_id = conn.execute("SELECT id FROM image WHERE path=?",
                                  (rel,)).fetchone()["id"]
            seen_img[rel] = img_id
        cur = conn.execute(
            "INSERT OR IGNORE INTO box (src_id, image_id, x1, y1, x2, y2, area, source)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (r["id"], img_id, bb[0], bb[1], bb[2], bb[3], area, "yolov5"))
        n_box += cur.rowcount
    conn.commit()
    fdb.finish_run(conn, run_id)

    print(f"\n들임: 사진 {n_img:,} · 상자 {n_box:,}"
          + (f" · 사진 없어 건너뜀 {n_missing:,}" if n_missing else ""))
    print(f"run {run_id} · {fdb.DEFAULT_DB if args.db is None else args.db}")


if __name__ == "__main__":
    main()
