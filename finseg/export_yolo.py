#!/usr/bin/env python3
"""검토를 YOLO-seg 학습 자료로 내보낸다.

    python -m finseg.export_yolo --out datasets/v1 --dry-run
    python -m finseg.export_yolo --out datasets/v1

판정 규칙은 여기 다시 쓰지 않는다 — `rules.box_states` 를 부른다. 검토 화면에
보이는 것이 곧 라벨이어야 하고, **갈라진 것은 눈에 띄지 않는다.**

## 크롭 하나가 학습 이미지 하나다

크롭은 상자마다 하나씩 있고, **그 안에 보이는 지느러미는 전부 라벨이 된다** —
자기 상자의 것이든 이웃의 것이든. 폴리곤을 DB 에 원본 좌표로 둔 것이 여기서
쓰인다. 어느 크롭으로든 옮겨 넣을 수 있다.

## 검토 안 된 것이 하나라도 들면 그 크롭을 뺀다

이것이 자료의 신뢰를 지키는 유일한 줄이다. 라벨이 없는 지느러미가 한 장이라도
들어가면 YOLO 는 **같은 모양을 어떤 크롭에서는 지느러미라 하고 어떤 크롭에서는
배경이라고** 배운다. `import_boxes` 가 이웃을 미리 끌어와 이 일이 드물게 만들지만,
막는 것은 여기다.

`not_fin` 은 다르다 — 사람이 "여기 없다" 고 말한 것이라 라벨 없이 두어도 된다.
YOLO 는 라벨 없는 자리를 배경으로 배우고, **SAM2 가 실제로 헷갈린 자리에 붙은
음성이라 무작위 배경보다 값어치가 크다.**

## 검증을 둘로 둔다

- `val`      — 크롭 20%, 관찰일별 층화.  학습이 되고 있나
- `val_date` — 관찰일 하나를 통째로.     **다른 날에 듣나**

`val` 은 좋은데 `val_date` 가 나쁘면 그 날의 바다 상태·역광·배의 위치를 외운
것이고, 새 관찰일에는 안 듣는다는 뜻이다.
"""
import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from finseg import crops as fcrops  # noqa: E402
from finseg import db as fdb        # noqa: E402
from finseg import rules            # noqa: E402

MIN_VISIBLE = 0.10   # 크롭에 이만큼도 안 보이는 조각은 라벨로 세지 않는다


def clip_to_crop(points, size):
    """폴리곤을 크롭 사각형 안으로 자른다 → (잘린 폴리곤, 보이는 비율).

    래스터로 자르고 윤곽을 다시 딴다. 폴리곤 클리핑 라이브러리를 들이지 않고
    끝나며, 마스크를 만들 때 쓴 것과 **같은 방식**이라 결과가 어긋나지 않는다.
    """
    import cv2
    import numpy as np
    pts = np.array([[int(round(x)), int(round(y))] for x, y in points], np.int32)
    if len(pts) < 3:
        return None, 0.0
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    pad = 2
    w = int(max(x1, size) - min(x0, 0)) + 2 * pad
    h = int(max(y1, size) - min(y0, 0)) + 2 * pad
    off = np.array([pad - min(int(x0), 0), pad - min(int(y0), 0)])
    full = np.zeros((h, w), np.uint8)
    cv2.fillPoly(full, [pts + off], 1)
    total = int(full.sum())
    if total == 0:
        return None, 0.0
    keep = np.zeros_like(full)
    keep[off[1]:off[1] + size, off[0]:off[0] + size] = 1
    inside = full * keep
    vis = int(inside.sum())
    if vis == 0:
        return None, 0.0
    cnts, _ = cv2.findContours(inside, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0.0
    c = max(cnts, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, 0.8, True)
    if len(c) < 3:
        return None, 0.0
    return [(float(p[0][0] - off[0]), float(p[0][1] - off[1])) for p in c], vis / total


def _dedup_labels(polys, size, iou_min=0.85):
    """한 크롭 안에서 거의 같은 폴리곤을 하나로 줄인다 → (남은 것, 지운 수)."""
    import numpy as np
    if len(polys) < 2:
        return polys, 0
    masks = [_raster(p, size) for p in polys]
    keep, dropped = [], 0
    for i, m in enumerate(masks):
        hit = False
        for j in keep:
            u = (m | masks[j]).sum()
            if u and (m & masks[j]).sum() / u >= iou_min:
                hit = True
                break
        if hit:
            dropped += 1
        else:
            keep.append(i)
    return [polys[i] for i in keep], dropped


def _raster(points, size):
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
    ap.add_argument("--crops", default=Path("crops"), type=Path)
    ap.add_argument("--out", type=Path, required=True, help="자료 꾸러미를 둘 곳")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--val-date", help="통째로 뺄 관찰일 (기본은 씨앗으로 고른다)")
    ap.add_argument("--seed", type=int, default=20260815)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    conn = fdb.init(args.db)

    # 1. 상자마다 라벨 상태를 받아 사진별로 모은다
    per_image = defaultdict(list)
    box_image = {}
    for d in rules.box_states(conn):
        per_image[d["image_id"]].append(d)
        box_image[d["box_id"]] = d["image_id"]

    crop_rows = {r["box_id"]: dict(r)
                 for r in conn.execute("SELECT * FROM crop")}
    meta = {r["id"]: dict(r) for r in conn.execute("SELECT * FROM image")}

    # 2. 크롭마다: 안에 든 상자가 전부 판정되었나, 그리고 라벨은 무엇인가
    kept, dropped = [], Counter()
    slivers = 0
    dup_labels = [0]
    runs = set()
    for box_id, crop in crop_rows.items():
        img_id = box_image[box_id]
        size = crop["w"]
        labels, pending_inside = [], False
        for d in per_image[img_id]:
            if not d["polygon"]:
                # 마스크가 없는 상자. 판정이 not_fin 이면 없어도 된다.
                if d["label"] != rules.BACKGROUND:
                    # 이 상자가 이 크롭에 들어오는지부터 본다
                    if _overlaps(d, crop):
                        pending_inside = True
                continue
            pts = fcrops.to_crop(fdb.loads_polygon(d["polygon"]), crop)
            clipped, vis = clip_to_crop(pts, size)
            if clipped is None or vis < MIN_VISIBLE:
                if clipped is not None:
                    slivers += 1
                continue
            if d["label"] == rules.PENDING:
                pending_inside = True
                break
            if d["label"] == rules.POSITIVE:
                labels.append(clipped)
                if d["run_id"]:
                    runs.add(d["run_id"])
        # **한 크롭 안에서 거의 같은 폴리곤은 하나로 줄인다.** `import_boxes` 가
        # 중복 상자를 미리 거르지만, 옛 검출기가 한 지느러미에 상자를 둘 붙여
        # 놓은 것이 이웃으로 딸려 들어올 수 있다. 같은 것을 두 번 적으면 YOLO 는
        # 겹친 개체 둘로 배운다.
        labels, dups = _dedup_labels(labels, size)
        dup_labels[0] += dups
        if pending_inside:
            dropped["검토 안 된 상자가 들어 있다"] += 1
            continue
        kept.append({"box_id": box_id, "crop": crop, "image_id": img_id,
                     "obsdate": meta[img_id]["obsdate"], "labels": labels})

    print(f"크롭 {len(crop_rows):,} 개 중 쓸 수 있는 것 {len(kept):,}")
    for k, v in dropped.items():
        print(f"  뺐다 — {k}: {v:,}")
    if slivers:
        print(f"  크롭 가장자리에 {MIN_VISIBLE:.0%} 미만만 보이는 조각 {slivers:,} 개는"
              f" 라벨로 세지 않았다")
    if dup_labels[0]:
        print(f"  한 크롭 안에서 겹친 라벨 {dup_labels[0]:,} 개를 하나로 줄였다")
    n_lab = sum(len(k["labels"]) for k in kept)
    n_empty = sum(1 for k in kept if not k["labels"])
    print(f"라벨 {n_lab:,} 개 · 라벨 없는(배경) 크롭 {n_empty:,} 개")
    if not kept:
        sys.exit("내보낼 것이 없다 — 먼저 검토할 것.")

    # 3. 나누기
    by_date = defaultdict(list)
    for k in kept:
        by_date[k["obsdate"]].append(k)
    rng = random.Random(args.seed)
    val_date = args.val_date or rng.choice(sorted(by_date))
    if val_date not in by_date:
        sys.exit(f"그런 관찰일이 없다: {val_date}")
    split = {}
    for d, items in sorted(by_date.items()):
        if d == val_date:
            for k in items:
                split[k["box_id"]] = "val_date"
            continue
        items = sorted(items, key=lambda k: k["box_id"])
        rng.shuffle(items)
        n_val = int(round(len(items) * args.val_frac))
        for i, k in enumerate(items):
            split[k["box_id"]] = "val" if i < n_val else "train"
    counts = Counter(split.values())
    print(f"\nval_date = {val_date} (통째로 뺀다)")
    print("  " + " · ".join(f"{s} {counts[s]:,}"
                            for s in ("train", "val", "val_date")))

    if args.dry_run:
        print("\n--dry-run 이라 아무것도 쓰지 않았다.")
        return

    # 4. 꾸러미
    out = args.out
    for s in ("train", "val", "val_date"):
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)
    run_id = fdb.start_run(conn, "export", params={
        "out": str(out), "val_date": val_date, "val_frac": args.val_frac,
        "seed": args.seed})

    for k in kept:
        s = split[k["box_id"]]
        name = f"{k['box_id']:08d}"
        shutil.copyfile(args.crops / k["crop"]["path"],
                        out / "images" / s / f"{name}.jpg")
        size = k["crop"]["w"]
        lines = []
        for poly in k["labels"]:
            # YOLO-seg: 분류 뒤에 정규화된 x y 가 이어진다
            flat = " ".join(f"{min(max(x / size, 0), 1):.6f} "
                            f"{min(max(y / size, 0), 1):.6f}" for x, y in poly)
            lines.append(f"0 {flat}")
        # 라벨이 없으면 **빈 파일**이다 — 지우는 것이 아니라. YOLO 는 그것을
        # 배경 이미지로 읽고, 배경을 10% 안팎 섞으라고 권한다.
        (out / "labels" / s / f"{name}.txt").write_text(
            "".join(f"{ln}\n" for ln in lines))

    # **경로는 상대로 적는다** — 절대경로면 2080ti 로 옮긴 곳에서 안 맞는다.
    # 다만 학습은 꾸러미 안에서 실행해야 한다 (ultralytics 는 path 를 실행
    # 디렉토리 기준으로 푼다).
    (out / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\n"
        f"# 통째로 뺀 관찰일 — 성적은 이쪽으로 읽는다\nval_date: images/val_date\n"
        "\nnames:\n  0: fin\n")

    manifest = {
        "git_sha": fdb.git_sha(),
        "created": fdb.now(),
        "export_run": run_id,
        "mask_runs": sorted(runs),
        "val_date": val_date,
        "val_frac": args.val_frac,
        "seed": args.seed,
        "counts": {**{s: counts[s] for s in ("train", "val", "val_date")},
                   "labels": n_lab, "background_crops": n_empty},
        "dropped": dict(dropped),
        "slivers_ignored": slivers,
        "duplicate_labels_merged": dup_labels[0],
        "obsdates": sorted(by_date),
        # **어느 크롭을 썼는지 적는다.** val_date 가 학습에 섞였는지를
        # 나중에 물을 수 있는 유일한 수단이다.
        "boxes": {s: sorted(b for b, v in split.items() if v == s)
                  for s in ("train", "val", "val_date")},
    }
    (out / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2))
    fdb.finish_run(conn, run_id)
    print(f"\n{out} · MANIFEST.json 에 커밋 해시와 쓴 크롭 목록을 적었다")


def _overlaps(d, crop):
    """상자가 크롭에 걸치나 (마스크가 없을 때의 대용)."""
    ox = min(d["x2"], crop["x1"]) - max(d["x1"], crop["x0"])
    oy = min(d["y2"], crop["y1"]) - max(d["y1"], crop["y0"])
    if ox <= 0 or oy <= 0:
        return False
    return ox * oy >= 0.3 * (d["x2"] - d["x1"]) * (d["y2"] - d["y1"])


if __name__ == "__main__":
    main()
