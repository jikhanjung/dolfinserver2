#!/usr/bin/env python3
"""상자마다 정사각형 크롭을 잘라 둔다.

    python -m finseg.crops --jobs 8
    python -m finseg.crops --box 1234 --show      # 한 장만 확인

## 왜 크롭인가

원본이 5472×3648 이고 지느러미 폭 중앙값이 103px 다. 통째로 넣으면 —
SAM2 의 이미지 인코더는 1024 로 줄이고 YOLO 는 `imgsz` 로 줄인다 — 지느러미가
20~25px 로 뭉개진다. 크롭을 640 으로 펴면 같은 지느러미가 300px 넘게 들어간다.

**SAM2.1 도 YOLO-seg 도 같은 크롭을 본다.** 그래야 학습한 모델을 SAM2.1 자리에
그대로 끼울 수 있고, 두 엔진의 숫자가 같은 자를 댄 것이 된다.

## 정사각형을 고집하는 이유

640×640 으로 펼 때 가로세로가 다르면 지느러미가 눌린다. **눌린 정도가 상자마다
다르면 모델이 배우는 형태가 흔들린다** — 지느러미의 윤곽이 곧 개체의 단서인
자료에서 치를 이유가 없는 대가다. 사진 가장자리에서는 정사각형을 줄이지 않고
**밀어서** 넣는다 (5472×3648 에 견줘 크롭이 훨씬 작으므로 늘 들어간다).

## 여유(`--pad`)

상자의 긴 변에 2.0 을 곱한다. 지느러미만 딱 맞게 자르면 물과의 경계가 잘려
모델이 "어디까지가 지느러미인가" 를 배울 수 없고, 옛 YOLOv5 의 상자가 조금
어긋나 있을 때 지느러미 끝이 크롭 밖으로 나간다.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from finseg import db as fdb  # noqa: E402

PHOTOS = Path(os.environ.get("FIN_PHOTOS", "/srv/dolfinserver/uploads"))
CROPS = Path(os.environ.get("FIN_CROPS", "crops"))
OUT = 640
PAD = 2.0


def crop_rect(box, img_w, img_h, pad=PAD):
    """상자를 감싸는 정사각형을 원본 좌표로 낸다 → (x0, y0, x1, y1) 배타.

    가장자리에서는 줄이지 않고 민다. 사진보다 큰 정사각형을 요구받으면
    그때만 사진 크기로 줄인다.
    """
    x1, y1, x2, y2 = box
    side = int(round(max(x2 - x1, y2 - y1) * pad))
    side = min(side, img_w, img_h)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    x0 = int(round(cx - side / 2))
    y0 = int(round(cy - side / 2))
    x0 = max(0, min(x0, img_w - side))          # 밀어 넣는다
    y0 = max(0, min(y0, img_h - side))
    return x0, y0, x0 + side, y0 + side


# ---- 좌표 사상 --------------------------------------------------------------
# **원본과 크롭을 오가는 식은 여기 둘뿐이다.** DB 의 폴리곤은 늘 원본 좌표이고
# 모델은 늘 크롭 좌표를 본다. 이 변환을 부르는 쪽마다 다시 쓰면 언젠가 한쪽이
# 반 화소씩 어긋나고, 그것은 마스크가 조금 밀린 모습으로만 나타나 눈에 안 띈다.

def scale_of(crop) -> float:
    """크롭 한 변 / 원본에서 잘린 한 변."""
    return crop["w"] / (crop["x1"] - crop["x0"])


def to_crop(points, crop):
    s = scale_of(crop)
    return [((x - crop["x0"]) * s, (y - crop["y0"]) * s) for x, y in points]


def to_orig(points, crop):
    s = scale_of(crop)
    return [(x / s + crop["x0"], y / s + crop["y0"]) for x, y in points]


def _one(job):
    """사진 한 장에 딸린 상자 전부를 한 번의 디코딩으로 처리한다.

    5MB 짜리 JPEG 를 푸는 데 0.3~0.5초 걸린다 — 상자마다 다시 열면 그만큼
    곱해진다. 표본에서는 사진 하나에 상자가 대개 하나지만, 쏠린 날에는
    한 장에 여럿 붙는다.
    """
    from PIL import Image
    rel, img_w, img_h, boxes, photos, crops_dir, out, pad = job
    try:
        with Image.open(Path(photos) / rel) as im:
            im = im.convert("RGB")
            done = []
            for bid, x1, y1, x2, y2 in boxes:
                r = crop_rect((x1, y1, x2, y2), img_w, img_h, pad)
                sub = im.crop(r).resize((out, out), Image.LANCZOS)
                # **상자 id 로 이름을 짓는다** — 사진 경로를 쓰면 같은 사진의
                # 여러 상자가 부딪히고, 순번을 쓰면 표본을 다시 뽑을 때 어긋난다.
                name = f"{bid:08d}.jpg"
                p = Path(crops_dir) / name[:3] / name   # 한 디렉토리에 몰지 않는다
                p.parent.mkdir(parents=True, exist_ok=True)
                sub.save(p, quality=92)
                done.append((bid, str(p.relative_to(crops_dir)),
                             r[0], r[1], r[2], r[3], out, out))
        return done, None
    except Exception as e:
        return [], f"{rel}: {e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=None)
    ap.add_argument("--photos", default=PHOTOS, type=Path)
    ap.add_argument("--crops", default=CROPS, type=Path, help="크롭을 둘 뿌리")
    ap.add_argument("--out", type=int, default=OUT, help="크롭 한 변 (화소)")
    ap.add_argument("--pad", type=float, default=PAD, help="상자 긴 변의 배수")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--redo", action="store_true", help="이미 있는 것도 다시 자른다")
    ap.add_argument("--limit", type=int, help="앞의 N 개만 (시험용)")
    args = ap.parse_args()

    conn = fdb.init(args.db)
    args.crops.mkdir(parents=True, exist_ok=True)

    q = ("SELECT b.id, b.x1, b.y1, b.x2, b.y2, i.path, i.width, i.height"
         "  FROM box b JOIN image i ON i.id = b.image_id")
    if not args.redo:
        q += " WHERE b.id NOT IN (SELECT box_id FROM crop)"
    q += " ORDER BY i.path, b.id"
    rows = conn.execute(q).fetchall()
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        print("자를 것이 없다. (--redo 로 다시 자를 수 있다)")
        return

    by_img = {}
    for r in rows:
        by_img.setdefault((r["path"], r["width"], r["height"]), []).append(
            (r["id"], r["x1"], r["y1"], r["x2"], r["y2"]))
    jobs = [(p, w, h, bs, str(args.photos), str(args.crops), args.out, args.pad)
            for (p, w, h), bs in by_img.items()]
    print(f"사진 {len(jobs):,} 장 · 상자 {len(rows):,} 개 · {args.jobs} 갈래")

    run_id = fdb.start_run(conn, "crop", params={
        "out": args.out, "pad": args.pad, "crops": str(args.crops)})
    n = 0
    errs = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for done, err in ex.map(_one, jobs, chunksize=8):
            if err:
                errs.append(err)
            for rec in done:
                conn.execute(
                    "INSERT OR REPLACE INTO crop (box_id, path, x0, y0, x1, y1, w, h)"
                    " VALUES (?,?,?,?,?,?,?,?)", rec)
                n += 1
            if n and n % 500 < len(done):
                conn.commit()
                print(f"  {n:,} / {len(rows):,}")
    conn.commit()
    fdb.finish_run(conn, run_id)
    print(f"크롭 {n:,} 개 → {args.crops}")
    for e in errs[:10]:
        print("  못 함:", e)
    if len(errs) > 10:
        print(f"  … 그 밖에 {len(errs) - 10} 건")


if __name__ == "__main__":
    main()
