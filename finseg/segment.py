#!/usr/bin/env python3
"""SAM2.1 에 상자를 프롬프트로 넣어 후보 마스크를 만든다. **[GPU]**

    python -m finseg.segment                      # 마스크 없는 상자 전부
    python -m finseg.segment --limit 50           # 먼저 조금만
    python -m finseg.segment --device cpu         # GPU 없이 (느리다)

DiaRUGA 는 프롬프트 없는 자동 분할(AMG)로 시작해 재현율 50~60% 에서 막혔다.
여기는 **상자가 이미 있다.** 상자를 프롬프트로 주면 SAM2 는 "이 안의 것 하나" 를
따는 일만 하면 되고, 그것은 훨씬 쉬운 문제다.

**크롭을 본다, 원본을 보지 않는다.** SAM2 의 이미지 인코더는 무엇을 주든 1024 로
줄인다 — 5472px 원본을 주면 103px 지느러미가 19px 이 되어 윤곽이 뭉개진다.
640 크롭을 주면 같은 지느러미가 300px 넘게 들어간다.

마스크는 **원본 화소 좌표의 폴리곤**으로 저장한다. 크롭 좌표로 두면 `--pad` 를
바꾸는 순간 저장된 것이 전부 뜻을 잃는다.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from finseg import db as fdb           # noqa: E402
from finseg import crops as fcrops     # noqa: E402

MODEL = os.environ.get("SAM2_MODEL", "facebook/sam2.1-hiera-base-plus")
CROPS = Path(os.environ.get("FIN_CROPS", "crops"))
# 폴리곤을 얼마나 단순하게 둘 것인가 (크롭 화소). 0.8 이면 640 크롭에서
# 눈에 안 보이는 정도이고, 점이 30~80 개로 떨어져 DB 와 라벨이 가벼워진다.
SIMPLIFY = 0.8


def autocast_dtype(device):
    """2080ti 는 Turing(sm_75) 이라 **bf16 이 없다.** 그때는 fp16 이어야 한다."""
    import torch
    if device != "cuda":
        return None
    major = torch.cuda.get_device_capability()[0]
    return torch.bfloat16 if major >= 8 else torch.float16


def mask_to_polygon(mask, simplify=SIMPLIFY):
    """이진 마스크에서 가장 큰 덩어리의 바깥 윤곽 하나. 못 찾으면 None.

    **가장 큰 것 하나만 쓴다.** 물보라나 옆 개체가 딸려 들어온 조각을 함께
    두면 학습 자료의 폴리곤이 두 덩어리가 되고, YOLO-seg 는 그것을 한 개체로
    배운다. 지느러미는 하나로 이어진 형태다.
    """
    import cv2
    import numpy as np
    m = mask.astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) <= 0:
        return None, 0
    if simplify:
        c = cv2.approxPolyDP(c, simplify, True)
    if len(c) < 3:
        return None, 0
    return [(float(p[0][0]), float(p[0][1])) for p in c], int(m.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=None)
    ap.add_argument("--crops", default=CROPS, type=Path)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--device", default=None, help="cuda | cpu (기본은 있는 것)")
    ap.add_argument("--simplify", type=float, default=SIMPLIFY)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--redo", action="store_true",
                    help="이미 마스크가 있는 상자도 다시 한다")
    ap.add_argument("--note")
    args = ap.parse_args()

    import numpy as np
    import torch
    from PIL import Image
    from sam2.build_sam import build_sam2_hf
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
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
    print(f"상자 {len(rows):,} 개 · {device} · {args.model}")

    sam = build_sam2_hf(args.model, device=device)
    predictor = SAM2ImagePredictor(sam)
    dtype = autocast_dtype(device)

    run_id = fdb.start_run(conn, "sam2", model=args.model, note=args.note,
                           params={"simplify": args.simplify, "device": device,
                                   "crops": str(args.crops)})
    n_ok = n_fail = 0
    for i, r in enumerate(rows, 1):
        img = np.array(Image.open(args.crops / r["path"]).convert("RGB"))
        # 상자를 크롭 좌표로 옮긴다. 마스크는 그 반대로 되돌린다.
        (cx1, cy1), (cx2, cy2) = fcrops.to_crop(
            [(r["bx1"], r["by1"]), (r["bx2"], r["by2"])], r)
        box = np.array([cx1, cy1, cx2, cy2], dtype=np.float32)

        ctx = (torch.autocast(device, dtype=dtype) if dtype
               else torch.inference_mode())
        with torch.inference_mode(), ctx:
            predictor.set_image(img)
            masks, scores, _ = predictor.predict(box=box[None, :],
                                                 multimask_output=True)
        # **점수가 가장 높은 것 하나.** 여러 개를 남기면 사람이 고를 것이 늘고,
        # 상자가 이미 무엇을 딸지 정해 주었으므로 고를 여지가 별로 없다.
        masks = masks.reshape(-1, masks.shape[-2], masks.shape[-1])
        scores = np.asarray(scores).reshape(-1)
        k = int(scores.argmax())
        poly_c, area_c = mask_to_polygon(masks[k] > 0, args.simplify)
        if poly_c is None:
            n_fail += 1
            continue

        poly_o = fcrops.to_orig(poly_c, r)
        s = fcrops.scale_of(r)
        conn.execute("UPDATE mask SET is_current=0 WHERE box_id=?", (r["box_id"],))
        conn.execute(
            "INSERT INTO mask (box_id, run_id, polygon, area, conf, is_current)"
            " VALUES (?,?,?,?,?,1)",
            (r["box_id"], run_id, fdb.dumps_polygon(poly_o),
             int(area_c / (s * s)), round(float(scores[k]), 4)))
        n_ok += 1
        if i % 100 == 0:
            conn.commit()
            print(f"  {i:,} / {len(rows):,}")
    conn.commit()
    fdb.finish_run(conn, run_id)
    print(f"마스크 {n_ok:,} 개" + (f" · 못 딴 것 {n_fail:,}" if n_fail else "")
          + f"  (run {run_id})")


if __name__ == "__main__":
    main()
