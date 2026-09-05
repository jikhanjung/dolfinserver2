"""**뒷날 크롭 격자** (`reid/v3-te`) — apex·뒷날·뒷삽입점만 잘라 다시 세운다.

`devlog/20260902_001` 3절의 레시피 그대로다. notch 의 상대 크기가 1.4~1.8배가
되어, "notch(2~5px)가 패치(14~16px)보다 작다" 는 천장을 **잘라서** 민다.

    apex        `out` 의 y 최솟값 점
    뒷삽입점     `base[1]`  (조각은 앞이 늘 왼쪽이다)
    뒷날        apex 오른쪽의 윤곽점들
    → 셋의 bbox 에 여백 6% · 128×128 로 **비율 무시하고** 늘린다

**원료는 `v3-full224` 의 조각이다** — 크롭 원본에서 다시 자르지 않는다.
좌표(`out`·`base`)가 조각 좌표계(0~1)라 조각에서 바로 잘리고, 그래야 챔피언을
낸 `v3-te` 와 같은 식이 된다. 차례·`box_id`·`facing` 은 그대로 나른다.

    python scripts/make_te.py            # → reid/v3-te/
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

SRC = Path("reid/v3-full224")     # 화면 차례(7,912)의 224 조각


def box_of(item, PAD, square, left):
    """한 조각의 뒷날 bbox (0~1 조각 좌표) → `(x0, y0, x1, y1)`."""
    out = np.asarray(item["out"], dtype=np.float64)
    base = np.asarray(item["base"], dtype=np.float64)
    apex = out[out[:, 1].argmin()]
    post = base[1]                       # 앞이 왼쪽이므로 뒤는 오른쪽 점
    tail = out[out[:, 0] > apex[0]]      # apex 오른쪽 윤곽 = 뒷날
    pts = np.vstack([apex[None], post[None]] + ([tail] if len(tail) else []))
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    x0 -= left                            # apex 왼쪽(지느러미 몸통)을 더 담는다
    px, py = (x1 - x0) * PAD, (y1 - y0) * PAD
    x0, y0, x1, y1 = x0 - px, y0 - py, x1 + px, y1 + py
    if square:
        # **비율을 지킨다** — 짧은 변을 긴 변에 맞춰 넓힌다. 안 그러면 뒷날
        # bbox 가 1:3 이라 정사각으로 늘릴 때 가로가 5배 늘어난다
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        h = max(x1 - x0, y1 - y0) / 2
        x0, y0, x1, y1 = cx - h, cy - h, cx + h, cy + h
    return (max(0.0, x0), max(0.0, y0), min(1.0, x1), min(1.0, y1))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reid/v3-te")
    ap.add_argument("--pad", type=float, default=0.06)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--square", action="store_true", help="비율을 지킨다")
    ap.add_argument("--left", type=float, default=0.0, help="apex 왼쪽을 더 담는다")
    opt = ap.parse_args()
    DST, OUT = Path(opt.out), opt.size
    z = np.load(SRC / "chips.npz")
    items = json.loads((SRC / "items.json").read_text(encoding="utf-8"))["items"]
    ids, chip = z["box_id"], z["chip"]
    if not (ids == np.array([it["id"] for it in items])).all():
        sys.exit(f"{SRC} 의 조각과 `items.json` 차례가 다르다")

    import torch
    import torch.nn.functional as Fn

    n, size = len(ids), chip.shape[1]
    out = np.zeros((n, OUT, OUT), dtype=chip.dtype)
    thin = 0
    for i, it in enumerate(items):
        x0, y0, x1, y1 = box_of(it, opt.pad, opt.square, opt.left)
        a, b = int(x0 * size), int(y0 * size)
        c, d = int(np.ceil(x1 * size)), int(np.ceil(y1 * size))
        # **너무 얇으면 그 자리를 넓힌다** — 뒷날이 거의 수직인 조각에서 폭이
        # 0 이 되면 `interpolate` 가 터진다. 드물지만 격자 전체가 멎는다
        if c - a < 4:
            c, a = min(size, a + 4), max(0, min(a, size - 4))
            thin += 1
        if d - b < 4:
            d, b = min(size, b + 4), max(0, min(b, size - 4))
            thin += 1
        x = torch.from_numpy(chip[i, b:d, a:c])[None, None]
        out[i] = Fn.interpolate(x, size=(OUT, OUT), mode="bilinear",
                                align_corners=False)[0, 0].numpy()

    DST.mkdir(parents=True, exist_ok=True)
    np.savez(DST / "chips.npz", box_id=ids, facing=z["facing"], chip=out)
    shutil.copy2(SRC / "items.json", DST / "items.json")
    (DST / "README.md").write_text(
        f"`{SRC}` 의 조각에서 **apex·뒷날·뒷삽입점의 bbox**(여백 {opt.pad:.0%}"
        f"{' · 비율 유지' if opt.square else ''}{f' · 왼쪽 +{opt.left}' if opt.left else ''})만 "
        f"잘라 {OUT}×{OUT} 로 늘린 격자. {n:,}장 · 차례는 화면 격자 그대로.\n"
        f"`devlog/20260902_001` 3절의 레시피 · `scripts/make_te.py` 가 만든다.\n"
        f"파생물이라 저장소에 안 담는다.\n", encoding="utf-8")
    print(f"{DST}/chips.npz  {n:,}장 · {OUT}px"
          + (f" (너무 얇아 넓힌 변 {thin})" if thin else ""))


if __name__ == "__main__":
    main()
