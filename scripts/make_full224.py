"""**화면이 보는 격자(`v3`, 7,912)의 224 조각 꾸러미를 만든다.**

왜 따로 만드나 — 챔피언이 배운 조각은 `v3-224`(7,888)인데 화면이 보는 격자는
`v3`(7,912)다. 앙상블 멤버는 **상자 차례가 격자와 한 톨도 안 다르게** 맞아야
실린다(`review/views._members`). 그래서 224 조각을 `v3` 차례로 다시 엮는다.

빠진 24개는 **`등지느러미가 아니다` 로 재판정된 상자**다 (08-27 격자를 뜬 뒤
09-01 사이에). 그래서 224 로 다시 뜰 수가 없다 — `reid.usable` 이 거른다.
**그 24줄만 128 조각을 224 로 늘려 채운다.** 지느러미가 아닌 것들이라 어차피
개체가 붙을 일이 없고, 0.3%다. 격자를 통째로 다시 뜨는 길(`v4`)이 제대로지만
그것은 조각·크롭·분류기를 레인으로 나르는 일이라 훨씬 비싸다.

    python scripts/make_full224.py            # → reid/v3-full224/

낸 것으로 ②를 돌린다:

    python manage.py reid_chips --out reid/v3-full224 --emb-only \
        --backbone dinov3l --deep <cls-*.blocks.pt> --emb-name emb-fullL.npz
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

SRC = Path("reid/v3")            # 화면이 보는 격자 (128 · 7,912)
REF = Path("reid/v3-224")        # 챔피언이 배운 조각 (224 · 7,888)
DST = Path("reid/v3-full224")


def main():
    a, b = np.load(SRC / "chips.npz"), np.load(REF / "chips.npz")
    ids = a["box_id"]
    pos = {int(v): i for i, v in enumerate(b["box_id"])}
    if not set(pos) <= set(int(v) for v in ids):
        sys.exit(f"{REF} 에 {SRC} 에 없는 상자가 있다 — 짝이 아니다")

    import torch
    import torch.nn.functional as Fn

    # **`npz` 는 꺼낼 때마다 통째로 푼다.** 줄마다 `b["chip"][j]` 를 쓰면
    # 1.6GB 를 7,888번 푸는 셈이고, 실제로 16분을 돌고도 안 끝났다.
    # 한 번만 꺼내 두고 그것을 벤다
    big, small = b["chip"], a["chip"]
    size = big.shape[1]
    out = np.zeros((len(ids), size, size), dtype=big.dtype)
    take = np.array([pos.get(int(v), -1) for v in ids])
    have = np.flatnonzero(take >= 0)
    out[have] = big[take[have]]
    up = np.flatnonzero(take < 0)
    if len(up):
        # **늘려서 채운 줄** — 원래 조각이 128 이라 진짜 224 가 아니다
        x = torch.from_numpy(small[up]).unsqueeze(1)
        out[up] = Fn.interpolate(x, size=(size, size), mode="bilinear",
                                 align_corners=False).squeeze(1).numpy()

    DST.mkdir(parents=True, exist_ok=True)
    # **안 누른다.** 이 기계에서만 쓰는 파생물이라 1.6GB 를 그대로 두는 것이
    # 낫다 — 누르면 19배 작아지지만 몇 분을 더 쓰고, 저쪽으로 나르는 것은
    # 이 파일이 아니라 여기서 나올 임베딩(32MB)이다
    np.savez(DST / "chips.npz", box_id=ids, facing=a["facing"], chip=out)
    shutil.copy2(SRC / "items.json", DST / "items.json")
    (DST / "README.md").write_text(
        f"`{SRC}` 차례({len(ids):,})에 `{REF}` 의 224 조각을 엮은 것.\n"
        f"**{len(up)}줄은 128 을 늘려 채웠다** — `등지느러미가 아니다` 로 재판정돼 "
        f"224 로 다시 뜰 수 없는 상자다: "
        f"{', '.join(str(int(ids[i])) for i in up)}\n\n"
        f"`scripts/make_full224.py` 가 만든다. 파생물이라 저장소에 안 담는다.\n",
        encoding="utf-8")
    n = json.loads((DST / "items.json").read_text(encoding="utf-8"))["n"]
    print(f"{DST}/chips.npz  {len(ids):,}장 · {size}px "
          f"(늘려 채운 것 {len(up)})")
    print(f"{DST}/items.json  n={n:,}")


if __name__ == "__main__":
    main()
