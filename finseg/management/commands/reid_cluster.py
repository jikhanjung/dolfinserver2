"""뒷날 notch으로 **닮은 것끼리 묶는다** — re-ID 의 첫 걸음.

    python manage.py reid_cluster --dry-run
    python manage.py reid_cluster --out reid/v1 --thres 0.08

## 왜 군집부터인가

개체명이 옛 DB 에 **166종 · 상자 435개**뿐이고 상자 5개 이상인 개체가 **다섯**
이다. 지도학습으로 임베딩을 배울 자료가 못 된다. 그래서 순서를 뒤집는다 —
**닮은 것끼리 먼저 묶어 사람에게 보이면 검토 한 번에 개체명이 여러 장씩 붙고,**
그것이 나중에 임베딩을 배울 자료가 된다.

사람이 답할 질문도 쉬워진다. "이게 누구냐" 가 아니라 **"이 둘이 같은 개체냐"** 다.

## 좌현·우현을 갈라 센다

같은 개체라도 왼쪽에서 본 것과 오른쪽에서 본 것은 다른 그림이다. 뒤집어
맞추면 서로 다른 두 면을 같은 것으로 만든다. 그래서 `facing` 으로 나눈 뒤
각각 묶는다 — 한 개체가 좌현 묶음 하나와 우현 묶음 하나로 나타나는 것이 맞다.

## 문턱은 실측으로 정한다

`--thres` 는 "이보다 가까우면 같은 묶음" 이다. 값을 모르니 `--dry-run` 이
여러 문턱에서 묶음이 어떻게 갈리는지 표로 낸다. **너무 크면 다 한 덩어리가
되고 너무 작으면 아무것도 안 묶인다** — 그 사이를 눈으로 고른다.

덜 묶는 것보다 더 묶는 쪽이 낫다. 사람이 보고 가르는 것이 다음 단계라,
**안 묶인 짝은 사람이 만날 기회조차 없다.**
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from django.core.management.base import BaseCommand

from finseg import reid, rules
from finseg.models import Box


class Command(BaseCommand):
    help = "뒷날 곡선으로 닮은 것끼리 묶는다 (re-ID 첫 걸음)"

    def add_arguments(self, p):
        p.add_argument("--out", help="묶음과 곡선을 적을 디렉토리")
        p.add_argument("--thres", type=float, default=0.08)
        p.add_argument("--min-area", type=int, default=reid.MIN_AREA)
        p.add_argument("--points", type=int, default=reid.N_POINTS)
        p.add_argument("--dry-run", action="store_true",
                       help="문턱을 안 정하고 여러 값에서 어떻게 갈리는지 본다")

    def handle(self, **o):
        w = self.stdout.write
        reid.MIN_AREA = o["min_area"]
        reid.N_POINTS = o["points"]

        boxes = (Box.objects.select_related("crop")
                 .prefetch_related("masks", "reviews").order_by("id"))
        why = Counter()
        items = []          # (box, facing, curve)
        for box in boxes:
            crop = getattr(box, "crop", None)
            if crop is None:
                why["크롭이 없다"] += 1
                continue
            st = rules.resolve(box)
            ok, reason = reid.usable(st)
            if not ok:
                why[reason] += 1
                continue
            c = reid.curve_of(st, crop)
            if not len(c):
                why["뒷날을 못 뽑았다"] += 1
                continue
            items.append((box, st["facing"], c))

        w(f"상자 {boxes.count():,} 중 쓸 수 있는 것 **{len(items):,}**")
        for k, v in why.most_common():
            w(f"  뺐다 — {k}: {v:,}")
        if not items:
            w("\n표본이 없다. `--min-area` 를 낮춰 볼 것.")
            return

        # **좌현·우현을 갈라 묶는다** — 뒤집어 합치면 두 면이 하나가 된다
        sides = {}
        for side in ("left", "right"):
            idx = [i for i, (_, f, _) in enumerate(items) if f == side]
            if not idx:
                continue
            d = reid.pairwise([items[i][2] for i in idx])
            sides[side] = (idx, d)
            w(f"\n{'왼쪽' if side == 'left' else '오른쪽'}이 앞 — 표본 {len(idx):,}")
            tri = d[np.triu_indices(len(idx), 1)]
            if len(tri):
                w(f"  거리 분포: 최소 {tri.min():.4f} · 5% {np.percentile(tri,5):.4f}"
                  f" · 중앙값 {np.percentile(tri,50):.4f} · 최대 {tri.max():.4f}")

        if o["dry_run"]:
            w("\n문턱별로 어떻게 갈리나 (묶음 = 둘 이상 든 것):")
            w(f"  {'문턱':>6} {'묶음':>6} {'묶인 상자':>9} {'가장 큰 묶음':>12}")
            for th in (0.02, 0.04, 0.06, 0.08, 0.10, 0.14, 0.20):
                ng = nb = big = 0
                for idx, d in sides.values():
                    for g in reid.cluster(d, th):
                        if len(g) > 1:
                            ng += 1
                            nb += len(g)
                            big = max(big, len(g))
                w(f"  {th:6.2f} {ng:6d} {nb:9d} {big:12d}")
            w("\n**너무 크면 다 한 덩어리가 되고 너무 작으면 아무것도 안 묶인다.**")
            w("가장 큰 묶음이 표본의 몇 %인지를 보고 고를 것 — 그것이 통째로")
            w("뭉친 신호다. `--out` 을 주면 그 문턱으로 적는다.")
            return

        out_groups, total = [], 0
        for side, (idx, d) in sides.items():
            for g in reid.cluster(d, o["thres"]):
                if len(g) < 2:
                    continue
                members = [items[idx[i]][0] for i in g]
                out_groups.append({
                    "side": side,
                    "boxes": [b.id for b in members],
                    "photos": [b.image.path for b in members],
                    # 묶음 안에서 가장 먼 두 장 — 사람이 먼저 볼 자리다.
                    # 가까운 것끼리는 어차피 같아 보인다
                    "spread": float(d[np.ix_(g, g)].max()),
                })
                total += len(members)
        out_groups.sort(key=lambda x: -len(x["boxes"]))
        w(f"\n문턱 {o['thres']} · 묶음 {len(out_groups):,} · 묶인 상자 {total:,}")
        for g in out_groups[:10]:
            w(f"  {g['side']:5s} {len(g['boxes']):3d}장 · 벌어짐 {g['spread']:.4f}"
              f" · box {g['boxes'][:6]}")

        if not o["out"]:
            w("\n`--out` 을 안 줘서 아무것도 쓰지 않았다.")
            return
        out = Path(o["out"])
        out.mkdir(parents=True, exist_ok=True)
        (out / "groups.json").write_text(json.dumps({
            "thres": o["thres"], "min_area": o["min_area"],
            "points": o["points"], "n_items": len(items),
            "groups": out_groups}, ensure_ascii=False, indent=1))
        # 곡선도 함께 적는다 — 문턱을 바꿔 보려고 다시 뽑을 이유가 없다
        np.savez_compressed(
            out / "curves.npz",
            box_id=np.array([b.id for b, _, _ in items]),
            facing=np.array([f for _, f, _ in items]),
            curve=np.stack([c for _, _, c in items]))
        w(f"\n{out}")
        w("  `curves.npz` 에 곡선이 들어 있다 — 문턱만 바꿔 다시 묶을 때 쓴다")
