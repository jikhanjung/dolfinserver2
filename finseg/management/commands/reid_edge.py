"""**뒷날이 되풀이되나** — 같은 개체를 다시 찍었을 때 뒷날이 같은 모양인가.

    python manage.py reid_edge                     # 재기만
    python manage.py reid_edge --by-chord          # 원본 해상도로 갈라서

## 무엇을 재나

진짜 notch 는 같은 개체를 다시 찍으면 같은 자리에 또 있고, 마스크 떨림은
프레임마다 다르다. 둘은 크기도(현의 2~5% 대 1.9%) 공간 주파수도 비슷해서
**필터로는 못 가르는데, 되풀이되느냐로는 갈린다.**

그래서 자는 AUC 다 — **같은 개체 짝의 거리가 다른 개체 짝의 거리보다 작은
비율**. 0.5 면 아무것도 아니고 1.0 이면 완벽하다.

**좌·우는 절대 안 섞는다** (`normalize` 가 한쪽을 거울처럼 뒤집는다).

**`같은 날` 과 `날 건너뜀` 을 가른다.** 같은 날 연속 프레임은 거의 같은
그림이라 쉽고, **실제로 쓰는 자리는 날을 건너뛴 쪽**이다. 이 갈림 자체가
자주 답을 준다 — 같은 날만 오르고 날을 건너뛰면 안 오르면, 그 자가 고르는
것은 개체가 아니라 **그 사진의 것**(빛·파도·각도·마스크 잡음)이다.

## 왜 명령으로 있나

2026-08-27 에 이 측정을 하고 **스크립트를 안 남겼다**. 2026-08-31 에 다시
재려니 처음부터 짜야 했고, 그때 **0.641 이 0.568 로 내려앉았다** — 짝이
6배가 되니 작은 표본이 부풀린 값이었던 것이 드러났다. 자료가 자라면 또
재야 하고, 그때 또 짜면 또 견줄 수 없다.

## 지금까지

    (개체 42 · 짝 ~900)          같은 날   날 건너뜀     2026-08-27
    (개체 51 · 날건너 짝 5,409)                          2026-08-31
      지금 마스크 · 밑동 현 틀     0.528     0.520   (그때 0.527)
      + 밝기 스냅                 0.555     0.554   (그때 0.587)
      + 뒷날 현 틀 · 호를 그대로   0.567     0.568   (그때 0.641)

**방향은 그대로인데 크기가 반이다.** 그리고 현이 큰 조각만 골라도 날을
건너뛰면 안 오른다 — 해상도가 천장이라는 가설이 여기서 확인되지 않았다.
"""
import sys
import time
from pathlib import Path

import numpy as np
from django.conf import settings
from django.core.management.base import BaseCommand

from finseg import geometry, reid, rules
from finseg.models import Box, Identification


def _auc(pos, neg):
    """P(같은 개체 거리 < 다른 개체 거리). Mann-Whitney U 로 센다."""
    if not pos or not neg:
        return float("nan")
    p, n = np.asarray(pos), np.asarray(neg)
    r = np.concatenate([p, n]).argsort().argsort()
    u = (r[:len(p)].sum() + len(p)) - len(p) * (len(p) + 1) / 2
    return float(1 - u / (len(p) * len(n)))


class Command(BaseCommand):
    help = "뒷날 모양이 같은 개체에서 되풀이되는 정도(AUC)를 잰다"

    def add_arguments(self, p):
        p.add_argument("--dir", default=str(settings.FIN_REID),
                       help="**격자가 아니라 DB 와 크롭을 읽는다** — 이 자는 "
                            "임베딩이 아니라 윤곽을 보므로 격자 판과 무관하다. "
                            "받아 두는 것은 어느 판을 보고 잰 것인지 적기 위해서다")
        p.add_argument("--points", type=int, default=reid.N_POINTS,
                       help="뒷날을 몇 점으로 다시 샘플할지")
        p.add_argument("--by-chord", action="store_true",
                       help="**원본 해상도로 갈라서 낸다.** notch 는 현의 2~5%% 인데 "
                            "현이 원본에서 189px(중앙)뿐이라 notch 가 4~9px 이다 — "
                            "해상도가 천장이면 현이 큰 것만 모을 때 올라야 한다. "
                            "2026-08-31 에는 **같은 날만 오르고 날을 건너뛰면 "
                            "안 올랐다**")
        p.add_argument("--min-pairs", type=int, default=30,
                       help="이보다 짝이 적은 갈래는 숫자를 안 낸다 — 폭이 넓어 "
                            "읽으면 오히려 잘못 읽는다")

    def handle(self, **o):
        w = self.stdout.write
        items = self._build(o["points"], w)
        if len(items) < 10:
            w("뒷날 벡터가 너무 적다")
            return
        ch = np.array([x["chord"] for x in items])
        w(f"\n뒷날 벡터가 나온 조각 **{len(items)}** · "
          f"개체 {len({x['ind'] for x in items})} · "
          f"좌 {sum(x['side'] == 'left' for x in items)} · "
          f"우 {sum(x['side'] == 'right' for x in items)}")
        w(f"밑동 현(원본 px) 중앙 {np.median(ch):.0f} · "
          f"90% {np.percentile(ch, 90):.0f} · 최대 {ch.max():.0f}")

        sets = [("전부", items)]
        if o["by_chord"]:
            for q in (50, 75, 90):
                t = np.percentile(ch, q)
                sets.append((f"현 상위 {100 - q}% (원본 ≥{t:.0f}px)",
                             [x for x in items if x["chord"] >= t]))
        for nm, sub in sets:
            self._table(nm, sub, o["min_pairs"], w)

    def _build(self, n_pts, w):
        import cv2

        ided = dict(Identification.objects.order_by("id")
                    .values_list("box_id", "individual_id"))
        root = Path(settings.FIN_CROPS)
        out, t0 = [], time.time()
        qs = (Box.objects.filter(id__in=list(ided))
              .select_related("crop", "image").prefetch_related("reviews", "masks"))
        for i, b in enumerate(qs):
            if i and i % 500 == 0:
                print(f"  {i} … {time.time() - t0:.0f}s", file=sys.stderr)
            st = rules.resolve(b)
            crop = getattr(b, "crop", None)
            if st.get("cls") != "fin" or not st.get("polygon") or crop is None:
                continue
            base = geometry.loads(st["base_line"])
            if len(base) != 2:
                continue
            pts = rules.final_points(st, crop)
            if len(pts) < 8:
                continue
            bc = geometry.to_crop(base, crop)
            raw, c1 = reid.normalize(pts, bc, st["facing"])
            te_raw = reid.trailing_edge(raw, n_pts) if c1 > 0 else np.empty((0, 2))
            img = cv2.imread(str(root / crop.path), cv2.IMREAD_GRAYSCALE)
            te_sn = np.empty((0, 2))
            if img is not None:
                sn, c2 = reid.normalize(reid.snap_to_edge(pts, img), bc, st["facing"])
                if c2 > 0:
                    te_sn = reid.trailing_edge(sn, n_pts)
            if not len(te_raw) or not len(te_sn):
                continue
            rc = reid.rear_chord(te_sn)
            if rc is None:
                continue
            ob = np.asarray(base, float)
            out.append(dict(ind=ided[b.id], side=st["facing"],
                            day=str(b.image.obsdate), a=te_raw, b=te_sn, c=rc,
                            chord=float(np.hypot(*(ob[1] - ob[0])))))
        return out

    def _table(self, nm, sub, min_pairs, w):
        w(f"\n  ## {nm} — 조각 {len(sub)} · 개체 {len({x['ind'] for x in sub})}")
        if len(sub) < 20:
            w("     조각이 너무 적어 안 잰다")
            return
        w(f"  {'자':<34}{'같은 날':>10}{'날 건너뜀':>12}   (같은개체 짝)")
        for key, lab in (("a", "지금 마스크 · 밑동 현 틀 · 좌표"),
                         ("b", "+ 밝기 스냅"),
                         ("c", "+ 뒷날 현 틀 · **호를 그대로**")):
            bins = {"same": ([], []), "cross": ([], [])}
            for side in ("left", "right"):
                g = [x for x in sub if x["side"] == side]
                for i in range(len(g)):
                    for j in range(i + 1, len(g)):
                        d = reid.distance(g[i][key], g[j][key])
                        if not np.isfinite(d):
                            continue
                        bb = bins["same" if g[i]["day"] == g[j]["day"] else "cross"]
                        bb[0 if g[i]["ind"] == g[j]["ind"] else 1].append(d)
            cells = []
            for k in ("same", "cross"):
                p, n = bins[k]
                cells.append(f"{_auc(p, n):.3f}" if len(p) >= min_pairs else
                             f"({len(p)}짝)")
            w(f"  {lab:<34}{cells[0]:>10}{cells[1]:>12}   "
              f"({len(bins['same'][0]):,} / {len(bins['cross'][0]):,})")
