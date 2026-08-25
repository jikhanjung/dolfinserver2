"""**뒷날 복잡도 자를 여럿 내고, 어느 것이 맞는지 잰다.**

    python manage.py reid_notch --dir reid/v2            # 재기만
    python manage.py reid_notch --dir reid/v2 --write    # 격자에 적는다

## 왜 다시 만드나

지금 `roughness`(자기 자신을 매끄럽게 편 것과의 차이)가 **사람 눈과 안 맞는다.**
상자에 넣은 조각과 안 만진 조각의 결각 분포가 사실상 같았고(0.0450 대 0.0443),
결각 사분위별 top-1 차이도 8pp뿐이었다(24.2% → 32.5%). 사람은 "매끈한 것이
많아 구분이 안 된다" 고 하는데 그 자는 그것을 못 잡는다.

## 무엇으로 맞는지 아나 — **개체를 맞히는 성적과의 상관**

좋은 복잡도 자라면 **그 값이 높은 지느러미가 실제로 잘 맞아야 한다.** 특징적일수록
알아보기 쉬운 것이 이 일의 전제이기 때문이다. 그래서 자마다

- 사분위로 갈라 **top-1 이 단조로 오르나**
- 값과 맞음/틀림의 **상관계수**

를 함께 낸다. `roughness` 를 같은 자리에 세워 함께 보인다 — **옛 자와 견주지
않으면 새 자가 나은지 알 수 없다.**

성적은 `reid_cls` 가 배운 분류기로 낸다 (kNN 이 아니라). 날로 갈라 배우고
재는 날에서만 센다.
"""
import json
from pathlib import Path

import numpy as np
from django.core.management.base import BaseCommand, CommandError

from finseg import reid
from finseg.management.commands.reid_cls import Command as Cls

MEASURES = ["tort", "ic04", "ic04_dip", "ic08", "ic08_dip", "ic16", "ic16_dip",
            "notches", "rough"]


class Command(BaseCommand):
    help = "뒷날 복잡도 자를 내고 개체 맞히기 성적과의 상관을 잰다"

    def add_arguments(self, p):
        p.add_argument("--dir", default="reid/v2")
        p.add_argument("--emb", default="emb-dinov2.npz")
        p.add_argument("--test-days", type=int, default=8)
        p.add_argument("--epochs", type=int, default=400,
                       help="**`reid_cls` 와 같은 값이라야 한다** — 성적이 그 명령과 다르면 자를 견줄 근거가 없다")
        p.add_argument("--write", action="store_true",
                       help="`items.json` 에 자들을 적는다 — 화면 정렬에 쓰려면")
        p.add_argument("--seed", type=int, default=20260825)

    def handle(self, **o):
        import torch
        import torch.nn.functional as Fn

        w = self.stdout.write
        root = Path(o["dir"])
        z = np.load(root / "curves.npz")
        curves, ids, fac = z["curve"], z["box_id"], z["facing"]
        items = json.loads((root / "items.json").read_text())["items"]
        if not (np.array([it["id"] for it in items]) == ids).all():
            raise CommandError("items.json 과 curves.npz 의 상자 순서가 다르다")
        day = np.array([it["day"] for it in items])
        rough = np.array([it.get("rough") if it.get("rough") is not None else np.nan
                          for it in items])

        w(f"조각 {len(ids):,} · 뒷날 {curves.shape[1]}점")
        M = {k: np.full(len(ids), np.nan) for k in MEASURES}
        M["rough"] = rough
        bad = 0
        for i, c in enumerate(curves):
            d = reid.edge_complexity(c)
            if d is None:
                bad += 1
                continue
            for k, v in d.items():
                M[k][i] = v
            if i and i % 1000 == 0:
                w(f"  {i:,}/{len(ids):,}")
        if bad:
            w(f"  못 잰 것 {bad:,}")

        # ---- 성적 — `reid_cls` 와 같은 규약 -------------------------------
        z2 = np.load(root / o["emb"])
        # **순서를 본다.** 다른 조각 갈래에서 만든 임베딩을 물리면 라벨이 통째로
        # 어긋난 채 그럴듯한 표가 나온다 — `reid_cls`·`reid_eval` 이 같은 것을 본다
        if not (z2["box_id"] == ids).all():
            raise CommandError(f"{o['emb']} 의 상자 순서가 items.json 과 다르다")
        emb = z2["emb"]
        X = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)
        cat = reid.catalog()
        of = {b: i for i, v in cat.items() for b in v}
        lab = np.array([of.get(int(b), -1) for b in ids])
        from collections import Counter
        days = sorted({d for d in day[lab >= 0]})
        cnt = Counter(day[lab >= 0].tolist())
        test_days = set(sorted(days, key=lambda d: -cnt[d])[:o["test_days"]])
        is_test = np.array([d in test_days for d in day])

        ok = np.zeros(len(ids), bool)
        used = np.zeros(len(ids), bool)
        for side in ("left", "right"):
            m = (fac == side) & (lab >= 0)
            tr = np.where(m & ~is_test)[0]
            te = np.where(m & is_test)[0]
            classes = sorted(set(lab[tr].tolist()))
            te = np.array([i for i in te if lab[i] in classes])
            if len(tr) < 10 or not len(te) or len(classes) < 2:
                continue
            k = {c: i for i, c in enumerate(classes)}
            # **`reid_cls` 를 다시 쓰지 않고 부른다.** 여기서 따로 짜면 에폭이나
            # 씨앗이 조금씩 어긋나고(실제로 2000 대 400이었다), 그러면 이 표의
            # 성적이 `reid_cls` 가 내는 성적과 다른 것이 되어 **자를 견줄 근거가
            # 사라진다.** 게다가 `ok[]` 가 아래 상관을 전부 먹인다
            W, B = Cls()._fit(X[tr], [k[int(v)] for v in lab[tr]], len(classes),
                              o["epochs"], 0.02, 1e-3, o["seed"])
            pred = (X[te] @ W.T + B).argmax(1)
            ok[te] = pred == np.array([k[int(v)] for v in lab[te]])
            used[te] = True
        n = int(used.sum())
        if n < 20:
            raise CommandError("잴 질의가 모자란다")
        w(f"\n재는 조각 {n} · top-1 {ok[used].mean():.1%}\n")

        w(f"  {'자':<12}{'상관':>7}   사분위별 top-1 (낮은 쪽 → 높은 쪽)")
        w("  " + "-" * 62)
        rows = []
        for key in MEASURES:
            v = M[key][used]
            g = ok[used]
            m = ~np.isnan(v)
            if m.sum() < 20 or np.std(v[m]) < 1e-12:
                continue
            r = float(np.corrcoef(v[m], g[m].astype(float))[0, 1])
            q = np.percentile(v[m], [25, 50, 75])
            band = []
            for lo, hi in ((-np.inf, q[0]), (q[0], q[1]), (q[1], q[2]), (q[2], np.inf)):
                sel = m & (M[key][used] > lo) & (M[key][used] <= hi)
                band.append(f"{g[sel].mean():.0%}" if sel.sum() >= 5 else "  ·")
            rows.append((abs(r), key, r, band))
            w(f"  {key:<12}{r:>+7.3f}   " + " → ".join(f"{b:>4}" for b in band))
        rows.sort(reverse=True)
        if rows:
            w(f"\n  **가장 센 자: `{rows[0][1]}` (상관 {rows[0][2]:+.3f})**"
              f" · 옛 자 `rough` 는 "
              f"{next((f'{r:+.3f}' for _, k, r, _ in rows if k == 'rough'), '—')}")

        if o["write"]:
            f = root / "items.json"
            d = json.loads(f.read_text())
            for n_, it in enumerate(d["items"]):
                for key in MEASURES:
                    if key != "rough" and not np.isnan(M[key][n_]):
                        it[key] = round(float(M[key][n_]), 4)
            f.write_text(json.dumps(d, ensure_ascii=False))
            w(f"\n{f} 에 자 {len(MEASURES)-1}개를 적었다")
