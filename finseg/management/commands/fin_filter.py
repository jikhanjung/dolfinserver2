"""**"이건 지느러미가 아니다" 를 따로 배운다** — 분할 엔진과 다른 모델로.

    python manage.py fin_filter --fit          # 사람 판정으로 배우고 재 본다
    python manage.py fin_filter --score reid/v2   # 격자에 든 것을 훑어 의심스러운 것을 낸다

## 왜 분할 엔진으로 안 되나

`seg-v3-s` 도 분류를 낸다. 그런데 그 어휘가 **셋뿐**이다 —
`fin`·`dolphin`·`nonfin`. `--group coarse` 로 학습해서 몸통·꼬리·주둥이가
`dolphin` 한 통에 들어가 있고, 분류는 분할의 부산물이라 그 힘도 약하다.
실제로 옛 상자에서 길어 온 조각 2,321장이 **전부 `fin` 으로 통과**했는데
그중에 몸통이 섞여 있다.

## 같은 라벨로 배우면서 무엇이 다른가 — **이것이 이 명령의 전제다**

사람 판정 3,005개는 **`seg-v3` 가 이미 학습에 쓴 것**이다. 같은 라벨을 다시
쓰는 것이라 "새 정보" 는 없다. 다른 것은 셋이다.

- **특징이 다르다.** DINOv2(자기지도 ViT)의 384차원 대 YOLO 분할 백본의 부산물
- **어휘가 다르다.** 열 갈래를 그대로 쓴다 — 몸통과 꼬리를 한 통에 안 넣는다
- **일이 다르다.** 분할은 윤곽을 따는 일이고 이것은 가르는 일만 한다

그래서 **더 나을 수도 있고 아닐 수도 있다. 재야 안다.** 그리고 재는 자리는
`seg-v3` 가 못 본 자리라야 한다 — 그것이 학습한 3,005개에서 견주면 그쪽이
외운 것을 성적으로 읽게 된다.

**결정적인 시험은 이것이다**: `seg-v3` 가 `fin` 이라고 통과시킨 2,321장을
이 필터로 훑어, 의심스럽다고 짚은 것이 **실제로 몸통이면** 이 필터는
분할 엔진이 못 하는 일을 하는 것이다. 사람이 그것만 보면 된다.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from finseg import geometry, rules
from finseg.models import Box, Crop

CACHE = "reid/fin-filter-emb.npz"


class Command(BaseCommand):
    help = "지느러미인지 아닌지를 DINOv2 특징 위에서 배운다"

    def add_arguments(self, p):
        p.add_argument("--fit", action="store_true")
        p.add_argument("--score", metavar="격자", help="이 격자의 조각을 훑는다")
        p.add_argument("--cache", default=CACHE)
        p.add_argument("--test-days", type=int, default=8)
        p.add_argument("--epochs", type=int, default=600)
        p.add_argument("--top", type=int, default=30)
        p.add_argument("--write", action="store_true",
                       help="`p(fin)` 을 격자의 `items.json` 에 적는다 — 화면이 "
                            "**의심스러운 것부터** 정렬할 수 있게. 자동으로 "
                            "버리지는 않는다: 전체 성적은 분할 엔진이 낫고 "
                            "**이 필터의 값은 그것이 놓친 것을 짚는 데 있다**")
        p.add_argument("--seed", type=int, default=20260825)

    # ---- 특징 ---------------------------------------------------------------

    def _embed_boxes(self, box_ids, w):
        """상자 자리를 크롭에서 오려 DINOv2 로 재운다. **캐시한다** — CPU 로
        장당 0.1초라 5,000장이면 10분이다."""
        import torch
        import torch.nn.functional as Fn
        from PIL import Image as PImage

        crops = {c.box_id: c for c in Crop.objects.filter(box_id__in=box_ids)}
        boxes = {b.id: b for b in Box.objects.filter(id__in=box_ids)}
        m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                           pretrained=True, verbose=False).eval()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        out, kept, buf, skipped = [], [], [], []

        def flush():
            if not buf:
                return
            x = torch.stack(buf)
            x = Fn.interpolate(x, size=224, mode="bilinear", align_corners=False)
            with torch.no_grad():
                out.append(m((x - mean) / std).numpy())
            buf.clear()

        for n, bid in enumerate(box_ids):
            c, b = crops.get(bid), boxes.get(bid)
            if c is None or b is None:
                continue
            f = settings.FIN_CROPS / c.path
            if not f.exists():
                continue
            # **상자 자리만 오린다** — 크롭은 상자의 두 배라 옆 개체가 함께
            # 들어와 있고, 그것까지 보면 "옆에 뭐가 있나" 를 배운다.
            # **좌표는 `geometry.to_crop` 이 낸다** — 크롭은 원본을 640 으로
            # 줄인 것이라 1:1 이 아니다. 손으로 빼면 어긋난다
            (qx1, qy1), (qx2, qy2) = geometry.to_crop(
                [(b.x1, b.y1), (b.x2, b.y2)], c)
            x1 = max(0, min(int(qx1), c.w - 1)); x2 = max(0, min(int(qx2), c.w))
            y1 = max(0, min(int(qy1), c.h - 1)); y2 = max(0, min(int(qy2), c.h))
            if x2 - x1 < 8 or y2 - y1 < 8:
                skipped.append(bid)
                continue
            im = PImage.open(f).convert("RGB").crop((x1, y1, x2, y2))
            buf.append(torch.from_numpy(
                np.asarray(im.resize((224, 224)), np.float32) / 255).permute(2, 0, 1))
            kept.append(bid)
            if len(buf) == 32:
                flush()
            if n and n % 500 == 0:
                w(f"  {n:,}/{len(box_ids):,}")
        flush()
        if skipped:
            # **못 잰 것을 세어서 말한다** — 조용히 적게 나오는 것이 가장 나쁘다
            w(f"  너무 작아 건너뛴 상자 {len(skipped):,}")
        return np.array(kept), np.concatenate(out) if out else np.zeros((0, 384))

    def _features(self, box_ids, o, w):
        p = Path(o["cache"])
        have = {}
        if p.exists():
            z = np.load(p)
            have = {int(b): i for i, b in enumerate(z["box_id"])}
            E0, I0 = z["emb"], z["box_id"]
        need = [b for b in box_ids if b not in have]
        if need:
            w(f"특징을 새로 잰다 {len(need):,}장 (DINOv2 · CPU)")
            ids, emb = self._embed_boxes(need, w)
            if p.exists():
                ids = np.concatenate([I0, ids])
                emb = np.concatenate([E0, emb])
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p, box_id=ids, emb=emb)
        z = np.load(p)
        pos = {int(b): i for i, b in enumerate(z["box_id"])}
        X = z["emb"] / np.maximum(np.linalg.norm(z["emb"], axis=1, keepdims=True), 1e-9)
        return X, pos

    # ---- 배우기 --------------------------------------------------------------

    def handle(self, **o):
        import torch
        import torch.nn.functional as Fn

        w = self.stdout.write
        boxes = list(Box.objects.prefetch_related("reviews", "masks")
                     .select_related("image"))
        state = {b.id: rules.resolve(b) for b in boxes}
        day = {b.id: str(b.image.obsdate) for b in boxes}
        human = {b: s["cls"] for b, s in state.items() if s.get("cls")}
        w(f"사람이 판정한 상자 {len(human):,} — "
          f"지느러미 {sum(1 for v in human.values() if v=='fin'):,} · "
          f"아닌 것 {sum(1 for v in human.values() if v!='fin'):,}")

        want = sorted(human)
        pool = []
        if o["score"]:
            it = json.loads((Path(o["score"]) / "items.json").read_text())["items"]
            pool = [i["id"] for i in it]
            want = sorted(set(want) | set(pool))
        X, pos = self._features(want, o, w)

        idx = np.array([pos[b] for b in sorted(human) if b in pos])
        ids = np.array([b for b in sorted(human) if b in pos])
        y = np.array([1 if human[b] == "fin" else 0 for b in ids])
        d = np.array([day[b] for b in ids])
        days = sorted(set(d.tolist()))
        cnt = Counter(d[y == 0].tolist())
        test_days = set(sorted(days, key=lambda x: -cnt[x])[:o["test_days"]])
        te = np.array([i for i, x in enumerate(d) if x in test_days])
        tr = np.array([i for i, x in enumerate(d) if x not in test_days])
        if not len(te) or not len(tr):
            raise CommandError("날 갈래가 안 선다")
        w(f"배우는 날 {len(days)-len(test_days)} ({len(tr):,}장) · "
          f"재는 날 {len(test_days)} ({len(te):,}장 · 아닌 것 {(y[te]==0).sum()})")

        torch.manual_seed(o["seed"])
        net = torch.nn.Linear(X.shape[1], 2)
        opt = torch.optim.AdamW(net.parameters(), lr=0.02, weight_decay=1e-3)
        xt = torch.from_numpy(X[idx[tr]].astype(np.float32))
        yt = torch.tensor(y[tr])
        # **아닌 것이 훨씬 적다** — 그대로 배우면 전부 "지느러미" 라 답해도 82%다
        wgt = torch.tensor([len(yt) / max((yt == 0).sum().item(), 1),
                            len(yt) / max((yt == 1).sum().item(), 1)],
                           dtype=torch.float32)
        for _ in range(o["epochs"]):
            opt.zero_grad()
            Fn.cross_entropy(net(xt), yt, weight=wgt).backward()
            opt.step()
        with torch.no_grad():
            p_te = Fn.softmax(net(torch.from_numpy(X[idx[te]].astype(np.float32))),
                              1)[:, 1].numpy()

        w("\n재는 날에서 — **`fin` 일 확률**로 자른다")
        w(f"  {'문턱':>6}{'지느러미를 지킨 비율':>20}{'아닌 것을 걸러낸 비율':>21}")
        for t in (0.3, 0.5, 0.7, 0.9):
            keep = p_te >= t
            w(f"  {t:>6.1f}{np.mean(keep[y[te]==1]):>19.1%}{np.mean(~keep[y[te]==0]):>20.1%}")

        # 엔진은 같은 자리에서 어떻게 하나 — **다만 이 날들도 seg-v3 가 배운 날이다**
        eng = []
        for b, i in zip(ids[te], range(len(te))):
            m = state[b].get("mask")
            eng.append(getattr(m, "cls", "") or "")
        eng = np.array(eng)
        ok = eng != ""
        if ok.sum():
            w(f"\n  분할 엔진(`seg-v3-s`)이 같은 {ok.sum()}장에서 —"
              f" 지느러미를 지킨 비율 {np.mean(eng[ok & (y[te]==1)]=='fin'):.1%}"
              f" · 아닌 것을 걸러낸 비율 {np.mean(eng[ok & (y[te]==0)]!='fin'):.1%}")
            w("  ** 이 날들은 `seg-v3` 가 **학습에 쓴 날**이라 그쪽에 유리한 자다.")

        if not o["score"]:
            return
        with torch.no_grad():
            pool_idx = np.array([pos[b] for b in pool if b in pos])
            pool_ids = np.array([b for b in pool if b in pos])
            p_pool = Fn.softmax(
                net(torch.from_numpy(X[pool_idx].astype(np.float32))), 1)[:, 1].numpy()
        order = np.argsort(p_pool)
        w(f"\n격자 {len(pool_ids):,}장 — **지느러미일 확률이 낮은 것부터** {o['top']}장")
        w("  (전부 `seg-v3` 가 `fin` 으로 통과시킨 것들이다)")
        for j in order[:o["top"]]:
            b = int(pool_ids[j])
            w(f"  상자 {b:>5} · {day.get(b,''):<11} p(fin)={p_pool[j]:.3f}   /photo/{b}")
        for t in (0.02, 0.05, 0.1, 0.3, 0.5):
            w(f"  문턱 {t:<4} 아래 {int((p_pool < t).sum()):>5,}장 "
              f"({(p_pool < t).mean():>5.1%})")

        if o["write"]:
            f = Path(o["score"]) / "items.json"
            d = json.loads(f.read_text())
            got = {int(b): float(v) for b, v in zip(pool_ids, p_pool)}
            n = 0
            for it in d["items"]:
                if it["id"] in got:
                    it["pfin"] = round(got[it["id"]], 4); n += 1
            f.write_text(json.dumps(d, ensure_ascii=False))
            w(f"\n{f} 에 `pfin` 을 {n:,}개 적었다 — 화면에서 "
              f"`지느러미가 아닐 듯` 으로 정렬된다.")
