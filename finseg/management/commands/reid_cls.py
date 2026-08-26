"""**아는 개체를 얼마나 맞히나** — 카탈로그를 클래스로 놓고 분류기를 배운다.

    python manage.py reid_cls --dir reid/v2 --emb emb-dinov2.npz
    python manage.py reid_cls --test-days 10 --epochs 400

## 왜 metric 이 아니라 분류인가

사람 re-ID 는 **처음 보는 사람**을 알아보는 문제라 학습 개체와 평가 개체를
겹치지 않게 나눈다. 감시 카메라가 날마다 모르는 사람을 만나기 때문이다.
**여기는 그 문제가 아니다** — 제주 남방큰돌고래는 개체군이 100~120이라
**개체 수의 천장이 개체군 크기**이고, 묻는 것은 "카탈로그의 몇 번인가" 다.

그래서 이 명령은 **닫힌 집합 분류**다. 실측으로 같은 자료에서 metric 쪽은
기준선을 못 넘었지만(`reid_probe --hold-individuals`) 닫힌 판에서는 얼린 특징
위의 선형 사영 하나가 28.0% → 43.0% 를 냈다. 그 자리를 제대로 파는 것이 여기다.

## 날로 가른다 — 조각으로 가르지 않는다

**같은 날 연속 프레임을 학습과 평가에 나눠 담으면 그날 조명을 외운 것이 성적이
된다.** 이 저장소가 이미 한 번 속은 자리다. 그래서 관찰일을 통째로 갈라
**배운 날과 재는 날이 안 겹친다.**

## 좌현과 우현은 따로 배운다

`reid.normalize` 가 한쪽 무리를 거울처럼 뒤집으므로 두 면의 특징은 서로 견줄 수
없다. 한 통에 넣고 배우면 모델이 개체가 아니라 **어느 쪽 면인가**를 배운다.
그래서 면마다 분류기를 따로 두고, 클래스는 개체다.

## 기준선을 같은 갈래에서 함께 낸다

**가장 닮은 조각의 개체를 고르는 것**(kNN)이 지금 화면이 하는 일이고 그것이
기준선이다. 배운 것과 안 배운 것을 다른 갈래에서 재면 무엇이 오른 것인지 알 수
없으므로, **같은 날 갈래·같은 질의·같은 후보**에 둘 다 댄다.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from finseg import reid


class Command(BaseCommand):
    help = "카탈로그를 클래스로 놓고 닫힌 집합 분류기를 배운다"

    def add_arguments(self, p):
        p.add_argument("--dir", default=str(settings.FIN_REID), help="기본은 **화면이 보는 격자**(`FIN_REID`) — 박아 두면 격자를 갈아 끼울 때마다 조용히 옛것을 잰다")
        p.add_argument("--emb", default="emb-dinov2.npz",
                       help="`--dir` 안의 임베딩 파일 이름")
        p.add_argument("--test-days", type=int, default=8)
        p.add_argument("--epochs", type=int, default=400)
        p.add_argument("--lr", type=float, default=0.02)
        p.add_argument("--wd", type=float, default=1e-3,
                       help="가중치 감쇠 — 개체당 조각이 적어 과적합이 쉽다")
        p.add_argument("--seed", type=int, default=20260825)
        p.add_argument("--fit-all", metavar="파일",
                       help="**재지 않고 배워서 저장한다** — 화면이 쓸 것이라 "
                            "날을 가르지 않고 아는 것을 전부 쓴다. 성적을 볼 때는 "
                            "이것을 쓰면 안 된다 (배운 것으로 배운 것을 잰다)")
        p.add_argument("--group", action="store_true",
                       help="**묶어서 묻는다** — 같은 날·같은 쪽의 한 개체 조각을 "
                            "한 묶음으로 보고 표를 모은다. 실제 화면이 하는 일이 "
                            "그것이고(`묶음 제안`), 판단 한 번이 여러 장을 덮는다")

    def _fit(self, X, y, n_cls, epochs, lr, wd, seed):
        """선형 한 층. **numpy 로 쓸 수 있게 가중치만 돌려준다** — 화면이 추론할
        때 torch 를 들이지 않아도 되게."""
        import torch
        import torch.nn.functional as Fn
        torch.manual_seed(seed)
        net = torch.nn.Linear(X.shape[1], n_cls)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
        xt = torch.from_numpy(X.astype(np.float32))
        yt = torch.tensor(y)
        for _ in range(epochs):
            opt.zero_grad()
            Fn.cross_entropy(net(xt), yt).backward()
            opt.step()
        return (net.weight.detach().numpy().astype(np.float32),
                net.bias.detach().numpy().astype(np.float32))

    def handle(self, **o):
        import torch
        import torch.nn.functional as Fn

        w = self.stdout.write
        root = Path(o["dir"])
        f = root / o["emb"]
        if not f.exists():
            raise CommandError(f"{f} 가 없다 — `reid_chips --emb-only` 로 만들 것")
        z = np.load(f)
        X = z["emb"] / np.maximum(np.linalg.norm(z["emb"], axis=1, keepdims=True), 1e-9)
        ids, fac = z["box_id"], z["facing"]
        items = json.loads((root / "items.json").read_text())["items"]
        if not (np.array([it["id"] for it in items]) == ids).all():
            raise CommandError("items.json 과 임베딩의 상자 순서가 다르다")
        day = np.array([it["day"] for it in items])

        cat = reid.catalog()
        of = {b: i for i, v in cat.items() for b in v}
        lab = np.array([of.get(int(b), -1) for b in ids])
        if (lab >= 0).sum() < 20:
            raise CommandError("정답이 너무 적다")

        # ---- 날로 가른다 ----------------------------------------------------
        days = sorted({d for d in day[lab >= 0]})
        cnt = Counter(day[lab >= 0].tolist())
        test_days = set(sorted(days, key=lambda d: -cnt[d])[:o["test_days"]])
        is_test = np.array([d in test_days for d in day])
        w(f"자료 {f} · {X.shape[1]}차원 · 정답 {int((lab>=0).sum()):,}조각 "
          f"· 개체 {len(cat)}")
        w(f"관찰일 {len(days)} → 재는 날 {len(test_days)} · 배우는 날 "
          f"{len(days)-len(test_days)}")

        if o["fit_all"]:
            return self._save(o, X, ids, fac, lab, cat, w)

        # **누수를 값으로 막는다.** 날로 가른다고 적어 두는 것과 실제로 안 겹치는
        # 것은 다르다 — 이 저장소가 "화면은 됐다는데 값이 안 닿는다" 로 겪은 종류다
        if set(day[is_test]) & set(day[~is_test]):
            raise CommandError("배우는 날과 재는 날이 겹친다 — 이 성적은 못 쓴다")

        torch.manual_seed(o["seed"])
        tot = {"cls1": 0, "cls5": 0, "knn1": 0, "knn5": 0, "n": 0}
        for side in ("left", "right"):
            m = (fac == side) & (lab >= 0)
            tr = np.where(m & ~is_test)[0]
            te = np.where(m & is_test)[0]
            classes = sorted(set(lab[tr].tolist()))
            # **배운 적 없는 개체는 닫힌 판의 질문이 아니다** — 그것은
            # `reid_probe --hold-individuals` 가 재는 열린 판의 몫이다
            te = np.array([i for i in te if lab[i] in classes])
            if len(tr) < 10 or not len(te) or len(classes) < 2:
                w(f"  {side}: 잴 것이 모자란다 (배움 {len(tr)} · 잼 {len(te)} "
                  f"· 개체 {len(classes)})")
                continue
            k = {c: i for i, c in enumerate(classes)}
            W, B = self._fit(X[tr], [k[int(v)] for v in lab[tr]], len(classes),
                             o["epochs"], o["lr"], o["wd"], o["seed"])
            logit = X[te] @ W.T + B

            # 기준선 — **배우는 날의 조각만** 후보로 둔다. 분류기가 본 것과
            # 같은 자료라야 공평하다
            knn = np.full((len(te), len(classes)), -np.inf)
            for c, ci in k.items():
                g = tr[lab[tr] == c]
                knn[:, ci] = (X[te] @ X[g].T).max(1)

            y = np.array([k[int(v)] for v in lab[te]])
            def hits(s, yy=None):
                yy = y if yy is None else yy
                r = np.argsort(-s, axis=1)
                return ((r[:, 0] == yy).sum(), (r[:, :5] == yy[:, None]).any(1).sum())
            if o["group"]:
                # **한 묶음은 (개체, 날, 쪽) 하나다** — 사람이 화면에서 묶는 단위와
                # 같다. 로짓은 평균하고 kNN 은 최고 닮음을 평균한다
                cells = {}
                for r, i in enumerate(te):
                    cells.setdefault((int(lab[i]), day[i]), []).append(r)
                idx = sorted(cells)
                logit = np.stack([logit[cells[c]].mean(0) for c in idx])
                knn = np.stack([knn[cells[c]].mean(0) for c in idx])
                y = np.array([k[c[0]] for c in idx])
                te = np.array(idx, dtype=object)
            c1, c5 = hits(logit)
            n1, n5 = hits(knn)
            w(f"  {side:<6} 개체 {len(classes):>2} · 배움 {len(tr):>3} · 잼 {len(te):>3}"
              f"   기준선(kNN) {n1/len(te):>6.1%} / {n5/len(te):>6.1%}"
              f"   **분류기 {c1/len(te):>6.1%} / {c5/len(te):>6.1%}**")
            tot["cls1"] += c1; tot["cls5"] += c5
            tot["knn1"] += n1; tot["knn5"] += n5; tot["n"] += len(te)

        if not tot["n"]:
            raise CommandError("잴 것이 없다 — `--test-days` 를 줄일 것")
        n = tot["n"]
        w(f"\n  {'':<14}{'top-1':>9}{'top-5':>9}   (질의 {n})")
        w(f"  {'기준선 (kNN)':<14}{tot['knn1']/n:>9.1%}{tot['knn5']/n:>9.1%}")
        w(f"  {'**분류기**':<14}{tot['cls1']/n:>9.1%}{tot['cls5']/n:>9.1%}")
        if tot["cls1"] <= tot["knn1"]:
            w("\n** 기준선을 못 넘었다 — 배운 것이 없다. 개체당 조각이 모자라거나"
              " 규제가 세거나 날 갈래가 너무 좁다.")

    def _save(self, o, X, ids, fac, lab, cat, w):
        """**화면이 쓸 분류기를 저장한다.** 날을 안 가르고 아는 것을 전부 쓴다 —
        성적을 재는 일과 실제로 쓰는 일은 다르다.

        `npz` 로 둔다. 추론은 `X @ W.T + b` 한 줄이라 화면 쪽에서 torch 를 들일
        이유가 없다.
        """
        out = {}
        for side in ("left", "right"):
            m = (fac == side) & (lab >= 0)
            idx = np.where(m)[0]
            classes = sorted(set(lab[idx].tolist()))
            if len(idx) < 10 or len(classes) < 2:
                w(f"  {side}: 배울 것이 모자란다 ({len(idx)}조각 · 개체 {len(classes)})")
                continue
            k = {c: i for i, c in enumerate(classes)}
            W_, B_ = self._fit(X[idx], [k[int(v)] for v in lab[idx]], len(classes),
                               o["epochs"], o["lr"], o["wd"], o["seed"])
            out[f"{side}_W"] = W_
            out[f"{side}_b"] = B_
            out[f"{side}_cls"] = np.array(classes, dtype=np.int64)
            w(f"  {side:<6} 개체 {len(classes):>2} · 조각 {len(idx):>3}")
        if not out:
            raise CommandError("배운 것이 없다")
        out["emb"] = np.array([o["emb"]])
        out["dim"] = np.array([X.shape[1]])
        out["n_labeled"] = np.array([int((lab >= 0).sum())])
        np.savez_compressed(o["fit_all"], **out)
        w(f"\n{o['fit_all']}  — 아는 것을 전부 써서 배웠다.")
        w("**이것으로 성적을 재지 말 것** — 배운 것으로 배운 것을 재게 된다.")
