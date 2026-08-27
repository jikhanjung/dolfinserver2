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


def split_days(days, cnt, k):
    """관찰일을 k벌로 나눈다. **큰 날부터 번갈아 담는다.**

    무작위로 담으면 33장짜리 날 여럿이 한 폴드에 몰려 그 폴드만 잼이 두꺼워진다.
    번갈아 담으면 폴드마다 크기가 비슷해지고, **씨앗을 안 타므로** 두 자를
    견줄 때 같은 문제를 푼다.

    번갈아 담기(`j % k`)로는 덜 고르다 — 큰 날 열 개를 5벌로 번갈아 담으면
    가장 무거운 폴드와 가벼운 폴드가 조각 12장 차이가 났다. **그때그때 가장
    가벼운 폴드에 담으면** 4장으로 준다. 폴드마다 잼이 비슷해야 폴드별 성적을
    나란히 읽을 수 있다.

    빈 폴드를 안 만든다 — 부르는 쪽이 `k <= len(days)` 를 이미 막는다.
    """
    fold = [[] for _ in range(k)]
    load = [0] * k
    for d in sorted(days, key=lambda d: (-cnt[d], d)):
        j = min(range(k), key=lambda i: (load[i], i))
        fold[j].append(d)
        load[j] += cnt[d]
    return fold


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
                       help="**L2 — 손잡이 값의 제곱에 벌점.** 큰 값을 세게 누르되 "
                            "**0 으로는 안 만든다.** 개체당 조각이 적어 과적합이 "
                            "쉽다. 384차원에서 0~1e-2 를 훑었을 때는 성적이 한 "
                            "자리도 안 움직였는데, 그때는 축을 안 자른 상태였다")
        p.add_argument("--l1", type=float, default=0.0,
                       help="**L1 — 손잡이 값의 절대값에 벌점.** 작은 값도 같은 "
                            "힘으로 밀어 **쓸모없는 축을 아예 0 으로 만든다.** "
                            "PCA 가 축을 잘라 이겼으니(384 → 256 에서 +2.4%p) "
                            "**라벨을 보고 자르는 쪽**도 물어볼 값이 있다. "
                            "가중치에만 걸고 편향에는 안 건다")
        p.add_argument("--no-bias-decay", action="store_true",
                       help="**편향을 감쇠에서 뺀다.** 관례인데 이 저장소는 지금 "
                            "`net.parameters()` 를 통째로 넘겨 편향에도 건다. "
                            "편향은 개체마다 하나씩(30개)이라 작을 것 같지만 "
                            "**재 본 적이 없다**")
        p.add_argument("--seed", type=int, default=20260825)
        p.add_argument("--fit-all", metavar="파일",
                       help="**재지 않고 배워서 저장한다** — 화면이 쓸 것이라 "
                            "날을 가르지 않고 아는 것을 전부 쓴다. 성적을 볼 때는 "
                            "이것을 쓰면 안 된다 (배운 것으로 배운 것을 잰다)")
        p.add_argument("--folds", type=int, default=0, metavar="K",
                       help="**재는 날을 고정하지 말고 돌린다.** 관찰일을 K벌로 "
                            "나눠 한 벌씩 빼면서 K번 배우고 잰다 — 모든 날이 한 "
                            "번씩 잼 쪽에 서므로 질의가 137 → 510 이 되고 눈금이 "
                            "0.73%p → 0.20%p 가 된다. 0 이면 `--test-days` 대로 "
                            "한 번만 (기본)")
        p.add_argument("--seeds", type=int, default=1, metavar="N",
                       help="씨앗을 N 번 흔들어 폭을 함께 낸다. **폴드 배정은 "
                            "안 흔든다** — 두 자를 견줄 때 같은 문제를 풀어야 "
                            "차이가 자 때문인지 문제 때문인지 갈린다")
        p.add_argument("--group", action="store_true",
                       help="**묶어서 묻는다** — 같은 날·같은 쪽의 한 개체 조각을 "
                            "한 묶음으로 보고 표를 모은다. 실제 화면이 하는 일이 "
                            "그것이고(`묶음 제안`), 판단 한 번이 여러 장을 덮는다")

    def _fit(self, X, y, n_cls, epochs, lr, wd, seed, l1=0.0, no_bias_decay=False):
        """선형 한 층. **numpy 로 쓸 수 있게 가중치만 돌려준다** — 화면이 추론할
        때 torch 를 들이지 않아도 되게.

        규제가 둘이다. **L2**(`wd`)는 `AdamW` 가 가중치를 매 걸음 직접 줄이는
        쪽으로 건다(decoupled — Adam 계열에서 손실에 더하는 옛 방식이 의도대로
        안 들어서 갈라 나온 것이다). **L1**(`l1`)은 손실에 더한다.
        """
        import torch
        import torch.nn.functional as Fn
        torch.manual_seed(seed)
        net = torch.nn.Linear(X.shape[1], n_cls)
        if no_bias_decay:
            groups = [{"params": [net.weight], "weight_decay": wd},
                      {"params": [net.bias], "weight_decay": 0.0}]
        else:
            groups = [{"params": list(net.parameters()), "weight_decay": wd}]
        opt = torch.optim.AdamW(groups, lr=lr)
        xt = torch.from_numpy(X.astype(np.float32))
        yt = torch.tensor(y)
        for _ in range(epochs):
            opt.zero_grad()
            loss = Fn.cross_entropy(net(xt), yt)
            # **가중치에만 건다** — 편향은 개체마다 하나뿐이라 죽일 축이 없다
            if l1:
                loss = loss + l1 * net.weight.abs().sum()
            loss.backward()
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

        if o["folds"]:
            return self._folds(o, X, lab, fac, day, days, cnt, w)

        tot = self._score(o, X, lab, fac, day, is_test, o["seed"], w)
        self._report(tot, w)

    def _score(self, o, X, lab, fac, day, is_test, seed, w):
        """한 갈래를 배우고 잰다. **자를 한 곳에 둔다** — 고정 갈래와 폴드가
        다른 식으로 채점하면 둘을 견줄 수 없다."""
        import numpy as np
        import torch

        torch.manual_seed(seed)
        tot = {"cls1": 0, "cls5": 0, "cls10": 0,
               "knn1": 0, "knn5": 0, "knn10": 0, "n": 0}
        for side in ("left", "right"):
            m = (fac == side) & (lab >= 0)
            tr = np.where(m & ~is_test)[0]
            te = np.where(m & is_test)[0]
            classes = sorted(set(lab[tr].tolist()))
            # **배운 적 없는 개체는 닫힌 판의 질문이 아니다** — 그것은
            # `reid_probe --hold-individuals` 가 재는 열린 판의 몫이다
            te = np.array([i for i in te if lab[i] in classes])
            if len(tr) < 10 or not len(te) or len(classes) < 2:
                if w:
                    w(f"  {side}: 잴 것이 모자란다 (배움 {len(tr)} · 잼 {len(te)} "
                      f"· 개체 {len(classes)})")
                continue
            k = {c: i for i, c in enumerate(classes)}
            W, B = self._fit(X[tr], [k[int(v)] for v in lab[tr]], len(classes),
                             o["epochs"], o["lr"], o["wd"], seed,
                             o["l1"], o["no_bias_decay"])
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
                return ((r[:, 0] == yy).sum(),
                        (r[:, :5] == yy[:, None]).any(1).sum(),
                        (r[:, :10] == yy[:, None]).any(1).sum())
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
            c1, c5, c10 = hits(logit)
            n1, n5, n10 = hits(knn)
            if w:
                w(f"  {side:<6} 개체 {len(classes):>2} · 배움 {len(tr):>3} · 잼 {len(te):>3}"
                  f"   기준선(kNN) {n1/len(te):>6.1%} / {n5/len(te):>6.1%}"
                  f"   **분류기 {c1/len(te):>6.1%} / {c5/len(te):>6.1%}**")
            tot["cls1"] += c1; tot["cls5"] += c5; tot["cls10"] += c10
            tot["knn1"] += n1; tot["knn5"] += n5; tot["knn10"] += n10
            tot["n"] += len(te)
            # **top-10 은 개체가 10마리를 넘어야 뜻이 있다** — 그 아래면 무조건
            # 100% 다. 지금은 면마다 27·31마리라 넘는다
            tot["cls10_ok"] = tot.get("cls10_ok", True) and len(classes) > 10
        return tot

    def _report(self, tot, w):
        if not tot["n"]:
            raise CommandError("잴 것이 없다 — `--test-days` 를 줄일 것")
        n = tot["n"]
        w(f"\n  {'':<14}{'top-1':>9}{'top-5':>9}{'top-10':>9}   (질의 {n})")
        w(f"  {'기준선 (kNN)':<14}{tot['knn1']/n:>9.1%}{tot['knn5']/n:>9.1%}"
          f"{tot['knn10']/n:>9.1%}")
        w(f"  {'**분류기**':<14}{tot['cls1']/n:>9.1%}{tot['cls5']/n:>9.1%}"
          f"{tot['cls10']/n:>9.1%}")
        w(self._top10_note(tot))
        if tot["cls1"] <= tot["knn1"]:
            w("\n** 기준선을 못 넘었다 — 배운 것이 없다. 개체당 조각이 모자라거나"
              " 규제가 세거나 날 갈래가 너무 좁다.")

    def _folds(self, o, X, lab, fac, day, days, cnt, w):
        """**재는 날을 돌린다** — 관찰일 K벌, 한 벌씩 빼면서 K번.

        ## 왜 조각이 아니라 날로 나누나

        `--test-days` 갈래와 같은 이유다. 같은 날 사진은 몇 초 사이에 연달아
        찍은 것이라 거의 같은 그림이고(정답 조각의 62%가 같은 날 3장 이상인
        묶음에 들어 있다), 그중 몇 장이 배움에 몇 장이 잼에 들어가면 모델은
        개체가 아니라 **그 날을 외운 것**이 된다.

        ## 왜 돌리나

        고정 갈래는 **가장 큰 8일**만 재고 나머지 41일 361조각은 성적에 한 번도
        안 들어간다. 질의가 137이면 **한 문제가 0.73%p** 라, 앞으로 잴 것들
        (백본 키우기·조각 해상도·2층 MLP·ArcFace)이 전부 몇 %p 짜리라 하나도
        못 가른다. 돌리면 510문항이 되어 눈금이 0.20%p 다.

        ## 폴드 배정은 씨앗을 안 탄다

        큰 날부터 번갈아 담아 폴드 크기를 맞추고, **그 배정을 고정한다.**
        두 자를 견줄 때 **같은 문제를 풀어야** 차이가 자 때문인지 문제 때문인지
        갈린다 — 씨앗은 머리의 초기값만 흔든다.

        ## 못 묻는 조각이 남는다

        관찰일이 하루뿐인 개체(8마리·21조각)는 어느 폴드에서도 질의가 못 된다 —
        그 날이 잼 쪽이면 배운 적이 없고 배움 쪽이면 물을 것이 없다.
        `/dataset` 의 `2일 이상` 축이 재는 것이 바로 이것이다.
        """
        import numpy as np

        K = o["folds"]
        if K < 2:
            raise CommandError("`--folds` 는 2 이상이라야 한다")
        if K > len(days):
            raise CommandError(f"관찰일이 {len(days)}일뿐이라 {K}벌로 못 나눈다")
        fold = split_days(days, cnt, K)

        w(f"**재는 날을 {K}벌로 돌린다** · 씨앗 {o['seeds']}개 "
          f"— 관찰일 {len(days)}일이 한 번씩 잼 쪽에 선다")
        runs = []
        for si in range(o["seeds"]):
            seed = o["seed"] + si
            tot = {"cls1": 0, "cls5": 0, "cls10": 0,
                   "knn1": 0, "knn5": 0, "knn10": 0, "n": 0}
            first = si == 0
            if first:
                w(f"\n  {'폴드':<5}{'재는날':>6}{'잼':>6}"
                  f"{'기준선 1/5/10':>22}{'분류기 1/5/10':>22}")
            for fi, ds in enumerate(fold):
                is_test = np.array([d in set(ds) for d in day])
                t = self._score(o, X, lab, fac, day, is_test, seed, None)
                for k2 in tot:
                    tot[k2] += t[k2]
                if first:
                    n = t["n"] or 1
                    w(f"  {fi + 1:<5}{len(ds):>6}{t['n']:>6}"
                      f"{t['knn1'] / n:>7.1%}{t['knn5'] / n:>7.1%}{t['knn10'] / n:>7.1%}"
                      f"{t['cls1'] / n:>8.1%}{t['cls5'] / n:>7.1%}{t['cls10'] / n:>7.1%}")
            runs.append(tot)

        n = runs[0]["n"]
        if not n:
            raise CommandError("잴 것이 없다")
        r0 = runs[0]
        w(f"\n  {'합':<5}{len(days):>6}{n:>6}"
          f"{r0['knn1'] / n:>7.1%}{r0['knn5'] / n:>7.1%}{r0['knn10'] / n:>7.1%}"
          f"{r0['cls1'] / n:>8.1%}{r0['cls5'] / n:>7.1%}{r0['cls10'] / n:>7.1%}")
        w(self._top10_note(r0))
        w(f"\n  **질의 {n}** — 한 문제가 {100 / n:.2f}%p "
          f"(고정 갈래 137문항이면 {100 / 137:.2f}%p 였다)")
        if o["seeds"] > 1:
            for key, nm in (("cls1", "분류기 top-1"), ("cls5", "분류기 top-5"),
                            ("cls10", "분류기 top-10")):
                v = np.array([r[key] / r["n"] for r in runs]) * 100
                w(f"  {nm:<12} 씨앗별 {' '.join(f'{x:.1f}' for x in v)}"
                  f"  →  **{v.mean():.1f} ± {v.std(ddof=0):.1f}**")
            w("  **폭이 씨앗 때문에 생긴 것이다** — 자료도 폴드도 같았다. "
              "두 자의 차이가 이 폭보다 작으면 그것은 아직 차이가 아니다.")

    def _top10_note(self, tot):
        """**top-10 을 다른 연구와 그대로 견주지 말 것.**

        finFindR 82% · CurvRank 83% · Kim et al. 84.8% 가 전부 top-10 이라 같은
        줄에 세우고 싶어지는데, **개체 수가 다르면 같은 자가 아니다.** 저쪽
        reference 는 79개체이고 우리는 42다 — 아무렇게나 찍어도 42마리에서
        열을 고르면 23.8%가 맞는데 79마리에서는 12.7%다.

        그래서 여기에 **찍어서 맞을 확률**을 함께 적는다. 숫자 옆에 그 값이
        없으면 다음에 보는 사람이 그대로 옮겨 적는다.
        """
        return ("  **top-10 을 남과 견줄 때는 개체 수를 함께 적을 것** — "
                "우리 42마리에서 찍어 맞을 확률이 23.8%다 "
                "(Kim et al. 은 79마리라 12.7%).")

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
                               o["epochs"], o["lr"], o["wd"], o["seed"],
                               o["l1"], o["no_bias_decay"])
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
