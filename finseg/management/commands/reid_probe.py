"""**사람이 붙인 정답으로 배우면 나아지나** — 가장 싼 형태로 먼저 묻는다.

    python manage.py reid_probe --dir reid/v2
    python manage.py reid_probe --dir reid/v2 --test-days 6 --epochs 60

지금 표에 앉은 `ResNet18 · 조각` 은 **학습 없는 ImageNet 특징**이다. 개체를
가르라고 배운 적이 없으니 그것이 출발선이고, 이 명령이 묻는 것은 하나다 —
**우리 정답으로 배우면 그 출발선을 넘나.**

## 왜 선형 사영인가

조각 화소부터 다시 배우는 것(`reid_train`)은 GPU 로 몇 시간이다. 그 전에
**512차원 특징 위에 선형 사영 하나**를 배워 본다. 몇 초면 끝나고, 여기서
안 오르면 픽셀부터 배우는 값도 의심해야 한다. **싼 질문을 먼저 한다.**

## 날로 가른다 — 이것이 이 시험의 전부다

같은 날 짝으로 배우고 같은 날 짝으로 재면 **그날 조명을 배운 것**을 성적으로
읽는다. 이 저장소가 이미 한 번 속은 자리다(`HANDOFF` 의 "자가 조명을 재고
있었다"). 그래서

- 관찰일을 통째로 갈라 **배운 날과 재는 날이 안 겹친다**
- 양성쌍도 **날을 건너뛴 것만** 쓴다 — 연속 프레임은 쉬운 짝이라 배울 것이 없다
- 좌현·우현을 안 섞는다 (`reid.normalize` 가 한쪽을 거울처럼 뒤집는다)

## 닫힌 판과 열린 판을 갈라 낸다

재는 날의 개체가 배운 날에도 나왔으면 **닫힌 판**(그 개체를 이미 봤다),
안 나왔으면 **열린 판**(처음 보는 개체)이다. 실제로 쓰는 자리는 열린 판인데
닫힌 판이 훨씬 잘 나오므로, **섞어서 하나로 말하면 낙관 쪽으로 기운다.**

**기준선을 같은 질의에 대고 함께 낸다.** 배운 것과 안 배운 것을 다른 판에서
재면 무엇이 오른 것인지 알 수 없다.
"""
import json
from pathlib import Path

import numpy as np
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from finseg import reid
from finseg.models import Box


def train_pairs(lab, day, fac, is_test):
    """배울 양성쌍 — **재는 쪽은 한 짝에도 안 든다.**

    새는 길이 둘이라 둘 다 막는다. (1) 재는 개체·날의 조각이 짝에 끼는 것,
    (2) 같은 날 짝으로 배우는 것 — 그것은 개체가 아니라 그날 조명을 배운다.
    """
    tr = np.where((lab >= 0) & ~is_test)[0]
    return [(a, b) for i, a in enumerate(tr) for b in tr[i + 1:]
            if lab[a] == lab[b] and day[a] != day[b] and fac[a] == fac[b]]


class Command(BaseCommand):
    help = "정답 위에 선형 사영을 배워 ImageNet 기준선과 견준다"

    def add_arguments(self, p):
        p.add_argument("--dir", default=str(settings.FIN_REID), help="기본은 **화면이 보는 격자**(`FIN_REID`) — 박아 두면 격자를 갈아 끼울 때마다 조용히 옛것을 잰다")
        p.add_argument("--test-days", type=int, default=6,
                       help="재는 데 쓸 관찰일 수 (나머지가 배우는 날)")
        p.add_argument("--hold-individuals", type=int, default=0,
                       help="**개체를 통째로 빼서 잰다** — 날로 가르면 개체가 "
                            "거의 안 빠져 열린 판을 못 잰다 (실측 10개뿐이었다). "
                            "이것이 실제로 쓰는 자리다: 처음 보는 개체")
        p.add_argument("--dim", type=int, default=128)
        p.add_argument("--epochs", type=int, default=60)
        p.add_argument("--lr", type=float, default=3e-3)
        p.add_argument("--temp", type=float, default=0.07)
        p.add_argument("--loss", choices=("infonce", "triplet"), default="infonce",
                       help="`triplet` 은 **batch-hard** 다 — 무리 안에서 가장 먼 "
                            "양성과 가장 가까운 음성을 골라 쓴다. 무작위 음성을 "
                            "쓰는 InfoNCE 와 달리 자료가 적을 때 강한 것으로 "
                            "알려진 사람 re-ID 의 표준 기준선이다")
        p.add_argument("--margin", type=float, default=0.3,
                       help="triplet 의 여백. 0 이면 soft-margin(softplus)")
        p.add_argument("--train-individuals", type=int, default=0,
                       help="배우는 개체를 이 수만큼으로 **줄인다** (0이면 전부). "
                            "재는 개체는 그대로 두고 이것만 흔들면 **개체 수가 "
                            "성적을 얼마나 움직이나**가 곡선으로 나온다 — "
                            "'자료를 두 배로 늘리면 되나' 는 짐작할 게 아니다")
        p.add_argument("--pk", default="8x4",
                       help="triplet 무리 짜기 — 개체 P × 장수 K")
        p.add_argument("--seed", type=int, default=20260824)

    def handle(self, **o):
        import torch
        import torch.nn.functional as Fn

        w = self.stdout.write
        root = Path(o["dir"])
        z = np.load(root / "emb.npz")
        ids, fac = z["box_id"], z["facing"]
        X = z["emb"] / np.linalg.norm(z["emb"], axis=1, keepdims=True)
        items = json.loads((root / "items.json").read_text())["items"]
        day = np.array([it["day"] for it in items])
        if not (np.array([it["id"] for it in items]) == ids).all():
            raise CommandError("items.json 과 emb.npz 의 상자 순서가 다르다")

        cat = reid.catalog()
        of = {b: i for i, v in cat.items() for b in v}
        lab = np.array([of.get(int(b), -1) for b in ids])
        if (lab >= 0).sum() < 20:
            raise CommandError("정답이 너무 적다 — `/reid` 에서 더 넣은 뒤 잴 것")

        # ---- 개체로 가르거나, 날로 가르거나 ---------------------------------
        if o["hold_individuals"]:
            # **개체를 뺀다.** 날로 가르면 그 개체가 다른 날에도 나와 모델이
            # 이미 본 개체가 된다 — 그것은 닫힌 판이고, 카탈로그에 없는 새
            # 개체를 만나는 실제 자리를 못 잰다.
            #
            # 한정: 재는 개체의 **날**은 배우는 데도 나온다. 그날 조명을
            # 지름길로 쓸 여지가 남지만, 양성쌍이 날을 건너뛴 것뿐이라
            # 조명만으로는 짝이 안 맞는다
            by = sorted(cat.items(), key=lambda kv: -len(kv[1]))
            cross_ok = [i for i, v in by
                        if len({str(d) for d in
                                Box.objects.filter(id__in=v)
                                .values_list("image__obsdate", flat=True)}) >= 2]
            held = set(cross_ok[:o["hold_individuals"]])
            is_test = np.array([int(l) in held for l in lab])
            # **배우는 개체만 줄인다** — 재는 쪽은 그대로 둬야 곡선의 점들이
            # 같은 자로 잰 것이 된다. 씨앗을 고정해 다시 뽑을 수 있게 한다
            train_ids = [i for i in cat if i not in held]
            if o["train_individuals"] and o["train_individuals"] < len(train_ids):
                rs = np.random.default_rng(o["seed"])
                keep = set(rs.choice(train_ids, size=o["train_individuals"],
                                     replace=False).tolist())
                drop = np.array([int(l) not in keep and int(l) not in held
                                 and l >= 0 for l in lab])
                lab = np.where(drop, -1, lab)
                train_ids = sorted(keep)
            w(f"개체 {len(cat)} → **재는 개체 {len(held)}** (날을 건너뛴 것 중 큰 것부터)"
              f" · **배우는 개체 {len(train_ids)}**")
            tr = np.where((lab >= 0) & ~is_test)[0]
            seen_ids = {int(lab[i]) for i in tr}
        else:
            is_test = None
            seen_ids = None

        # ---- 날로 가른다 ---------------------------------------------------
        # **정답이 많은 날부터 재는 쪽에 넣는다** — 잴 것이 없는 날을 골라 두면
        # 시험이 헐거워진다. 고를 때 순서를 씨앗으로 흔들지 않는다(재현되어야 한다)
        if is_test is None:
            days = sorted({d for d in day[lab >= 0]})
            cnt = {d: int(((day == d) & (lab >= 0)).sum()) for d in days}
            test_days = set(sorted(days, key=lambda d: -cnt[d])[:o["test_days"]])
            is_test = np.array([d in test_days for d in day])
            w(f"관찰일 {len(days)} (정답이 있는 날) → 재는 날 {len(test_days)}"
              f" · 배우는 날 {len(days) - len(test_days)}")
            w(f"  재는 날: {', '.join(sorted(test_days))}")

        # ---- 배울 짝 — 배우는 쪽 안에서, 날을 건너뛴 것만 --------------------
        pairs = train_pairs(lab, day, fac, is_test)
        if len(pairs) < 20:
            raise CommandError(
                f"배울 짝이 {len(pairs)}개뿐이다 — 날을 건너뛴 같은 개체가 모자란다.\n"
                f"  `--test-days` 를 줄이거나 개체 분류를 더 할 것.")
        w(f"배울 양성쌍 {len(pairs):,} (날을 건너뛴 것만)")

        # ---- 선형 사영 -----------------------------------------------------
        torch.manual_seed(o["seed"])
        Xt = torch.from_numpy(X.astype(np.float32))
        W = torch.nn.Linear(X.shape[1], o["dim"], bias=False)
        opt = torch.optim.Adam(W.parameters(), lr=o["lr"])
        P = torch.tensor(np.array(pairs))
        rng = np.random.default_rng(o["seed"])

        # ---- triplet 은 무리를 다르게 짠다 — **(개체, 좌/우)가 한 반이다** ----
        # 좌현과 우현은 견줄 수 없으므로(거울로 뒤집혀 있다) 한 무리에 안 섞는다.
        # 양성은 **날을 건너뛴 것**을 먼저 고른다 — 같은 날 짝은 조명이 답을
        # 알려 준다
        groups = {}
        for i in np.where((lab >= 0) & ~is_test)[0]:
            groups.setdefault((int(lab[i]), str(fac[i])), []).append(i)
        groups = {k: v for k, v in groups.items()
                  if len(v) >= 2 and len({day[j] for j in v}) >= 2}
        if o["loss"] == "triplet":
            if len(groups) < 4:
                raise CommandError(
                    f"triplet 을 짤 반이 {len(groups)}개뿐이다 — (개체, 좌/우)로 "
                    f"갈라 날을 건너뛴 것만 센 값이다. 개체 분류를 더 할 것.")
            w(f"triplet 반 {len(groups)}개 ((개체, 좌/우) · 날을 건너뛴 것만)")

        def pk_batch():
            """P개 반 × K장. **한 무리는 한쪽 면만** — 음성도 같은 면이라야
            어려운 음성이 뜻을 갖는다."""
            pp, kk = (int(x) for x in o["pk"].split("x"))
            side = rng.choice(["left", "right"])
            ks = [k for k in groups if k[1] == side] or list(groups)
            take = rng.choice(len(ks), size=min(pp, len(ks)), replace=False)
            idx, y = [], []
            for t in take:
                g = groups[ks[t]]
                # 날이 고루 섞이게 — 같은 날만 K장 뽑으면 배울 것이 없다
                by_day = {}
                for j in g:
                    by_day.setdefault(day[j], []).append(j)
                pick = []
                for d in rng.permutation(list(by_day)):
                    pick.append(rng.choice(by_day[d]))
                    if len(pick) >= kk:
                        break
                while len(pick) < min(kk, len(g)):
                    c = rng.choice(g)
                    if c not in pick:
                        pick.append(c)
                idx += list(pick); y += [t] * len(pick)
            return np.array(idx), np.array(y)

        for ep in range(o["epochs"]):
            tot = nb = 0
            if o["loss"] == "triplet":
                steps = max(1, len(pairs) // 32)
                for _ in range(steps):
                    idx, y = pk_batch()
                    if len(set(y.tolist())) < 2:
                        continue
                    z = Fn.normalize(W(Xt[idx]), dim=1)
                    d = torch.cdist(z, z)
                    same = torch.from_numpy(y[:, None] == y[None, :])
                    eye = torch.eye(len(y), dtype=torch.bool)
                    # **가장 먼 양성 · 가장 가까운 음성** — 이것이 batch-hard 다
                    dp = torch.where(same & ~eye, d, torch.zeros_like(d)).max(1).values
                    dn = torch.where(~same, d, torch.full_like(d, 1e9)).min(1).values
                    loss = (Fn.softplus(dp - dn) if o["margin"] <= 0
                            else Fn.relu(o["margin"] + dp - dn)).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
                    tot += loss.item(); nb += 1
            else:
                idx = rng.permutation(len(P))
                for i in range(0, len(idx), 128):
                    bp = P[idx[i:i + 128]]
                    if len(bp) < 4:
                        continue
                    za = Fn.normalize(W(Xt[bp[:, 0]]), dim=1)
                    zb = Fn.normalize(W(Xt[bp[:, 1]]), dim=1)
                    z = torch.cat([za, zb])
                    sim = z @ z.T / o["temp"]
                    sim.fill_diagonal_(-1e9)
                    m = len(bp)
                    tgt = torch.cat([torch.arange(m, 2 * m), torch.arange(0, m)])
                    loss = Fn.cross_entropy(sim, tgt)
                    opt.zero_grad(); loss.backward(); opt.step()
                    tot += loss.item(); nb += 1
            if ep % 10 == 9 or ep == 0:
                w(f"  {ep + 1:3d}에폭  손실 {tot / max(nb, 1):.4f}")
        with torch.no_grad():
            Y = Fn.normalize(W(Xt), dim=1).numpy()

        # ---- 잰다 — 기준선과 배운 것을 **같은 질의에** --------------------
        seen = seen_ids if seen_ids is not None else {
            int(lab[i]) for i in np.where((lab >= 0) & ~is_test)[0]}
        q = [i for i in np.where((lab >= 0) & is_test)[0]
             if ((lab == lab[i]) & (day != day[i]) & (fac == fac[i])).sum()]
        if not q:
            raise CommandError("재는 날에 잴 수 있는 질의가 없다")
        closed = [i for i in q if int(lab[i]) in seen]
        openq = [i for i in q if int(lab[i]) not in seen]
        w(f"\n질의 {len(q)} — 닫힌 판 {len(closed)} (배운 날에 나온 개체) ·"
          f" 열린 판 {len(openq)} (처음 보는 개체)")

        def run(F, qs, gallery):
            t1 = first = 0
            got = []
            for i in qs:
                ok = gallery.copy(); ok[i] = False
                ok &= (day != day[i]) & (fac == fac[i])
                if not (ok & (lab == lab[i])).any():
                    continue
                s = np.where(ok, F @ F[i], -np.inf)
                order = np.argsort(-s)
                order = order[ok[order]]
                hit = (lab[order] == lab[i])
                got.append((bool(hit[0]), int(np.argmax(hit)) + 1))
            if not got:
                return None
            return (np.mean([g[0] for g in got]),
                    np.median([g[1] for g in got]), len(got))

        lab_gal = lab >= 0
        all_gal = np.ones(len(ids), bool)
        w("")
        for name, qs in (("전체", q), ("닫힌 판", closed), ("열린 판", openq)):
            if not qs:
                continue
            w(f"{name} — 질의 {len(qs)}")
            w(f"  {'자':<22}{'좁은 후보 1등':>13}{'첫 정답':>8}"
              f"{'격자 전체 1등':>14}{'첫 정답':>8}")
            learned = f"+ 선형 사영 ({o['loss']})"
            for tag, F in (("ImageNet (기준선)", X), (learned, Y)):
                a = run(F, qs, lab_gal)
                b = run(F, qs, all_gal)
                if a is None or b is None:
                    continue
                w(f"  {tag:<22}{a[0]:>12.1%}{a[1]:>7.0f}위"
                  f"{b[0]:>13.1%}{b[1]:>7.0f}위")
            w("")
        w("**열린 판이 실제로 쓰는 자리다** — 닫힌 판은 그 개체를 이미 본 판이라"
          " 낙관 쪽으로 기운다.")
