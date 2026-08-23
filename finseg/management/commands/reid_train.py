"""조각 위에서 **개체를 가르는 특징을 배운다** — 개체명 없이. [GPU]

    python manage.py reid_train --epochs 40
    python manage.py reid_train --epochs 40 --out reid/emb-v1

## 정답 없이 어떻게 배우나

개체명이 옛 DB 에 166종·상자 435개뿐이고 **상자 5개 이상인 개체가 다섯**이다.
지도학습을 할 수 없다.

대신 **연속 프레임을 양성쌍으로 쓴다.** 같은 날 파일번호가 몇 안 떨어진 두
사진은 같은 개체일 가능성이 높다 — 몇 초 사이에 찍힌 것이다. 사람이 아무것도
안 붙여도 자료 자체가 그 짝을 알려 준다.

**완벽한 정답은 아니다.** 무리가 나란히 있으면 연속 프레임에 다른 개체가 찍힌다.
그래서 간격을 좁힐수록 정확하고 넓힐수록 많다 — `--gap` 이 그 거래다.

    간격 ≤1 → 85쌍 · ≤3 → 199 · ≤10 → 498 · ≤20 → 852

## 파일번호만으로는 모자라다 — **그림이 얼마나 바뀌었나**를 함께 본다

번호가 붙어 있어도 배가 돌거나 돌고래가 지나가면 장면이 통째로 바뀐다. 실측으로
간격 ≤3 짝의 그림 차이 중앙값이 **0.724** 였는데 **아무 짝이나가 0.977** 이다 —
절반 이상이 사실상 다른 장면이었다.

사진을 64×64 회색조로 줄이고 밝기를 지운 뒤(노출 차이를 없앤다) 화소 차이의
평균을 본다. 문턱을 좁히면 짝이 줄지만 **그중 진짜 같은 개체인 비율이 오른다**:

    제한 없음  짝 199  1등 12.1% · 10등 안 37.2% · 중앙값 24위
    ≤0.70     짝  92  1등 14.1% · 10등 안 42.4% · 중앙값 16위
    **≤0.50   짝  45  1등 13.3% · 10등 안 51.1% · 중앙값 10위**

그래서 **평가는 엄격하게, 학습은 넉넉하게** 잡는다(`--eval-diff` < `--diff`).
자가 흔들리면 배운 것이 있는지 알 수 없고, 학습 쪽은 대조학습이 잡음을 어느
정도 견딘다.

서명을 만드는 데 NAS 에서 599장에 5분이 걸렸다 — **캐시해 둔다.**

## 학습과 평가를 날로 가른다

**같은 짝으로 배우고 그 짝으로 재면 안 된다.** 이 저장소가 `val_date` 로 계속
지켜 온 규율이 여기서도 그대로 필요하다 — `val` 만 좋으면 그 날의 바다 상태를
외운 것이다(`CLAUDE.md`).

그래서 **관찰일 몇 개를 통째로 뺀다.** 뺀 날의 연속 프레임 짝으로만 잰다.
날을 안 가르고 짝만 가르면, 같은 날의 다른 짝을 통해 그 개체를 이미 봤을 수
있다.

## `fliplr` 을 쓰지 않는다

같은 개체라도 좌현과 우현은 다른 그림이다. 뒤집기 증강을 넣으면 두 면을 같은
것으로 배운다 (`TODOs` 의 re-ID 항목). 회전·크기·밝기는 넣되 **좌우 뒤집기는
절대 안 넣는다.**

## 무엇과 견주나

학습 없이 ImageNet 특징만으로 잰 값이 기준선이다.

    뒷날 곡선            1등  3.0% · 10등 안  9.5%
    조각 화소 그대로      1등  2.5% · 10등 안  6.5%
    ResNet18 (학습 없이)  1등 12.1% · 10등 안 37.2%

**이보다 못하면 배운 것이 없는 것이다.**
"""
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from django.core.management.base import BaseCommand, CommandError

from finseg import runs
from finseg.models import Box

CHIPS = "reid/v1/chips.npz"


def frame_no(path):
    m = re.search(r"(\d+)\.[A-Za-z]+$", path)
    return int(m.group(1)) if m else -1


class Command(BaseCommand):
    help = "연속 프레임을 양성쌍으로 삼아 조각 임베딩을 배운다 [GPU]"

    def add_arguments(self, p):
        p.add_argument("--chips", default=CHIPS)
        p.add_argument("--gap", type=int, default=10,
                       help="학습 양성쌍의 파일번호 간격 — 좁을수록 정확하고 "
                            "넓을수록 많다")
        p.add_argument("--eval-gap", type=int, default=3,
                       help="평가 양성쌍 — 좁게 잡아 정확도를 지킨다")
        p.add_argument("--diff", type=float, default=0.75,
                       help="학습 양성쌍이 허용하는 **전체 그림 차이**")
        p.add_argument("--eval-diff", type=float, default=0.5,
                       help="평가 양성쌍 — 좁게 잡는다. 자가 흔들리면 안 된다")
        p.add_argument("--sigs", default="reid/v1/sigs.npz",
                       help="사진 서명 캐시 (없으면 만든다 · NAS 에서 5분)")
        p.add_argument("--val-dates", nargs="*", help="통째로 뺄 관찰일")
        p.add_argument("--val-frac", type=float, default=0.25,
                       help="--val-dates 를 안 주면 이 비율만큼 날을 뽑는다")
        p.add_argument("--epochs", type=int, default=40)
        p.add_argument("--batch", type=int, default=48)
        p.add_argument("--lr", type=float, default=3e-4)
        p.add_argument("--temp", type=float, default=0.1)
        p.add_argument("--dim", type=int, default=128)
        p.add_argument("--seed", type=int, default=20260824)
        p.add_argument("--device", default="cuda")
        p.add_argument("--out")

    # ---- 자료 -----------------------------------------------------------
    def _load(self, o):
        f = Path(o["chips"])
        if not f.exists():
            raise CommandError(
                f"{f} 가 없다. 먼저 조각을 만들 것 — `reid_cluster --out reid/v1`")
        d = np.load(f, allow_pickle=True)
        ids, fac, chips = d["box_id"], d["facing"], d["chip"]
        meta = {b.id: (str(b.image.obsdate), b.image.path)
                for b in Box.objects.select_related("image")
                .filter(id__in=ids.tolist())}
        day = np.array([meta[i][0] for i in ids])
        num = np.array([frame_no(meta[i][1]) for i in ids])
        path = np.array([meta[i][1] for i in ids])
        return ids, fac, chips, day, num, path

    def _sigs(self, paths, cache):
        """사진마다 64×64 서명. **밝기를 지운다** — 노출 차이가 장면 변화로
        읽히면 안 된다. NAS 에서 읽는 것이 비싸므로 캐시한다."""
        from django.conf import settings
        from PIL import Image

        f = Path(cache)
        have = {}
        if f.exists():
            d = np.load(f, allow_pickle=True)
            have = {p: s for p, s in zip(d["path"], d["sig"])}
        want = [p for p in paths if p not in have]
        if want:
            self.stdout.write(f"  사진 서명 {len(want):,} 장을 만든다"
                              f" (NAS 라 느리다)…")
            for p in want:
                src = settings.FIN_PHOTOS / p
                if not src.exists():
                    continue
                im = Image.open(src)
                im.draft("L", (64, 64))          # 1/8 디코드 — 훨씬 빠르다
                a = np.asarray(im.convert("L").resize((64, 64)), np.float32) / 255
                have[p] = (a - a.mean()) / (a.std() + 1e-6)
            f.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(f, path=np.array(list(have)),
                                sig=np.stack(list(have.values())))
        return have

    def _pairs(self, fac, day, num, keep, gap, path, sigs, max_diff):
        """`keep` 안에서 (같은 날 · 같은 쪽 · 간격 이내 · **그림이 안 바뀐**) 짝."""
        out = []
        idx = np.where(keep)[0]
        by = defaultdict(list)
        for i in idx:
            by[(day[i], fac[i])].append(i)
        for group in by.values():
            for a in range(len(group)):
                for b in range(a + 1, len(group)):
                    i, j = group[a], group[b]
                    if not 0 < abs(num[i] - num[j]) <= gap:
                        continue
                    pa, pb = path[i], path[j]
                    if pa in sigs and pb in sigs:
                        if np.abs(sigs[pa] - sigs[pb]).mean() > max_diff:
                            continue
                    out.append((int(i), int(j)))
        return out

    # ---- 재는 자 --------------------------------------------------------
    def _rank(self, feats, fac, pairs):
        """짝이 서로의 최근접 몇 위인가 → (1등 %, 10등 안 %, 중앙값)."""
        F = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
        ranks = []
        for a, b in pairs:
            cand = np.where(fac == fac[a])[0]
            cand = cand[cand != a]
            d = 1 - F[cand] @ F[a]
            order = cand[np.argsort(d)]
            ranks.append(int(np.where(order == b)[0][0]) + 1)
        r = np.array(ranks)
        return 100 * (r == 1).mean(), 100 * (r <= 10).mean(), float(np.median(r))

    def handle(self, **o):
        import torch
        import torch.nn.functional as Fn
        import torchvision as tv

        w = self.stdout.write
        rng = np.random.default_rng(o["seed"])
        torch.manual_seed(o["seed"])

        ids, fac, chips, day, num, path = self._load(o)
        days = sorted(set(day.tolist()))
        sigs = self._sigs(sorted(set(path.tolist())), o["sigs"])

        # **날을 통째로 가른다.** 짝만 가르면 같은 날의 다른 짝을 통해 그
        # 개체를 이미 본 셈이 된다
        if o["val_dates"]:
            val_days = set(o["val_dates"])
            unknown = val_days - set(days)
            if unknown:
                raise CommandError(f"그런 관찰일이 없다: {sorted(unknown)}")
        else:
            k = max(1, int(round(len(days) * o["val_frac"])))
            val_days = set(rng.choice(days, k, replace=False).tolist())
        is_val = np.array([d in val_days for d in day])

        tr_pairs = self._pairs(fac, day, num, ~is_val, o["gap"],
                               path, sigs, o["diff"])
        va_pairs = self._pairs(fac, day, num, is_val, o["eval_gap"],
                               path, sigs, o["eval_diff"])
        w(f"조각 {len(ids):,} · 관찰일 {len(days)}")
        w(f"  뺀 날 {len(val_days)}: {sorted(val_days)}")
        w(f"  학습 조각 {int((~is_val).sum()):,} · 양성쌍 {len(tr_pairs):,}"
          f" (간격 ≤{o['gap']} · 그림차 ≤{o['diff']})")
        w(f"  평가 조각 {int(is_val.sum()):,} · 양성쌍 {len(va_pairs):,}"
          f" (간격 ≤{o['eval_gap']} · 그림차 ≤{o['eval_diff']})")
        if not tr_pairs or not va_pairs:
            raise CommandError(
                "짝이 모자란다. `--gap` 을 넓히거나 `--val-frac` 을 바꿀 것.")

        dev = torch.device(o["device"] if torch.cuda.is_available() else "cpu")
        X = torch.from_numpy(chips).float()          # (N, 128, 128)

        def backbone():
            m = tv.models.resnet18(
                weights=tv.models.ResNet18_Weights.IMAGENET1K_V1)
            m.fc = torch.nn.Identity()
            return m

        # 기준선 — 학습 전 ImageNet 특징
        net = backbone().to(dev).eval()
        head = torch.nn.Sequential(
            torch.nn.Linear(512, 512), torch.nn.ReLU(),
            torch.nn.Linear(512, o["dim"])).to(dev)
        mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

        def features(model, aug=False):
            outs = []
            for i in range(0, len(X), 64):
                x = X[i:i + 64].to(dev).unsqueeze(1)
                if aug:
                    x = augment(x, rng)
                x = x.repeat(1, 3, 1, 1)
                x = Fn.interpolate(x, size=224, mode="bilinear",
                                   align_corners=False)
                with torch.no_grad():
                    outs.append(model((x - mean) / std).cpu().numpy())
            return np.concatenate(outs)

        def augment(x, rng):
            """**좌우 뒤집기는 절대 안 넣는다** — 좌현과 우현이 하나가 된다.

            회전·크기·평행이동은 촬영 각도의 흔들림을 흉내 내고, 밝기·대비는
            물빛과 노출을 흉내 낸다. 지느러미 안쪽 무늬가 개체를 가르는 단서라
            (실루엣만 쓰면 12.1% → 2.5%로 무너졌다) 그 무늬를 지우지 않는 선에서.
            """
            n = x.shape[0]
            ang = torch.tensor(rng.uniform(-12, 12, n), dtype=torch.float32,
                               device=x.device) * np.pi / 180
            sc = torch.tensor(rng.uniform(0.88, 1.12, n), dtype=torch.float32,
                              device=x.device)
            tx = torch.tensor(rng.uniform(-.06, .06, n), dtype=torch.float32,
                              device=x.device)
            ty = torch.tensor(rng.uniform(-.06, .06, n), dtype=torch.float32,
                              device=x.device)
            c, s = torch.cos(ang) / sc, torch.sin(ang) / sc
            theta = torch.stack([torch.stack([c, -s, tx], 1),
                                 torch.stack([s, c, ty], 1)], 1)
            grid = Fn.affine_grid(theta, x.shape, align_corners=False)
            x = Fn.grid_sample(x, grid, align_corners=False)
            b = torch.tensor(rng.uniform(-.12, .12, n), dtype=torch.float32,
                             device=x.device).view(-1, 1, 1, 1)
            g = torch.tensor(rng.uniform(0.85, 1.2, n), dtype=torch.float32,
                             device=x.device).view(-1, 1, 1, 1)
            return ((x + b) * g).clamp(0, 1)

        base = features(net)
        t1, t10, med = self._rank(base, fac, va_pairs)
        w(f"\n기준선 (ImageNet 그대로): 1등 {t1:.1f}% · 10등 안 {t10:.1f}%"
          f" · 중앙값 {med:.0f}위")

        # ---- 학습 -------------------------------------------------------
        net = backbone().to(dev)
        opt = torch.optim.AdamW(
            list(net.parameters()) + list(head.parameters()),
            lr=o["lr"], weight_decay=1e-4)
        pairs = np.array(tr_pairs)
        run = runs.start("train", model="resnet18", params={
            "task": "reid", "gap": o["gap"], "eval_gap": o["eval_gap"],
            "val_dates": sorted(val_days), "epochs": o["epochs"],
            "batch": o["batch"], "lr": o["lr"], "temp": o["temp"],
            "dim": o["dim"], "n_chips": int(len(ids)),
            "diff": o["diff"], "eval_diff": o["eval_diff"],
            "n_train_pairs": len(tr_pairs), "n_val_pairs": len(va_pairs),
            "baseline_top1": float(t1)})
        best = (t1, t10, med, 0)
        for ep in range(1, o["epochs"] + 1):
            net.train(); head.train()
            rng.shuffle(pairs)
            tot = nb = 0
            for i in range(0, len(pairs), o["batch"]):
                bp = pairs[i:i + o["batch"]]
                if len(bp) < 4:
                    continue
                idx = np.concatenate([bp[:, 0], bp[:, 1]])
                x = X[idx].to(dev).unsqueeze(1)
                x = augment(x, rng).repeat(1, 3, 1, 1)
                x = Fn.interpolate(x, size=224, mode="bilinear",
                                   align_corners=False)
                z = Fn.normalize(head(net((x - mean) / std)), dim=1)
                # InfoNCE — 짝끼리 가깝게, 나머지는 멀게
                sim = z @ z.T / o["temp"]
                m = len(bp)
                sim.fill_diagonal_(-1e9)
                target = torch.cat([torch.arange(m, 2 * m),
                                    torch.arange(0, m)]).to(dev)
                loss = Fn.cross_entropy(sim, target)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss); nb += 1
            if ep % 2 or ep == o["epochs"]:
                net.eval()
                f = features(net)
                a1, a10, amed = self._rank(f, fac, va_pairs)
                mark = ""
                if a1 > best[0]:
                    best = (a1, a10, amed, ep); mark = " ←"
                    if o["out"]:
                        Path(o["out"]).mkdir(parents=True, exist_ok=True)
                        torch.save(net.state_dict(),
                                   Path(o["out"]) / "backbone.pt")
                        np.savez_compressed(Path(o["out"]) / "emb.npz",
                                            box_id=ids, facing=fac, emb=f)
                w(f"  {ep:3d}  손실 {tot/max(nb,1):.4f} · 1등 {a1:5.1f}%"
                  f" · 10등 안 {a10:5.1f}% · 중앙값 {amed:4.0f}위{mark}")
        runs.finish(run)
        w(f"\n최고 {best[3]}에폭 · 1등 **{best[0]:.1f}%** · 10등 안 {best[1]:.1f}%"
          f" · 중앙값 {best[2]:.0f}위")
        w(f"기준선 대비 1등 {t1:.1f}% → {best[0]:.1f}%")
        if best[0] <= t1:
            w("** 기준선을 못 넘었다 — 배운 것이 없다. 짝이 모자라거나"
              " 증강이 세거나 과적합이다.")
        if o["out"]:
            w(f"{o['out']}/backbone.pt · emb.npz")
