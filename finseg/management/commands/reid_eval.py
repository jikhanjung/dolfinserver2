"""**뒷날 곡선과 전체 모양 중 무엇이 개체를 가르나** — 사람이 붙인 정답으로 잰다.

    python manage.py reid_eval
    python manage.py reid_eval --dir reid/v1 --methods curve emb

`/reid` 에서 사람이 상자에 넣은 것이 정답이다. 스크립트로 한 번 돌리고 표만
남겼던 것을 명령으로 뽑는다 — **정답은 계속 늘고**, 늘 때마다 손으로 다시
하면 그 사이에 규칙이 조용히 달라진다.

## 자리를 둘 낸다 — 하나로는 못 읽는다

| | 후보 | 무엇을 답하나 |
|---|---|---|
| **좁은 후보** | 정답이 붙은 조각끼리 | **방법을 가른다** — 곡선이냐 전체 모양이냐 |
| **격자 전체** | 조각 1,735장 전부 | **실용성을 잰다** — 카탈로그를 실제로 뒤지는 자리 |

앞엣것만 보면 "1등 42%" 가 쓸 만해 보이는데, 후보가 수백 장뿐이라 그렇다.
뒤엣것만 보면 방법 사이의 차이가 바닥에 눌려 안 보인다.

## 같은 날은 후보에서 뺀다

**연속 프레임은 개체가 아니라 장면을 재게 한다.** 같은 날 같은 조우면 빛도
물빛도 자세도 닮아서, 자가 개체를 못 알아봐도 맞힌다. 실제로 무배경 검정에서
"실루엣만 쓰면 무너진다" 는 잘못된 읽기가 나왔는데, **그 검정이 조명에 상을
주고 있었기 때문**이었다 (`HANDOFF` 의 re-ID 절).

그래서 질의도 **날을 건너뛴 짝이 있는 것**만 쓴다. 후보에서 같은 날을 통째로
빼면, 그것이 없는 조각은 정답이 후보에 없어 잴 수가 없다.

## 좌현과 우현을 절대 섞지 않는다

`reid.normalize` 가 오른쪽 무리를 거울처럼 뒤집는다 (`reid.py` 의 부호 항목).
무리 안에서는 일관되지만 **섞으면 서로 다른 두 면이 같아 보인다** — `fliplr`
을 금지하는 것과 같은 이유다. 그래서 후보는 늘 같은 쪽만이고, 좌우가 갈린
정답 짝은 **잴 수 없는 짝**으로 따로 센다.

## 격자 전체의 1등은 아래로 치우친 자다

라벨이 없는 후보가 사실 같은 개체일 수 있는데 여기서는 오답으로 센다.
**참값은 이보다 높다.** 방법끼리 견주는 데는 문제가 없고(같은 자를 댄다),
절대값을 밖에 낼 때만 이 한정을 함께 낸다.

## `emb.npz` 가 무엇인지 함께 말한다

`ResNet18 · 조각` 줄이 무엇으로 잰 것인지는 파일에 달렸다. `reid_train` 은
`backbone.pt` 와 `emb.npz` 를 **함께** 쓰므로, 옆에 `backbone.pt` 가 없으면
그것은 **학습 없는 ImageNet 특징**이다. 명령이 그 판단을 화면에 적는다 —
학습한 것과 안 한 것이 같은 이름으로 표에 앉으면 안 된다.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from finseg import reid
from finseg.models import Box

# **기본은 화면이 보는 격자다** — 박아 두면 갈아 끼울 때마다 옛것을 잰다
# 자 이름 → (파일, 화면에 쓸 이름). 순서가 표의 순서다
METHODS = ["curve", "pixel", "silhouette", "emb"]
LABEL = {"curve": "뒷날 곡선", "pixel": "조각 화소 그대로",
         "silhouette": "조각 실루엣만", "emb": "ResNet18 · 조각"}


def cosine(mat):
    """행마다 길이를 1로. **0 벡터는 0으로 남긴다** — 나누면 nan 이 되고,
    nan 은 정렬에서 조용히 맨 앞이나 맨 뒤로 가 표를 거짓말하게 만든다."""
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    return np.divide(mat, n, out=np.zeros_like(mat), where=n > 0)


def curve_sims(curves, q):
    """곡선 하나 대 전부. **`reid.distance` 와 같은 식이어야 한다** —
    거리를 두 곳에 두면 언젠가 갈린다. 아래 `_check` 가 그것을 견준다."""
    d = np.hypot(curves[:, :, 0] - q[:, 0], curves[:, :, 1] - q[:, 1])
    return -(d.mean(1) + 0.5 * d.max(1))


class Command(BaseCommand):
    help = "re-ID 자들을 사람이 붙인 개체 정답에 비교한다"

    def add_arguments(self, p):
        p.add_argument("--dir", default=str(settings.FIN_REID))
        p.add_argument("--emb", help=f"기본은 <dir>/emb.npz")
        p.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
        p.add_argument("--as-of", type=int, metavar="개체판정번호",
                       help="그때의 정답으로 잰다 — **자가 좋아진 것인지 "
                            "정답이 늘어서인지를 가른다**")
        p.add_argument("--same-day", action="store_true",
                       help="같은 날도 후보에 둔다 — **자가 조명을 재게 된다**")

    # ---- 자료 ---------------------------------------------------------------

    def _load(self, o):
        root = Path(o["dir"])
        if not root.is_dir():
            raise CommandError(
                f"{root} 가 없다 — 조각을 아직 안 만들었거나 다른 기계에 있다.\n"
                f"  NAS 백업의 `derived/reid` 를 가져올 것 (`HANDOFF`).")
        items = json.loads((root / "items.json").read_text())["items"]
        ids = np.array([i["id"] for i in items])
        day = np.array([i["day"] for i in items])
        fac = np.array([i["facing"] for i in items])

        feat, why = {}, {}
        # `--emb` 는 통째로 다른 파일을 대는 것이라 그대로 쓴다. 나머지는
        # `--dir` 밑이다 — **한 번 두 번 이어 붙여 `reid/v1/reid/v1` 을 만들었다**
        for name, path, key in [("curve", root / "curves.npz", "curve"),
                                ("pixel", root / "chips.npz", "chip"),
                                ("silhouette", root / "chips.npz", "chip"),
                                ("emb", Path(o["emb"]) if o["emb"]
                                        else root / "emb.npz", "emb")]:
            if name not in o["methods"]:
                continue
            if not path.exists():
                why[name] = f"{path} 가 없다"
                continue
            z = np.load(path)
            # **순서가 같은지 본다.** 어긋나면 예외가 안 나고 성적만 나빠진다
            if not (z["box_id"] == ids).all():
                why[name] = f"{path.name} 의 상자 순서가 items.json 과 다르다"
                continue
            feat[name] = z[key]
        return ids, day, fac, feat, why

    def _check(self, curves):
        """벡터로 다시 쓴 거리가 `reid.distance` 와 같은지 표본으로 견준다."""
        rng = np.random.default_rng(20260824)
        for i in rng.integers(0, len(curves), 12):
            got = -curve_sims(curves, curves[i])
            want = np.array([reid.distance(curves[i], c) for c in curves[:64]])
            if not np.allclose(got[:64], want, atol=1e-6):
                raise CommandError(
                    "곡선 거리가 `reid.distance` 와 다르다 — 식이 두 곳에서"
                    " 갈렸다. 고치기 전에는 이 표를 믿으면 안 된다.")

    # ---- 재기 ---------------------------------------------------------------

    def _score(self, sims, valid, positive):
        """한 질의의 순위 → (1등인가, 첫 정답 순위, AP). 정답이 후보에 없으면 None."""
        pos = positive & valid
        if not pos.any():
            return None
        s = np.where(valid, sims, -np.inf)
        order = np.argsort(-s, kind="stable")
        order = order[valid[order]]
        hit = positive[order]
        first = int(np.argmax(hit)) + 1
        rank = np.arange(1, len(hit) + 1)
        ap = float((np.cumsum(hit) / rank)[hit].mean())
        return bool(hit[0]), first, ap

    def handle(self, **o):
        w = self.stdout.write
        ids, day, fac, feat, why = self._load(o)
        n = len(ids)

        # ---- 정답 ----------------------------------------------------------
        cat = reid.catalog(as_of=o["as_of"])   # 개체 → 상자들. **보류함은 안 센다**
        of = {}
        for ind, boxes in cat.items():
            for b in boxes:
                of[b] = ind
        lab = np.array([of.get(int(b), -1) for b in ids])
        if (lab >= 0).sum() < 2:
            raise CommandError("정답이 거의 없다 — `/reid` 에서 상자에 넣은 뒤 잴 것")

        same_ind = (lab[:, None] == lab[None, :]) & (lab[:, None] >= 0)
        same_day = day[:, None] == day[None, :]
        same_fac = fac[:, None] == fac[None, :]
        np.fill_diagonal(same_ind, False)

        # **잴 수 없는 짝을 세어서 말한다** — 안 세면 표가 그것을 삼킨다
        n_pair = int(same_ind.sum() // 2)
        n_cross = int((same_ind & ~same_day).sum() // 2)
        n_flip = int((same_ind & ~same_fac).sum() // 2)

        usable = same_ind & same_fac
        if not o["same_day"]:
            usable = usable & ~same_day
        q_idx = np.where(usable.any(1))[0]
        if not len(q_idx):
            raise CommandError(
                "잴 수 있는 질의가 없다 — 개체마다 사진이 한 날에만 있다.\n"
                "  다른 날의 같은 개체를 찾아 넣어야 이 자가 선다.")

        w(f"정답 조각 {int((lab >= 0).sum()):,} · 개체 {len(cat)} · 격자 {n:,}"
          + (f"  ← 개체판정 {o['as_of']}번까지의 정답" if o["as_of"] else ""))
        if o["as_of"]:
            w("** `--as-of` 는 `kind` 를 못 되살린다 — 그때 임시보관함이던 것이"
              " 지금 개체면 정답에 들어온다 (`reid.catalog`)")
        w(f"같은 개체 짝 {n_pair:,} — 날을 건너뛴 것 {n_cross:,}"
          f" · 좌우가 갈려 못 재는 것 {n_flip:,}")
        w(f"질의 {len(q_idx):,} 조각"
          + ("" if o["same_day"] else " (날을 건너뛴 짝이 있는 것만)"))
        if o["same_day"]:
            w("** `--same-day` — 같은 날이 후보에 있다. **자가 개체가 아니라"
              " 그날 조명을 재고 있을 수 있다.**")
        if "curve" in feat:
            self._check(feat["curve"])
        emb_p = Path(o["emb"]) if o["emb"] else Path(o["dir"]) / "emb.npz"
        if "emb" in feat:
            trained = (emb_p.parent / "backbone.pt").exists()
            w(f"`{emb_p}` — {'학습한 것 (`backbone.pt` 가 옆에 있다)' if trained else '**학습 없는 ImageNet 특징** (`backbone.pt` 가 없다)'}")
        for name, msg in why.items():
            w(f"** {LABEL[name]} 은 못 잰다 — {msg}")
        w("")

        # ---- 자마다, 자리마다 ------------------------------------------------
        rows = defaultdict(dict)
        for name in [m for m in METHODS if m in feat]:
            if name == "curve":
                sims_of = lambda i: curve_sims(feat["curve"], feat["curve"][i])
            else:
                x = feat[name].reshape(n, -1).astype(np.float32)
                if name == "silhouette":
                    x = (x > 0).astype(np.float32)
                x = cosine(x)
                sims_of = lambda i, x=x: x @ x[i]

            for place, gallery in [("narrow", lab >= 0), ("grid", np.ones(n, bool))]:
                got = []
                for i in q_idx:
                    valid = gallery.copy()
                    valid[i] = False
                    if not o["same_day"]:
                        valid &= ~same_day[i]
                    valid &= same_fac[i]
                    r = self._score(sims_of(i), valid, same_ind[i])
                    if r is not None:
                        got.append((int(lab[i]), int(valid.sum())) + r)
                if not got:
                    continue
                ind_of = np.array([g[0] for g in got])
                # **후보가 몇 장이었나를 함께 낸다.** 정답이 늘면 `좁은 후보`
                # 는 후보가 함께 늘어 어려워지고 `격자 전체` 는 후보가 그대로인
                # 채 정답만 늘어 쉬워진다 — **두 자가 반대로 움직인다.** 이
                # 열이 없으면 그 움직임이 자의 성적으로 읽힌다
                cand = np.array([g[1] for g in got], float)
                top1 = np.array([g[2] for g in got], float)
                first = np.array([g[3] for g in got], float)
                ap = np.array([g[4] for g in got], float)
                # **개체 평균도 낸다** — 조각이 40장인 개체 하나가 질의의
                # 5분의 1을 차지한다. 미시 평균만 보면 그 개체를 재는 것이다
                macro = np.mean([top1[ind_of == k].mean()
                                 for k in np.unique(ind_of)])
                rows[place][name] = (top1.mean(), np.median(first), ap.mean(),
                                     macro, len(got), np.median(cand))

        head = {"narrow": "좁은 후보 — 정답끼리 (**방법을 가른다**)",
                "grid":   f"격자 전체 {n:,}장 — 카탈로그를 뒤지는 자리 (**실용성**)"}
        for place in ("narrow", "grid"):
            if not rows[place]:
                continue
            w(head[place])
            hdr = (f"  {'자':<18} {'1등':>7} {'첫 정답':>8} {'mAP':>7} "
                   f"{'개체평균 1등':>12} {'질의':>6} {'후보':>7}")
            w(hdr)
            w("  " + "-" * (len(hdr) - 2))
            for name in [m for m in METHODS if m in rows[place]]:
                t1, med, ap, macro, nq, cand = rows[place][name]
                w(f"  {LABEL[name]:<18} {t1:>7.1%} {med:>7.0f}위 {ap:>7.1%} "
                  f"{macro:>12.1%} {nq:>6,} {cand:>7,.0f}")
            w("")
        if rows["grid"]:
            w("`격자 전체` 의 1등은 **아래로 치우친 자다** — 라벨 없는 후보가"
              " 사실 같은 개체여도 오답으로 센다.")
        w("**정답이 다른 두 판을 세로로 견주지 말 것** — `후보` 열이 함께"
          " 움직인다. 견줄 수 있는 것은 한 판 안에서 자끼리다"
          " (`--as-of` 로 같은 정답에 세울 것).")
