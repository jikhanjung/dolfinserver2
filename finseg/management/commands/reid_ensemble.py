"""**앙상블을 같은 자로 채점한다** — `reid_cls --save-logits` 가 남긴 로짓으로.

    python manage.py reid_ensemble --logits full-L.npz crop-L.npz
    python manage.py reid_ensemble --logits a.npz b.npz --weights 0.5 0.5
    python manage.py reid_ensemble --logits a.npz b.npz c.npz --each

**왜 명령으로 뽑나.** 2026-09-03 에 잰 89.2 는 GPU 자리의 일회성 스크립트가
낸 것이고, 화면(`review/views.py` 의 `_score`)이 앙상블을 따로 구현하면
**다른 것을 내면서 그걸 모른다.** 합치는 규칙은 `reid.ens_logits` 한 곳에
있고, 화면과 이 명령이 그것을 부른다 — 그러니 **이 명령이 89.2 를 재현하면
화면이 내는 것도 그 규칙이다.** 안 나오면 고칠 자리도 한 곳이다.

성적 셈은 `reid_cls` 와 같다: 폴드·씨앗마다 top-1/5/10 을 세고, 씨앗을 평균한
뒤 폴드를 합친다. **질의 단위다** — 2026-09-03 의 89.2 도 그 자리의 숫자다.

**묶어서 묻는 것(`--group`)은 여기서 못 한다.** 덤프가 `te`(줄번호)와 라벨은
들고 있어도 **날을 안 들고 있어서**다 — 묶는 단위가 (개체·날·쪽)이라 날이
없으면 묶을 수 없다. 격자를 열어 날을 붙이면 되지만, 그러면 이 명령이 어느
격자로 쟀는지에 매이게 된다. 필요해지면 덤프에 날을 함께 남기는 쪽이 맞다.

**멤버끼리 같은 질의가 같은 줄에 서야 한다.** 폴드 배정이 씨앗을 안 타므로
(`reid_cls` 의 `--save-logits` 도움말) `(fold, seed, side)` 로 맞추고,
`te`(질의 줄번호)와 `classes`(개체 차례)가 어긋나면 **멈춘다** — 조용히
어긋난 채 합치면 그 숫자가 무엇인지 아무도 모른다.
"""
import numpy as np
from django.core.management.base import BaseCommand, CommandError

from finseg import reid as R


def _entries(path):
    z = np.load(path, allow_pickle=True)
    if "entries" not in z:
        raise CommandError(f"{path} 에 `entries` 가 없다 — "
                           "`reid_cls --save-logits` 가 낸 파일인가")
    return list(z["entries"])


def _key(e):
    return (int(e["fold"]), int(e["seed"]), str(e["side"]))


def _hits(logit, y):
    r = np.argsort(-logit, axis=1)
    return (int((r[:, 0] == y).sum()),
            int((r[:, :5] == y[:, None]).any(1).sum()),
            int((r[:, :10] == y[:, None]).any(1).sum()))


class Command(BaseCommand):
    help = "여러 갈래의 로짓을 합쳐 같은 자로 채점한다"

    def add_arguments(self, p):
        p.add_argument("--logits", nargs="+", required=True, metavar="파일",
                       help="`reid_cls --save-logits` 가 낸 npz 들. **둘 이상**")
        p.add_argument("--weights", nargs="+", type=float, metavar="W",
                       help="멤버 가중치. 기본은 반반 — **지렛대가 아니다** "
                            "(실측에서 0.45~0.60 이 89.0~89.2 로 평평했다)")
        p.add_argument("--each", action="store_true",
                       help="멤버 단독 성적도 함께 낸다 — 앙상블이 무엇을 "
                            "보탰는지 보려면 그 줄이 있어야 한다")

    def handle(self, **o):
        w = self.stdout.write
        paths = o["logits"]
        if len(paths) < 2 and not o["each"]:
            raise CommandError("멤버가 둘은 있어야 앙상블이다 "
                               "(단독을 보려면 `--each`)")
        mem = [{_key(e): e for e in _entries(f)} for f in paths]

        # **칸이 안 맞으면 여기서 멈춘다.** 합쳐 놓고 나면 어긋난 것이 안 보인다.
        keys = sorted(set(mem[0]), key=lambda t: (t[0], t[1], t[2]))
        for f, m in zip(paths[1:], mem[1:]):
            miss = set(keys) ^ set(m)
            if miss:
                raise CommandError(
                    f"{f} 의 (폴드·씨앗·쪽) 이 첫 멤버와 다르다 — {sorted(miss)[:4]}…")
        for kk in keys:
            e0 = mem[0][kk]
            for f, m in zip(paths[1:], mem[1:]):
                e = m[kk]
                if not np.array_equal(e["te"], e0["te"]):
                    raise CommandError(f"{f} {kk} 의 질의 차례가 첫 멤버와 다르다")
                if not np.array_equal(e["classes"], e0["classes"]):
                    raise CommandError(f"{f} {kk} 의 개체 차례가 첫 멤버와 다르다")

        n_mem = len(paths)
        lanes = [("앙상블", list(range(n_mem)))]
        if o["each"]:
            lanes = [(f"단독 {i+1}", [i]) for i in range(n_mem)] + \
                    ([] if n_mem < 2 else lanes)

        w(f"\n로짓 {n_mem}벌 · 칸 {len(keys)}개 "
          f"(폴드 {len({k[0] for k in keys})} · 씨앗 {len({k[1] for k in keys})})")
        for f in paths:
            w(f"  {f}")

        w(f"\n{'':<10}{'질의':>7}{'top-1':>8}{'top-5':>8}{'top-10':>8}{'씨앗폭':>8}")
        for name, sel in lanes:
            ws = None
            if o["weights"] and len(sel) > 1:
                if len(o["weights"]) != n_mem:
                    raise CommandError("가중치 수가 멤버 수와 다르다")
                ws = [o["weights"][i] for i in sel]
            by_seed = {}
            for kk in keys:
                e0 = mem[0][kk]
                logit = R.ens_logits([mem[i][kk]["logit"] for i in sel], ws)
                classes = list(e0["classes"])
                y = np.array([classes.index(int(v)) for v in e0["y_ind"]])
                h = _hits(logit, y)
                s = by_seed.setdefault(kk[1], [0, 0, 0, 0])
                for j in range(3):
                    s[j] += h[j]
                s[3] += len(y)
            per = [(v[0] / v[3] * 100, v[1] / v[3] * 100, v[2] / v[3] * 100)
                   for v in by_seed.values()]
            n = sum(v[3] for v in by_seed.values()) // max(len(by_seed), 1)
            a = np.array(per)
            sd = a[:, 0].std() if len(a) > 1 else 0.0
            w(f"{name:<10}{n:>7,}{a[:,0].mean():>8.1f}{a[:,1].mean():>8.1f}"
              f"{a[:,2].mean():>8.1f}{'±' + f'{sd:.1f}':>8}")

        w("\n**이 숫자가 화면이 내는 것과 같은 규칙이다** — 둘 다 "
          "`reid.ens_logits` 를 부른다. 어긋나면 그 함수 하나를 고친다.")
        w("정답이 다른 판과 세로로 비교하려면 재는 쪽에 `--as-of` 를 붙여 뒀어야 한다.")
