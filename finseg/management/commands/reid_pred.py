"""**판정마다 "그때 모델은 뭐라고 했나" 를 채운다.**

    python manage.py reid_pred            # 안 채워진 것만
    python manage.py reid_pred --all      # 전부 다시
    python manage.py reid_pred --dry-run

## 왜 적어 두나

모델이 바뀌면 **그때 그 모델은 다시 못 만든다** — 블록도 머리도 `T` 도 바뀐다.
그러니 "제안이 얼마나 맞았나" 는 **그때 적어 두지 않으면 영영 못 묻는다.**
이 저장소가 `ip` 를 뒤늦게 붙여 옛 줄이 영영 빈 자리와 같은 종류다.

## 왜 판정 요청 안에서 안 하나

격자를 여는 데 **1.1초**가 든다(임베딩 세 벌이 100MB다). 판정마다 물리면 끌어
놓을 때마다 화면이 그만큼 멎는다. **모델은 다음 학습 바퀴 전까지 안 바뀌므로**
그 사이 아무 때나 채워도 같은 값이다 — 하루 한 번이면 된다.

## 읽을 때 붙는 한정 — **여기가 요점이다**

`pred` 가 맞았다고 다 같은 뜻이 아니다.

- **블록이 그 정답을 보고 배운 조각**은 자기 채점이다. 지금 멤버들은
  개체판정 **2,168** 까지의 정답으로 배웠으므로, 그 아래 번호는 성적으로 쓰면
  안 된다
- **`source` 가 `bulk`·`suggest` 인 것**은 사람이 **모델의 제안을 보고** 고른
  것이라 쉬운 쪽으로 치우쳐 있다. 2026-09-07 에 그것으로 99.3% 라는 못 믿을
  숫자가 나왔다
- **치우침 없는 자는 `source='hand'` 이고 번호가 2,168 을 넘는 것**이다 —
  사람이 제 순서(날짜·시간)로 만나 붙인 것
- **`bulk` 대 `bulk-x` 의 비도 성적이 아니다.** 그 화면의 일하는 법이
  *확실한 것만 옮기고 애매한 것은 그대로 둔다* 라, 애매한 것이 어느 갈래에도
  안 남는다 — **쉬운 쪽만 센 값**이라 실제보다 높다. 갈래는 "어느 길로 온
  판정인가" 를 가리는 데 쓰지, 그 자체를 정답률로 읽지 않는다

`pred_by` 가 어느 모델이 한 말인지 적는다. 그것이 없으면 위 첫 줄을 못 가른다.

## 세는 단위 — **상자별 마지막 줄**

`Identification` 은 쌓이는 테이블이라 한 상자에 여러 줄이 있다(옮길 때마다 늘고
되돌리기도 +2 다). 줄을 그대로 세면 **덧쓰인 옛 판단까지 성적에 든다** —
2026-09-07 에 그렇게 재서 49.1% 가 나왔고, 상자별 마지막 줄로 고치니 같은
자료가 99.3% 였다. `HANDOFF` 의 *"개체 판정 N 은 일한 양이 아니다"* 가 그 말이다.

    with last as (select i.* from finseg_identification i
                  where i.id = (select max(j.id) from finseg_identification j
                                where j.box_id = i.box_id))
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from finseg import reid as R


def model_tag(root):
    """지금 격자에 선 모델의 이름표. 명단과 온도가 바뀌면 함께 바뀐다."""
    f = root / "ensemble.json"
    if f.exists():
        spec = json.loads(f.read_text())
        names = "+".join(Path(m["cls"]).stem.replace("cls-", "")
                         for m in spec.get("members", []))
        t = spec.get("T")
        tag = f"{names}@T{t}" if t else names
    else:
        tag = "cls-dinov2"
    # 파일이 갈렸는데 이름이 같을 수 있다 — 머리의 sha 앞자리를 붙인다
    h = hashlib.sha256()
    for n in sorted(p.name for p in root.glob("cls-*.npz")):
        h.update(Path(root / n).read_bytes())
    return f"{tag}#{h.hexdigest()[:8]}"[:80]


class Command(BaseCommand):
    help = "판정마다 그때 모델의 1위와 확률을 채운다"

    def add_arguments(self, p):
        p.add_argument("--dir", default=str(settings.FIN_REID))
        p.add_argument("--all", action="store_true",
                       help="이미 채워진 것도 **다시** 적는다 — 모델을 갈아 "
                            "끼운 뒤 옛 말을 지우려는 것이 아니라면 쓰지 말 것")
        p.add_argument("--dry-run", action="store_true")

    def handle(self, **o):
        from finseg.models import Identification
        w = self.stdout.write
        root = Path(o["dir"])
        items_f = root / "items.json"
        if not items_f.exists():
            raise CommandError(f"격자가 없다: {items_f}")
        items = json.loads(items_f.read_text())["items"]
        ids = np.array([it["id"] for it in items])
        fac = np.array([it["facing"] for it in items])
        pos = {int(b): k for k, b in enumerate(ids)}

        spec = {}
        f = root / "ensemble.json"
        if f.exists():
            spec = json.loads(f.read_text())
        members = spec.get("members") or [{"emb": "emb-dinov2.npz",
                                           "cls": "cls-dinov2.npz", "w": 1}]
        T = float(spec.get("T", 1.0))
        mem = []
        for m in members:
            ef, cf = root / m["emb"], root / m["cls"]
            if not (ef.exists() and cf.exists()):
                raise CommandError(f"멤버 파일이 없다: {ef} · {cf}")
            z = np.load(ef)
            if not (z["box_id"] == ids).all():
                raise CommandError(f"{ef} 의 상자 차례가 격자와 다르다")
            E = z["emb"]
            X = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
            h = np.load(cf)
            head = {s: (h[f"{s}_W"], h[f"{s}_b"], h[f"{s}_cls"])
                    for s in ("left", "right") if f"{s}_W" in h}
            mem.append((X, head, float(m.get("w", 1))))
        tag = model_tag(root)
        w(f"모델 {tag}\n격자 {len(ids):,} · 멤버 {len(mem)} · T {T}")

        # **개체가 아닌 상자는 건너뛴다.** `임시보관함`·`식별 불가능`·
        # `지느러미 아님` 은 "어느 개체인가" 의 답이 아니고, 머리도 그것을
        # 배운 적이 없어 **구조상 영영 못 맞힌다.** 함께 세면 평균이 그만큼
        # 눌리는데 그것은 모델이 틀린 것이 아니다 — 실측으로 2026-09-07 에
        # `unid` 390장이 0%로 들어가 794장 50.5%가 됐다. 빼면 404장 99.3%다.
        qs = (Identification.objects.filter(individual__kind="")
              .exclude(individual=None).order_by("id"))
        if not o["all"]:
            qs = qs.filter(pred=None)
        rows = list(qs.values_list("id", "box_id"))
        w(f"채울 판정 {len(rows):,}")
        if not rows:
            return

        done = skip = 0
        upd = []
        for ident_id, box_id in rows:
            r = pos.get(int(box_id))
            if r is None:
                skip += 1
                continue
            side = str(fac[r])
            live = [m for m in mem if side in m[1]]
            if not live:
                skip += 1
                continue
            common = None
            for _, head, _w in live:
                s = {int(c) for c in head[side][2]}
                common = s if common is None else (common & s)
            common = sorted(common)
            if not common:
                skip += 1
                continue
            logits, ws = [], []
            for X, head, wt in live:
                W, b, classes = head[side]
                col = {int(c): i for i, c in enumerate(classes)}
                v = X[r] @ W.T + b
                logits.append(v[[col[c] for c in common]][None, :])
                ws.append(wt)
            # **화면과 같은 규칙이다** — `reid.ens_logits` 하나를 부른다
            sc = R.softmax(R.ens_logits(logits, ws)[0] / T)
            k = int(np.argmax(sc))
            upd.append((ident_id, int(common[k]), float(sc[k])))
            done += 1

        w(f"  낸 것 {done:,} · 못 낸 것 {skip:,}"
          f" (격자에 없거나 그 쪽을 배운 머리가 없다)")
        w("  **개체 아닌 상자(`임시보관함`·`식별 불가능`·`지느러미 아님`)는 "
          "아예 안 센다** — 어느 개체인가의 답이 아니다")
        if o["dry_run"]:
            w("--dry-run 이라 아무것도 쓰지 않았다.")
            return
        objs = []
        for ident_id, ind, p in upd:
            objs.append(Identification(id=ident_id, pred=ind, pred_p=p,
                                       pred_by=tag))
        Identification.objects.bulk_update(objs, ["pred", "pred_p", "pred_by"],
                                           batch_size=500)
        w(f"\n{done:,} 줄에 적었다.")
        w("**이것으로 성적을 낼 때는 갈래를 가릴 것** — 블록이 그 정답을 보고 "
          "배운 조각(개체판정 2,168 이하)은 자기 채점이고, `source` 가 "
          "`bulk`·`suggest` 인 것은 제안을 보고 고른 것이라 치우쳐 있다. "
          "치우침 없는 자는 **`source='hand'` 이고 그 번호를 넘는 것**이다.")
