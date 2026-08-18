"""검출기를 **사람이 인정한 상자**에 견준다 — 옛 YOLOv5 와 새것을 같은 자로.

    python manage.py eval_detect --date 2017-08-09
    python manage.py eval_detect --date 2017-08-09 --weights runs/detect-human-mosaic0/weights/best.pt
    python manage.py eval_detect --date 2017-08-09 --weights ... --conf 0.05 0.15 0.25

## 정답이 무엇인가

`export_detect` 와 같다 — 그 날 사진에서 **사람이 지느러미로 인정한 상자
전부**다: 옛 검출기가 낸 것 중 `not_fin` 이 아닌 것 + **사람이 직접 그린 것**.

**사람이 그린 것이 있어야 재현율이 뜻을 갖는다.** 사람이 거르기만 하고 채우지
않은 날은 분모가 검출기 출력과 같아져 재현율이 늘 100%로 나온다 — 그런 날은
경고를 낸다.

## 옛 검출기는 추론을 안 돌린다

그 날 DB 에 있는 `created_by='yolov5'` 상자가 곧 그 검출기의 출력이다. 문턱
0.25 · imgsz 640 으로 한 번 돌고 남은 자취이고(`detect_fin.py`), 그것을
다시 만들 이유가 없다.

## **사람이 본 새 상자**가 정밀도의 유일한 근거다

위 표의 `정밀도` 열은 **두 줄을 견줄 수 없다.** 정답 목록이 옛 검출기가 낸
것에서만 만들어졌으므로, 새 검출기가 더 뱉은 상자는 아무도 본 적이 없어
자동으로 틀린 것이 된다 (2016-03-15 에서 64.5%로 나왔다).

그래서 `infer_boxes` 가 그 상자들을 검토 화면에 올리고, 사람이 본 것을 여기서
**문턱별로** 되읽는다. `Box.conf` 가 남아 있어 검토를 한 번만 하면 문턱은
나중에 얼마든지 옮길 수 있다.

    python manage.py eval_detect                    # 사람이 본 것만 (날짜 없이도 된다)
    python manage.py eval_detect --date 2016-03-15  # 옛것과의 비교까지

## 문턱을 여러 개 주는 이유

**우리에게는 재현율이 정밀도보다 중요하다.** 헛것은 검토 화면에서 키 한 번에
걸러지지만, 놓친 지느러미는 걸러낼 기회조차 없다 — 그것이 이 저장소가 모든
숫자에 붙여 온 천장이다. 문턱을 낮추면 그 천장이 어디까지 올라가는지 보인다.
"""
import sqlite3
from collections import defaultdict
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from finseg.models import Run


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match(preds, truth, thr):
    """욕심껏 짝짓는다 → (맞힌 수, 못 찾은 정답 목록).

    한 정답에 여러 예측이 붙어도 **하나만 맞힌 것으로 센다** — 안 그러면 같은
    지느러미에 상자를 여럿 뱉는 모델이 재현율로 이득을 본다.
    """
    used, hit = set(), 0
    for p in sorted(preds, key=lambda x: -x[4] if len(x) > 4 else 0):
        best, bi = thr, None
        for i, t in enumerate(truth):
            if i in used:
                continue
            v = iou(p, t)
            if v >= best:
                best, bi = v, i
        if bi is not None:
            used.add(bi)
            hit += 1
    return hit, [t for i, t in enumerate(truth) if i not in used]


class Command(BaseCommand):
    help = "검출기를 사람이 인정한 상자에 견준다"

    def add_arguments(self, p):
        p.add_argument("--src-db", default="db/dolfinserver_prod_2026-08-17.sqlite3")
        p.add_argument("--photos", default=str(settings.FIN_PHOTOS))
        p.add_argument("--date", help="관찰일 (보통 val_date). 없으면 "
                                     "사람이 본 새 상자만 되읽는다")
        p.add_argument("--source", default="yolo11",
                       help="되읽을 새 상자의 `Box.source`")
        p.add_argument("--weights", help="견줄 새 검출기 (없으면 옛것만)")
        p.add_argument("--conf", type=float, nargs="+", default=[0.25])
        p.add_argument("--imgsz", type=int, default=1280)
        p.add_argument("--iou", type=float, default=0.5, help="짝짓기 문턱")
        p.add_argument("--device", default="0")

    def handle(self, **o):
        w = self.stdout.write
        if not o["date"]:
            self.reviewed(o)
            return
        src = Path(o["src_db"])
        if not src.exists():
            raise CommandError(f"옛 DB 가 없다: {src}")
        c = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
        q = lambda s, *a: c.execute(s, a).fetchall()

        imgs = {i: f for i, f in q(
            "select id, imagefile from dolfinrest_dolfinimage where obsdate=?",
            o["date"])}
        if not imgs:
            raise CommandError(f"그 관찰일에 사진이 없다: {o['date']}")
        not_fin = set(x[0] for x in q(
            "select dolfin_box_id from dolfinweb_userfinbox"
            " where not_fin=1 and dolfin_box_id is not null"))
        ph = ",".join(map(str, imgs))
        truth, old, n_hum = defaultdict(list), defaultdict(list), 0
        for bid, iid, by, coords in q(
                f"select id, dolfin_image_id, created_by, coords_str"
                f" from dolfinrest_dolfinbox where dolfin_image_id in ({ph})"):
            try:
                box = tuple(int(round(float(v))) for v in coords.split(","))
            except Exception:
                continue
            if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
                continue
            human = by != "yolov5"
            if not human:
                old[iid].append(box)          # 옛 검출기의 출력 그대로
            if bid not in not_fin:
                truth[iid].append(box)        # 사람이 인정한 것
                n_hum += human

        n_truth = sum(len(v) for v in truth.values())
        w(f"관찰일 {o['date']} · 사진 {len(imgs):,} 장")
        w(f"정답 {n_truth:,} 개 (그중 **사람이 그린 것 {n_hum:,}**)")
        if n_hum == 0:
            w("  ** 사람이 그린 상자가 0개다 — 거르기만 하고 채우지 않은 날이라")
            w("     정답이 검출기 출력과 같아진다. **재현율은 뜻이 없다.**")
        hdr = (f"{'검출기':<26} {'낸 것':>7} {'맞힘':>7} {'정밀도':>8}"
               f" {'재현율':>8} {'못 찾음':>8}")
        w("\n" + hdr)
        w("-" * len(hdr))

        def row(name, per_img):
            n = sum(len(v) for v in per_img.values())
            hit = miss = 0
            for iid in imgs:
                h, m = match(per_img.get(iid, []), truth.get(iid, []), o["iou"])
                hit += h
                miss += len(m)
            p = hit / n if n else 0.0
            r = hit / n_truth if n_truth else 0.0
            w(f"{name:<26} {n:>7,} {hit:>7,} {p:>7.1%} {r:>7.1%} {miss:>8,}")

        row("옛 YOLOv5 (DB, conf .25)", old)

        if o["weights"]:
            from ultralytics import YOLO
            model = YOLO(o["weights"])
            photos = Path(o["photos"])
            paths, ids = [], []
            for iid, rel in imgs.items():
                p = photos / rel
                if p.exists():
                    paths.append(str(p))
                    ids.append(iid)
            if len(paths) < len(imgs):
                w(f"  (사진 {len(imgs) - len(paths)} 장은 디스크에 없어 뺐다)")
            # **추론은 한 번만 한다.** 문턱마다 다시 돌리면 사진을 그 횟수만큼
            # 다시 읽는데, 4928×3280 JPEG 을 읽는 것이 추론보다 오래 걸린다.
            # 가장 낮은 문턱으로 한 번 받아 두고 나머지는 걸러 내면 같은 값이다.
            lo = min(o["conf"])
            raw = defaultdict(list)
            for i in range(0, len(paths), 8):
                chunk, cids = paths[i:i + 8], ids[i:i + 8]
                res = model.predict(chunk, imgsz=o["imgsz"], conf=lo,
                                    device=o["device"], verbose=False)
                for iid, r in zip(cids, res):
                    for b, sc in zip(r.boxes.xyxy.tolist(),
                                     r.boxes.conf.tolist()):
                        raw[iid].append((b[0], b[1], b[2], b[3], sc))
            for conf in sorted(o["conf"]):
                per = {iid: [b for b in v if b[4] >= conf]
                       for iid, v in raw.items()}
                row(f"새 검출기 conf {conf:.2f}", per)
            run = Run.objects.filter(model=o["weights"], kind="train").last()
            if run:
                w(f"\n  새 검출기 = run {run.id} 의 가중치")

        w(f"\n※ 짝짓기 IoU {o['iou']}. 한 정답에 예측이 여럿 붙어도 하나만 센다.")
        w("※ 정답은 **사람이 인정한 상자**다 — 사람도 놓친 것이 있으면 여기에도"
          " 없다. 재현율은 그 한정 안에서 읽을 것.")
        w("※ **위 `정밀도` 열은 두 줄을 견줄 수 없다** — 정답이 옛 검출기가 낸"
          " 것에서만 만들어져,")
        w("   새 검출기가 더 뱉은 상자는 아무도 본 적이 없어 자동으로 틀린 것이"
          " 된다. 아래를 볼 것.")
        self.reviewed(o)

    def reviewed(self, o):
        """**사람이 본 새 상자**로 문턱별 정밀도를 낸다.

        `infer_boxes` 가 들인 상자는 옛 상자와 안 겹치는 것뿐이다 — 옛 검출기가
        놓쳤거나 헛본 자리다. 사람이 `등지느러미` 라고 한 것이 곧 **옛 천장 위로
        올라간 만큼**이고, `아무것도아님` 이 헛것이다.
        """
        from finseg import rules
        from finseg.models import Box

        w = self.stdout.write
        qs = Box.objects.filter(source=o["source"]).prefetch_related("reviews")
        if o["date"]:
            qs = qs.filter(image__obsdate__in=([o["date"]] if isinstance(
                o["date"], str) else o["date"]))
        boxes = [b for b in qs if b.conf is not None]
        if not boxes:
            w(f"\n(`{o['source']}` 상자가 없다 — manage.py infer_boxes 로 들인다)")
            return
        judged = [(b.conf, rules.effective_review(b)) for b in boxes]
        judged = [(c, r) for c, r in judged if r is not None]
        w(f"\n사람이 본 새 상자 ({o['source']}) — 들인 것 {len(boxes):,} 중"
          f" **{len(judged):,}**")
        if not judged:
            w("  아직 아무도 안 봤다. 검토 화면의 `새 검출` 대기열에서 본다.")
            return

        cuts = [0.40, 0.25, 0.15, 0.05]
        hdr = (f"{'문턱':>8} {'본 것':>7} {'등지느러미':>10} {'딴 부위':>8}"
               f" {'헛것':>7} {'정밀도':>8}")
        w("\n" + hdr)
        w("-" * len(hdr))
        for cut in cuts:
            sel = [(c, r) for c, r in judged if c >= cut]
            if not sel:
                continue
            fin = sum(1 for _, r in sel if r.cls == "fin")
            none = sum(1 for _, r in sel if r.cls == "none")
            other = len(sel) - fin - none
            w(f"  ≥{cut:.2f} {len(sel):>7,} {fin:>10,} {other:>8,}"
              f" {none:>7,} {fin / len(sel):>7.1%}")
        w("\n※ 누적이다 — `≥0.25` 는 `≥0.40` 을 품는다.")
        w("※ `딴 부위` 는 꼬리·머리·몸통처럼 **돌고래이긴 한데 등지느러미가"
          " 아닌 것**이다. 검출기가 한 분류라 정밀도에서는 헛것과 같이 친다.")
        w("※ 여기 든 상자는 **옛 상자와 안 겹치는 것뿐**이다 — 옛 검출기가"
          " 놓쳤거나 헛본 자리. `등지느러미` 로 판정된 것이")
        w("   곧 **옛 천장 위로 올라간 만큼**이고, 그것이 이 단계의 목적이다.")
