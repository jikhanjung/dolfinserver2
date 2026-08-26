"""새 검출기가 낸 상자 중 **아직 아무도 안 본 것**을 들인다.

    python manage.py infer_boxes --weights runs/detect-merged-det/weights/best.pt \\
        --date 2016-03-15 --conf 0.05 --dry-run
    python manage.py infer_boxes --weights ... --date 2016-03-15 --conf 0.05
    python manage.py crops            # 이어서 — 크롭이 있어야 검토 화면에 뜬다

## 왜 필요한가 — **정밀도를 잴 방법이 없었다**

`eval_detect` 는 새 검출기가 낸 상자를 "사람이 인정한 상자" 에 견준다. 그런데 그
정답 목록은 **옛 검출기가 낸 것에서만** 만들어졌다. 새 검출기가 더 뱉은 상자는
아무도 본 적이 없어 자동으로 틀린 것이 되고, 정밀도가 64.5%로 나온다.

그 1,056개 중 진짜가 몇인지는 **사람이 보는 수밖에 없다.** 이 명령이 그것을
검토 화면에 올린다.

**옛 상자와 겹치는 것은 안 들인다.** 그것은 이미 두 바퀴 검토했고, 여기서 묻는
것은 "새로 뱉은 것이 진짜인가" 하나다. 겹침은 IoU 로 재되 **중심이 옛 상자 안에
들어가도 같은 것으로 본다** — 새 검출기가 지느러미를 더 좁게 따는 버릇이 있어
(`devlog/…_001` 4절) IoU 만 보면 같은 지느러미가 남처럼 보인다.

## **`fin.db` 만 보면 안 된다** — 옛 DB 도 함께 본다

`fin.db` 는 표본이다. 2016-03-15 의 사진 92장에 옛 DB 는 상자 189개를 들고 있는데
우리가 들여온 것은 **109개뿐**이다 (`import_boxes --boxes 2000` 이 골랐다).
`fin.db` 하고만 견주면 그 **80개가 "새 검출" 로 둔갑한다** — 옛 검출기가 이미
찾은 것을 사람에게 "새로 찾은 것" 이라며 보여 주고, 그 판정으로 정밀도를 재면
숫자가 통째로 틀린다.

같은 모양의 함정을 어제 세 번 만났다 (`devlog/…_002` 6절). 자를 대기 전에
**"이 목록이 무엇을 다 담고 있나"** 를 물어야 한다 — 여기서는 `fin.db` 가
"이미 아는 상자" 를 다 담고 있지 않았다.

## 들어온 상자는 무엇이 되나

`Box` 한 줄이다 — 옛 YOLOv5 상자와 같은 테이블에, `source` 와 `run` 과 `conf` 가
다를 뿐이다. 그래서 크롭·분할·검토·내보내기가 **전부 그대로 돌아간다.**
사람이 `등지느러미` 라고 하면 그 상자는 곧바로

- **검출 자료의 양성** — 옛 검출기가 놓쳐서 아무 라벨에도 없던 자리다
- **분할 자료의 양성** — 마스크를 뽑으면 그대로 크롭 표본이 된다

이 된다. `아무것도아님` 이라고 하면 **어려운 음성**이다. 어느 쪽이든 다음
바퀴에 쓰인다 — 검토가 헛돌지 않는다.

## 마스크는 나중이다

들일 때는 폴리곤이 없다. 검토 화면은 마스크 없이도 뜨고(`rules.resolve` 가
`mask=None` 을 받는다), 여기서 묻는 것은 "이 안에 지느러미가 있나" 라 윤곽이
필요 없다. 진짜로 밝혀진 것에만 `segment`/`infer` 를 돌리는 것이 싸다.
"""
from collections import defaultdict
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from finseg import runs
from finseg.models import Box, Image


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def inside(a, b):
    """`a` 의 중심이 `b` 안에 있나 — 좁게 딴 상자를 남으로 세지 않으려고."""
    cx, cy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    return b[0] <= cx <= b[2] and b[1] <= cy <= b[3]


class Command(BaseCommand):
    help = "새 검출기가 낸 상자 중 옛 상자와 안 겹치는 것을 들인다"

    def add_arguments(self, p):
        p.add_argument("--weights", required=True)
        p.add_argument("--src-db", default="db/dolfinserver_prod_2026-08-17.sqlite3",
                       help="옛 운영 DB — **겹침 판정에 반드시 넣는다** (위 문서). "
                            "'' 로 두면 fin.db 만 본다")
        p.add_argument("--photos", default=str(settings.FIN_PHOTOS))
        p.add_argument("--date", nargs="+",
                       help="관찰일 (없으면 표본에 든 날 전부)")
        p.add_argument("--conf", type=float, default=0.05,
                       help="이 아래는 아예 안 들인다. **`Box.conf` 에 남으므로 "
                            "문턱은 나중에 올릴 수 있다** — 낮게 받아 두는 편이 낫다")
        p.add_argument("--imgsz", type=int, default=1280)
        p.add_argument("--iou", type=float, default=0.4,
                       help="이 위로 겹치면 이미 있는 상자로 본다")
        p.add_argument("--min-side", type=int, default=12,
                       help="이보다 짧은 상자는 버린다 — 크롭을 640 으로 펴도 "
                            "사람이 못 가린다")
        p.add_argument("--source", default="yolo11",
                       help="`Box.source` 에 적을 이름")
        p.add_argument("--device", default="0")
        p.add_argument("--batch", type=int, default=8)
        p.add_argument("--dry-run", action="store_true")

    def handle(self, **o):
        w = self.stdout.write
        weights = Path(o["weights"])
        if not weights.exists():
            raise CommandError(f"가중치가 없다: {weights}")
        photos = Path(o["photos"])

        imgs = Image.objects.all()
        if o["date"]:
            imgs = imgs.filter(obsdate__in=o["date"])
        imgs = list(imgs)
        if not imgs:
            raise CommandError("그 관찰일에 사진이 없다 — 표본에 든 날인가?")

        # **이미 있는 상자 전부**가 겹침 판정의 기준이다. 옛 YOLOv5 것뿐 아니라
        # 앞서 들인 새 상자도 포함한다 — 두 번 돌려도 같은 것이 두 줄이 되면 안 된다.
        have = defaultdict(list)
        for b in Box.objects.filter(image__in=imgs).only(
                "image_id", "x1", "y1", "x2", "y2"):
            have[b.image_id].append((b.x1, b.y1, b.x2, b.y2))
        n_ours = sum(len(v) for v in have.values())
        # **옛 DB 의 상자도 넣는다.** `fin.db` 는 표본이라 같은 사진의 옛 상자를
        # 다 갖고 있지 않다 (위 문서). 안 넣으면 옛 검출기가 이미 찾은 것이
        # "새 검출" 로 둔갑하고 정밀도가 통째로 틀린다.
        n_src = 0
        src_db = Path(o["src_db"]) if o["src_db"] else None
        if src_db and src_db.exists():
            import sqlite3
            by_src = {im.src_id: im.id for im in imgs if im.src_id}
            c = sqlite3.connect(f"file:{src_db}?mode=ro", uri=True)   # 읽기 전용
            ph = ",".join(map(str, by_src))
            if ph:
                for sid, co in c.execute(
                        f"select dolfin_image_id, coords_str from"
                        f" dolfinrest_dolfinbox where dolfin_image_id in ({ph})"):
                    try:
                        b = tuple(int(round(float(v))) for v in co.split(","))
                    except Exception:
                        continue
                    if len(b) == 4 and b[2] > b[0] and b[3] > b[1]:
                        have[by_src[sid]].append(b)
                        n_src += 1
            c.close()
        elif src_db:
            raise CommandError(
                f"옛 DB 가 없다: {src_db}\n"
                "  **이것 없이 재면 안 된다** — fin.db 는 표본이라 옛 상자를 다\n"
                "  갖고 있지 않고, 빠진 것이 '새 검출' 로 둔갑한다.\n"
                "  정말 fin.db 만으로 견주려면 --src-db '' 로 뜻을 밝힐 것.")

        paths, ids, missing = [], [], 0
        for im in imgs:
            p = photos / im.path
            if p.exists():
                paths.append(str(p))
                ids.append(im)
            else:
                missing += 1
        if not paths:
            raise CommandError(f"사진이 하나도 없다 — {photos} 가 붙어 있나?")
        w(f"사진 {len(paths):,} 장" + (f" (디스크에 없어 뺀 것 {missing:,})"
                                       if missing else ""))
        w(f"이미 있는 상자 {n_ours + n_src:,} 개와 견준다"
          f" (fin.db {n_ours:,} + 옛 DB {n_src:,}"
          f" · IoU {o['iou']} 또는 중심이 안에)")
        if not n_src:
            w("  ** 옛 DB 를 안 봤다 — fin.db 는 표본이라 옛 상자를 다 갖고 있지")
            w("     않다. 빠진 것이 '새 검출' 로 둔갑할 수 있다.")

        from ultralytics import YOLO
        model = YOLO(str(weights))
        fresh, dup, tiny = [], 0, 0
        for i in range(0, len(paths), o["batch"]):
            chunk, cims = paths[i:i + o["batch"]], ids[i:i + o["batch"]]
            res = model.predict(chunk, imgsz=o["imgsz"], conf=o["conf"],
                                device=o["device"], verbose=False)
            for im, r in zip(cims, res):
                old = have[im.id]
                for xyxy, sc in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                    b = tuple(int(round(v)) for v in xyxy)
                    if b[2] - b[0] < o["min_side"] or b[3] - b[1] < o["min_side"]:
                        tiny += 1
                        continue
                    if any(iou(b, ob) >= o["iou"] or inside(b, ob) for ob in old):
                        dup += 1
                        continue
                    # **이번에 들인 것끼리도 겹치면 안 된다** — 같은 지느러미에
                    # 상자를 둘 뱉는 경우가 있고, 그러면 사람이 같은 것을 두 번 본다
                    old.append(b)
                    fresh.append((im, b, float(sc)))
            if (i // o["batch"]) % 20 == 0 and i:
                w(f"  {i:,} / {len(paths):,} 장 · 새 상자 {len(fresh):,}")

        w("")
        w(f"낸 것 중 이미 있는 상자와 같다  {dup:,}")
        w(f"너무 작아 버린 것             {tiny:,}")
        w(f"**새로 들일 상자             {len(fresh):,}**")
        if fresh:
            for lo, hi in ((0.4, 1.01), (0.25, 0.4), (0.15, 0.25), (0.0, 0.15)):
                n = sum(1 for _, _, c in fresh if lo <= c < hi)
                w(f"    conf {lo:.2f} ~ {hi if hi <= 1 else 1.0:.2f}   {n:,}")
        if o["dry_run"]:
            w("\n--dry-run 이라 아무것도 쓰지 않았다.")
            return
        if not fresh:
            return

        run = runs.start("detect", model=str(weights), params={
            "kind": "infer_boxes", "conf": o["conf"], "imgsz": o["imgsz"],
            "iou": o["iou"], "dates": o["date"] or [], "photos": len(paths),
            "new_boxes": len(fresh), "dup": dup, "tiny": tiny,
            "source": o["source"]})
        with transaction.atomic():
            Box.objects.bulk_create([
                Box(image=im, x1=b[0], y1=b[1], x2=b[2], y2=b[3],
                    area=(b[2] - b[0]) * (b[3] - b[1]),
                    source=o["source"], conf=c, run=run)
                for im, b, c in fresh], batch_size=500)
        runs.finish(run)
        w(f"\n상자 {len(fresh):,} 개를 들였다 (run {run.id})")
        w("**다음: manage.py crops** — 크롭이 있어야 검토 화면에 뜬다")
        w("      그 뒤 검토 화면의 `새 검출` 대기열에서 본다")
