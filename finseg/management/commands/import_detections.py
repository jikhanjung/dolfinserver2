"""다른 기계가 훑은 결과를 들인다 — **데스크톱 뷰어가 내는 것을 받는 자리다.**

    python manage.py import_detections scan.json --dry-run
    python manage.py import_detections scan.json

## 왜 필요한가

`infer_boxes` 는 이 기계(2080ti)에서만 돈다. 그런데 옛 운영 DB 를 날짜별로 펴
보니 **관찰일 255일 중 28일만 썼고 사진으로는 0.78%** 다. 나머지를 훑으려면
현장 노트북이나 다른 연구자의 기계가 함께 돌아야 하고, 그 결과가 이 루프로
돌아올 길이 있어야 한다.

**`infer_boxes` 와 같은 규칙으로 들인다** — 옛 DB 와 견줘 이미 아는 상자는
빼고, 새것만 `Box` 로 넣는다. 그래야 `새 검출` 대기열에 떠서 사람이 본다.

## 받는 모양

    { "model": "detect-v2", "imgsz": 1280, "conf": 0.05, "iou": 0.7,
      "photos": [{ "path": "nas/2019/08/22/DSC_1234.JPG",
                   "w": 4928, "h": 3280,
                   "boxes": [[x1, y1, x2, y2, conf], ...] }] }

- **경로는 사진 뿌리 기준 상대경로다** (`Image.path` 와 같은 모양). 절대경로를
  주면 기계마다 달라 못 잇는다
- **`model` 과 `conf` 를 함께 받는다.** 어느 검출기가 어느 문턱으로 낸 상자인지
  모르면 `Box.source`·`Box.conf` 를 못 채우고, 그러면 문턱별 정밀도를 되읽을
  수 없다 — 어제 그것으로 문턱을 정했다
- 상자는 **원본 사진 좌표**다. 크롭이나 레터박스 좌표가 아니다

## 사진이 DB 에 없으면

`Image` 를 만든다. 다만 **관찰일은 경로에서 못 읽는다** — 옛 DB 에 있으면
거기서 가져오고, 없으면 비워 둔다. `obsdate` 가 비면 `export_detect` 의 날짜
가르기에 안 잡히므로, 그런 사진이 몇 장인지 반드시 말한다.
"""
import json
from collections import Counter
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from finseg import runs
from finseg.management.commands.infer_boxes import inside, iou
from finseg.models import Box, Image


class Command(BaseCommand):
    help = "다른 기계가 훑은 결과(JSON)를 들인다"

    def add_arguments(self, p):
        p.add_argument("scan", help="스캔 결과 JSON")
        p.add_argument("--src-db", default="db/dolfinserver_prod_2026-08-17.sqlite3",
                       help="**옛 DB 도 함께 본다** — 아래 문서 참고")
        p.add_argument("--iou", type=float, default=0.4,
                       help="이 이상 겹치면 이미 아는 상자로 본다")
        p.add_argument("--min-side", type=int, default=12)
        p.add_argument("--source", help="기본은 JSON 의 `model`")
        p.add_argument("--dry-run", action="store_true")

    def handle(self, **o):
        import sqlite3

        w = self.stdout.write
        f = Path(o["scan"])
        if not f.exists():
            raise CommandError(f"그런 파일이 없다: {f}")
        try:
            scan = json.loads(f.read_text())
        except Exception as e:
            raise CommandError(f"JSON 을 못 읽었다: {e}")
        photos = scan.get("photos")
        if not photos:
            raise CommandError("`photos` 가 비어 있다")
        source = o["source"] or scan.get("model")
        if not source:
            raise CommandError(
                "어느 검출기가 낸 것인지 모른다 — JSON 에 `model` 을 적거나"
                " `--source` 로 줄 것. **모르면 `Box.source` 를 못 채우고,"
                " 그러면 나중에 문턱별 정밀도를 되읽을 수 없다.**")
        w(f"스캔 {f} · 모델 `{source}` · 사진 {len(photos):,}")
        w(f"  conf {scan.get('conf', '?')} · imgsz {scan.get('imgsz', '?')}"
          f" · 만든 때 {scan.get('scanned_at', '?')}")

        # **이미 아는 상자는 `fin.db` 만 보면 안 된다.** `fin.db` 는 표본이라
        # 그 사진의 옛 상자를 다 갖고 있지 않다 — 사진 92장에 옛 DB 는 189개인데
        # 우리는 109개만 들여왔다. 그것 하고만 견주면 나머지 80개가 "새 검출"
        # 로 둔갑한다 (`infer_boxes` 도입부 · `CLAUDE.md`).
        # **빈 문자열을 경로로 쓰면 안 된다.** `Path("")` 은 현재 디렉토리로
        # 풀려서 `.exists()` 가 참이고, 그러면 디렉토리를 sqlite 로 열려다
        # "disk I/O error" 가 난다. `--src-db ''` 는 "옛 DB 를 안 본다" 는 뜻이다
        # (`infer_boxes` 와 같다).
        src = Path(o["src_db"]) if o["src_db"] else None
        known, obsdate = {}, {}
        if src is not None and src.is_file():
            c = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
            rows = c.execute(
                "select i.imagefile, i.obsdate, b.coords_str"
                " from dolfinrest_dolfinbox b"
                " join dolfinrest_dolfinimage i on i.id=b.dolfin_image_id").fetchall()
            for rel, day, coords in rows:
                if day:
                    obsdate[rel] = str(day)
                try:
                    x1, y1, x2, y2 = (float(v) for v in coords.split(","))
                except Exception:
                    continue
                known.setdefault(rel, []).append((x1, y1, x2, y2))
            w(f"  옛 DB 에서 이미 아는 상자 {sum(len(v) for v in known.values()):,} 개")
        else:
            w(f"  ** 옛 DB 를 안 본다 ({src or '--src-db 가 비었다'})"
              f" — 이미 아는 상자를 못 걸러낸다.")
            w("     그러면 옛 검출기가 이미 찾은 것이 '새 검출' 로 둔갑한다.")

        for img in Image.objects.all().only("path", "id"):
            known.setdefault(img.path, [])
        for b in Box.objects.select_related("image").only(
                "x1", "y1", "x2", "y2", "image__path"):
            known.setdefault(b.image.path, []).append((b.x1, b.y1, b.x2, b.y2))

        drop = Counter()
        add = []          # (path, w, h, x1, y1, x2, y2, conf)
        for p in photos:
            rel = p.get("path")
            if not rel:
                drop["경로가 없다"] += 1
                continue
            olds = known.get(rel, [])
            for box in p.get("boxes", []):
                if len(box) < 4:
                    drop["상자 모양이 이상하다"] += 1
                    continue
                x1, y1, x2, y2 = (float(v) for v in box[:4])
                conf = float(box[4]) if len(box) > 4 else None
                if x2 - x1 < o["min_side"] or y2 - y1 < o["min_side"]:
                    drop["너무 작다"] += 1
                    continue
                cand = (x1, y1, x2, y2)
                # 겹침은 IoU 로 재되 **중심이 옛 상자 안에 들어가도 같은 것**으로
                # 본다 — 새 검출기가 좁게 따는 버릇이 있어 IoU 만 보면 같은
                # 지느러미가 남처럼 보인다 (`infer_boxes` 와 같은 규칙)
                if any(iou(cand, old) >= o["iou"] or inside(cand, old)
                       for old in olds):
                    drop["이미 아는 상자"] += 1
                    continue
                olds.append(cand)          # 이번 스캔 안에서도 겹치면 하나만
                add.append((rel, p.get("w"), p.get("h"), x1, y1, x2, y2, conf))

        w(f"\n들일 상자 {len(add):,} 개")
        for k, v in drop.most_common():
            w(f"  뺐다 — {k}: {v:,}")
        new_photos = {r[0] for r in add} - set(
            Image.objects.filter(path__in={r[0] for r in add})
            .values_list("path", flat=True))
        no_date = {r for r in new_photos if r not in obsdate}
        if new_photos:
            w(f"  DB 에 없던 사진 {len(new_photos):,} 장을 새로 만든다")
        if no_date:
            w(f"  ** 그중 {len(no_date):,} 장은 **관찰일을 모른다** — 옛 DB 에도"
              f" 없다.")
            w("     `obsdate` 가 비면 `export_detect` 의 날짜 가르기에 안 잡힌다.")
        if not add:
            w("\n들일 것이 없다.")
            return
        if o["dry_run"]:
            w("\n--dry-run 이라 아무것도 쓰지 않았다.")
            return

        run = runs.start("detect", model=source, params={
            "kind": "import_detections", "scan": str(f), "source": source,
            "conf": scan.get("conf"), "imgsz": scan.get("imgsz"),
            "iou": o["iou"], "photos": len(photos), "new_boxes": len(add),
            "dropped": dict(drop)})
        with transaction.atomic():
            imgs = {i.path: i for i in Image.objects.filter(
                path__in={r[0] for r in add})}
            for rel in new_photos:
                p0 = next(r for r in add if r[0] == rel)
                imgs[rel] = Image.objects.create(
                    path=rel, width=p0[1], height=p0[2],
                    obsdate=obsdate.get(rel) or None)
            Box.objects.bulk_create([
                Box(image=imgs[rel], x1=int(x1), y1=int(y1),
                    x2=int(x2), y2=int(y2),
                    area=int((x2 - x1) * (y2 - y1)),
                    conf=conf, run=run, source=source)
                for rel, _, _, x1, y1, x2, y2, conf in add])
        runs.finish(run)
        w(f"\nrun {run.id} · 상자 {len(add):,} 개를 들였다")
        w("  다음: manage.py crops   — 크롭이 있어야 검토 화면에 뜬다")
        w("  그다음 검토 화면의 `새 검출` 대기열에서 본다")
