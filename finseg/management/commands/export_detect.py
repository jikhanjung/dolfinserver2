"""**검출기** 학습 자료를 옛 운영 DB 에서 만든다 — 사진 한 장이 학습 이미지 하나다.

    python manage.py export_detect --out datasets/detect-human --dry-run
    python manage.py export_detect --out datasets/detect-human

지금까지의 `export_yolo` 는 **크롭**을 냈다. 그것은 상자를 받아 그 안을 분할하는
모델의 자료이고, 사진 전체에서 지느러미를 찾는 일과는 다르다 (`train.py` 의
"이 모델은 SAM2.1 자리에 들어가는 것이지 검출기가 아니다").

## 어느 사진을 쓰나 — **사람이 손댄 것만**

검출기를 학습시키려면 **그 사진의 지느러미가 빠짐없이 표시돼야** 한다. 그런데
옛 DB 의 상자 999,528개는 전부 검출기 출력이라, 그 검출기가 놓친 것은 어디에도
없다. 그대로 학습시키면 **놓친 지느러미를 배경이라고 가르치고** 새 검출기가 옛
천장을 그대로 물려받는다.

그래서 **시민과학자가 실제로 들여다본 사진만** 쓴다 — `dolfinweb_userfinbox` 에
판정이 붙었거나 사람이 상자를 그린 사진이다. 거기서는 사람이 틀린 것을 지우고
빠진 것을 채웠으므로 "빠짐없이" 에 가장 가깝다.

## 그래도 완전하지는 않다 — `--exclude-date` 가 있는 이유

날짜별로 사람이 한 일이 다르다. 실측:

    2016-03-16   검출기 1,005 · 사람이 더함 **0** · not_fin 255
    2017-08-09   검출기   175 · 사람이 더함 **169** · not_fin  51

앞의 날은 **거르기만 하고 더하지 않았다.** 그 사진들을 넣으면 놓친 지느러미가
배경으로 들어간다 — 우리가 피하려던 바로 그것이다. 뒤의 날은 사람이 채워
넣었고, 그날의 검출기 재현율이 175/344 = **51%** 로 드러난다.

**"사람이 봤다" 와 "사람이 채웠다" 는 다르다.** 전자만으로는 완전 라벨이
아니므로, 채우지 않은 날은 이름을 대어 뺀다.

## 옛 학습 자료를 함께 들인다 (`--legacy-labels`)

옛 YOLOv5(`dolfin_1280_s_100.pt`)를 학습시킬 때 쓴 라벨이 NAS 에 남아 있다 —
`DolFinID/TrainingData/<날짜>_<지명>/*.txt`, 1,910장. 이미 YOLO 형식이라 그대로
쓸 수 있고, **사람이 손댄 사진(817장)과 날짜가 거의 안 겹쳐** 서로 보완한다.

폴더 이름과 DB `dirname` 이 안 맞아서(`20170809_김녕` 대 `20170809_김녕_annotation`)
**날짜 + 파일 이름**으로 잇는다. 1,910 중 1,887 이 이어지고 전부 디스크에 있다.

**한 사진이 양쪽에 있으면 사람이 손댄 쪽을 쓴다.** 그쪽에는 사람이 그려 넣은
상자가 들어 있어 더 완전하다.

## `val_date` 는 **양쪽 다 안 본 날**이어야 한다

옛 학습 자료에 `20170809`·`20170810` 이 들어 있다. 그것을 val_date 로 두고 옛
검출기와 견주면 **옛것은 외운 날에서, 새것은 처음 보는 날에서** 재는 셈이라
비교가 기운다. 실제로 그렇게 재고 "새것이 못 이겼다" 고 읽을 뻔했다.
기본값 `2016-03-15` 는 옛 목록에 없고 사람이 상자를 109개 그려 넣은 날이다.

## `fin.db` 의 판정도 읽는다 (`--no-fin-db` 로 끈다)

**여기가 오래 끊겨 있던 자리다.** 검토 화면에서 인정한 상자가 다음 검출 학습에
안 들어가면 검토가 헛돈다 — 사람이 그린 상자 330개를 찾아 쓴 것과 똑같은 자리다.

`fin.db` 가 주는 것은 셋이고, **셋 다 옛 DB 에는 없다.**

1. **새 검출기가 찾아 사람이 인정한 상자** (`source='yolo11'` · `cls='fin'`).
   옛 검출기가 놓쳐 **아무 라벨에도 없던 자리**다 — 이것이 재현율의 천장을 연다
2. **사람이 헛것이라 한 옛 상자** (`cls` 가 `fin` 이 아닌 것). 옛 DB 에서는
   `fin` 으로 나가던 것이라 **틀린 라벨을 빼는 쪽**이다. 옛 DB 의 `not_fin` 은
   시민과학자가 표시한 것만 잡고, 이쪽은 전량 검토라 훨씬 촘촘하다
3. **그 사진들 자체.** 옛 DB 의 '손댄 사진' 과 54장밖에 안 겹친다

## `fin.db` 사진이 왜 "빠짐없이" 에 가까운가

이 자료의 요건은 하나다 — **그 사진의 지느러미가 빠짐없이 표시돼야 한다.**
`fin.db` 사진은 검출기 **둘**이 훑었고 그 결과를 사람이 전량 판정했다:

- 옛 검출기의 상자는 옛 DB 에서 그대로 가져온다 (`fin.db` 는 표본이라 일부만
  들여왔지만, 라벨은 옛 DB 에서 만들므로 안 들여온 것도 함께 나간다)
- 새 검출기가 `conf 0.05` 로 다시 훑어 안 겹치는 것을 690개 들였고
  (`infer_boxes` — **옛 DB 와 견줘서** 걸렀다), 그 690개에 전부 판정이 붙었다

그래서 **`--date` 로 훑은 날의 사진만** 쓴다. 안 훑은 날은 검출기 하나만 본
것이라 완전성이 한 단계 낮다 — 그 날짜는 `infer_boxes` 의 run 기록에서 읽는다.

**여전히 완전하지는 않다.** 둘 다 놓친 지느러미는 배경으로 들어가고, 옛 DB
상자 중 `import_boxes --min-area` 아래로 잘려 안 들여온 것은 아무도 안 봤다.
전자는 이 루프 안에서 열 방법이 없고(`HANDOFF` 의 "재현율에는 늘 한정이 붙는다"),
후자는 옛 검출기 출력이라 옛 라벨과 같은 지위다.

## 사진은 복사하지 않고 링크한다

원본이 4928×3280 · 장당 10MB 라 856장이면 8.5GB 다. 심볼릭 링크로 걸면 0바이트다
(ultralytics 가 따라간다). **크롭과 달리 원본은 하나뿐이고 바뀌지 않는다.**
"""
import json
import os
import random
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from finseg import nas, runs
from finseg.models import Review

NAMES = ["fin"]   # 검출기는 한 분류다 — 옛 DB 의 사람 판정이 fin/not_fin 뿐이다


class Command(BaseCommand):
    help = "옛 운영 DB 에서 검출기 학습 자료를 만든다 (사진 한 장 = 이미지 하나)"

    def add_arguments(self, p):
        p.add_argument("--src-db", default="db/dolfinserver_prod_2026-08-17.sqlite3")
        p.add_argument("--photos", default=str(settings.FIN_PHOTOS))
        p.add_argument("--out", required=True)
        p.add_argument("--val-date", default="2016-03-15",
                       help="통째로 뺄 관찰일 — **양쪽 다 안 본 날**이어야 한다")
        p.add_argument("--legacy-labels",
                       default=str(nas.root() / "DolFinID" / "TrainingData"),
                       help="옛 YOLOv5 학습 라벨 뿌리 ('' 면 안 쓴다)")
        p.add_argument("--exclude-date", nargs="*", default=["2016-03-16"],
                       help="사람이 '채우지는' 않은 날 — 기본값의 근거는 위 문서")
        p.add_argument("--no-fin-db", action="store_true",
                       help="검토 판정을 안 읽는다 — 옛 운영 DB 만으로 만든다")
        p.add_argument("--val-frac", type=float, default=0.2)
        p.add_argument("--seed", type=int, default=20260817)
        p.add_argument("--dry-run", action="store_true")

    def _from_reviews(self, excl):
        """`fin.db` 의 사람 판정 → (더할 상자, 뺄 옛 상자, 쓸 사진, 적을 말).

        좌표는 **원본 사진 좌표**다 (`Box` 가 그렇게 들고 있다) — 크롭 좌표로
        바꾸는 일은 없다. 옛 DB 와 잇는 열쇠는 `Image.src_id`·`Box.src_id` 다.
        """
        from finseg.models import Box, Run

        note = []
        # **훑은 날만 쓴다.** 새 검출기가 안 지나간 날은 검출기 하나만 본
        # 것이라 완전성이 한 단계 낮다. 날짜를 손으로 적지 않고 run 기록에서
        # 읽는다 — 손으로 적으면 다음에 더 훑고 나서 여기를 안 고친다.
        swept = set()
        for r in Run.objects.filter(kind="detect").order_by("-id"):
            for d in (r.params or {}).get("dates") or []:
                swept.add(str(d))
        if not swept:
            note.append("**`infer_boxes` 기록이 없다** — 새 검출기가 훑은 날을"
                        " 알 수 없어 `fin.db` 를 안 썼다.")
            self.swept = set()
            return defaultdict(list), set(), set(), note
        swept -= excl

        # 판정은 쌓이므로 **상자마다 가장 늦은 것**이 유효하다
        # (`rules.effective_review` 와 같은 규칙 — 여기서는 한 번에 훑는다)
        latest = {}
        for box_id, cls in (Review.objects.order_by("id")
                            .values_list("box_id", "cls")):
            latest[box_id] = cls

        add, reject, imgs = defaultdict(list), set(), set()
        n_seen = n_unreviewed = 0
        for b in Box.objects.select_related("image").filter(
                image__obsdate__in=sorted(swept), image__src_id__isnull=False):
            cls = latest.get(b.id)
            if cls is None:
                n_unreviewed += 1
                continue
            n_seen += 1
            imgs.add(b.image.src_id)
            if cls == "fin":
                # 옛 상자는 옛 DB 에서 그대로 나간다 — 여기서 더하면 두 번 된다
                if b.source != "yolov5":
                    add[b.image.src_id].append((b.x1, b.y1, b.x2, b.y2))
            elif b.src_id is not None:
                # **사람이 아니라고 한 옛 상자.** 옛 DB 에서는 `fin` 으로
                # 나가던 것이라, 빼는 것이 곧 틀린 라벨을 지우는 일이다
                reject.add(b.src_id)
        note.append(f"`fin.db` 판정 {n_seen:,} 건 · 사진 {len(imgs):,} 장"
                    f" (훑은 날 {len(swept)})")
        note.append(f"  더할 상자 **{sum(len(v) for v in add.values()):,}**"
                    f" (새 검출기가 찾아 사람이 인정한 것)")
        note.append(f"  뺄 옛 상자 **{len(reject):,}** (사람이 아니라고 한 것)")
        if n_unreviewed:
            note.append(f"  아직 판정이 없는 상자 {n_unreviewed:,} 개는 안 썼다")
        self.swept = swept
        return add, reject, imgs, note

    def handle(self, **o):
        from PIL import Image as PILImage
        w = self.stdout.write
        src = Path(o["src_db"])
        if not src.exists():
            raise CommandError(f"옛 DB 가 없다: {src}")
        # **읽기 전용으로만 연다** — 운영 자료의 사본이고 우리가 쓸 일이 없다
        c = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
        q = lambda s: c.execute(s).fetchall()

        excl = set(o["exclude_date"] or [])
        touched = set(x[0] for x in q(
            "select distinct dolfin_image_id from dolfinweb_userfinbox"
            " where dolfin_image_id is not null"))
        touched |= set(x[0] for x in q(
            "select distinct dolfin_image_id from dolfinrest_dolfinbox"
            " where created_by!='yolov5'"))
        if not touched:
            raise CommandError("사람이 손댄 사진이 없다")
        not_fin = set(x[0] for x in q(
            "select dolfin_box_id from dolfinweb_userfinbox"
            " where not_fin=1 and dolfin_box_id is not null"))

        # **검토 판정을 읽는다.** 여기가 루프에서 끊겨 있던 자리다 — 이것이
        # 없으면 검토 화면에서 인정한 상자가 다음 학습에 안 들어간다.
        touched_old = set(touched)
        self.swept = set()
        fin_add, fin_reject, fin_photos, fin_note = defaultdict(list), set(), set(), []
        if not o["no_fin_db"]:
            fin_add, fin_reject, fin_photos, fin_note = self._from_reviews(excl)
            not_fin |= fin_reject
            touched |= fin_photos
        ph = ",".join(map(str, touched))

        imgs = {i: (f, str(d)) for i, f, d in q(
            f"select id, imagefile, obsdate from dolfinrest_dolfinimage"
            f" where id in ({ph})")}
        photos = Path(o["photos"])

        per_img, dropped = defaultdict(list), Counter()
        for bid, iid, by, coords in q(
                f"select id, dolfin_image_id, created_by, coords_str"
                f" from dolfinrest_dolfinbox where dolfin_image_id in ({ph})"):
            if iid not in imgs:
                continue
            if imgs[iid][1] in excl:
                dropped["사람이 채우지 않은 날 (--exclude-date)"] += 1
                continue
            if bid in not_fin:
                dropped["사람이 '지느러미 아님' 이라 한 것"] += 1
                continue
            try:
                x1, y1, x2, y2 = (int(round(float(v))) for v in coords.split(","))
            except Exception:
                dropped["좌표를 못 읽었다"] += 1
                continue
            if x2 <= x1 or y2 <= y1:
                dropped["넓이가 없다"] += 1
                continue
            per_img[iid].append((x1, y1, x2, y2, by != "yolov5"))

        # **검토가 인정한 상자를 얹는다.** 옛 검출기가 놓쳐 아무 라벨에도 없던
        # 자리다 — `infer_boxes` 가 옛 DB 와 견줘 걸렀으므로 겹치지 않는다.
        n_add = 0
        for iid, boxes in fin_add.items():
            if iid not in imgs or imgs[iid][1] in excl:
                continue
            for x1, y1, x2, y2 in boxes:
                per_img[iid].append((x1, y1, x2, y2, True))
                n_add += 1
        if fin_note:
            for line in fin_note:
                w(line.replace("**", ""))
            w(f"  실제로 얹은 것 {n_add:,} 개")

        # **옛 학습 라벨을 잇는다.** 이미 YOLO 형식이라 그대로 옮겨 적는다.
        legacy = {}
        lab_root = Path(o["legacy_labels"]) if o["legacy_labels"] else None
        if lab_root and lab_root.exists():
            import os as _os
            bad_cls = 0
            for folder in sorted(_os.listdir(lab_root)):
                d = lab_root / folder
                if not d.is_dir() or folder == "img_test":
                    continue
                pre = folder.split("_")[0]
                if len(pre) != 8 or not pre.isdigit():
                    continue
                day = f"{pre[:4]}-{pre[4:6]}-{pre[6:]}"
                if day in excl:
                    continue
                rows = q(f"select id, filename, imagefile from"
                         f" dolfinrest_dolfinimage where obsdate='{day}'")
                byname = {}
                for iid, fn, imf in rows:
                    byname.setdefault(Path(fn).stem, (iid, imf))
                for f in _os.listdir(d):
                    if not f.endswith(".txt"):
                        continue
                    hit = byname.get(f[:-4])
                    if hit is None:
                        dropped["옛 라벨인데 DB 에 그 사진이 없다"] += 1
                        continue
                    iid, imf = hit
                    if iid in per_img:
                        dropped["옛 라벨인데 사람이 손댄 쪽을 쓴다"] += 1
                        continue
                    txt = (d / f).read_text().strip()
                    lines = [ln for ln in txt.splitlines() if ln.strip()]
                    if any(ln.split()[0] != "0" for ln in lines):
                        bad_cls += 1
                        continue
                    legacy[iid] = (imf, day, lines)
            if bad_cls:
                w(f"  ** 분류가 0 이 아닌 옛 라벨 {bad_cls} 장은 뺐다")

        kept = []
        for iid, boxes in per_img.items():
            rel, day = imgs[iid]
            p = photos / rel
            if not p.exists():
                dropped["사진 파일이 없다"] += 1
                continue
            kept.append({"id": iid, "rel": rel, "path": p, "day": day,
                         "boxes": boxes, "lines": None})
        for iid, (rel, day, lines) in legacy.items():
            p = photos / rel
            if not p.exists():
                dropped["옛 라벨인데 사진 파일이 없다"] += 1
                continue
            kept.append({"id": iid, "rel": rel, "path": p, "day": day,
                         "boxes": [], "lines": lines})
        if not kept:
            raise CommandError("내보낼 것이 없다")

        n_box = sum(len(k["boxes"]) or len(k["lines"] or []) for k in kept)
        n_hum = sum(1 for k in kept for b in k["boxes"] if b[4])
        n_leg = sum(1 for k in kept if k["lines"] is not None)
        w(f"\n사람이 손댄 사진 {len(imgs):,} 장 중 쓸 수 있는 것 {len(kept):,}"
          + (f" (그중 fin.db 가 들여온 사진 {len(fin_photos - touched_old):,})"
             if fin_photos else ""))
        for k, v in dropped.most_common():
            w(f"  뺐다 — {k}: {v:,}")
        w(f"상자 {n_box:,} 개 · 사진당 {n_box / len(kept):.2f}")
        w(f"  사람이 손댄 사진 {len(kept) - n_leg:,} 장 (그중 **사람이 그린 상자"
          f" {n_hum:,}개**) · 옛 학습 라벨 {n_leg:,} 장")
        by_date = defaultdict(list)
        for k in kept:
            by_date[k["day"]].append(k)
        w("  관찰일별: " + " · ".join(
            f"{d} {len(v)}장" for d, v in sorted(by_date.items())))

        # **상자가 하나도 없는 사진.** 사람이 그 사진의 상자를 전부 아니라고 한
        # 것이라 순수한 배경이다 — 어려운 음성이라 값이 있지만, 그것이 자료의
        # 몇 %인지는 눈에 보여야 한다 (권장 10% 언저리)
        n_bg = sum(1 for k in kept if not k["boxes"] and not (k["lines"] or []))
        if n_bg:
            w(f"  상자가 없는 사진 {n_bg:,} 장 ({100 * n_bg / len(kept):.1f}%)"
              f" — 사람이 그 사진의 상자를 전부 아니라고 한 것이다")

        rng = random.Random(o["seed"])
        val_date = o["val_date"] or rng.choice(sorted(by_date))
        if val_date not in by_date:
            raise CommandError(f"그런 관찰일이 없다: {val_date}")
        split = {}
        for d, items in sorted(by_date.items()):
            if d == val_date:
                split.update({k["id"]: "val_date" for k in items})
                continue
            items = sorted(items, key=lambda k: k["id"])
            rng.shuffle(items)
            n_val = int(round(len(items) * o["val_frac"]))
            for i, k in enumerate(items):
                split[k["id"]] = "val" if i < n_val else "train"
        counts = Counter(split.values())
        w(f"\nval_date = {val_date} (통째로 뺀다)")
        # **이 자가 무엇을 재나.** `val_date` 가 새 검출기로 훑은 날이면, 그날의
        # 정답에 **그 검출기가 찾아낸 상자가 들어가 있다.** 사람이 인정한 것이라
        # 정답이 맞기는 하지만, 그것으로 **그 검출기의 재현율을 재면 자기 답을
        # 채점받는 것**이다 — 이 저장소에서 네 번 걸린 바로 그 함정이다.
        # 다음 검출기를 재는 데는 아무 문제가 없다.
        if val_date in self.swept:
            w(f"  ** 경고: {val_date} 는 infer_boxes 가 훑은 날이다. 그날의"
              f" 정답에 지금 검출기가 찾은 상자가 섞여 있으므로,")
            w(f"     **그 검출기의** 재현율을 여기서 읽으면 높게 나온다."
              f" 다음 검출기를 재는 데는 문제가 없다.")
        w("  " + " · ".join(f"{s} {counts[s]:,}"
                            for s in ("train", "val", "val_date")))
        if o["dry_run"]:
            w("\n--dry-run 이라 아무것도 쓰지 않았다.")
            return

        out = Path(o["out"]).resolve()
        for s in ("train", "val", "val_date"):
            (out / "images" / s).mkdir(parents=True, exist_ok=True)
            (out / "labels" / s).mkdir(parents=True, exist_ok=True)
        run = runs.start("export", params={
            "task": "detect", "out": str(out), "src_db": str(src),
            "val_date": val_date, "exclude_date": sorted(excl),
            "val_frac": o["val_frac"], "seed": o["seed"]})
        sizes = []
        for k in kept:
            s = split[k["id"]]
            name = f"{k['id']:08d}"
            link = out / "images" / s / f"{name}.jpg"
            if not link.exists():
                os.symlink(k["path"].resolve(), link)
            if k["lines"] is not None:
                # 옛 라벨은 이미 정규화된 YOLO 형식이다 — 그대로 옮긴다
                (out / "labels" / s / f"{name}.txt").write_text(
                    "".join(f"{ln}\n" for ln in k["lines"]))
                continue
            with PILImage.open(k["path"]) as im:
                W, H = im.size          # 헤더만 읽는다 — 화소를 안 푼다
            lines = []
            for x1, y1, x2, y2, _ in k["boxes"]:
                cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
                bw, bh = (x2 - x1) / W, (y2 - y1) / H
                sizes.append((x2 - x1) * (y2 - y1))
                lines.append(f"0 {min(max(cx,0),1):.6f} {min(max(cy,0),1):.6f} "
                             f"{min(bw,1):.6f} {min(bh,1):.6f}")
            (out / "labels" / s / f"{name}.txt").write_text(
                "".join(f"{ln}\n" for ln in lines))

        (out / "data.yaml").write_text(
            "path: .\ntrain: images/train\nval: images/val\n"
            "# 통째로 뺀 관찰일 — 성적은 이쪽으로 읽는다.\n"
            "# `test` 로도 적어야 ultralytics 가 라벨을 찾는다\n"
            "test: images/val_date\nval_date: images/val_date\n\n"
            "names:\n  0: fin\n")
        sizes.sort() if sizes else sizes.append(0)
        (out / "MANIFEST.json").write_text(json.dumps({
            "git_sha": runs.git_sha(), "export_run": run.id, "task": "detect",
            "src_db": str(src), "val_date": val_date,
            "exclude_date": sorted(excl), "names": NAMES,
            # **자료가 무엇으로 만들어졌는지 여기 남는다.** `fin.db` 몫을
            # 안 적으면 다음 사람이 "옛 DB 만 읽는다" 는 옛 문서를 믿는다
            "fin_db": {"used": not o["no_fin_db"],
                       "added_boxes": n_add, "rejected_old": len(fin_reject),
                       "photos": len(fin_photos - touched_old)},
            "counts": {**{s: counts[s] for s in ("train", "val", "val_date")},
                       "images": len(kept), "boxes": n_box,
                       "human_drawn": n_hum, "legacy_images": n_leg,
                       "excluded": dict(dropped)},
            "box_area_px": {"min": sizes[0], "p50": sizes[len(sizes) // 2],
                            "max": sizes[-1]},
            "obsdates": {d: len(v) for d, v in sorted(by_date.items())},
        }, ensure_ascii=False, indent=2))
        runs.finish(run)
        w(f"\nrun {run.id} · {out}")
        w(f"  상자 넓이(px²) 최소 {sizes[0]:,} · 중앙값 "
          f"{sizes[len(sizes)//2]:,} · 최대 {sizes[-1]:,}")
        w("  사진은 **링크**다 — 원본은 하나뿐이고 바뀌지 않는다")
        w(f"\n다음: manage.py train --data {out} --model yolo11s.pt --imgsz 1280")
