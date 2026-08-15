"""기존 `dolfinserver` 의 상자를 표본 추출해 들인다.

    python manage.py import_boxes --dry-run
    python manage.py import_boxes --boxes 2000

**옛 DB 는 읽기 전용으로만 연다.** 운영 중인 웹서버가 같은 파일을 쓰고 있고,
SQLite 는 쓰기가 하나다.

## 무엇을 고르나

- **면적 15000 초과분에서만.** 옛 웹 UI 의 문턱(`MINIMUM_BOX_SIZE`)과 같다.
  그보다 작은 것은 폭이 100px 아래라 윤곽을 딸 것이 별로 없고, 사람이 검토해도
  판단이 흔들린다
- **관찰일별 층화.** 자료가 한 날짜에 쏠리면 그 날의 바다 상태·역광·배의 위치를
  외운다 — 형제 프로젝트가 슬라이드 한 장에 56% 쏠려 겪은 문제이고 **여기서는
  처음부터 피할 수 있다**
- **크롭에 들어오는 이웃 상자도 함께.** 크롭은 상자의 2배라 옆 개체의 지느러미가
  함께 들어온다 (표본의 9.2%). 그 지느러미에 라벨이 없으면 **YOLO 는 그것을
  배경으로 배운다.** 그냥 버리면 무리 지어 있는 개체만 골라 빠지므로 — 정작
  어려운 쪽이다 — 함께 들여 검토한다. 비용은 16% 늘어나는 것뿐이다
  (사진의 상자를 통째로 들이면 2.2배가 된다)
- **거의 겹치는 중복 상자는 하나로.** 옛 DB 에 IoU 1.0 짜리 쌍이 남아 있다.
  그냥 두면 같은 크롭이 두 벌 생기고 한 지느러미에 폴리곤이 겹쳐 붙는다

나누기(train/val/val_date)는 여기서 하지 않는다. **내보낼 때 정한다** — 검토가
얼마나 진행됐는지에 따라 달라지고, 여기서 굳히면 다시 못 고친다.
"""
import random
import sqlite3
from collections import defaultdict

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from finseg import runs
from finseg.geometry import crop_rect
from finseg.models import Box, Image

MIN_AREA = 15000

# `UserFinbox` 는 상자 하나에 여러 사람의 것이 붙을 수 있어 표시는 MAX 로 모은다
# (한 사람이라도 "아니다" 라 했으면 눈여겨볼 값어치가 있다).
SELECT_BOX = """
    SELECT b.id, b.dolfin_image_id, b.coords_str, b.obsdate,
           i.imagefile, i.exifdatetime,
           COALESCE(NULLIF(NULLIF(b.boxname, ''), 'None'), u.boxname, '') AS boxname,
           COALESCE(u.not_fin, 0)          AS not_fin,
           COALESCE(u.not_identifiable, 0) AS not_identifiable
      FROM dolfinrest_dolfinbox b
      JOIN dolfinrest_dolfinimage i ON i.id = b.dolfin_image_id
      LEFT JOIN (
          SELECT dolfin_box_id,
                 MAX(NULLIF(boxname, '')) AS boxname,
                 MAX(not_fin)             AS not_fin,
                 MAX(not_identifiable)    AS not_identifiable
            FROM dolfinweb_userfinbox
           WHERE dolfin_box_id IS NOT NULL
           GROUP BY dolfin_box_id
      ) u ON u.dolfin_box_id = b.id
"""


def bbox_of(coords_str):
    """`coords_str` 에서 상자를 뽑는다. 폴리곤이면 그 외접 상자다."""
    try:
        v = [int(round(float(x))) for x in coords_str.split(",")]
    except (ValueError, AttributeError):
        return None
    if len(v) < 4 or len(v) % 2:
        return None
    xs, ys = v[0::2], v[1::2]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def spread(items, n):
    """정렬된 것에서 n 개를 고르게 집는다 (앞뒤 끝을 포함한다)."""
    if n >= len(items):
        return list(items)
    if n == 1:
        return [items[len(items) // 2]]
    step = (len(items) - 1) / (n - 1)
    return [items[int(round(i * step))] for i in range(n)]


def iou(a, b):
    ox = min(a[2], b[2]) - max(a[0], b[0])
    oy = min(a[3], b[3]) - max(a[1], b[1])
    if ox <= 0 or oy <= 0:
        return 0.0
    inter = ox * oy
    ua = (a[2] - a[0]) * (a[3] - a[1])
    ub = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (ua + ub - inter)


class Command(BaseCommand):
    help = "옛 dolfinserver DB 에서 상자를 표본 추출해 들인다"

    def add_arguments(self, p):
        p.add_argument("--src", default=str(settings.FIN_SRC_DB))
        p.add_argument("--photos", default=str(settings.FIN_PHOTOS))
        p.add_argument("--boxes", type=int, default=2000, help="들일 상자 수")
        p.add_argument("--dates", type=int, default=20, help="쓸 관찰일 수")
        p.add_argument("--min-area", type=int, default=MIN_AREA)
        p.add_argument("--seed", type=int, default=20260815,
                       help="표본을 다시 뽑을 수 있게 씨앗을 적어 둔다")
        p.add_argument("--pad", type=float, default=2.0,
                       help="크롭 여유. crops 의 --pad 와 같아야 한다")
        p.add_argument("--margin", type=float, default=1.25,
                       help="이웃을 찾을 때 크롭보다 더 넓게 보는 배수")
        p.add_argument("--no-neighbors", action="store_true")
        p.add_argument("--dry-run", action="store_true")

    def handle(self, **o):
        from pathlib import Path
        src_path, photos = Path(o["src"]), Path(o["photos"])
        if not src_path.exists():
            raise CommandError(f"옛 DB 가 없다: {src_path}")
        src = sqlite3.connect(f"file:{src_path}?mode=ro", uri=True)
        src.row_factory = sqlite3.Row
        w = self.stdout.write

        w(f"옛 DB   {src_path}\n사진    {photos}")
        by_date = defaultdict(list)
        for r in src.execute(SELECT_BOX +
                             " WHERE b.coords_str != '' AND b.obsdate IS NOT NULL"):
            bb = bbox_of(r["coords_str"])
            if bb is None:
                continue
            area = (bb[2] - bb[0]) * (bb[3] - bb[1])
            if area >= o["min_area"]:
                by_date[r["obsdate"]].append((r, bb, area))
        total = sum(len(v) for v in by_date.values())
        w(f"면적 > {o['min_area']:,} 인 상자 {total:,} 개 / 관찰일 {len(by_date)} 일")

        per_date = max(1, o["boxes"] // o["dates"])
        usable = sorted(d for d, v in by_date.items() if len(v) >= per_date)
        if not usable:
            raise CommandError("쓸 만한 관찰일이 없다 — --boxes 나 --dates 를 줄일 것")
        picked = spread(usable, o["dates"])
        w(f"관찰일 {len(picked)} 일 선택 ({picked[0]} ~ {picked[-1]}), "
          f"날마다 {per_date} 개")

        rng = random.Random(o["seed"])
        chosen = []
        for d in picked:
            pool = sorted(by_date[d], key=lambda t: t[0]["id"])
            chosen.extend(rng.sample(pool, min(per_date, len(pool))))
        w(f"뽑힌 상자 {len(chosen):,} 개")

        if not o["no_neighbors"]:
            chosen, added = self._neighbors(src, chosen, o["pad"] * o["margin"])
            w(f"크롭에 들어오는 이웃 {added:,} 개를 함께 들인다 "
              f"→ 모두 {len(chosen):,} 개")

        chosen, dropped = self._dedup(chosen)
        if dropped:
            w(f"거의 겹치는 중복 상자 {dropped:,} 개를 걸렀다 → {len(chosen):,} 개")

        if o["dry_run"]:
            cnt = defaultdict(int)
            for r, _, _ in chosen:
                cnt[r["obsdate"]] += 1
            w("\n관찰일별:")
            for d in picked:
                w(f"  {d}  {cnt[d]:>4}  (후보 {len(by_date[d]):,})")
            w("\n--dry-run 이라 아무것도 쓰지 않았다.")
            return

        self._write(chosen, photos, o)

    # -- 이웃 폐포 -----------------------------------------------------------
    def _neighbors(self, src, chosen, pad, min_overlap=0.3):
        """크롭에 들어오는 이웃을 폐포로 끌어온다. 사진 안의 상자 수로 막혀 있다.

        `pad` 는 크롭보다 넉넉히 잡는다 — 나중에 크롭 여유를 조금 키워도 이웃
        목록이 어긋나지 않게. 어긋나면 라벨 없는 지느러미가 조용히 들어간다.
        """
        by_img = defaultdict(list)
        for item in chosen:
            by_img[item[0]["dolfin_image_id"]].append(item)
        pool = {}
        for sid in by_img:
            rows = []
            for r in src.execute(SELECT_BOX + " WHERE b.dolfin_image_id = ?"
                                              " AND b.coords_str != ''", (sid,)):
                bb = bbox_of(r["coords_str"])
                if bb:
                    rows.append((r, bb, (bb[2] - bb[0]) * (bb[3] - bb[1])))
            pool[sid] = rows

        added = 0
        for sid, seeds in by_img.items():
            have = {r["id"] for r, _, _ in seeds}
            queue = list(seeds)
            while queue:
                _, bb, _ = queue.pop()
                big = 10 ** 6          # 사진 크기를 모르니 자르지 않는 쪽으로
                c = crop_rect(bb, big, big, pad)
                for cand, cbb, carea in pool[sid]:
                    if cand["id"] in have:
                        continue
                    ox = min(cbb[2], c[2]) - max(cbb[0], c[0])
                    oy = min(cbb[3], c[3]) - max(cbb[1], c[1])
                    if ox <= 0 or oy <= 0:
                        continue
                    if ox * oy < min_overlap * (cbb[2] - cbb[0]) * (cbb[3] - cbb[1]):
                        continue
                    have.add(cand["id"])
                    item = (cand, cbb, carea)
                    queue.append(item)
                    by_img[sid].append(item)
                    added += 1
        return [it for v in by_img.values() for it in v], added

    def _dedup(self, chosen, iou_min=0.9):
        by_img = defaultdict(list)
        for item in chosen:
            by_img[item[0]["dolfin_image_id"]].append(item)
        out, removed = [], 0
        for items in by_img.values():
            keep = []
            for r, bb, area in sorted(items, key=lambda t: t[0]["id"]):
                if any(iou(bb, kbb) >= iou_min for _, kbb, _ in keep):
                    removed += 1
                else:
                    keep.append((r, bb, area))
            out.extend(keep)
        return out, removed

    # -- 쓰기 ----------------------------------------------------------------
    @transaction.atomic
    def _write(self, chosen, photos, o):
        from PIL import Image as PILImage      # 헤더만 읽는다
        run = runs.start("import", model=str(o["src"]), params={
            k: o[k] for k in ("boxes", "dates", "min_area", "seed", "pad", "margin")})
        n_img = n_box = n_missing = 0
        seen = {}
        for r, bb, area in chosen:
            rel = r["imagefile"]
            if not rel:
                n_missing += 1
                continue
            img = seen.get(rel)
            if img is None:
                full = photos / rel
                if not full.exists():
                    n_missing += 1
                    continue
                try:
                    with PILImage.open(full) as im:
                        size = im.size
                except Exception as e:
                    self.stdout.write(f"  못 읽음 {rel}: {e}")
                    n_missing += 1
                    continue
                img, created = Image.objects.get_or_create(
                    path=rel, defaults=dict(
                        src_id=r["dolfin_image_id"], obsdate=r["obsdate"],
                        exifdatetime=r["exifdatetime"] or None,
                        width=size[0], height=size[1]))
                n_img += int(created)
                seen[rel] = img
            _, created = Box.objects.get_or_create(src_id=r["id"], defaults=dict(
                image=img, x1=bb[0], y1=bb[1], x2=bb[2], y2=bb[3], area=area,
                source="yolov5", boxname=r["boxname"] or "",
                src_not_fin=bool(r["not_fin"]),
                src_not_identifiable=bool(r["not_identifiable"])))
            n_box += int(created)
        runs.finish(run)
        self.stdout.write(f"\n들임: 사진 {n_img:,} · 상자 {n_box:,}"
                          + (f" · 사진 없어 건너뜀 {n_missing:,}" if n_missing else ""))
        self.stdout.write(f"run {run.id}")
