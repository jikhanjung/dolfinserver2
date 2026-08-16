"""검토를 YOLO-seg 학습 자료로 내보낸다.

    python manage.py export_yolo --out datasets/v1 --dry-run
    python manage.py export_yolo --out datasets/v1

판정 규칙은 여기 다시 쓰지 않는다 — `finseg.rules` 를 부른다. 검토 화면에 보이는
것이 곧 라벨이어야 하고, **갈라진 것은 눈에 띄지 않는다.**

## 크롭 하나가 학습 이미지 하나다

크롭은 상자마다 하나씩 있고, **그 안에 보이는 것은 전부 라벨이 된다** — 자기
상자의 것이든 이웃의 것이든. 폴리곤을 원본 좌표로 둔 것이 여기서 쓰인다.

## 검토 안 된 것이 하나라도 들면 그 크롭을 뺀다

이것이 자료의 신뢰를 지키는 유일한 줄이다. 라벨 없는 지느러미가 한 장이라도
들어가면 YOLO 는 **같은 모양을 어떤 크롭에서는 지느러미라 하고 어떤 크롭에서는
배경이라고** 배운다. 가려져 조각만 보이는 것은 **빼지 않는다** — 보이는 만큼
라벨한다 (`rules.py` 의 2026-08-16 항목).

`cls='none'` 은 다르다 — 사람이 "여기 없다" 고 말한 것이라 라벨 없이 두어도 된다.
**SAM2 가 실제로 헷갈린 자리에 붙은 음성이라 무작위 배경보다 값어치가 크다.**

## 검증을 둘로 둔다

- `val`      — 크롭 20%, 관찰일별 층화.  학습이 되고 있나
- `val_date` — 관찰일 하나를 통째로.     **다른 날에 듣나**

`val` 은 좋은데 `val_date` 가 나쁘면 그 날의 바다 상태·역광·배의 위치를 외운
것이고, 새 관찰일에는 안 듣는다는 뜻이다.
"""
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from finseg import geometry, rules, runs
from finseg.models import CLASSES, Box, Crop

MIN_VISIBLE = 0.10   # 크롭에 이만큼도 안 보이는 조각은 라벨로 세지 않는다
# 분류 순서가 곧 YOLO 의 클래스 번호다. `none` 은 배경이라 라벨이 없다.
NAMES = [c for c, _ in CLASSES if c != "none"]


def clip_to_crop(points, size):
    """폴리곤을 크롭 사각형 안으로 자른다 → (잘린 폴리곤, 보이는 비율).

    래스터로 자르고 윤곽을 다시 딴다 — 마스크를 만들 때와 **같은 방식**이라
    결과가 어긋나지 않는다.
    """
    import cv2
    import numpy as np
    pts = np.array([[int(round(x)), int(round(y))] for x, y in points], np.int32)
    if len(pts) < 3:
        return None, 0.0
    lo, hi = pts.min(0), pts.max(0)
    pad = 2
    off = np.array([pad - min(int(lo[0]), 0), pad - min(int(lo[1]), 0)])
    w = int(max(hi[0], size) - min(lo[0], 0)) + 2 * pad
    h = int(max(hi[1], size) - min(lo[1], 0)) + 2 * pad
    full = np.zeros((h, w), np.uint8)
    cv2.fillPoly(full, [pts + off], 1)
    total = int(full.sum())
    if not total:
        return None, 0.0
    keep = np.zeros_like(full)
    keep[off[1]:off[1] + size, off[0]:off[0] + size] = 1
    inside = full * keep
    vis = int(inside.sum())
    if not vis:
        return None, 0.0
    cnts, _ = cv2.findContours(inside, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0.0
    c = cv2.approxPolyDP(max(cnts, key=cv2.contourArea), 0.8, True)
    if len(c) < 3:
        return None, 0.0
    return [(float(p[0][0] - off[0]), float(p[0][1] - off[1])) for p in c], vis / total


def dedup_labels(labels, size, iou_min=0.85):
    """한 크롭 안에서 거의 같은 폴리곤을 하나로 → (남은 것, 지운 수).

    들일 때 중복 상자를 거르지만, 옛 검출기가 한 지느러미에 상자를 둘 붙여 놓은
    것이 이웃으로 딸려 들어올 수 있다. 같은 것을 두 번 적으면 겹친 개체 둘로
    배운다.
    """
    if len(labels) < 2:
        return labels, 0
    masks = [geometry.rasterize(p, size) for _, p in labels]
    keep, dropped = [], 0
    for i, m in enumerate(masks):
        if any((m & masks[j]).sum() / max(1, (m | masks[j]).sum()) >= iou_min
               for j in keep):
            dropped += 1
        else:
            keep.append(i)
    return [labels[i] for i in keep], dropped


class Command(BaseCommand):
    help = "검토를 YOLO-seg 자료로 내보낸다"

    def add_arguments(self, p):
        p.add_argument("--crops", default=str(settings.FIN_CROPS))
        p.add_argument("--out", required=True)
        p.add_argument("--val-frac", type=float, default=0.2)
        p.add_argument("--val-date", help="통째로 뺄 관찰일 (기본은 씨앗으로)")
        p.add_argument("--seed", type=int, default=20260815)
        p.add_argument("--dry-run", action="store_true")

    def handle(self, **o):
        w = self.stdout.write
        boxes = list(Box.objects.select_related("image")
                     .prefetch_related("masks", "reviews"))
        states = {b.id: rules.resolve(b) for b in boxes}
        per_image = defaultdict(list)
        for b in boxes:
            per_image[b.image_id].append(b)
        crops = {c.box_id: c for c in Crop.objects.all()}

        kept, dropped = [], Counter()
        slivers = dups = 0
        used_runs = set()
        for box in boxes:
            crop = crops.get(box.id)
            if crop is None:
                continue
            labels, blocked = [], False
            for other in per_image[box.image_id]:
                st = states[other.id]
                if st["label"] == rules.BACKGROUND:
                    continue        # 라벨을 안 쓰는 것으로 충분하다
                pts = rules.final_points(st, crop)
                if not pts:
                    # 마스크가 없는 상자. 이 크롭에 걸치면 아직 말할 수 없다.
                    if self._overlaps(other, crop):
                        blocked = True
                        break
                    continue
                clipped, vis = clip_to_crop(pts, crop.w)
                if clipped is None or vis < MIN_VISIBLE:
                    if clipped is not None:
                        slivers += 1
                    continue
                if st["label"] in (rules.PENDING, rules.DROP):
                    blocked = True
                    break
                labels.append((NAMES.index(st["cls"]), clipped))
                if st["mask"]:
                    used_runs.add(st["mask"].run_id)
            if blocked:
                dropped["검토 안 됐거나 양날이 가린 상자가 들어 있다"] += 1
                continue
            labels, d = dedup_labels(labels, crop.w)
            dups += d
            kept.append({"box_id": box.id, "crop": crop, "labels": labels,
                         "obsdate": str(box.image.obsdate)})

        w(f"크롭 {len(crops):,} 개 중 쓸 수 있는 것 {len(kept):,}")
        for k, v in dropped.items():
            w(f"  뺐다 — {k}: {v:,}")
        if slivers:
            w(f"  가장자리에 {MIN_VISIBLE:.0%} 미만만 보이는 조각 {slivers:,} 개는"
              f" 라벨로 세지 않았다")
        if dups:
            w(f"  한 크롭 안에서 겹친 라벨 {dups:,} 개를 하나로 줄였다")
        n_lab = sum(len(k["labels"]) for k in kept)
        n_empty = sum(1 for k in kept if not k["labels"])
        by_cls = Counter(NAMES[c] for k in kept for c, _ in k["labels"])
        w(f"라벨 {n_lab:,} 개 {dict(by_cls)} · 배경 크롭 {n_empty:,} 개")
        if not kept:
            raise CommandError("내보낼 것이 없다 — 먼저 검토할 것.")

        by_date = defaultdict(list)
        for k in kept:
            by_date[k["obsdate"]].append(k)
        rng = random.Random(o["seed"])
        val_date = o["val_date"] or rng.choice(sorted(by_date))
        if val_date not in by_date:
            raise CommandError(f"그런 관찰일이 없다: {val_date}")
        split = {}
        for d, items in sorted(by_date.items()):
            if d == val_date:
                split.update({k["box_id"]: "val_date" for k in items})
                continue
            items = sorted(items, key=lambda k: k["box_id"])
            rng.shuffle(items)
            n_val = int(round(len(items) * o["val_frac"]))
            for i, k in enumerate(items):
                split[k["box_id"]] = "val" if i < n_val else "train"
        counts = Counter(split.values())
        w(f"\nval_date = {val_date} (통째로 뺀다)")
        w("  " + " · ".join(f"{s} {counts[s]:,}"
                            for s in ("train", "val", "val_date")))
        if o["dry_run"]:
            w("\n--dry-run 이라 아무것도 쓰지 않았다.")
            return

        out = Path(o["out"])
        for s in ("train", "val", "val_date"):
            (out / "images" / s).mkdir(parents=True, exist_ok=True)
            (out / "labels" / s).mkdir(parents=True, exist_ok=True)
        run = runs.start("export", params={"out": str(out), "val_date": val_date,
                                           "val_frac": o["val_frac"],
                                           "seed": o["seed"]})
        crops_dir = Path(o["crops"])
        for k in kept:
            s = split[k["box_id"]]
            name = f"{k['box_id']:08d}"
            shutil.copyfile(crops_dir / k["crop"].path,
                            out / "images" / s / f"{name}.jpg")
            size = k["crop"].w
            lines = []
            for cid, poly in k["labels"]:
                flat = " ".join(f"{min(max(x / size, 0), 1):.6f} "
                                f"{min(max(y / size, 0), 1):.6f}" for x, y in poly)
                lines.append(f"{cid} {flat}")
            # 라벨이 없으면 **빈 파일**이다 — 지우는 것이 아니라. YOLO 는 그것을
            # 배경 이미지로 읽고, 배경을 10% 안팎 섞으라고 권한다.
            (out / "labels" / s / f"{name}.txt").write_text(
                "".join(f"{ln}\n" for ln in lines))

        # **경로는 상대로 적는다** — 절대경로면 옮긴 곳에서 안 맞는다. 다만
        # 학습은 꾸러미 **안에서** 실행해야 한다 (ultralytics 는 path: 를 실행
        # 디렉토리 기준으로 푼다).
        names = "\n".join(f"  {i}: {n}" for i, n in enumerate(NAMES))
        (out / "data.yaml").write_text(
            "path: .\ntrain: images/train\nval: images/val\n"
            "# 통째로 뺀 관찰일 — 성적은 이쪽으로 읽는다\n"
            f"val_date: images/val_date\n\nnames:\n{names}\n")
        (out / "MANIFEST.json").write_text(json.dumps({
            "git_sha": runs.git_sha(),
            "export_run": run.id,
            "mask_runs": sorted(used_runs),
            "val_date": val_date, "val_frac": o["val_frac"], "seed": o["seed"],
            "names": NAMES,
            "counts": {**{s: counts[s] for s in ("train", "val", "val_date")},
                       "labels": n_lab, "background_crops": n_empty,
                       "by_class": dict(by_cls)},
            "dropped": dict(dropped), "slivers_ignored": slivers,
            "duplicate_labels_merged": dups,
            "obsdates": sorted(by_date),
            # **어느 크롭을 썼는지 적는다.** val_date 가 학습에 섞였는지를
            # 나중에 물을 수 있는 유일한 수단이다.
            "boxes": {s: sorted(b for b, v in split.items() if v == s)
                      for s in ("train", "val", "val_date")},
        }, ensure_ascii=False, indent=2))
        runs.finish(run)
        w(f"\n{out} · MANIFEST.json 에 커밋 해시와 쓴 크롭 목록을 적었다")

    @staticmethod
    def _overlaps(box, crop):
        ox = min(box.x2, crop.x1) - max(box.x1, crop.x0)
        oy = min(box.y2, crop.y1) - max(box.y1, crop.y0)
        if ox <= 0 or oy <= 0:
            return False
        return ox * oy >= 0.3 * (box.x2 - box.x1) * (box.y2 - box.y1)
