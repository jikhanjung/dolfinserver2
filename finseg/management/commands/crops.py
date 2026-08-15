"""상자마다 정사각형 크롭을 잘라 둔다.

    python manage.py crops --jobs 8

## 왜 크롭인가

원본이 5472×3648 이고 지느러미 폭 중앙값이 103px 다. 통째로 넣으면 —
SAM2 의 이미지 인코더는 1024 로 줄이고 YOLO 는 `imgsz` 로 줄인다 — 지느러미가
20~25px 로 뭉개진다. 크롭을 640 으로 펴면 같은 지느러미가 300px 넘게 들어간다.

**SAM2.1 도 YOLO-seg 도 같은 크롭을 본다.** 그래야 학습한 모델을 SAM2.1 자리에
그대로 끼울 수 있고, 두 엔진의 숫자가 같은 자를 댄 것이 된다.

여유(`--pad`)는 상자 긴 변의 2배다. 지느러미만 딱 맞게 자르면 물과의 경계가
잘려 모델이 "어디까지가 지느러미인가" 를 배울 수 없고, 옛 YOLOv5 의 상자가 조금
어긋나 있을 때 지느러미 끝이 크롭 밖으로 나간다.
"""
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from finseg import runs
from finseg.geometry import crop_rect
from finseg.models import Box, Crop


def _one(job):
    """사진 한 장에 딸린 상자 전부를 **한 번의 디코딩으로** 처리한다.

    5MB 짜리 JPEG 를 푸는 데 0.3~0.5초 걸린다 — 상자마다 다시 열면 그만큼
    곱해진다.
    """
    from PIL import Image as PILImage
    rel, img_w, img_h, boxes, photos, crops_dir, out, pad = job
    try:
        with PILImage.open(Path(photos) / rel) as im:
            im = im.convert("RGB")
            done = []
            for bid, x1, y1, x2, y2 in boxes:
                r = crop_rect((x1, y1, x2, y2), img_w, img_h, pad)
                sub = im.crop(r).resize((out, out), PILImage.LANCZOS)
                # **상자 id 로 이름을 짓는다** — 사진 경로를 쓰면 같은 사진의
                # 여러 상자가 부딪히고, 순번을 쓰면 표본을 다시 뽑을 때 어긋난다.
                name = f"{bid:08d}.jpg"
                p = Path(crops_dir) / name[:3] / name   # 한 디렉토리에 몰지 않는다
                p.parent.mkdir(parents=True, exist_ok=True)
                sub.save(p, quality=92)
                done.append((bid, f"{name[:3]}/{name}", *r, out, out))
        return done, None
    except Exception as e:
        return [], f"{rel}: {e}"


class Command(BaseCommand):
    help = "상자마다 정사각형 크롭을 만든다"

    def add_arguments(self, p):
        p.add_argument("--photos", default=str(settings.FIN_PHOTOS))
        p.add_argument("--crops", default=str(settings.FIN_CROPS))
        p.add_argument("--out", type=int, default=640, help="크롭 한 변 (화소)")
        p.add_argument("--pad", type=float, default=2.0, help="상자 긴 변의 배수")
        p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 1))
        p.add_argument("--redo", action="store_true")
        p.add_argument("--limit", type=int)

    def handle(self, **o):
        photos, crops_dir = Path(o["photos"]), Path(o["crops"])
        crops_dir.mkdir(parents=True, exist_ok=True)
        qs = Box.objects.select_related("image").order_by("image__path", "id")
        if not o["redo"]:
            qs = qs.filter(crop__isnull=True)
        boxes = list(qs[:o["limit"]] if o["limit"] else qs)
        if not boxes:
            self.stdout.write("자를 것이 없다. (--redo 로 다시 자를 수 있다)")
            return

        by_img = {}
        for b in boxes:
            by_img.setdefault((b.image.path, b.image.width, b.image.height),
                              []).append((b.id, b.x1, b.y1, b.x2, b.y2))
        jobs = [(p, w, h, bs, str(photos), str(crops_dir), o["out"], o["pad"])
                for (p, w, h), bs in by_img.items()]
        self.stdout.write(f"사진 {len(jobs):,} 장 · 상자 {len(boxes):,} 개 "
                          f"· {o['jobs']} 갈래")

        run = runs.start("crop", params={"out": o["out"], "pad": o["pad"],
                                         "crops": str(crops_dir)})
        n, errs = 0, []
        with ProcessPoolExecutor(max_workers=o["jobs"]) as ex:
            for done, err in ex.map(_one, jobs, chunksize=8):
                if err:
                    errs.append(err)
                for bid, path, x0, y0, x1, y1, w, h in done:
                    Crop.objects.update_or_create(box_id=bid, defaults=dict(
                        path=path, x0=x0, y0=y0, x1=x1, y1=y1, w=w, h=h))
                    n += 1
                if n and n % 500 < len(done):
                    self.stdout.write(f"  {n:,} / {len(boxes):,}")
        runs.finish(run)
        self.stdout.write(f"크롭 {n:,} 개 → {crops_dir}")
        for e in errs[:10]:
            self.stdout.write(f"  못 함: {e}")
        if len(errs) > 10:
            self.stdout.write(f"  … 그 밖에 {len(errs) - 10} 건")
