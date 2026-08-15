"""학습한 YOLO-seg 로 후보 마스크를 만든다 — SAM2.1 자리에 그대로 낀다. **[GPU]**

    python manage.py infer --weights runs/fin-seg/weights/best.pt --redo

`segment` 와 **같은 표에 같은 모양으로** 넣는다 (`Mask`, 원본 좌표 폴리곤).
그래야 검토 화면도 내보내기도 엔진을 모른 채로 돌고, 두 엔진의 숫자에 같은 자를
댈 수 있다. 마스크는 쌓이므로 (`is_current`) 같은 상자 위에서 나란히 견줄 수 있다.

## 크롭 하나에 지느러미 하나

크롭은 "가운데 것" 이라는 약속으로 만들어졌고 학습도 그렇게 시켰다(`mosaic=0`).
YOLO 가 여럿을 내면 **프롬프트 상자와 가장 많이 겹치는 것 하나**를 고른다 —
가운데에서 가장 가까운 것이 아니라 상자와 겹치는 것이다. 이웃 지느러미가 크롭
가운데로 밀려 들어온 경우에 그 둘이 갈린다.
"""
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand
from django.db import transaction

from finseg import geometry, runs
from finseg.models import Crop, Mask


class Command(BaseCommand):
    help = "학습한 YOLO-seg 로 후보 마스크를 만든다 [GPU]"

    def add_arguments(self, p):
        p.add_argument("--crops", default=str(settings.FIN_CROPS))
        p.add_argument("--weights", required=True)
        p.add_argument("--conf", type=float, default=0.10,
                       help="낮게 뽑아 둔다 — 운영점은 DB 에서 다시 고를 수 있다")
        p.add_argument("--imgsz", type=int, default=640)
        p.add_argument("--device", default="0")
        p.add_argument("--limit", type=int)
        p.add_argument("--redo", action="store_true")
        p.add_argument("--note", default="")

    def handle(self, **o):
        import numpy as np
        from ultralytics import YOLO

        crops_dir = Path(o["crops"])
        qs = Crop.objects.select_related("box").order_by("box_id")
        if not o["redo"]:
            qs = qs.exclude(box__masks__is_current=True)
        rows = list(qs[:o["limit"]] if o["limit"] else qs)
        if not rows:
            self.stdout.write("할 것이 없다. (--redo 로 다시 할 수 있다)")
            return
        self.stdout.write(f"상자 {len(rows):,} 개 · {o['weights']}")

        model = YOLO(o["weights"])
        run = runs.start("yolo", model=o["weights"], note=o["note"],
                         params={"conf": o["conf"], "imgsz": o["imgsz"]})
        n_ok = n_none = 0
        for i, crop in enumerate(rows, 1):
            box = crop.box
            res = model.predict(str(crops_dir / crop.path), imgsz=o["imgsz"],
                                conf=o["conf"], device=o["device"],
                                verbose=False, retina_masks=True)[0]
            if res.masks is None or not len(res.masks.data):
                n_none += 1
                continue
            (px1, py1), (px2, py2) = geometry.to_crop(
                [(box.x1, box.y1), (box.x2, box.y2)], crop)
            data = res.masks.data.cpu().numpy() > 0.5
            confs = res.boxes.conf.cpu().numpy()
            prompt = np.zeros(data.shape[1:], bool)
            prompt[max(0, int(py1)):int(py2) + 1, max(0, int(px1)):int(px2) + 1] = True
            overlaps = [(m & prompt).sum() for m in data]
            k = int(np.argmax(overlaps))
            if not overlaps[k]:
                n_none += 1
                continue
            poly, area = geometry.mask_to_polygon(data[k])
            if poly is None:
                n_none += 1
                continue
            s = crop.scale
            with transaction.atomic():
                Mask.objects.filter(box=box, is_current=True).update(is_current=False)
                Mask.objects.create(
                    box=box, run=run,
                    polygon=geometry.dumps(geometry.to_orig(poly, crop)),
                    area=int(area / (s * s)), conf=round(float(confs[k]), 4))
            n_ok += 1
            if i % 200 == 0:
                self.stdout.write(f"  {i:,} / {len(rows):,}")
        runs.finish(run)
        self.stdout.write(f"마스크 {n_ok:,} 개"
                          + (f" · 아무것도 못 낸 것 {n_none:,}" if n_none else "")
                          + f"  (run {run.id})")
        if n_none:
            self.stdout.write("  ↑ 이것이 이 엔진의 재현율 손실이다. SAM2 는"
                              " 상자를 받으면 늘 무언가를 냈다 — 견줄 때 함께 적을 것.")
