"""SAM2.1 에 상자를 프롬프트로 넣어 후보 마스크를 만든다. **[GPU]**

    python manage.py segment                  # 마스크 없는 상자 전부
    python manage.py segment --limit 50       # 먼저 조금만
    python manage.py segment --device cpu     # GPU 없이 (느리다)

형제 프로젝트는 프롬프트 없는 자동 분할(AMG)로 시작해 재현율 50~60% 에서 막혔다.
여기는 **상자가 이미 있다.** 상자를 프롬프트로 주면 SAM2 는 "이 안의 것 하나" 를
따는 일만 하면 되고, 그것은 훨씬 쉬운 문제다. 실측으로 겹친 개체에서도 앞쪽
지느러미를 가림 경계에서 정확히 잘라 낸다.

**크롭을 본다, 원본을 보지 않는다.** SAM2 의 이미지 인코더는 무엇을 주든 1024 로
줄인다 — 5472px 원본을 주면 103px 지느러미가 19px 이 되어 윤곽이 뭉개진다.

마스크는 **원본 화소 좌표의 폴리곤**으로 저장한다. 크롭 좌표로 두면 크롭 여유를
바꾸는 순간 저장된 것이 전부 뜻을 잃는다.
"""
import os
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand
from django.db import transaction

from finseg import geometry, runs
from finseg.models import Crop, Mask

MODEL = os.environ.get("SAM2_MODEL", "facebook/sam2.1-hiera-base-plus")
# 폴리곤을 얼마나 단순하게 둘 것인가 (크롭 화소). 0.8 이면 640 크롭에서 눈에
# 안 보이는 정도이고, 점이 30~80 개로 떨어져 DB 와 라벨이 가벼워진다.
SIMPLIFY = 0.8


def autocast_dtype(device):
    """**2080ti 는 Turing(sm_75) 이라 bf16 이 없다.** 그때는 fp16 이어야 한다."""
    import torch
    if device != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 \
        else torch.float16


class Command(BaseCommand):
    help = "SAM2.1 박스 프롬프트로 후보 마스크를 만든다 [GPU]"

    def add_arguments(self, p):
        p.add_argument("--crops", default=str(settings.FIN_CROPS))
        p.add_argument("--model", default=MODEL)
        p.add_argument("--device", default=None, help="cuda | cpu")
        p.add_argument("--simplify", type=float, default=SIMPLIFY)
        p.add_argument("--limit", type=int)
        p.add_argument("--box", type=int, nargs="+", help="이 상자들만")
        p.add_argument("--redo", action="store_true")
        p.add_argument("--note", default="")

    def handle(self, **o):
        import numpy as np
        import torch
        from PIL import Image as PILImage
        from sam2.build_sam import build_sam2_hf
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        device = o["device"] or ("cuda" if torch.cuda.is_available() else "cpu")
        crops_dir = Path(o["crops"])
        qs = Crop.objects.select_related("box").order_by("box_id")
        if not o["redo"]:
            qs = qs.exclude(box__masks__is_current=True)
        if o["box"]:
            qs = qs.filter(box_id__in=o["box"])
        rows = list(qs[:o["limit"]] if o["limit"] else qs)
        if not rows:
            self.stdout.write("할 것이 없다. (--redo 로 다시 할 수 있다)")
            return
        self.stdout.write(f"상자 {len(rows):,} 개 · {device} · {o['model']}")

        sam = build_sam2_hf(o["model"], device=device)
        predictor = SAM2ImagePredictor(sam)
        dtype = autocast_dtype(device)
        run = runs.start("sam2", model=o["model"], note=o["note"],
                         params={"simplify": o["simplify"], "device": device})

        n_ok = n_fail = 0
        for i, crop in enumerate(rows, 1):
            box = crop.box
            img = np.array(PILImage.open(crops_dir / crop.path).convert("RGB"))
            (cx1, cy1), (cx2, cy2) = geometry.to_crop(
                [(box.x1, box.y1), (box.x2, box.y2)], crop)
            prompt = np.array([cx1, cy1, cx2, cy2], dtype=np.float32)

            ctx = (torch.autocast(device, dtype=dtype) if dtype
                   else torch.inference_mode())
            with torch.inference_mode(), ctx:
                predictor.set_image(img)
                masks, scores, _ = predictor.predict(box=prompt[None, :],
                                                     multimask_output=True)
            # **점수가 가장 높은 것 하나.** 상자가 이미 무엇을 딸지 정해 주었으므로
            # 여럿을 남겨 사람이 고르게 할 값어치가 없다.
            masks = masks.reshape(-1, masks.shape[-2], masks.shape[-1])
            scores = np.asarray(scores).reshape(-1)
            k = int(scores.argmax())
            poly, area = geometry.mask_to_polygon(masks[k] > 0, o["simplify"])
            if poly is None:
                n_fail += 1
                continue
            s = crop.scale
            with transaction.atomic():
                Mask.objects.filter(box=box, is_current=True).update(is_current=False)
                Mask.objects.create(
                    box=box, run=run,
                    polygon=geometry.dumps(geometry.to_orig(poly, crop)),
                    area=int(area / (s * s)), conf=round(float(scores[k]), 4))
            n_ok += 1
            if i % 100 == 0:
                self.stdout.write(f"  {i:,} / {len(rows):,}")
        runs.finish(run)
        self.stdout.write(f"마스크 {n_ok:,} 개"
                          + (f" · 못 딴 것 {n_fail:,}" if n_fail else "")
                          + f"  (run {run.id})")
