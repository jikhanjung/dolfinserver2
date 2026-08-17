"""학습한 키포인트 모델로 **밑동 두 점을 제안한다.** [GPU]

    python manage.py infer_base --weights runs/pose-v1/weights/best.pt
    python manage.py infer_base --weights ... --redo   # 이미 채운 것도 다시

`Mask.base_line` 에 넣는다. 그러면 **화면 코드를 하나도 안 건드려도** 제안이
바뀐다 — `rules.resolve` 가 이미 "사람이 그은 것이 있으면 그것, 없으면 마스크의
것" 으로 고르게 되어 있기 때문이다(`review.base_line or mask.base_line`).
그 자리는 처음부터 삽입점 탐지기가 채우라고 비워 둔 칸이고, 두 번의 실패 뒤
비어 있었다 (`finseg/baseline.py`).

## 사람이 그은 것은 절대 안 건드린다

이 명령은 `Mask` 만 쓴다. `Review.base_line` 은 손대지 않으므로, 이미 검토한
상자는 화면에서든 내보내기에서든 **사람이 찍은 두 점을 그대로 쓴다.** 제안이
바뀌는 것은 아직 안 본 상자뿐이다.

## 여럿을 내면 프롬프트 상자와 겹치는 것을 고른다

`infer` 와 같은 규칙이다. 크롭은 "가운데 것 하나" 라는 약속이지만 이웃 지느러미가
걸쳐 들어오는 크롭이 20%쯤 되고, 그럴 때 **가운데에서 가장 가까운 것**이 아니라
**프롬프트 상자와 가장 많이 겹치는 것**이 옳다.

## 두 점의 순서

`export_pose` 가 **왼쪽 점을 늘 0번**으로 학습시켰으므로 여기서도 x 로 정렬해
넣는다. 화면의 두 손잡이도 왼쪽·오른쪽이라 그대로 맞는다.
"""
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from finseg import geometry, runs
from finseg.models import Crop, Mask


class Command(BaseCommand):
    help = "키포인트 모델로 밑동 두 점을 제안한다 [GPU]"

    def add_arguments(self, p):
        p.add_argument("--crops", default=str(settings.FIN_CROPS))
        p.add_argument("--weights", required=True)
        p.add_argument("--conf", type=float, default=0.10)
        p.add_argument("--imgsz", type=int, default=640)
        p.add_argument("--device", default="0")
        p.add_argument("--batch", type=int, default=16)
        p.add_argument("--limit", type=int)
        p.add_argument("--redo", action="store_true",
                       help="이미 base_line 이 있는 마스크도 다시 채운다")
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--note", default="")

    def handle(self, **o):
        w = self.stdout.write
        qs = Mask.objects.filter(is_current=True).select_related("box")
        if not o["redo"]:
            qs = qs.filter(base_line="")
        masks = list(qs.order_by("box_id"))
        crops = {c.box_id: c for c in Crop.objects.filter(
            box_id__in=[m.box_id for m in masks])}
        todo = [m for m in masks if m.box_id in crops]
        if o["limit"]:
            todo = todo[:o["limit"]]
        w(f"밑동을 채울 마스크 {len(todo):,} 개"
          f" ({'전부 다시' if o['redo'] else '아직 빈 것만'})")
        if not todo:
            w("할 것이 없다.")
            return
        if o["dry_run"]:
            w("--dry-run 이라 아무것도 쓰지 않았다.")
            return

        crops_dir = Path(o["crops"])
        missing = [m for m in todo
                   if not (crops_dir / crops[m.box_id].path).exists()]
        if missing:
            raise CommandError(f"크롭 파일이 없다: {len(missing)} 개"
                               f" (첫 번째: {crops[missing[0].box_id].path})")

        from ultralytics import YOLO
        model = YOLO(o["weights"])
        run = runs.start("yolo", model=o["weights"], params={
            "kind": "base", "conf": o["conf"], "imgsz": o["imgsz"],
            "redo": o["redo"]}, note=o["note"])

        filled = empty = 0
        for i in range(0, len(todo), o["batch"]):
            chunk = todo[i:i + o["batch"]]
            paths = [str(crops_dir / crops[m.box_id].path) for m in chunk]
            results = model.predict(paths, imgsz=o["imgsz"], conf=o["conf"],
                                    device=o["device"], verbose=False)
            hits = []
            for m, res in zip(chunk, results):
                crop = crops[m.box_id]
                kp = self._pick(res, m.box, crop)
                if kp is None:
                    empty += 1
                    continue
                m.base_line = geometry.dumps(geometry.to_orig(kp, crop))
                m.base_run = run
                hits.append(m)
            with transaction.atomic():
                Mask.objects.bulk_update(hits, ["base_line", "base_run"])
            filled += len(hits)
            w(f"  {min(i + o['batch'], len(todo)):>5}/{len(todo)}"
              f"  채움 {filled:,} · 못 찾음 {empty:,}", ending="\r")
        w("")
        runs.finish(run)
        w(f"run {run.id} · 밑동을 채운 마스크 {filled:,} 개"
          f" · 모델이 못 찾은 것 {empty:,} 개")
        w("사람이 그은 것은 안 건드렸다 — 이미 검토한 상자는 그대로다.")

    def _pick(self, res, box, crop):
        """여럿 중 프롬프트 상자와 가장 많이 겹치는 것의 두 점 (크롭 좌표).

        `box` 는 `Box` 모델이다 — 좌표 넷은 `.box` 로 꺼낸다.
        """
        kpts = getattr(res, "keypoints", None)
        if kpts is None or kpts.xy is None or len(kpts.xy) == 0:
            return None
        x1, y1, x2, y2 = box.box
        (px1, py1), (px2, py2) = geometry.to_crop([(x1, y1), (x2, y2)], crop)
        best, best_area = None, -1.0
        for j, b in enumerate(res.boxes.xyxy.tolist()):
            ix = max(0.0, min(b[2], px2) - max(b[0], px1))
            iy = max(0.0, min(b[3], py2) - max(b[1], py1))
            if ix * iy > best_area:
                best, best_area = j, ix * iy
        pts = kpts.xy[best].tolist()
        if len(pts) != 2:
            return None
        return sorted((float(x), float(y)) for x, y in pts)
