"""엔진 둘을 같은 자로 잰다 — SAM2.1 과 학습한 YOLO-seg.

    python manage.py eval_masks --runs 3 7
    python manage.py eval_masks --runs 3 7 --date 2019-06-17    # val_date 에서만

**정답은 사람이 채택한 마스크다.** 아래 직선까지 반영한 최종 폴리곤으로 잰다 —
화면에서 본 것이 곧 정답이어야 한다.

## 이 숫자에는 한정이 붙는다

정답이 **옛 YOLOv5 의 상자 위에서** 만들어졌다. 그 검출기가 아예 못 본 지느러미는
정답에도 없고 여기 계산에도 안 들어간다. 그러니 이것은 "지느러미를 얼마나 찾나"
가 아니라 **"주어진 상자 안의 윤곽을 얼마나 잘 따나"** 다.

**두 엔진에 같은 자를 대므로 견주기는 유효하다.** 절대값을 밖에 낼 때만 조심한다.
"""
from django.core.management.base import BaseCommand, CommandError

from finseg import geometry, rules
from finseg.models import Box, Crop, Mask, Run


class Command(BaseCommand):
    help = "마스크 run 들을 사람의 판정에 견준다"

    def add_arguments(self, p):
        p.add_argument("--runs", type=int, nargs="+", required=True)
        p.add_argument("--date", help="이 관찰일에서만 (보통 MANIFEST 의 val_date)")

    def handle(self, **o):
        import numpy as np
        qs = Box.objects.select_related("image").prefetch_related("masks", "reviews")
        if o["date"]:
            qs = qs.filter(image__obsdate=o["date"])
        crops = {c.box_id: c for c in Crop.objects.all()}

        truth = {}
        for box in qs:
            st = rules.resolve(box)
            if st["label"] != rules.POSITIVE or box.id not in crops:
                continue
            pts = rules.final_points(st, crops[box.id])
            if pts:
                truth[box.id] = pts
        if not truth:
            raise CommandError("정답이 없다 — 먼저 검토할 것"
                               + (f" ({o['date']})" if o["date"] else ""))
        w = self.stdout.write
        w(f"정답 {len(truth):,} 개" + (f" · 관찰일 {o['date']}" if o["date"] else "")
          + "\n")
        hdr = (f"{'run':>5} {'kind':<6} {'낸 것':>7} {'평균 IoU':>9} "
               f"{'≥0.5':>7} {'≥0.7':>7} {'≥0.9':>7}")
        w(hdr)
        w("-" * len(hdr))
        for rid in o["runs"]:
            run = Run.objects.filter(id=rid).first()
            if run is None:
                w(f"{rid:>5} — 그런 run 이 없다")
                continue
            got = {m.box_id: m for m in
                   Mask.objects.filter(run_id=rid, box_id__in=truth).order_by("id")}
            ious, produced = [], 0
            for box_id, tpts in truth.items():
                m = got.get(box_id)
                if m is None:
                    ious.append(0.0)      # 못 낸 것은 IoU 0 이다, 빼는 것이 아니라
                    continue
                produced += 1
                crop = crops[box_id]
                a = geometry.rasterize(
                    geometry.to_crop(geometry.loads(m.polygon), crop), crop.w)
                b = geometry.rasterize(tpts, crop.w)
                u = (a | b).sum()
                ious.append(float((a & b).sum() / u) if u else 0.0)
            v = np.array(ious)
            w(f"{rid:>5} {run.kind:<6} {produced:>7,} {v.mean():>9.3f} "
              f"{(v >= .5).mean():>7.1%} {(v >= .7).mean():>7.1%} "
              f"{(v >= .9).mean():>7.1%}")
            if produced < len(truth):
                w(f"      ↑ {len(truth) - produced:,} 개는 아무것도 못 냈다"
                  f" — IoU 0 으로 셌다")
        w("\n※ 정답이 옛 YOLOv5 상자 위에서 만들어졌다. 이 숫자는 '지느러미를"
          " 얼마나 찾나' 가 아니라 '상자 안의 윤곽을 얼마나 잘 따나' 다.")
