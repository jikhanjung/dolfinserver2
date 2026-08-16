"""엔진 둘을 같은 자로 잰다 — SAM2.1 과 학습한 YOLO-seg.

    python manage.py eval_masks --runs 3 7
    python manage.py eval_masks --runs 3 7 --date 2019-06-17    # val_date 에서만

**정답은 사람이 채택한 마스크다.** 아래 직선까지 반영한 최종 폴리곤으로 잰다 —
화면에서 본 것이 곧 정답이어야 한다.

## 이 숫자에는 한정이 붙는다

정답이 **옛 YOLOv5 의 상자 위에서** 만들어졌다. 그 검출기가 아예 못 본 지느러미는
정답에도 없고 여기 계산에도 안 들어간다. 그러니 이것은 "지느러미를 얼마나 찾나"
가 아니라 **"주어진 상자 안의 윤곽을 얼마나 잘 따나"** 다.

**두 엔진에 같은 자를 대므로 비교는 유효하다.** 절대값을 밖에 낼 때만 조심한다.
"""
from django.core.management.base import BaseCommand, CommandError

from finseg import evaluate, rules
from finseg.models import Box, Crop, Mask, Run


class Command(BaseCommand):
    help = "마스크 run 들을 사람의 판정에 비교한다"

    def add_arguments(self, p):
        p.add_argument("--runs", type=int, nargs="+", required=True)
        p.add_argument("--date", help="이 관찰일에서만 (보통 MANIFEST 의 val_date)")

    def handle(self, **o):
        qs = Box.objects.select_related("image").prefetch_related("masks", "reviews")
        if o["date"]:
            qs = qs.filter(image__obsdate=o["date"])
        crops = {c.box_id: c for c in Crop.objects.all()}

        # **계산은 `finseg.evaluate` 하나에만 있다** — 화면(`/compare`)이 같은
        # 숫자를 말해야 한다
        boxes = list(qs)
        truth = evaluate.truth_for(boxes, crops)
        states = {b.id: rules.resolve(b) for b in boxes if b.id in truth}
        indep = {b.id for b in boxes if b.id in truth and evaluate.independent(b)}
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
            ious = [evaluate.iou(got.get(b), t, crops[b], states[b])
                    for b, t in truth.items()]
            produced = sum(1 for b in truth if b in got)
            st = evaluate.score(ious)
            w(f"{rid:>5} {run.kind:<6} {produced:>7,} {st['mean']:>9.3f} "
              f"{st['p50']:>7.1%} {st['p70']:>7.1%} {st['p90']:>7.1%}")
            if indep:
                vi = [evaluate.iou(got.get(b), truth[b], crops[b], states[b])
                      for b in indep]
                si = evaluate.score(vi)
                w(f"{'':>5} {'└ 독립':<6} {len(indep):>7,} {si['mean']:>9.3f} "
                  f"{si['p50']:>7.1%} {si['p70']:>7.1%} {si['p90']:>7.1%}")
            if produced < len(truth):
                w(f"      ↑ {len(truth) - produced:,} 개는 아무것도 못 냈다"
                  f" — IoU 0 으로 셌다")
        w(f"\n※ '독립' 은 사람이 윗윤곽을 직접 다시 그린 {len(indep):,} 개다."
          " 나머지는 사람이 '통과' 를 누른 것이라 **정답이 곧 그때 현재였던"
          " 엔진의 출력**이고, 그 엔진에게는 자기 답을 채점하는 셈이 된다.")
        w("※ 정답이 옛 YOLOv5 상자 위에서 만들어졌다. 이 숫자는 '지느러미를"
          " 얼마나 찾나' 가 아니라 '상자 안의 윤곽을 얼마나 잘 따나' 다.")
