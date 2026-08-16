"""마스크 한 벌을 **현재로 올린다** — 엔진을 갈아 끼우는 자리.

    python manage.py promote --run 16 --dry-run
    python manage.py promote --run 16

`infer --compare-only` 로 나란히 쌓아 두고 `eval_masks` 로 재 본 뒤, 쓸 만하면
그것을 현재로 올린다. **비교와 교체를 갈라 둔 이유**가 이것이다 — 재 보기도
전에 현재를 넘기면 되돌릴 때 무엇이 원래였는지 알기 어렵다.

## 무엇이 따라 바뀌나

`rules.resolve` 는 현재 마스크 하나만 본다. 그래서 이 명령 하나로 검토 화면과
내보내기가 보는 것이 통째로 바뀐다. 함께 알아야 할 것이 셋이다.

**밑동 제안은 안 따라온다.** `base_line` 은 마스크에 붙어 있고 새 마스크는
비어 있다. 이어서 `infer_base --redo` 를 돌려야 제안이 다시 생긴다 — 안 그러면
검토 화면이 상자 아래 두 모서리로 돌아간다 (그 제안은 열에 아홉 틀린다).

**사람이 그은 것은 안 바뀐다.** `Review.polygon`·`base_line` 은 마스크보다
세고, `rules.resolve` 가 그것을 먼저 고른다.

**`verdict` 는 엔진에 딸린 판정이다.** "이 마스크가 맞다" 는 그때 그 마스크에
대한 말이었다 (`models.py`). 엔진이 바뀌면 다시 받아야 하고, `Review.mask_id`
가 어느 마스크를 보고 내린 판정인지 들고 있으므로 **검토 화면의 `엔진 바뀜`
대기열이 그것을 정확히 집어낸다.**

`cls` 는 다시 안 받아도 된다 — 상자 안에 무엇이 있나는 엔진과 무관하다.
"""
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from finseg import runs
from finseg.models import Mask, Review, Run


class Command(BaseCommand):
    help = "마스크 한 벌을 현재로 올린다 (엔진 교체)"

    def add_arguments(self, p):
        p.add_argument("--run", type=int, required=True)
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--note", default="")

    def handle(self, **o):
        w = self.stdout.write
        run = Run.objects.filter(id=o["run"]).first()
        if run is None:
            raise CommandError(f"그런 run 이 없다: {o['run']}")
        new = list(Mask.objects.filter(run_id=run.id).values_list("box_id", "id"))
        if not new:
            raise CommandError(f"run {run.id} 에 마스크가 없다")
        cur = dict(Mask.objects.filter(is_current=True)
                   .values_list("box_id", "id"))
        boxes = {b for b, _ in new}
        # 새 벌이 못 낸 상자는 옛 마스크를 그대로 둔다. 현재를 비우면 검토
        # 화면에서 그 상자가 통째로 사라지고, 그것은 자료를 잃는 것과 같다
        keep = set(cur) - boxes
        w(f"run {run.id} ({run.kind}) · {run.model}")
        w(f"  올릴 마스크 {len(new):,} 개")
        w(f"  옛 마스크를 그대로 두는 상자 {len(keep):,} 개"
          f" (새 벌이 아무것도 못 낸 것)")

        # 판정이 낡는 범위 — 다시 받아야 할 것이 몇인가
        latest = {}
        for r in Review.objects.order_by("id").values_list("box_id", "mask_id"):
            latest[r[0]] = r[1]
        new_by_box = dict(new)
        stale = sum(1 for b, mid in latest.items()
                    if b in new_by_box and mid != new_by_box[b])
        w(f"  **판정이 낡는 상자 {stale:,} 개** — 검토 화면의 `엔진 바뀜` 에 뜬다")
        if o["dry_run"]:
            w("\n--dry-run 이라 아무것도 쓰지 않았다.")
            return

        with transaction.atomic():
            Mask.objects.filter(is_current=True).exclude(
                box_id__in=keep).update(is_current=False)
            Mask.objects.filter(run_id=run.id).update(is_current=True)
        rec = runs.start("yolo" if run.kind == "yolo" else run.kind,
                         model=run.model,
                         params={"kind": "promote", "from_run": run.id,
                                 "promoted": len(new), "kept_old": len(keep),
                                 "stale_reviews": stale}, note=o["note"])
        runs.finish(rec)
        w(f"\nrun {run.id} 을 현재로 올렸다 (기록 run {rec.id})")
        w("**다음: manage.py infer_base --redo** — 밑동 제안은 마스크에 붙어"
          " 있어 따라오지 않는다")
