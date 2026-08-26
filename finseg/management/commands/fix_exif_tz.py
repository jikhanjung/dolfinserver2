"""이미 든 `exifdatetime` 이 9시간 밀린 것을 바로잡는다 (2026-08-27, 한 번짜리).

    python manage.py fix_exif_tz --dry-run
    python manage.py fix_exif_tz

**무엇이 어긋났나** — 옛 DB 는 `exifdatetime` 을 UTC 로 들고 있는데
`import_boxes` 가 시간대를 안 붙이고 넘겼다. Django 는 순진한 값을
`TIME_ZONE`(Asia/Seoul)로 읽어 **한 번 더 UTC 로 바꿨고**, 그래서 저장된 값이
참값보다 9시간 이르다. 화면에 오후 2시 48분이어야 할 것이 새벽 5시 48분으로
나왔다.

**원본 EXIF 로 확인한 뒤에 고친다.** `+9` 라는 짐작만으로 7,927줄을 옮기지
않는다 — 표본을 원본 파일과 대 보고, 어긋나면 아무것도 안 한다. 그리고
**이미 맞는 것은 안 건드린다**(다시 돌려도 안전하다).
"""
from datetime import timedelta

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from finseg.models import Image

SHIFT = timedelta(hours=9)
SAMPLE = 60


class Command(BaseCommand):
    help = "9시간 밀린 exifdatetime 을 원본 EXIF 로 확인하고 바로잡는다"

    def add_arguments(self, p):
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--sample", type=int, default=SAMPLE)

    def handle(self, *a, **o):
        from django.conf import settings
        from django.utils import timezone
        from PIL import Image as PImage

        w = self.stdout.write
        qs = Image.objects.exclude(exifdatetime=None)
        n = qs.count()
        if not n:
            w("고칠 것이 없다"); return

        # ---- 원본과 대 본다 -------------------------------------------------
        hit = miss = seen = 0
        for path, dt in qs.values_list("path", "exifdatetime").order_by("?")[:o["sample"]]:
            f = settings.FIN_PHOTOS / path
            if not f.exists():
                continue
            try:
                ex = PImage.open(f)._getexif() or {}
            except Exception:
                continue
            raw = ex.get(36867) or ex.get(306)
            if not raw:
                continue
            seen += 1
            want = str(raw).replace(":", "-", 2)          # `2016:03:15 14:48:12`
            got = timezone.localtime(dt + SHIFT).strftime("%Y-%m-%d %H:%M:%S")
            hit += (got == want)
            miss += (got != want)
        if not seen:
            raise CommandError("원본을 한 장도 못 읽었다 — NAS 가 붙어 있나 "
                               "(`FIN_PHOTOS`). **확인 못 한 채로는 안 고친다**")
        w(f"원본 {seen}장과 대 봤다 — +9 하면 맞는 것 {hit} · 안 맞는 것 {miss}")
        if hit / seen < 0.95:
            raise CommandError(
                "+9 가 답이 아니다 — 아무것도 안 고쳤다. **짐작으로 7,927줄을 "
                "옮기지 않는다**: 무엇이 어긋났는지 먼저 다시 볼 것")

        if o["dry_run"]:
            w(f"--dry-run — {n:,}줄을 +9 할 참이었다")
            return
        from django.db.models import F
        with transaction.atomic():
            # **한 벌로 옮긴다.** 줄마다 읽어 고치면 7,927번 오가고, 그 사이에
            # 멎으면 절반만 옮겨진 채로 남는다 — 그 상태는 겉으로 안 보인다.
            qs.update(exifdatetime=F("exifdatetime") + SHIFT)
        w(f"{n:,}줄을 +9 했다")
        s = qs.values_list("path", "exifdatetime").first()
        w(f"  보기: {s[0]} → {timezone.localtime(s[1]):%Y-%m-%d %H:%M} (KST)")
