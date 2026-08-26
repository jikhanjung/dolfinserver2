"""**re-ID 자리(GCP) → 작업 자리(m710q)** 로 보낼 행을 담는다. re-ID 자리에서 돈다.

    python manage.py export_from_reid_to_work --out /app/hostdb/_to_work.sqlite3

`export_from_work_to_reid` 의 짝이고, 담는 것이 정확히 반대다 — **개체와
개체판정만.** 이 자리가 주인인 것이 그 둘뿐이라 그 둘만 나간다.

받는 쪽은 **통째로 갈아 끼운다**. 병합이 아니라 갈아 끼우기인 것은 이 자리가
그 둘의 유일한 주인이기 때문이다 — 저쪽에 이쪽이 모르는 개체 판정이 있을 수
없으니 합칠 것이 없다.
"""
import sqlite3
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.db import connection

from .export_from_work_to_reid import REID_OWNS, WORK_OWNS


class Command(BaseCommand):
    help = "작업 자리로 보낼 개체·개체판정을 한 파일에 담는다"

    def add_arguments(self, p):
        p.add_argument("--out", required=True)

    def handle(self, *a, **o):
        out = Path(o["out"])
        out.unlink(missing_ok=True)
        w = self.stdout.write
        dst = sqlite3.connect(str(out))
        try:
            with connection.cursor() as cur:
                for t in REID_OWNS:
                    cur.execute("select sql from sqlite_master "
                                "where type='table' and name=%s", [t])
                    ddl = cur.fetchone()
                    if ddl is None:
                        raise CommandError(f"그런 테이블이 없다: {t}")
                    dst.execute(ddl[0])
                    cur.execute(f"pragma table_info({t})")
                    cols = [r[1] for r in cur.fetchall()]
                    cur.execute(f"select {','.join(cols)} from {t}")
                    rows = cur.fetchall()
                    dst.executemany(
                        f"insert into {t} values ({','.join('?' * len(cols))})", rows)
                    w(f"  {t:<22} {len(rows):>7,}")
                # **어느 상자를 가리키는지 함께 적는다.** 받는 쪽이 그 상자를
                # 다 갖고 있는지 넣기 전에 확인한다 — 없으면 FK 가 막는데,
                # 막히는 것보다 **먼저 말해 주는 편**이 고치기 쉽다.
                cur.execute("select distinct box_id from finseg_identification "
                            "union select rep_id from finseg_individual "
                            "where rep_id is not null")
                boxes = [r[0] for r in cur.fetchall() if r[0] is not None]
            dst.execute("create table _needs_box (id integer)")
            dst.executemany("insert into _needs_box values (?)", [(b,) for b in boxes])
            dst.execute("create table _lane (name text, note text)")
            dst.executemany("insert into _lane values (?,?)",
                            [(t, "re-ID 자리가 주인 — 통째로 갈아 끼운다") for t in REID_OWNS]
                            + [(t, "작업 자리가 주인 — 안 담는다") for t in WORK_OWNS])
            dst.commit()
            w(f"  가리키는 상자             {len(boxes):>7,}")
        finally:
            dst.close()
        w(f"\n{out}  ({out.stat().st_size / 1024:.0f} KB)")
        w("받는 쪽에서:  python manage.py import_from_reid_to_work --from <이 파일>")
