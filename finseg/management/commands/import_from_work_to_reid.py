"""**작업 자리 → re-ID 자리** 로 온 행을 upsert 한다. re-ID 자리에서 돈다.

    python manage.py import_from_work_to_reid --from /app/hostdb/rows.sqlite3

**`INSERT OR REPLACE` 를 쓰면 안 된다.** SQLite 에서 그것은 부딪히는 행을
**지우고 다시 넣는데**, `Identification.box` 가 `CASCADE` 라 그 순간 **이 자리가
쌓은 개체 판정이 함께 지워진다.** 상자 하나를 갱신하려다 그 상자에 달린 판정을
전부 잃는 것이다 — 조용히, 오류 없이. 그래서 참된 upsert(`ON CONFLICT DO
UPDATE`)만 쓴다: 행을 안 지우므로 CASCADE 가 안 돈다.

시험이 그것을 잰다 (`finseg/tests_lane.py`) — **재는 것은 "옮겨졌나" 가 아니라
"남의 레인을 안 건드렸나" 다.**
"""
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.db import connection, transaction

from .export_from_work_to_reid import REID_OWNS, WORK_OWNS


class _Stop(Exception):
    """`atomic` 을 되돌리려고 던진다 — 밖에서 사람이 읽을 말로 바꿔 낸다."""

    def __init__(self, msg=""):
        self.msg = msg


class Command(BaseCommand):
    help = "작업 자리에서 온 행을 upsert 한다 (개체 판정은 안 건드린다)"

    def add_arguments(self, p):
        p.add_argument("--from", dest="src", required=True)
        p.add_argument("--dry-run", action="store_true")

    def handle(self, *a, **o):
        src = Path(o["src"])
        if not src.is_file():
            raise CommandError(f"그런 파일이 없다: {src}")
        w = self.stdout.write

        # **Django 의 연결에 붙인다** — 화면이 열어 둔 바로 그 파일이라야 넣은
        # 것이 곧바로 보인다. `ATTACH` 는 트랜잭션 안에서 안 되므로 붙이는 것은
        # 밖에서 하고, 넣는 것만 `atomic` 안에서 한다.
        cur = connection.cursor()
        cur.execute("attach database %s as src", [str(src)])

        def count(t):
            cur.execute(f"select count(*) from main.{t}")
            return cur.fetchone()[0]

        try:
            cur.execute("select name from src.sqlite_master where type='table'")
            names = {r[0] for r in cur.fetchall()}
            # **저쪽 레인이 실려 오면 넣기 전에 멎는다.** 통째 덤프를 보냈다는
            # 뜻이고, 그대로 넣으면 개체 판정을 덮어쓴다.
            bad = names & set(REID_OWNS)
            if bad:
                raise CommandError(
                    f"이 파일에 남의 레인이 들어 있다: {sorted(bad)} — "
                    "`export_from_work_to_reid` 로 뜬 것이 맞는지 볼 것")

            before = {t: count(t) for t in REID_OWNS}
            lines = []
            try:
                with transaction.atomic():
                    for t in WORK_OWNS:
                        if t not in names:
                            lines.append(f"  {t:<22} (파일에 없다 — 건너뛴다)")
                            continue
                        cur.execute(f"pragma main.table_info({t})")
                        mine = [r[1] for r in cur.fetchall()]
                        cur.execute(f"pragma src.table_info({t})")
                        his = [r[1] for r in cur.fetchall()]
                        if mine != his:
                            raise _Stop(
                                f"{t} 의 칸이 다르다 — 한쪽이 마이그레이션을 "
                                f"덜 돌았다.\n  여기: {mine}\n  파일: {his}")
                        cols = ",".join(mine)
                        sets = ",".join(f"{c}=excluded.{c}" for c in mine if c != "id")
                        n0 = count(t)
                        # **`where true` 가 있어야 한다** — 없으면 SQLite 가
                        # `ON CONFLICT` 를 SELECT 의 일부로 읽으려다 멎는다.
                        cur.execute(
                            f"insert into main.{t} ({cols}) select {cols} "
                            f"from src.{t} where true "
                            f"on conflict(id) do update set {sets}")
                        lines.append(f"  {t:<22} {count(t):>7,}  (새로 {count(t) - n0:+,})")

                    # **넣고 나서 저쪽 레인을 다시 센다.** 안 건드렸다는 것을
                    # 말로만 두지 않는다 — 어긋나면 통째로 되돌린다.
                    for t in REID_OWNS:
                        if count(t) != before[t]:
                            raise _Stop(
                                f"{t} 이 {before[t]:,} → {count(t):,} 로 바뀌었다 — "
                                "**이 명령은 그 레인을 건드리면 안 된다.** 되돌렸다")
                    if o["dry_run"]:
                        raise _Stop("")
            except _Stop as e:
                if e.msg:
                    raise CommandError(e.msg)
                w("\n".join(lines))
                w("--dry-run 이라 되돌렸다.")
                return
            w("\n".join(lines))
            for t in REID_OWNS:
                w(f"  {t:<22} {count(t):>7,}  (그대로 — 저쪽 레인이다)")
        finally:
            cur.execute("detach database src")
