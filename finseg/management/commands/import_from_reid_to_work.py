"""**re-ID 자리 → 작업 자리** 로 온 개체·개체판정을 **통째로 갈아 끼운다.**

    python manage.py import_from_reid_to_work --from /tmp/_to_work.sqlite3

앞선 레인(`import_from_work_to_reid`)은 upsert 였는데 이쪽은 **지우고 넣는다.**
저쪽이 그 둘의 유일한 주인이라 합칠 것이 없어서다 — 여기 것은 사본일 뿐이다.

**그래서 이쪽이 더 위험하다.** 파일이 반쪽이면 사람의 판정을 지운다. 넣기 전에
넷을 본다:

  1. 담긴 것이 개체 레인뿐인가 (작업 레인이 섞여 오면 멎는다)
  2. 칸이 같은가 (한쪽이 마이그레이션을 덜 돌았나)
  3. **줄어들지 않는가** — 저쪽은 쌓기만 하는 자리라 줄면 사고다 (`--allow-shrink`)
  4. **가리키는 상자를 여기가 다 갖고 있는가** — 없으면 FK 가 막는다.
     그 답은 "먼저 `from_work_to_reid` 를 돌려라" 다

그리고 갈아 끼우기 **전에 한 벌 뜬다.** 되돌릴 자리 없이 지우지 않는다.
"""
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.db import connection, transaction

from .export_from_work_to_reid import REID_OWNS, WORK_OWNS


class _Stop(Exception):
    def __init__(self, msg=""):
        self.msg = msg


class Command(BaseCommand):
    help = "re-ID 자리에서 온 개체·개체판정으로 갈아 끼운다"

    def add_arguments(self, p):
        p.add_argument("--from", dest="src", required=True)
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--allow-shrink", action="store_true",
                       help="줄어드는 것을 받아들인다. **저쪽에서 정말 지운 것이 "
                            "맞는지 사람이 확인한 뒤에만**")

    def handle(self, *a, **o):
        src = Path(o["src"])
        if not src.is_file():
            raise CommandError(f"그런 파일이 없다: {src}")
        w = self.stdout.write

        cur = connection.cursor()
        cur.execute("attach database %s as src", [str(src)])

        def count(t, db="main"):
            cur.execute(f"select count(*) from {db}.{t}")
            return cur.fetchone()[0]

        try:
            cur.execute("select name from src.sqlite_master where type='table'")
            names = {r[0] for r in cur.fetchall()}
            bad = names & set(WORK_OWNS)
            if bad:
                raise CommandError(
                    f"이 파일에 남의 레인이 들어 있다: {sorted(bad)} — "
                    "`export_from_reid_to_work` 로 뜬 것이 맞는지 볼 것")
            missing = set(REID_OWNS) - names
            if missing:
                raise CommandError(
                    f"담겨야 할 것이 없다: {sorted(missing)} — **반쪽짜리 파일로 "
                    "갈아 끼우면 판정을 지운다.** 뜨는 쪽이 끝까지 돌았는지 볼 것")

            for t in REID_OWNS:
                cur.execute(f"pragma main.table_info({t})")
                mine = [r[1] for r in cur.fetchall()]
                cur.execute(f"pragma src.table_info({t})")
                his = [r[1] for r in cur.fetchall()]
                if mine != his:
                    raise CommandError(
                        f"{t} 의 칸이 다르다 — 한쪽이 마이그레이션을 덜 돌았다.\n"
                        f"  여기: {mine}\n  파일: {his}")

            before = {t: count(t) for t in REID_OWNS}
            coming = {t: count(t, "src") for t in REID_OWNS}
            for t in REID_OWNS:
                d = coming[t] - before[t]
                w(f"  {t:<22} {before[t]:>7,} → {coming[t]:>7,}  ({d:+,})")
                if d < 0 and not o["allow_shrink"]:
                    raise CommandError(
                        f"{t} 이 줄어든다 ({before[t]:,} → {coming[t]:,}). "
                        "**저쪽은 쌓기만 하는 자리라 줄면 사고다** — 반쪽 파일이거나 "
                        "옛 파일이다. 정말 지운 것이 맞으면 --allow-shrink")

            # **가리키는 상자를 여기가 다 갖고 있나.** FK 가 막기 전에 말한다 —
            # 막히면 "왜 안 되지" 지만, 여기서 말하면 "보내는 길을 먼저 돌려라" 다.
            if "_needs_box" in names:
                cur.execute("select count(*) from src._needs_box n "
                            "where not exists (select 1 from main.finseg_box b "
                            "where b.id = n.id)")
                gap = cur.fetchone()[0]
                if gap:
                    raise CommandError(
                        f"저쪽 판정이 가리키는 상자 {gap:,}개가 여기 없다 — "
                        "**`deploy/gcp/from_work_to_reid.sh` 를 먼저 돌릴 것.** "
                        "(저쪽이 이쪽보다 앞선 상자를 들고 있다는 뜻이다)")

            # **갈아 끼우기 전에 한 벌 뜬다.** 되돌릴 자리 없이 지우지 않는다.
            db = Path(connection.settings_dict["NAME"])
            snap = None
            if db.is_file() and not o["dry_run"]:
                snap = db.parent / f"fin.before-reid-import.{datetime.now():%Y-%m-%d_%H%M}.bak"
                s2 = sqlite3.connect(str(snap))
                connection.connection.backup(s2)
                s2.execute("pragma journal_mode=DELETE")
                s2.close()
                w(f"  뜬 것: {snap.name}")

            work_before = {t: count(t) for t in WORK_OWNS}
            try:
                with transaction.atomic():
                    # 개체판정을 먼저 지운다 — 개체를 먼저 지우면 CASCADE 로
                    # 함께 지워지는데, 순서를 못 박아 두는 편이 읽기 쉽다.
                    for t in ("finseg_identification", "finseg_individual"):
                        cur.execute(f"delete from main.{t}")
                    for t in ("finseg_individual", "finseg_identification"):
                        cur.execute(f"pragma main.table_info({t})")
                        cols = ",".join(r[1] for r in cur.fetchall())
                        cur.execute(f"insert into main.{t} ({cols}) "
                                    f"select {cols} from src.{t}")
                        # **번호도 저쪽 것에 맞춘다.** 여기서 쓸 일은 없지만,
                        # `FIN_ROLE=reid` 로 잠깐 띄워 보는 날 겹치지 않게.
                        cur.execute(f"update sqlite_sequence set seq="
                                    f"(select coalesce(max(id),0) from main.{t}) "
                                    f"where name='{t}'")
                    for t in WORK_OWNS:
                        if count(t) != work_before[t]:
                            raise _Stop(f"{t} 이 바뀌었다 — 이 명령은 그 레인을 "
                                        "건드리면 안 된다. 되돌렸다")
                    if o["dry_run"]:
                        raise _Stop("")
            except _Stop as e:
                if e.msg:
                    raise CommandError(e.msg)
                w("--dry-run 이라 되돌렸다.")
                return
            for t in WORK_OWNS:
                w(f"  {t:<22} {count(t):>7,}  (그대로 — 이쪽 레인이다)")
        finally:
            cur.execute("detach database src")
