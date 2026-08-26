"""**작업 자리(m710q) → re-ID 자리(GCP)** 로 보낼 행을 한 파일에 담는다.

    python manage.py export_from_work_to_reid --out /tmp/rows.sqlite3

이름에 방향이 들어 있다. 반대 방향은 `export_from_reid_to_work` 가 되고, 받는
쪽은 같은 이름의 `import_…` 이다 — **한 레인의 두 끝이 같은 이름을 쓴다.**

**무엇이 어느 쪽 것인가** (`HANDOFF` 의 `## 서버를 둘로 나눈다`):

    작업 자리가 주인   사진 · 상자 · 크롭 · 마스크 · 실행 · 판정
    re-ID 자리가 주인  개체 · 개체판정

그래서 이쪽에서 나가는 것은 늘 `WORK_OWNS` 뿐이고 **개체 쪽은 한 줄도 안
싣는다** — 실으면 저쪽이 쌓은 판정을 덮어쓴다. `fin.db` 를 통째로 보내지 않는
이유가 정확히 그것이다.

담는 그릇은 SQLite 파일이다. 같은 엔진이라 타입이 그대로 가고 받는 쪽은
`ATTACH` 한 줄로 붙인다 — JSON 으로 옮기다 날짜·NULL 에서 어긋날 자리가 없다.
"""
import sqlite3
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.db import connection

# **작업 자리가 주인인 테이블 전부.** 최소로 줄이지 않는 것은 FK 에 구멍을 안
# 내려는 것이다 — `Review.mask` 가 `Mask` 를, `Mask.run` 이 `Run` 을 가리킨다.
# 하나만 빼도 나중에 새 행이 갈 때 참조가 깨진다.
WORK_OWNS = ["finseg_image", "finseg_run", "finseg_box", "finseg_crop",
             "finseg_mask", "finseg_review"]

# **여기 것은 절대 안 싣는다.** re-ID 자리가 주인이다.
REID_OWNS = ["finseg_individual", "finseg_identification"]


class Command(BaseCommand):
    help = "re-ID 자리로 보낼 행을 한 파일에 담는다 (개체 판정은 안 담는다)"

    def add_arguments(self, p):
        p.add_argument("--out", required=True)

    def handle(self, *a, **o):
        out = Path(o["out"])
        out.unlink(missing_ok=True)
        w = self.stdout.write
        dst = sqlite3.connect(str(out))
        try:
            # **Django 의 연결로 읽는다.** 파일 경로로 따로 열면 시험(메모리 DB)
            # 에서 안 붙고, 무엇보다 화면이 열어 둔 것과 다른 것을 읽을 수 있다.
            with connection.cursor() as cur:
                for t in WORK_OWNS:
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
            # **무엇이 안 담겼는지도 적어 둔다.** 받는 쪽이 이것을 보고
            # "개체 판정이 안 왔다" 를 사고가 아니라 약속으로 읽는다.
            dst.execute("create table _lane (name text, note text)")
            dst.executemany("insert into _lane values (?,?)",
                            [(t, "작업 자리가 주인 — upsert 한다") for t in WORK_OWNS]
                            + [(t, "re-ID 자리가 주인 — 안 담는다") for t in REID_OWNS])
            dst.commit()
        finally:
            dst.close()
        w(f"\n{out}  ({out.stat().st_size / 1024**2:.0f} MB)")
        w("받는 쪽에서:  python manage.py import_from_work_to_reid --from <이 파일>")
