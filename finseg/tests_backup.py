"""시간별 백업의 계약을 잰다 (`.guides/web/data-safety.md`).

**게이트에 물리지 않으면 "검사가 있다" 는 착각만 생긴다** — 형제 프로젝트가
시험 47종을 갖고도 릴리스에서 한 번도 안 돌린 적이 있다. 그래서 여기 두고
`manage.py test` 가 잡게 한다 (`deploy/build.sh` 가 그것을 부른다).

재는 것은 **실패했을 때의 행동**이다. 잘 도는 백업은 눈에 보이지만, 계약이
지켜지는지는 손상됐을 때에만 드러난다 — 그리고 그때는 이미 늦다.
"""
import importlib.util
import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest import mock

from django.test import SimpleTestCase

SCRIPT = Path(__file__).resolve().parent.parent / "deploy" / "host" / "backup_db.py"


def make_db(path, sessions=3, reviews=5):
    c = sqlite3.connect(str(path))
    c.execute("create table finseg_review (id integer primary key)")
    c.execute("create table finseg_identification (id integer primary key)")
    c.execute("create table finseg_individual (id integer primary key)")
    c.execute("create table django_session (session_key text primary key,"
              " session_data text, expire_date text)")
    c.executemany("insert into finseg_review values (?)", [(i,) for i in range(1, reviews + 1)])
    c.executemany("insert into django_session values (?,?,?)",
                  [(f"key{i}", "x" * 400, "2026-01-01") for i in range(sessions)])
    c.commit()
    c.close()


class BackupContractTests(SimpleTestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.db = self.tmp / "fin.db"
        make_db(self.db)
        self.env = {
            "FIN_ROOT": str(self.tmp),
            "FIN_DB": str(self.db),
            "FIN_BACKUP_DIR": str(self.tmp / "backup" / "hourly"),
            "FIN_NAS_DIR": str(self.tmp / "nas-없음"),
            "FIN_SENTINEL": str(self.tmp / "INTEGRITY_FAIL"),
            "FIN_BACKUP_HOST": "testbox",
            "FIN_NOTIFY": "",            # 시험이 진짜 알림을 울리면 "늑대야" 가 된다
        }

    def load(self):
        """env 를 물린 채 새로 읽어 온다 — 모듈 수준에서 경로를 잡는 스크립트다."""
        with mock.patch.dict(os.environ, self.env, clear=False):
            spec = importlib.util.spec_from_file_location(f"bk{id(self)}", SCRIPT)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
        return mod

    def snaps(self, mod):
        return sorted(p.name for p in mod.LOCAL_DIR.glob("fin_testbox_*.sqlite3"))

    # ---- 정상 경로 --------------------------------------------------------
    def test_it_makes_a_snapshot_and_strips_the_session_table(self):
        """§7 세션 키가 곧 쿠키 값이다 — 사본을 읽은 사람이 그대로 재제출하면
        로그인된다. **`SECRET_KEY` 가 달라도 안 막힌다.**"""
        mod = self.load()
        self.assertEqual(mod.main(), 0)
        got = self.snaps(mod)
        self.assertEqual(len(got), 1)
        self.assertTrue(got[0].startswith("fin_testbox_"))   # 이름에 기계가 들어간다

        c = sqlite3.connect(str(mod.LOCAL_DIR / got[0]))
        self.assertEqual(c.execute("select count(*) from django_session").fetchone()[0], 0)
        self.assertEqual(c.execute("select count(*) from finseg_review").fetchone()[0], 5)
        # §5 단일 파일 — WAL 형제가 남으면 옮길 때 커밋 꼬리를 놓친다
        self.assertEqual(c.execute("pragma journal_mode").fetchone()[0].lower(), "delete")
        c.close()
        self.assertFalse((mod.LOCAL_DIR / f"{got[0]}-wal").exists())

    def test_it_keeps_only_the_retained_count(self):
        """§6 `RETAIN_COUNT × 주기 ≥ 오프사이트 간격` — 시간별 × 24 = 하루."""
        mod = self.load()
        mod.LOCAL_DIR.mkdir(parents=True, exist_ok=True)
        for i in range(30):
            (mod.LOCAL_DIR / f"fin_testbox_2026-01-01_{i:02d}.sqlite3").write_bytes(b"x")
        self.assertEqual(mod.main(), 0)
        self.assertEqual(len(self.snaps(mod)), mod.RETAIN_COUNT)

    def test_it_never_prunes_another_machines_lane(self):
        """곧 GCP 에도 진짜 `fin.db` 가 생기고 둘 다 같은 자리로 온다.
        **남의 갈래를 이쪽 셈으로 줄이면 그 기계는 제가 몇 벌 갖고 있는지
        모르는 채 줄어든다.**"""
        mod = self.load()
        mod.LOCAL_DIR.mkdir(parents=True, exist_ok=True)
        for i in range(30):
            (mod.LOCAL_DIR / f"fin_gcp_2026-01-01_{i:02d}.sqlite3").write_bytes(b"x")
        self.assertEqual(mod.main(), 0)
        self.assertEqual(len(list(mod.LOCAL_DIR.glob("fin_gcp_*.sqlite3"))), 30)

    # ---- 실패했을 때가 계약이다 -------------------------------------------
    def test_a_failed_integrity_check_preserves_the_good_copies(self):
        """§2 **이것이 계약의 핵심이다.** 형제 프로젝트가 반환값을 버리고 무조건
        정리해서, 손상되면 `RETAIN_COUNT` 시간 만에 **성한 스냅샷이 0개**가 됐다."""
        mod = self.load()
        mod.LOCAL_DIR.mkdir(parents=True, exist_ok=True)
        for i in range(30):
            (mod.LOCAL_DIR / f"fin_testbox_2026-01-01_{i:02d}.sqlite3").write_bytes(b"x")

        with mock.patch.object(mod, "integrity_ok", return_value=False):
            self.assertEqual(mod.main(), 1)

        self.assertEqual(len(self.snaps(mod)), 30)          # 하나도 안 지웠다
        self.assertTrue(mod.SENTINEL.exists())              # /healthz 가 degraded 를 낸다
        ev = list(mod.LOCAL_DIR.glob("*_INTEGRITY_FAIL.corrupt"))
        self.assertEqual(len(ev), 1)                        # 증거본은 딱 하나
        self.assertNotIn(ev[0].name, self.snaps(mod))       # prune glob 에 안 걸린다

    def test_hygiene_runs_after_the_gate_not_before(self):
        """§7 순서가 뒤집히면 **손상된 소스에서 `VACUUM` 이 터지고 그것을
        삼키느라 센티넬이 안 선다** — 안전망이 조용히 꺼진다."""
        mod = self.load()
        with mock.patch.object(mod, "integrity_ok", return_value=False), \
             mock.patch.object(mod, "strip_sessions") as strip:
            mod.main()
        strip.assert_not_called()

    def test_a_missing_source_does_not_prune(self):
        """소스가 없다고 있던 백업을 지우면, 되돌릴 자리가 그때 사라진다."""
        mod = self.load()
        mod.LOCAL_DIR.mkdir(parents=True, exist_ok=True)
        (mod.LOCAL_DIR / "fin_testbox_2026-01-01_00.sqlite3").write_bytes(b"x")
        self.db.unlink()
        self.assertEqual(mod.main(), 1)
        self.assertEqual(len(self.snaps(mod)), 1)

    def test_an_absent_nas_is_normal_not_a_failure(self):
        """§3 부재는 정상이다. **없는 것을 실패로 세면 시간마다 `exit 1` 이
        찍히고, 그 소음이 진짜 실패를 묻는다.**"""
        mod = self.load()
        with mock.patch.object(mod, "NAS_HOUR", -1):        # NAS 시각이 아니다
            self.assertEqual(mod.main(), 0)

    def test_the_sentinel_clears_once_a_good_snapshot_lands(self):
        """고쳐 놓고도 센티넬이 남으면 `/healthz` 가 영영 degraded 다."""
        mod = self.load()
        mod.SENTINEL.parent.mkdir(parents=True, exist_ok=True)
        mod.SENTINEL.write_text("옛 실패\n")
        self.assertEqual(mod.main(), 0)
        self.assertFalse(mod.SENTINEL.exists())
