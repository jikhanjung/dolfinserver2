"""레인이 갈려 있는지 잰다 (`HANDOFF` 의 `## 서버를 둘로 나눈다`).

**여기서 재는 것은 "옮겨졌나" 가 아니라 "안 건드렸나" 다.** 옮겨진 것은 눈에
보이지만, 남의 레인을 건드렸는지는 그 판정을 다시 찾을 때에야 드러나고 그때는
이미 늦다.
"""
import sqlite3
import tempfile
from datetime import date
from pathlib import Path

from django.core.management import CommandError, call_command
from django.db import connection
from django.test import TransactionTestCase

from finseg.models import Box, Identification, Image, Individual, Review


class LaneTests(TransactionTestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.out = self.tmp / "rows.sqlite3"
        img = Image.objects.create(path="nas/2016/03/15/a.JPG",
                                   obsdate=date(2016, 3, 15), width=100, height=100)
        self.box = Box.objects.create(image=img, x1=0, y1=0, x2=60, y2=60,
                                      source="yolov5", conf=0.9)
        Review.objects.create(box=self.box, cls="fin")

    def test_it_never_carries_the_other_lane(self):
        """개체 판정이 실려 가면 저쪽에서 그것을 덮어쓴다 — **통째 덤프를 보내는
        것이 정확히 그 사고다.**"""
        ind = Individual.objects.create(name="상자 1")
        Identification.objects.create(box=self.box, individual=ind)
        call_command("export_from_work_to_reid", out=str(self.out))
        con = sqlite3.connect(str(self.out))
        names = {r[0] for r in con.execute(
            "select name from sqlite_master where type='table'")}
        con.close()
        self.assertIn("finseg_box", names)
        self.assertNotIn("finseg_individual", names)
        self.assertNotIn("finseg_identification", names)

    def test_importing_a_box_does_not_delete_its_identifications(self):
        """**이 시험이 이 명령의 이유다.**

        `INSERT OR REPLACE` 는 부딪히는 행을 지우고 다시 넣는데
        `Identification.box` 가 `CASCADE` 라, 상자 하나를 갱신하려다 그 상자에
        달린 개체 판정이 **조용히 함께 지워진다.**
        """
        call_command("export_from_work_to_reid", out=str(self.out))          # 상자를 담아 두고
        ind = Individual.objects.create(name="상자 1")           # 저쪽에서 판정이 쌓인 척
        Identification.objects.create(box=self.box, individual=ind)
        Box.objects.filter(pk=self.box.pk).update(conf=0.5)     # 그 사이 상자가 바뀐 척

        call_command("import_from_work_to_reid", src=str(self.out))

        self.assertEqual(Identification.objects.count(), 1)     # 살아 있어야 한다
        self.assertEqual(Individual.objects.count(), 1)
        self.box.refresh_from_db()
        self.assertEqual(self.box.conf, 0.9)                    # 갱신은 됐다

    def test_it_refuses_a_file_that_carries_the_other_lane(self):
        """실수로 통째 덤프를 보냈을 때 **넣기 전에 멎어야 한다.**"""
        call_command("export_from_work_to_reid", out=str(self.out))
        con = sqlite3.connect(str(self.out))
        con.execute("create table finseg_identification (id integer primary key)")
        con.commit(); con.close()
        with self.assertRaises(CommandError) as e:
            call_command("import_from_work_to_reid", src=str(self.out))
        self.assertIn("남의 레인", str(e.exception))

    def test_it_refuses_when_the_columns_differ(self):
        """한쪽이 마이그레이션을 덜 돌았으면 **조용히 반쪽을 넣지 않는다.**"""
        call_command("export_from_work_to_reid", out=str(self.out))
        con = sqlite3.connect(str(self.out))
        con.execute("alter table finseg_box add column 새칸 text")
        con.commit(); con.close()
        with self.assertRaises(CommandError) as e:
            call_command("import_from_work_to_reid", src=str(self.out))
        self.assertIn("칸이 다르다", str(e.exception))

    def test_new_rows_arrive(self):
        """보내는 쪽에서 상자가 늘면 받는 쪽에 생겨야 한다 — **그것이 없으면
        새 격자의 조각을 개체에 넣을 수 없다**(FK 가 막는다)."""
        n0 = Box.objects.count()
        img = Image.objects.create(path="nas/2016/03/16/b.JPG",
                                   obsdate=date(2016, 3, 16), width=100, height=100)
        Box.objects.create(image=img, x1=1, y1=1, x2=50, y2=50, source="yolov5", conf=0.7)
        call_command("export_from_work_to_reid", out=str(self.out))
        Box.objects.all().delete()                     # 받는 쪽에 아직 없는 척
        call_command("import_from_work_to_reid", src=str(self.out))
        self.assertEqual(Box.objects.count(), n0 + 1)

    def test_dry_run_changes_nothing(self):
        call_command("export_from_work_to_reid", out=str(self.out))
        Box.objects.filter(pk=self.box.pk).update(conf=0.1)
        call_command("import_from_work_to_reid", src=str(self.out), dry_run=True)
        self.box.refresh_from_db()
        self.assertEqual(self.box.conf, 0.1)
