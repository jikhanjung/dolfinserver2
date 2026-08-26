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
from django.test import SimpleTestCase, TransactionTestCase

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


class ReturnLaneTests(TransactionTestCase):
    """**되받는 길은 갈아 끼운다** — 그래서 보내는 길보다 위험하다.

    보내는 길이 upsert 라 최악이 "덜 갔다" 인 데 비해, 이쪽은 최악이
    **"사람의 판정을 지웠다"** 다. 여기 시험은 전부 그 최악을 막는 자리다.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.out = self.tmp / "back.sqlite3"
        img = Image.objects.create(path="nas/2016/03/15/a.JPG",
                                   obsdate=date(2016, 3, 15), width=100, height=100)
        self.box = Box.objects.create(image=img, x1=0, y1=0, x2=60, y2=60,
                                      source="yolov5", conf=0.9)
        self.ind = Individual.objects.create(name="상자 1")
        Identification.objects.create(box=self.box, individual=self.ind)

    def test_it_carries_only_the_reid_lane(self):
        call_command("export_from_reid_to_work", out=str(self.out))
        con = sqlite3.connect(str(self.out))
        names = {r[0] for r in con.execute(
            "select name from sqlite_master where type='table'")}
        con.close()
        self.assertIn("finseg_individual", names)
        self.assertIn("finseg_identification", names)
        for t in ("finseg_box", "finseg_review", "finseg_mask"):
            self.assertNotIn(t, names)

    def test_it_replaces_rather_than_merges(self):
        """저쪽이 유일한 주인이라 합칠 것이 없다 — **여기 것은 사본이다.**
        저쪽에서 상자를 옮겼으면 여기 옛 자취가 남아 있으면 안 된다."""
        call_command("export_from_reid_to_work", out=str(self.out))   # 판정 1건을 담고
        Identification.objects.create(box=self.box, individual=None)  # 여기서만 늘어난 척
        self.assertEqual(Identification.objects.count(), 2)
        call_command("import_from_reid_to_work", src=str(self.out), allow_shrink=True)
        self.assertEqual(Identification.objects.count(), 1)           # 갈아 끼웠다

    def test_it_refuses_to_shrink(self):
        """**저쪽은 쌓기만 하는 자리라 줄면 사고다** — 반쪽 파일이거나 옛 파일이다."""
        call_command("export_from_reid_to_work", out=str(self.out))
        Identification.objects.create(box=self.box, individual=self.ind)
        with self.assertRaises(CommandError) as e:
            call_command("import_from_reid_to_work", src=str(self.out))
        self.assertIn("줄어든다", str(e.exception))
        self.assertEqual(Identification.objects.count(), 2)           # 안 건드렸다

    def test_it_refuses_a_half_file(self):
        """**반쪽짜리로 갈아 끼우면 판정을 지운다.**"""
        call_command("export_from_reid_to_work", out=str(self.out))
        con = sqlite3.connect(str(self.out))
        con.execute("drop table finseg_identification"); con.commit(); con.close()
        with self.assertRaises(CommandError) as e:
            call_command("import_from_reid_to_work", src=str(self.out))
        self.assertIn("담겨야 할 것이 없다", str(e.exception))
        self.assertEqual(Identification.objects.count(), 1)

    def test_it_says_to_run_the_forward_lane_first(self):
        """저쪽 판정이 여기 없는 상자를 가리키면 FK 가 막는다 — **막히기 전에
        무엇을 해야 하는지 말한다.**"""
        call_command("export_from_reid_to_work", out=str(self.out))
        con = sqlite3.connect(str(self.out))
        con.execute("insert into _needs_box values (999999)"); con.commit(); con.close()
        with self.assertRaises(CommandError) as e:
            call_command("import_from_reid_to_work", src=str(self.out))
        self.assertIn("from_work_to_reid", str(e.exception))

    def test_the_work_lane_is_untouched(self):
        n = (Box.objects.count(), Review.objects.count(), Image.objects.count())
        call_command("export_from_reid_to_work", out=str(self.out))
        call_command("import_from_reid_to_work", src=str(self.out))
        self.assertEqual(
            (Box.objects.count(), Review.objects.count(), Image.objects.count()), n)

    def test_dry_run_changes_nothing(self):
        call_command("export_from_reid_to_work", out=str(self.out))
        Identification.objects.create(box=self.box, individual=None)
        call_command("import_from_reid_to_work", src=str(self.out),
                     dry_run=True, allow_shrink=True)
        self.assertEqual(Identification.objects.count(), 2)


class OverlayFrameTests(SimpleTestCase):
    """**조각과 좌표가 같은 자로 세워졌나.**

    `frame()` 은 `facing` 으로 좌우를 뒤집는다. 그림을 구울 때와 좌표를 낼 때
    그 값이 다르면 **정확히 좌우반전된 윤곽**이 얹힌다 — 2026-08-27 에 격자의
    45%가 그렇게 어긋났고, 눈으로 보기 전에는 아무 오류도 안 났다.
    """

    def make(self, facing):
        from finseg import reid
        st = {"cls": "fin", "facing": facing,
              "polygon": "20,60 40,10 60,60", "base_line": "20,60 60,60",
              "base_partial": None, "review": None, "mask": None}
        crop = type("C", (), {"x0": 0, "y0": 0, "x1": 80, "y1": 80,
                              "w": 80, "h": 80, "scale": 1.0, "path": None})()
        return st, crop, reid

    def test_the_outline_follows_the_same_flip_as_the_image(self):
        """뒤집힌 그림 위에 안 뒤집힌 좌표를 얹으면 거울상이 된다."""
        st, crop, reid = self.make("right")
        left = reid.overlay({**st, "facing": "left"}, crop)
        right = reid.overlay(st, crop)
        self.assertIsNotNone(left)
        self.assertIsNotNone(right)
        # 같은 폴리곤인데 `facing` 만 다르면 **x 가 뒤집힌다** — 그것이 곧
        # 그림이 뒤집히는 방식이고, 좌표도 같이 뒤집혀야 맞는다
        lx = sorted(round(x, 3) for x, _ in left["out"])
        rx = sorted(round(1 - x, 3) for x, _ in right["out"])
        self.assertEqual(lx, rx)

    def test_the_front_point_is_first_whichever_side(self):
        """`base[0]` 이 앞이라는 규칙은 `reid.overlay` 한 곳에 있어야 한다 —
        화면이 그것 하나만 알면 되게."""
        for f in ("left", "right"):
            st, crop, reid = self.make(f)
            ov = reid.overlay(st, crop)
            self.assertLess(ov["base"][0][0], ov["base"][1][0], f)

    def test_no_facing_means_no_overlay(self):
        """짐작조차 못 한 것에 억지로 좌표를 내면 **어느 쪽으로 세운 것인지
        모르는 선**이 얹힌다. 안 내는 편이 낫다."""
        st, crop, reid = self.make("")
        self.assertIsNone(reid.overlay(st, crop))
