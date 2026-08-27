"""검토 화면이 **규칙을 말하는지** 시험한다.

밑동 현이라는 것을 화면이 말하지 않으면 사람이 스스로와 어긋나고, 어긋난 검토는
나중에 되살릴 수 없다 — 그래서 이것은 꾸밈이 아니라 자료의 요건이다. 문구는
`finseg.baseline` 에 있고 화면이 그것을 그대로 띄운다. 여기서 잡는 것은
**둘이 갈라지는 것**이다.
"""
import io
import json
import re
import tempfile
from datetime import date
from pathlib import Path

import numpy as np

from django.test import SimpleTestCase, TestCase, override_settings

from finseg import baseline, geometry, onnxdet, reid, rules
from review.urls import patterns_for

# **이 모듈이 re-ID 자리의 URLconf 노릇을 한다.** `/reid` 는 기본 역할(`work`)
# 에서 아예 안 걸리므로(`finweb/settings.py` 의 `FIN_ROLE`), 그 화면을 재는
# 시험은 그 자리로 옮겨 앉아야 한다 — `@override_settings(ROOT_URLCONF=...)`.
urlpatterns = patterns_for("reid")
from finseg.models import (Box, Crop, Identification, Image, Individual,
                           Mask, Review, Run)

BLOCK = re.compile(r'<div class="rule">(.*?)</div>', re.S)
TAG = re.compile(r"<[^>]+>")


class RuleShownTests(SimpleTestCase):

    def block(self):
        """규칙 칸의 HTML. **거기 있어야 한다** — 주석이나 스크립트가 아니라."""
        m = BLOCK.search(self.client.get("/review").content.decode())
        self.assertIsNotNone(m, "화면에 규칙 칸(.rule)이 없다")
        return m.group(1)

    def text(self):
        """규칙 칸의 글자만. 태그를 걷어 내야 `<b>` 가 문장을 가르지 않는다."""
        return TAG.sub("", self.block())

    def test_rule_is_on_screen(self):
        self.assertIn(baseline.RULE.replace("**", ""), self.text())

    def test_every_point_is_on_screen(self):
        """규칙을 `baseline` 에 더하면 화면에 저절로 뜬다."""
        text = self.text()
        for point in baseline.RULE_POINTS:
            self.assertIn(point.replace("**", ""), text)

    def test_emphasis_became_markup(self):
        """`**굵게**` 는 화면에서 태그가 된다 — 별표가 그대로 보이면 안 된다."""
        self.assertIn("<b>앞·뒤 삽입점을 잇는 직선</b>", self.block())
        self.assertNotIn("**", self.block())


class NewBoxQueueTests(TestCase):
    """**마스크 없는 상자**를 화면과 저장이 같은 뜻으로 다루나.

    새 검출기가 들인 상자에는 윤곽이 없다. 거기서 `verdict`("이 마스크가 맞나")
    는 물을 수 없는 말이라 화면이 못 누르게 하고 서버도 안 적는데, **둘 중
    하나만 고쳐지면 화면은 바뀌고 저장은 200 을 받으면서 값만 안 들어간다** —
    이 저장소에서 가장 자주 겪은 버그다.
    """

    def setUp(self):
        img = Image.objects.create(path="t/a.jpg", obsdate="2020-01-01",
                                   width=1000, height=800)
        self.old = Box.objects.create(image=img, x1=10, y1=10, x2=60, y2=60,
                                      source="yolov5")
        self.new = Box.objects.create(image=img, x1=500, y1=500, x2=560, y2=560,
                                      source="yolo11", conf=0.77)
        for b in (self.old, self.new):
            Crop.objects.create(box=b, path=f"t/{b.id}.jpg", x0=0, y0=0,
                                x1=640, y1=640, w=640, h=640)
        run = Run.objects.create(kind="sam2")
        self.mask = Mask.objects.create(box=self.old, run=run, is_current=True,
                                        polygon="20,20 50,20 35,50")

    def test_queue_lists_only_new_source_by_confidence(self):
        r = self.client.get("/api/batch?mode=new").json()
        self.assertEqual([t["box_id"] for t in r["tiles"]], [self.new.id])
        t = r["tiles"][0]
        self.assertEqual(t["source"], "yolo11")
        self.assertIsNone(t["mask_id"])
        self.assertAlmostEqual(t["conf"], 0.77)

    def test_todo_still_needs_a_mask(self):
        """`검토할 것` 에는 안 뜬다 — 거기서 묻는 것에 윤곽이 들어 있다."""
        r = self.client.get("/api/batch?mode=todo").json()
        self.assertEqual([t["box_id"] for t in r["tiles"]], [self.old.id])

    def test_verdict_is_not_recorded_without_a_mask(self):
        """`ok` 를 적으면 `label_of` 가 양성이라 하고 진행상황이 거짓말한다."""
        res = self.client.post("/api/review", data=json.dumps({"items": [
            {"box_id": self.new.id, "cls": "fin", "edges": "both",
             "verdict": "ok", "facing": ""}]}),
            content_type="application/json")
        self.assertEqual(res.status_code, 200)
        rv = Review.objects.get(box=self.new)
        self.assertEqual(rv.cls, "fin")
        self.assertEqual(rv.verdict, "")
        self.assertEqual(rules.label_of(rv), rules.PENDING)

    def test_verdict_is_recorded_when_a_mask_exists(self):
        self.client.post("/api/review", data=json.dumps({"items": [
            {"box_id": self.old.id, "cls": "fin", "edges": "both",
             "verdict": "ok", "mask_id": self.mask.id, "facing": ""}]}),
            content_type="application/json")
        rv = Review.objects.get(box=self.old)
        self.assertEqual(rv.verdict, "ok")
        self.assertEqual(rules.label_of(rv), rules.POSITIVE)

    def test_progress_counts_new_boxes_apart(self):
        """한 숫자에 섞으면 '다 끝났다' 로 보인다 — 엔진 바뀜에서 겪은 것."""
        self.client.post("/api/review", data=json.dumps({"items": [
            {"box_id": self.new.id, "cls": "none", "facing": ""}]}),
            content_type="application/json")
        c = rules.progress()
        self.assertEqual(c["새검출"], 1)
        self.assertEqual(c["새검출대기"], 0)
        self.assertEqual(c["새검출:헛것"], 1)


class HoldTests(TestCase):
    """**보류한 칸은 목록에서 빠지되 아무 판정도 안 남는다.**

    묶음 저장은 손 안 댄 칸까지 판정으로 적는다(`views.save`). 그래서 "모르겠다"
    를 말하려면 그 칸이 payload 에서 빠져야 하는데, **빠진 상태가 이미 정확한
    답**이다 — `rules.PENDING`("아직 사람이 말하지 않았다")이 그것이고,
    `rules.py` 가 크롭을 통째로 빼는 장치를 거기에만 남겨 두었다.

    보류를 표로 만들지 않은 이유가 여기 걸린다. 만들면 "모른다" 가 판정인 척
    남아 나중에 그것을 또 걸러야 하고, `effective_review()` 가 그것을 골라
    버리는 날이 온다. 그래서 **서버는 목록에서 빼 주기만 하고 아무것도 적지
    않는다.**
    """

    def setUp(self):
        img = Image.objects.create(path="t/h.jpg", obsdate="2020-01-01",
                                   width=1000, height=800)
        self.a = Box.objects.create(image=img, x1=10, y1=10, x2=60, y2=60,
                                    source="yolo11", conf=0.9)
        self.b = Box.objects.create(image=img, x1=200, y1=200, x2=260, y2=260,
                                    source="yolo11", conf=0.5)
        for box in (self.a, self.b):
            Crop.objects.create(box=box, path=f"t/{box.id}.jpg", x0=0, y0=0,
                                x1=640, y1=640, w=640, h=640)

    def ids(self, url):
        return [t["box_id"] for t in self.client.get(url).json()["tiles"]]

    def test_held_box_leaves_the_queue(self):
        self.assertEqual(self.ids("/api/batch?mode=new"), [self.a.id, self.b.id])
        self.assertEqual(self.ids(f"/api/batch?mode=new&hold={self.a.id}"),
                         [self.b.id])

    def test_count_and_pages_follow(self):
        """**쪽수를 세기 전에 뺀다.** 화면에서만 걸러 내면 한 쪽이 통째로 보류일
        때 빈 격자가 뜨는데, 화면은 그것이 끝인지 고장인지 말해 주지 않는다."""
        r = self.client.get(f"/api/batch?mode=new&hold={self.a.id}").json()
        self.assertEqual(r["total"], 1)
        self.assertEqual(r["pages"], 1)

    def test_holding_records_nothing(self):
        """보류는 `PENDING` 으로 남는다 — 그것이 정확한 상태다."""
        self.client.get(f"/api/batch?mode=new&hold={self.a.id}")
        self.assertFalse(Review.objects.filter(box=self.a).exists())
        self.assertEqual(rules.label_of(rules.effective_review(self.a)),
                         rules.PENDING)

    def test_garbage_is_ignored(self):
        """화면이 보내는 것이라 늘 숫자라는 보장이 없다 — 500 을 내면 안 된다."""
        r = self.client.get("/api/batch?mode=new&hold=,,x,-3,%20")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["total"], 2)

    def test_hold_works_on_the_todo_queue_too(self):
        """`검토할 것` 도 묶음 저장이라 같은 구멍이 있다."""
        run = Run.objects.create(kind="sam2")
        Mask.objects.create(box=self.a, run=run, is_current=True,
                            polygon="20,20 50,20 35,50")
        self.assertEqual(self.ids("/api/batch?mode=todo"), [self.a.id])
        self.assertEqual(self.ids(f"/api/batch?mode=todo&hold={self.a.id}"), [])


class ExportDetectReviewTests(TestCase):
    """**검토가 다음 검출 학습으로 이어지나.**

    오래 끊겨 있던 자리다 — `export_detect` 가 옛 운영 DB 만 읽어서, 검토
    화면에서 인정한 상자가 다음 학습에 안 들어갔다. 여기서 잡는 것은 그
    이음매다: 인정한 것이 **더해지고**, 아니라고 한 옛 상자가 **빠지고**,
    새 검출기가 안 훑은 날은 **안 쓰인다**.

    옛 DB 없이 돌 수 있게 `_from_reviews` 만 따로 잰다 — 그 함수는 Django
    쪽만 본다.
    """

    def cmd(self):
        from finseg.management.commands.export_detect import Command
        return Command()

    def setUp(self):
        # 새 검출기가 훑은 날은 run 기록에서 읽는다 — 손으로 적으면 다음에
        # 더 훑고 나서 여기를 안 고친다
        Run.objects.create(kind="detect", params={"kind": "infer_boxes",
                                                  "dates": ["2020-01-01"]})
        self.swept = Image.objects.create(path="t/s.jpg", src_id=11,
                                          obsdate="2020-01-01")
        self.unswept = Image.objects.create(path="t/u.jpg", src_id=22,
                                            obsdate="2020-06-01")

    def box(self, img, source, src_id=None, cls=None, **kw):
        b = Box.objects.create(image=img, x1=10, y1=20, x2=60, y2=80,
                               source=source, src_id=src_id, **kw)
        if cls:
            Review.objects.create(box=b, cls=cls)
        return b

    def test_accepted_new_box_is_added(self):
        b = self.box(self.swept, "yolo11", cls="fin")
        add, reject, imgs, _ = self.cmd()._from_reviews(set())
        self.assertEqual(add[11], [(b.x1, b.y1, b.x2, b.y2)])
        self.assertEqual(imgs, {11})

    def test_old_box_is_not_added_twice(self):
        """옛 상자는 옛 DB 에서 그대로 나간다 — 여기서 더하면 두 번 된다."""
        self.box(self.swept, "yolov5", src_id=777, cls="fin")
        add, reject, _, _ = self.cmd()._from_reviews(set())
        self.assertEqual(add, {})
        self.assertEqual(reject, set())

    def test_rejected_old_box_is_subtracted(self):
        """옛 DB 에서는 `fin` 으로 나가던 것이다 — 빼는 것이 곧 고치는 일이다."""
        self.box(self.swept, "yolov5", src_id=777, cls="none")
        self.box(self.swept, "yolov5", src_id=888, cls="person")
        _, reject, _, _ = self.cmd()._from_reviews(set())
        self.assertEqual(reject, {777, 888})

    def test_latest_review_wins(self):
        """판정은 쌓인다 — `rules.effective_review` 와 같은 규칙이어야 한다."""
        b = self.box(self.swept, "yolo11", cls="none")
        Review.objects.create(box=b, cls="fin")
        add, reject, _, _ = self.cmd()._from_reviews(set())
        self.assertEqual(len(add[11]), 1)

    def test_unswept_date_is_left_out(self):
        """검출기 하나만 본 날이라 완전성이 한 단계 낮다."""
        self.box(self.unswept, "yolo11", cls="fin")
        add, _, imgs, _ = self.cmd()._from_reviews(set())
        self.assertEqual(add, {})
        self.assertEqual(imgs, set())

    def test_excluded_date_is_left_out(self):
        self.box(self.swept, "yolo11", cls="fin")
        add, _, imgs, _ = self.cmd()._from_reviews({"2020-01-01"})
        self.assertEqual(imgs, set())

    def test_unreviewed_box_is_left_out(self):
        """판정이 없는 것은 아무 말도 안 한 것이다 (`rules.PENDING`)."""
        self.box(self.swept, "yolo11")
        add, _, imgs, note = self.cmd()._from_reviews(set())
        self.assertEqual(add, {})
        self.assertEqual(imgs, set())
        self.assertTrue(any("판정이 없는" in n for n in note))

    def test_without_a_sweep_record_nothing_is_used(self):
        """훑은 날을 모르면 완전성을 주장할 수 없다 — 조용히 쓰면 안 된다."""
        Run.objects.filter(kind="detect").delete()
        self.box(self.swept, "yolo11", cls="fin")
        add, reject, imgs, note = self.cmd()._from_reviews(set())
        self.assertEqual((add, reject, imgs), ({}, set(), set()))
        self.assertTrue(any("infer_boxes" in n for n in note))


class DrawFromScratchTests(TestCase):
    """**마스크가 없는 상자에서 윤곽을 처음부터 그릴 수 있나.**

    편집 화면은 이미 있는 점을 끌고 변 가운데를 눌러 늘리는 식이라, 점이 하나도
    없으면 **누를 데가 없다.** 분할 모델이 아무것도 못 낸 상자가 그렇고, 그중
    일부는 사람이 "등지느러미다" 라고 판정한 것이라 여기 말고는 윤곽을 만들
    길이 없다. 출발점은 프롬프트 상자다 — 밑동 첫 제안과 같은 이유로,
    **대충의 제안은 사람의 눈을 끌고 가지 않는다** (`finseg/baseline.py`).
    """

    def setUp(self):
        img = Image.objects.create(path="t/d.jpg", obsdate="2020-01-01",
                                   width=1000, height=800)
        self.box = Box.objects.create(image=img, x1=100, y1=200, x2=300, y2=400,
                                      source="yolo11", conf=0.4)
        self.crop = Crop.objects.create(box=self.box, path="t/d0.jpg",
                                        x0=50, y0=150, x1=690, y1=790,
                                        w=640, h=640)

    def test_prompt_box_comes_through_in_crop_coordinates(self):
        """**이 식은 `geometry` 한 곳에만 있어야 한다** — 저장할 때 되돌리는
        식과 어긋나면 사람이 그린 윤곽이 엉뚱한 자리에 적힌다."""
        html = self.client.get(f"/edit/{self.box.id}").content.decode()
        m = re.search(r"const BOX = (\[[^\]]*\])", html)
        self.assertIsNotNone(m, "편집 화면이 프롬프트 상자를 안 보낸다")
        got = json.loads(m.group(1))
        want = geometry.to_crop([(self.box.x1, self.box.y1),
                                 (self.box.x2, self.box.y2)], self.crop)
        self.assertEqual([round(v, 1) for xy in want for v in xy], got)

    def test_the_screen_offers_a_way_to_start(self):
        """화면이 말하지 않으면 없는 것과 같다 — `e`/`x` 를 못 찾은 그 값이다."""
        html = self.client.get(f"/edit/{self.box.id}").content.decode()
        self.assertIn("윤곽 새로 그리기", html)


class NoShapeQueueTests(TestCase):
    """**윤곽을 말할 수 없는 상자가 어디에도 안 뜨던 것.**

    판정이 붙으면 `검토할 것`·`새 검출` 에서 빠지고, 마스크가 없으면
    `엔진 바뀜` 에 안 걸리고, `verdict='fix'` 가 아니면 `교정 대기` 에도 안
    걸린다. 그동안 화면은 "남은 일 없다" 고 말하는데 `export_yolo` 는 그
    상자가 든 크롭을 통째로 뺐다 — **같은 크롭에 걸친 남의 상자까지 함께.**
    `교정 대기` 를 만든 것과 똑같은 자리라 같은 방식으로 푼다.
    """

    def setUp(self):
        img = Image.objects.create(path="t/n.jpg", obsdate="2020-01-01",
                                   width=1000, height=800)
        self.run = Run.objects.create(kind="sam2")
        self.mk = lambda **kw: Box.objects.create(
            image=img, x1=10, y1=10, x2=60, y2=60, **kw)

    def ids(self):
        r = self.client.get("/api/batch?mode=noshape").json()
        return [t["box_id"] for t in r["tiles"]]

    def test_a_judged_box_with_no_shape_shows_up(self):
        b = self.mk(source="yolo11", conf=0.3)
        Crop.objects.create(box=b, path="t/n0.jpg", x0=0, y0=0,
                            x1=640, y1=640, w=640, h=640)
        Review.objects.create(box=b, cls="fin")
        self.assertEqual(self.ids(), [b.id])
        self.assertEqual(rules.progress()["윤곽없음"], 1)

    def test_a_mask_takes_it_out(self):
        b = self.mk(source="yolo11", conf=0.3)
        Crop.objects.create(box=b, path="t/n1.jpg", x0=0, y0=0,
                            x1=640, y1=640, w=640, h=640)
        Review.objects.create(box=b, cls="fin")
        Mask.objects.create(box=b, run=self.run, is_current=True,
                            polygon="20,20 50,20 35,50")
        self.assertEqual(self.ids(), [])

    def test_a_hand_drawn_polygon_takes_it_out(self):
        """사람이 그렸으면 마스크가 없어도 말할 수 있다."""
        b = self.mk(source="yolo11", conf=0.3)
        Crop.objects.create(box=b, path="t/n2.jpg", x0=0, y0=0,
                            x1=640, y1=640, w=640, h=640)
        Review.objects.create(box=b, cls="fin", polygon="20,20 50,20 35,50")
        self.assertEqual(self.ids(), [])
        self.assertEqual(rules.progress()["윤곽없음"], 0)

    def test_none_is_not_in_it(self):
        """`아무것도아님` 은 배경으로 나간다 — 윤곽이 필요 없다."""
        b = self.mk(source="yolo11", conf=0.3)
        Crop.objects.create(box=b, path="t/n3.jpg", x0=0, y0=0,
                            x1=640, y1=640, w=640, h=640)
        Review.objects.create(box=b, cls="none")
        self.assertEqual(self.ids(), [])
        self.assertEqual(rules.progress()["윤곽없음"], 0)

    def test_fins_come_first_then_confidence(self):
        """도중에 멈춰도 얻는 것이 있게 값이 큰 것부터 낸다."""
        made = []
        for i, (cls, conf) in enumerate(
                [("body", 0.9), ("fin", 0.2), ("fin", 0.8)]):
            b = self.mk(source="yolo11", conf=conf)
            Crop.objects.create(box=b, path=f"t/n{i}x.jpg", x0=0, y0=0,
                                x1=640, y1=640, w=640, h=640)
            Review.objects.create(box=b, cls=cls)
            made.append((cls, conf, b.id))
        self.assertEqual(self.ids(), [made[2][2], made[1][2], made[0][2]])


class PolygonEditFlagTests(SimpleTestCase):
    """**점을 옮긴 것이 '고쳤다' 로 세어지나.**

    `dragVertices` 가 `polyMoved` 를 안 세워서, 점을 아무리 끌어도 저장이 그
    윤곽을 버렸다. 서버는 `polygon_edited` 가 거짓이면 폴리곤을 안 적는데
    (`views.save` — 안 건드린 제안이 사람의 판단인 척 남지 않게 하는 규칙)
    **화면은 200 을 받으므로 아무 표시도 안 났다.** 셋 중 둘을 그렇게 잃었다.

    브라우저 쪽을 돌려 볼 자리가 아직 없어 **글자로 잰다.** 성긴 시험이지만
    잡으려는 것은 분명하다 — `pts` 를 건드리는 길에 깃발이 빠지는 것.
    `itemOf` 가 두 축을 조용히 버린 것과 같은 종류다.
    """

    SRC = "review/templates/review/edit.html"

    def src(self):
        with open(self.SRC, encoding="utf-8") as f:
            return f.read()

    def block(self, name):
        """`function <name>(` 부터 다음 함수 정의 전까지."""
        s = self.src()
        i = s.index(f"function {name}(")
        j = s.find("\nfunction ", i + 1)
        return s[i:j if j > 0 else len(s)]

    def test_dragging_points_counts_as_an_edit(self):
        self.assertIn("polyMoved = true", self.block("dragVertices"),
                      "점을 끌어도 '고쳤다' 가 안 서면 저장이 윤곽을 버린다")

    def test_the_flag_waits_for_the_drag_threshold(self):
        """눌렀다 떼기만 한 것은 고친 것이 아니다 — `push()` 와 같은 자리다."""
        b = self.block("dragVertices")
        self.assertIn("dragging = true; push(); polyMoved = true;", b)

    def test_commit_checks_the_shape_not_just_the_flag(self):
        """깃발 하나에 기대면 다음에 또 같은 자리에서 샌다."""
        c = self.block("commit")
        self.assertIn("shape(T.points)", c)
        self.assertIn("seeded", c)

    def test_an_untouched_seed_still_does_not_save(self):
        """상자 네모가 사람이 그린 마스크인 척 들어가면 안 된다."""
        self.assertIn("seeded = shape(out)", self.block("seed"))

    def test_facing_can_be_set_on_the_edit_screen(self):
        """화면은 앞쪽을 그리고 저장도 실어 보내는데 **누를 자리가 없었다** —
        `HANDOFF` 는 여기서 `f` 가 된다고 적어 두었고 실제로는 안 됐다."""
        s = self.src()
        self.assertIn('e.key === "f"', s)
        self.assertIn("facingMoved = true", s)


class HandDrawnIsAnAnswerTests(TestCase):
    """**사람이 그린 윤곽이 자료에 닿나.**

    `verdict`("이 마스크가 맞나")는 마스크가 있어야 물을 수 있는 말이라 마스크
    없는 상자에는 서버가 안 적는다. 그런데 그 상자에 사람이 **직접 윤곽을
    그리면** 모양은 생겼는데 `verdict` 가 빈 채로 남아 `label_of` 가 `PENDING`
    을 냈고, `export_yolo` 는 그 상자가 든 크롭을 통째로 뺐다 — 그리는 일
    자체가 자료에 안 닿았다. `윤곽 없음` 을 다 비우고도 48개가 그랬다.
    """

    def setUp(self):
        img = Image.objects.create(path="t/hd.jpg", obsdate="2020-01-01",
                                   width=1000, height=800)
        self.box = Box.objects.create(image=img, x1=10, y1=10, x2=60, y2=60,
                                      source="yolo11", conf=0.5)
        Crop.objects.create(box=self.box, path="t/hd0.jpg", x0=0, y0=0,
                            x1=640, y1=640, w=640, h=640)

    def test_a_drawn_outline_is_positive_without_a_verdict(self):
        rv = Review.objects.create(box=self.box, cls="fin", verdict="",
                                   polygon="20,20 50,20 35,50")
        self.assertEqual(rules.label_of(rv), rules.POSITIVE)

    def test_nothing_drawn_is_still_pending(self):
        """빈 `verdict` 자체가 답이 되면 안 된다 — 그것은 안 물은 것이다."""
        rv = Review.objects.create(box=self.box, cls="fin", verdict="")
        self.assertEqual(rules.label_of(rv), rules.PENDING)

    def test_drawing_it_through_the_save_path_lands_as_positive(self):
        """화면이 보내는 그대로 넣어 본다 — 규칙과 저장이 갈라지면 못 잡는다."""
        res = self.client.post("/api/review", data=json.dumps({"items": [
            {"box_id": self.box.id, "cls": "fin", "edges": "both",
             "verdict": "fix", "facing": "",
             "polygon": [[20, 20], [50, 20], [35, 50]],
             "polygon_edited": True}]}),
            content_type="application/json")
        self.assertEqual(res.status_code, 200)
        rv = Review.objects.get(box=self.box)
        self.assertEqual(rv.verdict, "")      # 마스크가 없으니 안 적는다
        self.assertNotEqual(rv.polygon, "")   # 그린 것은 남는다
        self.assertEqual(rules.label_of(rv), rules.POSITIVE)

    def test_none_stays_background_even_with_a_polygon(self):
        rv = Review.objects.create(box=self.box, cls="none",
                                   polygon="20,20 50,20 35,50")
        self.assertEqual(rules.label_of(rv), rules.BACKGROUND)


class OnnxPreprocessTests(SimpleTestCase):
    """**전처리가 어긋나도 에러가 안 난다** — 상자가 조금 밀린 채로 나올 뿐이다.

    `.onnx` 하나를 파이썬·브라우저·(나중에) 데스크톱이 함께 쓰는데, 셋이 같은
    상자를 내야 한다. `finseg.onnxdet` 이 그 기준이고 여기서 그 산수를 잰다 —
    좌표 사상을 `geometry.py` 한 곳에만 두는 것과 같은 이유다.

    모델을 안 부른다. 부르는 시험은 가중치와 GPU 를 요구해서 `manage.py test`
    가 0.2초에 끝나는 것을 깬다 — 그쪽은 따로 돌린다.
    """

    def test_letterbox_keeps_the_aspect_and_centers(self):
        r, dx, dy, nw, nh = onnxdet.letterbox(4928, 3280, 1280)
        self.assertAlmostEqual(r, 1280 / 4928)
        self.assertEqual(nw, 1280)              # 긴 변이 꽉 찬다
        self.assertEqual(dx, 0)
        self.assertEqual(nh, round(3280 * r))
        self.assertEqual(dy, (1280 - nh) // 2)  # 짧은 변은 가운데
        self.assertAlmostEqual(nw / nh, 4928 / 3280, places=2)

    def test_a_square_photo_fills_the_canvas(self):
        r, dx, dy, nw, nh = onnxdet.letterbox(640, 640, 1280)
        self.assertEqual((dx, dy, nw, nh), (0, 0, 1280, 1280))

    def test_nms_drops_the_overlapping_and_keeps_the_far_one(self):
        boxes = np.array([[100., 100., 200., 200.],   # 확신 0.9
                          [105., 105., 205., 205.],   # 거의 같은 자리
                          [400., 400., 500., 500.]])  # 다른 자리
        keep = onnxdet.nms(boxes, np.array([0.9, 0.8, 0.7]), 0.7)
        self.assertEqual(sorted(keep), [0, 2])

    def test_nms_keeps_neighbours_that_merely_touch(self):
        """돌고래는 무리로 다녀 지느러미가 붙는다 — 너무 지우면 손해다."""
        boxes = np.array([[100., 100., 200., 200.],
                          [180., 100., 280., 200.]])   # IoU 약 0.11
        self.assertEqual(sorted(onnxdet.nms(boxes, np.array([0.9, 0.8]), 0.7)),
                         [0, 1])

    def test_boxes_come_back_in_original_coordinates(self):
        """**여백을 빼고 배율로 나눈다.** 이 두 줄이 어긋나면 상자가 밀린다."""
        w, h = 4928, 3280
        r, dx, dy, _, _ = onnxdet.letterbox(w, h, 1280)
        # 원본 한가운데의 상자를 레터박스 좌표로 옮겨 모델 출력인 척 만든다
        want = (2000.0, 1500.0, 2400.0, 1800.0)
        cx = ((want[0] + want[2]) / 2) * r + dx
        cy = ((want[1] + want[3]) / 2) * r + dy
        bw, bh = (want[2] - want[0]) * r, (want[3] - want[1]) * r
        out = np.array([[[cx], [cy], [bw], [bh], [0.9]]])   # (1, 5, 1)
        got = onnxdet.postprocess(out, (r, dx, dy, w, h))
        self.assertEqual(len(got), 1)
        for a, b in zip(got[0][:4], want):
            self.assertAlmostEqual(a, b, places=1)

    def test_low_confidence_is_dropped_before_nms(self):
        out = np.array([[[100.], [100.], [50.], [50.], [0.05]]])
        meta = (1.0, 0, 0, 1280, 1280)
        self.assertEqual(onnxdet.postprocess(out, meta, conf_thres=0.25), [])
        self.assertEqual(len(onnxdet.postprocess(out, meta, conf_thres=0.01)), 1)

    def test_it_takes_either_axis_order(self):
        """내보내기 판에 따라 `(1,5,N)` 이기도 `(1,N,5)` 이기도 하다."""
        meta = (1.0, 0, 0, 1280, 1280)
        a = np.array([[[100.], [100.], [50.], [50.], [0.9]]])      # (1,5,1)
        b = np.array([[[100., 100., 50., 50., 0.9]]])              # (1,1,5)
        self.assertEqual(onnxdet.postprocess(a, meta),
                         onnxdet.postprocess(b, meta))

    def test_downscaling_matches_cv2(self):
        """**부드럽게 줄이면 작은 지느러미가 사라진다.**

        학습(ultralytics)은 `cv2.resize(INTER_LINEAR)` 로 줄인 그림을 봤다.
        `PIL.Image.resize(BILINEAR)` 은 축소할 때 필터를 넓혀 안티에일리어싱을
        하는데, 그러면 **학습이 본 적 없는 그림**이 된다. 실측으로 29×21px
        상자의 확신이 0.36 → 0.17 로 반토막 났고 문턱 0.25 에서 없어졌다 —
        이 검출기가 여는 것의 3분의 2가 끄트머리만 보이는 작은 지느러미라
        하필 그쪽부터 먼저 사라진다.

        그래서 `PIL.Image.transform(AFFINE, BILINEAR)` 을 쓴다. **여기서 재는
        것은 그것이 정말 `cv2` 와 같은가**다 — 닮았다는 짐작이 아니라.
        """
        try:
            import cv2
        except ImportError:
            self.skipTest("cv2 가 없다 (검토 전용 설치)")
        from PIL import Image as PILImage
        rng = np.random.default_rng(3)
        a = (rng.random((512, 512, 3)) * 255).astype(np.uint8)
        a[63:66, 63:66] = 255              # 축소 배수보다 작은 밝은 점
        x, _ = onnxdet.preprocess(PILImage.fromarray(a), size=128)
        got = x[0].transpose(1, 2, 0) * 255
        want = cv2.resize(a, (128, 128),
                          interpolation=cv2.INTER_LINEAR).astype(np.float32)
        self.assertLess(np.abs(got - want).max(), 2.0,
                        "축소가 cv2 와 다르다 — 작은 지느러미가 조용히 사라진다")


class DetectPageTests(TestCase):
    """**받아 와야 하는 것이 없으면 화면이 왜 없는지 말하나.**

    `/detect` 는 저장소 밖의 것 둘에 기댄다 — `onnxruntime-web`(26MB)과
    내보낸 `.onnx`(37MB). 둘 다 `.gitignore` 라 새 클론에는 없다. 그때
    **조용히 빈 화면을 내면 무엇이 빠졌는지 알 길이 없다** — 끊어진 심볼릭
    링크가 빈 디렉토리처럼 보이던 것과 같은 함정이다(`CLAUDE.md`).
    """

    def page(self):
        return self.client.get("/detect").content.decode()

    def test_it_names_what_is_missing(self):
        html = self.page()
        # 시험 환경에는 받아 온 것이 없다 — 그러면 받는 법이 떠야 한다
        if "받아 와야 하는 것이 없다" in html:
            self.assertIn("onnxruntime-web", html)
            self.assertIn("yolo export", html)
        else:                       # 받아 둔 기계에서는 화면이 떠야 한다
            self.assertIn('id="drop"', html)

    def test_the_rule_numbers_come_from_one_place(self):
        """화면에 박아 두면 `onnxdet` 을 고쳐도 JS 가 옛 값을 쓴다.

        **받아 온 것이 없는 기계에서는 잴 것이 없다** — 그때 뜨는 것은 받는
        법이지 화면이 아니라서 `PAD`(114)가 나올 자리가 아예 없다. 형제 시험
        둘과 같은 자리에서 가린다. 그렇다고 이 시험이 헐거워지지는 않는다 —
        받아 둔 기계(2080ti)에서는 그대로 돈다.
        """
        html = self.page()
        if 'id="drop"' not in html:
            self.skipTest("`/detect` 에 받아 온 것이 없다 — 잴 화면이 없다")
        self.assertIn(str(onnxdet.IMGSZ), html)
        self.assertIn(str(onnxdet.PAD), html)

    def test_it_says_smoothing_must_be_off(self):
        """canvas 는 기본으로 부드럽게 줄인다 — 그러면 작은 지느러미가 사라진다."""
        html = self.page()
        if 'id="drop"' in html:
            self.assertIn("imageSmoothingEnabled = false", html)


class ImportDetectionsTests(TestCase):
    """**다른 기계가 훑은 결과를 들이는 자리.**

    옛 운영 DB 를 날짜별로 펴 보니 관찰일 255일 중 28일(사진의 0.78%)만 썼다.
    나머지를 훑으려면 이 기계 말고도 돌아야 하고, 그 결과가 루프로 돌아올 길이
    있어야 한다. `infer_boxes` 와 **같은 규칙**으로 들이는지가 여기서 잴 것이다.
    """

    def setUp(self):
        from django.core.management import call_command
        self.call = call_command
        img = Image.objects.create(path="t/i.jpg", obsdate="2020-01-01",
                                   width=1000, height=800)
        self.img = img
        Box.objects.create(image=img, x1=100, y1=100, x2=200, y2=200,
                           source="yolov5")

    def scan(self, boxes, path="t/i.jpg", **kw):
        import json as _json
        import tempfile
        d = {"model": "detect-v2", "conf": 0.05, "imgsz": 1280,
             "photos": [{"path": path, "w": 1000, "h": 800, "boxes": boxes}]}
        d.update(kw)
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        _json.dump(d, f)
        f.close()
        return f.name

    def run_it(self, boxes, **kw):
        out = io.StringIO()
        self.call("import_detections", self.scan(boxes, **kw),
                  src_db="", stdout=out)
        return out.getvalue()

    def test_a_new_box_is_taken(self):
        self.run_it([[400, 400, 500, 500, 0.42]])
        b = Box.objects.get(source="detect-v2")
        self.assertEqual((b.x1, b.y1, b.x2, b.y2), (400, 400, 500, 500))
        self.assertAlmostEqual(b.conf, 0.42)

    def test_a_box_we_already_know_is_dropped(self):
        """`새 검출` 은 **옛 상자와 안 겹치는 것뿐**이어야 한다."""
        self.run_it([[102, 102, 198, 198, 0.9]])
        self.assertFalse(Box.objects.filter(source="detect-v2").exists())

    def test_a_narrow_box_inside_an_old_one_is_also_known(self):
        """새 검출기가 좁게 따는 버릇이 있어 IoU 만 보면 남처럼 보인다."""
        self.run_it([[140, 140, 160, 160, 0.9]])
        self.assertFalse(Box.objects.filter(source="detect-v2").exists())

    def test_two_boxes_in_one_scan_do_not_double_up(self):
        self.run_it([[400, 400, 500, 500, 0.5], [402, 402, 498, 498, 0.4]])
        self.assertEqual(Box.objects.filter(source="detect-v2").count(), 1)

    def test_a_tiny_box_is_dropped(self):
        self.run_it([[400, 400, 405, 405, 0.9]])
        self.assertFalse(Box.objects.filter(source="detect-v2").exists())

    def test_it_refuses_when_the_model_is_unnamed(self):
        """어느 검출기가 낸 것인지 모르면 문턱별 정밀도를 되읽을 수 없다."""
        from django.core.management.base import CommandError
        f = self.scan([[400, 400, 500, 500, 0.5]])
        import json as _json
        d = _json.load(open(f)); d.pop("model")
        _json.dump(d, open(f, "w"))
        with self.assertRaises(CommandError):
            self.call("import_detections", f, src_db="", stdout=io.StringIO())

    def test_an_unknown_photo_is_made_and_the_missing_date_is_said(self):
        """`obsdate` 가 비면 `export_detect` 의 날짜 가르기에 안 잡힌다."""
        out = self.run_it([[400, 400, 500, 500, 0.5]], path="t/새것.jpg")
        img = Image.objects.get(path="t/새것.jpg")
        self.assertIsNone(img.obsdate)
        self.assertIn("관찰일을 모른다", out)

    def test_it_warns_when_the_old_db_is_absent(self):
        """`fin.db` 만 보면 옛 검출기가 이미 찾은 것이 '새 검출' 로 둔갑한다."""
        out = self.run_it([[400, 400, 500, 500, 0.5]])
        self.assertIn("이미 아는 상자를 못 걸러낸다", out)


@override_settings(ROOT_URLCONF="review.tests", FIN_ROLE="reid")
class ReidTests(TestCase):
    """**개체 판정은 쌓인다, 덮어쓰지 않는다** — `Review` 와 같은 규칙이다.

    카탈로그가 자라면 판정이 갈라지거나 합쳐진다. 자취가 없으면 언제 생각이
    바뀌었는지 잴 수 없다. 상자에서 뺀 것도 줄이 남는다.
    """

    def setUp(self):
        img = Image.objects.create(path="t/r.jpg", obsdate="2020-01-01",
                                   width=1000, height=800)
        self.a, self.b, self.c = [
            Box.objects.create(image=img, x1=10 * i, y1=10, x2=60 * (i + 1),
                               y2=60, source="yolov5") for i in range(3)]

    def box(self, **kw):
        return self.client.post("/api/reid/box", data=json.dumps(kw),
                                content_type="application/json")

    def assign(self, **kw):
        return self.client.post("/api/reid/assign", data=json.dumps(kw),
                                content_type="application/json")

    def test_a_box_is_made_without_a_name(self):
        """**먼저 모아 놓고 보아야 누구인지 정할 수 있다.** 이름부터 물으면
        거기서 손이 멎는다."""
        r = self.box()
        self.assertEqual(r.status_code, 200)
        self.assertTrue(Individual.objects.filter(id=r.json()["id"]).exists())
        self.assertTrue(r.json()["name"])

    def test_names_do_not_collide(self):
        a, b = self.box().json(), self.box().json()
        self.assertNotEqual(a["name"], b["name"])

    def test_renaming_refuses_a_taken_name(self):
        a, b = self.box().json(), self.box().json()
        self.assertEqual(self.box(id=b["id"], name=a["name"]).status_code, 400)

    def test_many_go_in_at_once(self):
        """같은 개체를 한 번에 넣는 것이 이 화면의 목적이다."""
        ind = self.box().json()
        self.assign(individual=ind["id"], boxes=[self.a.id, self.b.id])
        self.assertEqual(reid.catalog()[ind["id"]].sort(),
                         [self.a.id, self.b.id].sort())

    def test_moving_leaves_a_trail_and_the_latest_wins(self):
        one, two = self.box().json(), self.box().json()
        self.assign(individual=one["id"], boxes=[self.a.id])
        self.assign(individual=two["id"], boxes=[self.a.id])
        self.assertEqual(Identification.objects.filter(box=self.a).count(), 2)
        self.assertEqual(reid.effective_id(self.a).individual_id, two["id"])
        self.assertEqual(reid.catalog(), {two["id"]: [self.a.id]})

    def test_taking_it_out_leaves_the_catalog(self):
        """뺀 것도 답이다 — 자취는 남되 카탈로그에서는 빠진다."""
        ind = self.box().json()
        self.assign(individual=ind["id"], boxes=[self.a.id])
        self.assign(individual=None, boxes=[self.a.id])
        self.assertEqual(reid.catalog(), {})
        self.assertEqual(reid.decided([self.a.id]), {self.a.id})

    def test_it_refuses_an_unknown_box(self):
        self.assertEqual(self.assign(individual=9999,
                                     boxes=[self.a.id]).status_code, 404)
        self.assertFalse(Identification.objects.exists())

    def test_the_screen_says_what_is_missing(self):
        """빈 화면은 아무 말도 안 한다."""
        html = self.client.get("/reid").content.decode()
        if "분류할 것이 없다" in html:
            self.assertIn("items.json", html)
        else:
            self.assertIn("const ITEMS = ", html)

    def test_the_page_plants_the_csrf_cookie(self):
        """**끌어 넣을 때마다 POST 한다.** 쿠키가 없으면 서버가 403 을 내는데,
        화면은 "서버가 403 을 냈다" 만 말하고 무엇이 없는지는 안 말한다."""
        r = self.client.get("/reid")
        self.assertIn("csrftoken", r.cookies)

    def test_boxes_come_in_the_order_they_were_made(self):
        """이름순이면 `상자 10` 이 `상자 2` 앞에 오고, **이름을 고치는 순간
        그 상자가 목록에서 튀어 다닌다** — 방금 이름 붙인 것을 다시 찾아야 한다."""
        made = [self.box().json()["id"] for _ in range(3)]
        self.box(id=made[0], name="힣마지막이름")     # 이름순이면 맨 뒤로 갈 것
        html = self.client.get("/reid").content.decode()
        m = re.search(r"let BOXES = (\[.*?\]);", html, re.S)
        self.assertIsNotNone(m)
        # 보류함은 화면을 열 때 저절로 생긴다 — 개체가 아니라 자리다
        got = [b["id"] for b in json.loads(m.group(1)) if not b["hold"]]
        self.assertEqual(got, made)

    def test_a_representative_can_be_chosen_and_cleared(self):
        """**아무거나 하나를 보이면 하필 흐린 것이 대표가 된다** — 사람이 고른다."""
        ind = self.box().json()
        self.assign(individual=ind["id"], boxes=[self.a.id, self.b.id])
        r = self.box(id=ind["id"], rep=self.b.id)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(Individual.objects.get(id=ind["id"]).rep_id, self.b.id)
        self.box(id=ind["id"], rep=None)
        self.assertIsNone(Individual.objects.get(id=ind["id"]).rep_id)

    def test_the_page_carries_the_representative(self):
        ind = self.box().json()
        self.assign(individual=ind["id"], boxes=[self.a.id])
        self.box(id=ind["id"], rep=self.a.id)
        html = self.client.get("/reid").content.decode()
        m = re.search(r"let BOXES = (\[.*?\]);", html, re.S)
        # **번호로 찾는다.** 자리에 기대면 개체가 아닌 자리(임시보관함·
        # 지느러미 아님)가 앞에 서면서 깨진다
        got = {b["id"]: b for b in json.loads(m.group(1))}
        self.assertEqual(got[ind["id"]]["rep"], self.a.id)

    def test_the_holding_box_is_made_and_is_not_a_catalog_entry(self):
        """**보류함은 개체가 아니라 자리다.** 어디에 넣을지 모르겠는 것을
        아무 상자에나 넣으면 그 상자가 오염되고, 미분류로 두면 다음에 또 같은
        고민을 처음부터 한다. 그렇다고 카탈로그에 세면 개체 수가 틀린다."""
        self.client.get("/reid")
        hold = Individual.objects.get(kind="hold")
        self.assign(individual=hold.id, boxes=[self.a.id])
        self.assertEqual(reid.catalog(), {})           # 개체로 안 센다
        self.assertEqual(reid.effective_id(self.a).individual_id, hold.id)
        self.assertEqual(reid.decided([self.a.id]), {self.a.id})

    def test_moving_out_of_holding_leaves_holding(self):
        """**보류함에서 상자로 옮기면 보류함에서 빠져야 한다.**

        보류함은 개체가 아니라 자리라 `catalog()` 가 안 세는데, 그것만 덮여
        있었고 **거기서 나가는 길**은 안 덮여 있었다. 실제로 그 의심이 한 번
        나왔고 확인하는 데 손이 갔다 — 시험이 있으면 한 줄로 끝난다.
        """
        self.client.get("/reid")                 # 보류함이 여기서 생긴다
        hold = Individual.objects.get(kind="hold")
        ind = self.box().json()
        self.assign(individual=hold.id, boxes=[self.a.id, self.b.id])
        self.assign(individual=ind["id"], boxes=[self.a.id])

        # 유효 판정은 새 상자다. 보류함에는 안 남는다
        self.assertEqual(reid.effective_id(self.a).individual_id, ind["id"])
        latest = {}
        for box_id, i in (Identification.objects.order_by("id")
                          .values_list("box_id", "individual_id")):
            latest[box_id] = i
        self.assertEqual([b for b, i in latest.items() if i == hold.id],
                         [self.b.id])
        # 자취는 남는다 — 언제 생각이 바뀌었는지 잴 수 있어야 한다
        self.assertEqual(
            [x.individual_id for x in
             Identification.objects.filter(box=self.a).order_by("id")],
            [hold.id, ind["id"]])

    def test_the_page_shows_the_move(self):
        """화면이 받는 값도 함께 잰다 — DB 는 맞는데 화면이 안 따라오는 것이
        이 저장소에서 가장 자주 겪은 모양이다."""
        self.client.get("/reid")
        hold = Individual.objects.get(kind="hold")
        ind = self.box().json()
        self.assign(individual=hold.id, boxes=[self.a.id])
        self.assign(individual=ind["id"], boxes=[self.a.id])
        html = self.client.get("/reid").content.decode()
        m = re.search(r"const ITEMS = (\[.*?\]);", html, re.S)
        if m:                       # 조각 목록이 있는 기계에서만 잴 수 있다
            by = {i["id"]: i for i in json.loads(m.group(1))}
            if self.a.id in by:
                self.assertEqual(by[self.a.id]["in"], ind["id"])
        # 상자 목록의 장수도 보류함이 아니라 새 상자에 붙어야 한다
        b = json.loads(re.search(r"let BOXES = (\[.*?\]);", html, re.S).group(1))
        n = {x["id"]: x["n"] for x in b}
        self.assertEqual(n[ind["id"]], 1)
        self.assertEqual(n[hold.id], 0)


class ReidEvalTests(TestCase):
    """**자를 재는 시험이다.** `reid_eval` 이 내는 표가 판단의 근거가 되는데,
    순위 계산이 조용히 틀리면 그 표는 멀쩡해 보이면서 방법을 잘못 고르게 한다.

    화면이 아니라 **아는 답을 넣고 나오는 숫자**로 잰다.
    """

    def cmd(self):
        from finseg.management.commands import reid_eval
        return reid_eval.Command()

    def test_the_rank_of_the_first_hit_is_one_based(self):
        """`첫 정답 1위` 는 맨 앞이라는 뜻이다 — 0위가 나오면 표가 한 칸씩
        낙관 쪽으로 어긋난다."""
        import numpy as np
        c = self.cmd()
        sims = np.array([9.0, 8.0, 7.0])
        valid = np.array([True, True, True])
        pos = np.array([True, False, False])
        top1, first, ap = c._score(sims, valid, pos)
        self.assertTrue(top1)
        self.assertEqual(first, 1)
        self.assertAlmostEqual(ap, 1.0)

    def test_a_hidden_candidate_does_not_take_a_rank(self):
        """후보에서 뺀 것(같은 날·반대쪽)이 순위를 차지하면 안 된다.
        **자리만 차지하고 사라지면 정답의 순위가 뒤로 밀린다.**"""
        import numpy as np
        c = self.cmd()
        sims = np.array([9.0, 8.0, 7.0])
        valid = np.array([False, True, True])      # 맨 앞을 뺀다
        pos = np.array([False, False, True])
        top1, first, ap = c._score(sims, valid, pos)
        self.assertFalse(top1)
        self.assertEqual(first, 2)                 # 3위가 아니라 2위다

    def test_it_says_nothing_when_the_answer_is_not_a_candidate(self):
        """정답이 후보에 없으면 **못 잰 것**이지 0점이 아니다."""
        import numpy as np
        c = self.cmd()
        r = c._score(np.array([1.0, 2.0]), np.array([True, True]),
                     np.array([False, False]))
        self.assertIsNone(r)

    def test_average_precision_counts_every_hit(self):
        """정답이 둘이면 mAP 는 둘을 다 본다 — 첫 정답만 보는 자와 다르다."""
        import numpy as np
        c = self.cmd()
        # 순위 1·3 이 정답: (1/1 + 2/3) / 2
        _, first, ap = c._score(np.array([9.0, 8.0, 7.0]),
                                np.array([True, True, True]),
                                np.array([True, False, True]))
        self.assertEqual(first, 1)
        self.assertAlmostEqual(ap, (1.0 + 2 / 3) / 2)


class ReidChipRuleTests(TestCase):
    """**조각을 무엇으로 거르나 — 그 값이 자료에 적혀 있어야 한다.**

    `reid/v1` 은 넓이 문턱 15,000 으로 만들어졌는데 `reid.MIN_AREA` 는 그 뒤에
    30,000 이 되었다. 모르고 다시 만들면 **사람이 이미 분류한 조각 1,034장이
    격자에서 조용히 사라진다** — 화면은 멀쩡하고 수만 줄어든다.
    """

    def test_the_threshold_is_written_next_to_the_chips(self):
        """격자가 무엇으로 걸러졌는지 자료 자체가 들고 있어야 한다."""
        import json
        from pathlib import Path
        p = Path("reid/v1/items.json")
        if not p.exists():
            self.skipTest("이 기계에 조각이 없다")
        d = json.loads(p.read_text())
        # 옛 격자에는 아직 없다 — 그래서 새로 만드는 것부터 적는다
        if "min_area" in d:
            self.assertIsInstance(d["min_area"], int)

    def test_usable_reads_the_threshold_at_call_time(self):
        """`--min-area` 가 먹으려면 `usable` 이 모듈 값을 **부를 때** 읽어야 한다.
        기본값으로 묶여 있으면 옵션이 조용히 무시된다."""
        from finseg import reid
        img = Image.objects.create(path="t/c.jpg", obsdate="2020-01-01",
                                   width=1000, height=800)
        box = Box.objects.create(image=img, x1=0, y1=0, x2=140, y2=140,
                                 source="yolov5")     # 19,600px²
        state = {"cls": "fin", "polygon": "1,1 2,2 3,3", "base_line": "1,1 3,3",
                 "base_partial": "", "facing": "left", "review": None, "box": box}
        old = reid.MIN_AREA
        try:
            reid.MIN_AREA = 15000
            self.assertTrue(reid.usable(state)[0])
            reid.MIN_AREA = 30000
            ok, why = reid.usable(state)
            self.assertFalse(ok)
            self.assertIn("작다", why)
        finally:
            reid.MIN_AREA = old

    def test_an_unreviewed_box_only_lacks_the_facing(self):
        """**검토 안 된 상자에서 정말 비는 축이 무엇인가.**

        `edges`·`base_partial` 은 판정이 없으면 `usable` 이 통과값으로 읽는다 —
        `TODOs` 가 "대체가 없다" 고 적어 둔 두 축인데, 막는 것은 `facing` 이다.
        여기가 바뀌면 옛 상자를 들이는 경로가 통째로 달라진다.
        """
        from finseg import reid
        img = Image.objects.create(path="t/u.jpg", obsdate="2020-01-01",
                                   width=1000, height=800)
        box = Box.objects.create(image=img, x1=0, y1=0, x2=200, y2=200,
                                 source="yolov5")
        state = {"cls": "fin", "polygon": "1,1 2,2 3,3", "base_line": "1,1 3,3",
                 "base_partial": "", "facing": "", "review": None, "box": box}
        ok, why = reid.usable(state)
        self.assertFalse(ok)
        self.assertIn("앞쪽", why)
        state["facing"] = "left"                       # 기하 제안이 메우는 자리
        self.assertTrue(reid.usable(state)[0])


class ReidGroupTests(SimpleTestCase):
    """**묶는 규칙이 곧 제안의 품질이다.**

    실측으로 같은 날·같은 쪽에서 프레임 간격 ≤2 만 보면 75%, 조각 거리 ≤0.06
    만 보면 77%인데 **둘 다 보면 97%** 다. 하나라도 빠지면 잘못 묶인 것이
    한 번의 판단으로 여러 장에 번진다 — 낱장으로 틀리는 것보다 나쁘다.
    """

    def links(self, **kw):
        import numpy as np
        from finseg import reid
        n = len(kw["day"])
        return reid.group_links(np.arange(n), np.asarray(kw["emb"], float),
                                np.array(kw["day"]), np.array(kw["fac"]),
                                np.array(kw["frame"]))

    def test_close_and_consecutive_are_joined(self):
        g = self.links(emb=[[1, 0], [.999, .045]], day=["d", "d"],
                       fac=["left", "left"], frame=[10, 11])
        self.assertEqual([sorted(x) for x in g], [[0, 1]])

    def test_a_far_frame_is_not_joined_however_alike(self):
        """**닮았다고 잇지 않는다** — 같은 날 다른 개체가 닮은 일이 흔하다."""
        g = self.links(emb=[[1, 0], [1, 0]], day=["d", "d"],
                       fac=["left", "left"], frame=[10, 40])
        self.assertEqual(sorted(len(x) for x in g), [1, 1])

    def test_two_fins_in_one_photo_are_never_joined(self):
        """한 마리가 한 사진에 두 번 나올 수 없다 — 공짜로 얻는 제약이다."""
        g = self.links(emb=[[1, 0], [1, 0]], day=["d", "d"],
                       fac=["left", "left"], frame=[10, 10])
        self.assertEqual(sorted(len(x) for x in g), [1, 1])

    def test_sides_and_days_are_never_joined(self):
        g = self.links(emb=[[1, 0]] * 4, day=["d", "d", "e", "e"],
                       fac=["left", "right", "left", "left"], frame=[10, 11, 10, 11])
        self.assertEqual(sorted(len(x) for x in g), [1, 1, 2])

    def test_the_group_day_is_kept_out_of_its_own_suggestion(self):
        """**묶음의 날이 후보에 있으면 개체가 아니라 그날 조명을 맞힌다.**"""
        import numpy as np
        from finseg import reid
        emb = np.array([[1., 0], [1., 0], [0, 1.]])
        day = np.array(["d", "d", "e"])
        fac = np.array(["left"] * 3)
        # 개체 7 은 같은 날(d)에만 있고, 개체 9 는 다른 날(e)에 있다
        got = reid.suggest([0], emb, day, fac, {7: [1], 9: [2]})
        self.assertEqual([g[0] for g in got], [9])


class ReidChainTests(SimpleTestCase):
    """**화면이 이 값으로 정렬한다.** 없거나 어긋나면 `닮은 것끼리` 가 `NaN`
    비교가 되어 **조용히 아무 순서나 낸다** — 200 이 떨어지고 화면도 멀쩡한데
    정렬만 안 되는, 이 저장소가 가장 자주 겪은 종류다. 실제로 `reid/v2` 를
    만들 때 이 값을 빠뜨렸고 그대로 며칠 쓸 뻔했다.
    """

    def test_each_side_is_ranked_from_zero(self):
        import numpy as np
        from finseg import reid
        emb = np.array([[1., 0], [.9, .1], [0, 1.], [.1, .9], [.5, .5]])
        fac = np.array(["left", "left", "right", "right", "left"])
        out = reid.sim_chain(emb, fac)
        for side in ("left", "right"):
            r = sorted(out[fac == side].tolist())
            self.assertEqual(r, list(range(len(r))), f"{side} 쪽 등수가 0..n-1 이 아니다")

    def test_the_two_sides_are_never_woven_together(self):
        """좌·우를 한 줄에 섞으면 **서로 다른 두 면이 이웃**이 된다."""
        import numpy as np
        from finseg import reid
        emb = np.eye(4)
        fac = np.array(["left", "right", "left", "right"])
        out = reid.sim_chain(emb, fac)
        self.assertEqual(sorted(out[fac == "left"].tolist()), [0, 1])
        self.assertEqual(sorted(out[fac == "right"].tolist()), [0, 1])

    def test_neighbours_in_the_chain_are_more_alike_than_chance(self):
        """줄이 실제로 닮은 것끼리인가 — 그러라고 만든 값이다."""
        import numpy as np
        from finseg import reid
        rng = np.random.default_rng(3)
        # 세 무리를 만들어 둔다 — 줄이 무리를 따라 흘러야 한다
        emb = np.concatenate([rng.normal(c, .05, (12, 8))
                              for c in (np.r_[1., np.zeros(7)],
                                        np.r_[0, 1., np.zeros(6)],
                                        np.r_[0, 0, 1., np.zeros(5)])])
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        fac = np.array(["left"] * len(emb))
        order = np.argsort(reid.sim_chain(emb, fac))
        nb = np.mean([emb[order[i]] @ emb[order[i + 1]] for i in range(len(order) - 1)])
        rd = np.mean([emb[rng.integers(len(emb))] @ emb[rng.integers(len(emb))]
                      for _ in range(500)])
        self.assertGreater(nb, rd + .2, "줄 이웃이 무작위와 다를 바 없다")


class ReidProbeSplitTests(SimpleTestCase):
    """**배울 짝에 재는 쪽이 끼면 성적이 거짓말을 한다.**

    이 저장소가 이미 한 번 속은 자리다 — 같은 날 짝으로 배우고 같은 날로 재면
    개체가 아니라 **그날 조명**을 배운 것이 성적으로 나온다. 새는 길이 둘이라
    둘 다 시험한다.
    """

    def pairs(self, lab, day, fac, is_test):
        import numpy as np
        from finseg.management.commands.reid_probe import train_pairs
        return train_pairs(np.array(lab), np.array(day), np.array(fac),
                           np.array(is_test))

    def test_the_held_out_side_is_in_no_pair(self):
        # 0·1 은 배우는 쪽(개체 7), 2·3 은 재는 쪽(개체 9)
        got = self.pairs([7, 7, 9, 9], ["d1", "d2", "d1", "d2"],
                         ["left"] * 4, [False, False, True, True])
        self.assertEqual(got, [(0, 1)])

    def test_a_same_day_pair_is_not_learned(self):
        """같은 날 짝은 쉬운 짝이라 배울 것이 없고, **조명에 상을 준다.**"""
        got = self.pairs([7, 7], ["d1", "d1"], ["left", "left"], [False, False])
        self.assertEqual(got, [])

    def test_left_and_right_are_never_paired(self):
        """`reid.normalize` 가 한쪽을 거울처럼 뒤집는다 — 섞으면 서로 다른 두
        면이 같아 보인다 (`fliplr` 을 금지하는 것과 같은 이유)."""
        got = self.pairs([7, 7], ["d1", "d2"], ["left", "right"], [False, False])
        self.assertEqual(got, [])

    def test_unlabelled_chips_are_not_pairs(self):
        got = self.pairs([-1, -1], ["d1", "d2"], ["left", "left"], [False, False])
        self.assertEqual(got, [])


class ReidCatalogAsOfTests(TestCase):
    """**옛 정답에 새 자를 댈 수 있어야 한다.** 안 그러면 자가 좋아진 것인지
    정답이 늘어서인지 못 가른다 — 표가 쌓이는 것이 그러라고 있는 것이다.
    """

    def setUp(self):
        img = Image.objects.create(path="t/e.jpg", obsdate="2020-01-01",
                                   width=1000, height=800)
        self.a, self.b = [
            Box.objects.create(image=img, x1=10 * i, y1=10, x2=60 * (i + 1),
                               y2=60, source="yolov5") for i in range(2)]
        self.ind = Individual.objects.create(name="A")

    def test_it_gives_back_the_catalog_of_that_moment(self):
        from finseg import reid
        from finseg.models import Identification
        first = Identification.objects.create(box=self.a, individual=self.ind)
        Identification.objects.create(box=self.b, individual=self.ind)
        self.assertEqual(sorted(reid.catalog()[self.ind.id]),
                         sorted([self.a.id, self.b.id]))
        # 첫 판정까지만 — 뒤엣것은 아직 없던 일이다
        self.assertEqual(reid.catalog(as_of=first.id)[self.ind.id],
                         [self.a.id])

    def test_taking_a_fin_out_is_also_an_answer_at_that_moment(self):
        """뺀 것(`individual=NULL`)도 그때의 답이다 — 되살릴 때 함께 산다."""
        from finseg import reid
        from finseg.models import Identification
        Identification.objects.create(box=self.a, individual=self.ind)
        out = Identification.objects.create(box=self.a, individual=None)
        self.assertNotIn(self.ind.id, reid.catalog())
        self.assertEqual(reid.catalog(as_of=out.id - 1)[self.ind.id],
                         [self.a.id])


class MaskClassTests(TestCase):
    """**엔진이 무엇이라 했는지도 담는다.**

    분할 모델은 `coarse` 세 갈래를 내는데 그동안 폴리곤만 담고 분류를 버렸다.
    버리면 안 되는 이유는 **자동 경로에서 그것이 유일한 분류**이기 때문이다 —
    검출기는 클래스가 하나라 주둥이·몸통·사람을 다 "지느러미 같은 것" 으로
    잡아 온다.

    다만 **사람의 판정을 대신하지 않는다.** `rules.resolve` 는 여전히 `Review`
    를 본다.
    """

    def setUp(self):
        img = Image.objects.create(path="t/mc.jpg", obsdate="2020-01-01",
                                   width=1000, height=800)
        self.box = Box.objects.create(image=img, x1=10, y1=10, x2=60, y2=60,
                                      source="yolov5")
        self.run = Run.objects.create(kind="yolo")

    def test_the_engine_class_is_kept(self):
        m = Mask.objects.create(box=self.box, run=self.run, is_current=True,
                                polygon="20,20 50,20 35,50", cls="dolphin")
        self.assertEqual(Mask.objects.get(id=m.id).cls, "dolphin")

    def test_it_does_not_override_the_person(self):
        """사람이 `fin` 이라 했으면 엔진이 `nonfin` 이라 해도 `fin` 이다."""
        Mask.objects.create(box=self.box, run=self.run, is_current=True,
                            polygon="20,20 50,20 35,50", cls="nonfin")
        Review.objects.create(box=self.box, cls="fin", verdict="ok")
        st = rules.resolve(self.box)
        self.assertEqual(st["cls"], "fin")
        self.assertEqual(rules.label_of(st["review"]), rules.POSITIVE)

    def test_an_old_mask_without_a_class_still_works(self):
        """옛 마스크에는 이 칸이 없다 — 빈 값이어도 아무것도 안 깨져야 한다."""
        Mask.objects.create(box=self.box, run=self.run, is_current=True,
                            polygon="20,20 50,20 35,50")
        st = rules.resolve(self.box)
        self.assertEqual(st["mask"].cls, "")
        self.assertEqual(rules.label_of(st["review"]), rules.PENDING)


BACKUP_SQLITE = "finseg.management.commands.backup.sqlite3"


class _DiesMidDump:
    """`backup()` 이 **받는 쪽에 쓰기 시작한 뒤** 죽는 흉내.

    실제로도 그렇게 죽는다 — 페이지를 차례로 옮기는 일이라 중간에 끊기면
    받는 파일이 **반쯤 쓰인 채** 남는다. 원본이 통째로 깨진 경우와는 다르다:
    그때는 첫 쪽을 읽다가 죽어서 받는 쪽을 아예 안 건드리고, 그래서 그것으로는
    이 버그가 안 드러난다.
    """

    def __init__(self, real):
        self._r = real

    def __getattr__(self, name):
        return getattr(self._r, name)

    def connect(self, target, *a, **kw):
        return _Half(self._r.connect(target, *a, **kw), target, self._r)


class _Half:
    def __init__(self, con, path, real):
        self._c, self._p, self._r = con, path, real

    def __getattr__(self, name):
        return getattr(self._c, name)

    def backup(self, tgt, *a, **kw):
        Path(str(tgt._p)).write_bytes(b"half-written pages")
        raise self._r.OperationalError("disk I/O error")


class BackupTests(TestCase):
    """**확인 안 한 백업은 백업이 아니다.**

    형제 프로젝트가 프레임 229장을 잃은 것이 그 자리였다. `fin.db` 는 사람의
    판정이라 다시 만들 수 없고, 그것을 잃으면 오늘까지 한 검토가 통째로
    사라진다.

    시험은 메모리 DB 로 도므로 `--db` 로 진짜 파일을 대 준다 — 여기서 재는
    것은 **뜨고 · 읽어 보고 · 곁파일을 안 남기고 · 오래된 것을 지우는** 절차다.
    """

    def setUp(self):
        import sqlite3
        import tempfile
        self.tmp = Path(tempfile.mkdtemp())
        self.src = self.tmp / "fin.db"
        c = sqlite3.connect(self.src)
        c.execute("create table finseg_individual (id integer primary key,"
                  " name text)")
        c.executemany("insert into finseg_individual (name) values (?)",
                      [("JJ01",), ("JJ02",)])
        c.commit()
        c.execute("PRAGMA journal_mode=WAL")     # 우리 DB 와 같은 모드로
        c.close()

    def run_it(self, out, **kw):
        from django.core.management import call_command
        buf = io.StringIO()
        call_command("backup", out=str(out), db=str(self.src),
                     no_weights=True, stdout=buf, **kw)
        return buf.getvalue()

    def test_it_writes_a_readable_copy(self):
        import sqlite3
        out = self.tmp / "nas"
        out.mkdir()
        self.run_it(out)
        got = list((out / "db").glob("fin.db.*.bak"))
        self.assertEqual(len(got), 1)
        # **읽어 본다** — 뜬 줄 알고 지나가는 것이 가장 나쁘다
        c = sqlite3.connect(f"file:{got[0]}?mode=ro", uri=True)
        self.assertEqual(c.execute("PRAGMA integrity_check").fetchone()[0], "ok")
        self.assertEqual(
            c.execute("select count(*) from finseg_individual").fetchone()[0], 2)
        c.close()

    def test_no_wal_sidecars_are_left(self):
        """백업 옆에 `-wal` 이 놓여 있으면 **복원할 때 무엇이 진짜인지
        헷갈린다** — 하나만 옮기면 조용히 옛 상태가 된다."""
        out = self.tmp / "nas"
        out.mkdir()
        self.run_it(out)
        left = [p.name for p in (out / "db").iterdir()]
        self.assertEqual([p for p in left if p.endswith(("-wal", "-shm"))], [])

    def test_it_refuses_when_the_target_is_not_there(self):
        """**없는 데다 뜨면 뜬 줄 알고 지나간다** — NAS 가 안 붙었을 때다."""
        from django.core.management.base import CommandError
        with self.assertRaises(CommandError) as e:
            self.run_it("/그런/데는/없다/backup")
        self.assertIn("NAS", str(e.exception))

    def test_old_copies_are_dropped_but_the_newest_stays(self):
        # 이름에 기계가 들어가므로 fixture 도 같은 갈래에 둔다 — 기계 이름
        # 없이 뜬 것은 이제 **일부러 안 지운다**(어느 기계 것인지 모른다).
        # 그쪽은 `test_a_backup_from_before_the_machine_name_is_left_alone`
        out = self.tmp / "nas"
        d = out / "db"
        d.mkdir(parents=True)
        for day in ("2020-01-01", "2020-01-02", "2020-01-03"):
            (d / f"fin.db.{day}.2080ti.bak").write_bytes(b"x")
        self.run_it(out, keep=2, host="2080ti")
        left = sorted(p.name for p in d.glob("fin.db.*.2080ti.bak"))
        self.assertEqual(len(left), 2)
        self.assertIn(date.today().isoformat(), left[-1])

    # ---- 같은 날 두 번 뜰 때 -------------------------------------------
    #
    # 이름이 날짜라 **같은 날 두 번 뜨면 같은 파일이다.** 본 자리에 바로 쓰면
    # 뜨는 동안 옛 백업이 없는 시간이 생긴다. 여기서 재는 것은 그 창이 없다는
    # 것 — **백업이 가장 위험한 순간은 백업을 뜨는 순간이다.**

    def test_a_failed_dump_leaves_the_previous_backup_alone(self):
        """뜨다가 엎어져도 **어제까지의 백업은 그대로 있어야 한다.**

        고치기 전에는 본 자리에 바로 썼다 — 깨진 것을 잡기는 했지만
        **덮어쓴 다음에** 잡아서, 그때는 새것도 못 쓰고 옛것도 없었다.
        """
        import sqlite3
        from unittest.mock import patch
        out = self.tmp / "nas"
        d = out / "db"
        d.mkdir(parents=True)
        # **어제 것은 멀쩡한 DB 다.** 쓰레기 파일을 놔두면 옛 코드도 안
        # 건드려서 — `connect` 는 파일을 자르지 않는다 — 시험이 양쪽에서
        # 통과한다. 그러면 아무것도 재지 않는 시험이다
        keep = d / f"fin.db.{date.today().isoformat()}.bak"
        c = sqlite3.connect(keep)
        c.execute("create table 어제 (id integer primary key)")
        c.commit()
        c.close()
        OLD = keep.read_bytes()

        with patch(BACKUP_SQLITE, _DiesMidDump(sqlite3)):
            with self.assertRaises(sqlite3.OperationalError):
                self.run_it(out)

        # **쓰다가 죽어도 어제 것은 그대로다.** 옛 코드는 받는 자리가 곧
        # 어제 것이라 여기서 깨졌다
        self.assertEqual(keep.read_bytes(), OLD)
        # 반쯤 뜬 것을 남기지 않는다 — 남으면 다음 사람이 백업으로 본다
        self.assertEqual([p.name for p in d.iterdir() if ".part" in p.name], [])

    def test_a_stale_part_is_not_mistaken_for_a_backup(self):
        """지난번에 엎어져 남은 `.part` 가 있어도 이번 것이 제대로 선다."""
        import sqlite3
        out = self.tmp / "nas"
        d = out / "db"
        d.mkdir(parents=True)
        stale = d / f"fin.db.{date.today().isoformat()}.bak.part"
        stale.write_bytes(b"half-written junk")

        self.run_it(out)

        self.assertFalse(stale.exists())
        got = list(d.glob("fin.db.*.bak"))
        self.assertEqual(len(got), 1)
        c = sqlite3.connect(f"file:{got[0]}?mode=ro", uri=True)
        self.assertEqual(c.execute("PRAGMA integrity_check").fetchone()[0], "ok")
        c.close()

    def test_a_part_left_from_another_day_is_swept_too(self):
        """다른 날에 엎어져 남은 `.part` 도 치운다.

        그 이름으로는 **다시 뜰 일이 없어** 저절로 갈릴 기회가 없고,
        오래된 것 지우기는 `.bak` 로 끝나는 것만 보느라 지나친다 — 아무도
        안 치우면 NAS 에 영영 남는다.
        """
        out = self.tmp / "nas"
        d = out / "db"
        d.mkdir(parents=True)
        stale = d / "fin.db.2020-01-01.bak.part"
        stale.write_bytes(b"half-written junk")
        (d / "fin.db.2020-01-01.bak.part-wal").write_bytes(b"junk")

        self.run_it(out)

        self.assertEqual([p.name for p in d.iterdir() if ".part" in p.name], [])
        # 멀쩡한 백업은 건드리지 않는다 — 쓸어 낸 것은 찌꺼기뿐이다
        self.assertEqual(len(list(d.glob("fin.db.*.bak"))), 1)

    # ---- 기계가 둘일 때 -------------------------------------------------
    #
    # 날짜만으로는 **기계 사이에서도 부딪친다.** 2026-08-24 에 실제로 그랬다 —
    # m710q 가 저녁에 뜬 것이 2080ti 가 낮에 올려 둔 것을 갈아 치웠다. 그날은
    # 나중 것이 상위집합이라 잃은 게 없었지만 **그것은 운이었다.**

    def test_two_machines_do_not_overwrite_each_other(self):
        """다른 기계가 같은 날 떠도 **서로의 것을 안 건드린다.**"""
        import sqlite3
        out = self.tmp / "nas"
        out.mkdir()
        self.run_it(out, host="m710q")
        theirs = out / "db" / f"fin.db.{date.today().isoformat()}.m710q.bak"
        self.assertTrue(theirs.exists())
        THEIRS = theirs.read_bytes()

        # 두 번째 기계가 판정을 더 얹고 같은 날 뜬다
        c = sqlite3.connect(self.src)
        c.execute("insert into finseg_individual (name) values ('JJ03')")
        c.commit()
        c.close()
        self.run_it(out, host="2080ti")

        ours = out / "db" / f"fin.db.{date.today().isoformat()}.2080ti.bak"
        self.assertTrue(ours.exists())
        self.assertEqual(theirs.read_bytes(), THEIRS)   # 남의 것은 그대로다
        c = sqlite3.connect(f"file:{ours}?mode=ro", uri=True)
        self.assertEqual(
            c.execute("select count(*) from finseg_individual").fetchone()[0], 3)
        c.close()

    def test_pruning_only_touches_its_own_lane(self):
        """`--keep` 이 **남의 갈래를 줄이지 않는다.**

        그 기계는 제가 몇 벌 갖고 있는지 모르는 채 남의 손에 줄어든다.
        """
        out = self.tmp / "nas"
        d = out / "db"
        d.mkdir(parents=True)
        for day in ("2020-01-01", "2020-01-02", "2020-01-03"):
            (d / f"fin.db.{day}.m710q.bak").write_bytes(b"x")
            (d / f"fin.db.{day}.2080ti.bak").write_bytes(b"x")

        self.run_it(out, keep=2, host="2080ti")

        self.assertEqual(len(list(d.glob("fin.db.*.m710q.bak"))), 3)   # 안 건드렸다
        self.assertEqual(len(list(d.glob("fin.db.*.2080ti.bak"))), 2)  # 제 것만 줄였다

    def test_a_backup_from_before_the_machine_name_is_left_alone(self):
        """기계 이름 없이 뜬 옛것은 **어느 기계 것인지 알 수 없어 안 지운다.**"""
        out = self.tmp / "nas"
        d = out / "db"
        d.mkdir(parents=True)
        old = d / "fin.db.2020-01-01.bak"
        old.write_bytes(b"x")

        got = self.run_it(out, keep=1, host="2080ti")

        self.assertTrue(old.exists())
        self.assertIn("기계 이름 없이", got)

    def test_the_manifest_is_split_by_machine_too(self):
        """MANIFEST 도 날짜뿐이면 같이 부딪친다."""
        import json
        out = self.tmp / "nas"
        out.mkdir()
        self.run_it(out, host="m710q")
        self.run_it(out, host="2080ti")
        got = sorted(p.name for p in out.glob("MANIFEST.*.json"))
        self.assertEqual(len(got), 2)
        d = json.loads((out / got[0]).read_text())
        self.assertIn("host", d)

    def test_dumping_twice_in_a_day_carries_the_newer_rows(self):
        """갈아 끼우기가 **정말 갈아 끼우는지** 본다 — 옛것을 지키느라
        새것을 안 쓰면 그것대로 조용히 옛 상태가 된다."""
        import sqlite3
        out = self.tmp / "nas"
        out.mkdir()
        self.run_it(out)

        c = sqlite3.connect(self.src)
        c.execute("insert into finseg_individual (name) values ('JJ03')")
        c.commit()
        c.close()
        self.run_it(out)

        got = list((out / "db").glob("fin.db.*.bak"))
        self.assertEqual(len(got), 1)          # 날짜가 같으니 한 벌이다
        c = sqlite3.connect(f"file:{got[0]}?mode=ro", uri=True)
        self.assertEqual(
            c.execute("select count(*) from finseg_individual").fetchone()[0], 3)
        c.close()


class RoleTests(SimpleTestCase):
    """**어느 길이 걸리는지는 `FIN_ROLE` 이 정한다.**

    화면을 정리하려는 것이 아니다. 개체를 만들고 지느러미를 넣는 일이 두
    자리에서 일어나면 `Individual`·`Identification` 의 번호가 겹치고, 그러면
    합치는 길이 없다 (`HANDOFF.md` 의 `## 서버를 둘로 나눈다`).

    **미들웨어가 아니라 URLconf 에서 뺀다** — 경로가 없으면 실수로 도는 길이
    없다. 그래서 시험도 "403 이 나오나" 가 아니라 **"길이 아예 없나"** 를 묻는다.
    """
    # 이 넷이 `Identification`·`Individual` 에 쓰는 전부다. 하나라도 `work`
    # 쪽에 새면 그 자리가 조용히 두 번째 주인이 된다.
    #
    # **`reid_cls_set` 이 여기 있었다가 사라졌다** — `/reid` 에서 "지느러미가
    # 아니다" 를 쓰던 자리인데, 그것은 `Individual` 이 아니라 `Review` 에
    # 쓰는 길이었다. 걷어냈으므로 이제 `Review` 의 주인도 이 기계 하나다.
    WRITES_INDIVIDUALS = {"reid", "reid_box", "reid_assign", "catalog"}

    def names(self, role):
        from review.urls import patterns_for
        return {p.name for p in patterns_for(role)}

    def test_reid_role_has_only_the_reid_screens(self):
        got = self.names("reid")
        self.assertTrue(self.WRITES_INDIVIDUALS <= got)
        for gone in ("index", "edit", "photo", "compare", "detect", "save", "batch"):
            self.assertNotIn(gone, got)

    def test_work_role_cannot_reach_anything_that_writes_individuals(self):
        """**이 시험이 이 갈래의 이유다.** `/reid` 는 열기만 해도 보류함
        `Individual` 을 하나 만든다 (`views.reid` 의 `get_or_create`) — 구경도
        쓰기라, 링크를 숨기는 것으로는 안 막힌다."""
        got = self.names("work")
        self.assertEqual(self.WRITES_INDIVIDUALS & got, set())
        self.assertIn("index", got)
        self.assertIn("edit", got)

    def test_healthz_is_in_both(self):
        """**배포가 뒤바뀐 것을 밖에서 잡는 자리**라 역할을 안 탄다."""
        for role in ("work", "reid"):
            self.assertIn("healthz", self.names(role))

    def test_an_unknown_role_stops_the_process(self):
        """조용히 `work` 로 떨어지면 **개체 판정을 받을 자리가 안 받는다.**"""
        with self.assertRaises(ValueError):
            import os
            from importlib import reload
            old = os.environ.get("FIN_ROLE")
            os.environ["FIN_ROLE"] = "reid-collector"      # 오타를 흉내낸다
            try:
                import finweb.settings
                reload(finweb.settings)
            finally:
                if old is None:
                    os.environ.pop("FIN_ROLE", None)
                else:
                    os.environ["FIN_ROLE"] = old
                import finweb.settings
                reload(finweb.settings)


class HealthzTests(TestCase):
    """`/healthz` 는 **살아 있나** 만 묻는 자리가 아니다 — 무엇으로 떴는지와
    안전망이 막혔는지를 함께 말한다 (`.guides/web/data-safety.md` §2)."""

    def test_it_says_which_role_it_came_up_as(self):
        """**배포가 뒤바뀐 것을 여기서 잡는다.** 개체 분류를 받을 자리가
        `work` 로 떠 있으면 화면은 멀쩡히 200 을 내면서 그날 판정을 한 건도
        못 받는다."""
        d = self.client.get("/healthz").json()
        self.assertEqual(d["role"], "work")
        self.assertEqual(d["status"], "ok")
        self.assertIn("version", d)

    def test_a_blocked_backup_shows_as_degraded_but_still_200(self):
        """503 은 "트래픽 보내지 말라" 는 뜻이라 배포 스크립트의 liveness
        대기를 멈춰 세운다 — **게이트가 아니라 배포 장애가 된다.**"""
        import tempfile
        with tempfile.TemporaryDirectory() as t:
            f = Path(t) / "INTEGRITY_FAIL"
            f.write_text("2026-08-26T00:00:00 integrity_check failed\n")
            with self.settings(FIN_SENTINEL=f):
                r = self.client.get("/healthz")
            self.assertEqual(r.status_code, 200)
            d = r.json()
            self.assertEqual(d["status"], "degraded")
            self.assertIn("integrity_check failed", d["integrity"])


@override_settings(FIN_ACCESS_CODE="열려라-참깨-0123456789")
class AccessCodeTests(TestCase):
    """문. **기본이 막힘이고 예외만 적는다** (`review/gate.py`).

    이 앱에는 `login_required` 가 하나도 없고 쓰기 경로가 열려 있다 — 그것이
    쓰는 것은 다시 만들 수 없는 개체 판정이다.
    """

    def setUp(self):
        from django.core.cache import cache
        cache.clear()               # 잠금 셈이 시험끼리 새면 안 된다

    def test_a_stranger_is_sent_to_the_door(self):
        r = self.client.get("/review")
        self.assertEqual(r.status_code, 302)
        self.assertTrue(r["Location"].startswith("/enter?next="))

    def test_the_door_remembers_where_you_were_going(self):
        """링크를 받고 들어온 사람이 문을 지난 뒤 처음 화면으로 떨어지면
        그 링크가 무엇이었는지 잃는다."""
        r = self.client.get("/edit/7")
        self.assertIn("next=%2Fedit%2F7", r["Location"])
        r = self.client.post("/enter", {"code": "열려라-참깨-0123456789",
                                        "next": "/edit/7"})
        self.assertEqual(r.status_code, 302)
        self.assertEqual(r["Location"], "/edit/7")

    def test_it_will_not_bounce_you_to_another_site(self):
        """`next=//남의집` 을 그대로 받으면 이 화면이 남의 자리로 보내는
        발판이 된다."""
        r = self.client.post("/enter", {"code": "열려라-참깨-0123456789",
                                        "next": "//example.com/x"})
        self.assertEqual(r["Location"], "/")

    def test_healthz_stays_outside_the_door(self):
        """smoke 와 배포가 그것을 읽는다 — 코드를 모르는 자리에서도 살아
        있는지는 물을 수 있어야 한다."""
        self.assertEqual(self.client.get("/healthz").status_code, 200)

    def test_a_wrong_code_does_not_open_anything(self):
        self.client.post("/enter", {"code": "틀린것"})
        self.assertEqual(self.client.get("/review").status_code, 302)

    def test_it_locks_after_too_many_tries(self):
        """짧은 코드를 쓰면 이것만으로 못 막는다 — 그래도 **긁어 보는 것**은
        여기서 멎는다."""
        from review import gate
        for _ in range(gate.MAX_TRIES):
            self.client.post("/enter", {"code": "틀린것"})
        r = self.client.post("/enter", {"code": "열려라-참깨-0123456789"})
        self.assertEqual(r.status_code, 200)          # 맞아도 안 열린다
        self.assertIn("분 뒤에", r.content.decode())

    def test_the_session_key_is_replaced_when_you_pass(self):
        """문 앞에서 받은 키를 그대로 쓰면 **남이 미리 심어 둔 키로 안까지
        들어오게 된다** (session fixation) — 여기서는 그 키를 흉내 낸다."""
        s = self.client.session
        s["심어둔것"] = 1
        s.save()
        planted = s.session_key
        self.client.cookies["sessionid"] = planted
        self.client.post("/enter", {"code": "열려라-참깨-0123456789"})
        self.assertNotEqual(self.client.cookies["sessionid"].value, planted)

    def test_no_code_no_door(self):
        """m710q 의 개발·시험 자리는 그대로 둔다."""
        with self.settings(FIN_ACCESS_CODE=""):
            self.assertEqual(self.client.get("/review").status_code, 200)


@override_settings(FIN_ACCESS_CODE="열려라-참깨-0123456789")
class HealthzOutsideTheDoorTests(TestCase):
    def test_it_answers_but_does_not_count_out_loud(self):
        """`/healthz` 는 smoke 가 읽어야 해서 문 밖에 있다. 그 김에 상자·개체가
        몇인지까지 알려 주면 **문을 세운 뜻이 절반 없어진다.**"""
        d = self.client.get("/healthz").json()
        self.assertEqual(d["status"], "ok")
        self.assertIn("role", d)
        self.assertIn("reid_items", d)          # 자료가 왔는지는 밖에서도 봐야 한다
        for hidden in ("box", "individual", "identification", "db"):
            self.assertNotIn(hidden, d)

    def test_inside_it_tells_everything(self):
        self.client.post("/enter", {"code": "열려라-참깨-0123456789"})
        d = self.client.get("/healthz").json()
        self.assertIn("box", d)
        self.assertIn("db", d)


class ReidCandidateQueueTests(TestCase):
    """**격자로 나가기 전에 한 번 거르는 자리.**

    `/reid` 가 판정을 안 쓰게 되면서(`Review` 의 주인을 한 곳으로 두려고)
    몸통·바위를 걸러 낼 자리가 없어졌다. 검토 화면의 대기열이 그것을 받는다.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        img = Image.objects.create(path="nas/2016/03/15/a.JPG", obsdate=date(2016, 3, 15),
                                   width=100, height=100)
        self.boxes = [Box.objects.create(image=img, x1=0, y1=0, x2=60, y2=60,
                                         source="yolov5", conf=0.9) for _ in range(3)]
        for b in self.boxes:
            Crop.objects.create(box=b, path=f"000/{b.id:08d}.jpg",
                                x0=0, y0=0, x1=60, y1=60, w=640, h=640)
        items = {"n": 3, "items": [
            {"id": self.boxes[0].id, "day": "2016-03-15", "facing": "L", "pfin": 0.02},
            {"id": self.boxes[1].id, "day": "2016-03-15", "facing": "L", "pfin": 0.98},
            {"id": self.boxes[2].id, "day": "2016-03-15", "facing": "L"},
        ]}
        (self.tmp / "items.json").write_text(json.dumps(items), encoding="utf-8")

    def ids(self):
        with self.settings(FIN_REID=self.tmp):
            r = self.client.get("/api/batch?mode=reid&n=24").json()
        return [t["box_id"] for t in r["tiles"]], r["total"]

    def test_the_most_suspicious_comes_first(self):
        """`fin_filter` 가 적어 둔 `p(fin)` 이 낮은 것부터 — **의심스러운 것을
        먼저 치워야 격자가 빨리 깨끗해진다.**"""
        got, total = self.ids()
        self.assertEqual(total, 3)
        self.assertEqual(got[0], self.boxes[0].id)      # p(fin) 0.02
        self.assertEqual(got[1], self.boxes[1].id)      # 0.98
        self.assertEqual(got[2], self.boxes[2].id)      # 값이 없으면 뒤로

    def test_what_a_person_already_judged_drops_out(self):
        """한 번 본 것을 또 보여 주면 **볼 때마다 같은 판단을 다시 한다.**"""
        Review.objects.create(box_id=self.boxes[0].id, cls="body")
        got, total = self.ids()
        self.assertEqual(total, 2)
        self.assertNotIn(self.boxes[0].id, got)

    def test_no_grid_is_not_an_error(self):
        """조각을 아직 안 만든 자리가 있다 — 빈 대기열이지 오류가 아니다."""
        with self.settings(FIN_REID=self.tmp / "없는곳"):
            r = self.client.get("/api/batch?mode=reid").json()
        self.assertEqual(r["total"], 0)


class HomeTests(TestCase):
    """홈은 **한 바퀴 전체가 어디까지 왔나**를 말하는 자리다."""

    def test_it_does_not_offer_a_road_this_seat_does_not_have(self):
        """`work` 자리에서 `/reid` 는 404 다. 그것을 내밀면 누른 사람은 화면이
        깨진 줄 알지, **이 자리가 그 일을 안 하기로 한 줄 모른다.**"""
        html = self.client.get("/").content.decode()
        self.assertNotIn('href="/reid"', html)
        self.assertNotIn('href="/catalog"', html)
        self.assertIn("reid` 자리에서 한다", html)      # 어디서 하는지는 말한다
        self.assertIn("개체 (re-ID)", html)             # 셈은 그대로 보인다

    @override_settings(FIN_ROLE="reid", ROOT_URLCONF="review.tests")
    def test_the_reid_seat_gets_the_road(self):
        # `reid` 자리에는 홈이 `/reid` 로 걸려 있다 (`urls.patterns_for`).
        from review.urls import patterns_for
        self.assertIn("reid", {p.name for p in patterns_for("reid")})

    def test_filtering_comes_before_classifying(self):
        """거른 결과가 다음 격자에 실리고, 그 격자로 분류한다 — **순서가 곧
        자료가 흐르는 방향이다.**"""
        html = self.client.get("/").content.decode()
        self.assertLess(html.index("re-ID 후보 거르기"), html.index("개체 (re-ID)"))


class ReidOrderTests(TestCase):
    """찍힌 차례로 볼 수 있어야 한다 — **같은 무리를 잇달아 찍은 것이 붙어 있다.**"""

    def setUp(self):
        from django.utils import timezone
        self.tmp = Path(tempfile.mkdtemp())
        self.ids = []
        # 같은 관찰일인데 UTC 로는 전날인 이른 아침 — 여기서 어긋난다
        for h, m in ((5, 48), (5, 49), (18, 2)):
            img = Image.objects.create(
                path=f"nas/2016/03/15/{h}{m}.JPG", obsdate=date(2016, 3, 15),
                width=100, height=100,
                exifdatetime=timezone.make_aware(
                    __import__("datetime").datetime(2016, 3, 15, h, m)))
            b = Box.objects.create(image=img, x1=0, y1=0, x2=60, y2=60,
                                   source="yolov5", conf=0.9)
            self.ids.append(b.id)
        (self.tmp / "items.json").write_text(json.dumps({"items": [
            {"id": i, "day": "2016-03-15", "facing": "L"} for i in reversed(self.ids)]}),
            encoding="utf-8")
        (self.tmp / "look").mkdir()

    def items(self):
        with self.settings(FIN_REID=self.tmp, FIN_ROLE="reid",
                           ROOT_URLCONF="review.tests"):
            html = self.client.get("/reid").content.decode()
        return json.loads(re.search(r"const ITEMS = (\[.*?\]);", html, re.S).group(1))

    def test_every_chip_gets_a_capture_time(self):
        got = self.items()
        self.assertEqual(len(got), 3)
        self.assertTrue(all(i["at"] for i in got))

    def test_the_time_is_local_so_it_matches_the_day_it_shows(self):
        """UTC 그대로 쓰면 이른 아침 것이 **전날로 보인다** — 화면이 말하는 날과
        정렬 축의 날이 어긋나면 사람이 그것을 오류로 읽는다."""
        for i in self.items():
            self.assertEqual(i["at"][:10], i["day"])

    def test_a_chip_without_exif_falls_back_to_the_day(self):
        """없는 것이 맨 앞으로 몰리면 그 순서에 뜻이 없어진다."""
        Image.objects.update(exifdatetime=None)
        self.assertTrue(all(i["at"] == "2016-03-15" for i in self.items()))


class SpecialBoxTests(TestCase):
    """**개체가 아닌 자리 둘** (`Individual.KINDS`).

    `지느러미 아님` 이 있는 이유는 re-ID 자리가 `Review` 를 못 쓰기 때문이다 —
    주인이 작업 자리 하나다. 거기서는 **제 레인으로 말해 두고**, 되받은 뒤
    여기서 사람이 진짜 분류를 골라 옮겨 적는다.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "items.json").write_text('{"items": []}', encoding="utf-8")
        (self.tmp / "look").mkdir()
        img = Image.objects.create(path="a.JPG", obsdate=date(2016, 3, 15),
                                   width=100, height=100)
        self.box = Box.objects.create(image=img, x1=0, y1=0, x2=60, y2=60,
                                      source="yolov5", conf=0.9)
        Crop.objects.create(box=self.box, path=f"000/{self.box.id:08d}.jpg",
                            x0=0, y0=0, x1=60, y1=60, w=640, h=640)
        # **마이그레이션이 이미 만들어 둔다** — 자리는 자료가 아니라 뼈대라,
        # 화면을 한 번도 안 연 기계에서도 있어야 한다
        self.nf = Individual.objects.get(kind="notfin")

    def test_opening_the_screen_makes_both_boxes(self):
        with self.settings(FIN_REID=self.tmp, FIN_ROLE="reid",
                           ROOT_URLCONF="review.tests"):
            self.client.get("/reid")
        self.assertEqual(
            sorted(Individual.objects.exclude(kind="").values_list("kind", "name")),
            [("hold", "임시보관함"), ("notfin", "지느러미 아님")])

    def test_only_one_box_per_kind(self):
        """둘이 되면 어느 쪽에 넣었는지에 따라 결과가 갈리는데, **그 사실은
        눈에 안 띈다.**"""
        from django.db import IntegrityError, transaction
        # 제약을 건드리면 그 트랜잭션이 깨진다 — 안쪽에 하나 더 두어 감싼다
        with self.assertRaises(IntegrityError), transaction.atomic():
            Individual.objects.create(kind="notfin", name="또 하나")

    def test_a_special_box_is_not_an_individual(self):
        """세면 성적이 그만큼 부푼다."""
        from finseg import reid
        Identification.objects.create(box=self.box, individual=self.nf)
        self.assertEqual(reid.catalog(), {})

    def test_what_the_other_seat_marked_comes_to_the_queue(self):
        Identification.objects.create(box=self.box, individual=self.nf)
        r = self.client.get("/api/batch?mode=notfin").json()
        self.assertEqual(r["total"], 1)
        self.assertEqual(r["tiles"][0]["box_id"], self.box.id)

    def test_once_a_person_writes_it_down_it_leaves_the_queue(self):
        """옮겨 적기가 끝난 것을 또 보여 주면 **볼 때마다 같은 판단을 다시 한다.**"""
        Identification.objects.create(box=self.box, individual=self.nf)
        Review.objects.create(box=self.box, cls="body")
        self.assertEqual(self.client.get("/api/batch?mode=notfin").json()["total"], 0)

    def test_calling_it_a_fin_keeps_it_in_the_queue(self):
        """저쪽이 아니라 했는데 여기서 `fin` 이라 적었으면 **아직 안 옮긴 것**이다."""
        Identification.objects.create(box=self.box, individual=self.nf)
        Review.objects.create(box=self.box, cls="fin")
        self.assertEqual(self.client.get("/api/batch?mode=notfin").json()["total"], 1)


class FlipMarkTests(TestCase):
    """**뒤집힌 것만 표시한다.**

    조각은 앞쪽이 늘 왼쪽에 오게 세운 것이라, 원래 오른쪽이 앞이던 사진은
    좌우로 뒤집혀 있다. 원본과 나란히 놓고 볼 때 그것을 모르면 **다른
    지느러미로 읽는다.**
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "look").mkdir()
        ids = []
        for f in ("left", "right"):
            img = Image.objects.create(path=f"{f}.JPG", obsdate=date(2016, 3, 15),
                                       width=100, height=100)
            b = Box.objects.create(image=img, x1=0, y1=0, x2=60, y2=60,
                                   source="yolov5", conf=0.9)
            ids.append((b.id, f))
        (self.tmp / "items.json").write_text(json.dumps({"items": [
            {"id": i, "day": "2016-03-15", "facing": f, "rough": 0.1}
            for i, f in ids]}), encoding="utf-8")
        self.ids = dict((f, i) for i, f in ids)

    def page(self):
        with self.settings(FIN_REID=self.tmp, FIN_ROLE="reid",
                           ROOT_URLCONF="review.tests"):
            return self.client.get("/reid").content.decode()

    def test_the_page_carries_which_way_each_chip_faces(self):
        got = {i["id"]: i["facing"]
               for i in json.loads(re.search(r"const ITEMS = (\[.*?\]);",
                                             self.page(), re.S).group(1))}
        self.assertEqual(got[self.ids["right"]], "right")
        self.assertEqual(got[self.ids["left"]], "left")

    def test_only_the_flipped_one_gets_a_mark(self):
        """둘 다 표를 달면 격자가 글자로 덮인다 — **절반에 다는 것으로 충분하다.**"""
        html = self.page()
        self.assertIn('i.facing === "right"', html)      # 표는 `right` 에만
        self.assertNotIn('"우"', html)                    # 좌·우를 다 적지 않는다

    def test_the_mark_does_not_hide_with_the_outline_toggle(self):
        """뒤집혔다는 것은 그림 자체의 성질이라 윤곽을 끄고 보아도 알아야 한다."""
        html = self.page()
        self.assertNotIn("body:not(.ovon) .fin .flip", html)


@override_settings(FIN_ROLE="reid", ROOT_URLCONF="review.tests")
class NicknameTests(TestCase):
    """**번호와 별명은 다른 칸이다.**

    `JTA001 (제돌이)` 처럼 한 칸에 섞으면 정렬도 중복 검사도 옛 카탈로그와
    맞춰 보는 일도 전부 그 괄호를 파싱하게 되고, 별명이 붙거나 바뀔 때마다
    **번호가 든 문자열을 건드리게 된다.**
    """

    def setUp(self):
        self.ind = Individual.objects.create(name="JTA001")

    def post(self, body):
        return self.client.post("/api/reid/box", json.dumps(body),
                                content_type="application/json").json()

    def test_a_nickname_can_be_set_and_cleared(self):
        self.assertEqual(self.post({"id": self.ind.id, "nick": "제돌이"})["nick"], "제돌이")
        self.assertEqual(self.post({"id": self.ind.id, "nick": ""})["nick"], "")

    def test_nicknames_may_repeat(self):
        """겹치는 별명은 사람이 알아보는 문제이지 **자료가 깨지는 문제가 아니다** —
        개체를 가리키는 것은 번호다."""
        other = Individual.objects.create(name="JTA002")
        self.post({"id": self.ind.id, "nick": "제돌이"})
        self.assertEqual(self.post({"id": other.id, "nick": "제돌이"})["nick"], "제돌이")

    def test_the_number_still_may_not_repeat(self):
        Individual.objects.create(name="JTA002")
        self.assertIn("error", self.post({"id": self.ind.id, "name": "JTA002"}))

    def test_setting_a_nickname_leaves_the_number_alone(self):
        """별명 갈래가 이름 갈래보다 뒤에 있으면 "이름이 비었다" 를 낸다."""
        self.post({"id": self.ind.id, "nick": "춘삼이"})
        self.ind.refresh_from_db()
        self.assertEqual(self.ind.name, "JTA001")
        self.assertEqual(self.ind.nickname, "춘삼이")


class ContextMenuTests(TestCase):
    """**여기서 못 하는 일은 차림표에 안 올린다.**

    `원본 사진`·`윤곽 고치기` 는 `work` 자리의 길이라 `reid` 자리에서는 404 다.
    내밀면 누른 사람은 화면이 깨진 줄 알지, **이 자리가 그 일을 안 하기로 한
    줄 모른다.**
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "items.json").write_text('{"items": []}', encoding="utf-8")
        (self.tmp / "look").mkdir()

    def page(self, role):
        with self.settings(FIN_REID=self.tmp, FIN_ROLE=role,
                           ROOT_URLCONF="review.tests" if role == "reid" else None):
            return self.client.get("/reid" if role == "reid" else "/").content.decode()

    def test_the_reid_seat_knows_its_role(self):
        self.assertIn('const ROLE = "reid"', self.page("reid"))

    def test_the_dead_entries_are_behind_the_role(self):
        """길이 있는지를 화면이 스스로 판단하게 둔다 — 두 자리가 같은 템플릿을
        쓰므로 지워 버리면 `work` 쪽에서 그 길이 없어진다."""
        html = self.page("reid")
        self.assertIn('if (ROLE !== "reid")', html)
        self.assertIn("if (!rows.length) return false", html)


class FoldSplitTests(SimpleTestCase):
    """**재는 날을 돌린다.** 고정 갈래는 가장 큰 8일만 재고 나머지 41일
    361조각은 성적에 한 번도 안 들어간다 — 질의 137이면 한 문제가 0.73%p 라
    앞으로 잴 것들(백본·조각 해상도·2층 MLP·ArcFace)이 전부 그 눈금 아래다.
    """

    def days(self, sizes):
        return [f"d{i}" for i in range(len(sizes))], \
               {f"d{i}": n for i, n in enumerate(sizes)}

    def test_every_day_is_measured_exactly_once(self):
        """**이것이 폴드의 전부다** — 모든 관찰일이 한 번씩 잼 쪽에 선다."""
        from finseg.management.commands.reid_cls import split_days
        days, cnt = self.days([33, 30, 26, 26, 25, 21, 21, 20, 5, 1, 1])
        fold = split_days(days, cnt, 5)
        seen = [d for f in fold for d in f]
        self.assertEqual(sorted(seen), sorted(days))
        self.assertEqual(len(seen), len(set(seen)))

    def test_the_big_days_are_spread_out(self):
        """무작위로 담으면 큰 날이 한 폴드에 몰려 그 폴드만 잼이 두꺼워진다.
        큰 날부터 번갈아 담으면 폴드 크기가 비슷해진다."""
        from finseg.management.commands.reid_cls import split_days
        days, cnt = self.days([33, 30, 26, 26, 25, 21, 21, 20, 19, 17])
        loads = [sum(cnt[d] for d in f) for f in split_days(days, cnt, 5)]
        self.assertLess(max(loads) - min(loads), 10)

    def test_the_split_does_not_move(self):
        """**씨앗을 안 탄다.** 두 자를 견줄 때 같은 문제를 풀어야 차이가 자
        때문인지 문제 때문인지 갈린다."""
        from finseg.management.commands.reid_cls import split_days
        days, cnt = self.days([9, 8, 7, 6, 5, 4, 3, 2, 1])
        self.assertEqual(split_days(days, cnt, 3), split_days(days, cnt, 3))


class TemplateCommentTests(SimpleTestCase):
    """**Django 에서 `{# … #}` 는 한 줄짜리만 주석이다.**

    템플릿 토크나이저의 `tag_re` 에 `re.DOTALL` 이 없어서 `.` 이 줄바꿈을 안
    문다 — 여러 줄로 쓰면 주석으로 안 잡히고 **글자 그대로 페이지에 찍힌다.**
    여러 줄은 `{% comment %}` 여야 한다.

    이 저장소가 이것을 **두 번** 밟았다(`devlog/20260827_001` 8절, 그리고
    `/catalog` 의 조각 주석 — 반복문 안이라 **563번** 찍혔고 페이지의 절반이
    그것이었다). 눈으로는 잘 안 걸린다 — 편집기에서는 멀쩡한 주석으로 보이고
    화면에서도 그냥 글자가 하나 더 있는 것처럼 보인다. **세어서 잡는다.**
    """

    def test_no_hash_comment_spans_lines(self):
        import re
        from pathlib import Path as P
        bad = []
        for p in sorted(P("review/templates").rglob("*.html")):
            t = p.read_text(encoding="utf-8")
            for m in re.finditer(r"\{#", t):
                end = t.find("\n", m.start())
                seg = t[m.start(): end if end != -1 else len(t)]
                if "#}" not in seg:
                    bad.append(f"{p}:{t[:m.start()].count(chr(10)) + 1}")
        self.assertEqual(bad, [], "여러 줄 주석은 `{% comment %}` 로 쓸 것: "
                                 + ", ".join(bad))


class NewBoxRaceTests(TestCase):
    """**둘이 동시에 `새 상자` 를 눌러도 한 상자를 나눠 쓰지 않는다.**

    전에는 이름을 `count()+1` 로 짓고 `get_or_create` 했다. 같은 `n` 이 나오면
    뒤엣사람이 `made=False` 로 앞엣사람의 상자를 받아 든 채 **새로 만든 줄
    안다** — 그 뒤로 둘의 조각이 한 상자에 섞이고, 섞였다는 것을 아무도 모른다.
    """

    def post(self):
        with self.settings(FIN_ROLE="reid", ROOT_URLCONF="review.tests"):
            r = self.client.post("/api/reid/box", "{}",
                                 content_type="application/json")
        return json.loads(r.content)

    def test_two_presses_are_two_boxes(self):
        a, b = self.post(), self.post()
        self.assertNotEqual(a["id"], b["id"])
        self.assertNotEqual(a["name"], b["name"])
        self.assertTrue(a["made"] and b["made"])

    def test_a_name_taken_between_the_count_and_the_insert_is_stepped_over(self):
        """세어서 고르는 것은 여전히 짐작이다 — **짐작이 틀렸을 때 조용히
        넘어가지 않는다**는 것이 다르다. 이름이 UNIQUE 라 DB 가 정한다."""
        Individual.objects.create(name="상자 1")     # count()+1 이 집을 자리
        r = self.post()
        self.assertTrue(r["made"])
        self.assertNotEqual(r["name"], "상자 1")
        self.assertEqual(Individual.objects.filter(name=r["name"]).count(), 1)

    def test_a_named_box_is_still_shared(self):
        """이름을 손으로 준 것은 `get_or_create` 그대로다 — 같은 번호를 두
        사람이 적었으면 그것은 같은 개체를 가리킨 것이다."""
        with self.settings(FIN_ROLE="reid", ROOT_URLCONF="review.tests"):
            body = json.dumps({"name": "JTA001"})
            a = json.loads(self.client.post("/api/reid/box", body,
                                            content_type="application/json").content)
            b = json.loads(self.client.post("/api/reid/box", body,
                                            content_type="application/json").content)
        self.assertEqual(a["id"], b["id"])


class DatasetStateTests(TestCase):
    """**아직 안 붙인 것을 보는 자리.** 세는 것은 `reid.dataset_state()` 하나다 —
    화면이 제 나름대로 세면 나중에 명령으로 잴 때 숫자가 갈리고, 그때 어느 쪽이
    맞는지 아무도 모른다."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.boxes = []
        for i, (d, fac) in enumerate([("2016-03-15", "left"),
                                      ("2016-03-15", "left"),
                                      ("2017-05-02", "right")]):
            img = Image.objects.create(path=f"{i}.JPG", width=100, height=100,
                                       obsdate=date.fromisoformat(d))
            self.boxes.append(Box.objects.create(image=img, x1=0, y1=0, x2=60,
                                                 y2=60, source="yolov5"))
        self.ind = Individual.objects.create(name="JTA001", nickname="제돌이")
        for b in self.boxes:
            Identification.objects.create(box=b, individual=self.ind)
        (self.tmp / "items.json").write_text(json.dumps({"items": [
            {"id": b.id, "facing": f}
            for b, f in zip(self.boxes, ["left", "left", "right"])]}),
            encoding="utf-8")

    def state(self, **goal):
        with self.settings(FIN_REID=self.tmp):
            return reid.dataset_state(goal or None)

    def test_it_counts_what_is_there(self):
        d = self.state()
        self.assertEqual((d["n_ind"], d["n_chip"]), (1, 3))
        self.assertEqual(d["rows"][0]["days"], 2)

    def test_the_side_comes_from_the_grid_not_the_judgments(self):
        """분류기가 면을 가를 때 보는 것이 격자다 — DB 의 `facing` 은 사람이
        누른 것만이라 절반 넘게 비어 있고, 그것으로 세면 좌·우 합이 조각 수와
        안 맞는다."""
        r = self.state()["rows"][0]
        self.assertEqual((r["left"], r["right"], r["unknown"]), (2, 1, 0))

    def test_what_the_grid_does_not_know_is_counted_apart(self):
        """좌·우 합이 조각 수와 안 맞는 채로 두면 **어디가 빈 것인지 안 보인다.**"""
        (self.tmp / "items.json").write_text('{"items": []}', encoding="utf-8")
        r = self.state()["rows"][0]
        self.assertEqual((r["left"], r["right"], r["unknown"]), (0, 0, 3))

    def test_it_says_how_far_from_the_goal(self):
        d = self.state(per_individual=10, days_per_individual=2)
        self.assertEqual(d["rows"][0]["need_chips"], 7)
        self.assertEqual(d["rows"][0]["need_days"], 0)      # 2일이라 됐다
        self.assertEqual(d["short_chips"], 7)

    def test_a_one_day_individual_is_short_a_day(self):
        """하루뿐인 개체는 날로 가르면 배우는 쪽이나 재는 쪽 **한 곳에만**
        들어간다 — 많고 적음이 아니라 되나 안 되나다."""
        other = Individual.objects.create(name="JTA002")
        Identification.objects.create(box=self.boxes[0], individual=other)
        row = [r for r in self.state()["rows"] if r["name"] == "JTA002"][0]
        self.assertEqual(row["need_days"], 1)

    def test_the_ones_still_short_come_first(self):
        """보고 나서 무엇을 할지 사람이 또 정해야 하면 대시보드로 끝난다."""
        done = Individual.objects.create(name="JTA009")
        for i in range(10):
            img = Image.objects.create(path=f"d{i}.JPG", width=10, height=10,
                                       obsdate=date(2018, 1, 1 + i % 2))
            b = Box.objects.create(image=img, x1=0, y1=0, x2=9, y2=9,
                                   source="yolov5")
            Identification.objects.create(box=b, individual=done)
        names = [r["name"] for r in self.state()["rows"]]
        self.assertEqual(names, ["JTA001", "JTA009"])   # 모자란 것이 위

    def test_the_page_is_only_on_the_reid_seat(self):
        """`Identification` 의 주인이 하나라 그것을 보는 자리도 거기다."""
        from review.urls import patterns_for
        self.assertIn("dataset", {p.name for p in patterns_for("reid")})
        self.assertNotIn("dataset", {p.name for p in patterns_for("work")})

    def test_the_menu_says_what_this_is_and_which_build(self):
        """자리가 넷이라 **어느 판으로 떠 있는지가 늘 보여야 한다.** `/healthz`
        가 이미 내지만 그건 밖에서 curl 로 물을 때이고, 화면을 보고 있는 사람은
        `/healthz` 를 안 친다."""
        from finweb.version import __version__
        with self.settings(FIN_REID=self.tmp, FIN_ROLE="reid",
                           ROOT_URLCONF="review.tests"):
            html = self.client.get("/dataset").content.decode()
        self.assertIn("DolfinServer2", html)
        self.assertIn(f"v{__version__}", html)

    def test_the_menu_offers_it(self):
        with self.settings(FIN_REID=self.tmp, FIN_ROLE="reid",
                           ROOT_URLCONF="review.tests"):
            html = self.client.get("/dataset").content.decode()
        self.assertIn('href="/dataset"', html)
        self.assertIn("자료 상태", html)


class ClientIpTests(TestCase):
    """**클라이언트가 적어 보낸 값을 믿지 않는다.**

    앞에 nginx 가 있어 `REMOTE_ADDR` 은 도커 브리지 주소다. 그래서 헤더를
    읽어야 하는데 `X-Forwarded-For` 는 **누구나 먼저 적어 보낼 수 있는 자리**라,
    전에 그 맨 앞을 쓰던 것은 헤더 한 줄로 잠금을 피하고 기록에 아무 주소나
    남길 수 있다는 뜻이었다.
    """

    def req(self, **meta):
        from django.test import RequestFactory
        return RequestFactory().get("/", **meta)

    def test_the_header_nginx_overwrites_wins(self):
        """nginx 는 `X-Real-IP` 를 `$remote_addr` 로 **덮어쓴다** — 클라이언트가
        적어 보낸 것은 거기서 지워진다."""
        from review import gate
        r = self.req(HTTP_X_REAL_IP="203.0.113.9",
                     HTTP_X_FORWARDED_FOR="1.2.3.4", REMOTE_ADDR="172.18.0.5")
        self.assertEqual(gate.client_ip(r), "203.0.113.9")

    def test_the_forwarded_chain_is_read_from_the_back(self):
        """`$proxy_add_x_forwarded_for` 는 제가 본 주소를 **뒤에 덧붙인다.**
        앞쪽은 남이 적은 것일 수 있어도 맨 뒤 하나는 nginx 가 적은 것이다."""
        from review import gate
        r = self.req(HTTP_X_FORWARDED_FOR="9.9.9.9, 203.0.113.9",
                     REMOTE_ADDR="172.18.0.5")
        self.assertEqual(gate.client_ip(r), "203.0.113.9")

    def test_without_a_proxy_the_peer_is_the_answer(self):
        """`runserver` 앞에는 nginx 가 없다."""
        from review import gate
        self.assertEqual(gate.client_ip(self.req(REMOTE_ADDR="10.0.0.7")), "10.0.0.7")

    def test_what_goes_into_the_data_must_be_an_address(self):
        """빈 칸이 곧 **"못 알아냈다"** 다 — 자리표시를 넣으면 "모른다" 와
        "알아냈는데 이 값이다" 가 한 칸에서 섞인다."""
        from review import gate
        # `client_ip` 이 잠금용으로 내는 `"?"` — 캐시 열쇠로는 되지만
        # 자료 칸에는 못 넣는다. SQLite 는 검사 없이 받아 버린다
        self.assertIsNone(gate.recordable_ip(self.req(REMOTE_ADDR="?")))
        self.assertIsNone(gate.recordable_ip(self.req(HTTP_X_REAL_IP="어디선가")))
        self.assertEqual(gate.recordable_ip(self.req(REMOTE_ADDR="10.0.0.7")),
                         "10.0.0.7")


class IdentificationStampTests(TestCase):
    """판정에 **때와 자리**를 남긴다. `reviewer` 가 늘 비어 있어서(로그인이
    없다) 누가 했는지를 가리키는 것이 이 둘뿐이다."""

    def setUp(self):
        img = Image.objects.create(path="a.JPG", obsdate=date(2016, 3, 15),
                                   width=100, height=100)
        self.box = Box.objects.create(image=img, x1=0, y1=0, x2=60, y2=60,
                                      source="yolov5", conf=0.9)
        self.ind = Individual.objects.create(name="JTA001")

    def assign(self, **meta):
        with self.settings(FIN_ROLE="reid", ROOT_URLCONF="review.tests"):
            self.client.post("/api/reid/assign",
                             json.dumps({"individual": self.ind.id,
                                         "boxes": [self.box.id]}),
                             content_type="application/json", **meta)
        return Identification.objects.latest("id")

    def test_it_writes_down_where_it_came_from(self):
        self.assertEqual(self.assign(HTTP_X_REAL_IP="203.0.113.9").ip, "203.0.113.9")

    def test_a_spoofed_chain_does_not_get_written_down(self):
        """맨 앞을 읽던 때라면 `1.2.3.4` 가 남았다."""
        self.assertEqual(
            self.assign(HTTP_X_FORWARDED_FOR="1.2.3.4, 203.0.113.9").ip,
            "203.0.113.9")

    def test_an_unknown_place_is_left_empty(self):
        self.assertIsNone(self.assign(REMOTE_ADDR="?").ip)

    def test_the_time_was_already_there(self):
        """`at` 은 `auto_now_add` 다 — 새로 붙인 것은 자리뿐이다."""
        self.assertIsNotNone(self.assign(REMOTE_ADDR="10.0.0.7").at)


class ReviewStampTests(TestCase):
    """검토 판정에도 같은 것을 남긴다. `Review` 의 주인은 **작업 자리**인데,
    그렇지 않은 데서 들어온 줄이 있으면 여기서 보인다."""

    def setUp(self):
        img = Image.objects.create(path="a.JPG", obsdate=date(2016, 3, 15),
                                   width=100, height=100)
        self.box = Box.objects.create(image=img, x1=0, y1=0, x2=60, y2=60,
                                      source="yolov5")

    def post(self, n=1, **meta):
        self.client.post("/api/review", json.dumps({"items": [
            {"box_id": self.box.id, "cls": "fin", "verdict": "ok",
             "edges": "both", "facing": "left"} for _ in range(n)]}),
            content_type="application/json", **meta)

    def test_it_writes_down_where_it_came_from(self):
        self.post(HTTP_X_REAL_IP="203.0.113.9")
        self.assertEqual(Review.objects.latest("id").ip, "203.0.113.9")

    def test_a_spoofed_chain_does_not_get_written_down(self):
        self.post(HTTP_X_FORWARDED_FOR="1.2.3.4, 203.0.113.9")
        self.assertEqual(Review.objects.latest("id").ip, "203.0.113.9")

    def test_one_save_is_one_place_and_one_moment(self):
        """한 번의 저장이 판정 여럿을 만드는데, 그 여럿은 같은 사람이 같은
        자리에서 같은 순간에 누른 것이다 — 줄마다 다시 읽을 것이 아니다."""
        self.post(n=3, HTTP_X_REAL_IP="203.0.113.9")
        self.assertEqual({r.ip for r in Review.objects.all()}, {"203.0.113.9"})

    def test_an_unknown_place_is_left_empty(self):
        self.post(REMOTE_ADDR="?")
        self.assertIsNone(Review.objects.latest("id").ip)


class CatalogTests(TestCase):
    """카탈로그도 **여기서 못 가는 길은 안 낸다.**

    `/photo` 는 `work` 자리의 길인데 `/catalog` 은 `reid` 자리에만 걸려 있어
    (`urls.patterns_for`) 조각을 누르면 **전부 404** 였다. `_nav.html`·홈 카드·
    우클릭 차림표에 이미 건 규칙을 **네 번째 자리에서 또 빠뜨린 것**이다.
    """

    def setUp(self):
        img = Image.objects.create(path="a.JPG", obsdate=date(2016, 3, 15),
                                   width=100, height=100)
        self.box = Box.objects.create(image=img, x1=0, y1=0, x2=60, y2=60,
                                      source="yolov5", conf=0.9)
        ind = Individual.objects.create(name="JTA001")
        Identification.objects.create(box=self.box, individual=ind)

    def page(self, role):
        with self.settings(FIN_ROLE=role, ROOT_URLCONF="review.tests"):
            return self.client.get("/catalog").content.decode()

    def test_the_reid_seat_does_not_offer_the_photo_road(self):
        self.assertNotIn(f'href="/photo/{self.box.id}"', self.page("reid"))

    def test_it_still_says_which_box_and_which_day(self):
        """길은 없어도 **무엇인지는 말한다** — 링크가 아니라 `title` 이 하던 일이다."""
        self.assertIn(f'title="상자 {self.box.id} · 2016-03-15"', self.page("reid"))

    def test_the_road_comes_back_where_it_exists(self):
        """지우지 않고 역할로 가른 값 — 뒷날 `/catalog` 이 `work` 에도 걸리면
        **저절로 되살아난다.**"""
        self.assertIn(f'href="/photo/{self.box.id}"', self.page("work"))

    def test_the_box_list_does_not_carry_the_note(self):
        """상자 목록에서는 메모를 안 보인다 — 쓰는 자리는 `/catalog` 이다.
        안 쓰는 것을 상자마다 실으면 4.7MB 짜리 페이지가 그만큼 더 는다."""
        tmp = Path(tempfile.mkdtemp())
        (tmp / "items.json").write_text('{"items": []}', encoding="utf-8")
        ind = Individual.objects.get(name="JTA001")
        ind.note = "두 마리가 섞였을 수도"
        ind.save()
        with self.settings(FIN_REID=tmp, FIN_ROLE="reid",
                           ROOT_URLCONF="review.tests"):
            html = self.client.get("/reid").content.decode()
        self.assertNotIn("두 마리가 섞였을 수도", html)

    def test_the_three_fields_are_editable_where_you_recognise_the_animal(self):
        """번호를 붙이는 사람은 조각 여러 장을 한꺼번에 봐야 하고, 그것을 펴
        놓는 화면이 여기다. `/reid` 로 건너가 다시 찾아 치게 하면 **그 왕복이
        일의 절반이 된다.**"""
        html = self.page("reid")
        for attr in (f'data-id="', 'data-nick="', 'data-note="'):
            self.assertIn(attr, html)
        self.assertIn("csrftoken", html)      # POST 하려면 쿠키가 있어야 한다

    def test_it_carries_the_nickname_and_the_note(self):
        ind = Individual.objects.get(name="JTA001")
        ind.nickname, ind.note = "제돌이", "왼쪽만 있다"
        ind.save()
        with self.settings(FIN_ROLE="reid", ROOT_URLCONF="review.tests"):
            r = self.client.get("/catalog").context["rows"][0]
        self.assertEqual((r["nick"], r["note"]), ("제돌이", "왼쪽만 있다"))

    def test_a_note_can_be_written_and_cleared(self):
        """**"모르겠다" 를 적을 자리다.** 가장 잦은 것이 "묶음은 맞는 것 같은데
        어느 개체인지 모르겠다" 인데, 보류함(조각을 빼는 것)에도 이름을 안
        건드리는 것(안 본 것과 구별이 안 된다)에도 안 맞는다."""
        ind = Individual.objects.get(name="JTA001")
        with self.settings(FIN_ROLE="reid", ROOT_URLCONF="review.tests"):
            def put(v):
                return self.client.post(
                    "/api/reid/box", json.dumps({"id": ind.id, "note": v}),
                    content_type="application/json")
            put("두 마리가 섞였을 수도")
            ind.refresh_from_db()
            self.assertEqual(ind.note, "두 마리가 섞였을 수도")
            put("")                       # 비울 수 있다 — 빈 값이 곧 답이다
            ind.refresh_from_db()
            self.assertEqual(ind.note, "")

    def test_writing_a_note_does_not_touch_the_number(self):
        """`if ind_id` 갈래가 먼저 채 가면 "이름이 비었다" 가 난다 — 별명을
        이름 갈래 앞에 둔 것과 같은 이유다."""
        ind = Individual.objects.get(name="JTA001")
        with self.settings(FIN_ROLE="reid", ROOT_URLCONF="review.tests"):
            r = self.client.post("/api/reid/box",
                                 json.dumps({"id": ind.id, "note": "메모"}),
                                 content_type="application/json")
        self.assertEqual(r.status_code, 200)
        ind.refresh_from_db()
        self.assertEqual(ind.name, "JTA001")

    def test_it_carries_the_last_day_for_the_screen_to_sort_by(self):
        """화면이 `최근에 본 것부터` 로 고쳐 세울 때 쓴다. `span` 에서 잘라
        쓰게 두면 한 날짜뿐인 개체에서 갈린다 — 그때는 `~` 가 없다."""
        with self.settings(FIN_ROLE="reid", ROOT_URLCONF="review.tests"):
            rows = self.client.get("/catalog").context["rows"]
        self.assertEqual(rows[0]["last"], "2016-03-15")


class TileStampTests(TestCase):
    """조각에 **찍힌 때**를 적는다. `날짜·시간순` 으로 세워 놓고도 **어디서
    끊기는지는 눈으로 못 본다** — 잇달아 찍은 것이 같은 개체인 자리가 많다."""

    def setUp(self):
        from django.utils import timezone
        import datetime
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "look").mkdir()
        img = Image.objects.create(
            path="a.JPG", obsdate=date(2016, 3, 15), width=100, height=100,
            exifdatetime=timezone.make_aware(datetime.datetime(2016, 3, 15, 5, 48, 12)))
        b = Box.objects.create(image=img, x1=0, y1=0, x2=60, y2=60,
                               source="yolov5", conf=0.9)
        (self.tmp / "items.json").write_text(json.dumps({"items": [
            {"id": b.id, "day": "2016-03-15", "facing": "left", "rough": 0.1}]}),
            encoding="utf-8")

    def test_the_stamp_is_minutes_not_seconds(self):
        """초까지 넣으면 글자가 조각을 덮는다."""
        with self.settings(FIN_REID=self.tmp, FIN_ROLE="reid",
                           ROOT_URLCONF="review.tests"):
            html = self.client.get("/reid").content.decode()
        got = json.loads(re.search(r"const ITEMS = (\[.*?\]);", html, re.S).group(1))
        self.assertEqual(got[0]["at"], "2016-03-15 05:48:12")
        self.assertIn("i.at.slice(0, 16)", html)      # 화면은 분까지만 낸다
