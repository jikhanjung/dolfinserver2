"""검토 화면이 **규칙을 말하는지** 시험한다.

밑동 현이라는 것을 화면이 말하지 않으면 사람이 스스로와 어긋나고, 어긋난 검토는
나중에 되살릴 수 없다 — 그래서 이것은 꾸밈이 아니라 자료의 요건이다. 문구는
`finseg.baseline` 에 있고 화면이 그것을 그대로 띄운다. 여기서 잡는 것은
**둘이 갈라지는 것**이다.
"""
import io
import json
import re

import numpy as np

from django.test import SimpleTestCase, TestCase

from finseg import baseline, geometry, onnxdet, reid, rules
from finseg.models import (Box, Crop, Identification, Image, Individual,
                           Mask, Review, Run)

BLOCK = re.compile(r'<div class="rule">(.*?)</div>', re.S)
TAG = re.compile(r"<[^>]+>")


class RuleShownTests(SimpleTestCase):

    def block(self):
        """규칙 칸의 HTML. **거기 있어야 한다** — 주석이나 스크립트가 아니라."""
        m = BLOCK.search(self.client.get("/").content.decode())
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
        """화면에 박아 두면 `onnxdet` 을 고쳐도 JS 가 옛 값을 쓴다."""
        html = self.page()
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
        self.assertEqual(json.loads(m.group(1))[0]["rep"], self.a.id)

    def test_the_holding_box_is_made_and_is_not_a_catalog_entry(self):
        """**보류함은 개체가 아니라 자리다.** 어디에 넣을지 모르겠는 것을
        아무 상자에나 넣으면 그 상자가 오염되고, 미분류로 두면 다음에 또 같은
        고민을 처음부터 한다. 그렇다고 카탈로그에 세면 개체 수가 틀린다."""
        self.client.get("/reid")
        hold = Individual.objects.get(holding=True)
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
        hold = Individual.objects.get(holding=True)
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
        hold = Individual.objects.get(holding=True)
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
