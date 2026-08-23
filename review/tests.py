"""검토 화면이 **규칙을 말하는지** 시험한다.

밑동 현이라는 것을 화면이 말하지 않으면 사람이 스스로와 어긋나고, 어긋난 검토는
나중에 되살릴 수 없다 — 그래서 이것은 꾸밈이 아니라 자료의 요건이다. 문구는
`finseg.baseline` 에 있고 화면이 그것을 그대로 띄운다. 여기서 잡는 것은
**둘이 갈라지는 것**이다.
"""
import json
import re

from django.test import SimpleTestCase, TestCase

from finseg import baseline, geometry, rules
from finseg.models import Box, Crop, Image, Mask, Review, Run

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
