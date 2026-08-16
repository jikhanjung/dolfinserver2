"""표로 시험할 값어치가 있는 것 둘 — **판정 규칙**과 **좌표 사상**.

    python manage.py test finseg

이 둘을 고른 이유는 **틀려도 눈에 안 띄기 때문**이다. 화면은 그대로 뜨고
학습도 그대로 돌아간다. 판정 규칙이 어긋나면 화면에서 뺀 것이 학습에 들어가
있고, 좌표 사상이 어긋나면 마스크가 몇 화소 밀린 모습으로만 나타난다. 둘 다
성적표에 조용히 섞여 들어오고, 그때는 어느 바퀴부터 틀렸는지 알 수 없다.

**표로 적는다.** 축이 넷이라 조합이 말로는 안 잡히고, 나중에 규칙을 고칠 때
무엇을 무너뜨렸는지 여기가 말해 준다.

데이터베이스가 필요한 것은 `effective_review` 하나뿐이다 — 나머지는 저장하지
않은 인스턴스로 시험한다.
"""
from django.test import SimpleTestCase, TestCase

from finseg import baseline, geometry, rules
from finseg.models import EDGES, Box, Crop, Image, Mask, Review, Run


def review(cls="fin", verdict="ok", edges="both", polygon="", base_line=""):
    """저장하지 않은 판정 하나."""
    return Review(cls=cls, verdict=verdict, edges=edges,
                  polygon=polygon, base_line=base_line)


class LabelTests(SimpleTestCase):
    """`rules.label_of` — 판정 한 건이 학습 자료에서 무엇이 되는가."""

    def test_table(self):
        for name, r, want in [
            # 아무 말도 없는 것과 "없다" 고 말한 것은 **다르다.** 앞을 배경으로
            # 쓰면 SAM2 가 놓친 지느러미까지 배경이라고 가르치게 된다.
            ("판정 없음",        None,                              rules.PENDING),
            ("분류 비어 있음",   review(cls=""),                    rules.PENDING),
            ("아무것도아님",     review(cls="none", verdict="", edges=""),
                                                                    rules.BACKGROUND),
            ("등지느러미 통과",  review(),                          rules.POSITIVE),
            ("꼬리 통과",        review(cls="fluke"),               rules.POSITIVE),
            ("주둥이 통과",      review(cls="rostrum"),             rules.POSITIVE),
            ("기타 통과",        review(cls="other"),               rules.POSITIVE),
            # 가려진 것도 **버리지 않는다.** 보이는 실루엣은 맞는 것이고,
            # `edges` 는 re-ID 축이지 학습 축이 아니다 (rules.py 2026-08-16)
            ("앞날만",           review(edges="leading"),           rules.POSITIVE),
            ("뒷날만",           review(edges="trailing"),          rules.POSITIVE),
            ("끄트머리",         review(edges="tip"),               rules.POSITIVE),
            ("머리",             review(cls="head"),                rules.POSITIVE),
            ("몸통",             review(cls="body"),                rules.POSITIVE),
            ("새",               review(cls="bird"),                rules.POSITIVE),
            ("교정 전",          review(verdict="fix"),             rules.PENDING),
            ("윗윤곽 교정",      review(verdict="fix", polygon="1,1 2,2 3,1"),
                                                                    rules.POSITIVE),
            ("아래 직선 교정",   review(verdict="fix", base_line="1,9 9,9"),
                                                                    rules.POSITIVE),
        ]:
            with self.subTest(name):
                self.assertEqual(rules.label_of(r), want)

    def test_edges_never_drops(self):
        """**`edges` 는 학습 자료에서 아무것도 가르지 않는다.**

        전에는 `neither` 가 크롭을 통째로 버렸다. 그것을 되돌린 이유는
        `rules.py` 의 2026-08-16 항목에 있다 — 끄트머리만 보인다는 것은 앞에
        온전한 지느러미가 있다는 뜻이라, 버리면 가장 좋은 표본이 함께 사라진다.
        """
        for e, _ in EDGES:
            with self.subTest(e):
                self.assertEqual(rules.label_of(review(edges=e)), rules.POSITIVE)
                self.assertEqual(
                    rules.label_of(review(verdict="fix", polygon="1,1 2,2 3,1",
                                          edges=e)), rules.POSITIVE)
        self.assertNotIn(rules.DROP,
                         [rules.label_of(review(edges=e)) for e, _ in EDGES])

    def test_none_is_background_regardless(self):
        """`cls='none'` 이면 다른 축을 보지 않는다 — 상자 안에 아무것도 없다."""
        self.assertEqual(
            rules.label_of(review(cls="none", verdict="fix", edges="tip")),
            rules.BACKGROUND)


class ResolveTests(SimpleTestCase):
    """`rules.resolve` — 사람의 교정이 마스크를 이긴다."""

    def setUp(self):
        self.box = Box(x1=10, y1=10, x2=20, y2=20)
        self.mask = Mask(polygon="0,0 5,0 5,5", base_line="0,5 5,5",
                         base_partial=False)

    def test_review_overrides_mask(self):
        r = review(verdict="fix", polygon="1,1 9,1 9,9", base_line="1,9 9,9")
        st = rules.resolve(self.box, mask=self.mask, review=r)
        self.assertEqual(st["polygon"], "1,1 9,1 9,9")
        self.assertEqual(st["base_line"], "1,9 9,9")
        self.assertEqual(st["label"], rules.POSITIVE)

    def test_falls_back_to_mask(self):
        """사람이 안 건드린 축은 마스크 것이 그대로 온다."""
        st = rules.resolve(self.box, mask=self.mask, review=review())
        self.assertEqual(st["polygon"], "0,0 5,0 5,5")
        self.assertEqual(st["base_line"], "0,5 5,5")

    def test_partial_is_tri_state(self):
        """`base_partial` 은 NULL 이 '제안 그대로' 다 — False 와 다르다."""
        self.mask.base_partial = True
        st = rules.resolve(self.box, mask=self.mask, review=review())
        self.assertTrue(st["base_partial"])          # 판정이 말하지 않았다 → 마스크
        st = rules.resolve(self.box, mask=self.mask,
                           review=Review(cls="fin", verdict="ok", edges="both",
                                         base_partial=False))
        self.assertFalse(st["base_partial"])         # 사람이 아니라고 했다


class EffectiveReviewTests(TestCase):
    """**멀티유저의 갈림길.** 지금은 가장 늦은 것이 이긴다."""

    def setUp(self):
        img = Image.objects.create(path="a/b.jpg", width=100, height=100)
        self.box = Box.objects.create(image=img, x1=10, y1=10, x2=30, y2=30)
        self.run = Run.objects.create(kind="sam2")

    def test_latest_wins(self):
        Review.objects.create(box=self.box, cls="none")
        Review.objects.create(box=self.box, cls="fin", verdict="ok", edges="both")
        self.assertEqual(rules.effective_review(self.box).cls, "fin")
        self.assertEqual(rules.label_of(rules.effective_review(self.box)),
                         rules.POSITIVE)

    def test_rejected_is_kept(self):
        """고쳐 매겨도 **지운 자취가 남는다** — 어려운 음성이고 분모다."""
        Review.objects.create(box=self.box, cls="none")
        Review.objects.create(box=self.box, cls="fin", verdict="ok", edges="both")
        self.assertEqual(self.box.reviews.count(), 2)

    def test_current_mask_only(self):
        old = Mask.objects.create(box=self.box, run=self.run, polygon="0,0 1,0 1,1",
                                  is_current=False)
        new = Mask.objects.create(box=self.box, run=self.run, polygon="2,2 3,2 3,3")
        self.assertEqual(rules.effective_mask(self.box).id, new.id)
        self.assertNotEqual(rules.effective_mask(self.box).id, old.id)


def crop(x0=100, y0=200, side=320, w=640):
    """원본 (x0,y0) 에서 `side` 정사각형을 잘라 `w` 로 편 크롭."""
    return Crop(path="c.jpg", x0=x0, y0=y0, x1=x0 + side, y1=y0 + side, w=w, h=w)


class GeometryTests(SimpleTestCase):
    """**이 식은 `geometry` 에 둘뿐이다.** 어긋나면 마스크가 밀린 모습으로만 난다."""

    def test_polygon_text_roundtrip(self):
        self.assertEqual(geometry.loads("1,2 3,4"), [(1, 2), (3, 4)])
        self.assertEqual(geometry.dumps([(1.4, 2.6), (3.0, 4.0)]), "1,3 3,4")
        self.assertEqual(geometry.loads(""), [])

    def test_crop_orig_roundtrip(self):
        c = crop()
        pts = [(100, 200), (420, 200), (260, 360), (419.5, 519.5)]
        back = geometry.to_orig(geometry.to_crop(pts, c), c)
        for (x, y), (bx, by) in zip(pts, back):
            self.assertAlmostEqual(x, bx, places=6)
            self.assertAlmostEqual(y, by, places=6)

    def test_crop_corners(self):
        """크롭의 왼쪽 위는 (0,0), 오른쪽 아래는 한 변이다."""
        c = crop()
        self.assertEqual(geometry.to_crop([(c.x0, c.y0)], c), [(0.0, 0.0)])
        self.assertEqual(geometry.to_crop([(c.x1, c.y1)], c), [(640.0, 640.0)])
        self.assertEqual(c.scale, 2.0)

    def test_crop_rect_is_square_and_centered(self):
        x0, y0, x1, y1 = geometry.crop_rect((1000, 1000, 1100, 1080), 5472, 3648, 2.0)
        self.assertEqual(x1 - x0, y1 - y0)
        self.assertEqual(x1 - x0, 200)               # 긴 변 100 × pad 2.0
        self.assertEqual((x0 + x1) / 2, 1050)        # 상자 가운데
        self.assertEqual((y0 + y1) / 2, 1040)

    def test_crop_rect_pushes_at_edge(self):
        """가장자리에서는 **줄이지 않고 민다** — 눌린 지느러미를 만들지 않는다."""
        x0, y0, x1, y1 = geometry.crop_rect((10, 10, 110, 110), 5472, 3648, 2.0)
        self.assertEqual((x0, y0), (0, 0))
        self.assertEqual((x1 - x0, y1 - y0), (200, 200))
        x0, y0, x1, y1 = geometry.crop_rect((5400, 3600, 5460, 3640), 5472, 3648, 2.0)
        self.assertEqual((x1, y1), (5472, 3648))
        self.assertEqual((x1 - x0, y1 - y0), (120, 120))

    def test_crop_rect_clamps_to_image(self):
        """사진보다 큰 정사각형을 요구받을 때만 줄인다."""
        x0, y0, x1, y1 = geometry.crop_rect((100, 100, 1700, 1200), 5472, 3648, 2.0)
        self.assertEqual(x1 - x0, 3200)              # 3648 보다 작으니 그대로
        x0, y0, x1, y1 = geometry.crop_rect((100, 100, 2000, 1200), 5472, 3648, 2.0)
        self.assertEqual((x1 - x0, y1 - y0), (3648, 3648))   # 사진 높이로 줄었다
        self.assertGreaterEqual(x0, 0)
        self.assertGreaterEqual(y0, 0)
        self.assertLessEqual(x1, 5472)
        self.assertLessEqual(y1, 3648)


class BaselineTests(SimpleTestCase):
    """밑동 현 — 아래를 자르는 식."""

    SQUARE = [(10, 10), (90, 10), (90, 90), (10, 90)]

    def test_cut_below_removes_lower_half(self):
        out = baseline.cut_below(self.SQUARE, (0, 50), (100, 50), 128)
        self.assertLessEqual(max(y for _, y in out), 52)
        self.assertGreaterEqual(len(out), 3)

    def test_cut_below_without_line_is_identity(self):
        self.assertEqual(baseline.cut_below(self.SQUARE, None, None, 128),
                         self.SQUARE)

    def test_chord_and_height(self):
        length, deg = baseline.chord((0, 0), (10, 0))
        self.assertAlmostEqual(length, 10.0)
        self.assertAlmostEqual(deg, 0.0)
        # 기울어져도 뜻이 같아야 한다 — 현에 **수직인** 거리로 잰다
        h = baseline.height_above([(0, 0), (10, 0), (5, -8)], (0, 0), (10, 0))
        self.assertAlmostEqual(h, 8.0)

    def test_propose_is_box_bottom(self):
        p0, p1 = baseline.propose_from_box((10, 20, 60, 80))
        self.assertEqual((p0, p1), ((10.0, 80.0), (60.0, 80.0)))


class FinalPointsTests(SimpleTestCase):
    """`rules.final_points` — 화면과 내보내기가 **같은 것**을 본다."""

    def state(self, cls="fin", base_line="0,250 640,250"):
        return {"polygon": "0,0 640,0 640,640 0,640", "base_line": base_line,
                "cls": cls, "base_partial": False}

    def test_fin_is_cut(self):
        c = crop(x0=0, y0=0, side=640, w=640)
        pts = rules.final_points(self.state(), c)
        self.assertLessEqual(max(y for _, y in pts), 252)

    def test_fluke_is_not_cut(self):
        """밑동 현은 **등지느러미에만** 뜻이 있다."""
        c = crop(x0=0, y0=0, side=640, w=640)
        pts = rules.final_points(self.state(cls="fluke"), c)
        self.assertGreater(max(y for _, y in pts), 600)

    def test_no_base_is_not_cut(self):
        c = crop(x0=0, y0=0, side=640, w=640)
        pts = rules.final_points(self.state(base_line=""), c)
        self.assertGreater(max(y for _, y in pts), 600)

    def test_empty_polygon(self):
        c = crop(x0=0, y0=0, side=640, w=640)
        st = self.state()
        st["polygon"] = ""
        self.assertEqual(rules.final_points(st, c), [])
