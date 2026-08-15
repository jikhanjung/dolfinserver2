"""검토 화면이 **규칙을 말하는지** 시험한다.

밑동 현이라는 것을 화면이 말하지 않으면 사람이 스스로와 어긋나고, 어긋난 검토는
나중에 되살릴 수 없다 — 그래서 이것은 꾸밈이 아니라 자료의 요건이다. 문구는
`finseg.baseline` 에 있고 화면이 그것을 그대로 띄운다. 여기서 잡는 것은
**둘이 갈라지는 것**이다.
"""
import re

from django.test import SimpleTestCase

from finseg import baseline

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
