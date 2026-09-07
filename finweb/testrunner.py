"""**시험은 이 기계의 자료를 안 본다.**

`manage.py test` 가 `fin.db` 를 안 건드리는 것은 Django 가 시험용 DB 를 따로
세우기 때문인데, **파일 쪽에는 그런 장치가 없었다.** `FIN_REID`·`FIN_CROPS`
는 그냥 settings 의 값이라, 시험이 운영 격자(`reid/v3`, 임베딩 세 벌 100MB)를
그대로 열 수 있다.

2026-09-07 에 `FIN_REID` 기본값을 지운 `reid/v1` 에서 `reid/v3` 로 옮기다가
드러났다 — **시험이 4.1초에서 9.7초가 됐다.** 없는 자리를 가리키던 덕에 여태
안 걸렸을 뿐, 자료를 읽고 있었던 것이다. 빈 격자를 물리면 262개가 그대로
통과하니 **필요해서 읽은 것도 아니었다.**

그것이 왜 나쁜가:

- **기계마다 다른 것을 잰다.** 격자를 갈아 끼운 기계에서는 다른 것이 실린다
- **없는 기계에서는 안 도는 시험**이 생길 수 있다 — 새로 받은 자리에서
  `manage.py test` 가 처음 하는 일이 "왜 이게 안 돌지" 가 된다
- 자료를 쓰는 시험은 **제 것을 만들어 쓴다**(`self.settings(FIN_REID=self.tmp)`).
  이미 그렇게 쓰고 있으므로, 기본을 막아도 잃는 것이 없다

**기본이 막힘이고 예외만 적는다** — 문(`review/gate.py`)을 그렇게 세운 것과
같은 규칙이다.
"""
import tempfile
from pathlib import Path

from django.test.runner import DiscoverRunner
from django.test.utils import override_settings


class FinTestRunner(DiscoverRunner):
    """자료 경로를 빈 임시 자리로 돌려놓고 시험을 돌린다."""

    def setup_test_environment(self, **kwargs):
        super().setup_test_environment(**kwargs)
        self._tmp = tempfile.TemporaryDirectory(prefix="fin-test-")
        root = Path(self._tmp.name)
        grid = root / "reid"
        grid.mkdir()
        # 빈 격자 하나는 둔다 — 없는 자리와 빈 자리는 다른 상태이고,
        # 시험이 재려는 것은 대개 뒤엣것이다
        (grid / "items.json").write_text('{"n": 0, "items": []}')
        (root / "crops").mkdir()
        self._over = override_settings(FIN_REID=grid, FIN_CROPS=root / "crops",
                                       FIN_PHOTOS=root / "photos")
        self._over.enable()

    def teardown_test_environment(self, **kwargs):
        self._over.disable()
        self._tmp.cleanup()
        super().teardown_test_environment(**kwargs)
