"""주소. **어느 것이 걸리는지는 `FIN_ROLE` 이 정한다.**

개체 분류(`/reid`)와 나머지를 갈라 둔 것은 화면을 정리하려는 것이 아니라
**`Individual`·`Identification` 의 주인을 하나로 두려는 것**이다
(`HANDOFF.md` 의 `## 서버를 둘로 나눈다`).

**미들웨어로 403 을 내지 않고 여기서 뺀다.** 경로가 아예 없으면 실수로 도는
길이 없다. 템플릿이 `{% url %}` 를 하나도 안 쓰고 링크를 경로로 직접 적어서
(`href="/reid"`) 빼도 `reverse` 가 안 깨진다 — 대신 `_nav.html` 이 이 역할에
없는 링크를 안 그린다.
"""
from django.conf import settings
from django.urls import path

from . import views

# 어느 역할에서나 있는 것. `/healthz` 는 밖에서 **이 자리가 어느 역할로
# 떴는지**를 확인하는 자리다 — 배포가 뒤바뀐 것을 그것으로 잡는다.
common = [
    path("healthz", views.healthz, name="healthz"),
    # 문 (`review/gate.py`). 코드가 없는 자리에서는 `/` 로 돌려보낸다.
    path("enter", views.enter, name="enter"),
]

# 검출·분할·밑동 — 사람이 마스크와 밑동을 보는 쪽
work = [
    path("", views.home, name="home"),
    path("review", views.index, name="index"),
    path("api/batch", views.batch, name="batch"),
    path("api/review", views.save, name="save"),
    path("api/progress", views.progress, name="progress"),
    path("compare", views.compare, name="compare"),
    path("detect", views.detect, name="detect"),
    path("photo/<int:box_id>", views.photo, name="photo"),
    path("edit/<int:box_id>", views.edit, name="edit"),
]

# re-ID — 개체를 만들고 지느러미를 넣는 쪽. **`Identification`·`Individual` 에
# 쓰는 것은 전부 여기 있다.**
reid = [
    path("", views.reid, name="home"),
    path("reid", views.reid, name="reid"),
    path("catalog", views.catalog, name="catalog"),
    path("api/reid/box", views.reid_box, name="reid_box"),
    path("api/reid/groups", views.reid_groups, name="reid_groups"),
    path("api/reid/suggest", views.reid_suggest, name="reid_suggest"),
    path("api/reid/assign", views.reid_assign, name="reid_assign"),
    path("reid/chip/<int:box_id>.png", views.reid_chip, name="reid_chip"),
]

def patterns_for(role):
    """그 역할에서 걸리는 길. **판단이 여기 한 줄로 있어야 시험이 그것을 잰다.**"""
    return common + (reid if role == "reid" else work)


urlpatterns = patterns_for(settings.FIN_ROLE)
