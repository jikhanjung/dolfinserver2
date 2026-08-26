"""템플릿이 이 자리의 역할을 알아야 한다.

`_nav.html` 이 **이 역할에 없는 링크를 안 그린다.** 안 그러면 차림표가 404 로
가는 길을 내밀고, 누른 사람은 화면이 깨진 줄 안다 — 실은 그 자리가 그 일을
안 하기로 한 것이다 (`finweb/settings.py` 의 `FIN_ROLE`).
"""
from django.conf import settings


def role(request):
    return {"fin_role": settings.FIN_ROLE}
