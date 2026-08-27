"""템플릿이 **이 자리가 무엇인지**를 알아야 한다 — 역할과 판.

`_nav.html` 이 **이 역할에 없는 링크를 안 그린다.** 안 그러면 차림표가 404 로
가는 길을 내밀고, 누른 사람은 화면이 깨진 줄 안다 — 실은 그 자리가 그 일을
안 하기로 한 것이다 (`finweb/settings.py` 의 `FIN_ROLE`).

**판도 같은 줄에 낸다.** 자리가 넷이고(운영 둘·시험·개발) 배포가 뒤바뀌거나
낡은 이미지로 뜨는 일이 실제로 있었다 (2026-08-26 에 시험 자리가 옛 태그로
떴다 — `deploy/env.example` 주석). `/healthz` 가 이미 그것을 내지만 그건
**밖에서 curl 로 물을 때**이고, 화면을 보고 있는 사람은 `/healthz` 를 안
친다. 늘 보이는 자리에 적어 두면 **말 안 해도 눈에 걸린다.**
"""
from django.conf import settings

from finweb.version import __version__


def role(request):
    return {"fin_role": settings.FIN_ROLE, "fin_version": __version__}
