"""**어느 `fin.db` 를 열고 있나** 를 명령마다 말한다.

기본 자리가 저장소 밖(`/srv/dolfinserver2/db/`)으로 옮겨 가면서 생기는 함정이
하나 있다 — 그 자리가 없는 기계에서 명령을 돌리면 **SQLite 가 빈 파일을 만들고
아무 말 없이 돈다.** 상자 0개로 도는 `crops` 는 오류를 안 내고 그냥 아무것도
안 한다. 그러면 "왜 아무 일도 안 일어나지" 를 한참 뒤에 묻게 된다.

그래서 시스템 검사로 **먼저 말한다.** 검사는 `manage.py` 명령마다 도는 자리라
빠뜨릴 길이 없다.
"""
from pathlib import Path

from django.conf import settings
from django.core.checks import Warning, register


@register()
def db_is_where_we_think(app_configs, **kwargs):
    name = Path(settings.DATABASES["default"]["NAME"])
    if name.exists():
        return []
    return [Warning(
        f"`fin.db` 가 그 자리에 없다: {name}",
        hint="운영 자리는 /srv/dolfinserver2/db/fin.db 다. 다른 기계·시험·개발이면 "
             "FIN_DB 로 대 줄 것 — 안 주면 빈 DB 가 만들어지고 명령이 아무 말 "
             "없이 0건을 처리한다. 시험 사본은 deploy/host/test_db.sh 가 뜬다.",
        id="finseg.W001",
    )]
