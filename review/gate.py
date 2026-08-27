"""문 하나. **코드를 맞혀야 안으로 들어온다** (`settings.FIN_ACCESS_CODE`).

왜 이것이 필요한가 — 이 앱에는 `login_required` 가 하나도 없고, `/api/reid/assign`
같은 **쓰기 경로가 그대로 열려 있다.** 그것이 쓰는 것은 사람의 개체 판정이고
다시 만들 수 없다. 주소를 아는 사람이 지울 수 있으면 안 된다.

**막는 자리를 미들웨어로 둔 이유**는 뷰마다 데코레이터를 붙이면 새로 만든 뷰
하나가 빠지는 날이 오기 때문이다. 여기서는 **기본이 막힘이고 예외만 적는다** —
검토 화면이 "기본값으로 두고 예외만 누른다" 로 만들어진 것과 같은 이유다.
"""
import hmac
import time

from django.conf import settings
from django.core.cache import cache
from django.shortcuts import redirect
from django.utils.http import urlencode

# 문 밖에 두는 것. **`/healthz` 가 여기 있어야 한다** — smoke 와 배포가 그것을
# 읽고, 코드를 모르는 자리에서도 살아 있는지는 물을 수 있어야 한다. 대신
# `/healthz` 는 세는 수만 내고 자료를 안 낸다.
OPEN = ("/enter", "/healthz", "/static/", "/favicon.ico")

# 코드를 몇 번 틀리면 그 주소를 잠근다. **짧은 코드를 쓰면 이것만으로는 못
# 막는다** — 코드 자체가 길어야 한다 (`FIN_ACCESS_CODE` 주석).
MAX_TRIES = 10
LOCK_SECONDS = 15 * 60


def client_ip(request):
    """누가 두드렸나. **클라이언트가 적어 보낸 값을 믿지 않는다.**

    앞에 nginx 가 있어서 `REMOTE_ADDR` 은 도커 브리지 주소(`172.x`)다. 그래서
    헤더를 보아야 하는데, `X-Forwarded-For` 는 **클라이언트가 먼저 적어 보낼 수
    있는 자리**다. 전에 그 **맨 앞**을 썼는데 거기가 바로 그 적어 보낸 값이라,
    헤더 한 줄로 잠금(`note_failure`)을 피하고 기록에 아무 주소나 남길 수 있었다.

    - `X-Real-IP` — nginx 가 `$remote_addr` 로 **덮어쓴다**(`proxy_set_header`).
      클라이언트가 적어 보내도 지워지므로 이것이 가장 믿을 만하다
    - `X-Forwarded-For` 의 **맨 뒤** — nginx 가 `$proxy_add_x_forwarded_for` 로
      제가 본 주소를 **뒤에 덧붙인다.** 앞쪽은 남이 적은 것일 수 있어도
      맨 뒤 하나는 nginx 가 적은 것이다
    - `REMOTE_ADDR` — nginx 가 없는 자리(`runserver`)에서는 이것이 맞다

    **두 nginx.conf 가 둘 다 저 두 헤더를 건다** (`deploy/nginx.conf` ·
    `deploy/gcp/nginx.conf`). 앞에 프록시를 하나 더 놓게 되면 여기를 함께 볼 것.
    """
    real = (request.META.get("HTTP_X_REAL_IP") or "").strip()
    if real:
        return real
    fwd = request.META.get("HTTP_X_FORWARDED_FOR", "")
    if fwd:
        return fwd.rsplit(",", 1)[-1].strip()
    return request.META.get("REMOTE_ADDR", "?")


def recordable_ip(request):
    """자료에 남길 주소. **말이 되는 것만 남기고 아니면 `None`.**

    `client_ip` 은 잠금에 쓰려고 못 알아내면 `"?"` 를 낸다 — 캐시 열쇠로는
    그것으로 충분하지만 **자료에 넣을 값은 아니다.** `GenericIPAddressField`
    가 그런 값을 SQLite 에서는 그대로 받아 버려서(검사는 `full_clean` 때만
    돈다), 나중에 세는 쪽이 `"?"` 를 주소 하나로 센다.

    빈 칸이 곧 **"못 알아냈다"** 이고, 그것이 자리표시를 넣는 것보다 낫다.
    """
    import ipaddress
    v = client_ip(request)
    try:
        return str(ipaddress.ip_address(v))
    except ValueError:
        return None


def locked_for(request):
    """남은 잠금 시간(초). 0 이면 안 잠겼다."""
    until = cache.get(f"fin-lock:{client_ip(request)}")
    return max(0, int(until - time.time())) if until else 0


def note_failure(request):
    key = f"fin-try:{client_ip(request)}"
    n = (cache.get(key) or 0) + 1
    cache.set(key, n, LOCK_SECONDS)
    if n >= MAX_TRIES:
        cache.set(f"fin-lock:{client_ip(request)}", time.time() + LOCK_SECONDS,
                  LOCK_SECONDS)
    return n


def clear_failures(request):
    cache.delete(f"fin-try:{client_ip(request)}")
    cache.delete(f"fin-lock:{client_ip(request)}")


def matches(given):
    """**`==` 를 쓰지 않는다.** 문자열 비교는 다른 글자가 나오는 순간 멈춰서,
    맞은 글자 수가 걸린 시간에 새어 나온다.

    바이트로 바꿔서 넘긴다 — `compare_digest` 는 ASCII 밖 글자가 든 **문자열**을
    거절한다(`TypeError`). 코드에 한글을 넣는 것을 막을 이유가 없다.
    """
    return hmac.compare_digest(str(given or "").encode("utf-8"),
                               settings.FIN_ACCESS_CODE.encode("utf-8"))


class AccessCodeMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if settings.FIN_ACCESS_CODE and not request.session.get("fin_ok"):
            path = request.path
            if not any(path == o or path.startswith(o) for o in OPEN):
                # `next` 로 돌려보낸다 — 링크를 받고 들어온 사람이 문을 지난 뒤
                # 처음 화면으로 떨어지면 그 링크가 무엇이었는지 잃는다.
                # **감싸서 넣는다** — 원래 주소에 `?`·`&` 가 있으면 그대로
                # 붙였을 때 그 뒤가 `/enter` 의 인자로 읽힌다.
                return redirect("/enter?" + urlencode({"next": request.get_full_path()}))
        return self.get_response(request)
