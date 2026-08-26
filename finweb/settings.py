"""Django 설정.

**`fin.db` 가 곧 Django 의 기본 데이터베이스다.** 스키마의 주인은
`finseg/models.py` 하나이고, 파이프라인도 검토 UI 도 같은 ORM 을 쓴다 —
주인이 둘이면 언젠가 갈라지고, 갈라진 것은 눈에 띄지 않는다.

지금은 한 사람이 쓰지만 **멀티유저로 갈 것을 전제로** auth·session·admin 을
처음부터 켜 둔다. 나중에 붙이면 `Review.reviewer` 를 문자열에서 FK 로 옮기는
마이그레이션을 실제 자료 위에서 해야 한다.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.environ.get("FIN_SECRET_KEY", "dev-only-not-a-secret")
DEBUG = os.environ.get("FIN_DEBUG", "1") == "1"
ALLOWED_HOSTS = os.environ.get("FIN_ALLOWED_HOSTS", "*").split(",")
CSRF_TRUSTED_ORIGINS = [
    o for o in os.environ.get("FIN_TRUSTED_ORIGINS", "").split(",") if o
]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "finseg",
    "review",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    # **세션 뒤에 온다** — 세션을 읽어야 문을 지난 적이 있는지 안다.
    "review.gate.AccessCodeMiddleware",
]

ROOT_URLCONF = "finweb.urls"
WSGI_APPLICATION = "finweb.wsgi.application"

TEMPLATES = [{
    "BACKEND": "django.template.backends.django.DjangoTemplates",
    "APP_DIRS": True,
    "DIRS": [],
    "OPTIONS": {"context_processors": [
        "django.template.context_processors.request",
        "django.contrib.auth.context_processors.auth",
        "django.contrib.messages.context_processors.messages",
        # 차림표가 이 자리의 역할을 알아야 없는 링크를 안 그린다
        "review.context.role",
    ]},
}]

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        # **운영 자리는 `/srv/dolfinserver2/db/` 다** (2026-08-26). 저장소
        # `db/` 를 쓰지 않는다 — 검토 화면이 컨테이너로 도는데, 그것이 보는
        # 파일과 파이프라인이 쓰는 파일이 다르면 **판정이 한쪽에만 쌓인다.**
        # 형제 프로젝트들과 자리도 같아져서(`/srv/<proj>/db/`) 백업 레인과
        # 배포가 같은 곳을 본다.
        #
        # **다른 기계·시험·개발은 `FIN_DB` 로 대 준다.** 시험 자리는 NAS
        # 백업을 복사해 쓴다 (`deploy/host/test_db.sh`) — 사본으로 돌려야
        # 무엇을 해도 사람의 판정이 안 다치고, 덤으로 백업이 성한지도 잰다.
        "NAME": Path(os.environ.get("FIN_DB", "/srv/dolfinserver2/db/fin.db")),
        # **SQLite 는 쓰기가 하나다.** 파이프라인이 도는 중에 검토를 저장하면
        # 잠긴다 — 형제 프로젝트가 그것으로 프레임 229장을 잃었다. WAL 과
        # timeout 은 완충일 뿐 해결이 아니므로, 운영 자리는 한 곳으로 둔다.
        "OPTIONS": {"timeout": 20, "init_command": "PRAGMA journal_mode=WAL;"},
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation."
             "UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "ko-kr"
TIME_ZONE = "Asia/Seoul"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
# **저장소 밖에서 받아 오는 것들이 여기 있다** — `onnxruntime-web`(26MB)과
# 내보낸 `.onnx`(37MB). 둘 다 `.gitignore` 다. 없으면 `/detect` 가 왜 없는지
# 화면에서 말한다 (`review.views.detect`).
STATICFILES_DIRS = [BASE_DIR / "static"]
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ---- 이 자리가 무슨 일을 하나 ---------------------------------------------
# **개체를 만들고 지느러미를 개체에 넣는 일은 한 자리에서만 한다.** 그래야
# `Individual`·`Identification` 의 주인이 하나로 남고, 두 자리의 판정을 합칠
# 일이 없다 (`HANDOFF.md` 의 `## 서버를 둘로 나눈다`).
#
#   work  검출·분할·밑동·검토 — `/` `/review` `/edit` `/compare` `/detect` `/photo`
#   reid  개체 분류만        — `/reid` `/catalog` `/api/reid/*`
#
# **막는 자리가 앱이어야 한다.** nginx 로만 막으면 정작 새는 자리가 안 막힌다 —
# `runserver` 로 띄울 때는 앞에 nginx 가 없고, 구멍은 거기서 열린다:
# **`/reid` 를 열기만 해도 보류함 `Individual` 이 하나 생긴다**
# (`review.views.reid` 의 `get_or_create`). URLconf 에서 아예 빼면 그 길이 없다.
#
# 기본은 `work` — 여기 습관을 안 바꾼다. GCP 쪽 `.env` 에만 `reid` 를 적는다.
# 시간별 백업이 `integrity_check` 에 걸리면 여기 센티넬을 세운다. **DB 옆에
# 둔다** — 호스트의 `db/` 가 그대로 컨테이너에 마운트되므로 cron(호스트)과
# 화면(컨테이너)이 같은 파일을 본다 (`deploy/host/backup_db.py`).
FIN_SENTINEL = Path(os.environ.get(
    "FIN_SENTINEL", Path(DATABASES["default"]["NAME"]).parent / "INTEGRITY_FAIL"))

# ---- 접속 코드 -------------------------------------------------------------
# **이 앱에는 사람마다의 인증이 없다.** `login_required` 도 없고 쓰기 경로가
# 그대로 열려 있어, 주소를 아는 사람은 누구나 판정을 써 넣을 수 있다 — 그리고
# 그것은 다시 만들 수 없는 자료다. 코드 하나로 문을 막는다.
#
# **이것이 인증의 대신이지 인증은 아니다.** 다음 셋을 알고 쓴다:
#   · 누가 했는지가 안 남는다 (`Review.reviewer` 가 NULL 이다)
#   · 한 사람만 뺄 수 없다 — 코드를 바꾸면 다 같이 나간다
#   · **TLS 없이 공개 주소로 열면 코드도 세션 쿠키도 평문으로 간다.**
#     지금 GCP 는 tailnet 에만 열려 있고 그 구간은 암호화된다
#
# 빈 값이면 문이 없다 — m710q 의 개발·시험 자리는 그대로 둔다.
FIN_ACCESS_CODE = os.environ.get("FIN_ACCESS_CODE", "")
# 코드를 맞힌 뒤 얼마나 유지되나. 분류하는 일은 며칠씩 이어지므로 길게 둔다.
SESSION_COOKIE_AGE = int(os.environ.get("FIN_SESSION_DAYS", "30")) * 86400
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = "Lax"
# **이 자리가 https 뒤에 있나.** 손잡이 하나로 셋을 켠다 — 따로 두면 하나를
# 빠뜨리고, 빠뜨린 것이 조용히 틀린다.
#
# `SECURE_PROXY_SSL_HEADER` 를 안 켜면 **코드를 맞게 넣어도 403** 이 난다:
# Django 는 nginx 뒤라 스스로를 http 로 알고 `http://<호스트>` 를 옳은 출처로
# 삼는데, 브라우저는 `Origin: https://<호스트>` 를 보낸다. 스킴 한 글자가
# 어긋나서 CSRF 가 막는다 — 포트 때문에 겪은 것과 같은 종류다.
#
# nginx 가 `X-Forwarded-Proto` 를 늘 덮어쓰고 컨테이너는 127.0.0.1 에만
# 바인드하므로 그 헤더를 속일 자리가 없다.
FIN_HTTPS = os.environ.get("FIN_HTTPS", "0") == "1"
if FIN_HTTPS:
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True

FIN_ROLE = os.environ.get("FIN_ROLE", "work")
if FIN_ROLE not in ("work", "reid"):
    raise ValueError(f"FIN_ROLE 은 work 아니면 reid 다: {FIN_ROLE!r}")

LOGIN_URL = "/admin/login/"
LOGIN_REDIRECT_URL = "/"

# ---- 저장소 밖의 것들 -------------------------------------------------------
# 사진은 NAS, 크롭과 학습 자료는 파생물이다. 전부 gitignore 다.
FIN_PHOTOS = Path(os.environ.get("FIN_PHOTOS", "/srv/dolfinserver/uploads"))
FIN_CROPS = Path(os.environ.get("FIN_CROPS", BASE_DIR / "crops"))
# re-ID 조각 꾸러미. **격자를 갈아 끼울 수 있어야 한다** — 옛 상자에서 후보를
# 더 뽑아 섞어 보는 일이 이 값을 바꿔 가며 도는 일이다 (`reid_chips --out`)
FIN_REID = Path(os.environ.get("FIN_REID", BASE_DIR / "reid" / "v1"))
FIN_SRC_DB = Path(os.environ.get(
    "FIN_SRC_DB", BASE_DIR.parent / "dolfinserver" / "db.sqlite3"))
