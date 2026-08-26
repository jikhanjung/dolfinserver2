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
