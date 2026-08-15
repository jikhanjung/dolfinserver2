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
        "NAME": Path(os.environ.get("FIN_DB", BASE_DIR / "fin.db")),
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
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

LOGIN_URL = "/admin/login/"
LOGIN_REDIRECT_URL = "/"

# ---- 저장소 밖의 것들 -------------------------------------------------------
# 사진은 NAS, 크롭과 학습 자료는 파생물이다. 전부 gitignore 다.
FIN_PHOTOS = Path(os.environ.get("FIN_PHOTOS", "/srv/dolfinserver/uploads"))
FIN_CROPS = Path(os.environ.get("FIN_CROPS", BASE_DIR / "crops"))
FIN_SRC_DB = Path(os.environ.get(
    "FIN_SRC_DB", BASE_DIR.parent / "dolfinserver" / "db.sqlite3"))
