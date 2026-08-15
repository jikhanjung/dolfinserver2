from django.conf import settings
from django.contrib import admin
from django.urls import include, path
from django.views.static import serve

urlpatterns = [
    path("", include("review.urls")),
    path("admin/", admin.site.urls),
]

if settings.DEBUG:
    # 크롭은 개발 서버가 직접 낸다. 배포하면 nginx 가 맡는다
    # (`.guides/web/operations.md`).
    urlpatterns += [
        path("crops/<path:path>", serve, {"document_root": settings.FIN_CROPS}),
    ]
