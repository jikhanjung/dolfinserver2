"""관리자 화면. 자료를 눈으로 훑고 판정을 고쳐 매길 때 쓴다.

멀티유저로 가면 여기서 계정을 만든다 — 그래서 `auth` 를 처음부터 켜 두었다.
"""
from django.contrib import admin

from .models import Box, Crop, Image, Mask, Review, Run


@admin.register(Image)
class ImageAdmin(admin.ModelAdmin):
    list_display = ("id", "path", "obsdate", "width", "height")
    list_filter = ("obsdate",)
    search_fields = ("path",)


@admin.register(Box)
class BoxAdmin(admin.ModelAdmin):
    list_display = ("id", "image", "area", "boxname", "src_not_fin",
                    "src_not_identifiable")
    list_filter = ("source", "src_not_fin", "src_not_identifiable")
    search_fields = ("boxname",)
    raw_id_fields = ("image",)


@admin.register(Crop)
class CropAdmin(admin.ModelAdmin):
    list_display = ("box_id", "path", "w", "h")
    raw_id_fields = ("box",)


@admin.register(Run)
class RunAdmin(admin.ModelAdmin):
    list_display = ("id", "kind", "model", "git_sha", "host", "started_at",
                    "finished_at")
    list_filter = ("kind", "host")


@admin.register(Mask)
class MaskAdmin(admin.ModelAdmin):
    list_display = ("id", "box_id", "run", "conf", "area", "base_partial",
                    "is_current")
    list_filter = ("run", "is_current", "base_partial")
    raw_id_fields = ("box", "run")


@admin.register(Review)
class ReviewAdmin(admin.ModelAdmin):
    list_display = ("id", "box_id", "cls", "verdict", "edges", "base_partial",
                    "reviewer", "at")
    list_filter = ("cls", "verdict", "edges", "reviewer")
    raw_id_fields = ("box", "mask")
