from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("api/batch", views.batch, name="batch"),
    path("api/review", views.save, name="save"),
    path("api/progress", views.progress, name="progress"),
    path("compare", views.compare, name="compare"),
    path("detect", views.detect, name="detect"),
    path("reid", views.reid, name="reid"),
    path("catalog", views.catalog, name="catalog"),
    path("api/reid/box", views.reid_box, name="reid_box"),
    path("api/reid/groups", views.reid_groups, name="reid_groups"),
    path("api/reid/suggest", views.reid_suggest, name="reid_suggest"),
    path("api/reid/assign", views.reid_assign, name="reid_assign"),
    path("reid/chip/<int:box_id>.png", views.reid_chip, name="reid_chip"),
    path("photo/<int:box_id>", views.photo, name="photo"),
    path("edit/<int:box_id>", views.edit, name="edit"),
]
