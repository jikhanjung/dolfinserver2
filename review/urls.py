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
    path("api/reid", views.reid_save, name="reid_save"),
    path("reid/chip/<int:box_id>.png", views.reid_chip, name="reid_chip"),
    path("photo/<int:box_id>", views.photo, name="photo"),
    path("edit/<int:box_id>", views.edit, name="edit"),
]
