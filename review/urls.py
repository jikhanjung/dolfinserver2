from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("api/batch", views.batch, name="batch"),
    path("api/review", views.save, name="save"),
    path("api/progress", views.progress, name="progress"),
    path("compare", views.compare, name="compare"),
    path("photo/<int:box_id>", views.photo, name="photo"),
    path("edit/<int:box_id>", views.edit, name="edit"),
]
