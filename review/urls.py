from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("api/batch", views.batch, name="batch"),
    path("api/review", views.save, name="save"),
    path("api/progress", views.progress, name="progress"),
]
