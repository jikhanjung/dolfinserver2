from django.apps import AppConfig


class FinsegConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "finseg"
    verbose_name = "지느러미 분할"

    def ready(self):
        # **어느 `fin.db` 를 열었나** 를 명령마다 말한다 (`finseg/checks.py`).
        from . import checks  # noqa: F401
