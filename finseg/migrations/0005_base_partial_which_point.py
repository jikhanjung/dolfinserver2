"""`base_partial` 을 참/거짓에서 **어느 삽입점이 짐작인가**로 넓힌다.

`0002`~`0004` 는 `choices` 만 건드려 SQL 상 no-op 이었지만 **이것은 진짜 자료
변경**이다. 참/거짓에서 문자열로 바뀌므로 SQLite 가 테이블을 다시 만든다.

그래서 `AlterField` 하나로 맡기지 않고 **더하고·옮기고·지우고·이름 바꾸기**로
쪼갰다. 자동 변환에 맡기면 참이 `"True"` 인지 `"1"` 인지 백엔드가 정하고,
사람의 판정 1,100건 위에서 그것을 확인할 방법이 없다.

옛 참은 `"unknown"` 이 된다 — 그때는 어느 쪽인지 묻지 않았으므로 앞인지 뒤인지
알 수 없다. **모른다는 사실 자체가 자료**라 `"both"` 같은 것으로 지어내지 않고
그대로 남긴다. 되짚어 볼 때 사람이 고친다.
"""
from django.db import migrations, models

BASE_PARTIAL = [
    ("", "다 봤다"),
    ("front", "앞삽입점 짐작"),
    ("rear", "뒷삽입점 짐작"),
    ("both", "둘 다 짐작"),
    ("unknown", "불완전 — 어느 쪽인지 모름"),
]


def to_text(apps, schema_editor):
    """참 → `unknown`, 거짓 → `""`, NULL 은 NULL 그대로."""
    for name in ("Mask", "Review"):
        model = apps.get_model("finseg", name)
        for row in model.objects.all().only("id", "base_partial"):
            old = row.base_partial
            row.base_partial_new = None if old is None else ("unknown" if old else "")
            row.save(update_fields=["base_partial_new"])


def to_bool(apps, schema_editor):
    """되돌리기 — 빈 값이 아니면 참이다. 어느 쪽이었는지는 여기서 잃는다."""
    for name in ("Mask", "Review"):
        model = apps.get_model("finseg", name)
        for row in model.objects.all().only("id", "base_partial_new"):
            new = row.base_partial_new
            row.base_partial = None if new is None else bool(new)
            row.save(update_fields=["base_partial"])


class Migration(migrations.Migration):

    dependencies = [("finseg", "0004_alter_review_cls")]

    operations = [
        migrations.AddField(
            model_name="mask", name="base_partial_new",
            field=models.CharField(max_length=10, choices=BASE_PARTIAL,
                                   blank=True, default=""),
        ),
        migrations.AddField(
            model_name="review", name="base_partial_new",
            field=models.CharField(max_length=10, choices=BASE_PARTIAL,
                                   null=True, blank=True, default=None),
        ),
        migrations.RunPython(to_text, to_bool),
        migrations.RemoveField(model_name="mask", name="base_partial"),
        migrations.RemoveField(model_name="review", name="base_partial"),
        migrations.RenameField(model_name="mask", old_name="base_partial_new",
                               new_name="base_partial"),
        migrations.RenameField(model_name="review", old_name="base_partial_new",
                               new_name="base_partial"),
        migrations.AlterField(
            model_name="mask", name="base_partial",
            field=models.CharField(max_length=10, choices=BASE_PARTIAL,
                                   blank=True, default="",
                                   help_text="짐작해서 찍은 삽입점"),
        ),
        migrations.AlterField(
            model_name="review", name="base_partial",
            field=models.CharField(max_length=10, choices=BASE_PARTIAL,
                                   null=True, blank=True, default=None,
                                   help_text="사람의 판단. NULL 이면 제안 그대로"),
        ),
    ]
