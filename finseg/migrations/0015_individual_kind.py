"""`holding`(참/거짓) → `kind`(갈래). 그리고 `지느러미 아님` 자리를 만든다.

**자료를 옮기는 단계를 사이에 둔다.** `makemigrations` 가 만든 것은 옛 칸을
그냥 지우는데, 그러면 임시보관함에 든 312장이 **진짜 개체로 둔갑한다** —
`catalog()` 가 그것을 세기 시작하고, 성적이 조용히 바뀐다.

갈래를 둘로 늘린 이유는 `notfin` 때문이다. re-ID 자리(GCP)는 `Review` 를 못
쓴다(주인이 작업 자리 하나다). 그런데 분류하다 보면 "이건 지느러미가 아니다"
를 반드시 만난다 — 그것을 **제 레인(개체)으로 말해 두면**, 되받은 뒤 작업
자리에서 사람이 진짜 분류를 골라 `Review` 로 옮겨 적는다.
"""
from django.db import migrations, models


def to_kind(apps, schema_editor):
    Individual = apps.get_model("finseg", "Individual")
    Individual.objects.filter(holding=True).update(kind="hold")
    # 이름도 함께 바꾼다 — `보류함` 은 "곧 뺄 것" 처럼 읽히는데, 실제로는
    # 몇 달을 거기 머무는 것이 있다. `임시보관함` 이 그 쓰임에 맞다.
    Individual.objects.filter(kind="hold", name="보류함").update(name="임시보관함")
    if not Individual.objects.filter(kind="notfin").exists():
        Individual.objects.create(kind="notfin", name="지느러미 아님")


def back(apps, schema_editor):
    Individual = apps.get_model("finseg", "Individual")
    Individual.objects.filter(kind="notfin").delete()
    Individual.objects.filter(kind="hold").update(holding=True, name="보류함")


class Migration(migrations.Migration):

    dependencies = [("finseg", "0014_mask_cls")]

    operations = [
        migrations.AddField(
            model_name="individual",
            name="kind",
            field=models.CharField(
                blank=True, default="",
                choices=[("", "개체"), ("hold", "임시보관함"), ("notfin", "지느러미 아님")],
                help_text="비어 있으면 진짜 개체다", max_length=10),
        ),
        migrations.RunPython(to_kind, back),
        migrations.RemoveField(model_name="individual", name="holding"),
        migrations.AddConstraint(
            model_name="individual",
            constraint=models.UniqueConstraint(
                condition=models.Q(("kind", ""), _negated=True),
                fields=("kind",), name="one_box_per_kind"),
        ),
    ]
