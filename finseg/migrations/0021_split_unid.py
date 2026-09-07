"""`식별 불가능` 을 둘로 가른다 — **사진 품질**과 **뒷날이 밋밋한 것**.

가르는 축은 하나다: **다른 사진이면 되나.** 앞엣것은 되고 뒤엣것은 안 된다.

**둘을 자동으로 가를 수 있는지는 아직 모른다** — 2026-09-07 에 자를 넷 재고
다 어긋났다. 이 마이그레이션이 하는 일은 **사람이 갈라 준 답을 담을 자리**를
만드는 것이고, 자는 그 답이 몇십 장 모인 뒤에 고른다.

자리는 자료가 아니라 뼈대다 — 화면을 한 번도 안 연 기계에서도 있어야 한다
(`0015`·`0019` 와 같은 이유). **옛 `unid` 는 그대로 둔다**: 이미 든 390장은
"아직 안 가른 것" 이고, 다시 보게 하지 않는다. 앞으로 누를 때부터 갈리면 된다.
"""
from django.db import migrations, models


def make(apps, schema_editor):
    Individual = apps.get_model("finseg", "Individual")
    for kind, name in (("unid_photo_qual", "식별 불가능 · 사진 품질"),
                       ("unid_too_smooth", "식별 불가능 · 뒷날이 밋밋")):
        if not Individual.objects.filter(kind=kind).exists():
            Individual.objects.create(kind=kind, name=name)


def back(apps, schema_editor):
    (apps.get_model("finseg", "Individual").objects
     .filter(kind__in=("unid_photo_qual", "unid_too_smooth")).delete())


class Migration(migrations.Migration):

    dependencies = [
        ('finseg', '0020_identification_pred_identification_pred_by_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='individual',
            name='kind',
            field=models.CharField(blank=True, choices=[('', '개체'), ('hold', '임시보관함'), ('notfin', '지느러미 아님'), ('unid', '식별 불가능'), ('unid_photo_qual', '식별 불가능 · 사진 품질'), ('unid_too_smooth', '식별 불가능 · 뒷날이 밋밋')], default='', help_text='비어 있으면 진짜 개체다', max_length=20),
        ),
        migrations.RunPython(make, back),
    ]
