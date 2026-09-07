"""`식별 불가능` 을 둘로 가른다 — **사진 품질**과 **뒷날이 밋밋한 것**.

가르는 축은 하나다: **다른 사진이면 되나.** 앞엣것은 되고 뒤엣것은 안 된다.

실측이 그 둘을 정반대로 가른다 (2026-09-07 · 배포된 3자 앙상블) — 흐린 것은
모델의 확신이 낮아 지금 자로 잘 걸리는데(`unid` 전체와 개체 조각의 AUROC
0.971), **밋밋한 것은 확신이 되레 높다.** 매끈한 뒷날이 카탈로그의 다른 매끈한
개체와 잘 닮아 **자신 있게 잘못 찍는다.** 확신 하나로는 뒤엣것을 못 거른다.

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
