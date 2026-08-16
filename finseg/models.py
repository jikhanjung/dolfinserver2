"""`fin.db` 의 스키마.

**원본은 이 데이터베이스다.** 사진도 가중치도 학습 자료도 다시 만들 수 있지만
사람의 판정은 못 만든다.

## 판정이 여러 축인 이유

한 칸에 예/아니오 하나가 아니라 축이 넷이다. 각각 **아래로 흘러가는 곳이
다르기 때문**이고, 하나로 뭉치면 나중에 되살릴 수 없다.

| 축 | 값 | 어디에 쓰나 |
|---|---|---|
| `cls` | fin · fluke · rostrum · head · body · bird · pectoral · other · none | 학습의 분류. **`none` 만 배경이다** |
| `verdict` | ok · fix | 마스크 윤곽이 맞나 |
| `edges` | both · leading · trailing · tip | **re-ID 전용.** 어느 날로 매칭할 수 있나 |
| `base_line`·`base_partial` | 두 점 · 어느 삽입점이 짐작인가 | 아래 경계. 형태 계측을 믿어도 되나 |
| `facing` | 앞이 왼쪽·오른쪽 | **re-ID.** 좌현·우현을 가르고 현에 부호를 준다 |

**꼬리지느러미를 배경으로 두지 않는 이유**가 여기 있다. 물 위의 검은 삼각형이라
등지느러미와 실루엣이 거의 같은데, 배경으로 가르치면 모델이 "이 삼각형은 아니다"
라는 날카로운 결정을 배워야 한다. 명시적 분류로 주면 훨씬 쉽게 배운다 —
시각적으로 가까운 어려운 음성은 배경보다 별도 분류가 낫다.

## `head`·`body`·`bird`·`pectoral` 을 나중에 더한 이유 (2026-08-16)

첫 500장을 검토하고서야 **옛 YOLOv5 가 무엇을 헛보는지**가 드러났다. 새의 날개,
그리고 각도에 따라 잡히는 **돌고래 몸통**이다. 둘 다 처음에는 `none` 으로 갔다.

그것이 위 문단과 어긋난다는 것이 이 변경의 이유다. 새 날개는 물 위의 삼각형이라
꼬리지느러미와 같은 자리에 있고, **몸통은 더 나쁘다** — 같은 동물의 같은 색·질감을
"배경" 이라고 가르치는 셈이라, 반쯤 잠긴 지느러미의 재현율을 깎을 수 있다.

`head` 는 `rostrum`(주둥이)만으로는 모자라서 더했다. 수면 위로 올라오는 것은
주둥이 끝만이 아니라 머리 전체인 경우가 많고, 그 둘은 실루엣이 다르다.
`pectoral`(가슴지느러미)도 같은 줄기다 — 물 위로 드는 순간의 실루엣이
등지느러미와 닮았고, `body` 로 뭉뚱그리면 그 닮음을 가르칠 수 없다.

**`CLASSES` 의 순서가 곧 YOLO 클래스 번호다** (`export_yolo.NAMES`). 그래서
자료를 한 번이라도 내보낸 뒤에는 중간에 끼워 넣으면 안 된다 — 이미 쓴 라벨
파일의 번호가 통째로 뜻을 잃는다. 지금까지는 `--dry-run` 뿐이라 자유로웠다.
그 뒤로는 **`none` 바로 앞에 덧붙이기만** 할 것.

**`other` 는 남겨 둔다.** 아직 이름 붙일 만큼 자주 안 나온 것들의 자리이고,
`other` 가 늘어나면 그때 또 쪼개면 된다 — 이번이 그렇게 한 것이다.

`edges` 는 **지금 안 적어 두면 re-ID 갈 때 전부 다시 검토해야 하는 것**이다.
photo-ID 는 주로 뒷날의 결각으로 개체를 가리므로, 뒷날만 보이는 지느러미는
여전히 쓸모가 있고 앞날만 보이는 것은 값어치가 훨씬 떨어진다. 끄트머리만
보이는 것은 re-ID 에 못 쓴다 — **그래도 검출 학습에는 그대로 쓴다**
(`rules.py` 의 2026-08-16 항목).

## 판정을 상자와 마스크로 나눈 이유

형제 프로젝트는 판정이 하나였다 — 자동 분할이 개체와 마스크를 함께 냈으니
"이것이 대상인가" 와 "이 마스크가 맞는가" 를 가를 이유가 없었다. 여기는 상자가
먼저 있고 마스크가 뒤에 붙으므로 두 판단이 갈린다.

- **`cls='none'`** — 상자 안에 아무것도 없다. 옛 YOLOv5 의 오검출이다.
  **엔진을 갈아도 유효하다**
- **`verdict='fix'`** — 대상은 맞는데 마스크가 틀렸다. **엔진에 딸린 판정**이라
  분할기를 바꾸면 다시 받아야 한다

## 여러 사람이 볼 것을 전제로 한다

`Review` 는 **덮어쓰지 않고 쌓인다.** 한 상자에 여러 사람의 판정이 붙을 수 있고
같은 사람이 고쳐 매길 수도 있다. 무엇이 유효한지 고르는 규칙은 `rules.py`
하나에만 있다 — 지금은 "사람마다 최신, 그 뒤 합의" 이고, 사람이 늘면 그 함수만
바꾼다.
"""
from django.conf import settings
from django.db import models

# 기본값이 맨 앞이다 — 검토는 **누르지 않으면 등지느러미·양날 온전·통과**다.
# 순서가 곧 검토 화면의 숫자키다. **새 분류는 `none` 앞에 붙인다** — 뒤에 붙이면
# 손에 익은 자리가 밀린다.
CLASSES = [
    ("fin", "등지느러미"),
    ("fluke", "꼬리지느러미"),
    ("rostrum", "주둥이"),
    ("head", "머리"),
    ("body", "몸통"),
    ("bird", "새"),
    ("pectoral", "가슴지느러미"),
    ("other", "기타"),
    ("none", "아무것도아님"),
]
# **보이는 것으로 이름 붙인다.** 앞의 `neither`("둘 다 가림")는 못 본 것을
# 말했는데, 검토하는 사람은 보이는 것을 보고 누르지 안 보이는 것을 헤아려
# 누르지 않는다. 넷 다 **순수한 re-ID 축**이다 — 학습에서 빼는 일은 안 한다.
EDGES = [
    ("both", "양날"),
    ("leading", "앞날만"),
    ("trailing", "뒷날만"),
    ("tip", "끄트머리"),
]
# 밑동 두 점 중 **실제로 못 보고 짐작한 것**. `edges` 와 짝이지만 재는 것이
# 다르다 — `edges` 는 "어느 날로 매칭할 수 있나", 이것은 "밑동 현을 정규화
# 기준으로 믿어도 되나" 다.
#
# **어느 쪽인지를 남기는 이유**: re-ID 는 주로 뒷날 결각으로 개체를 가린다.
# 뒷삽입점이 짐작이면 결각을 재는 쪽 끝이 흔들려 값이 크게 떨어지지만,
# 앞삽입점만 짐작이면 현 각도는 흔들려도 뒷날 자체는 온전해 여전히 쓸 만하다.
# 참/거짓 하나로 두면 나중에 이 둘을 한꺼번에 버리거나 한꺼번에 쓰는 수밖에
# 없다. 지금 안 갈라 두면 그때 전부 다시 검토해야 한다.
#
# `unknown` 은 이 축을 참/거짓으로 쓰던 때의 것이다. **어느 쪽인지 모른다는
# 사실 자체가 자료**라 지어내지 않고 그대로 둔다 — 되짚어 볼 때 고친다.
BASE_PARTIAL = [
    ("", "다 봤다"),
    ("front", "앞삽입점 짐작"),
    ("rear", "뒷삽입점 짐작"),
    ("both", "둘 다 짐작"),
    ("unknown", "불완전 — 어느 쪽인지 모름"),
]
# **밑동 두 점 중 어느 쪽이 앞(머리 쪽)인가.** 저장된 두 점은 화면 좌표라
# 왼쪽·오른쪽일 뿐이고, 앞·뒤는 돌고래가 어느 쪽을 보느냐에 달렸다. 그 한 비트를
# 안 적으면 세 가지가 함께 막힌다 — `base_partial` 의 앞·뒤가 뜻을 잃고, re-ID 의
# 좌현·우현을 못 가르며(`CLAUDE.md` 6단계), 밑동 현의 각도가 부호를 못 갖는다.
#
# **`edges` 처럼 비어 있게 두면 안 된다.** 그 축은 900장 넘게 기본값 하나로
# 남아 있었다 — 사람이 안 누르는 축은 안 채워진다. 그래서 손으로 100장쯤 받아
# 기하 규칙(앞날이 가파르다 — `baseline.py` 의 기울기 −41 대 0.5~2.2)의 정확도를
# 재고, 충분하면 제안으로 바꿔 예외만 누르게 한다.
FACING = [
    ("", "모름"),
    ("left", "앞이 왼쪽"),
    ("right", "앞이 오른쪽"),
]
VERDICTS = [("ok", "통과"), ("fix", "교정 필요")]
RUN_KINDS = [("import", "들이기"), ("crop", "크롭"), ("sam2", "SAM2.1"),
             ("yolo", "YOLO"), ("export", "내보내기"), ("train", "학습")]


class Image(models.Model):
    """사진 한 장. 화소는 저장소 밖(NAS)에 있고 여기는 경로만 든다."""
    src_id = models.IntegerField(unique=True, null=True,
                                 help_text="옛 dolfinrest_dolfinimage.id")
    path = models.CharField(max_length=300, unique=True,
                            help_text="사진 뿌리 기준 상대경로")
    obsdate = models.DateField(null=True, db_index=True)
    exifdatetime = models.DateTimeField(null=True, blank=True)
    width = models.IntegerField(null=True)
    height = models.IntegerField(null=True)

    class Meta:
        ordering = ["obsdate", "path"]

    def __str__(self):
        return self.path


class Box(models.Model):
    """SAM2 에 넣을 프롬프트 상자. 지금은 옛 YOLOv5 가 낸 것이다."""
    src_id = models.IntegerField(unique=True, null=True,
                                 help_text="옛 dolfinrest_dolfinbox.id")
    image = models.ForeignKey(Image, on_delete=models.CASCADE, related_name="boxes")
    x1 = models.IntegerField()
    y1 = models.IntegerField()
    x2 = models.IntegerField()
    y2 = models.IntegerField()
    area = models.IntegerField(null=True)
    source = models.CharField(max_length=30, default="yolov5")
    # 옛 DB 에 이미 붙어 있던 것. **버리지 않고 들고 온다** — 개체명은 re-ID 의
    # 씨앗이고, 시민과학자가 남긴 표시는 공짜로 얻는 어려운 음성이다.
    boxname = models.CharField(max_length=100, blank=True, default="",
                               db_index=True, help_text="JTA### 개체명")
    src_not_fin = models.BooleanField(default=False)
    src_not_identifiable = models.BooleanField(default=False)

    class Meta:
        ordering = ["id"]

    @property
    def box(self):
        return self.x1, self.y1, self.x2, self.y2


class Crop(models.Model):
    """원본에서 잘라 낸 자리. 마스크 좌표를 원본으로 되돌리는 데 쓴다.

    사진 가장자리에서는 정사각형이 못 되므로 **실제 잘린 사각형**을 그대로 적는다.
    """
    box = models.OneToOneField(Box, on_delete=models.CASCADE, related_name="crop")
    path = models.CharField(max_length=300, help_text="크롭 뿌리 기준 상대경로")
    x0 = models.IntegerField()
    y0 = models.IntegerField()
    x1 = models.IntegerField()
    y1 = models.IntegerField()
    w = models.IntegerField()
    h = models.IntegerField()

    @property
    def scale(self):
        """크롭 한 변 / 원본에서 잘린 한 변."""
        return self.w / (self.x1 - self.x0)


class Run(models.Model):
    """무엇이 언제 무엇으로 돌았나. 마스크는 반드시 하나에 매인다.

    **어떤 코드가 만든 자료인지 나중에 반드시 묻게 된다.**
    """
    kind = models.CharField(max_length=20, choices=RUN_KINDS, db_index=True)
    model = models.CharField(max_length=300, blank=True, default="",
                             help_text="모델 id 또는 가중치 경로")
    git_sha = models.CharField(max_length=40, blank=True, default="")
    host = models.CharField(max_length=100, blank=True, default="")
    params = models.JSONField(default=dict, blank=True)
    started_at = models.DateTimeField(auto_now_add=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    note = models.TextField(blank=True, default="")

    class Meta:
        ordering = ["-id"]

    def __str__(self):
        return f"run {self.id} {self.kind}"


class Mask(models.Model):
    """후보 마스크. **덮어쓰지 않고 쌓는다** — 엔진을 갈아도 같은 상자 위에서
    나란히 견줄 수 있어야 한다.

    폴리곤은 **원본 화소 좌표**다. 크롭 좌표로 두면 크롭 여유(`--pad`)를 바꾸는
    순간 저장된 것이 전부 뜻을 잃는다.

    아래를 자를 직선은 **마스크를 잘라서 저장하지 않고 따로 든다** — 삽입점을
    정하는 규칙을 고쳐도 SAM2 가 낸 것을 잃지 않고, 사람이 고칠 것도 점 두 개로
    끝난다.
    """
    box = models.ForeignKey(Box, on_delete=models.CASCADE, related_name="masks")
    run = models.ForeignKey(Run, on_delete=models.CASCADE, related_name="masks")
    polygon = models.TextField(help_text='"x,y x,y ..." 원본 화소 좌표. 자르기 전')
    area = models.IntegerField(null=True)
    conf = models.FloatField(null=True)
    base_line = models.TextField(blank=True, default="",
                                 help_text='"x,y x,y" 앞·뒤 삽입점')
    base_partial = models.CharField(max_length=10, choices=BASE_PARTIAL,
                                    blank=True, default="",
                                    help_text="짐작해서 찍은 삽입점")
    is_current = models.BooleanField(default=True, db_index=True)
    # **밑동 제안을 누가 만들었나.** `run` 은 폴리곤을 만든 엔진이고 이것은 아래
    # 직선을 채운 엔진이라 서로 다르다 — SAM2 가 딴 마스크 위에 키포인트 모델이
    # 두 점을 얹는다. 한 칸에 뭉치면 "이 제안이 어느 코드에서 나왔나" 를 물을 수
    # 없고, 그 물음은 반드시 나온다.
    base_run = models.ForeignKey(Run, on_delete=models.SET_NULL, null=True,
                                 blank=True, related_name="base_masks",
                                 help_text="base_line 을 채운 실행")

    class Meta:
        ordering = ["-id"]
        indexes = [models.Index(fields=["box", "is_current"])]


class Review(models.Model):
    """사람의 판정. **쌓인다** — 고쳐 매긴 자취가 남아야 하고, 여러 사람이
    같은 상자를 볼 수 있다.

    지운 것도 남는다 — 어려운 음성 표본이고 재현율의 분모다.
    """
    box = models.ForeignKey(Box, on_delete=models.CASCADE, related_name="reviews")
    mask = models.ForeignKey(Mask, on_delete=models.SET_NULL, null=True, blank=True,
                             related_name="reviews")
    cls = models.CharField(max_length=10, choices=CLASSES)
    verdict = models.CharField(max_length=10, choices=VERDICTS, blank=True,
                               default="", help_text="cls='none' 이면 빈 값")
    edges = models.CharField(max_length=10, choices=EDGES, blank=True, default="")
    facing = models.CharField(max_length=10, choices=FACING, blank=True,
                              default="",
                              help_text="밑동 두 점 중 앞(머리 쪽)이 어디인가")
    polygon = models.TextField(blank=True, default="",
                               help_text="위 윤곽 교정본. 없으면 마스크 그대로")
    base_line = models.TextField(blank=True, default="",
                                 help_text="아래 직선 교정본. 없으면 제안 그대로")
    base_partial = models.CharField(max_length=10, choices=BASE_PARTIAL,
                                    null=True, blank=True, default=None,
                                    help_text="사람의 판단. NULL 이면 제안 그대로")
    reviewer = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.PROTECT,
                                 null=True, blank=True, related_name="reviews")
    at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-id"]
        indexes = [models.Index(fields=["box", "-id"]),
                   models.Index(fields=["reviewer", "-id"])]
