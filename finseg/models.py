"""`fin.db` 의 스키마.

**원본은 이 데이터베이스다.** 사진도 가중치도 학습 자료도 다시 만들 수 있지만
사람의 판정은 못 만든다.

## 판정이 여러 축인 이유

한 칸에 예/아니오 하나가 아니라 축이 넷이다. 각각 **아래로 흘러가는 곳이
다르기 때문**이고, 하나로 뭉치면 나중에 되살릴 수 없다.

| 축 | 값 | 어디에 쓰나 |
|---|---|---|
| `cls` | fin · fluke · rostrum · head · body · bird · pectoral · person · other · none | 학습의 분류. **`none` 만 배경이다** |
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
`person` 은 배 위나 물속의 사람이다. 수면 위로 나온 어두운 세로 형태라 옛
검출기가 자주 집는데, `none`(배경)에 두면 그 닮음을 "배경" 이라고 가르친다.

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
    # 돌고래의 몸
    ("fin", "등지느러미"),
    ("fluke", "꼬리지느러미"),
    ("rostrum", "주둥이"),
    ("head", "머리"),
    ("body", "몸통"),
    ("pectoral", "가슴지느러미"),
    # 돌고래가 아닌 것
    ("bird", "새"),
    ("person", "사람"),
    ("other", "기타"),
    # 배경
    ("none", "아무것도아님"),
]

# **화면의 키는 분류 순서와 따로 둔다.** 순서는 YOLO 클래스 번호라 한 번
# 내보내면 못 바꾸는데, 키는 손에 익는 것이라 사람 쪽 사정으로 정해야 한다.
# 둘을 붙여 두었더니 분류가 열이 되는 순간 `아무것도아님` 이 9 에서 0 으로
# 밀렸다 — 가장 자주 누르는 키가 자료 구조 사정으로 움직인 것이다.
#
# 가르는 축은 **돌고래냐 아니냐**다. 숫자는 돌고래의 몸, 글자는 그 밖의 것,
# 배경은 9. `e`·`p`·`f`·`x` 는 다른 축이 쓰므로 피한다.
CLASS_KEYS = {
    "fin": "1", "fluke": "2", "rostrum": "3",
    "head": "4", "body": "5", "pectoral": "6",
    "bird": "q", "person": "w", "other": "r",
    "none": "9",
}
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
# **학습 라벨을 만들 때 쓰는 묶음.** DB 는 분류를 잘게 들고, 내보낼 때만 묶는다 —
# 사람의 판정을 뭉개는 것이 아니라 그 판정에서 라벨을 만드는 방식을 정하는 것이다.
#
# 잘게 둔 채로 내보내면 가슴지느러미 3장짜리 클래스가 생긴다. 그것은 배우는 게
# 아니라 잡음이고 클래스별 성적도 뜻을 잃는다. `models.py` 첫 문단의 논지는
# **"어려운 음성을 배경에 두지 말라"** 였지 "잘게 쪼개라" 가 아니었으므로,
# 묶어도 그 값은 그대로 남는다 — 묶인 것도 여전히 **라벨이 붙은 양성**이다.
#
# `none` 은 어느 묶음에도 없다. 그것만 배경이다.
CLASS_GROUPS = {
    "all": [(c, ko, [c]) for c, ko in CLASSES if c != "none"],
    "coarse": [
        ("fin", "등지느러미", ["fin"]),
        ("dolphin", "돌고래(등지느러미 아님)",
         ["fluke", "rostrum", "head", "body", "pectoral"]),
        ("nonfin", "돌고래 아님", ["bird", "person", "other"]),
    ],
}
VERDICTS = [("ok", "통과"), ("fix", "교정 필요")]
RUN_KINDS = [("import", "들이기"), ("crop", "크롭"), ("sam2", "SAM2.1"),
             ("yolo", "YOLO"), ("detect", "검출"), ("export", "내보내기"),
             ("train", "학습")]


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
    """분할기에 넣을 프롬프트 상자. **이제 낸 것이 둘이다.**

    처음에는 옛 YOLOv5 가 낸 것뿐이었고 `source` 는 그것을 적어 두기만 했다
    (`src_id` 로 옛 DB 를 되짚을 수 있으니). 2026-08-18 에 우리 검출기가 낸 상자를
    들이면서 두 칸이 필요해졌다.

    **`run`** — 어느 실행이 이 상자를 냈나. `Mask.run` 과 같은 이유다
    (`Run` 문서: "어떤 코드가 만든 자료인지 나중에 반드시 묻게 된다"). 옛 상자는
    비어 있다 — 그것을 낸 실행은 이 저장소 밖에서 돌았다.

    **`conf`** — 검출기가 말한 확신. **이것이 없으면 문턱을 못 고른다.** 새
    검출기의 정밀도는 문턱에 따라 64%에서 84%까지 움직이는데, 사람이 검토한 뒤
    "conf 0.4 위에서는 몇 %가 진짜였나" 를 물으려면 상자마다 그 값이 남아 있어야
    한다. 안 남기면 추론을 처음부터 다시 돌리는 수밖에 없다.
    """
    src_id = models.IntegerField(unique=True, null=True,
                                 help_text="옛 dolfinrest_dolfinbox.id")
    image = models.ForeignKey(Image, on_delete=models.CASCADE, related_name="boxes")
    x1 = models.IntegerField()
    y1 = models.IntegerField()
    x2 = models.IntegerField()
    y2 = models.IntegerField()
    area = models.IntegerField(null=True)
    source = models.CharField(max_length=30, default="yolov5", db_index=True)
    conf = models.FloatField(null=True, blank=True,
                             help_text="검출기가 말한 확신. 옛 상자는 비어 있다")
    # `Run` 은 아래에 있다 — 문자열로 건다
    run = models.ForeignKey("Run", on_delete=models.SET_NULL, null=True, blank=True,
                            related_name="boxes", help_text="이 상자를 낸 실행")
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
    나란히 비교할 수 있어야 한다.

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
    # **엔진이 무엇이라 했나.** 분할 모델은 `coarse` 세 갈래(`fin`·`dolphin`·
    # `nonfin`)를 내는데 그동안 그 판정을 버리고 폴리곤만 담았다.
    #
    # 버리면 안 되는 이유는 **자동 경로에서 그것이 유일한 분류**이기 때문이다.
    # 검출기는 클래스가 하나라 주둥이·몸통·사람을 다 "지느러미 같은 것" 으로
    # 잡아 오고, 사람이 안 본 사진에서는 그것을 가릴 다른 수가 없다.
    #
    # 실측(사람이 분류한 상자 2,799개): `fin` 98.5% · `nonfin` 97.1% ·
    # `dolphin` 81.5%. 딴 부위를 지느러미로 잘못 부르는 것이 1.2% 다.
    #
    # **사람의 판정을 대신하지 않는다** — `rules.resolve` 는 여전히 `Review`
    # 를 본다. 이것은 아무도 안 본 상자에서만 쓸 값이다.
    cls = models.CharField(max_length=10, blank=True, default="",
                           help_text="엔진이 낸 묶음 분류 (fin·dolphin·nonfin)")
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


class Individual(models.Model):
    """개체 하나. **이름은 사람이 붙인다.**

    옛 DB 의 `boxname` 과 **섞지 않는다.** 그것은 시민과학자가 붙인 것이고
    166개 중에 `까치`·`황여새` 같은 새 이름이 섞여 있어 그대로 믿을 수 없다.
    `Box.source` 로 어느 검출기가 낸 상자인지 갈라 온 것과 같은 이유다 —
    **누가 말한 것인지 모르면 나중에 그것으로 무엇을 잴 수 없다.**

    좌현·우현을 한 개체 아래 함께 둔다. 같은 동물이니 그것이 옳고, 다만
    **닮음을 잴 때는 절대 섞지 않는다** (`reid.frame` 의 부호 주석).
    """
    # **개체를 가리키는 것은 이것 하나다** — `JTA001` 같은 카탈로그 번호.
    # 판정도 성적도 분류기도 전부 `id` 로 묶으므로 이 값을 고쳐도 자료가
    # 안 흔들린다 (`catalog()` · `reid_cls` · `Identification`).
    name = models.CharField(max_length=50, unique=True, help_text="카탈로그 번호")
    # **별명은 있을 수도 없을 수도 있다** (`제돌이`·`춘삼이`). 번호와 한 칸에
    # 섞지 않는 이유는 — `JTA001 (제돌이)` 로 두면 정렬도 중복 검사도 옛
    # 카탈로그와 맞춰 보는 일도 전부 그 괄호를 파싱하게 되고, 별명이 붙거나
    # 바뀔 때마다 **번호가 든 문자열을 건드리게 된다.**
    #
    # **UNIQUE 를 안 건다.** 겹치는 별명은 사람이 알아보는 문제이지 자료가
    # 깨지는 문제가 아니다 — 개체를 가리키는 것은 번호다.
    nickname = models.CharField(max_length=50, blank=True, default="",
                                help_text="별명 — 있는 개체만")
    note = models.TextField(blank=True, default="")
    # 옛 DB 의 이름과 이어졌다면 적어 둔다. 믿어서가 아니라 **되짚으려고** 다
    src_name = models.CharField(max_length=50, blank=True, default="",
                                help_text="옛 dolfinrest_dolfinbox.boxname")
    # **대표 사진.** 상자에 든 것을 다 늘어놓으면 목록이 금세 길어져 상자
    # 사이를 오가기가 어려워진다. 그렇다고 아무거나 하나를 보이면 하필 물에
    # 잠겼거나 흐린 것이 대표가 되어 **그 상자가 누구인지 알아볼 수 없다** —
    # 그래서 사람이 고른다. 안 고르면 든 것 중 첫 번째를 쓴다.
    rep = models.ForeignKey("Box", on_delete=models.SET_NULL, null=True,
                            blank=True, related_name="represents",
                            help_text="이 개체를 대표하는 상자(사진)")
    # **개체가 아닌 자리들.** 분류하다 보면 "이 개체다" 라고 못 하는 것이
    # 반드시 나오는데, 아무 상자에나 넣으면 그 상자가 오염되고 미분류로 두면
    # 다음에 또 같은 고민을 처음부터 한다. 그래서 **개체가 아닌 자리를 갈래로
    # 둔다** — `catalog()` 가 여기 든 것을 안 센다.
    #
    # `hold`   특징은 뚜렷한데 지금은 어느 상자인지 모르겠는 것
    # `notfin` **지느러미가 아닌 것.** re-ID 자리는 `Review` 를 못 쓰므로
    #          (주인이 작업 자리 하나다) 여기에 넣어 두면, 되받은 뒤 작업
    #          자리에서 **사람이 진짜 분류를 골라 `Review` 로 옮겨 적는다**
    #          (검토 화면의 `저쪽에서 아니라 한 것` 대기열)
    KINDS = [("", "개체"), ("hold", "임시보관함"), ("notfin", "지느러미 아님")]
    kind = models.CharField(max_length=10, choices=KINDS, blank=True, default="",
                            help_text="비어 있으면 진짜 개체다")
    at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["name"]
        constraints = [
            # **갈래마다 하나뿐이다.** 둘이 되면 어느 쪽에 넣었는지에 따라
            # 결과가 갈리는데, 그 사실은 눈에 안 띈다.
            models.UniqueConstraint(fields=["kind"], condition=~models.Q(kind=""),
                                    name="one_box_per_kind"),
        ]

    def __str__(self):
        return self.name


class Identification(models.Model):
    """"이 상자는 이 개체다" — **쌓인다, 덮어쓰지 않는다.**

    `Review` 와 같은 규칙이다. 개체 판정은 되짚어 고치는 일이 잦고(카탈로그가
    자라면서 갈라지거나 합쳐진다), 그 자취가 없으면 언제 생각이 바뀌었는지
    잴 수 없다. 유효한 것은 **가장 늦은 것**이다.

    `individual` 이 비면 **"이 묶음에서 뺐다"** 는 뜻이다. 아니라고 말한 것도
    자료다 — 그것이 어려운 음성이고, 다음 바퀴에서 같은 짝을 또 보여 주지
    않게 해 준다.
    """
    box = models.ForeignKey(Box, on_delete=models.CASCADE,
                            related_name="identifications")
    individual = models.ForeignKey(Individual, on_delete=models.CASCADE,
                                   null=True, blank=True,
                                   related_name="identifications",
                                   help_text="NULL 이면 '이 개체가 아니다'")
    # **무엇을 보고 정했나.** 군집이 보여 준 묶음에서 왔는지, 사람이 직접
    # 찾은 것인지. 나중에 "그 군집이 얼마나 맞았나" 를 이것으로 잰다
    source = models.CharField(max_length=20, default="human",
                              help_text="human · cluster:<run> …")
    reviewer = models.ForeignKey(settings.AUTH_USER_MODEL,
                                 on_delete=models.PROTECT, null=True, blank=True,
                                 related_name="identifications")
    at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-id"]
        indexes = [models.Index(fields=["box", "-id"]),
                   models.Index(fields=["individual", "-id"])]
