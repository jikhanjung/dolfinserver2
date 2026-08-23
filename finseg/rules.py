"""**판정 규칙은 여기 하나뿐이다.**

화면에 보이는 것이 곧 학습 자료여야 한다. 규칙을 검토 UI 와 내보내기가 따로
쓰면 언젠가 갈라지고, **갈라진 것은 눈에 띄지 않는다** — 화면에서는 지운
마스크가 학습에는 들어가 있는 식이다.

## 무엇이 유효한가

- **마스크** — 상자마다 `is_current=True` 인 것 하나
- **판정** — `effective_review()` 가 고른다. 지금은 **가장 늦은 것**이고,
  한 사람이 쓰는 동안에는 그것이 곧 "그 사람의 최신" 이다

**여럿이 볼 때를 대비해 이 선택을 함수 하나로 뽑아 두었다.** 사람이 늘면
`effective_review()` 만 바꾼다 — 사람마다 최신을 고른 뒤 합의(다수결·신뢰가중)
로 가는 것이 자연스럽고, 그때 판정 표는 손대지 않아도 된다. 지금 이 자리에
"가장 늦은 것" 을 흩뿌려 두면 그 이전이 불가능해진다.

## 판정이 학습 자료에서 무엇이 되는가

| | 라벨 |
|---|---|
| `cls='none'` | **배경.** 사람이 "여기 아무것도 없다" 고 말한 자리다 |
| `cls` 가 `none` 이 아님 + `verdict='ok'` | **양성**, 그 분류로 |
| `verdict='fix'` + 교정(윗윤곽 **또는** 아래 직선) | **양성**, 교정본으로 |
| `verdict='fix'` (교정 전) | 아직 아니다 → 크롭을 뺀다 |
| 판정 없음 | 아직 아니다 → 크롭을 뺀다 |

**`none` 과 "판정 없음" 은 다르다.** 앞은 사람이 "없다" 고 말한 것이라 배경으로
쓸 수 있고, 뒤는 아무 말도 없는 것이라 배경으로 쓰면 SAM2 가 놓친 지느러미까지
"배경" 이라고 가르치게 된다.

`leading`(앞날만)·`trailing`(뒷날만)·`tip`(끄트머리)은 **버리지 않는다.** 보이는
실루엣은 틀린 것이 아니라 맞는 것이고, 가려진 개체는 검출 모델이 늘 다루는
것이다. 다만 re-ID 는 주로 뒷날의 결각으로 개체를 가리므로 그 구분을 남겨 둔다.

## `edges='neither'` 로 크롭을 버리던 것을 되돌렸다 (2026-08-16)

**`edges` 는 학습에서 아무것도 가르지 않는다.** 원래는 `neither`(앞날도 뒷날도
가림)면 그 상자가 든 크롭을 통째로 뺐다 — ultralytics 가 ignore 영역을 못 받으니
조각을 양성이라 하지도, 배경이라 하지도 않으려는 절충이었다. 두 가지가 그것을
뒤집었다.

**첫째, 버리는 대상이 하필 가장 좋은 표본이다.** 끄트머리만 보인다는 것은
**앞에 무언가가 가리고 있다**는 뜻이고, 그 앞의 것은 십중팔구 온전한
지느러미다. 크롭을 버리면 그 온전한 것이 함께 사라진다. 실측으로도 크롭의
20%에 다른 상자가 10% 이상 들어 있고 하나를 버리면 남의 크롭이 중앙값 1개
막히는데, 가림이 있는 자리에서는 이 확률이 20%가 아니라 거의 1이다. **가장 좋은
표본을 골라서 버리는 규칙이었다.**

**둘째, 조각에 라벨을 붙이는 것이 표준이다.** 인스턴스 분할 라벨은 보이는
부분만 칠한다(modal). 가려진 물체를 보이는 만큼 라벨하는 것이 COCO 이래의
관행이고, 모델은 그렇게 가림을 배운다. 위 문단이 `leading`·`trailing` 에 대해
이미 같은 말을 하고 있었다 — 끄트머리에만 다른 잣대를 댈 이유가 없었다.
정도의 차이지 종류의 차이가 아니다.

너무 작은 조각은 `export_yolo.MIN_VISIBLE` 이 따로 거른다.

**크롭을 통째로 버리는 장치는 `PENDING` 에만 남았다.** 그것은 "모른다" 라서
정말 말할 수 없는 경우고, 끄트머리는 "안다, 조각이다" 라서 말할 수 있다.
"""
from finseg import baseline, geometry
from finseg.models import Mask, Review

POSITIVE = "positive"      # 학습 자료의 양성
BACKGROUND = "background"  # 라벨 없이 배경으로
PENDING = "pending"        # 아직 사람이 말하지 않았다
DROP = "drop"              # 라벨로도 배경으로도 쓸 수 없다 — 크롭을 뺀다.
                           # **지금은 아무 규칙도 이것을 내지 않는다** (아래 문서의
                           # 2026-08-16 항목). 내보내기는 계속 받아 주므로, 그런
                           # 경우가 다시 생기면 `label_of` 에서 내기만 하면 된다


def effective_review(box):
    """이 상자에 유효한 판정 하나. 없으면 None.

    **여기가 멀티유저의 갈림길이다.** 지금은 가장 늦은 것이고, 사람이 늘면
    사람마다 최신을 고른 뒤 합의로 간다. 부르는 쪽은 바뀌지 않는다.
    """
    return box.reviews.order_by("-id").first()


def effective_mask(box):
    return box.masks.filter(is_current=True).order_by("-id").first()


def label_of(review):
    """판정 한 건이 학습 자료에서 무엇이 되는가."""
    if review is None or not review.cls:
        return PENDING
    if review.cls == "none":
        return BACKGROUND
    # **사람이 그린 윤곽이 있으면 그것이 곧 답이다.**
    #
    # `verdict` 는 "이 마스크가 맞나" 라, 마스크가 있어야 물을 수 있는 말이다 —
    # 마스크 없는 상자에는 서버가 아예 안 적는다(`review.views.save`). 그런데
    # 그 상자에 **사람이 직접 윤곽을 그리면** 모양은 생겼는데 `verdict` 가 빈
    # 채로 남아 여기서 `PENDING` 이 됐고, `export_yolo` 는 그 상자가 든 크롭을
    # 통째로 뺐다 — **그리는 일 자체가 자료에 안 닿았다.** `윤곽 없음` 대기열을
    # 다 비우고도 48개가 그 상태였다.
    #
    # 물음이 아예 다르다는 것이 요점이다. 엔진이 낸 마스크는 사람이 "맞다" 고
    # 해 줘야 자료가 되지만, **사람이 그린 것은 그 자체가 사람의 판단**이라
    # 다시 물을 것이 없다.
    if review.polygon:
        return POSITIVE
    # `edges` 는 **여기서 아무것도 가르지 않는다** — re-ID 축이지 학습 축이
    # 아니다. 끄트머리만 보이는 것도 보이는 만큼 라벨한다 (아래 문서 참고).
    if review.verdict == "ok":
        return POSITIVE
    if review.verdict == "fix":
        # 교정은 윗윤곽이든 아래 직선이든 하나면 끝난 것으로 본다. 대부분의
        # 교정은 삽입점 두 개를 옮기는 것이고 그때 윗윤곽은 SAM2 것 그대로다.
        return POSITIVE if (review.polygon or review.base_line) else PENDING
    return PENDING


def is_stuck(review):
    """**고쳐야 한다고 해 놓고 아직 안 고친 것.**

    `label_of` 는 이것을 `PENDING` 으로 낸다 — 한 번도 안 본 상자와 같은 값이다.
    자료로 나가지 않는다는 점에서는 같지만 **다시 만나는 길이 다르다**: 안 본
    것은 대기열에 나오고, 이것은 판정이 이미 붙어 있어 어디에도 안 나온다.
    그래서 따로 셀 수 있어야 한다.
    """
    return bool(review and review.verdict == "fix"
                and not review.polygon and not review.base_line
                and review.cls and review.cls != "none")


def resolve(box, mask=None, review=None):
    """상자 하나의 지금 상태. 내보내기·검토 UI·진행상황이 **전부 이것을 부른다.**

    폴리곤과 직선은 **원본 좌표 문자열** 그대로 낸다 — 크롭으로 옮기는 것은
    쓰는 쪽이 `geometry.to_crop` 으로 한다.
    """
    mask = mask if mask is not None else effective_mask(box)
    review = review if review is not None else effective_review(box)
    polygon = (review.polygon if review and review.polygon
               else (mask.polygon if mask else ""))
    base = (review.base_line if review and review.base_line
            else (mask.base_line if mask else ""))
    partial = None
    if review is not None and review.base_partial is not None:
        partial = review.base_partial
    elif mask is not None:
        partial = mask.base_partial
    return {
        "box": box,
        "mask": mask,
        "review": review,
        "label": label_of(review),
        "cls": review.cls if review else "",
        "facing": review.facing if review else "",
        "polygon": polygon,      # 자르기 전의 윗윤곽 (원본 좌표)
        "base_line": base,       # 아래 직선 두 점 (원본 좌표)
        "base_partial": partial or "",
    }


def final_points(state, crop):
    """아래 직선까지 반영한 **최종** 폴리곤 (크롭 좌표).

    자른 결과를 DB 에 담지 않고 쓸 때 만든다 — 삽입점을 정하는 규칙을 고치면
    저장된 것을 전부 다시 만들어야 하는데, 두 점만 들고 있으면 그럴 일이 없다.
    자르는 식도 `baseline` 하나뿐이라 화면과 내보내기가 어긋나지 않는다.
    """
    if not state["polygon"]:
        return []
    pts = geometry.to_crop(geometry.loads(state["polygon"]), crop)
    base = geometry.loads(state["base_line"])
    # 밑동 현은 **등지느러미에만** 뜻이 있다. 꼬리·주둥이에는 긋지 않는다.
    if len(base) != 2 or state["cls"] != "fin":
        return pts
    p0, p1 = geometry.to_crop(base, crop)
    return baseline.cut_below(pts, p0, p1, crop.w)


def progress():
    """지금 어디까지 왔나."""
    from collections import Counter

    from finseg.models import Box
    c = Counter()
    masks = {m.box_id: m for m in Mask.objects.filter(is_current=True)}
    reviews = {}
    for r in Review.objects.order_by("id"):
        reviews[r.box_id] = r          # 나중 것이 이긴다 (effective_review 와 같다)
    for box in Box.objects.all():
        c["상자"] += 1
        review = reviews.get(box.id)
        mask = masks.get(box.id)
        c[label_of(review)] += 1
        # **새 검출기가 들인 상자는 따로 센다.** 옛 상자와 묻는 것이 다르다 —
        # 저쪽은 "이 윤곽이 맞나" 고 이쪽은 "여기 정말 지느러미가 있나" 다.
        # 머리말이 이 수를 안 말하면 `검토 2,315 / 2,315` 로 다 끝난 것처럼
        # 보이는데, 그것은 엔진을 갈았을 때 겪은 것과 같은 함정이다.
        if box.source != "yolov5":
            c["새검출"] += 1
            if review is None:
                c["새검출대기"] += 1
            elif review.cls == "none":
                c["새검출:헛것"] += 1
            else:
                c["새검출:진짜"] += 1
        # **"안 본 것" 과 "고치다 만 것" 을 가른다.** 둘 다 `PENDING` 이지만
        # 앞은 대기열에 나오고 뒤는 안 나온다 — 한 칸에 뭉쳐 두면 `x` 만 누르고
        # 지나간 상자가 조용히 자료에서 빠진 채 아무 데도 안 잡힌다.
        if is_stuck(review):
            c["교정대기"] += 1
        if mask is None:
            c["마스크없음"] += 1
        # **윤곽을 말할 수 없는 상자.** 판정은 붙었는데(그래서 `검토할 것`·
        # `새 검출` 에서 빠졌고) 마스크도 사람 윤곽도 없다. `label_of` 가
        # `PENDING` 이라 내보낼 때 **그 상자가 든 크롭이 통째로 빠지고, 같은
        # 크롭에 걸친 남의 상자까지 함께 빠진다.** 대기열 다섯 어디에도 안
        # 걸려서 화면은 "남은 일 없다" 고 말하는데 자료는 조용히 준다 —
        # `교정대기` 를 따로 센 것과 같은 이유다.
        if (review is not None and review.cls and review.cls != "none"
                and mask is None and not review.polygon):
            c["윤곽없음"] += 1
        if review is not None:
            c["검토함"] += 1
            # **판정이 어느 마스크를 보고 내려졌나.** 엔진을 갈아 끼우면
            # "검토함" 은 그대로인데 그 판정들은 옛 마스크에 대한 말이 된다 —
            # 한 숫자로만 보면 다 끝난 것처럼 보이고, 정작 남은 일이 안 보인다.
            if mask is not None and review.mask_id != mask.id:
                c["엔진바뀜"] += 1
            c[f"분류:{review.cls}"] += 1
            if review.edges and review.edges != "both":
                c[f"날:{review.edges}"] += 1
    return c
