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
| `cls∈{fin,fluke,rostrum,other}` + `verdict='ok'` | **양성**, 그 분류로 |
| `verdict='fix'` + 교정(윗윤곽 **또는** 아래 직선) | **양성**, 교정본으로 |
| `verdict='fix'` (교정 전) | 아직 아니다 → 크롭을 뺀다 |
| **`edges='neither'`** | **뺀다.** 라벨로도 배경으로도 쓸 수 없다 |
| 판정 없음 | 아직 아니다 → 크롭을 뺀다 |

**`none` 과 "판정 없음" 은 다르다.** 앞은 사람이 "없다" 고 말한 것이라 배경으로
쓸 수 있고, 뒤는 아무 말도 없는 것이라 배경으로 쓰면 SAM2 가 놓친 지느러미까지
"배경" 이라고 가르치게 된다.

**`edges='neither'` 는 `none` 이 아니다.** 지느러미가 있기는 한데 앞날도 뒷날도
가려 라벨을 붙일 수 없는 것이다. 배경이라 하면 지느러미를 배경이라 가르치는
것이고, 보이는 조각을 양성이라 하면 조각을 지느러미라 가르치는 것이다. 객체
검출에서 이럴 때 쓰는 ignore 영역을 ultralytics 는 받지 않으므로, **그 상자가
든 크롭을 통째로 뺀다** — 검토 안 된 상자에 쓰는 것과 같은 장치다.

`leading`(앞날만)·`trailing`(뒷날만)은 **버리지 않는다.** 보이는 실루엣은 틀린
것이 아니라 맞는 것이고, 가려진 개체는 검출 모델이 늘 다루는 것이다. 다만
re-ID 는 주로 뒷날의 결각으로 개체를 가리므로 그 구분을 남겨 둔다.
"""
from finseg import baseline, geometry
from finseg.models import Mask, Review

POSITIVE = "positive"      # 학습 자료의 양성
BACKGROUND = "background"  # 라벨 없이 배경으로
PENDING = "pending"        # 아직 사람이 말하지 않았다
DROP = "drop"              # 라벨로도 배경으로도 쓸 수 없다 — 크롭을 뺀다


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
    if review.edges == "neither":
        return DROP
    if review.verdict == "ok":
        return POSITIVE
    if review.verdict == "fix":
        # 교정은 윗윤곽이든 아래 직선이든 하나면 끝난 것으로 본다. 대부분의
        # 교정은 삽입점 두 개를 옮기는 것이고 그때 윗윤곽은 SAM2 것 그대로다.
        return POSITIVE if (review.polygon or review.base_line) else PENDING
    return PENDING


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
        "polygon": polygon,      # 자르기 전의 윗윤곽 (원본 좌표)
        "base_line": base,       # 아래 직선 두 점 (원본 좌표)
        "base_partial": bool(partial),
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
        if mask is None:
            c["마스크없음"] += 1
        if review is not None:
            c["검토함"] += 1
            c[f"분류:{review.cls}"] += 1
            if review.edges and review.edges != "both":
                c[f"날:{review.edges}"] += 1
    return c
