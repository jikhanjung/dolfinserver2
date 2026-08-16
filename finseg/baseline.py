"""지느러미의 아래 경계 — 앞뒤 삽입점을 잇는 직선.

## 규칙

**위 윤곽은 실제 경계를 따라간다. 아래는 앞·뒤 삽입점(지느러미가 등과 만나는
자리)을 잇는 직선으로 자른다.** 돌고래 photo-ID 에서 쓰는 fin base chord 와 같고,
점 두 개면 결정되므로 사람이 고칠 때도 **끄는 것이 두 개뿐**이다.

수면선에서 자르는 방법도 있지만 그것은 같은 개체라도 자세에 따라 아래 끝이
달라진다. 개체 식별이 목적이라면 해부학적 자리가 맞다.

이 규칙은 **등지느러미에만** 적용한다. 꼬리지느러미·주둥이에는 뜻이 없다.

## 자동으로 찾으려다 실패한 것 — 다시 시도하지 말 것

마스크의 위 옆모습(열마다 가장 위 화소)을 뽑아 **기울기가 완만(등)에서
가파름(날)으로 바뀌는 자리**를 삽입점으로 잡으려 했다. 40장에 대 보니
**양쪽 다 찾은 것이 2장뿐**이었다. 실측한 옆모습이 이유를 말해 준다.

    타일 3   y : 457 443 429 │ 210 201 205 215 229 248 272 299 331 358 431
          기울기: 0.0 -0.2 -0.8│-41.0  0.2  0.3  0.7  1.0  1.0  1.2  0.7  1.1  2.0
                 └─ 등(완만) ┘└앞날(수직)┘└─── 뒷날 ~ 등: 구분되지 않는다 ───┘

- **앞날은 거의 수직**이라 쉽게 잡힌다 (기울기 −41)
- **뒷날은 기울기가 0.5~2.2 로 등과 같은 크기다.** 뒷삽입점은 기울기가 아니라
  완만한 **곡률** 변화이고, 문턱으로는 잡히지 않는다
- 개체에 따라서는 지느러미와 등이 위 윤곽에서 하나의 매끈한 경사로 이어져
  전환점 자체가 없다

사람 눈에는 보이지만 마스크 기하만으로는 안 된다. **그래서 사람이 찍는다.**

## 지금 하는 것

1. **상자 아래 두 모서리**를 첫 제안으로 준다. 옛 YOLOv5 상자의 아래 변이
   지느러미 밑동 언저리라 출발점으로 쓸 만하고, 무엇보다 **어느 쪽으로도
   편향되지 않는다** — 틀린 자동 제안은 사람의 눈을 끌고 가지만 대충의 제안은
   그러지 않는다
2. 사람이 두 점을 끈다
3. 300장쯤 쌓이면 **키포인트 모델로 제안한다.** 640 크롭에 점 두 개를 얹는
   회귀는 쉬운 문제이고 ultralytics 가 그대로 지원한다. 그 뒤로는 사람이
   어긋난 것만 고친다
"""
import numpy as np

# 검토 화면에 그대로 띄우는 문구. **규칙과 화면을 한 곳에 둔다** — 화면이 규칙을
# 말하지 않으면 사람이 스스로와 어긋나고, 화면에만 적어 두면 규칙이 여기서
# 바뀔 때 화면이 따라오지 않는다.
RULE = ("아래 경계는 **앞·뒤 삽입점을 잇는 직선**(밑동 현)이다. "
        "위 윤곽은 실제 경계를 따라간다.")

RULE_POINTS = [
    "삽입점은 지느러미가 **등과 만나는 자리**다 — 수면선이 아니다. "
    "수면은 같은 개체라도 자세에 따라 달라진다",
    "물에 잠겨 삽입점이 안 보이면 **짐작해서 긋고** p 로 불완전을 표시한다. "
    "그 표시가 나중에 형태 계측에서 가릴 수 있는 유일한 단서다",
    "밑동 현은 **등지느러미에만** 긋는다. 꼬리·주둥이·기타에는 뜻이 없다",
    "첫 제안은 상자 아래 두 모서리라 **대충이다.** 맞다고 여기지 말 것 — "
    "자동 탐지는 두 번 실패했고 그 기록이 finseg/baseline.py 에 있다",
    "두 점을 옮기면 그 자체로 **교정함(fix)** 이 된다",
]


def upper_profile(mask):
    """열마다 가장 위 화소 → (xs, ys). 마스크가 없는 열은 뺀다.

    윗윤곽만 보는 계산에 쓴다. SAM2 가 흔들리는 곳은 아래쪽(수면·물보라)뿐이라
    위 옆모습은 믿을 수 있다.
    """
    cols = np.where(mask.any(axis=0))[0]
    if len(cols) == 0:
        return np.array([]), np.array([])
    ys = mask[:, cols].argmax(axis=0)
    return cols, ys.astype(float)


def propose_from_box(box, mask=None):
    """첫 제안 — 상자 아래 두 모서리 → (왼쪽 점, 오른쪽 점).

    `box` 는 크롭 좌표의 (x1, y1, x2, y2). 마스크를 주면 그 아래 끝을 넘지
    않도록만 다듬는다 (상자가 마스크보다 아래로 처져 있는 경우).
    """
    x1, y1, x2, y2 = box
    y = float(y2)
    if mask is not None and mask.any():
        rows = np.where(mask.any(axis=1))[0]
        y = min(y, float(rows[-1]))
    return (float(x1), y), (float(x2), y)


def cut_below(polygon, p0, p1, size):
    """직선 아래를 잘라 낸 폴리곤. 직선이 없으면 그대로 돌려준다.

    래스터로 자르고 윤곽을 다시 딴다 — 마스크를 만들 때와 **같은 방식**이라
    결과가 어긋나지 않는다.
    """
    import cv2
    if p0 is None or p1 is None:
        return polygon
    pts = np.array([[int(round(x)), int(round(y))] for x, y in polygon], np.int32)
    if len(pts) < 3:
        return polygon
    m = np.zeros((size, size), np.uint8)
    cv2.fillPoly(m, [pts], 1)
    yy, xx = np.mgrid[0:size, 0:size]
    (x0, y0), (x1, y1) = p0, p1
    if abs(x1 - x0) < 1e-6:
        keep = xx <= max(x0, x1)
    else:
        keep = yy <= y0 + (xx - x0) * ((y1 - y0) / (x1 - x0))
    m = m & keep.astype(np.uint8)
    if not m.any():
        return polygon
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return polygon
    c = max(cnts, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, 0.8, True)
    if len(c) < 3:
        return polygon
    return [(float(p[0][0]), float(p[0][1])) for p in c]


def chord(p0, p1):
    """밑동 현의 길이와 기울기 — 형태 계측의 기본이다 (re-ID 에서 쓴다)."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    return float(np.hypot(dx, dy)), float(np.degrees(np.arctan2(dy, dx)))


def height_above(polygon, p0, p1):
    """현에서 가장 먼 점까지의 거리 = 지느러미 높이.

    현이 기울어 있어도 뜻이 같도록 **현에 수직인 거리**로 잰다.
    """
    if not polygon or p0 is None or p1 is None:
        return 0.0
    a = np.array(polygon, float)
    d = np.array([p1[0] - p0[0], p1[1] - p0[1]], float)
    n = np.hypot(*d)
    if n < 1e-6:
        return 0.0
    d /= n
    rel = a - np.array(p0, float)
    perp = np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0])
    return float(perp.max())


# **꼭대기를 기준으로 넓이가 큰 쪽이 앞이다.** 지느러미가 뒤로 젖혀져 있으니
# 꼭대기 앞쪽에 살이 더 많다. 사람이 찍은 156장에 대 보니 넓이비 문턱 1.3 에서
# **149장(95%)에 적용해 정확도 100%** 였다 (문턱 없이는 98.7%).
#
# ## 먼저 세웠다 틀린 규칙 셋 — 다시 시도하지 말 것
#
# 위 문단의 "앞날은 거의 수직(기울기 −41)" 을 근거로 **가파른 쪽이 앞**이라고
# 예측했는데 실측 **26.9%** 로 거의 정반대였다. 그 −41 은 타일 하나의 예시였고
# 일반화되지 않는다. "꼭대기에서 가까운 쪽이 앞" 은 17.3%(뒤집으면 82.7%),
# "끝 기울기가 가파른 쪽이 앞" 은 73.1% 로 둘 다 모자랐다.
#
# **세 예측이 모두 빗나갔고 값이 나온 것은 안 세웠던 넷째다.** 재 보기 전에
# 제안으로 돌렸으면 넷에 하나가 틀린 채로 2,315장에 얹혔을 것이다.
FACING_MIN_RATIO = 1.3


def propose_facing(points, size, min_ratio=FACING_MIN_RATIO):
    """윗윤곽으로 앞쪽을 짐작한다 → ("left" | "right" | "", 넓이비).

    **애매하면 아무 말도 안 한다.** 넓이비가 문턱 아래면 빈 값을 돌려준다 —
    그럴듯한 오답은 사람의 눈을 끌고 가고, 그것이 수면선 제안을 버린 이유다.
    """
    if not points or len(points) < 3:
        return "", 0.0
    m = _np_rasterize(points, size)
    cols = np.where(m.any(axis=0))[0]
    if len(cols) < 8:
        return "", 0.0
    ys = m[:, cols].argmax(axis=0)
    apex = cols[int(np.argmin(ys))]
    left = float(m[:, cols[cols <= apex]].sum())
    right = float(m[:, cols[cols >= apex]].sum())
    ratio = max(left, right) / max(min(left, right), 1.0)
    if ratio < min_ratio:
        return "", ratio
    return ("left" if left > right else "right"), ratio


def _np_rasterize(points, size):
    from finseg import geometry
    return geometry.rasterize(points, size)
