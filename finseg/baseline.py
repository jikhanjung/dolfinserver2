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
