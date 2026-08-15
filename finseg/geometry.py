"""폴리곤 표기와 원본↔크롭 좌표 사상.

**이 변환식은 여기 둘뿐이다.** DB 의 폴리곤은 늘 원본 좌표이고 모델은 늘 크롭
좌표를 본다. 부르는 쪽마다 다시 쓰면 언젠가 한쪽이 반 화소씩 어긋나고, 그것은
마스크가 조금 밀린 모습으로만 나타나 눈에 안 띈다.

폴리곤은 `"x,y x,y ..."` 로 적는다. 원본 화소 좌표이고 정수다. 눈으로 읽히고
grep 되며, **정규화는 내보낼 때 한다** — DB 안에서 정규화하면 사진 크기가 함께
있어야 뜻이 통하고, 그 둘이 어긋나는 사고가 조용히 지나간다.
"""


def dumps(points) -> str:
    return " ".join(f"{int(round(x))},{int(round(y))}" for x, y in points)


def loads(s):
    if not s:
        return []
    return [tuple(int(v) for v in p.split(",")) for p in s.split()]


def to_crop(points, crop):
    s = crop.scale
    return [((x - crop.x0) * s, (y - crop.y0) * s) for x, y in points]


def to_orig(points, crop):
    s = crop.scale
    return [(x / s + crop.x0, y / s + crop.y0) for x, y in points]


def crop_rect(box, img_w, img_h, pad):
    """상자를 감싸는 정사각형을 원본 좌표로 낸다 → (x0, y0, x1, y1) 배타.

    **정사각형을 고집한다.** 640 으로 펼 때 가로세로가 다르면 지느러미가 눌리고,
    눌린 정도가 상자마다 다르면 모델이 배우는 형태가 흔들린다 — 윤곽이 곧 개체의
    단서인 자료에서 치를 이유가 없는 대가다.

    가장자리에서는 줄이지 않고 **민다**. 사진보다 큰 정사각형을 요구받을 때만
    사진 크기로 줄인다.
    """
    x1, y1, x2, y2 = box
    side = int(round(max(x2 - x1, y2 - y1) * pad))
    side = min(side, img_w, img_h)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    x0 = max(0, min(int(round(cx - side / 2)), img_w - side))
    y0 = max(0, min(int(round(cy - side / 2)), img_h - side))
    return x0, y0, x0 + side, y0 + side


def mask_to_polygon(mask, simplify=0.8):
    """이진 마스크 → 가장 큰 덩어리의 바깥 윤곽. 못 찾으면 (None, 0).

    **가장 큰 것 하나만 쓴다.** 물보라나 옆 개체가 딸려 들어온 조각을 함께 두면
    폴리곤이 두 덩어리가 되고 YOLO-seg 는 그것을 한 개체로 배운다. 지느러미는
    하나로 이어진 형태다.
    """
    import cv2
    import numpy as np
    m = mask.astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) <= 0:
        return None, 0
    if simplify:
        c = cv2.approxPolyDP(c, simplify, True)
    if len(c) < 3:
        return None, 0
    return [(float(p[0][0]), float(p[0][1])) for p in c], int(m.sum())


def rasterize(points, size):
    """폴리곤 → 크롭 크기의 이진 마스크. IoU 계산과 자르기에 쓴다."""
    import cv2
    import numpy as np
    m = np.zeros((size, size), np.uint8)
    pts = np.array([[int(round(x)), int(round(y))] for x, y in points], np.int32)
    if len(pts) >= 3:
        cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)
