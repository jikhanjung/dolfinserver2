"""ONNX 검출기의 **전처리와 NMS — 이 규칙의 원본이 여기다.**

`.onnx` 하나를 세 곳이 쓴다: 이 모듈(파이썬), 브라우저(`/detect` 의 JS),
그리고 나중에 데스크톱 뷰어. **셋이 같은 상자를 내야 한다.** 여기가 기준이고,
다른 둘은 이것을 옮긴 것이다 — 어긋나면 여기를 고치고 나머지를 따라오게 한다.

## 왜 따로 두나 — 어긋나도 에러가 안 난다

전처리가 학습 때와 다르면 **예외 없이 성적만 나빠진다.** 색 순서를 뒤집어도,
레터박스 대신 늘려도, 여백을 잘못 빼도 상자는 나온다 — 조금 틀린 자리에.
이 저장소가 좌표 사상을 `geometry.py` 한 곳에만 두는 이유와 같은 문제다
(`devlog/…0816_002`).

그래서 **`ultralytics` 와 대조하는 시험**을 함께 둔다(`review/tests.py`).
원본과 견주지 않으면 어긋난 것을 알 방법이 없다.

## 맞춰야 하는 것

| | |
|---|---|
| 크기 | **레터박스** — 비율을 지키고 남는 데를 회색(114)으로 채운다 |
| 줄이는 법 | **필터를 넓히지 않는 2×2 표본** — 아래 문단 |
| 색 | **RGB** (OpenCV 로 읽으면 BGR 이라 뒤집어야 한다) |
| 값 | `/255` → 0~1 |
| 축 | NCHW (1, 3, H, W) · float32 |

## 줄일 때 부드럽게 하면 안 된다 — 실측

`PIL.Image.resize(BILINEAR)` 은 **축소할 때 필터를 넓혀 안티에일리어싱**을 한다.
ultralytics 는 `cv2.resize(INTER_LINEAR)` 를 쓰는데 그것은 안 한다. 4928 →
1280 은 3.85배 축소라 이 차이가 크다. 같은 사진 한 장(DSC_7067)에서:

| 줄이는 법 | 큰 상자 | 큰 상자 | **작은 상자 (29×21px)** |
|---|---|---|---|
| `PIL.resize` BILINEAR | 0.8851 | 0.8460 | **0.1651** |
| `cv2` INTER_LINEAR | 0.8923 | 0.8537 | **0.3612** |
| `PIL.transform` AFFINE+BILINEAR | 0.8923 | 0.8537 | **0.3691** |
| `PIL.resize` NEAREST | 0.8925 | 0.8553 | 0.3393 |
| (원본 `.pt`) | 0.8912 | 0.8524 | 0.3442 |

**큰 상자는 0.007 차이인데 작은 상자는 확신이 절반이 된다.** 문턱 0.25 에서
그 상자가 사라졌다. 부드럽게 만든 것이 **학습 때 본 적 없는 그림**이라서다.

이 프로젝트에서 특히 나쁘다 — 새 검출기가 여는 것의 3분의 2가 **끄트머리만
보이는 작은 지느러미**다(`tip` 67.1%). 바로 그것들이 먼저 사라진다.

그래서 `PIL.transform` 을 쓴다. `cv2` 와 같은 2×2 표본이고 **의존성이 안 는다**
— 뒷날 데스크톱 뷰어에서 opencv 를 안 넣어도 된다.

**브라우저도 같은 함정이 있다.** canvas 의 `drawImage` 는 기본으로 부드럽게
줄인다. `ctx.imageSmoothingEnabled = false` 로 꺼야 하고, 그러면 위 표의
NEAREST(0.3393) 자리라 쓸 만하다.

## 모델이 내는 것

`(1, 4+nc, N)` 이다 — 우리는 한 분류라 `(1, 5, 8400)`. 앞 넷이
`cx, cy, w, h`(**레터박스 안의 화소 좌표**), 다섯째가 확신이다.
**NMS 는 안 들어 있다** — 일부러 뺐다. 정렬하고 반복문으로 지우는 일이라
GPU 셰이더에 안 맞아서, 브라우저에서 그 연산자만 CPU 로 떨어지면 이득이
깎인다. 밖에 두면 **문턱을 다시 안 돌리고 바꿔 볼 수 있다**는 값도 따라온다.
"""
import numpy as np

# 학습 때 ultralytics 가 쓰는 값과 같아야 한다
PAD = 114
IMGSZ = 1280
CONF = 0.25
IOU = 0.7


def letterbox(w, h, size=IMGSZ):
    """원본 크기 → (배율, 왼쪽 여백, 위 여백, 새 너비, 새 높이).

    **그림을 안 건드리고 숫자만 낸다** — 파이썬은 PIL, 브라우저는 canvas 로
    각자 그리되 이 숫자는 같아야 한다.
    """
    r = min(size / w, size / h)
    nw, nh = round(w * r), round(h * r)
    return r, (size - nw) // 2, (size - nh) // 2, nw, nh


def preprocess(img, size=IMGSZ):
    """PIL 이미지 → (입력 텐서, 되돌리기에 쓸 값들).

    `img` 는 RGB 여야 한다. 여기서 한 번 더 `convert("RGB")` 하는 것은
    회색조·팔레트 사진이 섞여 들어와도 축이 셋이 되게 하려는 것이다.
    """
    from PIL import Image

    img = img.convert("RGB")
    w, h = img.size
    r, dx, dy, nw, nh = letterbox(w, h, size)
    canvas = Image.new("RGB", (size, size), (PAD, PAD, PAD))
    # **`resize` 가 아니라 `transform` 이다.** `resize` 는 축소할 때 필터를
    # 넓혀 부드럽게 만드는데, 학습은 그런 그림을 본 적이 없다 — 작은
    # 지느러미의 확신이 절반이 된다(도입부 실측). `transform` 은 2×2 표본만
    # 써서 `cv2.INTER_LINEAR` 과 같다.
    small = img.transform((nw, nh), Image.AFFINE,
                          (w / nw, 0, 0, 0, h / nh, 0), Image.BILINEAR)
    canvas.paste(small, (dx, dy))
    x = np.asarray(canvas, dtype=np.float32) / 255.0      # HWC, 0~1
    x = x.transpose(2, 0, 1)[None]                        # NCHW
    return np.ascontiguousarray(x), (r, dx, dy, w, h)


def nms(boxes, scores, iou_thres=IOU):
    """겹치는 상자 중 확신 높은 것만 남긴다. 남길 것의 번호를 낸다.

    **자기 출력끼리** 견주는 자리다 — `infer_boxes` 의 `--iou` 는 옛 DB 의
    상자와 견주는 것이라 이름만 같고 하는 일이 다르다.
    """
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        union = area[i] + area[rest] - inter
        # 넓이가 0인 상자가 섞여도 0으로 나누지 않는다
        iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        order = rest[iou <= iou_thres]
    return keep


def postprocess(out, meta, conf_thres=CONF, iou_thres=IOU, nc=1):
    """모델 출력 → **원본 사진 좌표**의 상자 목록 `[(x1,y1,x2,y2,conf), …]`.

    `nc` 는 분류 수다 — 이 검출기는 하나(`fin`)라 칸이 `4+1=5` 개다.

    `meta` 는 `preprocess` 가 낸 것이다. **여백을 빼고 배율로 나눈다** —
    이 두 줄이 어긋나면 상자가 조금씩 밀린 채로 나오고, 그것은 에러가 아니라
    그냥 나쁜 결과로만 보인다.
    """
    r, dx, dy, w, h = meta
    p = np.asarray(out)
    if p.ndim == 3:
        p = p[0]
    # **축이 어느 쪽인지는 칸 수로 가린다.** 내보내기 판에 따라 `(4+nc, N)`
    # 이기도 `(N, 4+nc)` 이기도 한데, "짧은 쪽이 칸" 이라고 어림하면 상자가
    # 딱 4+nc 개일 때 뒤집힌다 — 시험에서 그렇게 걸렸다. 우리는 한 분류라
    # 칸이 다섯이라는 것을 알고 있으니 그것으로 판단한다.
    nf = 4 + nc
    if p.shape[-1] != nf:
        if p.shape[0] != nf:
            raise ValueError(f"모르는 출력 모양이다: {p.shape} (칸 {nf} 개를 기대)")
        p = p.T
    scores = p[:, 4]
    m = scores >= conf_thres
    p, scores = p[m], scores[m]
    if not len(p):
        return []
    cx, cy, bw, bh = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
    keep = nms(boxes, scores, iou_thres)
    out_boxes = []
    for i in keep:
        x1, y1, x2, y2 = boxes[i]
        # 레터박스를 되돌린다 — 여백 먼저, 배율은 그다음
        x1 = (x1 - dx) / r
        y1 = (y1 - dy) / r
        x2 = (x2 - dx) / r
        y2 = (y2 - dy) / r
        out_boxes.append((
            float(np.clip(x1, 0, w)), float(np.clip(y1, 0, h)),
            float(np.clip(x2, 0, w)), float(np.clip(y2, 0, h)),
            float(scores[i])))
    return out_boxes


def detect(session, img, conf_thres=CONF, iou_thres=IOU, size=IMGSZ):
    """onnxruntime 세션 하나로 사진 한 장. 편의 함수다."""
    x, meta = preprocess(img, size)
    name = session.get_inputs()[0].name
    out = session.run(None, {name: x})[0]
    return postprocess(out, meta, conf_thres, iou_thres)


# ---- 분할 (2단) --------------------------------------------------------------
#
# **검출과 분할은 두 단이다.** 원본을 640으로 줄이면 지느러미가 16px 이 되고
# notch 는 그 그림에 아예 없다 — `CLAUDE.md` 가 "한 모델로 합치기" 를 안 하기로
# 한 이유다. 그래서 1280으로 상자를 찾고, 상자마다 크롭을 떠서 640으로 분할한다.
#
# 크롭 규칙은 `geometry.crop_rect` 하나뿐이다. 여기서 다시 쓰지 않는다 —
# 학습 자료를 만든 식과 어긋나면 모델이 본 적 없는 틀이 들어간다.

SEG_IMGSZ = 640
SEG_NC = 3          # fin · dolphin · nonfin (`models.CLASS_GROUPS['coarse']`)
CROP_PAD = 2.0      # 상자 긴 변의 배수 (`crops --pad`)
MASK_THRES = 0.5


def seg_masks(out0, out1, box_in_crop, conf_thres=CONF, iou_thres=IOU,
              size=SEG_IMGSZ, nc=SEG_NC):
    """YOLO-seg 출력 → (마스크, 상자, 확신, 분류) 하나. 없으면 None.

    `out0` 은 `(1, 4+nc+32, N)`, `out1` 은 `(1, 32, 160, 160)` 이다.
    마스크는 **크롭 좌표의 `size`×`size` 불리언 배열**로 낸다.

    ## 여럿 나오면 프롬프트 상자와 가장 많이 겹치는 것

    크롭은 "가운데 것 하나" 라는 약속으로 만들었지만 이웃 지느러미가 걸쳐
    들어오는 크롭이 20%쯤 된다. 그때 **가운데에서 가장 가까운 것**이 아니라
    **상자와 가장 많이 겹치는 것**이 옳다 — `infer`·`infer_base` 와 같은 규칙이다.
    """
    p = np.asarray(out0)
    if p.ndim == 3:
        p = p[0]
    nf = 4 + nc + 32
    if p.shape[-1] != nf:
        if p.shape[0] != nf:
            raise ValueError(f"모르는 출력 모양이다: {p.shape} (칸 {nf} 개를 기대)")
        p = p.T
    cls_scores = p[:, 4:4 + nc]
    scores = cls_scores.max(1)
    cls = cls_scores.argmax(1)
    m = scores >= conf_thres
    p, scores, cls = p[m], scores[m], cls[m]
    if not len(p):
        return None
    cx, cy, bw, bh = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
    keep = nms(boxes, scores, iou_thres)
    if not keep:
        return None
    # 프롬프트 상자와 겹치는 넓이가 가장 큰 것
    bx1, by1, bx2, by2 = box_in_crop
    best, best_ov = None, -1.0
    for i in keep:
        x1, y1, x2, y2 = boxes[i]
        ov = (max(0.0, min(x2, bx2) - max(x1, bx1))
              * max(0.0, min(y2, by2) - max(y1, by1)))
        if ov > best_ov:
            best, best_ov = i, ov
    proto = np.asarray(out1)[0]                 # (32, 160, 160)
    coef = p[best, 4 + nc:]                     # (32,)
    k, mh, mw = proto.shape
    z = (coef @ proto.reshape(k, -1)).reshape(mh, mw)
    mask = 1.0 / (1.0 + np.exp(-z))             # 시그모이드
    # 160 → size. **상자 밖은 지운다** — ultralytics 와 같은 규칙이다
    from PIL import Image
    big = np.asarray(Image.fromarray((mask * 255).astype(np.uint8))
                     .resize((size, size), Image.BILINEAR)) / 255.0
    x1, y1, x2, y2 = boxes[best]
    out = np.zeros((size, size), bool)
    xs, ys = slice(max(0, int(x1)), int(np.ceil(x2))), \
        slice(max(0, int(y1)), int(np.ceil(y2)))
    out[ys, xs] = big[ys, xs] > MASK_THRES
    return out, boxes[best], float(scores[best]), int(cls[best])
