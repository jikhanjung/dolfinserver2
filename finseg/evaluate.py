"""엔진을 사람의 판정에 비교하는 계산 — **여기 하나뿐이다.**

명령(`eval_masks`)과 화면(`/compare`)이 같은 숫자를 말해야 한다. 둘로 갈라
두면 표에서는 0.89 인데 화면에서는 0.91 인 날이 오고, **갈라진 것은 눈에 띄지
않는다** (`rules.py` 를 한 곳에 둔 것과 같은 이유).

**정답은 사람이 채택한 마스크**이고, 아래 직선까지 반영한 최종 폴리곤이다 —
화면에서 본 것이 곧 정답이어야 한다.

## 못 낸 것은 IoU 0 이다

엔진이 아무것도 안 낸 상자를 **빼면 안 된다.** 빼고 평균을 내면 "잘 낸 것만
골라 잰" 숫자가 되어 재현율 손실이 성적으로 둔갑한다.
"""
from finseg import geometry, rules


def truth_for(boxes, crops):
    """상자 목록 → {상자번호: 최종 폴리곤(크롭 좌표)}. 양성만 담는다."""
    out = {}
    for box in boxes:
        st = rules.resolve(box)
        if st["label"] != rules.POSITIVE or box.id not in crops:
            continue
        pts = rules.final_points(st, crops[box.id])
        if pts:
            out[box.id] = pts
    return out


def iou(mask, tpts, crop):
    """마스크 하나와 정답의 IoU. 마스크가 없으면 0."""
    if mask is None:
        return 0.0
    a = geometry.rasterize(
        geometry.to_crop(geometry.loads(mask.polygon), crop), crop.w)
    b = geometry.rasterize(tpts, crop.w)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def score(ious):
    """IoU 목록 → 요약. 못 낸 것(0)도 들어 있어야 한다."""
    import numpy as np
    v = np.array(ious, dtype=float) if ious else np.zeros(0)
    if not len(v):
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p70": 0.0, "p90": 0.0}
    return {"n": len(v), "mean": float(v.mean()),
            "p50": float((v >= .5).mean()), "p70": float((v >= .7).mean()),
            "p90": float((v >= .9).mean())}
