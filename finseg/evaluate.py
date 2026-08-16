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


def iou(mask, tpts, crop, state=None):
    """마스크 하나와 정답의 IoU. 마스크가 없으면 0.

    **후보도 정답과 같은 밑동에서 자른다.** 정답은 `final_points` 로 잘린
    폴리곤인데 후보를 안 자르면 IoU 가 두 가지를 섞어 잰다 — 잘라낸 양과
    실제 윤곽 차이다. 밑동 현은 사람의 판단이고 어느 엔진의 것도 아니므로
    양쪽에 똑같이 대는 것이 맞다.

    안 자르고 재던 때 SAM2 가 자기 출력을 정답으로 삼은 2,035장에서 0.936 이
    나왔는데, 1 이 아닌 이유가 오로지 밑동 아래를 잘라낸 만큼이었다.
    """
    if mask is None:
        return 0.0
    pts = geometry.to_crop(geometry.loads(mask.polygon), crop)
    if state is not None:
        pts = rules.final_points({**state, "polygon": mask.polygon}, crop)
    a = geometry.rasterize(pts, crop.w)
    b = geometry.rasterize(tpts, crop.w)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def independent(box):
    """**정답이 어느 엔진의 출력도 아닌가.**

    사람이 윗윤곽을 직접 다시 그린 상자만 참이다. 나머지는 사람이 "통과" 를
    누른 것이라 정답이 곧 그때 현재였던 엔진의 출력이고, 그 엔진을 그것으로
    채점하면 자기 답을 채점하는 셈이다.

    **이것은 SAM2 만의 문제가 아니다.** 엔진을 갈아 끼우고 다시 검토하면
    다음 정답의 대부분이 새 엔진의 출력이 된다 — 바퀴를 돌 때마다 되풀이된다.
    """
    r = rules.effective_review(box)
    return bool(r and r.polygon)


def score(ious):
    """IoU 목록 → 요약. 못 낸 것(0)도 들어 있어야 한다."""
    import numpy as np
    v = np.array(ious, dtype=float) if ious else np.zeros(0)
    if not len(v):
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p70": 0.0, "p90": 0.0}
    return {"n": len(v), "mean": float(v.mean()),
            "p50": float((v >= .5).mean()), "p70": float((v >= .7).mean()),
            "p90": float((v >= .9).mean())}
