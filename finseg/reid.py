"""개체 식별의 첫 걸음 — **뒷날 notch으로 닮은 것끼리 묶는다.**

## 왜 군집부터인가

임베딩을 지도학습으로 배우려면 같은 개체의 사진이 여러 장 있어야 한다. 그런데
옛 DB 의 개체명은 **166종 · 상자 435개**뿐이고, **상자 5개 이상인 개체가
다섯**이다. 게다가 그 이름 중에 `까치`·`황여새` 같은 새 이름이 섞여 있다 —
다른 용도로 쓰인 칸이라 그대로 믿을 수 없다.

그러니 순서가 반대다. **닮은 것끼리 먼저 묶어 사람에게 보이면 검토 한 번에
개체명이 여러 장씩 붙는다.** 그것이 나중에 임베딩을 배울 자료가 된다. 사람이
답할 질문도 "이게 누구냐" 가 아니라 **"이 둘이 같은 개체냐"** 라 훨씬 쉽다.

## 밑동 현이 정규화의 기준이다

`baseline.py` 가 밑동을 **수면선이 아니라 앞·뒤 삽입점**으로 정한 이유가 이것
이다 — 수면선은 같은 개체라도 자세에 따라 움직여서 그날 물결을 재게 된다.
해부학적 자리라야 회전·크기를 지울 수 있다.

    앞삽입점을 원점으로, 밑동 현을 x축으로, 현 길이를 1로.

그래서 `base_partial`("두 점 중 짐작한 것") 이 여기서 결정적이다. **뒷삽입점이
짐작이면 notch 를 재는 쪽 끝이 흔들린다** — 그 축을 참/거짓 하나로 두지 않고
앞·뒤를 갈라 둔 것이 여기서 값을 한다(`models.BASE_PARTIAL`).

## 좌현·우현을 뒤집어 합치면 안 된다

같은 개체라도 왼쪽에서 본 것과 오른쪽에서 본 것은 다른 그림이다. 뒤집어
맞추면 서로 다른 두 면을 같은 것으로 배운다. `facing` 이 그것을 가른다.
**임베딩을 배울 때 `fliplr` 을 쓰면 안 되는 것도 같은 이유다** (`TODOs`).
"""
from pathlib import Path

import numpy as np

from finseg import baseline, geometry

# 뒷날을 몇 점으로 다시 샘플할지. notch 는 작은 흠이라 너무 성기면 뭉개지고,
# 너무 촘촘하면 마스크의 들쭉날쭉함까지 재게 된다
N_POINTS = 64

# 표본으로 쓸 최소 넓이(px²). 한 변 ~174px. `TODOs` 가 "notch 는 하필 사람이
# 알아보는 문턱 근처에 있다" 고 적어 둔 자리라 **실측으로 정해야 한다** —
# 이 값은 출발점이지 답이 아니다
MIN_AREA = 30000


def usable(state):
    """이 상자를 re-ID 표본으로 쓸 수 있나 → (된다, 안 되는 이유).

    **거르는 축이 전부 검토에서 남긴 것이다.** 분할에는 필요 없어 보여도 적어
    둔 것들이 여기서 필터가 된다 (`CLAUDE.md` 의 "나중에 붙이려면 전부 다시
    검토해야 한다").
    """
    if state["cls"] != "fin":
        return False, "등지느러미가 아니다"
    if not state["polygon"]:
        return False, "윤곽이 없다"
    if len(geometry.loads(state["base_line"])) != 2:
        return False, "밑동 두 점이 없다"
    # 뒷날이 가려졌으면 notch 를 볼 수 없다. `tip`(끄트머리만)이 대부분이다.
    # `edges` 는 `rules.resolve` 가 안 내므로 판정에서 직접 읽는다 — 중앙
    # 함수에 칸을 더하면 부르는 쪽 전부가 영향을 받는다
    edges = (state["review"].edges if state["review"] else "") or "both"
    if edges not in ("both", "trailing"):
        return False, f"뒷날이 안 보인다 ({edges})"
    # **뒷삽입점이 짐작이면 재는 쪽 끝이 흔들린다.** 앞쪽만 짐작인 것은 쓴다 —
    # 현 각도는 흔들려도 뒷날 자체는 온전하다
    if state["base_partial"] in ("rear", "both", "unknown"):
        return False, f"뒷삽입점이 짐작이다 ({state['base_partial']})"
    if not state["facing"]:
        return False, "앞쪽을 모른다 (좌현·우현을 못 가른다)"
    box = state["box"]
    if (box.x2 - box.x1) * (box.y2 - box.y1) < MIN_AREA:
        return False, "너무 작다"
    return True, ""


def normalize(pts, base, facing):
    """윤곽을 **밑동 현 기준**으로 옮긴다 → (정규화된 점들, 현 길이).

    앞삽입점이 원점, 현이 x축, 현 길이가 1. 회전과 크기가 지워진다.

    `facing` 이 어느 쪽이 앞인지 말한다. **뒤집지 않는다** — 좌현과 우현은
    다른 그림이고, 맞추려고 뒤집으면 둘을 같은 것으로 만든다.
    """
    (ax, ay), (bx, by) = base
    # 두 점은 왼쪽·오른쪽으로만 저장된다 (`infer_base` 의 "두 점의 순서").
    # 어느 쪽이 앞인지는 `facing` 이 안다
    if facing == "right":
        (ax, ay), (bx, by) = (bx, by), (ax, ay)
    vx, vy = bx - ax, by - ay
    chord = float(np.hypot(vx, vy))
    if chord < 1e-6:
        return np.empty((0, 2)), 0.0
    c, s = vx / chord, vy / chord          # 현을 x축으로 돌리는 회전
    p = np.asarray(pts, float) - (ax, ay)
    out = np.stack([p[:, 0] * c + p[:, 1] * s,
                    -p[:, 0] * s + p[:, 1] * c], 1) / chord
    # **지느러미가 위로 서게 하되, 부호를 자료에서 읽는다.**
    #
    # `facing` 이 `right` 면 두 점을 맞바꾸므로 현의 방향이 뒤집히고, 그러면
    # 수직 성분의 부호도 함께 뒤집힌다. 처음에는 `*= -1` 을 무조건 걸었는데
    # 그러면 한쪽만 맞고 다른 쪽은 아래로 뒤집힌 채 남는다 — 실측으로
    # **왼쪽 371개는 +y, 오른쪽 288개는 −y** 였고, 조각을 그렸더니 오른쪽이
    # 통째로 canvas 밖으로 나가 납작한 띠만 남았다.
    #
    # **이것은 한쪽 무리를 거울처럼 뒤집는다.** 좌현과 우현을 **절대 한 통에
    # 넣지 않는 한** 안전하다 — 무리 안에서는 부호가 일관되기 때문이다.
    # 섞으면 서로 다른 두 면이 같아 보인다 (`fliplr` 을 금지하는 것과 같은 이유).
    if len(out) and out[:, 1].mean() < 0:
        out[:, 1] *= -1
    else:
        out[:, 1] *= 1
    return out, chord


def tip_index(pts):
    """밑동 현에서 가장 먼 점 — 지느러미 끝.

    정규화 뒤라 현이 x축이므로 **y 가 가장 큰 점**이다. 삽입점을 기울기로
    찾으려던 시도는 두 번 실패했지만(`baseline.py`), 끝을 찾는 것은 그것과
    다른 문제다 — 여기서는 최댓값 하나면 된다.
    """
    return int(np.argmax(pts[:, 1])) if len(pts) else -1


def trailing_edge(pts, n=N_POINTS):
    """정규화된 윤곽 → **뒷날만** `n` 점으로. 없으면 빈 배열.

    끝에서 뒷삽입점(1, 0)까지의 구간이다. 폴리곤은 닫힌 고리라 끝점에서 두
    갈래로 갈리는데, **뒷삽입점 쪽으로 가는 쪽**을 고른다.

    호 길이로 고르게 다시 샘플한다 — 원래 꼭짓점은 `approxPolyDP` 가 남긴
    것이라 간격이 들쭉날쭉하고, 그대로 재면 촘촘한 데를 무겁게 센다.
    """
    if len(pts) < 3:
        return np.empty((0, 2))
    t = tip_index(pts)
    if t < 0:
        return np.empty((0, 2))
    # 뒷삽입점은 정규화 뒤 (1, 0) 이다 — 거기에 가장 가까운 꼭짓점을 찾는다
    d = np.hypot(pts[:, 0] - 1.0, pts[:, 1])
    r = int(np.argmin(d))
    if r == t:
        return np.empty((0, 2))
    n_pts = len(pts)
    fwd = [(t + i) % n_pts for i in range((r - t) % n_pts + 1)]
    back = [(t - i) % n_pts for i in range((t - r) % n_pts + 1)]
    # **뒤쪽으로 가는 길**은 두 갈래다. 짧은 쪽이 아니라 **x 가 커지는 쪽**을
    # 고른다 — 앞날을 거쳐 돌아오는 길이 더 짧을 수 있다
    def score(path):
        q = pts[path]
        return float(np.mean(np.diff(q[:, 0]))) if len(q) > 1 else -1e9
    path = fwd if score(fwd) > score(back) else back
    q = pts[path]
    if len(q) < 2:
        return np.empty((0, 2))
    # 호 길이로 고르게
    seg = np.hypot(*np.diff(q, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-9:
        return np.empty((0, 2))
    want = np.linspace(0, s[-1], n)
    return np.stack([np.interp(want, s, q[:, 0]),
                     np.interp(want, s, q[:, 1])], 1)


def curve_of(state, crop):
    """상자 하나 → 뒷날 곡선 (`N_POINTS`×2). 못 만들면 빈 배열.

    윤곽은 **밑동 아래를 자른 것**을 쓴다 (`rules.final_points` 와 같은 규칙).
    안 자르면 밑동 밑의 몸통이 곡선에 섞인다.
    """
    from finseg import rules

    pts = rules.final_points(state, crop)
    if len(pts) < 3:
        return np.empty((0, 2))
    base = geometry.to_crop(geometry.loads(state["base_line"]), crop)
    norm, chord = normalize(pts, base, state["facing"])
    if chord <= 0:
        return np.empty((0, 2))
    return trailing_edge(norm)


def distance(a, b):
    """뒷날 곡선 둘 사이의 거리. 작을수록 닮았다.

    같은 길이로 다시 샘플했으므로 점끼리 견준다. **평균이 아니라 평균+최대를
    섞는다** — notch 는 국소적인 흠이라, 평균만 보면 한 군데 깊게 팬 차이가
    전체 평균에 묻힌다. 그 흠이 곧 개체의 단서다.
    """
    if len(a) != len(b) or not len(a):
        return float("inf")
    d = np.hypot(*(a - b).T)
    return float(d.mean() + 0.5 * d.max())


def pairwise(curves):
    """거리 행렬. 표본이 수백 개라 그냥 다 잰다."""
    n = len(curves)
    m = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            m[i, j] = m[j, i] = distance(curves[i], curves[j])
    return m


def cluster(dist, thres, linkage="complete"):
    """거리 행렬 → 묶음 목록 (번호들).

    ## 단일 연결은 여기서 못 쓴다 — 실측

    처음에 단일 연결로 짰다. "같은 개체의 사진 둘이 각도 때문에 멀어 보여도
    중간 각도의 사진 하나가 둘을 이어 준다" 는 논리였고, 사람이 가르는 것이
    다음 단계니 더 묶는 편이 낫다고 봤다. **틀렸다.**

        문턱 0.02 → 묶음 31 · 가장 큰 것 35
        문턱 0.04 → 묶음 37 · **가장 큰 것 223** (묶인 385개의 58%)

    한 다리 건너 이어지는 사슬이 표본의 절반을 한 덩어리로 만든다. 그런
    묶음은 사람이 볼 수 없어 **아무것도 안 묶은 것과 같다.**

    그래서 **완전 연결**이 기본이다 — 묶음 안의 **가장 먼 두 장**이 문턱 안에
    들어야 한다. 사슬이 안 생기고, "이 묶음은 다 비슷하다" 가 실제로 참이 된다.
    각도가 벌어진 짝은 묶음으로는 못 잡지만 `neighbours()` 가 잡는다.

    `scipy` 를 안 쓴다 — 표본이 수백 개라 필요가 없고 의존성을 더할 이유가 없다.
    """
    n = len(dist)
    if linkage == "single":
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(n):
            for j in range(i + 1, n):
                if dist[i, j] <= thres:
                    a, b = find(i), find(j)
                    if a != b:
                        parent[a] = b
        groups = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)
        return sorted(groups.values(), key=len, reverse=True)

    # 완전 연결: 가장 가까운 두 묶음을 합치되, **묶음끼리의 거리는 그 안의
    # 가장 먼 짝**으로 잰다. 그 거리가 문턱을 넘으면 멈춘다
    groups = [[i] for i in range(n)]
    d = dist.astype(float).copy()
    np.fill_diagonal(d, np.inf)
    alive = list(range(n))
    while len(alive) > 1:
        sub = d[np.ix_(alive, alive)]
        k = int(np.argmin(sub))
        a, b = divmod(k, len(alive))
        if sub[a, b] > thres:
            break
        ia, ib = alive[a], alive[b]
        groups[ia] = groups[ia] + groups[ib]
        # 합친 묶음까지의 거리는 **둘 중 먼 쪽**
        d[ia, :] = np.maximum(d[ia, :], d[ib, :])
        d[:, ia] = d[ia, :]
        d[ia, ia] = np.inf
        alive.remove(ib)
    return sorted((groups[i] for i in alive), key=len, reverse=True)


def sim_chain(emb, facing):
    """닮은 것끼리 한 줄로 편 등수 → 각 조각의 `sim`.

    격자를 훑을 때 **이웃이 닮아 있어야 눈이 덜 움직인다.** 묶음으로 보이는
    것은 그만뒀지만(묶음이 밝기로 뭉쳤고 특이한 개체가 오히려 빠졌다) 순서로
    돕는 것은 남았다 — 판단은 사람이 하고 모델은 순서만 돕는다.

    **좌·우를 따로 편다.** `normalize` 가 한쪽을 거울처럼 뒤집으므로 한 줄에
    섞으면 서로 다른 두 면이 이웃이 된다. 화면도 쪽을 먼저 가르고 그 안에서
    이 등수를 쓴다.

    **가장 외딴 것에서 출발한다** — 아무 데서나 시작하면 돌 때마다 줄이
    달라지고, 사람이 "아까 그 근처" 를 못 찾는다. 여기서 출발하면 줄이
    자료로 정해진다.
    """
    n = len(emb)
    out = np.zeros(n, int)
    X = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)
    for side in np.unique(facing):
        idx = np.where(facing == side)[0]
        if len(idx) < 2:
            for r, i in enumerate(idx):
                out[i] = r
            continue
        Z = X[idx]
        S = Z @ Z.T
        np.fill_diagonal(S, -np.inf)
        cur = int(np.argmin(S.max(1)))          # 가장 외딴 것
        left = np.ones(len(idx), bool)
        left[cur] = False
        order = [cur]
        for _ in range(len(idx) - 1):
            s = np.where(left, S[cur], -np.inf)
            cur = int(np.argmax(s))
            left[cur] = False
            order.append(cur)
        for r, j in enumerate(order):
            out[idx[j]] = r
    return out


# ---- 뒷날이 얼마나 복잡한가 -------------------------------------------------
#
# **`roughness` 가 사람 눈과 안 맞는다.** 상자에 넣은 조각과 안 만진 조각의
# notch 분포가 사실상 같았고(0.0450 대 0.0443), notch 사분위별 top-1 차이도
# 8pp뿐이었다(24.2% → 32.5%). 사람은 "매끈한 게 많아서 구분이 안 된다" 고
# 하는데 그 자는 그것을 못 잡는다.
#
# 문헌은 **곡률**로 수렴한다. 여기 셋을 함께 낸다 — **어느 것이 맞는지는
# 재서 정한다** (`reid_notch` 가 개체 맞히기 성적과의 상관을 낸다).
#
# 1. **적분 곡률**(CurvRank) — 각 점에 반지름 r 원을 씌워 곡선 한쪽에 들어온
#    면적의 비. 0.5 면 직선이다. **여러 r 로 재면 작은 notch 와 큰 굴곡이 다른
#    척도에서 잡힌다.** 자세·시점에 강해 큰돌고래 top-1 95%를 낸 자다
# 2. **notch 개수** — 곡률의 극값을 척도별로 세어, 큰 척도까지 살아남는 것만
#    센다. 사람이 "notch" 이라 부르는 것이 그것이다
# 3. **호 길이 / 현 길이** — 가장 단순한 자. 매끈함의 대용으로는 꽤 세다


def _resample(curve, n):
    """호 길이로 고르게 다시 뽑는다."""
    q = np.asarray(curve, float)
    if len(q) < 3:
        return np.empty((0, 2))
    seg = np.hypot(*np.diff(q, axis=0).T)
    t = np.concatenate([[0.0], np.cumsum(seg)])
    if t[-1] < 1e-9:
        return np.empty((0, 2))
    want = np.linspace(0, t[-1], n)
    return np.stack([np.interp(want, t, q[:, 0]),
                     np.interp(want, t, q[:, 1])], 1)


def integral_curvature(curve, r, n=256):
    """적분 곡률 — 점마다 반지름 `r` 안에서 곡선이 얼마나 휘었나. 0.5 가 직선.

    원 안에 든 곡선 조각과 그 **현** 사이의 면적을 원 넓이로 나눈 것에 0.5 를
    더한 값이다. 파인 자리는 0.5 아래, 볼록한 자리는 위로 간다.

    `r` 은 **현 길이 기준**이다(정규화된 곡선이라 현이 1). 작은 `r` 은 톱니를
    보고 큰 `r` 은 전체 굴곡을 본다 — **그래서 여러 척도로 재야 한다.**
    """
    q = _resample(curve, n)
    if not len(q):
        return np.empty(0)
    out = np.full(len(q), 0.5)
    for i in range(len(q)):
        d = np.hypot(*(q - q[i]).T)
        near = d <= r
        if near.sum() < 3:
            continue
        # 이어진 구간만 — 곡선이 되돌아와 원에 다시 들어오는 것은 안 센다.
        # **`in inside` 로 찾으면 점마다 배열을 훑어 O(n²) 가 된다**
        lo = i
        while lo > 0 and near[lo - 1]:
            lo -= 1
        hi = i
        while hi + 1 < len(q) and near[hi + 1]:
            hi += 1
        seg = q[lo:hi + 1]
        if len(seg) < 3:
            continue
        # 현과 곡선 사이의 부호 있는 면적 (신발끈 공식)
        a, b = seg[0], seg[-1]
        poly = np.vstack([seg, a])
        area = 0.5 * np.sum(poly[:-1, 0] * poly[1:, 1] - poly[1:, 0] * poly[:-1, 1])
        out[i] = 0.5 + float(area) / (np.pi * r * r)
    return out


def edge_complexity(curve, radii=(0.04, 0.08, 0.16), n=256):
    """뒷날 하나 → 복잡도 자 여럿. 못 재면 None.

    **하나로 안 줄인다.** 어느 자가 사람 눈에 가까운지 모르는 채로 하나를
    고르면, 그 자가 틀렸을 때 알아챌 방법이 없다 — `roughness` 가 그랬다.
    """
    q = _resample(curve, n)
    if not len(q):
        return None
    chord = float(np.hypot(*(q[-1] - q[0])))
    arc = float(np.sum(np.hypot(*np.diff(q, axis=0).T)))
    out = {"tort": arc / chord if chord > 1e-9 else 0.0}
    first = None
    for r in radii:
        ic = integral_curvature(q, r, n)
        if first is None:
            first = ic                 # 아래 notch 세기에서 다시 안 잰다
        key = f"ic{int(r*100):02d}"
        out[key] = float(ic.std())            # 척도별 흔들림
        # **파인 쪽만** 따로 — notch 는 한쪽으로만 파인다
        out[key + "_dip"] = float(np.mean(np.maximum(0.5 - ic, 0)))
    # notch 개수 — 가장 작은 척도에서 눈에 띄게 파인 골을 센다
    dip = 0.5 - first
    thr = max(0.02, float(dip.std()) * 1.5)
    on, cnt = False, 0
    for v in dip:
        if v > thr and not on:
            cnt += 1; on = True
        elif v <= thr * 0.5:
            on = False
    out["notches"] = cnt
    return out


# ---- 묶어서 묻기 -------------------------------------------------------------
#
# **한 장씩 묻지 말고 묶어서 묻는다.** 실측으로 5장을 묶으면 top-1 이 22.7% →
# 40%, top-5 는 68% → **80%** 가 되고, 무엇보다 **사람의 판단 한 번이 5장을
# 덮는다.**
#
# 묶는 근거는 둘을 곱한다 — 같은 날·같은 쪽에서
#
#     프레임 간격 ≤2 만            같은 개체일 확률 75%
#     조각 거리 ≤0.06 만           77%
#     **둘 다**                    **97%**
#
# 서로 다른 종류의 증거라서 곱하면 는다(하나는 그림, 하나는 촬영 순서).
# **거리 하나로는 안 된다** — 같은 개체의 거리 중앙값(0.100)과 남 중 가장
# 닮은 것들(하위 10%가 0.103)이 같은 자리라, 문턱을 넓히면 남이 섞인다.
# 좁게 잡아 **정밀도만** 가져가고 못 묶인 것은 사람이 보던 대로 본다.
LINK_DIST = 0.06
LINK_GAP = 2


def group_links(idx, emb, day, facing, frame, d_max=LINK_DIST, gap_max=LINK_GAP):
    """조각 번호들 → 묶음 목록. **같은 날·같은 쪽·가까운 프레임·닮은 것**만 잇는다.

    이어진 것을 타고 번지게 둔다(연결 요소). 사슬이 길어질 걱정은 문턱이
    좁아서 작다 — 넓히면 `cluster()` 가 겪은 사슬 문제가 그대로 생긴다.

    **한 사진 안의 둘은 절대 안 잇는다** — 한 마리가 한 사진에 두 번 나올 수
    없다. 공짜로 얻는 제약이라 안 쓸 이유가 없다 (실제로 이 제약으로 카탈로그에서
    겹친 상자 2건을 찾았다).
    """
    idx = np.asarray(idx)
    if len(idx) < 2:
        return [[int(i)] for i in idx]
    X = emb[idx] / np.maximum(np.linalg.norm(emb[idx], axis=1, keepdims=True), 1e-9)
    parent = list(range(len(idx)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = idx[a], idx[b]
            if day[i] != day[j] or facing[i] != facing[j]:
                continue
            if frame[i] < 0 or frame[j] < 0:
                continue
            if frame[i] == frame[j]:      # 같은 사진 — 다른 개체다
                continue
            if abs(int(frame[i]) - int(frame[j])) > gap_max:
                continue
            if 1.0 - float(X[a] @ X[b]) > d_max:
                continue
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    out = {}
    for a in range(len(idx)):
        out.setdefault(find(a), []).append(int(idx[a]))
    return sorted(out.values(), key=lambda g: -len(g))


def suggest(group, emb, day, facing, catalog_idx, k=5):
    """묶음 하나 → 닮은 개체 `k`개 [(개체, 점수)…]. 큰 것부터.

    개체마다 **그 개체가 가진 조각 중 가장 닮은 것**을 보고, 그것을 묶음 회원
    전체에 대해 평균한다. **묶음의 날은 후보에서 뺀다** — 같은 날 것이 끼면
    개체가 아니라 그날 조명을 맞히게 된다.
    """
    if not len(group):
        return []
    X = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)
    d0 = {day[i] for i in group}
    f0 = facing[group[0]]
    out = []
    for ind, members in catalog_idx.items():
        m = [i for i in members
             if day[i] not in d0 and facing[i] == f0 and i not in group]
        if not m:
            continue
        m = np.asarray(m)
        out.append((ind, float(np.mean([(X[m] @ X[i]).max() for i in group]))))
    out.sort(key=lambda t: -t[1])
    return out[:k]


def neighbours(dist, k=10):
    """번호마다 **가장 닮은 k 개**와 그 거리.

    묶음과 달리 문턱이 필요 없다. 사람에게 "이 지느러미와 닮은 것 열 장" 을
    보여 주고 같은 개체인지만 고르게 하는 자리다 — 문턱을 고르는 일 자체가
    답을 모르는 상태에서 하는 추측이라, **첫 바퀴에는 이쪽이 낫다.**
    """
    out = []
    for i in range(len(dist)):
        row = dist[i].copy()
        row[i] = np.inf
        idx = np.argsort(row)[:k]
        out.append([(int(j), float(row[j])) for j in idx if np.isfinite(row[j])])
    return out


# ---- 표준 조각 (chip) --------------------------------------------------------
#
# **곡선을 손으로 뽑는 대신 그림 자체를 견준다.**
#
# 뒷날 곡선은 끝점을 찾고 폴리곤을 걷는 단계가 있고, 그 둘이 다 깨지기 쉽다
# (마스크가 조금만 들쭉날쭉해도 끝점이 옮겨 간다). 그리고 뒷날만 쓰느라
# 가로세로비·앞날 휨·전체 실루엣을 버린다 — 그것도 개체의 단서다.
#
# 대신 **회전·크기를 표준화한 뒤 마스크 밖을 지운 조각**을 만든다. 무엇이
# 개체를 가르는지 사람이 짐작하지 않고, 배우는 쪽이 정하게 둔다.
#
# 배경을 지우는 이유는 그것이 **그날 바다**이기 때문이다. 안 지우면 같은 날
# 찍힌 것끼리 물빛이 닮아 개체보다 먼저 묶인다 — 밑동을 수면선으로 잡지 않은
# 것과 같은 논리다(`baseline.py`).

CHIP = 128          # 조각 한 변
CHIP_PAD = 0.15     # 현 길이 기준 여백. 끝이 잘리면 notch 가 함께 잘린다


def frame(base, pts, facing):
    """크롭 좌표 → 정규화 좌표의 **2×3 어파인 하나**. 못 만들면 None.

    앞삽입점이 원점, 밑동 현이 +x, 현 길이가 1, 지느러미가 +y.

    **여기가 유일한 변환식이다.** 마스크와 그림이 같은 행렬을 써야 조각이
    어긋나지 않는다 — 두 벌로 두면 몇 화소씩 밀리고, 밀린 조각은 "모델이 못
    땄다" 로 읽힌다.
    """
    (ax, ay), (bx, by) = base
    if facing == "right":
        (ax, ay), (bx, by) = (bx, by), (ax, ay)
    vx, vy = bx - ax, by - ay
    chord = float(np.hypot(vx, vy))
    if chord < 1e-6:
        return None, 0.0
    c, s = vx / chord, vy / chord
    R = np.array([[c, s], [-s, c]]) / chord
    q = (np.asarray(pts, float) - (ax, ay)) @ R.T
    # **부호를 자료에서 읽는다.** `facing` 이 `right` 면 현의 방향이 뒤집혀
    # 수직 성분의 부호도 뒤집힌다 — 무조건 `-1` 을 걸면 한쪽만 맞는다.
    # 실측으로 왼쪽 371개는 +y, 오른쪽 288개는 −y 였고, 그대로 그렸더니
    # 오른쪽이 통째로 canvas 밖으로 나가 납작한 띠만 남았다.
    #
    # **이것은 한쪽 무리를 거울처럼 뒤집는다.** 좌현·우현을 절대 한 통에 넣지
    # 않는 한 안전하다 (`fliplr` 을 금지하는 것과 같은 이유).
    sgn = 1.0 if q[:, 1].mean() >= 0 else -1.0
    A = np.array([[R[0, 0], R[0, 1]], [R[1, 0] * sgn, R[1, 1] * sgn]])
    t = -A @ np.array([ax, ay])
    return np.hstack([A, t.reshape(2, 1)]), chord


def chip(state, crop, size=CHIP, pad=CHIP_PAD, color=False, cut=True):
    """상자 하나 → **회전·크기를 맞춘** 조각.

    - `cut=True` (기본) — 마스크 밖을 지운다. **모델이 먹는 것**이다. 배경은
      그날 바다라, 안 지우면 같은 날 찍힌 것끼리 물빛이 닮아 개체보다 먼저
      묶인다
    - `cut=False`, `color=True` — **사람이 볼 것**이다. 배경과 색이 있어야
      판단이 쉽다. 지느러미가 물에 얼마나 잠겼는지, 빛이 어느 쪽에서 오는지,
      옆에 다른 개체가 있는지가 다 판단에 든다

    **정렬은 두 경우가 같다.** 같은 `frame()` 을 쓰므로 사람이 보는 그림과
    모델이 먹는 그림이 어긋나지 않는다 — 사람이 "이건 달라" 라고 한 것이
    모델에게는 다른 자리였다면 그 판정을 못 쓴다.

    돌려주는 것은 `(size, size)` 또는 `(size, size, 3)` float32 (0~1).
    못 만들면 `None`.
    """
    import cv2
    from django.conf import settings
    from PIL import Image

    from finseg import rules

    pts = rules.final_points(state, crop)
    if len(pts) < 3:
        return None
    base = geometry.to_crop(geometry.loads(state["base_line"]), crop)
    if len(base) != 2:
        return None
    F, chord = frame(base, pts, state["facing"])
    if F is None:
        return None

    # 정규화 좌표를 조각 화소로. 가로는 `-pad ~ 1+pad`, 세로도 같은 배율
    k = size / (1.0 + 2 * pad)
    S = np.array([[k, 0.0, pad * k], [0.0, -k, size - pad * k]])
    M = (S[:, :2] @ F[:, :2])
    M = np.hstack([M, (S[:, :2] @ F[:, 2:3] + S[:, 2:3])]).astype(np.float32)

    xy = (np.asarray(pts, float) @ M[:, :2].T) + M[:, 2]
    m = np.zeros((size, size), np.uint8)
    cv2.fillPoly(m, [np.round(xy).astype(np.int32)], 255)
    if not m.any():
        return None

    f = getattr(crop, "path", None)
    p_img = settings.FIN_CROPS / f if f else None
    if p_img is None or not p_img.exists():
        # 그림이 없으면 **실루엣만** 낸다 — 아무것도 안 내는 것보다 낫다
        return (m > 0).astype(np.float32)
    img = np.asarray(Image.open(p_img).convert("RGB" if color else "L"), np.uint8)
    warp = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR,
                          borderValue=0)
    if not cut:
        return warp.astype(np.float32) / 255.0
    if warp.ndim == 3:
        m = m[:, :, None]
    return np.where(m > 0, warp, 0).astype(np.float32) / 255.0


def overlay(state, crop, n=24):
    """조각 위에 얹을 **윤곽과 밑동**. `chip()` 과 같은 자로 옮긴다.

    돌려주는 것은 그림 크기로 나눈 0~1 좌표다 —
    `{"out": [[x,y]…], "base": [앞, 뒤], "facing": …}`. **밑동은 앞(머리 쪽)이
    먼저다.** 크기로 나누는 것은
    `look/`(사람이 보는 것)과 `chips/`(모델이 먹는 것)의 **크기가 달라서**다.
    같은 `frame()` 을 쓰므로 어느 쪽에 얹어도 자리가 맞는다.

    **꼭짓점을 줄여서 낸다.** 마스크는 중간값이 65개고 많으면 461개인데,
    격자 화면은 조각을 한꺼번에 다 실어 보내는 자리라 그대로 실으면 페이지가
    몇 배가 된다. notch 모양을 보는 데 24점이면 남는다.

    **그림에 구워 넣지 않는 이유**는 `chips/` 에 선이 들어가면 모델이 그 선을
    배우기 때문이다. 좌표로 내면 사람이 보는 쪽에만 얹힌다.
    """
    import cv2

    from finseg import rules

    # **앞쪽을 모르면 안 낸다.** `frame()` 이 `facing` 으로 좌우를 뒤집으므로,
    # 빈 값으로 부르면 **안 뒤집힌 좌표**가 나온다 — 그림이 뒤집혀 구워졌다면
    # 그 위에 거울상이 얹힌다. 없는 것보다 나쁜 것이 어긋난 것이다.
    if not state.get("facing"):
        return None
    pts = rules.final_points(state, crop)
    if len(pts) < 3:
        return None
    base = geometry.to_crop(geometry.loads(state["base_line"]), crop)
    if len(base) != 2:
        return None
    F, _ = frame(base, pts, state["facing"])
    if F is None:
        return None
    # `chip()` 의 `S` 와 같되 크기를 1 로 둔다 — 그래서 곧바로 0~1 이다.
    k = 1.0 / (1.0 + 2 * CHIP_PAD)
    S = np.array([[k, 0.0, CHIP_PAD * k], [0.0, -k, 1.0 - CHIP_PAD * k]])
    M = S[:, :2] @ F[:, :2]
    t = S[:, :2] @ F[:, 2:3] + S[:, 2:3]

    def put(a):
        return ((np.asarray(a, float) @ M.T) + t.ravel())

    xy = put(pts).astype(np.float32)
    # `approxPolyDP` 의 문턱을 올려 가며 점 수를 맞춘다. 한 번에 정하는 식이
    # 없어서 훑는데, 열 번이면 닿는다.
    eps, out = 0.002, xy
    for _ in range(12):
        got = cv2.approxPolyDP(xy.reshape(-1, 1, 2), eps, True).reshape(-1, 2)
        out = got
        if len(got) <= n:
            break
        eps *= 1.6
    # **밑동 두 점을 앞(머리 쪽)부터 낸다.** `frame()` 이 안에서 `facing` 에
    # 따라 뒤바꾸므로, 원래 순서 그대로 옮기면 좌현·우현에서 앞뒤가 반대로
    # 나온다. 여기서 같은 규칙으로 맞춰 두면 **화면은 "0번이 앞" 하나만 알면
    # 된다** — 규칙이 두 곳에 있으면 언젠가 갈라진다.
    b = list(base[::-1] if state["facing"] == "right" else base)
    return {"out": [[round(float(x), 4), round(float(y), 4)] for x, y in out],
            "base": [[round(float(x), 4), round(float(y), 4)] for x, y in put(b)],
            "facing": state["facing"]}


# ---- 개체 판정 ---------------------------------------------------------------

def effective_id(box):
    """이 상자에 유효한 개체 판정 하나. 없으면 None.

    `rules.effective_review` 와 같은 규칙이다 — **가장 늦은 것.** 여기가
    멀티유저의 갈림길이고, 사람이 늘면 사람마다 최신을 고른 뒤 합의로 간다.
    """
    return box.identifications.order_by("-id").first()


# ---- 1차 목표 ---------------------------------------------------------------
#
# **왜 이 숫자인가.** 넷 다 재어 둔 것에서 나왔지 정해 놓고 시작한 것이 아니다.
#
# `per_individual` = 10 — 개체당 조각 수와 top-1 의 관계를 재 보면 **문턱이
# 10 에 있다**: 4~9장 11.1% · 10~19장 30.6% · 20장 이상 33.8%. 10 을 넘길 때
# +19.5pp 이고 그 뒤로는 +3.2pp 다. **몇 마리를 40장으로 만드는 것보다 모두를
# 10장으로 올리는 편이 훨씬 세다.**
#
# `days_per_individual` = 2 — `reid_cls` 가 **관찰일을 통째로 갈라** 배우는 날과
# 재는 날을 안 겹치게 한다. 그래서 하루에만 찍힌 개체는 배우는 쪽이나 재는 쪽
# **한 곳에만 들어가고 다른 쪽에서는 아예 안 쓰인다.** 조각 수와 달리 이건
# 많고 적음이 아니라 **되나 안 되나**다.
#
# `individuals` = 60 — 제주 남방큰돌고래 개체군이 100~120 이다. 절반은 넘겨야
# 카탈로그라 부를 수 있다. (Kim et al. 2024 의 reference 는 79개체였다.)
#
# `chips` = 1,200 — 위 셋을 채우면 그 언저리다. 덤으로, 그 논문이 fine-tuning
# 에 쓴 실사진이 35개체 × 22장 = **770장**이라 1,200 은 그것을 넘긴다 —
# 백본 마지막 블록을 녹이는 일이 "자료가 모자라서 안 된다" 가 아니게 되는
# 첫 지점이다.
#
# **성적으로 검수하지 말 것.** 개체가 42 → 60 이 되면 문제 자체가 어려워져서
# top-1 이 안 오르거나 내릴 수 있다. 그것은 뒷걸음이 아니다
# (`HANDOFF` 의 `정답이 다른 두 판을 세로로 견주지 말 것`).
GOAL = {
    "individuals": 60,
    "chips": 1200,
    "per_individual": 10,
    "days_per_individual": 2,
}


def dataset_state(goal=None):
    """지금 자료가 목표 어디쯤인가. **세는 자리는 여기 하나다.**

    화면(`/dataset`)과 뒷날 명령이 같은 함수를 부른다 — 두 벌로 두면 한쪽만
    고쳐지고, 그때 어느 쪽 숫자가 맞는지 아무도 모른다 (`rules.py`·
    `evaluate.py` 를 한 곳에 둔 것과 같은 이유다).

    **날은 `items.json` 이 아니라 사진에서 읽는다.** 격자는 갈아 끼우는 것이라
    (`FIN_REID`) 거기서 읽으면 격자를 바꿀 때마다 진척이 흔들린다.

    **면은 반대로 격자에서 읽는다.** 분류기가 좌현·우현을 따로 배울 때 보는
    것이 격자의 `facing` 이고(`reid_chips --auto-facing` 이 제안한 것까지
    들어 있다), DB 의 것은 **사람이 누른 것만**이라 절반 넘게 비어 있다.
    "실효는 면마다 반이다" 를 말하려는 칸이니 **모델이 보는 쪽**을 세야 한다.
    격자에 없으면 DB 로 물러서고, 그래도 모르면 `unknown` 으로 따로 센다 —
    좌·우 합이 조각 수와 안 맞는 채로 두면 어디가 빈 것인지 안 보인다.
    """
    import json
    from collections import defaultdict

    from django.conf import settings

    from finseg import rules
    from finseg.models import Box, Individual

    g = dict(GOAL, **(goal or {}))
    cat = catalog()
    ids = [b for v in cat.values() for b in v]
    day = dict(Box.objects.filter(id__in=ids).values_list("id", "image__obsdate"))
    facing = {}
    items = Path(settings.FIN_REID) / "items.json"
    if items.exists():
        for it in json.loads(items.read_text())["items"]:
            if it.get("facing"):
                facing[int(it["id"])] = it["facing"]
    rest = [b for b in ids if b not in facing]
    if rest:
        for b in Box.objects.filter(id__in=rest).prefetch_related("reviews", "masks"):
            facing[b.id] = rules.resolve(b).get("facing") or ""
    names = dict(Individual.objects.values_list("id", "name"))
    nicks = dict(Individual.objects.values_list("id", "nickname"))

    rows = []
    for ind, boxes in cat.items():
        days = {str(day.get(b)) for b in boxes if day.get(b)}
        side = defaultdict(int)
        for b in boxes:
            side[facing.get(b, "")] += 1
        rows.append({
            "id": ind, "name": names.get(ind, f"#{ind}"),
            "nick": nicks.get(ind, ""),
            "n": len(boxes), "days": len(days),
            "left": side["left"], "right": side["right"],
            "unknown": len(boxes) - side["left"] - side["right"],
            "need_chips": max(0, g["per_individual"] - len(boxes)),
            "need_days": max(0, g["days_per_individual"] - len(days)),
        })
    # **아직 모자란 것 중 가장 조금 모자란 것부터.** 이 화면이 대시보드로
    # 끝나면 보고 나서 무엇을 할지는 여전히 사람이 정해야 한다 — 문턱에
    # 가장 가까운 개체를 위에 놓으면 그대로 다음에 할 일 목록이 된다.
    rows.sort(key=lambda r: (
        not (r["need_chips"] or r["need_days"]),        # 다 된 것은 아래로
        r["need_chips"] + r["need_days"] * 3,           # 날이 더 비싸다
        -r["n"]))
    return {
        "goal": g,
        "rows": rows,
        "n_ind": len(rows),
        "n_chip": sum(r["n"] for r in rows),
        "at_chips": sum(1 for r in rows if not r["need_chips"]),
        "at_days": sum(1 for r in rows if not r["need_days"]),
        "at_both": sum(1 for r in rows if not (r["need_chips"] or r["need_days"])),
        "short_chips": sum(r["need_chips"] for r in rows),
        "n_unknown": sum(r["unknown"] for r in rows),
    }


def catalog(as_of=None):
    """개체 → 그 개체로 확정된 상자 번호들. **뺀 것은 안 센다.**

    `as_of` 에 `Identification` 번호를 주면 **그때의 카탈로그**를 돌려준다.
    표가 쌓이는 것이 그러라고 있는 것이다 (`models.Identification`) — 자가
    좋아진 것인지 정답이 늘어서인지는 **옛 정답에 새 자를 대 봐야** 갈린다.

    다만 `kind` 는 개체의 **지금** 상태라 되살릴 수 없다. 그때 임시보관함이던
    것이 지금 개체면 여기 들어온다 — 부르는 쪽이 그 한정을 함께 말한다.
    """
    from collections import defaultdict

    from finseg.models import Identification
    qs = Identification.objects.order_by("id")
    if as_of is not None:
        qs = qs.filter(id__lte=as_of)
    latest = {}
    for i in qs.values_list("box_id", "individual_id"):
        latest[i[0]] = i[1]
    # **개체가 아닌 자리는 카탈로그에 안 넣는다** — 임시보관함도, 지느러미가
    # 아니라고 한 것도. 세면 성적이 그만큼 부풀고, 그 사실이 눈에 안 띈다
    from finseg.models import Individual
    hold = set(Individual.objects.exclude(kind="")
               .values_list("id", flat=True))
    out = defaultdict(list)
    for box_id, ind in latest.items():
        if ind is not None and ind not in hold:
            out[ind].append(box_id)
    return dict(out)


def decided(box_ids):
    """이미 답한 상자 번호들 — 개체를 붙였든 아니라고 했든.

    묶음을 다시 보여 줄 때 **이미 답한 것을 또 묻지 않으려고** 쓴다.
    아니라고 말한 것도 답이다.
    """
    from finseg.models import Identification
    return set(Identification.objects.filter(box_id__in=box_ids)
               .values_list("box_id", flat=True))


# ---- 얼마나 특징적인가 --------------------------------------------------------
#
# **특이한 개체가 오히려 안 보인다.** 묶음은 둘 이상 든 것만 내는데, 특이하다는
# 것은 남과 안 닮았다는 뜻이라 혼자 남기 쉽다. 그런데 앞날이 패였거나 뒷날이
# 많이 울퉁불퉁한 개체가 **사람이 가장 쉽게 알아보는 것**이고, 첫 정답을
# 만들기에 가장 값이 크다.
#
# 그래서 notch 를 재서 **특징적인 것부터** 보여 준다. 재는 법은 볼록껍질과의
# 차이다 — 파인 자리가 곧 notch 가고, 그 깊이가 곧 특징의 크기다.


# ---- 밝기 경계로 스냅 --------------------------------------------------------
#
# **마스크는 넓이는 맞는데 경계가 notch 크기에서 어긋난다.** 분할 모델의 마스크는
# prototype 격자(입력의 1/4 — 640 크롭이면 한 칸이 4px)에서 나오는데 진짜 notch 는
# 크롭 6~16px 이라 **한 칸 반에서 네 칸**이다. `retina_masks` 가 640 으로 늘려
# 저장하지만 **늘린다고 정보가 생기지 않는다.**
#
# 640 크롭에는 원본 화소가 다 들어 있으니(원본 ~286px 영역을 640으로 편 것),
# 그 격자를 벗어나 **밝기 기울기가 가장 센 자리로 옮겨 붙인다.** 학습이 없다 —
# 지느러미는 물 위의 검은 형체라 경계 대비가 세다.
#
# 실측(`devlog/20260827_003`): 뒷날 벡터가 되풀이되는 정도(AUC, 날을 건너뛴 짝)가
# **0.527 → 0.625**. 그리고 이것으로 다시 잰 거칢으로 "notch 있는 것" 을 고르면
# 날을 건너뛴 성적이 **단조롭게 오른다**(0.643 → 0.676) — 모델 마스크의 거칢으로
# 고르면 같은 날만 오르고 날을 건너뛰면 안 오른다. **그 거칢은 notch 가 아니라
# 마스크 잡음을 재고 있었다.**

SNAP_SEARCH = 4.0     # 법선으로 훑을 거리(크롭 px). ±10 은 파도·반사로 뛰어 되레 나빠진다
SNAP_STEP = 1.5       # 다시 샘플할 간격. 성기면 옮겨 봐야 notch 모양이 안 생긴다
SNAP_MED = 3          # 튐을 거르는 중앙값 창(점 수)


def snap_to_edge(pts, gray, search=SNAP_SEARCH, step=SNAP_STEP, med=SNAP_MED):
    """폴리곤(크롭 좌표) → **밝기 경계에 붙인 폴리곤.**

    각 점을 제 법선으로 훑어 밝기 기울기가 가장 센 자리로 옮긴다.

    **먼저 촘촘히 다시 샘플한다.** 꼭짓점 간격이 크롭 15px 인데 notch 는
    6~16px 이라, 점이 성기면 옮겨 봐야 notch 모양이 안 생긴다.

    **평균이 아니라 중앙값으로 튐을 거른다.** 물보라·파도 마루·반사가 진짜
    경계보다 센 자리가 있어 한 점만 튀는데, 그 튐은 모양으로는 notch 와
    똑같다 — notch 를 재려는 판에 가짜 notch 를 만드는 셈이다. 그런데 5칸
    **평균**(창 7.5px)을 쓰면 창이 notch 폭(6~16px)과 같은 크기라 **진짜
    notch 도 함께 뭉갠다.** 중앙값은 한 점짜리 튐만 없애고 여러 점이 함께
    들어간 계단은 남긴다 — notch 가 그 계단이다.
    """
    import cv2

    p = np.asarray(pts, float)
    if len(p) < 4 or gray is None:
        return pts
    # 고르게 다시 샘플 (닫힌 고리)
    q = np.vstack([p, p[:1]])
    t = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(q, axis=0).T))])
    if t[-1] < 1e-6:
        return pts
    want = np.arange(0, t[-1], step)
    p = np.stack([np.interp(want, t, q[:, 0]), np.interp(want, t, q[:, 1])], 1)

    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.2)
    gm = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, 3), cv2.Sobel(g, cv2.CV_32F, 0, 1, 3))
    H, W = gm.shape
    d = np.roll(p, -1, 0) - np.roll(p, 1, 0)
    L = np.hypot(*d.T)[:, None]
    L[L < 1e-6] = 1.0
    nrm = np.stack([-d[:, 1], d[:, 0]], 1) / L
    best = np.zeros(len(p))
    mx = None
    for k, off in enumerate(np.arange(-search, search + 1e-6, 0.5)):
        z = p + nrm * off
        v = cv2.remap(gm, np.clip(z[:, 0], 0, W - 1).astype(np.float32)[None],
                      np.clip(z[:, 1], 0, H - 1).astype(np.float32)[None],
                      cv2.INTER_LINEAR)[0]
        if mx is None:
            mx = v.copy()
            best[:] = off
        else:
            hit = v > mx
            mx[hit] = v[hit]
            best[hit] = off
    if med >= 3:
        h = med // 2
        pad = np.r_[best[-h:], best, best[:h]]
        best = np.median(np.stack([pad[i:i + len(best)] for i in range(med)]), 0)
    out = p + nrm * best[:, None]
    return [(float(x), float(y)) for x, y in out]


def roughness(state, crop, n=256, smooth=0.08):
    """윤곽이 얼마나 **울퉁불퉁한가** → dict. 길이는 밑동 현 기준이다.

    ## 볼록껍질로 재면 안 된다 — 실측

    처음에 `convexityDefects` 로 쟀다. 그랬더니 깊이 중앙값이 **0.187**(현의
    19%)이 나오고 659개 중 641개가 "특징적" 이었다. 진짜 notch 는 현의 2~5%다.
    볼록껍질과의 차이는 notch 가 아니라 **끝에서 밑동으로 내려오는 자연스러운
    휨**을 재고 있었다. 다 특징적이면 아무것도 특징적이지 않다.

    그래서 **자기 자신을 매끄럽게 편 것과 견준다.** 낮은 주파수(전체 모양)를
    빼면 높은 주파수(패임·톱니)만 남는다. `smooth` 는 미는 창의 크기이고
    윤곽 길이의 비다 — 크면 큰 휨까지 notch으로 세고, 작으면 마스크의
    들쭉날쭉함까지 센다.

    앞날·뒷날을 갈라 낸다. **앞날이 패인 개체와 뒷날이 톱니인 개체는 사람이
    가장 쉽게 알아보는 둘**이고, 첫 정답을 만들기에 값이 가장 크다.
    """
    from finseg import rules

    pts = rules.final_points(state, crop)
    if len(pts) < 8:
        return None
    base = geometry.to_crop(geometry.loads(state["base_line"]), crop)
    if len(base) != 2:
        return None
    F, chord = frame(base, pts, state["facing"])
    if F is None:
        return None
    q = (np.asarray(pts, float) @ F[:, :2].T) + F[:, 2]

    # 밑동 현 위(위쪽 윤곽)만 본다 — 아래는 우리가 그은 직선이라 뜻이 없다
    tip = int(np.argmax(q[:, 1]))
    # 호 길이로 고르게 다시 샘플한다. 꼭짓점 간격이 들쭉날쭉해서 그냥 쓰면
    # 촘촘한 데를 무겁게 센다
    closed = np.vstack([q, q[:1]])
    seg = np.hypot(*np.diff(closed, axis=0).T)
    t = np.concatenate([[0.0], np.cumsum(seg)])
    if t[-1] < 1e-6:
        return None
    want = np.linspace(0, t[-1], n, endpoint=False)
    r = np.stack([np.interp(want, t, closed[:, 0]),
                  np.interp(want, t, closed[:, 1])], 1)

    # 고리를 따라 미는 평균 — 낮은 주파수(전체 모양)
    k = max(3, int(round(n * smooth)) | 1)
    pad = np.vstack([r[-(k // 2):], r, r[:k // 2]])
    ker = np.ones(k) / k
    low = np.stack([np.convolve(pad[:, 0], ker, "valid"),
                    np.convolve(pad[:, 1], ker, "valid")], 1)[:n]
    dev = np.hypot(*(r - low).T)          # 높은 주파수만 남는다

    # 밑동 아래(우리가 그은 직선)는 뺀다
    up = r[:, 1] > 0.02
    if up.sum() < 8:
        return None
    tx = float(q[tip, 0])
    front = up & (r[:, 0] < tx)
    rear = up & (r[:, 0] >= tx)
    f = lambda m: (float(dev[m].mean()), float(dev[m].max())) if m.sum() > 3 \
        else (0.0, 0.0)
    fm, fx = f(front)
    rm, rx = f(rear)
    return {"chord": chord,
            "front_mean": fm, "front_max": fx,
            "rear_mean": rm, "rear_max": rx,
            "total": float(dev[up].mean()),
            "max": float(dev[up].max())}
