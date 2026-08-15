"""**판정 규칙은 여기 하나뿐이다.**

화면에 보이는 것이 곧 학습 자료여야 한다. 규칙을 검토 UI 와 내보내기가 따로
쓰면 언젠가 갈라지고, **갈라진 것은 눈에 띄지 않는다** — 화면에서는 지운
마스크가 학습에는 들어가 있는 식이다. DiaRUGA 가 `data._apply_review` 하나로
묶어 둔 것과 같은 이유다.

## 유효한 것

- **마스크**는 상자마다 `is_current=1` 인 것 하나. 엔진을 갈면 새 마스크가
  현재가 되고 옛 것은 남는다
- **판정**은 상자마다 `review.id` 가 가장 큰 것 하나. 고쳐 매길 수 있어야 하고,
  고친 자취는 남아야 한다

## 라벨

| 판정 | 학습 자료에서 |
|---|---|
| `ok` | **양성.** 마스크 그대로 |
| `fix` + 교정 폴리곤 | **양성.** 교정본으로 |
| `fix` (교정 전) | **아직 아니다** — 그 상자가 든 크롭을 통째로 뺀다 |
| `not_fin` | **아무것도 안 쓴다.** YOLO 는 라벨 없는 자리를 배경으로 배운다 |
| 판정 없음 | **크롭을 통째로 뺀다** |

마지막 두 줄이 갈리는 것이 중요하다. `not_fin` 은 사람이 "여기 지느러미가 없다"
고 말한 것이라 배경으로 쓸 수 있지만, **판정이 없는 것은 아무 말도 없는 것**이다.
그것을 배경으로 쓰면 SAM2 가 놓친 지느러미까지 "배경" 이라고 가르치게 된다.
"""

# 상자의 현재 마스크
CURRENT_MASK = """
    SELECT m.* FROM mask m WHERE m.box_id = ? AND m.is_current = 1
"""

# 상자의 최신 판정 — 상자마다 review.id 가 가장 큰 행 하나
LATEST_REVIEW_JOIN = """
    LEFT JOIN review r ON r.id = (
        SELECT MAX(r2.id) FROM review r2 WHERE r2.box_id = b.id
    )
"""

POSITIVE = "positive"      # 학습 자료의 양성
BACKGROUND = "background"  # 라벨 없이 배경으로
PENDING = "pending"        # 아직 사람이 말하지 않았다


def label_of(verdict, fix_polygon):
    """판정 한 건이 학습 자료에서 무엇이 되는가.

    `verdict` 는 `review.verdict` 이거나 판정이 없으면 None,
    `fix_polygon` 은 `review.polygon` (교정본) 이다.
    """
    if verdict is None:
        return PENDING
    if verdict == "not_fin":
        return BACKGROUND
    if verdict == "ok":
        return POSITIVE
    if verdict == "fix":
        return POSITIVE if fix_polygon else PENDING
    raise ValueError(f"모르는 판정: {verdict}")


def polygon_of(mask_polygon, fix_polygon):
    """학습에 쓸 폴리곤. 교정본이 있으면 그것이 이긴다."""
    return fix_polygon or mask_polygon


def box_states(conn, where="", params=()):
    """상자마다 (id, image_id, 마스크, 판정, 라벨) 을 낸다.

    내보내기와 검토 UI 와 진행 상황이 **전부 이 함수를 부른다.**
    """
    q = f"""
        SELECT b.id            AS box_id,
               b.image_id      AS image_id,
               b.x1, b.y1, b.x2, b.y2,
               m.id            AS mask_id,
               m.polygon       AS mask_polygon,
               m.conf          AS conf,
               m.run_id        AS run_id,
               r.verdict       AS verdict,
               r.polygon       AS fix_polygon,
               r.at            AS reviewed_at
          FROM box b
          LEFT JOIN mask m ON m.box_id = b.id AND m.is_current = 1
          {LATEST_REVIEW_JOIN}
          {where}
    """
    for row in conn.execute(q, params):
        d = dict(row)
        d["label"] = label_of(d["verdict"], d["fix_polygon"])
        d["polygon"] = polygon_of(d["mask_polygon"], d["fix_polygon"])
        yield d


def progress(conn):
    """지금 어디까지 왔나."""
    from collections import Counter
    c = Counter()
    for d in box_states(conn):
        c[d["label"]] += 1
        if d["mask_id"] is None:
            c["마스크없음"] += 1
        if d["verdict"]:
            c[f"판정:{d['verdict']}"] += 1
    c["상자"] = sum(1 for _ in conn.execute("SELECT 1 FROM box"))
    return c
