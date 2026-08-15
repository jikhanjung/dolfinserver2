"""마스크 검토 — 격자에서 고르고, 아래 경계는 점 두 개로 정한다.

**기본값으로 두고 예외만 누른다.** 상자가 이미 "여기 지느러미가 있다" 고 말했고
SAM2 는 그 안의 것 하나를 딸 뿐이라, 사람이 하는 일은 틀린 것을 집어내는 것이다.

    누르지 않음   등지느러미 · 양날 온전 · 마스크 통과
    1~5           분류 (등지느러미·꼬리·주둥이·기타·아무것도아님)
    e             날 (양날 → 앞날만 → 뒷날만 → 둘 다 가림)
    점을 끈다      아래 경계를 고친다 (그 자체로 '교정함' 이 된다)
    x             마스크 윤곽이 틀렸다
    Enter         저장하고 다음 묶음

판정 규칙은 여기 다시 쓰지 않는다 — `finseg.rules` 를 부른다. 화면에 보이는
것이 곧 학습 자료여야 하기 때문이다.
"""
import json

from django.conf import settings
from django.db import transaction
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET, require_POST

from finseg import baseline, geometry, rules
from finseg.models import CLASSES, EDGES, Box, Crop, Review


def _tile(state, crop):
    """격자 한 칸. **폴리곤과 직선을 크롭 좌표로 바꿔 보낸다.**

    DB 는 원본 좌표로 들고 있고 화면은 크롭을 본다. 이 변환이 `geometry` 하나에만
    있어야 저장할 때 되돌리는 식과 어긋나지 않는다.
    """
    box, review = state["box"], state["review"]
    pts = geometry.to_crop(geometry.loads(state["polygon"]), crop) \
        if state["polygon"] else []
    base = geometry.loads(state["base_line"])
    if len(base) == 2:
        base = geometry.to_crop(base, crop)
    else:
        # 제안이 없으면 상자 아래 두 모서리에서 만든다 — 편향 없는 출발점이다.
        # 틀린 자동 제안은 사람의 눈을 끌고 가지만 대충의 제안은 그러지 않는다.
        (bx1, by1), (bx2, by2) = geometry.to_crop(
            [(box.x1, box.y1), (box.x2, box.y2)], crop)
        base = list(baseline.propose_from_box((bx1, by1, bx2, by2)))
    hint = []
    if box.src_not_fin:
        hint.append("옛 표시: 지느러미 아님")
    if box.src_not_identifiable:
        hint.append("옛 표시: 식별 불가")
    if box.boxname:
        hint.append(box.boxname)
    return {
        "box_id": box.id,
        "mask_id": state["mask"].id if state["mask"] else None,
        "crop": f"/crops/{crop.path}",
        "size": crop.w,
        "conf": state["mask"].conf if state["mask"] else None,
        # 옛 DB 에서 시민과학자가 남긴 표시. 판정을 대신하지 않고 **눈에 띄게만**
        # 한다 — 다른 사람의 판단이고, 여기서 다시 받는 것이 이 검토다.
        "hint": " · ".join(hint),
        "cls": (review.cls if review else "") or "fin",
        "edges": (review.edges if review else "") or "both",
        "verdict": (review.verdict if review else "") or "ok",
        "base_partial": state["base_partial"],
        "points": [[round(x, 1), round(y, 1)] for x, y in pts],
        "base": [[round(x, 1), round(y, 1)] for x, y in base],
    }


def index(request):
    return render(request, "review/grid.html", {
        "classes": json.dumps(CLASSES, ensure_ascii=False),
        "edges": json.dumps(EDGES, ensure_ascii=False),
    })


@require_GET
def batch(request):
    n = min(int(request.GET.get("n", 24)), 100)
    redo = request.GET.get("redo") == "1"
    qs = (Box.objects.filter(masks__is_current=True)
          .select_related("crop").prefetch_related("masks", "reviews")
          .distinct().order_by("id"))
    if not redo:
        qs = qs.filter(reviews__isnull=True)
    tiles = []
    for box in qs[:n * 2]:
        crop = getattr(box, "crop", None)
        if crop is None:
            continue
        tiles.append(_tile(rules.resolve(box), crop))
        if len(tiles) >= n:
            break
    return JsonResponse({"tiles": tiles})


@require_POST
@transaction.atomic
def save(request):
    """판정을 **쌓는다**. 덮어쓰지 않는다 — 고쳐 매긴 자취가 남아야 하고,
    나중에 여러 사람의 판정을 합의로 모을 수 있어야 한다.
    """
    body = json.loads(request.body or "{}")
    valid_cls = {c for c, _ in CLASSES}
    valid_edges = {e for e, _ in EDGES}
    crops = {c.box_id: c for c in Crop.objects.filter(
        box_id__in=[it["box_id"] for it in body.get("items", [])])}
    n = 0
    for it in body.get("items", []):
        cls = it.get("cls")
        if cls not in valid_cls:
            return JsonResponse({"error": f"모르는 분류: {cls}"}, status=400)
        edges = it.get("edges") or ""
        verdict = it.get("verdict") or ""
        if cls == "none":
            verdict, edges = "", ""
        else:
            if edges not in valid_edges:
                return JsonResponse({"error": f"모르는 날 상태: {edges}"}, status=400)
            if verdict not in ("ok", "fix"):
                return JsonResponse({"error": f"모르는 판정: {verdict}"}, status=400)

        base_str = ""
        crop = crops.get(it["box_id"])
        # **사람이 끈 것만 저장한다.** 안 건드린 제안까지 적으면 나중에 제안
        # 규칙을 고쳐도 옛 제안이 사람의 판단인 척 남는다.
        if it.get("base_moved") and crop is not None and len(it.get("base") or []) == 2:
            base_str = geometry.dumps(geometry.to_orig(it["base"], crop))

        Review.objects.create(
            box_id=it["box_id"], mask_id=it.get("mask_id"), cls=cls,
            verdict=verdict, edges=edges, base_line=base_str,
            base_partial=bool(it.get("base_partial")),
            reviewer=request.user if request.user.is_authenticated else None)
        n += 1
    return JsonResponse({"saved": n, "progress": dict(rules.progress())})


@require_GET
def progress(request):
    return JsonResponse(dict(rules.progress()))
