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
import re
from pathlib import Path

from django.conf import settings
from django.db import transaction
from django.db.models import Max
from django.http import FileResponse, Http404, JsonResponse
from django.shortcuts import get_object_or_404, render
from django.utils.html import escape
from django.utils.safestring import mark_safe
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_GET, require_POST

from finseg import baseline, evaluate, geometry, onnxdet, rules
from finseg.models import (BASE_PARTIAL, CLASS_KEYS, CLASSES, EDGES, FACING,
                           Box, Crop, Identification, Image, Mask, Review, Run)


def _tile(state, crop):
    """격자 한 칸. **폴리곤과 직선을 크롭 좌표로 바꿔 보낸다.**

    DB 는 원본 좌표로 들고 있고 화면은 크롭을 본다. 이 변환이 `geometry` 하나에만
    있어야 저장할 때 되돌리는 식과 어긋나지 않는다.
    """
    box, review = state["box"], state["review"]
    pts = geometry.to_crop(geometry.loads(state["polygon"]), crop) \
        if state["polygon"] else []
    base = geometry.loads(state["base_line"])
    # **이 두 점이 사람이 찍은 것인가, 기계의 제안인가.** 화면이 이것을 모르면
    # 되짚어 보다가 딴 축만 고쳐 저장할 때 `base_moved` 가 거짓이라 밑동이 빈
    # 채로 새 판정이 쌓이고, 그것이 앞의 것을 덮어 **사람이 찍은 두 점이
    # 버려진다.** 실제로 그렇게 76건을 잃었다.
    base_human = bool(review and review.base_line)
    poly_human = bool(review and review.polygon)
    if len(base) == 2:
        base = geometry.to_crop(base, crop)
    else:
        # 제안이 없으면 상자 아래 두 모서리에서 만든다 — 편향 없는 출발점이다.
        # 틀린 자동 제안은 사람의 눈을 끌고 가지만 대충의 제안은 그러지 않는다.
        (bx1, by1), (bx2, by2) = geometry.to_crop(
            [(box.x1, box.y1), (box.x2, box.y2)], crop)
        base = list(baseline.propose_from_box((bx1, by1, bx2, by2)))
    # **앞쪽 제안.** 사람이 아직 안 말했을 때만 낸다 — 말했으면 그것이 답이다.
    # 제안은 **저장까지 간다** — 화면이 안 누른 기본값을 판정으로 적는 것과
    # 같다 (`cls`·`edges`·`verdict`). 밑동 제안을 안 적는 것과 사정이 다르다:
    # 그것은 85%가 틀리지만 이 규칙은 문턱 위에서 149/149 였다.
    #
    # 두 가지를 조심한다. (1) `state["cls"]` 는 **아직 안 본 상자에서 빈 값**이라
    # 그것으로 거르면 제안이 가장 필요한 것들이 전부 빠진다 — 화면이 쓰는 기본값
    # `fin` 으로 판단한다. (2) **밑동 아래를 자른 뒤**의 폴리곤으로 재야 한다.
    # 안 자르면 밑동 밑의 몸통이 넓이에 섞여 좌우 비가 뒤집힌다.
    cls = (review.cls if review else "") or "fin"
    facing_hint = ""
    if cls == "fin" and not state["facing"]:
        cut = rules.final_points({**state, "cls": "fin"}, crop)
        facing_hint, _ = baseline.propose_facing(cut, crop.w)
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
        # 마스크가 있으면 그 확신, 없으면 **상자를 낸 검출기의 확신**이다.
        # 새 검출기가 들인 상자는 아직 마스크가 없고(`infer_boxes`), 거기서는
        # 이 값이 곧 문턱을 고르는 근거가 된다.
        "conf": state["mask"].conf if state["mask"] else box.conf,
        # **어느 검출기가 낸 상자인가.** 화면이 이것을 말해야 하는 이유는
        # 판단이 달라지기 때문이다 — 옛 상자는 "이 마스크가 맞나" 를 묻지만
        # 새 상자는 "여기 정말 지느러미가 있나" 를 묻는다.
        "source": box.source,
        # 옛 DB 에서 시민과학자가 남긴 표시. 판정을 대신하지 않고 **눈에 띄게만**
        # 한다 — 다른 사람의 판단이고, 여기서 다시 받는 것이 이 검토다.
        "hint": " · ".join(hint),
        "cls": cls,
        "edges": (review.edges if review else "") or "both",
        "verdict": (review.verdict if review else "") or "ok",
        "base_human": base_human,
        "poly_human": poly_human,
        "base_partial": state["base_partial"],
        "facing": state["facing"],
        "facing_hint": facing_hint,
        "points": [[round(x, 1), round(y, 1)] for x, y in pts],
        "base": [[round(x, 1), round(y, 1)] for x, y in base],
    }


def _emph(s):
    """`**굵게**` 만 알아듣는다. 규칙 문구를 파이썬 쪽에 두려고 쓴다."""
    return mark_safe(re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escape(s)))


# **이 화면이 `csrftoken` 쿠키의 유일한 출처다.** 저장은 fetch 로 하므로 폼이
# 없고, 폼이 없으니 `{% csrf_token %}` 도 없다 — 그러면 Django 는 쿠키를 낼 일이
# 없고 `/api/review` 가 403 으로 막힌다. 한동안 됐던 것은 브라우저에 `/admin/`
# 에서 받은 쿠키가 남아 있었기 때문이고, 그것이 지워지자 저장이 통째로 멈췄다.
@ensure_csrf_cookie
def index(request):
    # **규칙은 화면에 늘 떠 있어야 한다.** 밑동 현이라는 것을 화면이 말하지
    # 않으면 검토가 스스로와 어긋나고, 어긋난 검토는 나중에 되살릴 수 없다.
    # 문구는 `finseg.baseline` 에 있다 — 규칙을 고치면 화면이 따라온다.
    return render(request, "review/grid.html", {
        "classes": json.dumps(CLASSES, ensure_ascii=False),
        "class_keys": json.dumps(CLASS_KEYS, ensure_ascii=False),
        "edges": json.dumps(EDGES, ensure_ascii=False),
        "partials": json.dumps(BASE_PARTIAL, ensure_ascii=False),
        "facings": json.dumps(FACING, ensure_ascii=False),
        "rule": _emph(baseline.RULE),
        "rule_points": [_emph(p) for p in baseline.RULE_POINTS],
    })


def _tiles_for(boxes):
    """상자 목록 → 칸 목록. 크롭 없는 상자는 뺀다 (화면에 낼 것이 없다)."""
    out = []
    for box in boxes:
        crop = getattr(box, "crop", None)
        if crop is not None:
            out.append(_tile(rules.resolve(box), crop))
    return out


def _reviewed_order(cls=""):
    """검토를 **마친 순서**대로 상자 번호. 앞이 먼저 본 것이다.

    `Review` 는 쌓이므로 한 상자에 여러 건이 붙는다. **처음 본 때**로 자리를
    잡는다 — 고쳐 매겼다고 순서가 바뀌면 "1페이지" 가 어제와 달라진다.

    `cls` 를 주면 그 분류만 낸다. 거르는 기준은 **유효 판정**(가장 늦은 것)이라,
    `아무것도아님` 으로 넘겼다가 나중에 `사람` 으로 고친 상자는 `사람` 에서
    나온다 — 자취가 아니라 지금 값으로 본다.
    """
    seen, order, latest = set(), [], {}
    for box_id, c in (Review.objects.order_by("id")
                      .values_list("box_id", "cls")):
        latest[box_id] = c
    for box_id in (Review.objects.order_by("at", "id")
                   .values_list("box_id", flat=True)):
        if box_id in seen:
            continue
        seen.add(box_id)
        if not cls or latest.get(box_id) == cls:
            order.append(box_id)
    return order


@require_GET
def batch(request):
    """`mode=todo` 는 아직 안 본 것을 앞에서부터, `mode=done` 은 **이미 본 것을
    본 순서대로 페이지로** 낸다.

    되짚어 보는 길이 있어야 하는 이유는 하나다 — **기준은 검토하면서 선다.**
    첫 페이지를 매길 때의 기준과 열 번째 페이지의 기준이 같은지는 돌아가 봐야만
    알 수 있고, 그것이 이 자료에서 가장 되살리기 어려운 것이다.
    """
    n = min(int(request.GET.get("n", 24)), 100)
    page = max(int(request.GET.get("page", 1)), 1)
    mode = request.GET.get("mode", "todo")
    if mode not in ("todo", "done", "stuck", "stale", "new", "noshape"):
        mode = "todo"
    cls = request.GET.get("cls", "")
    if cls not in {c for c, _ in CLASSES}:
        cls = ""
    # **보류.** 판단이 안 서는 칸을 지나가려는 것이다. 묶음 저장은 손 안 댄
    # 칸까지 전부 판정으로 적으므로(`save`), 화면에서 빼지 않으면 "모르겠다"
    # 를 말할 자리가 없다 — 찍는 쪽이든 안 찍는 쪽이든 없는 판단이 들어간다.
    #
    # **서버에서 뺀다.** 화면에서만 걸러 내면 한 쪽이 통째로 보류일 때 빈
    # 격자가 뜨고 쪽수도 어긋난다. 여기서 빼면 `total`·`pages` 가 곧바로 맞는다.
    #
    # 어디에도 저장하지 않는다. 보류는 **아무 말도 안 한 것**(`rules.PENDING`)
    # 이고 그것이 이미 정확한 상태다 — 표를 만들면 "모른다" 가 판정인 척
    # 남는다. 화면이 이번 판(sessionStorage)만 기억하고, 새로 열면 다시 만난다.
    hold = {int(v) for v in request.GET.get("hold", "").split(",")
            if v.strip().isdigit()}
    # **마스크를 요구하지 않는다.** 옛 상자는 전부 마스크가 있어 이 줄이 걸러
    # 내는 것이 없었지만, 새 검출기가 들인 상자는 아직 없다 — 그것을 묻는 것은
    # "이 안에 지느러미가 있나" 지 "이 윤곽이 맞나" 가 아니라서 마스크가 필요
    # 없다. `_tile`·`rules.resolve` 는 마스크 없이도 돈다.
    base = (Box.objects.select_related("crop")
            .prefetch_related("masks", "reviews"))

    if mode == "new":
        # **새 검출기가 낸 상자 중 아직 안 본 것.** 확신이 높은 것부터 낸다 —
        # 거기가 "진짜인데 옛 검출기가 놓친 것" 이 가장 많이 모여 있고, 문턱을
        # 어디로 잡을지도 위에서부터 훑어야 보인다.
        qs = (base.exclude(source="yolov5").filter(reviews__isnull=True)
              .distinct().order_by("-conf", "id"))
        ids = None
    elif mode == "stale":
        # **판정이 본 마스크와 지금 현재가 다른 것.** 엔진을 갈아 끼우면
        # `verdict`("이 마스크가 맞다")가 그때 그 마스크에 대한 말이 되어
        # 낡는다 (`models.py` 의 "엔진에 딸린 판정"). `Review.mask_id` 가
        # 어느 마스크를 보고 내린 판정인지 들고 있어 정확히 집어낼 수 있다.
        # `cls` 는 다시 안 받아도 된다 — 상자 안에 무엇이 있나는 엔진과 무관하다.
        cur = dict(Mask.objects.filter(is_current=True)
                   .values_list("box_id", "id"))
        latest = {}
        for bid, mid, at in (Review.objects.order_by("id")
                             .values_list("box_id", "mask_id", "at")):
            latest[bid] = (mid, at)
        ids = [b for b, (mid, _) in sorted(latest.items(), key=lambda x: x[1][1])
               if b in cur and mid != cur[b]]
        qs = None
    elif mode == "noshape":
        # **윤곽을 말할 수 없는 상자.** 판정은 붙었는데(그래서 `검토할 것`·
        # `새 검출` 에서 빠졌고) 마스크도 사람 윤곽도 없다 — 분할 모델이 이
        # 크롭에서 아무것도 못 냈고 사람도 아직 안 그린 자리다.
        #
        # **여기 말고는 다시 만날 길이 없었다.** `새 검출` 은 판정이 붙어서,
        # `엔진 바뀜` 은 마스크가 없어서, `교정 대기` 는 `verdict='fix'` 가
        # 아니라서 안 걸린다. 그동안 화면은 "남은 일 없다" 고 말했는데
        # `export_yolo` 는 **그 상자가 든 크롭을 통째로 뺐다** — 같은 크롭에
        # 걸친 남의 상자까지 함께. `교정 대기` 를 만든 것과 똑같은 자리다.
        #
        # **등지느러미부터, 그다음 확신이 높은 것부터.** 값이 가장 큰 것이
        # 먼저 와야 도중에 멈춰도 얻는 것이 있다.
        cur = set(Mask.objects.filter(is_current=True)
                  .values_list("box_id", flat=True))
        latest = {}
        for bid, cls, poly in (Review.objects.order_by("id")
                               .values_list("box_id", "cls", "polygon")):
            latest[bid] = (cls, poly)
        want = [b for b, (cl, poly) in latest.items()
                if cl and cl != "none" and not poly and b not in cur]
        conf = dict(Box.objects.filter(id__in=want).values_list("id", "conf"))
        ids = sorted(want, key=lambda b: (latest[b][0] != "fin",
                                          -(conf.get(b) or 0)))
        qs = None
    elif mode == "stuck":
        # **고쳐야 한다고 해 놓고 안 고친 것.** 판정이 붙어 있어 `todo` 에 안
        # 나오고, 자료로도 안 나간다 — 여기 말고는 다시 만날 길이 없다.
        latest = (Review.objects.values("box_id")
                  .annotate(mid=Max("id")).values_list("mid", flat=True))
        ids = list(Review.objects.filter(
            id__in=latest, verdict="fix", polygon="", base_line="")
            .exclude(cls="none").exclude(cls="")
            .order_by("at", "id").values_list("box_id", flat=True))
        qs = None
    elif mode == "done":
        ids = _reviewed_order(cls)
        qs = None
    else:
        # **`todo` 는 마스크가 있는 것만이다** — 여기서 묻는 것에 "이 윤곽이
        # 맞나" 가 들어 있다. 마스크 없는 새 상자는 `new` 로 간다. 둘을 한
        # 대기열에 섞으면 서로 다른 두 물음이 한 칸에서 뒤섞인다.
        ids = None
        qs = (base.filter(masks__is_current=True, reviews__isnull=True)
              .distinct().order_by("id"))

    # **보류한 것은 목록에서 빠진다** — 쪽수를 세기 전에 뺀다.
    if hold:
        if qs is not None:
            qs = qs.exclude(id__in=hold)
        else:
            ids = [i for i in ids if i not in hold]
    total = qs.count() if qs is not None else len(ids)

    # **쪽수를 먼저 정하고 자른다.** 범위를 넘긴 요청은 마지막 쪽으로 접어
    # 준다 — 안 그러면 빈 격자가 뜨는데 화면이 "끝이다" 인지 "뭔가 잘못됐다"
    # 인지 말해 주지 않는다. `todo` 는 검토할수록 쪽수가 줄어서 화면이 든 값이
    # 늘 조금 낡아 있고, 끝에서 `]` 를 한 번 더 누르면 곧바로 걸린다.
    pages = max((total + n - 1) // n, 1)
    page = min(page, pages)
    lo = (page - 1) * n

    if ids is None:
        tiles = _tiles_for(qs[lo:lo + n * 2])[:n]
    else:
        want = ids[lo:lo + n]
        by_id = {b.id: b for b in base.filter(id__in=want)}
        tiles = _tiles_for([by_id[i] for i in want if i in by_id])

    return JsonResponse({
        "tiles": tiles, "mode": mode, "page": page, "total": total,
        "pages": pages, "cls": cls,
    })


@require_POST
@transaction.atomic
def save(request):
    """판정을 **쌓는다**. 덮어쓰지 않는다 — 고쳐 매긴 자취가 남아야 하고,
    나중에 여러 사람의 판정을 합의로 모을 수 있어야 한다.
    """
    body = json.loads(request.body or "{}")
    valid_cls = {c for c, _ in CLASSES}
    valid_edges = {e for e, _ in EDGES}
    valid_partial = {p for p, _ in BASE_PARTIAL}
    valid_facing = {f for f, _ in FACING}
    ids = [it["box_id"] for it in body.get("items", [])]
    crops = {c.box_id: c for c in Crop.objects.filter(box_id__in=ids)}
    # **마스크가 없는 상자에는 `verdict` 를 안 적는다.** "이 마스크가 맞나" 는
    # 마스크가 있어야 물을 수 있는 말이다 (`models.py` 의 "엔진에 딸린 판정").
    # 새 검출기가 들인 상자는 아직 윤곽이 없고, 거기서 사람이 답한 것은
    # `cls` 하나 — "여기 정말 지느러미가 있나" 뿐이다.
    #
    # `ok` 를 적어 두면 `label_of` 가 그것을 **양성**이라 하고, 진행상황이
    # 분할 자료가 다 된 것처럼 말한다. 비워 두면 `PENDING` 이라 남은 일로
    # 잡히고, 나중에 `infer` 가 마스크를 얹으면 **`엔진 바뀜` 대기열에
    # 저절로 뜬다** — 그때 윤곽을 판정하면 된다.
    masked = set(Mask.objects.filter(box_id__in=ids, is_current=True)
                 .values_list("box_id", flat=True))
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
            if it["box_id"] not in masked:
                verdict = ""        # 물을 수 없는 말이다 (위 주석)
            elif verdict not in ("ok", "fix"):
                return JsonResponse({"error": f"모르는 판정: {verdict}"}, status=400)

        facing = it.get("facing") or ""
        if facing not in valid_facing:
            return JsonResponse({"error": f"모르는 앞쪽 표시: {facing}"},
                                status=400)
        partial = it.get("base_partial") or ""
        if partial not in valid_partial:
            return JsonResponse({"error": f"모르는 아래경계 표시: {partial}"},
                                status=400)
        base_str = poly_str = ""
        crop = crops.get(it["box_id"])
        # **윗윤곽 교정본도 사람이 그린 것만 저장한다** — 밑동과 같은 규칙이다.
        # 안 그리면 `mask.polygon` 이 그대로 쓰이고(`rules.resolve`), 그래야
        # 나중에 마스크를 다시 뽑았을 때 옛 윤곽이 사람의 것인 척 남지 않는다.
        if it.get("polygon_edited") and crop is not None \
                and len(it.get("polygon") or []) >= 3:
            poly_str = geometry.dumps(geometry.to_orig(it["polygon"], crop))
        # **사람이 끈 것만 저장한다.** 안 건드린 제안까지 적으면 나중에 제안
        # 규칙을 고쳐도 옛 제안이 사람의 판단인 척 남는다.
        if it.get("base_moved") and crop is not None and len(it.get("base") or []) == 2:
            base_str = geometry.dumps(geometry.to_orig(it["base"], crop))

        Review.objects.create(
            box_id=it["box_id"], mask_id=it.get("mask_id"), cls=cls,
            verdict=verdict, edges=edges, base_line=base_str, polygon=poly_str,
            base_partial=partial, facing=facing,
            reviewer=request.user if request.user.is_authenticated else None)
        n += 1
    return JsonResponse({"saved": n, "progress": dict(rules.progress())})


# **CSRF 쿠키를 여기서 심는다.** 이 화면은 끌어 넣을 때마다 POST 하는데,
# 쿠키가 없으면 서버가 403 을 내고 화면은 "서버가 403 을 냈다" 만 말한다 —
# 무엇이 없는지는 안 말한다. 격자 화면(`index`)이 이미 같은 이유로 붙어 있다.
@ensure_csrf_cookie
@require_GET
def reid(request):
    """**사람이 손으로 분류하는 자리.** 지느러미를 상자에 끌어 넣는다.

    군집을 먼저 만들어 보였는데 쓸모가 없었다 — 묶음이 밝기로 뭉친 정도가
    68%였고, 특이한 개체는 남과 안 닮아 혼자 남아서 오히려 안 보였다.
    **모델이 판단하고 사람이 고치는 것이 아니라, 사람이 판단하고 모델은
    순서만 돕는다.**

    정렬 셋을 준다. `sim` 은 임베딩으로 닮은 것끼리 한 줄로 편 것이다 — 이웃이
    닮아 있어야 눈이 덜 움직인다. `rough` 는 결각이 깊은 것부터. `day` 는 날짜순.

    상자 하나가 `Individual` 하나다. **이름은 나중에 붙인다** — 먼저 모아 놓고
    보아야 그 개체가 누구인지 정할 수 있고, 이름부터 물으면 손이 멎는다.
    """
    from finseg.models import Individual

    root = Path(settings.FIN_REID)
    items_f = root / "items.json"
    ready = items_f.exists() and ((root / "look").is_dir()
                                  or (root / "chips").is_dir())
    items, boxes = [], []
    if ready:
        items = json.loads(items_f.read_text(encoding="utf-8")).get("items", [])
        # 지금 어느 상자에 들어 있나 — **가장 늦은 판정이 이긴다**
        latest = {}
        for box_id, ind in (Identification.objects.order_by("id")
                            .values_list("box_id", "individual_id")):
            latest[box_id] = ind
        # **이미 "지느러미가 아니다" 라고 한 것은 표를 달고 보인다.** 격자에서
        # 지워 버리면 방금 무엇을 했는지 확인할 길이 없고, 새로고침하면 그
        # 판정이 화면에서 사라져 같은 것을 또 누르게 된다
        judged = {}
        for box_id, cls in (Review.objects.filter(box_id__in=[i["id"] for i in items])
                            .exclude(cls="").order_by("id")
                            .values_list("box_id", "cls")):
            judged[box_id] = cls
        for it in items:
            it["in"] = latest.get(it["id"])
            c = judged.get(it["id"])
            if c and c != "fin":
                it["notfin"] = c
        # **만든 순서로 낸다.** 기본은 이름순인데(`Individual.Meta`) 그러면
        # `상자 10` 이 `상자 2` 앞에 오고, 무엇보다 **이름을 고치는 순간 그
        # 상자가 목록에서 튀어 다닌다** — 방금 이름 붙인 것을 눈으로 다시
        # 찾아야 한다. 분류는 만든 순서대로 쌓이는 일이라 그 순서가 맞다.
        # **보류함은 늘 있다.** 어디에 넣을지 모르겠는 것이 반드시 나오는데,
        # 아무 상자에나 넣으면 그 상자가 오염되고 미분류로 두면 다음에 또
        # 같은 고민을 처음부터 한다
        hold, _ = Individual.objects.get_or_create(
            holding=True, defaults={"name": "보류함"})
        boxes = [{"id": i.id, "name": i.name, "rep": i.rep_id,
                  "hold": i.holding,
                  "n": sum(1 for v in latest.values() if v == i.id)}
                 for i in Individual.objects.order_by("id")]
    # **키는 검토 화면과 같은 것을 쓴다** (`models.CLASS_KEYS`) — 두 화면에서
    # 다른 키를 누르게 하면 손이 헷갈리고, 그 순간 잘못된 분류가 남는다.
    # 여기서는 `fin` 만 뺀다 — 격자에 있다는 것 자체가 "지느러미로 봤다" 이므로
    notfin = {k: c for c, k in CLASS_KEYS.items() if c != "fin"}
    # **상자만 따로 창으로 뽑는다.** 격자와 상자 목록을 두 화면에 나눠 놓으면
    # 끌어 넣는 거리가 짧아지고 목록을 굴릴 일이 없다. 같은 템플릿을 쓰되
    # 격자를 감춘다 — 끌기·놓기·저장 코드를 그대로 물려받으려는 것이다
    return render(request, "review/reid.html", {
        "only_boxes": request.GET.get("only") == "boxes",
        "ready": ready,
        "notfin_keys": json.dumps(notfin, ensure_ascii=False),
        "notfin_names": json.dumps(dict(CLASSES), ensure_ascii=False),
        "items": json.dumps(items, ensure_ascii=False),
        "boxes": json.dumps(boxes, ensure_ascii=False),
        "n_items": len(items),
    })


@require_GET
def catalog(request):
    """**지금까지 붙인 개체를 한 장에 편다.**

    `/reid` 은 일하는 자리라 상자를 하나씩 열어 본다. 그런데 카탈로그가 자라면
    **전체를 한눈에 보아야 하는 순간**이 온다 — 둘이 사실 같은 개체는 아닌지,
    한 상자에 두 개체가 섞이지 않았는지는 나란히 놓고 보아야 보인다.

    개체마다 **가진 조각을 전부** 낸다. 대표 한 장만 보이는 `/reid` 의 상자
    목록과 반대다.

    ## 날짜를 조각마다 적는다

    개체의 증거가 되는 것은 **날을 건너뛴 것**이다. 같은 날 연속 프레임은 짝을
    수백 개 만들어도 증거로는 약하다. 그래서 날이 바뀌는 자리에 금을 긋고
    `날 n` 을 함께 낸다 — 한 날짜뿐인 개체는 그것만으로 눈에 띈다.

    **보류함은 개체가 아니다** (`Individual.holding`). `reid.catalog()` 가 안
    세므로 여기에도 안 나온다.
    """
    from finseg import reid
    from finseg.models import Individual

    cat = reid.catalog()
    names = dict(Individual.objects.values_list("id", "name"))
    reps = dict(Individual.objects.values_list("id", "rep_id"))
    day = dict(Box.objects.filter(id__in=[b for v in cat.values() for b in v])
               .values_list("id", "image__obsdate"))
    facing = {}
    for b in Box.objects.filter(
            id__in=[b for v in cat.values() for b in v]).prefetch_related(
            "reviews", "masks"):
        facing[b.id] = rules.resolve(b).get("facing") or ""

    rows = []
    for ind, boxes in cat.items():
        # 날짜순으로 편다 — 같은 날 것이 붙어 있어야 "이 날 이 개체" 가 읽힌다
        fins = sorted(boxes, key=lambda b: (str(day.get(b) or ""), b))
        days = sorted({str(day.get(b)) for b in fins if day.get(b)})
        rows.append({
            "id": ind, "name": names.get(ind, f"#{ind}"), "rep": reps.get(ind),
            "n": len(fins), "days": len(days),
            "span": f"{days[0]} ~ {days[-1]}" if len(days) > 1 else
                    (days[0] if days else ""),
            "fins": [{"id": b, "day": str(day.get(b) or ""),
                      "facing": facing.get(b, "")} for b in fins],
        })
    # **날을 건너뛴 개체부터.** 그것이 re-ID 에 값을 하는 쪽이고, 한 날짜뿐인
    # 개체는 아직 "그 날 그 무리" 이지 개체의 증거가 아니다
    rows.sort(key=lambda r: (-r["days"], -r["n"]))
    return render(request, "review/catalog.html", {
        "rows": rows,
        "n_ind": len(rows),
        "n_fin": sum(r["n"] for r in rows),
        "n_cross": sum(1 for r in rows if r["days"] >= 2),
    })


@require_POST
@transaction.atomic
def reid_box(request):
    """상자를 새로 만들거나 이름을 고친다.

    **이름 없이도 만들어진다.** 먼저 모아 놓고 보아야 누구인지 정할 수 있는데,
    이름부터 물으면 거기서 손이 멎는다. 임시 이름은 겹치지 않게 붙인다.
    """
    from finseg.models import Individual

    body = json.loads(request.body or "{}")
    ind_id = body.get("id")
    name = (body.get("name") or "").strip()
    if ind_id and "rep" in body:
        # **대표를 고른다.** 상자에 든 것을 다 보이면 목록이 길어지고, 아무거나
        # 하나를 보이면 하필 흐린 것이 대표가 되어 누구인지 알아볼 수 없다
        ind = Individual.objects.filter(id=ind_id).first()
        if ind is None:
            return JsonResponse({"error": "그런 상자가 없다"}, status=404)
        rep = body["rep"]
        ind.rep_id = int(rep) if rep else None
        ind.save(update_fields=["rep"])
        return JsonResponse({"id": ind.id, "rep": ind.rep_id})
    if ind_id:
        ind = Individual.objects.filter(id=ind_id).first()
        if ind is None:
            return JsonResponse({"error": "그런 상자가 없다"}, status=404)
        if not name:
            return JsonResponse({"error": "이름이 비었다"}, status=400)
        if Individual.objects.filter(name=name).exclude(id=ind.id).exists():
            return JsonResponse({"error": f"'{name}' 은 이미 있다"}, status=400)
        ind.name = name
        ind.save(update_fields=["name"])
        return JsonResponse({"id": ind.id, "name": ind.name})
    if not name:
        n = Individual.objects.count() + 1
        while Individual.objects.filter(name=f"상자 {n}").exists():
            n += 1
        name = f"상자 {n}"
    ind, made = Individual.objects.get_or_create(name=name)
    return JsonResponse({"id": ind.id, "name": ind.name, "made": made})


def _reid_pool():
    """격자의 조각들과 그것을 묶는 데 필요한 축들. **자료를 읽는 자리는 여기 하나**."""
    import numpy as np

    root = Path(settings.FIN_REID)
    items = json.loads((root / "items.json").read_text(encoding="utf-8"))["items"]
    ids = np.array([it["id"] for it in items])
    day = np.array([it["day"] for it in items])
    fac = np.array([it["facing"] for it in items])
    # **어느 임베딩이냐가 묶음의 질을 정한다.** DINOv2 가 거리 ≤0.06 에서
    # 77% 대 62%(ResNet18)로 앞선다 — 있으면 그것을 쓴다
    for name in ("emb-dinov2.npz", "emb.npz"):
        f = root / name
        if f.exists():
            z = np.load(f)
            if (z["box_id"] == ids).all():
                emb = z["emb"]
                break
    else:
        return None
    path = dict(Box.objects.filter(id__in=[int(b) for b in ids])
                .values_list("id", "image__path"))
    frame = np.array([int(m.group(1))
                      if (m := re.search(r"(\d+)\.[A-Za-z]+$", path.get(int(b)) or ""))
                      else -1 for b in ids])
    # **배운 분류기가 있으면 그것을 쓴다.** 같은 갈래에서 재 보니 kNN 25.5% 대
    # 분류기 44.5% 로 거의 두 배다. 없으면 kNN 으로 떨어진다 — 화면이 멈추지는
    # 않되, 무엇으로 낸 것인지는 응답이 말한다
    cls = None
    f = root / "cls-dinov2.npz"
    if f.exists():
        z = np.load(f)
        cls = {side: (z[f"{side}_W"], z[f"{side}_b"], z[f"{side}_cls"])
               for side in ("left", "right") if f"{side}_W" in z}
        cls["_n_labeled"] = int(z["n_labeled"][0]) if "n_labeled" in z else 0
    return {"ids": ids, "day": day, "fac": fac, "emb": emb, "frame": frame,
            "cls": cls, "pos": {int(b): i for i, b in enumerate(ids)}}


def _score(P, g, cat_idx, k=5):
    """묶음 하나 → [(개체, 점수, 확률인가)…]. 분류기가 있으면 그것을 쓴다.

    **확률과 닮음은 다른 값이다.** 분류기는 클래스 위의 확률을 내므로 그대로
    읽으면 되고(합이 1), 코사인 닮음은 다 0.9대라 절대값이 뜻이 없어 순위로만
    읽어야 한다. 화면이 둘을 구별해 보여야 사람이 잘못 믿지 않는다.
    """
    import numpy as np
    from finseg import reid as R

    cls = P.get("cls")
    side = str(P["fac"][g[0]])
    if cls and side in cls:
        W, b, classes = cls[side]
        X = P["emb"] / np.maximum(np.linalg.norm(P["emb"], axis=1, keepdims=True), 1e-9)
        # **묶음은 로짓을 평균한다** — 표를 모으는 것이 낱장보다 낫다
        logit = (X[g] @ W.T + b).mean(0)
        e = np.exp(logit - logit.max())
        prob = e / e.sum()
        order = np.argsort(-prob)[:k]
        # 카탈로그에서 사라진 개체(뺐거나 합친 것)는 내지 않는다
        return [(int(classes[i]), float(prob[i]), True)
                for i in order if int(classes[i]) in cat_idx], True
    return [(ind, sc, False)
            for ind, sc in R.suggest(g, P["emb"], P["day"], P["fac"], cat_idx, k)], False


@require_GET
def reid_groups(request):
    """**아직 안 넣은 조각들을 묶어서 내놓는다** — 한 장씩 묻지 않으려고.

    실측: 5장을 묶으면 top-1 이 22.7% → 40%, top-5 는 68% → **80%** 가 되고
    사람의 판단 한 번이 5장을 덮는다. 묶는 근거는 `reid.group_links` 한 곳이다.
    """
    import numpy as np
    from finseg import reid as R

    P = _reid_pool()
    if P is None:
        return JsonResponse({"error": "임베딩이 없다 — `reid_chips` 를 먼저 돌릴 것"},
                            status=400)
    from finseg.models import Individual

    latest = {}
    for box_id, ind in (Identification.objects.order_by("id")
                        .values_list("box_id", "individual_id")):
        latest[box_id] = ind
    free = np.array([i for i, b in enumerate(P["ids"]) if int(b) not in latest])
    groups = [g for g in R.group_links(free, P["emb"], P["day"], P["fac"], P["frame"])
              if len(g) >= int(request.GET.get("min", 2))]
    cat_idx = {ind: [P["pos"][b] for b in boxes if b in P["pos"]]
               for ind, boxes in R.catalog().items()}
    names = dict(Individual.objects.values_list("id", "name"))
    n = int(request.GET.get("n", 20))
    out = []
    is_cls = False
    for g in groups[:n]:
        sg, is_cls = _score(P, g, cat_idx)
        out.append({
            "boxes": [int(P["ids"][i]) for i in g],
            "day": str(P["day"][g[0]]), "facing": str(P["fac"][g[0]]),
            "suggest": [{"id": ind, "name": names.get(ind, str(ind)),
                         "score": round(sc, 4)} for ind, sc, _ in sg],
        })
    # **무엇으로 낸 점수인지 화면이 알아야 한다** — 확률과 닮음은 읽는 법이 다르다
    unknown = 0
    if P.get("cls"):
        known = set()
        for side in ("left", "right"):
            if side in P["cls"]:
                known |= set(int(x) for x in P["cls"][side][2])
        unknown = len([i for i in cat_idx if i not in known])
    return JsonResponse({"groups": out, "n_free": int(len(free)),
                         "n_groups": len(groups), "prob": bool(is_cls),
                         "unknown": unknown})


@require_POST
def reid_suggest(request):
    """고른 것들을 한 묶음으로 보고 닮은 개체를 낸다 — 사람이 직접 묶었을 때."""
    from finseg import reid as R
    from finseg.models import Individual

    P = _reid_pool()
    if P is None:
        return JsonResponse({"error": "임베딩이 없다"}, status=400)
    body = json.loads(request.body or "{}")
    g = [P["pos"][int(b)] for b in body.get("boxes", []) if int(b) in P["pos"]]
    if not g:
        return JsonResponse({"error": "고른 것이 없다"}, status=400)
    cat_idx = {ind: [P["pos"][b] for b in boxes if b in P["pos"]]
               for ind, boxes in R.catalog().items()}
    names = dict(Individual.objects.values_list("id", "name"))
    sg, is_cls = _score(P, g, cat_idx, k=int(body.get("k", 5)))
    return JsonResponse({"prob": is_cls,
                         "suggest": [{"id": i, "name": names.get(i, str(i)),
                                      "score": round(s, 4)} for i, s, _ in sg]})


@require_POST
@transaction.atomic
def reid_cls_set(request):
    """**격자에서 바로 "이건 지느러미가 아니다" 라고 말한다.**

    옛 상자를 2,877개 길어 오면서 몸통·꼬리 같은 것이 섞여 들어왔다. 분할
    엔진이 걸러 주기는 하는데 어휘가 셋뿐이라(`fin`·`dolphin`·`nonfin`) 놓치는
    것이 있고, 그것이 격자에 앉아 있으면 **사람이 볼 때마다 같은 판단을 다시
    한다.**

    판정은 `Review` 로 남는다 — 검토 화면이 쓰는 바로 그 표다. 그래야
    `rules.resolve` 가 그 분류를 내고 `reid.usable` 이 다음 격자에서 걸러낸다.
    **여기서 따로 표를 만들지 않는다** — 두 곳에 두면 화면이 거른 것과 자료로
    나가는 것이 갈린다.

    `verdict` 는 안 적는다. 그것은 "이 마스크가 맞나" 를 묻는 말이라 분류와
    다른 축이고, 여기서는 묻지 않았다.
    """
    body = json.loads(request.body or "{}")
    cls = body.get("cls", "")
    if cls not in dict(CLASSES):
        return JsonResponse({"error": f"그런 분류가 없다: {cls}"}, status=400)
    box_ids = [int(b) for b in body.get("boxes", [])]
    if not box_ids:
        return JsonResponse({"error": "고른 것이 없다"}, status=400)
    user = request.user if request.user.is_authenticated else None
    masks = {m.box_id: m for m in Mask.objects.filter(box_id__in=box_ids,
                                                      is_current=True)}
    Review.objects.bulk_create([
        Review(box_id=b, mask=masks.get(b), cls=cls, reviewer=user)
        for b in box_ids])
    return JsonResponse({"saved": len(box_ids), "cls": cls})


@require_POST
@transaction.atomic
def reid_assign(request):
    """지느러미를 상자에 넣거나 뺀다. **쌓는다, 덮어쓰지 않는다.**

    `individual` 이 없으면 뺀 것이다(`NULL`). 아니라고 말한 것도 자료이고,
    옮겨 다닌 자취가 있어야 나중에 판단이 언제 바뀌었는지 잴 수 있다.
    """
    from finseg.models import Individual

    body = json.loads(request.body or "{}")
    ind_id = body.get("individual")
    ind = None
    if ind_id:
        ind = Individual.objects.filter(id=ind_id).first()
        if ind is None:
            return JsonResponse({"error": "그런 상자가 없다"}, status=404)
    box_ids = [int(b) for b in body.get("boxes", [])]
    if not box_ids:
        return JsonResponse({"error": "고른 것이 없다"}, status=400)
    user = request.user if request.user.is_authenticated else None
    Identification.objects.bulk_create([
        Identification(box_id=b, individual=ind, source=body.get("source", "hand"),
                       reviewer=user) for b in box_ids])
    n = Identification.objects.filter(individual=ind).values("box").distinct().count() \
        if ind else 0
    return JsonResponse({"saved": len(box_ids),
                         "individual": ind.id if ind else None, "n": n})


@require_GET
def reid_chip(request, box_id):
    """조각 한 장. 저장소 밖이라 정적 서빙이 없어 여기서 낸다.

    **사람이 보는 것과 모델이 먹는 것이 다르다.** 사람에게는 색과 배경이 있는
    큰 그림(`look/`)을 낸다 — 지느러미가 물에 얼마나 잠겼는지, 빛이 어느
    쪽에서 오는지, 옆에 다른 개체가 있는지가 다 판단에 든다. 모델이 먹는
    조각(`chips/`)은 배경을 지운 회색조이고, 그것은 **그날 바다를 안 배우게**
    하려는 것이다.

    정렬은 둘이 같다(`reid.frame`). 사람이 "이건 달라" 라고 한 것이 모델에게는
    다른 자리였다면 그 판정을 못 쓴다.
    """
    root = Path(settings.FIN_REID)
    for sub_dir, ext, mime in (("look", "jpg", "image/jpeg"),
                               ("chips", "png", "image/png")):
        f = root / sub_dir / f"{int(box_id):08d}.{ext}"
        if f.exists():
            return FileResponse(open(f, "rb"), content_type=mime)
    raise Http404("그 조각이 없다")


@require_GET
def detect(request):
    """**브라우저에서 검출기를 돌린다** — 사진을 떨구면 상자를 그린다.

    서버 GPU 를 안 쓴다. 2080ti 는 학습이 잡고 있는 시간이 길고, 그동안 서버
    추론은 줄을 서야 한다 — `onnxruntime-web` 이 보는 사람의 GPU(WebGPU)를
    쓰면 그와 무관해진다.

    **NMS 는 JS 가 한다.** 정렬하고 반복문으로 지우는 일이라 GPU 셰이더에 안
    맞아서, 모델 안에 넣으면 그 연산자만 CPU 로 떨어져 이득이 깎인다. 밖에
    두면 **문턱을 다시 안 돌리고 바꿔 볼 수 있다**는 값이 따라온다 — 이
    프로젝트에서 `conf` 를 어디로 둘지가 계속 문제였다.

    규칙의 원본은 `finseg/onnxdet.py` 다. 여기 JS 는 그것을 옮긴 것이고,
    **어긋나면 파이썬 쪽을 고치고 여기를 따라오게 한다.**

    받아 온 것 둘이 없으면 화면이 왜 없는지 말한다 — 조용히 빈 화면을 내면
    무엇이 빠졌는지 알 길이 없다.
    """
    model = Path(settings.BASE_DIR) / "static" / "models" / "detect-v2.onnx"
    # **분할은 있으면 얹고 없으면 상자만 그린다.** 두 단이라 검출만으로도
    # 화면이 돌아가야 한다 — 없다고 아무것도 안 나오면 무엇이 빠졌는지 모른다
    seg = Path(settings.BASE_DIR) / "static" / "models" / "seg-v2.onnx"
    ort = Path(settings.BASE_DIR) / "static" / "vendor" / "ort"
    # **`.mjs` 하나만 보고 있으면 안 된다.** 묶음은 실행 중에 `.wasm` 을 이름으로
    # 불러오고, 그 이름이 판마다 바뀐다(1.27 은 `jsep` 이 아니라 `asyncify` 다).
    # 하나라도 없으면 화면이 "no available backend found" 만 내는데 그것만 보고는
    # 무엇이 빠졌는지 알 수 없다 — 여기서 미리 세어 이름을 대 준다.
    need = ["ort.webgpu.bundle.min.mjs",
            "ort-wasm-simd-threaded.asyncify.mjs",
            "ort-wasm-simd-threaded.asyncify.wasm",
            "ort-wasm-simd-threaded.mjs",
            "ort-wasm-simd-threaded.wasm"]
    missing = [f for f in need if not (ort / f).exists()]
    return render(request, "review/detect.html", {
        "model_ok": model.exists(),
        "seg_ok": seg.exists(),
        "seg_imgsz": onnxdet.SEG_IMGSZ,
        "seg_nc": onnxdet.SEG_NC,
        "crop_pad": onnxdet.CROP_PAD,
        "mask_thres": onnxdet.MASK_THRES,
        "ort_ok": not missing,
        "missing": missing,
        "imgsz": onnxdet.IMGSZ,
        "conf": onnxdet.CONF,
        "iou": onnxdet.IOU,
        "pad": onnxdet.PAD,
    })


@require_GET
def edit(request, box_id):
    """윗윤곽 고치기 — **격자에서는 못 한다.** 190px 칸에서 꼭짓점을 집는 것은
    밑동 점이 화면에서 2.6px 였던 것과 같은 문제다. 크게 띄우고 확대까지 되어야
    비로소 고칠 수 있다.

    격자와 같은 `_tile` 을 쓴다 — 화면에 보이는 것이 곧 저장될 것이어야 한다.
    """
    box = get_object_or_404(
        Box.objects.select_related("crop", "image")
        .prefetch_related("masks", "reviews"), pk=box_id)
    crop = getattr(box, "crop", None)
    if crop is None:
        raise Http404("크롭이 없는 상자다")
    img = box.image
    src = settings.FIN_PHOTOS / img.path
    # **원본이 있으면 그것을 깐다.** 640 크롭은 원본의 300px 언저리를 늘린 것이라
    # 확대할수록 뭉개진다. 원본을 크롭 좌표에 얹으면 좌표계는 그대로 두고 —
    # 저장 경로가 안 바뀐다 — 화질이 살아나고 크롭 밖 주변까지 보인다.
    geom = {"x0": crop.x0, "y0": crop.y0, "scale": crop.scale,
            "iw": img.width or 0, "ih": img.height or 0}
    # **프롬프트 상자를 크롭 좌표로 함께 보낸다.** 마스크가 아예 없는 상자에서
    # 사람이 윤곽을 처음부터 그릴 때 출발점이 된다 — 이 화면은 이미 있는 점을
    # 끌고 변 가운데를 눌러 늘리는 식이라, **점이 하나도 없으면 누를 데가 없다.**
    (bx1, by1), (bx2, by2) = geometry.to_crop(
        [(box.x1, box.y1), (box.x2, box.y2)], crop)
    return render(request, "review/edit.html", {
        "tile": json.dumps(_tile(rules.resolve(box), crop), ensure_ascii=False),
        "boxrect": json.dumps([round(bx1, 1), round(by1, 1),
                               round(bx2, 1), round(by2, 1)]),
        "box": box, "img": img,
        "photo": f"/photos/{img.path}" if src.exists() else "",
        "geom": json.dumps(geom),
        "facings": json.dumps(FACING, ensure_ascii=False),
        "classes": json.dumps(CLASSES, ensure_ascii=False),
    })


@require_GET
def home(request):
    """**작업대** — 이 저장소가 무엇을 만들고 있는지 한 장에.

    화면을 그때그때 필요에 따라 만들다 보니 서로 오갈 길이 없었다. 그런데 이
    저장소가 하는 일은 하나다 — **사진 한 장에서 개체 이름까지 가는 사슬**을
    위해 가중치 네 벌(검출·분할·밑동·개체)을 함께 굴리는 것. 그 사슬을 차림표가
    스스로 말해야 새로 오는 사람도, 두 주 뒤의 나도 어디부터 손댈지 안다.

    **각 칸에서 노란 수가 사람이 할 일이다.** 자동으로 도는 것과 사람이 해야
    하는 것을 눈으로 갈라 놓지 않으면, 다 된 것처럼 보이는 화면 뒤에 남은 일이
    쌓인다 (`새 검출` 대기열을 만들 때 겪은 것이다).
    """
    from finseg import rules
    from finseg.models import Individual

    c = rules.progress()
    def latest(kind, like=""):
        """가장 최근 run 의 **가중치 이름**(`runs/<이름>/weights/best.pt` 의 가운데).

        **`kind` 를 틀리면 조용히 하드코딩된 이름이 나간다.** 검출은
        `infer_boxes` 가 `kind="detect"` 로 적고 분할·밑동은 `kind="yolo"` 다.
        그리고 `facebook/sam2.1-…` 처럼 두 토막짜리 모델 이름도 있어 자리를
        세어 꺼내면 터진다 — 꼴이 맞을 때만 꺼낸다.
        """
        qs = Run.objects.filter(kind=kind)
        if like:
            qs = qs.filter(model__contains=like)
        r = qs.order_by("-id").first()
        parts = Path(r.model).parts if r and r.model else ()
        return parts[-3] if len(parts) >= 3 else None

    n_ident = Identification.objects.count()
    cat_n = Individual.objects.filter(holding=False).count()
    pool, filed, held, unjudged = 0, 0, 0, 0
    root = Path(settings.FIN_REID)
    if (root / "items.json").exists():
        items = json.loads((root / "items.json").read_text())["items"]
        pool = len(items)
        latest_id = {}
        for b, i in (Identification.objects.order_by("id")
                     .values_list("box_id", "individual_id")):
            latest_id[b] = i
        # **보류함은 개체가 아니라 자리다** (`Individual.holding`) — `catalog()` 가
        # 안 세는 것을 여기서 세면 "분류가 852장 됐다" 고 말하게 된다.
        # 실제로 그렇게 말하고 있었고, 그중 312장은 아직 답을 못 정한 것이다
        hold = set(Individual.objects.filter(holding=True).values_list("id", flat=True))
        filed = sum(1 for it in items
                    if latest_id.get(it["id"]) and latest_id[it["id"]] not in hold)
        held = sum(1 for it in items if latest_id.get(it["id"]) in hold)
        # **"지느러미가 아니다" 도 답이다.** 그것은 `Review` 로만 남고
        # `Identification` 을 안 만드는데, 그 둘을 안 세면 이미 답한 조각이
        # 영영 "남은 일" 로 남는다 — 이 화면이 막으려는 바로 그 함정의 반대다
        done_cls = set(Review.objects.filter(
            box_id__in=[i["id"] for i in items]).exclude(cls="")
            .exclude(cls="fin").values_list("box_id", flat=True))
        unjudged = sum(1 for it in items
                       if it["id"] not in latest_id and it["id"] not in done_cls)

    stages = [
        {"name": "검출 — 사진에서 지느러미를 찾는다",
         "weights": latest("detect") or latest("yolo", "detect") or "—",
         "why": "옛 YOLOv5 를 대신할 우리 검출기. 재현율의 천장을 여는 자리다.",
         "nums": [("상자", f"{c['상자']:,}", False),
                  ("새 검출", f"{c.get('새검출', 0):,}", False),
                  ("아직 안 본 새 검출", f"{c.get('새검출대기', 0):,}",
                   c.get("새검출대기", 0) > 0)],
         "acts": [("새 검출 보기", "/review?queue=new", c.get("새검출대기", 0) > 0),
                  ("브라우저에서 돌려 보기", "/detect", False)]},
        {"name": "분할 — 지느러미 윤곽을 딴다",
         "weights": latest("yolo", "seg") or "—",
         "why": "윤곽이 곧 개체의 단서다. 사람은 기본값을 두고 예외만 누른다.",
         "nums": [("검토함", f"{c.get('검토함', 0):,} / {c['상자']:,}", False),
                  ("교정 대기", f"{c.get('교정대기', 0):,}", c.get("교정대기", 0) > 0),
                  ("윤곽 없음", f"{c.get('윤곽없음', 0):,}", c.get("윤곽없음", 0) > 0)],
         "acts": [("검토하기", "/review", c.get("교정대기", 0) > 0),
                  ("엔진 비교", "/compare", False)]},
        {"name": "밑동 — 두 삽입점을 찍는다",
         "weights": latest("yolo", "pose") or latest("base") or "—",
         "why": "회전·크기를 지우는 기준선. 이것이 흔들리면 개체가 아니라 각도를 잰다.",
         "nums": [("사람이 그린 밑동",
                   f"{Review.objects.exclude(base_line='').values('box').distinct().count():,}",
                   False)],
         "acts": [("윤곽·밑동 고치기", "/review", False)]},
        {"name": "개체 (re-ID) — 카탈로그의 몇 번인지 답한다",
         "weights": "cls-dinov2" if (root / "cls-dinov2.npz").exists() else "",
         "why": "지금 여기를 판다. 얼린 DINOv2 위의 분류기 — 아는 개체 top-1 44.5%.",
         "nums": [("개체", f"{cat_n}", False),
                  ("격자", f"{pool:,}", False),
                  ("상자에 든 것", f"{filed:,}", False),
                  ("보류함", f"{held:,}", held > 0),
                  ("아직 안 만진 것", f"{unjudged:,}", unjudged > 0)],
         "acts": [("분류하기", "/reid", unjudged > 0),
                  ("카탈로그 보기", "/catalog", False)]},
    ]
    return render(request, "review/home.html", {
        "stages": stages,
        "n_image": f"{Image.objects.count():,}",
        "n_box": f"{Box.objects.count():,}",
        # **`distinct()` 에 정렬이 끼어든다.** `Image.Meta.ordering` 이
        # `path` 를 SELECT 에 넣어 사진마다 한 줄이 된다 — 관찰일이
        # 아니라 사진 수가 나온다. `order_by()` 로 그 정렬을 지운다
        "n_day": Image.objects.order_by().values("obsdate")
                 .distinct().count(),
        "n_review": f"{Review.objects.count():,}",
        "n_ident": f"{n_ident:,}",
    })


@require_GET
def compare(request):
    """엔진 둘을 **눈으로** 비교한다 — `/compare?runs=5,16&date=2019-06-17`

    `eval_masks` 는 평균과 비율을 내지만 **어디서 틀리는지는 말해 주지 않는다.**
    다음 개선점은 늘 나쁜 쪽 꼬리에 있으므로 **낮은 IoU 부터** 보여 준다.

    계산은 `finseg.evaluate` 를 부른다 — 표와 화면이 다른 숫자를 말하면
    어느 쪽을 믿을지 알 수 없다.
    """
    ids = [int(x) for x in request.GET.get("runs", "").replace(",", " ").split()
           if x.strip().isdigit()][:4]
    date = request.GET.get("date", "")
    order = request.GET.get("order", "worst")
    page = max(int(request.GET.get("page", 1)), 1)
    n = min(int(request.GET.get("n", 24)), 60)

    runs = list(Run.objects.filter(id__in=ids)) if ids else []
    runs.sort(key=lambda r: ids.index(r.id))
    boxes = Box.objects.select_related("image").prefetch_related("masks", "reviews")
    if date:
        boxes = boxes.filter(image__obsdate=date)
    crops = {c.box_id: c for c in Crop.objects.all()}
    rows, summary = [], []
    if runs:
        truth = evaluate.truth_for(boxes, crops)
        got = {r.id: {m.box_id: m for m in
                      Mask.objects.filter(run_id=r.id, box_id__in=truth)}
               for r in runs}
        for box_id, tpts in truth.items():
            crop = crops[box_id]
            per = [evaluate.iou(got[r.id].get(box_id), tpts, crop) for r in runs]
            rows.append({
                "box_id": box_id, "crop": f"/crops/{crop.path}", "size": crop.w,
                "truth": [[round(x, 1), round(y, 1)] for x, y in tpts],
                "ious": [round(v, 3) for v in per],
                # 첫 run 을 기준으로 삼는다 — 보통 옛 엔진이 앞에 온다
                "delta": round(per[-1] - per[0], 3) if len(per) > 1 else 0.0,
                "polys": [
                    [[round(x, 1), round(y, 1)] for x, y in geometry.to_crop(
                        geometry.loads(got[r.id][box_id].polygon), crop)]
                    if box_id in got[r.id] else []
                    for r in runs],
            })
        for i, r in enumerate(runs):
            st = evaluate.score([x["ious"][i] for x in rows])
            st.update(run=r.id, kind=r.kind, model=r.model,
                      produced=sum(1 for x in rows if x["polys"][i]))
            summary.append(st)
        key = {"worst": lambda x: x["ious"][0], "best": lambda x: -x["ious"][0],
               "gain": lambda x: -x["delta"], "loss": lambda x: x["delta"],
               "box": lambda x: x["box_id"]}.get(order, lambda x: x["ious"][0])
        rows.sort(key=key)
    pages = max((len(rows) + n - 1) // n, 1)
    page = min(page, pages)
    return render(request, "review/compare.html", {
        "runs": runs, "date": date, "order": order, "page": page, "pages": pages,
        "total": len(rows), "summary": summary,
        "rows": json.dumps(rows[(page - 1) * n:page * n], ensure_ascii=False),
        "all_runs": Run.objects.filter(kind__in=("sam2", "yolo")).order_by("-id")[:20],
    })


@require_GET
def photo(request, box_id):
    """원본 사진 한 장 — **크롭만 봐서는 못 가리는 것들이 있다.**

    640 크롭은 상자 주변을 `pad` 만큼만 담는다. 겹쳐 헤엄치는 무리에서 어느
    지느러미가 어느 몸에 붙었는지, 물보라인지 뒷날인지는 **주변을 봐야** 갈린다.
    그래서 상자를 표시한 원본을 따로 띄운다 — 이 상자는 노랑, 같은 사진의 다른
    상자는 흐리게. 어느 것을 보다 왔는지 잃지 않는다.

    사진은 저장소 밖(`FIN_PHOTOS`)에 있고 NAS 가 안 붙어 있을 수 있다. **없으면
    없다고 말한다** — 깨진 이미지 아이콘은 왜 안 뜨는지 말해 주지 않는다.
    """
    box = get_object_or_404(Box.objects.select_related("image"), pk=box_id)
    img = box.image
    src = settings.FIN_PHOTOS / img.path
    rect = lambda b: {"id": b.id, "x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2}
    others = [rect(b) for b in Box.objects.filter(image=img).exclude(pk=box.pk)]
    return render(request, "review/photo.html", {
        "box": box, "img": img,
        "exists": src.exists(),
        "src": f"/photos/{img.path}",
        "expect": str(src),
        "here": json.dumps(rect(box)),
        "others": json.dumps(others),
        "crop": f"/crops/{box.crop.path}" if hasattr(box, "crop") else "",
    })


@require_GET
def progress(request):
    return JsonResponse(dict(rules.progress()))
