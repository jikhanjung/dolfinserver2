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

from django.conf import settings
from django.db import transaction
from django.db.models import Max
from django.http import Http404, JsonResponse
from django.shortcuts import get_object_or_404, render
from django.utils.html import escape
from django.utils.safestring import mark_safe
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_GET, require_POST

from finseg import baseline, evaluate, geometry, rules
from finseg.models import (BASE_PARTIAL, CLASS_KEYS, CLASSES, EDGES, FACING,
                           Box, Crop, Mask, Review, Run)


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
    if mode not in ("todo", "done", "stuck", "stale", "new"):
        mode = "todo"
    cls = request.GET.get("cls", "")
    if cls not in {c for c, _ in CLASSES}:
        cls = ""
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
        ids, total = None, qs.count()
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
        total, qs = len(ids), None
    elif mode == "stuck":
        # **고쳐야 한다고 해 놓고 안 고친 것.** 판정이 붙어 있어 `todo` 에 안
        # 나오고, 자료로도 안 나간다 — 여기 말고는 다시 만날 길이 없다.
        latest = (Review.objects.values("box_id")
                  .annotate(mid=Max("id")).values_list("mid", flat=True))
        ids = list(Review.objects.filter(
            id__in=latest, verdict="fix", polygon="", base_line="")
            .exclude(cls="none").exclude(cls="")
            .order_by("at", "id").values_list("box_id", flat=True))
        total, qs = len(ids), None
    elif mode == "done":
        ids = _reviewed_order(cls)
        total, qs = len(ids), None
    else:
        # **`todo` 는 마스크가 있는 것만이다** — 여기서 묻는 것에 "이 윤곽이
        # 맞나" 가 들어 있다. 마스크 없는 새 상자는 `new` 로 간다. 둘을 한
        # 대기열에 섞으면 서로 다른 두 물음이 한 칸에서 뒤섞인다.
        ids = None
        qs = (base.filter(masks__is_current=True, reviews__isnull=True)
              .distinct().order_by("id"))
        total = qs.count()

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
    return render(request, "review/edit.html", {
        "tile": json.dumps(_tile(rules.resolve(box), crop), ensure_ascii=False),
        "box": box, "img": img,
        "photo": f"/photos/{img.path}" if src.exists() else "",
        "geom": json.dumps(geom),
        "facings": json.dumps(FACING, ensure_ascii=False),
        "classes": json.dumps(CLASSES, ensure_ascii=False),
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
