#!/usr/bin/env python3
"""마스크 검토 — 격자에서 고르고, 필요한 것만 다시 그린다.

    uvicorn finseg.review.app:app --host 0.0.0.0 --port 8900

**한 사람이 쓰는 도구다.** 인증도 권한도 진행상태 배정도 없다 — 그런 것이
필요해지는 것은 시민과학으로 넘어갈 때이고, 그때는 이 저장소의 일이 아니다.

## 격자에서 무엇을 하나

거의 다 통과다. 상자가 이미 "여기 지느러미가 있다" 고 말했고 SAM2 는 그 안의
것 하나를 딸 뿐이라, 사람이 하는 일은 **틀린 것을 집어내는 것**이다. 그래서
기본값이 `ok` 이고 누르는 것이 예외다.

    누르지 않음 → ok        마스크가 맞다
    한 번        → not_fin  상자 안에 지느러미가 없다 (옛 YOLOv5 의 오검출)
    두 번        → fix      지느러미는 맞는데 윤곽이 틀렸다 → 교정 대기열
    세 번        → ok 로 돌아온다

`not_fin` 과 `fix` 를 가르는 이유는 `db.py` 에 적었다 — 앞은 엔진을 갈아도
살아남고 뒤는 다시 받아야 한다.
"""
import json
import os
import sys
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from finseg import crops as fcrops  # noqa: E402
from finseg import db as fdb        # noqa: E402
from finseg import rules            # noqa: E402

HERE = Path(__file__).resolve().parent
CROPS = Path(os.environ.get("FIN_CROPS", "crops")).resolve()
REVIEWER = os.environ.get("FIN_REVIEWER", os.environ.get("USER", "?"))

app = FastAPI(title="finseg review")
app.mount("/crops", StaticFiles(directory=CROPS), name="crops")
templates = Jinja2Templates(directory=str(HERE / "templates"))


def conn():
    return fdb.connect()


def _tile(row, crop):
    """격자 한 칸에 필요한 것. **폴리곤은 크롭 좌표로 바꿔 보낸다.**

    DB 는 원본 좌표로 들고 있고 화면은 크롭을 본다. 이 변환이 한 군데(`crops`)
    에만 있어야 저장할 때 되돌리는 식과 어긋나지 않는다.
    """
    poly = fdb.loads_polygon(row["polygon"]) if row["polygon"] else []
    return {
        "box_id": row["box_id"],
        "mask_id": row["mask_id"],
        "crop": f"/crops/{crop['path']}",
        "size": crop["w"],
        "conf": row["conf"],
        "verdict": row["verdict"],
        "points": [[round(x, 1), round(y, 1)]
                   for x, y in fcrops.to_crop(poly, crop)],
    }


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "grid.html",
                                      {"reviewer": REVIEWER})


@app.get("/api/batch")
def batch(n: int = 24, redo: bool = False):
    """다음 묶음. `redo` 면 이미 판정한 것도 다시 보여 준다."""
    c = conn()
    where = "WHERE m.id IS NOT NULL"
    if not redo:
        where += " AND r.id IS NULL"
    out = []
    for row in rules.box_states(c, where + " ORDER BY b.id"):
        crop = c.execute("SELECT * FROM crop WHERE box_id=?",
                         (row["box_id"],)).fetchone()
        if crop is None:
            continue
        out.append(_tile(row, crop))
        if len(out) >= n:
            break
    return {"tiles": out, "reviewer": REVIEWER}


@app.post("/api/review")
async def save(request: Request):
    """판정을 쌓는다. **덮어쓰지 않는다** — 고쳐 매긴 자취가 남아야 한다."""
    body = await request.json()
    items = body.get("items", [])
    c = conn()
    n = 0
    for it in items:
        v = it.get("verdict")
        if v not in fdb.VERDICTS:
            return JSONResponse({"error": f"모르는 판정: {v}"}, status_code=400)
        c.execute("INSERT INTO review (box_id, mask_id, verdict, polygon,"
                  " reviewer, at) VALUES (?,?,?,?,?,?)",
                  (it["box_id"], it.get("mask_id"), v, it.get("polygon"),
                   body.get("reviewer") or REVIEWER, fdb.now()))
        n += 1
    c.commit()
    return {"saved": n, "progress": dict(rules.progress(c))}


@app.get("/api/progress")
def progress():
    return dict(rules.progress(conn()))


@app.get("/api/fixqueue")
def fixqueue(n: int = 50):
    """`fix` 라 표시됐는데 아직 다시 그리지 않은 것."""
    c = conn()
    out = []
    for row in rules.box_states(
            c, "WHERE r.verdict='fix' AND (r.polygon IS NULL OR r.polygon='')"
               " ORDER BY b.id"):
        crop = c.execute("SELECT * FROM crop WHERE box_id=?",
                         (row["box_id"],)).fetchone()
        if crop:
            out.append(_tile(row, crop))
        if len(out) >= n:
            break
    return {"tiles": out}
