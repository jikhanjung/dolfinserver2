"""`fin.db` 의 스키마와 여는 법.

**원본은 이 파일 하나다.** 사진도 가중치도 학습 자료도 다시 만들 수 있지만
사람의 판정은 못 만든다.

## 판정을 상자와 마스크로 나눈 이유

DiaRUGA 는 판정이 하나였다 — SAM2 AMG 가 개체와 마스크를 함께 냈으니 "이것이
규조각인가" 와 "이 마스크가 맞는가" 를 가를 이유가 없었다. 여기는 상자가 먼저
있고 마스크가 뒤에 붙으므로 두 판단이 갈린다.

- **`not_fin`** — 상자 안에 지느러미가 없다. YOLOv5 의 오검출이다.
  **엔진을 갈아도 유효하다.** 다시 물을 일이 없다
- **`fix`** — 지느러미는 맞는데 마스크가 틀렸다. **엔진에 딸린 판정**이라
  분할기를 바꾸면 다시 봐야 한다

그래서 `review` 가 `box_id` 와 `mask_id` 를 둘 다 든다. 상자에 붙는 판정은
분할기를 갈아도 살아남고, 마스크에 붙는 판정만 다시 받는다.
"""
import os
import sqlite3
import subprocess
import socket
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB = Path(os.environ.get("FIN_DB", "fin.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS image (
    id            INTEGER PRIMARY KEY,
    src_id        INTEGER UNIQUE,          -- dolfinrest_dolfinimage.id
    path          TEXT NOT NULL UNIQUE,    -- 사진 뿌리 기준 상대경로
    obsdate       TEXT,                    -- YYYY-MM-DD
    exifdatetime  TEXT,
    width         INTEGER,
    height        INTEGER
);
CREATE INDEX IF NOT EXISTS image_obsdate ON image(obsdate);

CREATE TABLE IF NOT EXISTS box (
    id        INTEGER PRIMARY KEY,
    src_id    INTEGER UNIQUE,              -- dolfinrest_dolfinbox.id
    image_id  INTEGER NOT NULL REFERENCES image(id),
    x1        INTEGER NOT NULL,
    y1        INTEGER NOT NULL,
    x2        INTEGER NOT NULL,
    y2        INTEGER NOT NULL,
    area      INTEGER,
    source    TEXT DEFAULT 'yolov5'        -- 이 상자를 만든 것
);
CREATE INDEX IF NOT EXISTS box_image ON box(image_id);

-- 원본에서 잘라 낸 자리. 마스크 좌표를 원본으로 되돌리는 데 쓴다.
-- 사진 가장자리에서는 정사각형이 못 되므로 실제 잘린 사각형을 그대로 적는다.
CREATE TABLE IF NOT EXISTS crop (
    id       INTEGER PRIMARY KEY,
    box_id   INTEGER NOT NULL UNIQUE REFERENCES box(id),
    path     TEXT NOT NULL,                -- 크롭 뿌리 기준 상대경로
    x0       INTEGER NOT NULL,             -- 원본에서의 왼쪽 위
    y0       INTEGER NOT NULL,
    x1       INTEGER NOT NULL,             -- 원본에서의 오른쪽 아래 (배타)
    y1       INTEGER NOT NULL,
    w        INTEGER NOT NULL,             -- 저장된 크롭의 크기
    h        INTEGER NOT NULL
);

-- 무엇이 언제 무엇으로 돌았나. 마스크는 반드시 하나에 매인다.
CREATE TABLE IF NOT EXISTS run (
    id           INTEGER PRIMARY KEY,
    kind         TEXT NOT NULL,            -- sam2 | yolo | export | train
    model        TEXT,                     -- 모델 id 또는 가중치 경로
    git_sha      TEXT,
    host         TEXT,
    params_json  TEXT,
    started_at   TEXT,
    finished_at  TEXT,
    note         TEXT
);

-- 후보 마스크. **덮어쓰지 않고 쌓는다** — 엔진을 갈아도 같은 상자 위에서
-- 나란히 견줄 수 있어야 한다.
CREATE TABLE IF NOT EXISTS mask (
    id          INTEGER PRIMARY KEY,
    box_id      INTEGER NOT NULL REFERENCES box(id),
    run_id      INTEGER NOT NULL REFERENCES run(id),
    polygon     TEXT NOT NULL,             -- "x,y x,y ..." 원본 화소 좌표
    area        INTEGER,
    conf        REAL,
    is_current  INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS mask_box ON mask(box_id, is_current);
CREATE INDEX IF NOT EXISTS mask_run ON mask(run_id);

-- 사람의 판정. 한 마스크에 여러 번 달릴 수 있고 **가장 늦은 것이 유효하다**
-- (고쳐 매길 수 있어야 한다). 지운 것도 남는다 — 어려운 음성 표본이고
-- 재현율의 분모다.
CREATE TABLE IF NOT EXISTS review (
    id        INTEGER PRIMARY KEY,
    box_id    INTEGER NOT NULL REFERENCES box(id),
    mask_id   INTEGER REFERENCES mask(id),  -- not_fin 은 마스크 없이도 달린다
    verdict   TEXT NOT NULL,                -- ok | fix | not_fin
    polygon   TEXT,                         -- 교정본. 없으면 마스크 그대로
    reviewer  TEXT,
    at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS review_box ON review(box_id, id);
CREATE INDEX IF NOT EXISTS review_mask ON review(mask_id, id);
"""

VERDICTS = ("ok", "fix", "not_fin")


def connect(path=None, readonly=False) -> sqlite3.Connection:
    """`fin.db` 를 연다. WAL 이라 검토 중에 추론이 읽어도 막히지 않는다.

    **쓰기는 여전히 하나다.** 파이프라인이 도는 중에 검토를 저장하면 잠긴다 —
    DiaRUGA 가 프레임 229장을 그렇게 잃었다. `busy_timeout` 은 그 완충일 뿐이지
    해결이 아니다.
    """
    path = Path(path or DEFAULT_DB)
    if readonly:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    else:
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=15000")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init(path=None) -> sqlite3.Connection:
    conn = connect(path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def git_sha() -> str:
    """지금 코드의 판. **어떤 코드가 만든 자료인지 나중에 반드시 묻게 된다.**"""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=Path(__file__).resolve().parent,
                             capture_output=True, text=True, timeout=5)
        sha = out.stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               cwd=Path(__file__).resolve().parent,
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return f"{sha}+dirty" if dirty else sha
    except Exception:
        return "?"


def start_run(conn, kind, model=None, params=None, note=None) -> int:
    import json
    cur = conn.execute(
        "INSERT INTO run (kind, model, git_sha, host, params_json, started_at, note)"
        " VALUES (?,?,?,?,?,?,?)",
        (kind, model, git_sha(), socket.gethostname(),
         json.dumps(params or {}, ensure_ascii=False), now(), note))
    conn.commit()
    return cur.lastrowid


def finish_run(conn, run_id) -> None:
    conn.execute("UPDATE run SET finished_at=? WHERE id=?", (now(), run_id))
    conn.commit()


# ---- 폴리곤 -----------------------------------------------------------------
# "x,y x,y ..." 로 적는다. 원본 화소 좌표이고 정수다. 눈으로 읽히고 grep 되며,
# 정규화는 내보낼 때 한다 — DB 안에서 정규화하면 사진 크기가 함께 있어야 뜻이
# 통하고, 그 둘이 어긋나는 사고가 조용히 지나간다.

def dumps_polygon(points) -> str:
    return " ".join(f"{int(round(x))},{int(round(y))}" for x, y in points)


def loads_polygon(s: str):
    if not s:
        return []
    return [tuple(int(v) for v in p.split(",")) for p in s.split()]
