"""**다시 만들 수 없는 것을 NAS 로 뜬다.**

    python manage.py backup --dry-run
    python manage.py backup

## 무엇이 다시 만들 수 없나

| | 다시 만들 수 있나 |
|---|---|
| **`fin.db`** | **아니다** — 사람의 판정 6,000여 건과 개체 분류가 들어 있다 |
| **가중치** | 사실상 아니다 — 자료가 같아도 16시간을 다시 써야 하고 씨앗이 달라진다 |
| 크롭 · 자료 꾸러미 · 조각 | 된다 — `fin.db` 와 사진에서 다시 뽑는다 |
| 사진 | NAS 가 원본이다 |
| 코드 | git 에 있다 |

그래서 기본은 **`fin.db` 와 가중치만**이다. 나머지는 뜨는 값보다 자리 값이 크다.

## `--derived` 는 다른 일이다 — **다른 기계로 옮기려는 것**

크롭과 re-ID 조각은 다시 만들 수 있지만, **다시 만들려면 사진(NAS)과 시간이
든다.** GPU 없는 기계에서 검토·분류만 하려는데 크롭부터 다시 자르고 있으면
그 자리에서 반나절이 간다.

그래서 `--derived` 를 주면 그것들도 함께 뜬다. **한 벌만 두고 바뀐 것만
옮긴다**(크기·mtime 이 같으면 건너뛴다) — 크롭 3,005장을 날마다 다시 쓸 이유가
없다.

**다만 `fin.db` 는 한 기계에서만 연다.** 옮겨 가서 일했으면 거기서 다시 떠서
가져와야 하고, 양쪽에서 동시에 열면 어느 쪽 판정이 이기는지 아무도 모른다.

## `fin.db` 는 날짜별, 가중치는 이름별

`fin.db` 는 날마다 바뀌므로 **날짜를 붙여 쌓는다** — 어제 것으로 돌아갈 수
있어야 하고, 하루 지나 알아채는 사고가 실제로 있다.

가중치는 학습이 끝나면 **안 바뀐다.** 날마다 뜨면 같은 271MB 를 되풀이해
쌓을 뿐이다. 그래서 run 이름별로 한 벌만 두고, **내용이 같으면 건너뛴다**
(sha256). 그날 무엇이 있었는지는 `MANIFEST.json` 이 적는다.

## 열려 있는 DB 를 그냥 복사하지 않는다

`shutil.copy` 는 쓰는 중인 sqlite 를 반쯤 복사할 수 있고, **그렇게 깨진 것은
복원할 때가 되어서야 드러난다.** `sqlite3.backup()` 은 잠금을 지켜 가며 뜨므로
검토 화면이 열려 있어도 안전하다.

뜬 뒤에 **`PRAGMA integrity_check` 로 읽어 본다.** 확인 안 한 백업은 백업이
아니다 — 형제 프로젝트가 프레임 229장을 잃은 것이 그 자리였다.
"""
import hashlib
import json
import shutil
import sqlite3
from datetime import date
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from finseg import nas, runs

# **기계마다 NAS 를 보는 길이 다르다** — 뿌리는 `finseg/nas.py` 한 곳이다
OUT = str(nas.root() / "dolfinserver2_backup")


def sha256(path, n=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(n):
            h.update(chunk)
    return h.hexdigest()


class Command(BaseCommand):
    help = "fin.db 와 가중치를 NAS 로 뜬다 (다시 만들 수 없는 것만)"

    def add_arguments(self, p):
        p.add_argument("--out", default=OUT)
        p.add_argument("--db", help="기본은 settings 의 것")
        p.add_argument("--keep", type=int, default=30,
                       help="`fin.db` 를 며칠 치 남길지 (0이면 안 지운다)")
        p.add_argument("--no-weights", action="store_true")
        p.add_argument("--derived", action="store_true",
                       help="크롭·re-ID 조각도 함께 (다른 기계로 옮길 때)")
        p.add_argument("--dry-run", action="store_true")

    def handle(self, **o):
        w = self.stdout.write
        out = Path(o["out"])
        today = date.today().isoformat()
        # **뜰 곳을 먼저 본다.** DB 를 다 읽고 나서 "거기 없다" 를 만나면
        # 그 시간이 헛것이 되고, 무엇보다 **없는 데다 뜨면 뜬 줄 알고 지나간다**
        w(f"뜰 곳 {out}")
        if not out.parent.exists():
            raise CommandError(
                f"{out.parent} 가 없다 — NAS 가 안 붙어 있는 것 같다.\n"
                f"  본 자리: {' · '.join(nas.ROOTS)} (`FIN_NAS` 로 대 줄 수 있다)\n"
                f"  마운트를 먼저 볼 것. **없는 데다 뜨면 뜬 줄 알고 지나간다.**")

        db = Path(o["db"] or settings.DATABASES["default"]["NAME"])
        if not db.exists():
            raise CommandError(
                f"DB 가 없다: {db}\n"
                f"  메모리 DB 로 도는 중이면 `--db` 로 파일을 대 줄 것.")

        # ---- fin.db — 날짜별 -------------------------------------------
        dst = out / "db" / f"fin.db.{today}.bak"
        w(f"\n{db} → {dst.name}  ({db.stat().st_size / 1e6:.0f}MB)")
        if not o["dry_run"]:
            dst.parent.mkdir(parents=True, exist_ok=True)
            src = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            tgt = sqlite3.connect(dst)
            try:
                src.backup(tgt)
                # **곁딸린 `-wal`·`-shm` 을 남기지 않는다.** 우리 DB 가 WAL
                # 모드라 뜬 것도 그렇게 되는데, 백업 옆에 그 둘이 놓여 있으면
                # **복원할 때 무엇이 진짜인지 헷갈린다** — 셋을 다 옮겨야 하는
                # 것처럼 보이고, 하나만 옮기면 조용히 옛 상태가 된다.
                # `DELETE` 로 바꾸면 내용이 본 파일 하나로 합쳐진다
                tgt.execute("PRAGMA journal_mode=DELETE")
            finally:
                tgt.close()
                src.close()
            # **뜬 것을 읽어 본다.** 확인 안 한 백업은 백업이 아니다.
            # 표 이름을 못 박지 않는다 — 스키마가 바뀌면 백업이 깨지는데,
            # 그때 멎어야 할 이유가 없다. 있는 것만 센다
            chk = sqlite3.connect(f"file:{dst}?mode=ro", uri=True)
            ok = chk.execute("PRAGMA integrity_check").fetchone()[0]
            have = {r[0] for r in chk.execute(
                "select name from sqlite_master where type='table'")}
            count = lambda t: (chk.execute(f"select count(*) from {t}").fetchone()[0]
                               if t in have else None)
            n_rev, n_id = count("finseg_review"), count("finseg_identification")
            chk.close()
            for side in ("-wal", "-shm"):
                p_side = dst.with_name(dst.name + side)
                if p_side.exists():
                    p_side.unlink()
            if ok != "ok":
                raise CommandError(f"뜬 DB 가 깨졌다: {ok}")
            w("  integrity_check ok"
              + (f" · 판정 {n_rev:,}" if n_rev is not None else "")
              + (f" · 개체 판정 {n_id:,}" if n_id is not None else "")
              + f" · 표 {len(have)} 개")

        # 오래된 것 지우기
        olds = sorted((out / "db").glob("fin.db.*.bak")) if (out / "db").exists() else []
        if o["keep"] and len(olds) > o["keep"]:
            drop = olds[:len(olds) - o["keep"]]
            w(f"  {o['keep']}일 치만 남긴다 — {len(drop)} 개를 지운다")
            for f in drop:
                if not o["dry_run"]:
                    f.unlink()

        # ---- 가중치 — 이름별, 같으면 건너뛴다 ----------------------------
        kept = []
        if not o["no_weights"]:
            w("")
            for f in sorted(Path("runs").glob("*/weights/best.pt")):
                name = f.parent.parent.name
                tgt = out / "weights" / name / "best.pt"
                same = tgt.exists() and sha256(tgt) == sha256(f)
                kept.append({"run": name, "size": f.stat().st_size,
                             "sha256": sha256(f)})
                w(f"  {name:22s} {f.stat().st_size / 1e6:5.0f}MB"
                  + ("  (같다 — 건너뛴다)" if same else "  → 뜬다"))
                if same or o["dry_run"]:
                    continue
                tgt.parent.mkdir(parents=True, exist_ok=True)
                # **`copy2` 가 아니라 `copyfile` 이다.** NAS 는 drvfs 로 붙어
                # 있어 파일 시각을 못 바꾼다 — `copy2` 는 내용을 다 옮기고
                # 나서 `utime` 에서 `PermissionError` 를 낸다. 우리가 지킬 것은
                # 내용이지 시각이 아니고, 언제 떴는지는 파일 이름과
                # `MANIFEST` 가 든다
                shutil.copyfile(f, tgt)
                for extra in ("results.csv", "args.yaml"):
                    src_e = f.parent.parent / extra
                    if src_e.exists():
                        shutil.copyfile(src_e, tgt.parent / extra)

        # ---- 파생물 — 다른 기계로 옮길 때만 -----------------------------
        if o["derived"]:
            w("")
            for name, src_d in (("crops", Path("crops")),
                                ("reid", Path("reid"))):
                if not src_d.is_dir():
                    continue
                n_new = n_same = 0
                for f in sorted(src_d.rglob("*")):
                    if not f.is_file():
                        continue
                    tgt = out / "derived" / name / f.relative_to(src_d)
                    # **크기와 mtime 으로 가린다** — 3,005장에 sha256 을 걸면
                    # 뜨는 것보다 재는 것이 오래 걸린다. 파생물이라 그 정도면 된다
                    if tgt.exists() and tgt.stat().st_size == f.stat().st_size:
                        n_same += 1
                        continue
                    if not o["dry_run"]:
                        tgt.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(f, tgt)
                    n_new += 1
                w(f"  {name:8s} 새로 뜬 것 {n_new:,} · 같아서 건너뛴 것 {n_same:,}")

        if o["dry_run"]:
            w("\n--dry-run 이라 아무것도 쓰지 않았다.")
            return

        (out / f"MANIFEST.{today}.json").write_text(json.dumps({
            "date": today, "git_sha": runs.git_sha(),
            "db": {"file": dst.name, "size": dst.stat().st_size,
                   "sha256": sha256(dst),
                   "reviews": n_rev, "identifications": n_id},
            "weights": kept,
        }, ensure_ascii=False, indent=1))
        w(f"\n{out}")
        if o["derived"]:
            w("  크롭·조각도 함께 떴다 (`--derived`) — **다른 기계로 옮기려는"
              " 것**이지 백업이 아니다")
        else:
            w("  **다시 만들 수 없는 것만 든다** — 크롭·자료 꾸러미·조각은 안"
              " 뜬다. `fin.db` 와 사진에서 다시 뽑는다")
        w("  **`fin.db` 는 한 기계에서만 연다.** 옮겨 가서 일했으면 거기서 다시"
          " 떠서 가져올 것 —")
        w("  양쪽에서 동시에 열면 어느 쪽 판정이 이기는지 아무도 모른다")
