#!/usr/bin/env python3
"""`fin.db` 시간별 백업 — SQLite 온라인 백업 + 무결성 게이트.

`.guides/web/data-safety.md` 의 백업 계약 구현. **`fin.db` 는 사람의 판정이라
다시 만들 수 없고**, 가이드가 말하는 "operator-entered data" 가 정확히 그것이다.
형제 구현(`/srv/dolfinserver/scripts/backup_db.py`)에서 구조를 가져왔다.

계약에서 여기가 지키는 것:

- §1 시간별 트랙. 형제들은 하루 한 번인데 여기는 **시간마다** 뜬다 —
     `fin.db` 가 36MB 라 24벌이 900MB 밖에 안 되고, 사람이 판정을 넣는 화면이
     하루 종일 돈다. 하루치를 잃는 것과 한 시간치를 잃는 것은 다르다
- §2 **스냅샷**(라이브 소스가 아니라)에 `integrity_check` → 실패하면 prune 을
     건너뛰고 증거본 하나를 남기고 센티넬을 세운다. 이 규칙을 어긴 형제가
     실제로 **성한 스냅샷 0개**가 된 적이 있다
- §4 NAS(하위 트랙)는 라이브 DB 가 아니라 **검증된 로컬 스냅샷**을 복사한다
- §5 스냅샷을 `journal_mode=DELETE` 로 내려 단일 파일을 보장한다
- §6 `RETAIN_COUNT × 주기 ≥ 오프사이트 간격` → 시간별 + 일 1회 NAS = **24**
- §7 반출 위생: **검증 이후** `django_session` 삭제 + `VACUUM`

**이름에 기계가 들어간다** (`fin_<기계>_<날짜>_<시>.sqlite3`). 곧 GCP 에도
진짜 `fin.db` 가 생기고 둘 다 같은 NAS 로 오는데, 이름이 같으면 **나중에 뜬
기계가 먼저 뜬 기계의 것을 갈아 치운다** — 이 저장소가 한 번 겪은 일이다.
그래서 prune 도 제 기계 갈래만 본다.

cron:
    0 * * * * python3 /srv/dolfinserver2/scripts/backup_db.py \\
                  >> /srv/dolfinserver2/logs/backup.log 2>&1
"""
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(os.environ.get("FIN_ROOT", "/srv/dolfinserver2"))
DB_SOURCE = Path(os.environ.get("FIN_DB", ROOT / "db" / "fin.db"))
LOCAL_DIR = Path(os.environ.get("FIN_BACKUP_DIR", ROOT / "backup" / "hourly"))
# **빈 값이면 오프사이트 트랙이 아예 없다는 뜻이다.** GCP 에는 NAS 가 없고,
# 계약 §1 이 말하는 대로 **오프사이트가 프로덕션에서 당겨간다**(m710q 가
# `pull_gcp_backup.sh` 로 가져온다). 없는 것을 매일 "없다" 고 적으면 그 소음이
# 진짜 실패를 묻는다 (§3 — 부재는 정상이다).
_nas = os.environ.get("FIN_NAS_DIR", "/nas/JikhanJung/dolfinserver2_backup/daily")
NAS_DIR = Path(_nas) if _nas else None

# **기계 이름이 들어간다.** 같은 NAS 에 m710q 와 GCP 것이 함께 온다.
HOST = os.environ.get("FIN_BACKUP_HOST") or socket.gethostname().split(".")[0]
PREFIX = f"fin_{HOST}"
SUFFIX = ".sqlite3"

# `/healthz` 가 stat 하는 센티넬 — **DB 파일 옆에 둔다.** 호스트의 `db/` 가
# 그대로 컨테이너에 마운트되므로 cron(호스트)과 앱(컨테이너)이 같은 파일을 본다.
SENTINEL = Path(os.environ.get("FIN_SENTINEL", DB_SOURCE.parent / "INTEGRITY_FAIL"))

# §6 관계이지 튜닝값이 아니다 — 시간별 × 24 = 하루, NAS 가 하루 한 번이므로
# 그 사이 어느 시각으로도 되돌아갈 수 있다. 늘리는 것은 무손실이다(안 지운다).
RETAIN_COUNT = 24
NAS_HOUR = 4              # 이 시각에 뜬 것만 NAS 로 올린다 (하위 트랙 = 일 1회)
NAS_DAILY_DAYS = 90       # NAS 는 90일 + 매달 1일 영구

MIN_FREE_GB = 2           # DB 가 36MB 라 한 회분의 수십 배
NAS_TIMEOUT = 15          # NFS 가 멎었을 때 cron 이 매달리지 않도록

_notify = os.environ.get("FIN_NOTIFY", "/home/jikhanjung/scripts/notify-telegram.sh")
NOTIFY = Path(_notify) if _notify else None


def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def notify_fail(msg):
    """무인 실패는 사람이 이미 보는 채널로 (`.guides/web/operations.md`)."""
    if NOTIFY is not None and NOTIFY.is_file() and os.access(NOTIFY, os.X_OK):
        try:
            subprocess.run([str(NOTIFY), f"⚠️ dolfinserver2 백업({HOST}): {msg}"],
                           timeout=30, capture_output=True)
        except (subprocess.SubprocessError, OSError):
            pass


def fail(msg):
    log(f"ERROR: {msg}")
    notify_fail(msg)
    return 1


def nas_available():
    if NAS_DIR is None:
        return False
    try:
        return subprocess.run(["test", "-d", str(NAS_DIR)],
                              timeout=NAS_TIMEOUT).returncode == 0
    except subprocess.TimeoutExpired:
        log(f"WARN: NAS 응답 없음 ({NAS_TIMEOUT}s 초과) — NAS 트랙 건너뜀")
        return False
    except OSError:
        return False


def check_disk_space():
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(LOCAL_DIR).free / (1024 ** 3)
    if free_gb < MIN_FREE_GB:
        log(f"ABORT: 여유 디스크 {free_gb:.2f} GB < {MIN_FREE_GB} GB")
        return False
    return True


def make_snapshot(tmp):
    """온라인 백업 API. **컨테이너가 DB 를 연 채여도 일관 스냅샷을 얻는다** —
    `cp` 는 WAL 이 붙어 있으면 커밋 꼬리를 놓친 반쪽을 가져온다."""
    src = dst = None
    try:
        # WAL 소스는 `-shm` 이 없으면 read-only 로 못 연다(컨테이너가 내려가
        # 있을 때). 읽기 전용을 먼저 대 보고 안 되면 일반 연결로 — 백업 API 는
        # 읽기만 한다.
        try:
            src = sqlite3.connect(f"file:{DB_SOURCE}?mode=ro", uri=True)
            src.execute("SELECT 1 FROM sqlite_schema LIMIT 1")
        except sqlite3.Error:
            if src is not None:
                src.close()
            src = sqlite3.connect(str(DB_SOURCE))
        dst = sqlite3.connect(str(tmp))
        src.backup(dst)
        dst.execute("PRAGMA journal_mode=DELETE")      # §5
        return True
    except (sqlite3.Error, OSError) as e:
        log(f"ERROR: 스냅샷 생성 실패 — {e}")
        return False
    finally:
        for c in (dst, src):
            if c is not None:
                c.close()


def integrity_ok(path):
    """§2 **스냅샷을** 검사한다. 라이브를 검사하면 긴 읽기 트랜잭션이 붙고,
    스냅샷이 성하지 않다는 것이 곧 소스가 성하지 않다는 뜻이다."""
    conn = None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        got = conn.execute("PRAGMA integrity_check").fetchone()[0]
        if got != "ok":
            log(f"ERROR: integrity_check 실패 — {got[:200]}")
            return False
        return True
    except sqlite3.Error as e:
        log(f"ERROR: integrity_check 예외 — {e}")
        return False
    finally:
        if conn is not None:
            conn.close()


def judgment_counts(path):
    """**무엇을 지켰는지 로그에 남긴다.** 파일 크기는 판정이 몇 건인지 말해
    주지 않는다 — 되돌릴 때 고르는 기준이 그 수다."""
    try:
        c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        n = c.execute("select (select count(*) from finseg_review), "
                      "(select count(*) from finseg_identification), "
                      "(select count(*) from finseg_individual)").fetchone()
        c.close()
        return f"판정 {n[0]:,} · 개체판정 {n[1]:,} · 개체 {n[2]:,}"
    except sqlite3.Error:
        return "(셀 수 없음)"


def strip_sessions(path):
    """§7 반출 위생 — **검증 이후에만.**

    세션 키가 곧 쿠키 값이라 사본을 읽은 사람이 그대로 재제출하면 로그인된다
    (`SECRET_KEY` 가 달라도 안 막힌다 — 공격이 복호화를 안 한다). `VACUUM` 은
    필수다: `DELETE` 로는 닿지 않는 free page 에 지난 세션이 남아 있다.

    검증보다 먼저 돌리면 **손상된 소스에서 `VACUUM` 이 터지고 그것을 삼키느라
    센티넬이 안 선다** — 안전망이 조용히 꺼진다.
    """
    conn = None
    try:
        conn = sqlite3.connect(str(path))
        n = conn.execute("SELECT COUNT(*) FROM django_session").fetchone()[0]
        conn.execute("DELETE FROM django_session")
        conn.commit()
        conn.execute("VACUUM")
        conn.commit()
        log(f"반출 위생: django_session {n}행 삭제 + VACUUM")
        return True
    except sqlite3.Error as e:
        log(f"ERROR: 반출 위생 실패 — {e}")
        return False
    finally:
        if conn is not None:
            conn.close()


def mine(directory):
    """**제 기계 갈래만.** 남의 갈래를 이쪽 셈으로 줄이면 그 기계는 제가 몇 벌
    갖고 있는지 모르는 채 줄어든다."""
    try:
        return sorted(directory.glob(f"{PREFIX}_*{SUFFIX}"))
    except OSError as e:
        log(f"WARN: 목록 조회 실패 ({directory}) — {e}")
        return []


def prune_hourly():
    """§6 최근 `RETAIN_COUNT` 벌만. **성공 경로에서만 부른다.**"""
    files = mine(LOCAL_DIR)
    if len(files) <= RETAIN_COUNT:
        return
    for f in files[:len(files) - RETAIN_COUNT]:
        try:
            f.unlink()
        except OSError:
            pass
    log(f"시간별 정리: {len(files) - RETAIN_COUNT}개 삭제 (최근 {RETAIN_COUNT}벌 유지)")


def prune_nas():
    """계층형: `NAS_DAILY_DAYS` 초과분 중 **매달 1일은 남긴다**(12월 1일은 영구)."""
    if NAS_DIR is None:
        return
    now, deleted = datetime.now(), 0
    for f in mine(NAS_DIR):
        stamp = f.name[len(PREFIX) + 1:-len(SUFFIX)]
        try:
            dt = datetime.strptime(stamp.split("_")[0], "%Y-%m-%d")
        except ValueError:
            continue                       # 이름 규칙 밖 → 안 건드린다
        if (now - dt).days <= NAS_DAILY_DAYS or dt.day == 1:
            continue
        try:
            f.unlink()
            deleted += 1
        except OSError:
            pass
    if deleted:
        log(f"NAS 정리: {deleted}개 삭제 ({NAS_DAILY_DAYS}일 초과, 월초 보존)")


def report(directory, label):
    files = mine(directory)
    if not files:
        log(f"{label}: (없음)")
        return
    total = sum(f.stat().st_size for f in files) / (1024 ** 2)
    log(f"{label}: {len(files)}개 · {total:.0f} MB · 최신 {files[-1].name}")


def main():
    now = datetime.now()
    log(f"===== dolfinserver2 백업 시작 ({HOST}) =====")

    if not DB_SOURCE.is_file():
        return fail(f"소스 DB 없음 ({DB_SOURCE})")
    if not check_disk_space():
        return fail("디스크 여유 부족")

    stamp = now.strftime("%Y-%m-%d_%H")
    final = LOCAL_DIR / f"{PREFIX}_{stamp}{SUFFIX}"
    tmp = LOCAL_DIR / f"{PREFIX}_{stamp}{SUFFIX}.tmp"
    tmp.unlink(missing_ok=True)

    # --- 1. 스냅샷 ---
    if not make_snapshot(tmp):
        tmp.unlink(missing_ok=True)
        return fail("스냅샷 생성 실패 — 기존 백업은 정리하지 않는다")

    # --- 2. 무결성 게이트 (§2) ---
    if not integrity_ok(tmp):
        evidence = LOCAL_DIR / f"{PREFIX}_{stamp}_INTEGRITY_FAIL.corrupt"
        try:
            tmp.replace(evidence)          # 확장자가 달라 prune glob 에 안 걸린다
            log(f"증거본 보존: {evidence.name}")
        except OSError:
            tmp.unlink(missing_ok=True)
        try:
            SENTINEL.parent.mkdir(parents=True, exist_ok=True)
            SENTINEL.write_text(f"{now.isoformat()} integrity_check failed\n")
        except OSError:
            pass
        return fail("integrity_check 실패 — 소스 손상 의심. "
                    "기존 백업 보존, 사람이 판단할 것")

    counts = judgment_counts(tmp)

    # --- 3. 반출 위생 (§7) — 검증 이후 ---
    if not strip_sessions(tmp):
        tmp.unlink(missing_ok=True)
        return fail("반출 위생 실패 — 세션이 남은 사본은 내보내지 않는다")

    # --- 4. 확정 (원자 rename) ---
    try:
        os.chmod(tmp, 0o640)
        tmp.replace(final)
    except OSError as e:
        tmp.unlink(missing_ok=True)
        return fail(f"확정 rename 실패 — {e}")
    log(f"시간별 완료: {final.name} ({final.stat().st_size / 1024**2:.0f} MB · {counts})")

    SENTINEL.unlink(missing_ok=True)

    # --- 5. NAS 트랙 (§4) — 라이브가 아니라 **검증된 스냅샷**을 복사 ---
    # §3 required 축: NAS 부재는 정상(집 네트워크가 끊길 수 있다), 있는데
    # 실패하는 것이 진짜 실패다. 어느 쪽이든 NAS 정리는 하지 않는다.
    nas_ok = False
    nas_present = False
    if NAS_DIR is not None and now.hour == NAS_HOUR:
        nas_present = nas_available()
        if nas_present:
            NAS_DIR.mkdir(parents=True, exist_ok=True)
            nas_final = NAS_DIR / final.name
            nas_tmp = NAS_DIR / f"{final.name}.tmp"
            try:
                shutil.copyfile(final, nas_tmp)
                os.replace(nas_tmp, nas_final)
                try:
                    os.chmod(nas_final, 0o640)
                except OSError:
                    pass                   # 일부 NFS 는 chmod 를 안 받는다
                if integrity_ok(nas_final):        # 수신 측 재검증 (§4)
                    log(f"NAS 완료: {nas_final.name}")
                    nas_ok = True
                else:
                    log("ERROR: NAS 사본 검증 실패 — NAS 정리 건너뜀")
                    notify_fail("NAS 사본 integrity_check 실패")
            except OSError as e:
                log(f"ERROR: NAS 복사 실패 — {e}")
                notify_fail(f"NAS 복사 실패: {e}")
                try:
                    nas_tmp.unlink(missing_ok=True)
                except OSError:
                    pass
        else:
            log(f"WARN: NAS 없음 ({NAS_DIR}) — 건너뜀, 정리도 안 한다")

    # --- 6. 정리 (성공한 트랙만) ---
    prune_hourly()
    if nas_ok:
        prune_nas()

    report(LOCAL_DIR, "시간별")
    if nas_ok:
        report(NAS_DIR, "NAS")

    log("===== 백업 완료 =====")
    return 1 if (nas_present and not nas_ok) else 0


if __name__ == "__main__":
    sys.exit(main())
