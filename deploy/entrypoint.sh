#!/bin/bash
# 컨테이너가 하는 일은 셋뿐이다 — 정적파일 모으기 · 마이그레이션 · gunicorn.
# **자료를 나르지 않는다.** 크롭도 조각도 DB 도 호스트에 있고 여기서는 마운트로
# 볼 뿐이다. 배포가 자료를 만지기 시작하면 "재배포하면 판정이 날아가나" 를
# 매번 물어야 한다 (`.guides/web/`, fsis2026 `DEPLOY.md` 의 불변식).
set -e
umask 002

# ── 비-root 드롭 ────────────────────────────────────────────────────────────
# root 로 시작해서(collectstatic·소유권 정리) 서비스 프로세스만 내린다.
# 대상 uid 는 DB 디렉토리(`/app/hostdb`) 주인을 따른다 — 프로세스 uid 와 마운트
# 주인이 어긋나면 WAL 을 못 써서 검토 저장이 조용히 막힌다.
TARGET_UID="${APP_RUN_UID:-}"
TARGET_GID="${APP_RUN_GID:-}"
if [ -z "$TARGET_UID" ] && [ -d /app/hostdb ]; then
    TARGET_UID=$(stat -c %u /app/hostdb 2>/dev/null || echo "")
    TARGET_GID=$(stat -c %g /app/hostdb 2>/dev/null || echo "")
fi

run() { if [ -n "$DROP" ]; then $DROP "$@"; else "$@"; fi; }

if [ -z "$TARGET_UID" ] || [ "$TARGET_UID" = "0" ]; then
    echo "entrypoint: hostdb 마운트가 없거나 root 소유 — root 로 돈다"
    DROP=""
else
    : "${TARGET_GID:=$TARGET_UID}"
    echo "entrypoint: 권한을 ${TARGET_UID}:${TARGET_GID} 로 내린다"
    # 작은 마운트만 -R 한다. crops·photos 는 수천 장이라 걷는 값이 아깝고,
    # 어차피 읽기만 한다.
    chown -R "${TARGET_UID}:${TARGET_GID}" /app/hostdb /app/staticfiles 2>/dev/null || true
    # HOME 이 `/`(못 씀)로 잡히는 것을 막는다 — gosu 는 미등록 uid 에 그렇게 준다.
    DROP="gosu ${TARGET_UID}:${TARGET_GID} env HOME=/tmp"
fi

run python manage.py collectstatic --noinput
# **사람의 판정이 든 DB 위에서 돈다.** 스키마를 지우거나 좁히는 마이그레이션은
# 백업을 먼저 뜬 뒤에 배포할 것 (CLAUDE.md 함정 목록).
run python manage.py migrate --noinput

# 제어 소켓은 안 쓴다. 기본값이 작업 디렉토리(`/app`, root 소유)에 만들려 들어
# 권한을 내린 뒤에는 매번 `Control server error: Permission denied` 를 찍는다 —
# 돌기는 도는데 로그가 지저분해지고, 그러면 진짜 오류가 안 보인다.
# SQLite 는 쓰기가 하나다. 워커를 늘려도 쓰기는 줄을 서고, WAL 과 timeout(20초)이
# 완충일 뿐이다 — 읽기가 대부분인 검토 화면이라 2 로 둔다.
exec $DROP gunicorn finweb.wsgi:application \
    --bind 0.0.0.0:8000 \
    --no-control-socket \
    --workers "${GUNICORN_WORKERS:-2}" \
    --timeout "${GUNICORN_TIMEOUT:-120}" \
    --access-logfile -
