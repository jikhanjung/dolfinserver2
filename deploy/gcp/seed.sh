#!/bin/bash
# **m710q 에서** 돌린다. GCP 로 자료를 민다 — 처음 한 번, 그리고 격자를 갈아
# 끼울 때마다.
#
#   deploy/gcp/seed.sh            # 전부 (처음)
#   deploy/gcp/seed.sh --no-db    # 조각·크롭만 (그 뒤로는 늘 이것)
#
# **`--no-db` 가 그 뒤의 기본이다.** GCP 가 뜬 뒤로 그쪽 `fin.db` 는 개체
# 판정의 **주인**이다 — 여기서 밀어 넣으면 그 판정이 사라진다. 상자·크롭을
# 늘렸으면 그것만 나르는 길(`reid_import`)이 따로 서야 한다 (`TODOs.md`).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HOST="${FIN_GCP_HOST:-dolfinid}"
ROOT=/srv/dolfinserver2
HOURLY=/srv/dolfinserver2/backup/hourly
WITH_DB=1
[ "${1:-}" = "--no-db" ] && WITH_DB=0

if [ "$WITH_DB" = "1" ]; then
    # **검증된 스냅샷을 보낸다** (`.guides/web/data-safety.md` §4) — 라이브
    # DB 를 rsync 하면 WAL 이 붙은 채로 반쪽이 갈 수 있다. 덤으로 세션도
    # 이미 지워져 있다(§7 반출 위생).
    SRC="$(ls -t "$HOURLY"/fin_*.sqlite3 2>/dev/null | head -1)"
    [ -n "$SRC" ] || { echo "시간별 스냅샷이 없다 — 먼저: python3 /srv/dolfinserver2/scripts/backup_db.py"; exit 1; }
    echo "▶ DB  $(basename "$SRC")"
    if ssh "$HOST" "[ -s $ROOT/db/fin.db ]"; then
        echo "  **거기 DB 가 이미 있다.** 개체 판정의 주인은 그쪽이다 — 덮지 않는다."
        echo "  정말 갈아 끼우려면 그 기계에서 백업을 뜬 뒤 손으로 옮길 것."
        exit 1
    fi
    rsync -a --info=progress2 "$SRC" "$HOST:$ROOT/db/fin.db"
fi

echo "▶ 조각 (reid)"
rsync -a --info=stats1 --delete "$REPO/reid/" "$HOST:$ROOT/reid/"
echo "▶ 크롭"
rsync -a --info=stats1 "$REPO/crops/" "$HOST:$ROOT/crops/"

echo ""
echo "다음 — GCP 에서:"
echo "  sudo $ROOT/_incoming/bootstrap.sh     # DB 가 생겼으니 cron 이 이제 걸린다"
echo "  cd $ROOT && docker compose up -d && ./smoke.sh"
