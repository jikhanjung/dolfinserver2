#!/bin/bash
# **작업 자리(m710q) → re-ID 자리(GCP)** 레인을 한 번 돌린다. 여기서 돌린다.
#
#   deploy/gcp/from_work_to_reid.sh              # 행 + 조각·크롭 파일
#   deploy/gcp/from_work_to_reid.sh --rows-only  # 행만
#
# 나르는 것은 **작업 자리가 주인인 것뿐**이다 — 사진·상자·크롭·마스크·실행·판정.
# 개체와 개체판정은 저쪽이 주인이라 한 줄도 안 간다(명령이 그것을 보장하고,
# 받는 쪽이 넣기 전에 다시 확인한다).
#
# **왜 필요한가**: 상자를 더 들이고 격자를 새로 만들면 그 격자가 가리키는
# `Box` 행이 저쪽에 없다. 조각은 보이는데 **개체에 넣으려는 순간 FK 가 막는다.**
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# **`python3` 를 그냥 부르지 않는다** — cron 에는 venv 가 없다 (`deploy/_venv.sh`)
source "$REPO/deploy/_venv.sh"
HOST="${FIN_GCP_HOST:-dolfinid}"
ROOT=/srv/dolfinserver2
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
ROWS="$TMP/from_work.sqlite3"

echo "▶ 담는다"
cd "$REPO" && "$PY" manage.py export_from_work_to_reid --out "$ROWS"

echo "▶ 보낸다"
# `db/` 로 보내는 것은 그 디렉토리가 컨테이너의 `/app/hostdb` 라서다 —
# 명령이 컨테이너 안에서 돌아야 화면이 열어 둔 바로 그 DB 에 들어간다.
rsync -a --info=progress2 "$ROWS" "$HOST:$ROOT/db/_from_work.sqlite3"

echo "▶ 넣는다 (먼저 헛돌려 본다)"
ssh "$HOST" "cd $ROOT && docker compose exec -T web python manage.py \
    import_from_work_to_reid --from /app/hostdb/_from_work.sqlite3 --dry-run"
echo "▶ 넣는다"
ssh "$HOST" "cd $ROOT && docker compose exec -T web python manage.py \
    import_from_work_to_reid --from /app/hostdb/_from_work.sqlite3 && \
    rm -f $ROOT/db/_from_work.sqlite3"

if [ "${1:-}" != "--rows-only" ]; then
    echo "▶ 조각"
    rsync -a --info=stats1 --delete "$REPO/reid/" "$HOST:$ROOT/reid/"
    echo "▶ 크롭"
    rsync -a --info=stats1 "$REPO/crops/" "$HOST:$ROOT/crops/"
fi

echo ""
ssh "$HOST" "curl -s https://reid.nopeoplestime.info/healthz" | python3 -m json.tool | head -5
