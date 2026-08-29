#!/bin/bash
# **re-ID 자리(GCP) → 작업 자리(m710q)** 레인. 여기서 돌린다.
#
#   deploy/gcp/from_reid_to_work.sh
#
# 나르는 것은 **개체와 개체판정뿐**이고, 받는 쪽에서 **통째로 갈아 끼운다** —
# 저쪽이 그 둘의 유일한 주인이라 합칠 것이 없다.
#
# **언제 돌리나**: 분류기를 다시 배우기 직전(`reid_cls --fit-all`)과 성적을 잴
# 때(`reid_eval`). **판정이 사라질 걱정으로 돌리는 것이 아니다** — 그것은
# 백업이 맡는다(저쪽 매시 + 여기 05시 당겨오기). 이 레인은 **쓰려고** 돈다.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# **`python3` 를 그냥 부르지 않는다** — cron 에는 venv 가 없다 (`deploy/_venv.sh`)
source "$REPO/deploy/_venv.sh"
HOST="${FIN_GCP_HOST:-dolfinid}"
ROOT=/srv/dolfinserver2
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "▶ 저쪽에서 담는다"
ssh "$HOST" "cd $ROOT && docker compose exec -T web python manage.py \
    export_from_reid_to_work --out /app/hostdb/_to_work.sqlite3"

echo "▶ 가져온다"
rsync -a "$HOST:$ROOT/db/_to_work.sqlite3" "$TMP/back.sqlite3"
ssh "$HOST" "rm -f $ROOT/db/_to_work.sqlite3"

echo "▶ 헛돌려 본다"
cd "$REPO" && "$PY" manage.py import_from_reid_to_work --from "$TMP/back.sqlite3" --dry-run
echo "▶ 갈아 끼운다"
"$PY" manage.py import_from_reid_to_work --from "$TMP/back.sqlite3"

echo ""
echo "다음 — 새 판정으로 분류기를 다시 배운다:"
echo "  python manage.py reid_cls --epochs 2000                    # 성적"
echo "  python manage.py reid_cls --epochs 2000 --fit-all reid/v3/cls-dinov2.npz"
echo "  deploy/gcp/from_work_to_reid.sh    # 그 분류기를 저쪽 화면에 준다"
