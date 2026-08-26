#!/bin/bash
# GCP 에서 판을 갈아 끼운다. **거기서** 돌린다.
#
#   /srv/dolfinserver2/deploy.sh X.Y.Z
#
# 순서가 곧 안전이다 — **스냅샷 먼저**, 그 다음 스왑. 되돌릴 자리 없이 바꾸면
# 마이그레이션이 스키마를 좁혔을 때 돌아갈 곳이 없다.
set -euo pipefail

ROOT=/srv/dolfinserver2
VERSION="${1:-}"
[ -n "$VERSION" ] || { echo "사용법: $0 X.Y.Z"; exit 1; }
cd "$ROOT"

echo "=== [1/5] 배포 전 스냅샷 ==="
# 시간별 트랙과 **자리를 나눈다** — 여기 것은 배포마다 하나이고, 시간별
# retention 이 이것을 지우면 안 된다. 형제가 두 곳이 같은 디렉토리를 서로 다른
# 수치로 정리하다 부딪힌 적이 있다.
mkdir -p backup/pre_deploy
SNAP="backup/pre_deploy/fin_$(date +%Y-%m-%d_%H%M)_pre_$VERSION.sqlite3"
python3 - "$SNAP" <<'PY'
import sqlite3, sys
src = sqlite3.connect("file:/srv/dolfinserver2/db/fin.db?mode=ro", uri=True)
dst = sqlite3.connect(sys.argv[1])
src.backup(dst); dst.execute("PRAGMA journal_mode=DELETE")
got = dst.execute("PRAGMA integrity_check").fetchone()[0]
n = dst.execute("select (select count(*) from finseg_identification),"
                " (select count(*) from finseg_individual)").fetchone()
dst.close(); src.close()
print(f"  {sys.argv[1]}  integrity={got}  개체판정 {n[0]:,} · 개체 {n[1]:,}")
assert got == "ok", "스냅샷이 성하지 않다 — 배포를 멈춘다"
PY
# 스냅샷은 최근 20벌만. **시간별 것과 다른 디렉토리라 서로 안 건드린다.**
ls -t backup/pre_deploy/*.sqlite3 2>/dev/null | tail -n +21 | xargs -r rm -f

echo "=== [2/5] 안내 페이지 ==="
touch maintenance.flag
trap 'rm -f "$ROOT/maintenance.flag"' EXIT

echo "=== [3/5] 이미지 $VERSION ==="
# **이미 있으면 안 당긴다.** 지금은 레지스트리에 안 올리고 m710q 에서
# `docker save | ssh docker load` 로 실어 나른다 (`deploy/gcp/ship.sh`) —
# 이미지에 자료는 없지만 공개 레지스트리에 올리는 것은 따로 정할 일이다.
if ! docker image inspect "honestjung/dolfinserver2:$VERSION" >/dev/null 2>&1; then
    docker pull "honestjung/dolfinserver2:$VERSION"
fi
sed -i "s/^IMAGE_TAG=.*/IMAGE_TAG=$VERSION/" .env

echo "=== [4/5] 스왑 ==="
docker compose up -d

echo "=== [5/5] smoke ==="
./smoke.sh "$VERSION"

rm -f maintenance.flag
trap - EXIT
echo ""
echo "되돌리려면:  $ROOT/deploy.sh <이전 X.Y.Z>"
echo "  (DB 는 그대로 둔다. 스키마를 되돌려야 하면 backup/pre_deploy/ 에서 사람이 고른다 —"
echo "   **어느 것을 되돌릴지는 기계가 정할 수 없다.**)"
