#!/bin/bash
# 시험 자리의 DB 를 **NAS 백업에서** 새로 뜬다.
#
#   deploy/host/test_db.sh
#
# 운영 것을 그대로 물리지 않는 이유가 둘이다 — 시험에서 무엇을 해도 사람의
# 판정이 안 다치고, **덤으로 백업이 성한지를 잰다.** 백업에서 복원해 화면이
# 도는 것을 본 적이 없으면 그것은 백업이 아니라 파일일 뿐이다.
set -euo pipefail

ROOT=/srv/dolfinserver2-test
HOURLY=/srv/dolfinserver2/backup/hourly
NAS=/nas/JikhanJung/dolfinserver2_backup

[ -d "$ROOT/db" ] || { echo "$ROOT 가 없다 — 먼저: sudo deploy/host/bootstrap.sh --test"; exit 1; }

# **검증을 통과한 스냅샷을 쓴다** (`.guides/web/data-safety.md` §4) — 라이브
# DB 를 복사하지 않는다. 시간별 것이 가장 새것이고 이미 `integrity_check` 와
# 반출 위생을 지난 파일이다. 없으면 NAS 갈래로 물러선다.
SRC="$(ls -t "$HOURLY"/fin_*.sqlite3 2>/dev/null | head -1)"
[ -n "$SRC" ] || SRC="$(ls -t "$NAS"/daily/fin_*.sqlite3 "$NAS"/db/fin.db.*.bak 2>/dev/null | head -1)"
[ -n "$SRC" ] || { echo "쓸 백업이 없다 — 먼저: python3 $HOURLY/../../scripts/backup_db.py"; exit 1; }
echo "▶ $(basename "$SRC")"

# `cp` 가 아니라 `.backup` 이다 — 백업에 `-wal` 이 붙어 있으면 `cp` 는 커밋
# 꼬리를 놓친 반쪽을 가져온다.
sqlite3 "file:$SRC?mode=ro" ".backup '$ROOT/db/fin.db'"
echo -n "  integrity: "; sqlite3 "$ROOT/db/fin.db" "PRAGMA integrity_check;" | head -1
sqlite3 -column "$ROOT/db/fin.db" "
  select '  상자 '||(select count(*) from finseg_box)
      ||' · 판정 '||(select count(*) from finseg_review)
      ||' · 개체판정 '||(select count(*) from finseg_identification)
      ||' · 개체 '||(select count(*) from finseg_individual);"
echo ""
echo "다음 — 역할을 골라 띄운다:"
echo "  sed -i 's/^FIN_ROLE=.*/FIN_ROLE=reid/' $ROOT/.env"
echo "  cd $ROOT && docker compose up -d      # http://m710q:8086"
