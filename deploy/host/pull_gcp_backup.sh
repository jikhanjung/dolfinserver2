#!/bin/bash
# **m710q 에서** 돈다. GCP 의 검증된 스냅샷을 당겨 온다.
#
#   deploy/host/pull_gcp_backup.sh
#   0 5 * * * /home/jikhanjung/projects/dolfinserver2/deploy/host/pull_gcp_backup.sh \
#                 >> /srv/dolfinserver2/logs/pull-gcp.log 2>&1
#
# **오프사이트가 프로덕션에서 당겨간다** (`.guides/web/data-safety.md` §1) —
# 반대로 하면 백업 서버가 운영에 쓰기 권한을 갖게 된다. GCP 에는 NAS 가 없어서
# 그쪽 오프사이트가 곧 이 기계이고, 여기서 다시 NAS 로 간다.
#
# **라이브 DB 를 가져오지 않는다** (§4) — 그쪽 시간별 트랙이 `integrity_check`
# 와 반출 위생을 이미 지난 파일을 가져오고, **받아서 다시 검사한다.**
set -euo pipefail

HOST="${FIN_GCP_HOST:-dolfinid}"
REMOTE=/srv/dolfinserver2/backup/hourly
LOCAL=/srv/dolfinserver2/backup/offsite
NAS=/nas/JikhanJung/dolfinserver2_backup/offsite
KEEP_DAYS=90
STALE_HOURS=26            # 그쪽 시간별이 멎었거나 게이트가 막고 있다는 뜻
NOTIFY=/home/jikhanjung/scripts/notify-telegram.sh

log() { echo "[$(date '+%F %T')] $*"; }
warn() {
    log "WARN: $*"
    [ -x "$NOTIFY" ] && "$NOTIFY" "⚠️ dolfinserver2 GCP 백업 당겨오기: $*" >/dev/null 2>&1 || true
}

mkdir -p "$LOCAL"
log "===== GCP 백업 당겨오기 시작 ====="

NEWEST="$(ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST" \
          "ls -t $REMOTE/fin_*.sqlite3 2>/dev/null | head -1" || true)"
if [ -z "$NEWEST" ]; then
    warn "그쪽에 스냅샷이 없다 — 시간별 cron 이 안 돌고 있을 수 있다"
    exit 1
fi

# **신선도 게이트.** 파일이 있다고 살아 있는 것이 아니다 — 멎은 cron 은 어제
# 것을 그대로 두고, 그것을 조용히 계속 가져오면 백업이 도는 줄 안다.
AGE_H="$(ssh -o BatchMode=yes "$HOST" \
         "echo \$(( ( \$(date +%s) - \$(stat -c %Y '$NEWEST') ) / 3600 ))")"
log "가장 새것 $(basename "$NEWEST") · ${AGE_H}시간 전"
[ "$AGE_H" -le "$STALE_HOURS" ] || warn "그쪽 스냅샷이 ${AGE_H}시간째 안 바뀐다 (게이트가 막고 있나?)"

TMP="$LOCAL/.$(basename "$NEWEST").tmp"
rsync -a --info=stats0 "$HOST:$NEWEST" "$TMP"

# **받아서 다시 검사한다** (§4). 이 갈래가 가장 오래 남는 사본이라,
# "가장 값진 백업이 가장 덜 검증됐다" 가 정확히 피해야 할 자리다.
GOT="$(sqlite3 "file:$TMP?mode=ro" "PRAGMA integrity_check;" 2>&1 | head -1)"
if [ "$GOT" != "ok" ]; then
    rm -f "$TMP"
    warn "받은 사본이 성하지 않다 ($GOT) — 채택하지 않고 정리도 건너뛴다"
    exit 1
fi
N="$(sqlite3 "file:$TMP?mode=ro" "select (select count(*) from finseg_identification)||' / '||(select count(*) from finseg_individual);")"
mv "$TMP" "$LOCAL/$(basename "$NEWEST")"
log "채택: $(basename "$NEWEST") · 개체판정/개체 $N"

# NAS 로 한 벌 더. **없으면 없는 대로 둔다** — 부재는 정상이고, 정리도 안 한다.
if [ -d "$(dirname "$NAS")" ]; then
    mkdir -p "$NAS"
    cp "$LOCAL/$(basename "$NEWEST")" "$NAS/.$(basename "$NEWEST").tmp"
    mv "$NAS/.$(basename "$NEWEST").tmp" "$NAS/$(basename "$NEWEST")"
    log "NAS: $NAS/$(basename "$NEWEST")"
else
    log "NAS 없음 — 건너뜀 (정리도 안 한다)"
fi

# 성공했을 때만 정리한다. 매달 1일은 남긴다.
find "$LOCAL" -name 'fin_*.sqlite3' -mtime +$KEEP_DAYS ! -name '*-01_*' -delete 2>/dev/null || true
log "보유 $(ls "$LOCAL"/fin_*.sqlite3 2>/dev/null | wc -l) 벌"
log "===== 끝 ====="
