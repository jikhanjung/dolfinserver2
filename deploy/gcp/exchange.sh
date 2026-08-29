#!/bin/bash
# **두 레인을 한 번에 돌린다.** m710q 에서, 하루 한 번.
#
#   deploy/gcp/exchange.sh
#   30 6 * * * FIN_PY=$HOME/venv/dolfinserver2/bin/python \
#                  /home/jikhanjung/projects/dolfinserver2/deploy/gcp/exchange.sh \
#                  >> /srv/dolfinserver2/logs/exchange.log 2>&1
#
# **`FIN_PY` 를 적어 두는 것은 뜻을 보이려는 것**이지 없으면 안 도는 것은
# 아니다 — `deploy/_venv.sh` 가 스스로도 찾는다.
#
# **되받기가 먼저다.** 저쪽 판정을 가져와야 분류기를 다시 배울 수 있고, 그
# 분류기가 다시 저쪽 화면으로 가는 것이 한 바퀴다. 순서가 뒤집히면 그날 것은
# 하루 늦게 반영된다.
#
# 06:30 인 것은 05:00 의 백업 당겨오기 뒤라서다 — 그쪽이 먼저 돌아야 저쪽
# 스냅샷이 오늘 것으로 있다.
#
# **분류기를 여기서 다시 배우지 않는다.** `reid_cls` 는 몇 분씩 걸리고 무엇보다
# **성적이 바뀌는 일이라 사람이 보고 있어야 한다** — 자동으로 갈아 끼우면
# 어느 날 나빠진 것을 며칠 뒤에 안다.
#
# **끝난 것을 파일에 적는다** (`.guides/web/operations.md` §5). 로그만 남기는
# cron 은 검사의 환상이다 — 호스트 crontab 에는 `MAILTO` 가 없어서 죽어도
# 아무 데도 안 간다. 2026-08-27~29 에 사흘 내리 죽었는데 **로그를 볼 일이
# 없어서 몰랐다.** 그래서 상태를 `db/` 옆에 적고 **`/healthz` 가 그것을
# 읽는다** — 이미 사람이 보는 자리에 얹는 것이 요점이다.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

# **컨테이너가 볼 수 있는 자리라야 한다.** `logs/` 는 안 걸려 있고 `db/` 가
# `/app/hostdb` 로 걸린다 — 화면이 읽어야 하니 거기 둔다.
STATUS="${FIN_EXCHANGE_STATUS:-/srv/dolfinserver2/db/exchange_status.txt}"
STAGE=시작
note() { printf 'at=%s\nok=%s\nstage=%s\nnote=%s\n' \
             "$(date '+%F %T')" "$1" "$STAGE" "${2:-}" > "$STATUS" 2>/dev/null || true; }
# **`ERR` 이 아니라 `EXIT` 이다.** `ERR` 은 `exit` 을 안 잡는데, 사흘을 죽인 그
# 경우가 정확히 `exit` 이다 — `_venv.sh` 가 python 을 못 찾아 스스로 나간다.
finish() {
    local rc=$?
    [ "$rc" -eq 0 ] && [ "$STAGE" = 끝 ] && return 0
    note 0 "$STAGE 에서 멎었다 (rc=$rc)"
    echo "!!!!! 주고받기 실패 — $STAGE (rc=$rc)"
    return 0
}
trap finish EXIT

# **환경부터 본다.** 사흘을 죽인 것이 여기였다. ssh 로 저쪽 것을 담아 온
# **뒤에** 죽으면 저쪽에 임시 파일이 남고, 무엇보다 1초에 알 것을 1분 뒤에
# 안다 (`deploy/_venv.sh` 가 `import django` 까지 해 본다)
source "$REPO/deploy/_venv.sh"

echo "===== $(date '+%F %T') 주고받기 ====="
echo "    python  $PY"
STAGE=되받기
echo "--- 되받기 (GCP → 여기) ---"
deploy/gcp/from_reid_to_work.sh
STAGE=보내기
echo "--- 보내기 (여기 → GCP) ---"
deploy/gcp/from_work_to_reid.sh
STAGE=끝
note 1 "$("$PY" manage.py shell -c 'from finseg.models import Individual, Identification
print("개체", Individual.objects.filter(kind="").count(),
      "· 개체판정", Identification.objects.count())' 2>/dev/null | tail -1)"
echo "===== 끝 ====="
echo "분류기를 다시 배우려면 (손으로):"
echo "  python manage.py reid_cls --epochs 2000 --fit-all reid/v3/cls-dinov2.npz"
echo "  deploy/gcp/from_work_to_reid.sh"
