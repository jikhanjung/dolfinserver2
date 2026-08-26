#!/bin/bash
# **두 레인을 한 번에 돌린다.** m710q 에서, 하루 한 번.
#
#   deploy/gcp/exchange.sh
#   30 6 * * * /home/jikhanjung/projects/dolfinserver2/deploy/gcp/exchange.sh \
#                  >> /srv/dolfinserver2/logs/exchange.log 2>&1
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
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

echo "===== $(date '+%F %T') 주고받기 ====="
echo "--- 되받기 (GCP → 여기) ---"
deploy/gcp/from_reid_to_work.sh
echo "--- 보내기 (여기 → GCP) ---"
deploy/gcp/from_work_to_reid.sh
echo "===== 끝 ====="
echo "분류기를 다시 배우려면 (손으로):"
echo "  python manage.py reid_cls --epochs 2000 --fit-all reid/v3/cls-dinov2.npz"
echo "  deploy/gcp/from_work_to_reid.sh"
