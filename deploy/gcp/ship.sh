#!/bin/bash
# **m710q 에서** 돌린다. 이미지를 ssh 로 실어 나른다.
#
#   deploy/gcp/ship.sh 0.3.0
#
# 레지스트리를 안 거치는 이유는 값이 아니라 **정하지 않았기 때문**이다 —
# 형제들은 Docker Hub 를 쓰지만(`honestjung/cdgts` 등), 이 이미지를 공개
# 레지스트리에 올릴지는 따로 정할 일이다. 157MB 라 tailscale 로 나르면 몇 분이고,
# 그 사이 아무것도 밖으로 안 나간다. push 로 바꾸려면 `build.sh --push` 하나면 된다.
set -euo pipefail
V="${1:-}"; HOST="${FIN_GCP_HOST:-dolfinid}"
[ -n "$V" ] || { echo "사용법: $0 X.Y.Z"; exit 1; }
IMG="honestjung/dolfinserver2:$V"

docker image inspect "$IMG" >/dev/null || { echo "$IMG 가 여기 없다 — 먼저: deploy/build.sh $V"; exit 1; }
echo "▶ $IMG → $HOST  ($(docker image inspect -f '{{.Size}}' "$IMG" | awk '{printf "%.0f MB", $1/1024/1024}'))"
docker save "$IMG" | gzip -1 | ssh "$HOST" 'gunzip | docker load'
ssh "$HOST" "docker image inspect $IMG >/dev/null && echo '  실렸다'"
