#!/bin/bash
# **m710q 에서** 돌린다. GCP 로 호스트 파일을 밀고 bootstrap 을 부른다.
#
#   deploy/gcp/install.sh
#
# 형제(cdGTS)는 이미지에서 호스트 파일을 꺼내는 self-heal 방식인데, 여기는
# 아직 이미지를 Docker Hub 로 안 올렸다. **먼저 도는 길을 만들고**, 이미지
# 경로는 push 를 시작할 때 맞춘다 (`TODOs.md`).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HOST="${FIN_GCP_HOST:-dolfinid}"
ROOT=/srv/dolfinserver2

echo "▶ $HOST 로 호스트 파일을 민다"
ssh "$HOST" "sudo install -d -o \$(id -u) -g \$(id -g) -m 775 $ROOT $ROOT/_incoming"
rsync -a --info=stats0 \
    "$REPO/deploy/gcp/docker-compose.yml" \
    "$REPO/deploy/gcp/nginx.conf" \
    "$REPO/deploy/gcp/bootstrap.sh" \
    "$REPO/deploy/gcp/deploy.sh" \
    "$REPO/deploy/gcp/smoke.sh" \
    "$REPO/deploy/host/backup_db.py" \
    "$REPO/deploy/maintenance.html" \
    "$REPO/deploy/env.example" \
    "$HOST:$ROOT/_incoming/"
ssh "$HOST" "chmod +x $ROOT/_incoming/*.sh $ROOT/_incoming/backup_db.py"

echo "▶ bootstrap"
ssh "$HOST" "sudo $ROOT/_incoming/bootstrap.sh"
