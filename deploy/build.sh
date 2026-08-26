#!/bin/bash
# 이미지 하나 굽는다. 버전 → 테스트 → build (→ push).
#
#   deploy/build.sh 0.1.0            # 굽기만
#   deploy/build.sh 0.1.0 --push     # Docker Hub 까지 (GCP 로 보낼 때)
#   deploy/build.sh 0.1.0 --no-test
#
# **테스트가 깨지면 push 전에 멈춘다.** 깨진 이미지가 레지스트리에 올라가면
# 그때부터는 배포 사고가 아니라 회수 작업이 된다.
#
# 이 기계(m710q)는 빌드 자리이자 시험 자리다. 그래서 최소한 지킬 것은
# **호스트에서 라이브 DB 에 쓰지 않는 것** — 빌드는 DB 를 안 건드리고
# 마이그레이션은 컨테이너 entrypoint 만 돌린다. `manage.py test` 도
# 인메모리 DB 를 쓴다 (`fin.db` 를 안 연다).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-honestjung/dolfinserver2}"
VERSION_FILE="$REPO/finweb/version.py"

VERSION=""
PUSH=0
RUN_TESTS=1
for a in "$@"; do
    case "$a" in
        --push) PUSH=1 ;;
        --no-test|--fast) RUN_TESTS=0 ;;
        *) VERSION="$a" ;;
    esac
done
[ -n "$VERSION" ] || { echo "사용법: deploy/build.sh X.Y.Z [--push] [--no-test]"; exit 1; }

cd "$REPO"
echo "▶ $IMAGE:$VERSION  (호스트 $(hostname))"

if [ "$RUN_TESTS" = "1" ]; then
    echo "=== [1/3] 테스트 ==="
    # 판정 규칙·좌표 사상·규칙 표시. `fin.db` 를 안 건드린다.
    python manage.py test
else
    echo "=== [1/3] 테스트 건너뜀 ==="
fi

echo "=== [2/3] 버전 $VERSION ==="
CURRENT="$(grep -oP '__version__ = "\K[^"]+' "$VERSION_FILE")"
if [ "$CURRENT" != "$VERSION" ]; then
    sed -i "s/__version__ = \"$CURRENT\"/__version__ = \"$VERSION\"/" "$VERSION_FILE"
    echo "  version.py: $CURRENT → $VERSION"
else
    echo "  version.py: 이미 $VERSION"
fi

# 새 자리를 세울 때 쓰는 본도 함께 올린다 — 안 그러면 `bootstrap.sh` 가
# 낡은 태그를 심어 **옛 이미지로 떠 놓고 새 판인 줄 안다.**
sed -i "s/^IMAGE_TAG=.*/IMAGE_TAG=$VERSION/" deploy/env.example
echo "  env.example: IMAGE_TAG=$VERSION"

echo "=== [3/3] build ==="
docker build -f deploy/Dockerfile -t "$IMAGE:$VERSION" -t "$IMAGE:latest" .

if [ "$PUSH" = "1" ]; then
    echo "=== push ==="
    docker push "$IMAGE:$VERSION"
    docker push "$IMAGE:latest"
fi

echo ""
echo "다음 — 시험 자리에 걸기:"
echo "  sed -i 's/^IMAGE_TAG=.*/IMAGE_TAG=$VERSION/' /srv/dolfinserver2/.env"
echo "  cd /srv/dolfinserver2 && docker compose up -d"
