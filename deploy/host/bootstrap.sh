#!/bin/bash
# `/srv` 자리를 만들고 nginx 를 건다. **root 가 필요하다** — /srv 는 root
# 소유고 그룹 쓰기가 없다.
#
#   sudo deploy/host/bootstrap.sh            # 운영  /srv/dolfinserver2      nginx 8085
#   sudo deploy/host/bootstrap.sh --test     # 시험  /srv/dolfinserver2-test nginx 8086
#
# 여러 번 돌려도 된다. **이미 있는 것은 안 덮는다** — `.env`(SECRET_KEY 가 들어
# 있다)도, `db/fin.db`(사람의 판정)도 손대지 않는다.
#
# **자료를 넣지 않는다.** 운영 DB 는 이 자리에 그대로 살고, 크롭·조각은 저장소
# 것을 읽기전용으로 걸며, 시험 DB 는 `deploy/host/test_db.sh` 가 NAS 백업에서
# 뜬다.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OWNER="${SUDO_USER:-jikhanjung}"
GROUP=devops

if [ "${1:-}" = "--test" ]; then
    KIND=시험; ROOT=/srv/dolfinserver2-test; PORT=8086
    CONF=/etc/nginx/sites-available/dolfinserver2-test.conf; SRC="$REPO/deploy/nginx.test.conf"
    COMPOSE="$REPO/deploy/docker-compose.test.yml"; ROLE=reid
else
    KIND=운영; ROOT=/srv/dolfinserver2; PORT=8085
    CONF=/etc/nginx/sites-available/dolfinserver2.conf; SRC="$REPO/deploy/nginx.conf"
    COMPOSE="$REPO/deploy/docker-compose.yml"; ROLE=work
fi

[ "$(id -u)" = "0" ] || { echo "root 로 돌려야 한다: sudo $0 ${1:-}"; exit 1; }

echo "▶ $KIND — $ROOT (소유 $OWNER:$GROUP · 역할 $ROLE · nginx $PORT)"
# setgid — 안에 새로 생기는 것이 devops 그룹을 물려받는다. 컨테이너가 uid 를
# hostdb 주인에서 읽어 가므로 db/ 의 주인이 곧 서비스 프로세스의 uid 다.
install -d -o "$OWNER" -g "$GROUP" -m 2775 \
    "$ROOT" "$ROOT/db" "$ROOT/staticfiles" "$ROOT/logs" \
    "$ROOT/scripts" "$ROOT/backup" "$ROOT/backup/hourly"

# 사진은 운영 자리 하나만 둔다 — NAS 로 가는 링크라 두 벌일 이유가 없다.
if [ "$KIND" = "운영" ]; then
    install -d -o "$OWNER" -g "$GROUP" -m 2775 "$ROOT/photos"
    if [ ! -e "$ROOT/photos/nas" ]; then
        # DB 가 경로를 `nas/2016/03/15/…` 로 들고 있어 **이 이름 그대로** 있어야 한다.
        ln -s /nas/JikhanJung/dolfinimage "$ROOT/photos/nas"
        chown -h "$OWNER:$GROUP" "$ROOT/photos/nas"
        echo "  photos/nas → /nas/JikhanJung/dolfinimage"
    fi
fi

echo "▶ compose · 안내 페이지"
install -o "$OWNER" -g "$GROUP" -m 664 "$COMPOSE" "$ROOT/docker-compose.yml"
install -o "$OWNER" -g "$GROUP" -m 664 "$REPO/deploy/maintenance.html" "$ROOT/maintenance.html"

echo "▶ .env"
if [ -f "$ROOT/.env" ]; then
    echo "  이미 있다 — 안 덮는다 (IMAGE_TAG·FIN_ROLE 만 바꿀 것)"
else
    SECRET=$(python3 -c 'import secrets; print(secrets.token_urlsafe(50))')
    sed -e "s|^FIN_SECRET_KEY=.*|FIN_SECRET_KEY=$SECRET|" \
        -e "s|^FIN_ROLE=.*|FIN_ROLE=$ROLE|" "$REPO/deploy/env.example" > "$ROOT/.env"
    # compose 가 저장소 자리를 알아야 크롭·조각을 읽기전용으로 건다.
    echo "REPO=$REPO" >> "$ROOT/.env"
    chown "$OWNER:$GROUP" "$ROOT/.env"; chmod 600 "$ROOT/.env"
    echo "  새로 만들었다 (SECRET_KEY 를 뽑고 FIN_ROLE=$ROLE 로 뒀다)"
fi

# 시간별 백업은 **운영 자리에만** 건다. 시험 자리의 DB 는 백업에서 뜬 사본이라
# 그것을 또 뜨는 것은 값이 없고, 로그만 두 배가 된다.
if [ "$KIND" = "운영" ]; then
    echo "▶ 백업 스크립트"
    install -o "$OWNER" -g "$GROUP" -m 775 "$REPO/deploy/host/backup_db.py" "$ROOT/scripts/backup_db.py"
    echo "  $ROOT/scripts/backup_db.py — cron 은 아래를 볼 것"
    echo "    0 * * * * python3 $ROOT/scripts/backup_db.py >> $ROOT/logs/backup.log 2>&1"
fi

echo "▶ nginx"
install -m 644 "$SRC" "$CONF"
ln -sfn "$CONF" "/etc/nginx/sites-enabled/$(basename "$CONF")"
nginx -t && systemctl reload nginx
echo "  $PORT 로 열었다"

echo ""
if [ "$KIND" = "시험" ]; then
    echo "다음 — DB 를 NAS 백업에서 뜨고 (사용자 계정으로):"
    echo "  deploy/host/test_db.sh"
fi
echo "  cd $ROOT && docker compose up -d"
