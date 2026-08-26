#!/bin/bash
# GCP(`dolfinid-2`) 에서 도는 자리 만들기. **거기서** 돌린다 —
# `deploy/gcp/install.sh` 가 파일을 밀어 넣고 이것을 불러 준다.
#
#   sudo /srv/dolfinserver2/bootstrap.sh
#
# 여러 번 돌려도 된다. **이미 있는 것은 안 덮는다** — `.env`(SECRET_KEY)도
# `db/fin.db`(사람의 판정)도 손대지 않는다.
set -euo pipefail

ROOT=/srv/dolfinserver2
OWNER="${SUDO_USER:-honestjung}"
GROUP=devops
CONF=/etc/nginx/sites-available/dolfinserver2.conf

[ "$(id -u)" = "0" ] || { echo "root 로 돌려야 한다: sudo $0"; exit 1; }
[ -d "$ROOT/_incoming" ] || { echo "$ROOT/_incoming 이 없다 — install.sh 로 파일을 먼저 밀 것"; exit 1; }
IN="$ROOT/_incoming"

echo "▶ 자리 (소유 $OWNER:$GROUP · 역할 reid · nginx tailnet:8085)"
install -d -o "$OWNER" -g "$GROUP" -m 2775 \
    "$ROOT" "$ROOT/db" "$ROOT/crops" "$ROOT/reid" "$ROOT/staticfiles" \
    "$ROOT/logs" "$ROOT/scripts" "$ROOT/backup" "$ROOT/backup/hourly"

echo "▶ compose · 안내 페이지 · 백업 스크립트"
install -o "$OWNER" -g "$GROUP" -m 664 "$IN/docker-compose.yml" "$ROOT/docker-compose.yml"
install -o "$OWNER" -g "$GROUP" -m 664 "$IN/maintenance.html"   "$ROOT/maintenance.html"
install -o "$OWNER" -g "$GROUP" -m 775 "$IN/backup_db.py"       "$ROOT/scripts/backup_db.py"
install -o "$OWNER" -g "$GROUP" -m 775 "$IN/smoke.sh"           "$ROOT/smoke.sh"
install -o "$OWNER" -g "$GROUP" -m 775 "$IN/deploy.sh"          "$ROOT/deploy.sh"

echo "▶ .env"
if [ -f "$ROOT/.env" ]; then
    echo "  이미 있다 — 안 덮는다 (IMAGE_TAG 만 바꿀 것)"
else
    SECRET=$(python3 -c 'import secrets; print(secrets.token_urlsafe(50))')
    sed -e "s|^FIN_SECRET_KEY=.*|FIN_SECRET_KEY=$SECRET|" \
        -e "s|^FIN_ROLE=.*|FIN_ROLE=reid|" "$IN/env.example" > "$ROOT/.env"
    # **사진이 없다.** 화면이 "이 기계에 없다" 고 말하게 그대로 둔다.
    sed -i "s|^FIN_PHOTOS=.*|FIN_PHOTOS=/app/photos|" "$ROOT/.env"
    chown "$OWNER:$GROUP" "$ROOT/.env"; chmod 600 "$ROOT/.env"
    echo "  새로 만들었다 (SECRET_KEY 를 뽑고 FIN_ROLE=reid 로 뒀다)"
fi

echo "▶ nginx (tailnet 전용)"
install -m 644 "$IN/nginx.conf" "$CONF"
ln -sfn "$CONF" /etc/nginx/sites-enabled/dolfinserver2.conf
nginx -t && systemctl reload nginx
echo "  100.126.94.48:8085 로 열었다 — **공개 포트가 아니다**"

echo "▶ cron"
CRON="0 * * * * FIN_NAS_DIR= /usr/bin/python3 $ROOT/scripts/backup_db.py >> $ROOT/logs/backup.log 2>&1"
if sudo -u "$OWNER" crontab -l 2>/dev/null | grep -q "dolfinserver2/scripts/backup_db.py"; then
    echo "  이미 걸려 있다"
else
    # **DB 를 넣기 전에 걸면 매시 '소스 없음' 으로 알림이 운다.** seed 뒤에 건다.
    if [ -s "$ROOT/db/fin.db" ]; then
        (sudo -u "$OWNER" crontab -l 2>/dev/null; \
         echo "# dolfinserver2 — fin.db 시간별 백업. 오프사이트는 m710q 가 당겨간다"; \
         echo "$CRON") | sudo -u "$OWNER" crontab -
        echo "  걸었다"
    else
        echo "  **아직 안 건다** — db/fin.db 가 없다. seed 뒤에 이것을 다시 돌릴 것"
    fi
fi

echo ""
echo "다음 — m710q 에서 자료를 밀고:"
echo "  deploy/gcp/seed.sh"
echo "그 다음 여기서:"
echo "  cd $ROOT && docker compose up -d && ./smoke.sh"
