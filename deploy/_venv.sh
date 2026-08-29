# **cron 에는 venv 가 없다.** Django 를 부르는 배포 스크립트는 이것을 먼저 읽고
# `$PY` 를 쓴다. 부르기 전에 `REPO` 가 서 있어야 한다.
#
#   source "$REPO/deploy/_venv.sh"
#   "$PY" manage.py <명령>
#
# **왜 있나.** 2026-08-27~29, 주고받기(`exchange.sh`)가 사흘 내리 죽었다 —
# `python3 manage.py` 가 cron 의 PATH 에서 시스템 파이썬을 잡아
# `ModuleNotFoundError: No module named 'django'` 로 **첫 줄에서** 끝났다.
# 한 번도 성공한 적이 없다.
#
# **사흘을 몰랐던 까닭이 둘이다.** 오류가 파이썬 traceback 이라 로그를 봐도
# "cron 에 venv 가 없다" 로 안 읽혔고, 무엇보다 **같은 crontab 의 다른 둘은
# 멀쩡히 돌았다** — `backup_db.py` 와 `pull_gcp_backup.sh` 는 표준 라이브러리
# (sqlite3)만 써서 시스템 파이썬으로 충분하다. 로그가 세 줄 중 두 줄은 `끝` 을
# 찍고 있어 **"cron 은 되고 있다" 로 보였다.**
#
# 그래서 여기서 하는 일은 찾는 것만이 아니라 **`import django` 까지 해 보는
# 것**이다 — 있는 것과 되는 것은 다르고, 갈리는 자리가 정확히 여기였다.

_fin_py() {
    local c
    for c in "${FIN_PY:-}" "$REPO/.venv/bin/python" \
             "$HOME/venv/dolfinserver2/bin/python" "$(command -v python3 || true)"; do
        [ -n "$c" ] && [ -x "$c" ] \
            && "$c" -c 'import django' >/dev/null 2>&1 && { echo "$c"; return 0; }
    done
    return 1
}

if ! PY="$(_fin_py)"; then
    # **사람이 읽을 말로 죽는다.** traceback 은 무엇이 없는지는 말해도
    # 무엇을 하라는지는 안 말한다
    {
        echo "!! Django 를 든 python 을 못 찾았다 — 아무것도 안 하고 멈춘다."
        echo "   cron 이면 PATH 에 venv 가 없어서다. crontab 줄에 대 줄 것:"
        echo "     FIN_PY=\$HOME/venv/dolfinserver2/bin/python"
        echo "   찾아본 자리:"
        echo "     FIN_PY   ${FIN_PY:-(안 줬다)}"
        echo "     $REPO/.venv/bin/python"
        echo "     $HOME/venv/dolfinserver2/bin/python"
        echo "     PATH     $(command -v python3 || echo '(python3 이 없다)')"
    } >&2
    exit 1
fi
export PY
