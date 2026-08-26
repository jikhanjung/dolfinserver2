#!/bin/bash
# 떴는지, **그리고 무엇으로 떴는지** 본다.
#
#   /srv/dolfinserver2/smoke.sh [기대버전]
#
# 200 만 보고 넘어가면 안 되는 이유가 이 앱에 하나 더 있다 — **역할이 뒤바뀌어도
# 화면은 멀쩡히 200 을 낸다.** 개체 판정을 받을 자리가 `work` 로 떠 있으면
# 그날 판정을 한 건도 못 받는다.
set -euo pipefail

URL="${FIN_SMOKE_URL:-http://127.0.0.1:8012}"
WANT_VERSION="${1:-}"
WANT_ROLE="${FIN_SMOKE_ROLE:-reid}"

for i in $(seq 1 30); do
    code=$(curl -s -o /tmp/smoke.json -w "%{http_code}" "$URL/healthz" || echo 000)
    [ "$code" = "200" ] && break
    sleep 2
done
[ "$code" = "200" ] || { echo "FAIL: /healthz 가 $code"; cat /tmp/smoke.json 2>/dev/null; exit 1; }

read -r status role version items < <(python3 -c "
import json; d=json.load(open('/tmp/smoke.json'))
print(d.get('status'), d.get('role'), d.get('version'), d.get('reid_items'))")

fail=0
echo "  status  $status";  [ "$status" = "ok" ] || { echo "    ↑ ok 가 아니다"; fail=1; }
echo "  role    $role";    [ "$role" = "$WANT_ROLE" ] || { echo "    ↑ $WANT_ROLE 이어야 한다"; fail=1; }
echo "  version $version"
if [ -n "$WANT_VERSION" ] && [ "$version" != "$WANT_VERSION" ]; then
    echo "    ↑ $WANT_VERSION 이어야 한다 — 옛 이미지가 떠 있다"; fail=1
fi
# **격자가 비어도 화면은 200 이다.** 조각이 덜 왔는지 분류가 안 됐는지는
# 화면만 봐서 안 갈린다 — 그래서 여기서 센다.
echo "  격자    $items"
[ "$items" -gt 0 ] 2>/dev/null || { echo "    ↑ 격자가 비었다"; fail=1; }

[ "$fail" = "0" ] && echo "PASS" || { echo "FAIL"; exit 1; }
