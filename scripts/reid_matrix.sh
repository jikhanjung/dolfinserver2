#!/bin/bash
# **re-ID 한 판을 통째로 잰다** — 백본 × 블록 수 × 조각 크기.
#
#   scripts/reid_matrix.sh                 # 있는 대로
#   SEEDS=3 scripts/reid_matrix.sh         # 폭까지
#
# **GPU 에서 돌리라고 만든 것이다.** m710q(CPU 4코어)에서는 한 칸에 30분~3시간이라
# 씨앗을 못 흔든다. RTX 8000 이면 씨앗 3으로 전부 돌려도 몇십 분이다.
#
# ## 미리 알 것
#
# - **bf16 을 켜지 말 것.** 2080ti·RTX 8000 은 Turing(sm_75)이라 그것이 없다
# - **`--batch` 를 키울 것.** 기본 16은 CPU 기준이다. 48GB 면 128도 된다
# - **`--as-of` 를 붙일 것.** 정답이 늘면 문제가 달라져 세로로 못 견준다
# - 얼린 갈래는 GPU 를 안 쓴다 (선형 한 층이라 옮기는 비용이 더 크다)
# - **`DIR128` 과 `DIR224` 는 같은 날 뜬 짝이어야 한다.** 격자는 그때그때의
#   사람 판정으로 걸러지므로, 날이 다르면 상자 수가 달라지고 **그 차이가
#   조각 크기의 차이인 척한다.** 실제로 `reid/v3`(08-27, 7,912)와
#   224(09-01, 7,888)가 24개 어긋났다 — 그래서 `reid/v3-128` 을 함께 떴다
#   (`HANDOFF` 의 2026-09-01 절 · 묶음의 `README.md`)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SEEDS=${SEEDS:-1}
BATCH=${BATCH:-64}
DIR128=${DIR128:-$FIN_REID}
DIR224=${DIR224:-}          # 224 조각 격자. 없으면 건너뛴다
BLOCKS=${BLOCKS:-"1 2 3 4 6"}

say() { echo; echo "########## $* ($(date '+%m-%d %H:%M')) ##########"; }
cls() { python manage.py reid_cls --folds 5 --seeds "$SEEDS" --batch "$BATCH" "$@" 2>&1 \
        | grep -vE "Warning|warn|xFormers" | grep -E "^자료|캐시 [0-9,]+장|로 돈다|^  합|분류기 top"; }

say "얼린 채 (기준)"
cls --dir "$DIR128"

for n in $BLOCKS; do
  say "dinov2 · ${n}블록"
  cls --dir "$DIR128" --unfreeze "$n"
done

if [ -n "$DIR224" ]; then
  say "224 조각 · 얼린 채"
  cls --dir "$DIR224"
  for n in $BLOCKS; do
    say "224 조각 · ${n}블록"
    cls --dir "$DIR224" --unfreeze "$n"
  done
fi

# **백본을 바꿀 때는 임베딩을 먼저 뽑는다** — kNN 기준선이 그것으로 난다.
# `--no-chain` 이 없으면 `items.json` 의 `sim` 을 덮어써 **화면 정렬이 바뀐다**
for bb in dinov3s dinov2b; do
  f="emb-$bb.npz"
  if [ ! -f "$DIR128/$f" ]; then
    say "$bb 임베딩"
    python manage.py reid_chips --out "$DIR128" --emb-only --no-chain \
        --backbone "$bb" --emb-name "$f" 2>&1 | tail -2
  fi
  say "$bb · 얼린 채"
  cls --dir "$DIR128" --emb "$f"
  for n in $BLOCKS; do
    say "$bb · ${n}블록"
    cls --dir "$DIR128" --emb "$f" --backbone "$bb" --unfreeze "$n"
  done
done
say "끝"
