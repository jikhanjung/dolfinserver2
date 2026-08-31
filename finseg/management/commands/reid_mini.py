"""**남에게 줄 수 있는 최소 한 벌**로 뜬다 — 조각과 라벨만.

    python manage.py reid_mini --out reid-mini.npz

## 왜 있나

밖에서(다른 기계·다른 사람) 이 문제를 돌려 보려면 무엇이 필요한지가 안
드러나 있었다. 사진 74GB 도 크롭 715MB 도 `fin.db` 도 필요 없다 — **조각이
이미 밑동으로 회전·크기를 맞춰 잘려 있어서**, 정답이 붙은 것만 뽑으면
**5MB 남짓**이다.

## 라벨이 셋이다 — 하나만 주면 성적이 부푼다

    individual   맞혀야 할 것 (닫힌 판)
    facing       **좌·우를 섞으면 안 된다**
    day          **날로 갈라야 한다**

`day` 가 왜 있나: 무작위로 train/test 를 가르면 **같은 날 연속 프레임이 양쪽에
나뉜다.** 몇 초 사이에 찍힌 거의 같은 그림이라 외운 것을 다시 맞히는 셈이다.
실측으로 같은 날 짝은 AUC 0.698, 날을 건너뛴 짝은 0.480 이다 —
**그 갈림이 이 과제의 전부**다.

`facing` 이 왜 있나: 조각을 세울 때 `reid.normalize` 가 앞쪽이 늘 왼쪽에 오게
**한쪽을 거울처럼 뒤집는다.** 좌·우를 한 통에 넣으면 같은 개체가 두 갈래로
흩어진다. 그래서 이 저장소는 좌·우 모델을 따로 배우고 따로 잰다.

**규칙을 npz 안에 글로 적어 둔다** (`README`). 받는 사람이 그것을 모르고
무작위로 가르면 부푼 숫자를 내는데, 부풀었다는 것이 안 보인다.
"""
import numpy as np
from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
from pathlib import Path

RULES = """이 꾸러미로 재현하려면 규칙 셋을 지켜야 한다.

1. 좌·우를 따로 배우고 따로 잰다 (`facing`).
   조각은 앞쪽이 늘 왼쪽에 오게 세운 것이라 한쪽이 거울상이다.
   섞으면 같은 개체가 두 갈래로 흩어진다.

2. 날로 가른다 (`day`). 무작위로 가르면 안 된다.
   같은 날 연속 프레임은 몇 초 사이에 찍힌 거의 같은 그림이다.
   실측: 같은 날 짝 AUC 0.698 · 날 건너뛴 짝 0.480.

3. 닫힌 판이다. 배운 적 없는 개체는 질의에서 뺀다.
   제주 남방큰돌고래는 개체군이 100~120이라 개체 수의 천장이 개체군 크기다.
   묻는 것은 "카탈로그의 몇 번인가" 이지 "처음 보는 개체인가" 가 아니다.

성적 (5폴드 · 관찰일을 5벌로 돌린다 · 배운 적 없는 개체는 뺀다):
   기준선 kNN (얼린 DINOv2 ViT-S/14)   top-1 26.7%
   + 선형 한 층                              49.5%
   + 마지막 블록까지 녹임                     59.9%
   + 마지막 두 블록까지 녹임                  63.9%
"""


class Command(BaseCommand):
    help = "조각과 라벨만 담은 최소 꾸러미를 낸다 (밖에서 돌려 보라고)"

    def add_arguments(self, p):
        p.add_argument("--out", required=True, help="낼 `.npz` 파일")
        p.add_argument("--dir", default=str(settings.FIN_REID),
                       help="격자. 기본은 화면이 보는 것")
        p.add_argument("--chips", default="chips.npz")

    def handle(self, **o):
        from finseg import reid
        from finseg.models import Box

        w = self.stdout.write
        root = Path(o["dir"])
        f = root / o["chips"]
        if not f.exists():
            raise CommandError(f"{f} 가 없다 — `reid_chips` 가 만든다")
        z = np.load(f)
        ids, chip, fac = z["box_id"], z["chip"], z["facing"]

        cat = reid.catalog()
        of = {b: i for i, v in cat.items() for b in v}
        if not of:
            raise CommandError("정답이 하나도 없다")
        day = dict(Box.objects.filter(id__in=list(of))
                   .values_list("id", "image__obsdate"))
        sel = np.flatnonzero([int(b) in of for b in ids])
        # **날이 없는 것은 뺀다.** 규칙 2를 못 지키는 줄이라 넣으면 받는 쪽이
        # 그 줄만 무작위로 가르게 된다
        sel = np.array([i for i in sel if day.get(int(ids[i]))])
        if not len(sel):
            raise CommandError("날이 붙은 정답이 없다")

        np.savez_compressed(
            o["out"],
            box_id=ids[sel], chip=chip[sel], facing=fac[sel],
            individual=np.array([of[int(b)] for b in ids[sel]], dtype=np.int64),
            day=np.array([str(day[int(b)]) for b in ids[sel]]),
            README=np.array([RULES]),
        )
        n_ind = len({of[int(b)] for b in ids[sel]})
        size = Path(o["out"]).stat().st_size
        w(f"{o['out']}  {size / 1e6:.1f} MB")
        w(f"  조각 {len(sel):,} · 개체 {n_ind} · 관찰일 "
          f"{len({str(day[int(b)]) for b in ids[sel]})}")
        w(f"  좌 {int((fac[sel] == 'left').sum())} · "
          f"우 {int((fac[sel] == 'right').sum())} · "
          f"조각 {chip.shape[1]}×{chip.shape[2]}")
        w("  **규칙 셋이 `README` 칸에 들어 있다** — 좌우·날·닫힌 판")
