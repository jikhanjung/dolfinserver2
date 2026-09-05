"""**아는 개체를 얼마나 맞히나** — 카탈로그를 클래스로 놓고 분류기를 배운다.

    python manage.py reid_cls --dir reid/v2 --emb emb-dinov2.npz
    python manage.py reid_cls --test-days 10 --folds 5 --seeds 3

## 왜 metric 이 아니라 분류인가

사람 re-ID 는 **처음 보는 사람**을 알아보는 문제라 학습 개체와 평가 개체를
겹치지 않게 나눈다. 감시 카메라가 날마다 모르는 사람을 만나기 때문이다.
**여기는 그 문제가 아니다** — 제주 남방큰돌고래는 개체군이 100~120이라
**개체 수의 천장이 개체군 크기**이고, 묻는 것은 "카탈로그의 몇 번인가" 다.

그래서 이 명령은 **닫힌 집합 분류**다. 실측으로 같은 자료에서 metric 쪽은
기준선을 못 넘었지만(`reid_probe --hold-individuals`) 닫힌 판에서는 얼린 특징
위의 선형 사영 하나가 28.0% → 43.0% 를 냈다. 그 자리를 제대로 파는 것이 여기다.

## 날로 가른다 — 조각으로 가르지 않는다

**같은 날 연속 프레임을 학습과 평가에 나눠 담으면 그날 조명을 외운 것이 성적이
된다.** 이 저장소가 이미 한 번 속은 자리다. 그래서 관찰일을 통째로 갈라
**배운 날과 재는 날이 안 겹친다.**

## 좌현과 우현은 따로 배운다

`reid.normalize` 가 한쪽 무리를 거울처럼 뒤집으므로 두 면의 특징은 서로 견줄 수
없다. 한 통에 넣고 배우면 모델이 개체가 아니라 **어느 쪽 면인가**를 배운다.
그래서 면마다 분류기를 따로 두고, 클래스는 개체다.

## 기준선을 같은 갈래에서 함께 낸다

**가장 닮은 조각의 개체를 고르는 것**(kNN)이 지금 화면이 하는 일이고 그것이
기준선이다. 배운 것과 안 배운 것을 다른 갈래에서 재면 무엇이 오른 것인지 알 수
없으므로, **같은 날 갈래·같은 질의·같은 후보**에 둘 다 댄다.
"""
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from finseg import reid


def split_days(days, cnt, k):
    """관찰일을 k벌로 나눈다. **큰 날부터 번갈아 담는다.**

    무작위로 담으면 33장짜리 날 여럿이 한 폴드에 몰려 그 폴드만 잼이 두꺼워진다.
    번갈아 담으면 폴드마다 크기가 비슷해지고, **씨앗을 안 타므로** 두 자를
    견줄 때 같은 문제를 푼다.

    번갈아 담기(`j % k`)로는 덜 고르다 — 큰 날 열 개를 5벌로 번갈아 담으면
    가장 무거운 폴드와 가벼운 폴드가 조각 12장 차이가 났다. **그때그때 가장
    가벼운 폴드에 담으면** 4장으로 준다. 폴드마다 잼이 비슷해야 폴드별 성적을
    나란히 읽을 수 있다.

    빈 폴드를 안 만든다 — 부르는 쪽이 `k <= len(days)` 를 이미 막는다.
    """
    fold = [[] for _ in range(k)]
    load = [0] * k
    for d in sorted(days, key=lambda d: (-cnt[d], d)):
        j = min(range(k), key=lambda i: (load[i], i))
        fold[j].append(d)
        load[j] += cnt[d]
    return fold


class Command(BaseCommand):
    help = "카탈로그를 클래스로 놓고 닫힌 집합 분류기를 배운다"

    def add_arguments(self, p):
        p.add_argument("--dir", default=str(settings.FIN_REID), help="기본은 **화면이 보는 격자**(`FIN_REID`) — 박아 두면 격자를 갈아 끼울 때마다 조용히 옛것을 잰다")
        p.add_argument("--emb", default="emb-dinov2.npz",
                       help="`--dir` 안의 임베딩 파일 이름")
        p.add_argument("--test-days", type=int, default=8)
        p.add_argument("--epochs", type=int, default=2000,
                       help="**400 이었다 — 덜 배운 자리였다** (2026-08-29). 같은 폴드에서 400 → 800 한 칸에 top-1 이 +1.6%%p 뛰고 1200 부터 평평하다. 스크립트들(`exchange.sh`·`from_reid_to_work.sh`)은 이미 `--epochs 2000` 을 안내하고 있었는데 **기본값이 400 이고 재는 명령이 기본값을 써서**, 기록에 남은 성적이 전부 덜 배운 것이었다. 안내와 기본값을 한 자리로 모은다")
        p.add_argument("--lr", type=float, default=0.02)
        p.add_argument("--wd", type=float, default=1e-3,
                       help="**L2 — 손잡이 값의 제곱에 벌점.** 큰 값을 세게 누르되 "
                            "**0 으로는 안 만든다.** 개체당 조각이 적어 과적합이 "
                            "쉽다. 384차원에서 0~1e-2 를 훑었을 때는 성적이 한 "
                            "자리도 안 움직였는데, 그때는 축을 안 자른 상태였다")
        p.add_argument("--l1", type=float, default=0.0,
                       help="**L1 — 손잡이 값의 절대값에 벌점.** 작은 값도 같은 "
                            "힘으로 밀어 **쓸모없는 축을 아예 0 으로 만든다.** "
                            "PCA 가 축을 잘라 이겼으니(384 → 256 에서 +2.4%%p) "
                            "**라벨을 보고 자르는 쪽**도 물어볼 값이 있다. "
                            "가중치에만 걸고 편향에는 안 건다")
        p.add_argument("--unfreeze", type=int, default=0, metavar="N",
                       help="**백본 마지막 N블록을 녹여 함께 배운다** (0 이면 지금처럼 "
                            "얼린 특징 위의 선형 한 층). 얼린 백본 위에서 진 것들"
                            "(ViT-B · 2층 MLP · ArcFace 여백)과 **축이 다르다** — "
                            "저것들은 특징을 안 건드렸다. `TODOs` 가 걸어 둔 조건"
                            "(조각 1,200)은 2026-08-29 에 넘었다. "
                            "**앞 블록은 한 번만 계산해 캐시한다** — 얼려 뒀으니 "
                            "매 에폭 같은 값이 나오고, 그것만으로 4.3배 빠르다"
                            "(m710q CPU 에서 72 → 16.6 ms/장). "
                            "**그래서 증강을 못 쓴다**: 앞이 매번 같은 입력을 본다. "
                            "증강까지 하려면 캐시를 버려야 하고 4배가 든다")
        p.add_argument("--deep-epochs", type=int, default=20, metavar="N",
                       help="`--unfreeze` 일 때의 에폭. 얼린 갈래의 `--epochs`(2000)와 "
                            "**자릿수가 다르다** — 저쪽은 384차원 위의 선형 한 층을 "
                            "밑바닥부터 배우고, 이쪽은 이미 배운 블록을 조금 옮긴다")
        p.add_argument("--lr-deep", type=float, default=1e-4,
                       help="`--unfreeze` 일 때의 학습률. 얼린 갈래의 0.02 를 그대로 "
                            "쓰면 배운 블록이 무너진다")
        p.add_argument("--lr-blocks", type=float, default=None, metavar="LR",
                       help="블록 쪽만 다른 학습률. 안 주면 `--lr-deep` 그대로 — "
                            "머리는 밑바닥부터 배우고 블록은 배운 것을 옮기는 "
                            "것이라, 둘을 가르는 것이 층별 학습률의 절반이다")
        p.add_argument("--lr-decay", type=float, default=1.0, metavar="G",
                       help="블록 학습률을 꼭대기에서 아래로 갈수록 G 배씩 줄인다 "
                            "(ViT 관례 0.65~0.9). 1.0 이면 지금과 같다 — 기본을 "
                            "바꾸지 않아야 앞의 성적표가 그대로 선다")
        p.add_argument("--batch", type=int, default=16,
                       help="`--unfreeze` 일 때의 묶음 크기")
        p.add_argument("--device", default=None,
                       help="cuda | cpu. 안 주면 있으면 cuda. **`--unfreeze` 에만 "
                            "든다** — 얼린 갈래는 384차원 위의 선형 한 층이라 "
                            "GPU 로 옮기는 값이 없다(옮기는 비용이 더 크다). "
                            "**bf16 은 안 쓴다**: Turing(sm_75, 2080ti·RTX 8000)에 "
                            "그것이 없어 조용히 느려지거나 터진다 "
                            "(`segment.py` 의 `autocast_dtype`)")
        p.add_argument("--backbone", default="dinov2",
                       help="`--unfreeze` 가 녹일 백본. **`--emb` 가 어느 백본으로 "
                            "뽑힌 것인지와 같아야 한다** — 다르면 캐시와 임베딩이 "
                            "다른 그물에서 나와 성적이 아무 뜻이 없다. 차원이 안 "
                            "맞으면 멈춘다. 이름은 `reid_chips` 의 `BACKBONES` 와 "
                            "같다 (`dinov2` S/14 · `dinov2b` B/14 · `dinov2l` L/14)")
        p.add_argument("--chips", default="chips.npz",
                       help="`--unfreeze` 가 읽을 조각 꾸러미 (`reid_chips` 가 쓴다)")
        p.add_argument("--as-of", type=int, default=None, metavar="개체판정번호",
                       help="**그때의 카탈로그로 잰다** — `Identification` 번호가 "
                            "이 값 이하인 정답만 쓴다(`reid_eval` 과 같은 손잡이). "
                            "**자료가 늘면 문제가 달라져 성적을 세로로 못 견준다** — "
                            "2026-08-29 하루에도 44.9 → 49.5 였는데 코드가 아니라 "
                            "판정이 는 것이었다. 문제를 고정하고 자·자료를 흔들려면 "
                            "이것이 필요하다. **`kind` 는 못 되살린다** — 그때 "
                            "임시보관함이던 것이 지금 개체면 개체로 센다")
        p.add_argument("--min-chips", type=int, default=0, metavar="N",
                       help="**조각이 N장 이상인 개체만** 놓고 배운다. `자료가 더 "
                            "있으면 나아지나` 를 묻는 자리다 — 잰 곡선이 4~9장 "
                            "11.1%% → 10~19장 30.6%% → 20장 이상 33.8%% 였다. "
                            "**혼자 쓰면 답이 안 나온다**: 개체당 조각이 느는 "
                            "동시에 **개체 수가 줄어** 문제가 쉬워진다. "
                            "`--sample-ind` 로 개체 수만 같게 맞춘 대조군을 함께 "
                            "돌릴 것")
        p.add_argument("--sample-ind", type=int, default=0, metavar="K",
                       help="**개체를 K마리만 아무렇게나 골라** 놓고 배운다. "
                            "`--min-chips` 의 대조군이다 — 클래스 수를 같게 맞춰 "
                            "놓으면 남는 차이가 조각 수에서 온 것이 된다. "
                            "고르는 씨앗은 `--ind-seed` 다(머리 씨앗과 따로 둔다 — "
                            "**같은 개체 무리를 놓고 머리만 흔들어야** 둘을 가른다)")
        p.add_argument("--ind-seed", type=int, default=0,
                       help="`--sample-ind` 가 개체를 고르는 씨앗")
        p.add_argument("--calib", action="store_true",
                       help="**확신이 읽히나** 를 함께 낸다. 성적은 `무엇을 골랐나` "
                            "만 재는데, 화면은 그 옆에 `98.3%%` 같은 퍼센트를 "
                            "띄운다 — 그 숫자가 뜻이 없으면 사람이 잘못 믿는다. "
                            "`1등 확률` 과 `1·2등 차이` 를 각각 자로 써서 확신 "
                            "상위 몇 %%의 정확도를 내고, **화면이 말한 퍼센트와 "
                            "실제가 얼마나 벌어지는지**(ECE)를 잰다")
        p.add_argument("--arcface", action="store_true",
                       help="**여백 손실로 바꾼다** (ArcFace). 지금 손실은 맞히기만 "
                            "하면 되는데, 이것은 `x` 와 개체 자를 둘 다 길이 1 로 "
                            "재서 점수를 **각도의 코사인**으로 만들고, 정답 개체에만 "
                            "각도를 `--arc-m` 만큼 더해 놓고 채점한다 — "
                            "`조금 이기면 안 되고 여백만큼 이겨라`. 같은 개체를 한 "
                            "점으로 모으고 개체끼리 밀어내라는 뜻이다. "
                            "**열린 판에서는 이미 진 가족이다**(`devlog/20260824_004` "
                            "의 InfoNCE·triplet) — 다만 그때 잰 자가 열린 판이었고 "
                            "**닫힌 판에서는 잰 적이 없다**")
        p.add_argument("--arc-m", type=float, default=0.3, metavar="M",
                       help="여백(라디안). 얼굴에서 쓰는 값은 0.5 인데 그쪽은 "
                            "클래스당 사진이 수백 장이다. 우리는 개체당 평균 18장이라 "
                            "작게 잡고 시작한다")
        p.add_argument("--arc-s", type=float, default=30.0, metavar="S",
                       help="코사인이 [-1,1] 이라 그대로면 softmax 가 안 선다. "
                            "**순위는 안 바뀌고 확률만 선다** — 저장할 때도 이 값을 "
                            "곱해 둬서 화면의 퍼센트가 제 값이 되게 한다")
        p.add_argument("--hidden", type=int, default=0, metavar="N",
                       help="**숨은 층을 하나 넣는다** (384→N→개체, ReLU). 0 이면 "
                            "지금까지 쓰던 선형 한 층이다. 한 층으로는 축들의 "
                            "가중합밖에 못 말하는데, 두 층은 **조합**을 말한다 — "
                            "`축 7이 크고 동시에 축 40이 작을 때` 같은 것. "
                            "**손잡이가 확 는다**: 384→56 이 21,560개인데 "
                            "384→256→56 은 113,000개이고 배울 조각은 367장이다. "
                            "미뤄 뒀던 이유가 그것이고(`TODOs`), 조각이 563 → "
                            "1,221 이 된 2026-08-29 에 다시 물어보려고 붙였다")
        p.add_argument("--no-bias-decay", action="store_true",
                       help="**편향을 감쇠에서 뺀다.** 관례인데 이 저장소는 지금 "
                            "`net.parameters()` 를 통째로 넘겨 편향에도 건다. "
                            "편향은 개체마다 하나씩(30개)이라 작을 것 같지만 "
                            "**재 본 적이 없다**")
        p.add_argument("--seed", type=int, default=20260825)
        p.add_argument("--fit-all", metavar="파일",
                       help="**재지 않고 배워서 저장한다** — 화면이 쓸 것이라 "
                            "날을 가르지 않고 아는 것을 전부 쓴다. 성적을 볼 때는 "
                            "이것을 쓰면 안 된다 (배운 것으로 배운 것을 잰다)")
        p.add_argument("--folds", type=int, default=0, metavar="K",
                       help="**재는 날을 고정하지 말고 돌린다.** 관찰일을 K벌로 "
                            "나눠 한 벌씩 빼면서 K번 배우고 잰다 — 모든 날이 한 "
                            "번씩 잼 쪽에 서므로 질의가 137 → 510 이 되고 눈금이 "
                            "0.73%%p → 0.20%%p 가 된다. 0 이면 `--test-days` 대로 "
                            "한 번만 (기본)")
        p.add_argument("--seeds", type=int, default=1, metavar="N",
                       help="씨앗을 N 번 흔들어 폭을 함께 낸다. **폴드 배정은 "
                            "안 흔든다** — 두 자를 견줄 때 같은 문제를 풀어야 "
                            "차이가 자 때문인지 문제 때문인지 갈린다")
        p.add_argument("--save-logits", metavar="파일",
                       help="폴드·씨앗마다 질의별 로짓을 npz 로 남긴다 — 두 갈래를 "
                            "따로 배워 로짓을 평균하는 **앙상블을 같은 자로 재기** "
                            "위한 것이다. 폴드 배정이 씨앗을 안 타므로 갈래끼리 "
                            "같은 질의가 같은 줄에 선다")
        p.add_argument("--group", action="store_true",
                       help="**묶어서 묻는다** — 같은 날·같은 쪽의 한 개체 조각을 "
                            "한 묶음으로 보고 표를 모은다. 실제 화면이 하는 일이 "
                            "그것이고(`묶음 제안`), 판단 한 번이 여러 장을 덮는다")

    def _prefix(self, root, chips, ids, unfreeze, w, keep_rows, backbone, dim,
                device="cpu"):
        """조각 → **얼린 앞부분을 통과한 토큰**. 한 번만 만들어 두고 돌려 쓴다.

        마지막 N블록만 녹이면 앞 블록은 얼려 있으므로 **매 에폭 같은 값**이
        나온다. 그것을 다시 계산하는 것이 CPU 에서 값을 다 먹는다 —
        재 보니 마지막 블록만 녹인 학습(72ms/장)이 얼린 forward(76ms/장)와
        거의 같았다. 캐시하면 16.6ms/장이다.

        **그래서 증강을 못 쓴다.** 앞이 매번 같은 입력을 본다 — 증강까지
        하려면 이 함수를 버리고 조각부터 매번 돌려야 한다.

        `reid_chips` 의 전처리를 그대로 쓴다 (128 → 224 · ImageNet 정규화) —
        **다른 식으로 넣으면 얼린 특징과 견줄 수 없다.**

        **정답이 있는 조각만 만든다.** 격자가 7,912장인데 정답은 1,300장이고,
        정답 없는 것은 배움에도 잼에도 안 들어간다 — 전부 만들면 497초에
        3.1GB 였다. 낸 `pos` 가 원래 줄번호를 캐시 줄번호로 옮긴다.
        """
        import torch
        import torch.nn.functional as Fn

        z = np.load(root / chips)
        if not (z["box_id"] == ids).all():
            raise CommandError(f"{chips} 와 임베딩의 상자 차례가 다르다")
        from finseg import backbone as BB
        if BB.kind(backbone) not in ("v2", "v3", "timm"):
            raise CommandError(
                f"`--unfreeze` 는 ViT·Swin 만 녹인다 — `--backbone {backbone}` 은 "
                f"아니다. 쓸 수 있는 것: "
                f"{', '.join(k for k in BB.BACKBONES if BB.kind(k) in ('v2', 'v3', 'timm'))}")
        m = BB.load(backbone).to(device)
        pre, tail, keep = BB.split(m, unfreeze)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        if device != "cpu":
            w(f"    {device} 로 돈다")
        # **차원이 안 맞으면 여기서 멈춘다.** `--emb` 를 바꾸고 `--backbone` 을
        # 안 바꾸면 캐시와 임베딩이 다른 그물에서 나오는데, kNN 은 임베딩으로
        # 재고 분류기는 캐시로 재므로 **둘이 다른 그물인 채 나란히 찍힌다.**
        # Swin 은 `embed_dim` 이 줄기 폭(192)이고 특징 폭은 `num_features`(1536)다.
        # DINO ViT 는 둘이 같다 — 특징 폭 쪽을 묻는다.
        feat_dim = getattr(m, "num_features", None) or m.embed_dim
        if feat_dim != dim:
            raise CommandError(
                f"`--backbone {backbone}` 은 {feat_dim}차원인데 `--emb` 는 "
                f"{dim}차원이다 — 같은 백본으로 맞출 것")
        rows = np.flatnonzero(keep_rows)
        pos = np.full(len(ids), -1, dtype=np.int64)
        pos[rows] = np.arange(len(rows))
        C = torch.from_numpy(z["chip"][rows])
        out = []
        t0 = time.time()
        for i in range(0, len(C), 32):
            if i and i % 640 == 0:
                w(f"    캐시 {i:,}/{len(C):,} … {time.time() - t0:.0f}s")
            x = C[i:i + 32].unsqueeze(1).repeat(1, 3, 1, 1)
            x = Fn.interpolate(x, size=BB.input_size(backbone), mode="bilinear",
                               align_corners=False)
            h, aux = pre(((x - mean) / std).to(device))
            # **캐시는 CPU 에 둔다.** 정답 1,300장이면 513MB 라 VRAM 에 얹어도
            # 되지만, 격자가 자라면 먼저 터지는 자리가 여기다. 묶음마다 올린다
            out.append(h.cpu())
        w(f"    캐시 {len(C):,}장 (격자 {len(ids):,} 중 정답 있는 것) · "
          f"{time.time() - t0:.0f}s · "
          f"{sum(t.numel() for t in out) * 4 / 1e6:.0f} MB")
        # `aux` 는 조각 크기가 한 가지라 한 벌이면 된다 (v3 의 RoPE · v2 는 None)
        return torch.cat(out), (m, tail, aux, device), keep, pos

    def _fit_deep(self, pre, tr, y, n_cls, ctx, keep, o, seed):
        """캐시된 토큰 위에서 **마지막 N블록 + 머리**를 배운다 → numpy 로 낸다.

        낸 것은 `_logits_deep` 이 먹는다. 얼린 갈래처럼 `(W, b)` 로 못 접는다 —
        블록이 배웠으므로 추론에 그 블록이 필요하다.
        """
        import copy
        import torch
        import torch.nn.functional as Fn

        m, tail, aux, device = ctx
        torch.manual_seed(seed)
        blocks = copy.deepcopy(keep)          # **폴드마다 처음부터** — 안 그러면
        for p in blocks.parameters():         # 앞 폴드가 배운 것이 새어 든다
            p.requires_grad_(True)
        blocks.train()
        head = torch.nn.Linear(getattr(m, "num_features", None) or m.embed_dim,
                               n_cls).to(device)
        # 층별 학습률 — 머리는 `lr_deep`, 블록은 `lr_blocks`(없으면 같다)에서
        # 꼭대기부터 아래로 `lr_decay` 배씩. 기본값이면 그룹이 갈려도 수학은
        # 한 그룹과 같아 앞의 성적표가 그대로 선다.
        lr_b = o["lr_blocks"] if o["lr_blocks"] is not None else o["lr_deep"]
        n_b = len(blocks)
        groups = [{"params": head.parameters(), "lr": o["lr_deep"]}]
        groups += [{"params": blk.parameters(),
                    "lr": lr_b * o["lr_decay"] ** (n_b - 1 - i)}
                   for i, blk in enumerate(blocks)]
        opt = torch.optim.AdamW(groups, weight_decay=o["wd"])
        H = pre[tr]
        yt = torch.tensor(y).to(device)
        B = o["batch"]
        g = torch.Generator().manual_seed(seed)
        for _ in range(o["deep_epochs"]):
            for i in torch.randperm(len(H), generator=g).split(B):
                opt.zero_grad()
                loss = Fn.cross_entropy(head(tail(H[i].to(device), blocks, aux)), yt[i])
                loss.backward()
                opt.step()
        blocks.eval()
        return blocks, head

    def _logits_deep(self, net, ctx, pre, idx, batch=32):
        import torch
        _m, tail, aux, device = ctx
        blocks, head = net
        out = []
        with torch.no_grad():
            for i in range(0, len(idx), batch):
                h = pre[idx[i:i + batch]].to(device)
                out.append(head(tail(h, blocks, aux)).cpu().numpy())
        return np.concatenate(out)

    def _fit(self, X, y, n_cls, epochs, lr, wd, seed, l1=0.0, no_bias_decay=False,
             hidden=0, arcface=False, arc_m=0.3, arc_s=30.0):
        """선형 한 층, 또는 `hidden` 을 주면 숨은 층 하나를 더 (ReLU).
        **numpy 로 쓸 수 있게 가중치만 돌려준다** — 화면이 추론할 때 torch 를
        들이지 않아도 되게. 낸 것은 `_logits` 가 그대로 먹는다.

        규제가 둘이다. **L2**(`wd`)는 `AdamW` 가 가중치를 매 걸음 직접 줄이는
        쪽으로 건다(decoupled — Adam 계열에서 손실에 더하는 옛 방식이 의도대로
        안 들어서 갈라 나온 것이다). **L1**(`l1`)은 손실에 더한다.

        **`ReLU` 가 없으면 두 층은 한 층과 같다** — 행렬 두 번 곱하기는 한 번으로
        접힌다. 층을 진짜 두 층으로 만드는 것이 그 한 줄이다.

        `arcface` 면 손실만 바뀌고 **낸 것의 모양은 그대로다** — 추론이
        `정규화한 자 · x` 라 결국 선형 한 층이고, `arc_s` 를 곱해 두면 화면의
        `X @ W.T + b` 가 그대로 돈다(편향은 0). 여백은 배울 때만 쓴다.

        **`--wd` 는 arcface 에서 거의 일을 안 한다** — 자를 정규화해서 쓰므로
        크기를 줄여도 방향이 그대로다. 이 저장소가 `L2 는 이 자료의 지렛대가
        아니다` 를 이미 한 번 겪은 자리와 같은 종류다.
        """
        import torch
        import torch.nn.functional as Fn
        torch.manual_seed(seed)
        if hidden:
            net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden),
                                      torch.nn.ReLU(),
                                      torch.nn.Linear(hidden, n_cls))
            ws = [net[0].weight, net[2].weight]
            bs = [net[0].bias, net[2].bias]
        else:
            net = torch.nn.Linear(X.shape[1], n_cls)
            ws, bs = [net.weight], [net.bias]
        if no_bias_decay:
            groups = [{"params": ws, "weight_decay": wd},
                      {"params": bs, "weight_decay": 0.0}]
        else:
            groups = [{"params": list(net.parameters()), "weight_decay": wd}]
        opt = torch.optim.AdamW(groups, lr=lr)
        xt = torch.from_numpy(X.astype(np.float32))
        yt = torch.tensor(y)
        last = net[2] if hidden else net
        for _ in range(epochs):
            opt.zero_grad()
            if arcface:
                # `cos(θ+m) = cosθ·cos m − sinθ·sin m` — `acos` 를 안 거친다.
                # 각도를 직접 구하면 0 근처에서 기울기가 튀는데, 이렇게 쓰면
                # 코사인만으로 끝나 그 자리가 없다
                cos = Fn.normalize(xt) @ Fn.normalize(last.weight).T
                cos = cos.clamp(-1 + 1e-7, 1 - 1e-7)
                sin = (1 - cos ** 2).clamp_min(1e-12).sqrt()
                phi = cos * math.cos(arc_m) - sin * math.sin(arc_m)
                # **θ+m 이 180° 를 넘으면 여백이 도로 벌이 아니라 상이 된다**
                # (코사인이 다시 올라간다). 그 자리는 여백을 안 준다
                phi = torch.where(cos > math.cos(math.pi - arc_m), phi,
                                  cos - math.sin(math.pi - arc_m) * arc_m)
                hot = Fn.one_hot(yt, n_cls).bool()
                loss = Fn.cross_entropy(torch.where(hot, phi, cos) * arc_s, yt)
            else:
                loss = Fn.cross_entropy(net(xt), yt)
            # **가중치에만 건다** — 편향은 개체마다 하나뿐이라 죽일 축이 없다.
            # 두 층이면 두 판 다 건다: 미뤄 뒀던 걱정이 손잡이 수라, 규제를
            # 새로 는 판에만 안 걸면 재려던 것과 다른 것을 재게 된다
            if l1:
                loss = loss + l1 * sum(t.abs().sum() for t in ws)
            loss.backward()
            opt.step()
        if arcface:
            # **추론에는 여백이 없다.** 정규화한 자에 `arc_s` 를 곱해 두면
            # `X @ W.T + b` 한 줄이 그대로 `s·cosθ` 가 된다 — 화면 코드를
            # 안 고치고도 확률이 제 값으로 선다. 편향은 0 이다
            import torch as _t
            with _t.no_grad():
                Wn = (Fn.normalize(last.weight) * arc_s).numpy().astype(np.float32)
            zero = np.zeros(n_cls, dtype=np.float32)
            if hidden:
                return (ws[0].detach().numpy().astype(np.float32),
                        bs[0].detach().numpy().astype(np.float32), Wn, zero)
            return (Wn, zero)
        out = []
        for t in [x for pair in zip(ws, bs) for x in pair]:
            out.append(t.detach().numpy().astype(np.float32))
        return tuple(out)

    @staticmethod
    def _logits(net, X):
        """`_fit` 이 낸 것으로 점수를 낸다. **자를 한 곳에 둔다** — 배운 자리와
        재는 자리가 다른 식으로 셈하면 그 성적은 아무 뜻이 없다."""
        if len(net) == 2:
            W, b = net
            return X @ W.T + b
        W1, b1, W2, b2 = net
        h = X @ W1.T + b1
        np.maximum(h, 0, out=h)                 # ReLU
        return h @ W2.T + b2

    def handle(self, **o):
        import torch
        import torch.nn.functional as Fn

        w = self.stdout.write
        root = Path(o["dir"])
        f = root / o["emb"]
        if not f.exists():
            raise CommandError(f"{f} 가 없다 — `reid_chips --emb-only` 로 만들 것")
        z = np.load(f)
        X = z["emb"] / np.maximum(np.linalg.norm(z["emb"], axis=1, keepdims=True), 1e-9)
        ids, fac = z["box_id"], z["facing"]
        items = json.loads((root / "items.json").read_text())["items"]
        if not (np.array([it["id"] for it in items]) == ids).all():
            raise CommandError("items.json 과 임베딩의 상자 순서가 다르다")
        day = np.array([it["day"] for it in items])

        # **둘을 같이 걸지 않는다.** arcface 는 특징을 정규화해서 각도를 재는데,
        # 숨은 층을 두면 그 앞이 ReLU 뒤 값이라 추론 때도 정규화해야 한다 —
        # 그러면 화면의 `X @ W.T + b` 한 줄이 안 맞는다. MLP 는 이미 졌으니
        # (2026-08-29) 둘을 붙일 값이 없다.
        if o["arcface"] and o["hidden"]:
            raise CommandError("`--arcface` 와 `--hidden` 은 같이 못 쓴다 — "
                               "숨은 층을 두면 추론 때도 특징을 정규화해야 한다")

        cat = reid.catalog(as_of=o["as_of"])
        # **개체를 걸러 내는 자리는 여기 하나다** — `of` 를 만들기 전에 걸러야
        # 뺀 개체의 조각이 `lab = -1` 이 되어 배움에서도 잼에서도 함께 빠진다.
        # 뒤에서 걸러 내면 그 조각이 다른 개체의 후보로 남아 문제가 달라진다.
        if o["min_chips"]:
            cat = {k: v for k, v in cat.items() if len(v) >= o["min_chips"]}
        if o["sample_ind"]:
            rng = np.random.default_rng(o["ind_seed"])
            keys = sorted(cat)
            pick = rng.choice(len(keys), size=min(o["sample_ind"], len(keys)),
                              replace=False)
            cat = {keys[i]: cat[keys[i]] for i in sorted(pick)}
        if len(cat) < 2:
            raise CommandError(f"남은 개체가 {len(cat)}마리다 — 거르기를 늦출 것")
        of = {b: i for i, v in cat.items() for b in v}
        lab = np.array([of.get(int(b), -1) for b in ids])
        if (lab >= 0).sum() < 20:
            raise CommandError("정답이 너무 적다")

        # ---- 날로 가른다 ----------------------------------------------------
        days = sorted({d for d in day[lab >= 0]})
        cnt = Counter(day[lab >= 0].tolist())
        test_days = set(sorted(days, key=lambda d: -cnt[d])[:o["test_days"]])
        is_test = np.array([d in test_days for d in day])
        w(f"자료 {f} · {X.shape[1]}차원 · 정답 {int((lab>=0).sum()):,}조각 "
          f"· 개체 {len(cat)}"
          + (f"  ← 개체판정 {o['as_of']:,}번까지의 정답" if o["as_of"] else ""))
        w(f"관찰일 {len(days)} → 재는 날 {len(test_days)} · 배우는 날 "
          f"{len(days)-len(test_days)}")

        if o["fit_all"]:
            return self._save(o, X, ids, fac, lab, cat, w)

        # **누수를 값으로 막는다.** 날로 가른다고 적어 두는 것과 실제로 안 겹치는
        # 것은 다르다 — 이 저장소가 "화면은 됐다는데 값이 안 닿는다" 로 겪은 종류다
        if set(day[is_test]) & set(day[~is_test]):
            raise CommandError("배우는 날과 재는 날이 겹친다 — 이 성적은 못 쓴다")

        # **캐시는 한 번만 만든다** — 폴드·씨앗이 바뀌어도 얼린 앞부분은 같다.
        # 이것이 `--unfreeze` 를 이 기계에서 돌릴 만하게 하는 전부다.
        pre = None
        if o["unfreeze"]:
            w(f"\n마지막 {o['unfreeze']}블록을 녹인다 · 에폭 {o['deep_epochs']} · "
              f"lr {o['lr_deep']} · 묶음 {o['batch']}")
            dev = o["device"] or ("cuda" if torch.cuda.is_available() else "cpu")
            pre, ctx, keep, pos = self._prefix(root, o["chips"], ids,
                                               o["unfreeze"], w, lab >= 0,
                                               o["backbone"], X.shape[1], dev)
            o["_deep"] = (pre, ctx, keep, pos)

        if o["folds"]:
            return self._folds(o, X, lab, fac, day, days, cnt, w, len(cat))

        tot = self._score(o, X, lab, fac, day, is_test, o["seed"], w)
        self._report(tot, w, len(cat), o["calib"])

    def _score(self, o, X, lab, fac, day, is_test, seed, w):
        """한 갈래를 배우고 잰다. **자를 한 곳에 둔다** — 고정 갈래와 폴드가
        다른 식으로 채점하면 둘을 견줄 수 없다."""
        import numpy as np
        import torch

        torch.manual_seed(seed)
        tot = {"cls1": 0, "cls5": 0, "cls10": 0,
               "knn1": 0, "knn5": 0, "knn10": 0, "n": 0, "cal": []}
        for side in ("left", "right"):
            m = (fac == side) & (lab >= 0)
            tr = np.where(m & ~is_test)[0]
            te = np.where(m & is_test)[0]
            classes = sorted(set(lab[tr].tolist()))
            # **배운 적 없는 개체는 닫힌 판의 질문이 아니다** — 그것은
            # `reid_probe --hold-individuals` 가 재는 열린 판의 몫이다
            te = np.array([i for i in te if lab[i] in classes])
            if len(tr) < 10 or not len(te) or len(classes) < 2:
                if w:
                    w(f"  {side}: 잴 것이 모자란다 (배움 {len(tr)} · 잼 {len(te)} "
                      f"· 개체 {len(classes)})")
                continue
            k = {c: i for i, c in enumerate(classes)}
            yk = [k[int(v)] for v in lab[tr]]
            if o["unfreeze"]:
                # **캐시 줄번호로 옮겨서 쓴다** — 정답 있는 것만 만들어 뒀다
                pre, ctx, keep, pos = o["_deep"]
                deep = self._fit_deep(pre, pos[tr], yk, len(classes),
                                      ctx, keep, o, seed)
                logit = self._logits_deep(deep, ctx, pre, pos[te])
            else:
                net = self._fit(X[tr], yk, len(classes),
                                o["epochs"], o["lr"], o["wd"], seed,
                                o["l1"], o["no_bias_decay"], o["hidden"],
                                o["arcface"], o["arc_m"], o["arc_s"])
                logit = self._logits(net, X[te])

            if o.get("save_logits"):
                # 묶기(`--group`) 전의 날것을 남긴다 — 앙상블은 질의 단위로
                # 합친 뒤에 같은 규칙으로 묶어야 자가 하나가 된다
                o.setdefault("_dump", []).append(dict(
                    fold=o.get("_fold", -1), seed=seed, side=side,
                    te=te.copy(), y_ind=lab[te].copy(),
                    classes=np.array(classes), logit=logit.copy()))

            # 기준선 — **배우는 날의 조각만** 후보로 둔다. 분류기가 본 것과
            # 같은 자료라야 공평하다
            knn = np.full((len(te), len(classes)), -np.inf)
            for c, ci in k.items():
                g = tr[lab[tr] == c]
                knn[:, ci] = (X[te] @ X[g].T).max(1)

            y = np.array([k[int(v)] for v in lab[te]])
            def hits(s, yy=None):
                yy = y if yy is None else yy
                r = np.argsort(-s, axis=1)
                return ((r[:, 0] == yy).sum(),
                        (r[:, :5] == yy[:, None]).any(1).sum(),
                        (r[:, :10] == yy[:, None]).any(1).sum())
            if o["group"]:
                # **한 묶음은 (개체, 날, 쪽) 하나다** — 사람이 화면에서 묶는 단위와
                # 같다. 로짓은 평균하고 kNN 은 최고 닮음을 평균한다
                cells = {}
                for r, i in enumerate(te):
                    cells.setdefault((int(lab[i]), day[i]), []).append(r)
                idx = sorted(cells)
                logit = np.stack([logit[cells[c]].mean(0) for c in idx])
                knn = np.stack([knn[cells[c]].mean(0) for c in idx])
                y = np.array([k[c[0]] for c in idx])
                te = np.array(idx, dtype=object)
            if o["calib"]:
                # **화면이 내는 그 값을 그대로 잰다** — `review/views.py` 의
                # `_score` 도 로짓에 softmax 를 씌워 퍼센트로 낸다. 딴 식으로
                # 셈해서 재면 화면이 아니라 다른 것을 재게 된다
                ex = np.exp(logit - logit.max(1, keepdims=True))
                pr = ex / ex.sum(1, keepdims=True)
                srt = -np.sort(-pr, axis=1)
                tot["cal"].append(np.stack(
                    [srt[:, 0], srt[:, 0] - srt[:, 1],
                     (np.argmax(logit, axis=1) == y).astype(float)], 1))
            c1, c5, c10 = hits(logit)
            n1, n5, n10 = hits(knn)
            if w:
                w(f"  {side:<6} 개체 {len(classes):>2} · 배움 {len(tr):>3} · 잼 {len(te):>3}"
                  f"   기준선(kNN) {n1/len(te):>6.1%} / {n5/len(te):>6.1%}"
                  f"   **분류기 {c1/len(te):>6.1%} / {c5/len(te):>6.1%}**")
            tot["cls1"] += c1; tot["cls5"] += c5; tot["cls10"] += c10
            tot["knn1"] += n1; tot["knn5"] += n5; tot["knn10"] += n10
            tot["n"] += len(te)
            # **top-10 은 개체가 10마리를 넘어야 뜻이 있다** — 그 아래면 무조건
            # 100% 다. 지금은 면마다 27·31마리라 넘는다
            tot["cls10_ok"] = tot.get("cls10_ok", True) and len(classes) > 10
        return tot

    def _report(self, tot, w, n_ind, o_calib=False):
        if not tot["n"]:
            raise CommandError("잴 것이 없다 — `--test-days` 를 줄일 것")
        n = tot["n"]
        w(f"\n  {'':<14}{'top-1':>9}{'top-5':>9}{'top-10':>9}   (질의 {n})")
        w(f"  {'기준선 (kNN)':<14}{tot['knn1']/n:>9.1%}{tot['knn5']/n:>9.1%}"
          f"{tot['knn10']/n:>9.1%}")
        w(f"  {'**분류기**':<14}{tot['cls1']/n:>9.1%}{tot['cls5']/n:>9.1%}"
          f"{tot['cls10']/n:>9.1%}")
        w(self._top10_note(tot, n_ind))
        if o_calib:
            self._calib_note(tot, w)
        if tot["cls1"] <= tot["knn1"]:
            w("\n** 기준선을 못 넘었다 — 배운 것이 없다. 개체당 조각이 모자라거나"
              " 규제가 세거나 날 갈래가 너무 좁다.")

    def _folds(self, o, X, lab, fac, day, days, cnt, w, n_ind):
        """**재는 날을 돌린다** — 관찰일 K벌, 한 벌씩 빼면서 K번.

        ## 왜 조각이 아니라 날로 나누나

        `--test-days` 갈래와 같은 이유다. 같은 날 사진은 몇 초 사이에 연달아
        찍은 것이라 거의 같은 그림이고(정답 조각의 62%가 같은 날 3장 이상인
        묶음에 들어 있다), 그중 몇 장이 배움에 몇 장이 잼에 들어가면 모델은
        개체가 아니라 **그 날을 외운 것**이 된다.

        ## 왜 돌리나

        고정 갈래는 **가장 큰 8일**만 재고 나머지 41일 361조각은 성적에 한 번도
        안 들어간다. 질의가 137이면 **한 문제가 0.73%p** 라, 앞으로 잴 것들
        (백본 키우기·조각 해상도·2층 MLP·ArcFace)이 전부 몇 %p 짜리라 하나도
        못 가른다. 돌리면 510문항이 되어 눈금이 0.20%p 다.

        ## 폴드 배정은 씨앗을 안 탄다

        큰 날부터 번갈아 담아 폴드 크기를 맞추고, **그 배정을 고정한다.**
        두 자를 견줄 때 **같은 문제를 풀어야** 차이가 자 때문인지 문제 때문인지
        갈린다 — 씨앗은 머리의 초기값만 흔든다.

        ## 못 묻는 조각이 남는다

        관찰일이 하루뿐인 개체(8마리·21조각)는 어느 폴드에서도 질의가 못 된다 —
        그 날이 잼 쪽이면 배운 적이 없고 배움 쪽이면 물을 것이 없다.
        `/dataset` 의 `2일 이상` 축이 재는 것이 바로 이것이다.
        """
        import numpy as np

        K = o["folds"]
        if K < 2:
            raise CommandError("`--folds` 는 2 이상이라야 한다")
        if K > len(days):
            raise CommandError(f"관찰일이 {len(days)}일뿐이라 {K}벌로 못 나눈다")
        fold = split_days(days, cnt, K)

        w(f"**재는 날을 {K}벌로 돌린다** · 씨앗 {o['seeds']}개 "
          f"— 관찰일 {len(days)}일이 한 번씩 잼 쪽에 선다")
        runs = []
        for si in range(o["seeds"]):
            seed = o["seed"] + si
            tot = {"cls1": 0, "cls5": 0, "cls10": 0,
                   "knn1": 0, "knn5": 0, "knn10": 0, "n": 0, "cal": []}
            first = si == 0
            if first:
                w(f"\n  {'폴드':<5}{'재는날':>6}{'잼':>6}"
                  f"{'기준선 1/5/10':>22}{'분류기 1/5/10':>22}")
            for fi, ds in enumerate(fold):
                is_test = np.array([d in set(ds) for d in day])
                o["_fold"] = fi
                t = self._score(o, X, lab, fac, day, is_test, seed, None)
                for k2 in tot:
                    tot[k2] += t[k2]
                if first:
                    n = t["n"] or 1
                    w(f"  {fi + 1:<5}{len(ds):>6}{t['n']:>6}"
                      f"{t['knn1'] / n:>7.1%}{t['knn5'] / n:>7.1%}{t['knn10'] / n:>7.1%}"
                      f"{t['cls1'] / n:>8.1%}{t['cls5'] / n:>7.1%}{t['cls10'] / n:>7.1%}")
            runs.append(tot)

        if o.get("save_logits") and o.get("_dump"):
            np.savez_compressed(o["save_logits"],
                                entries=np.array(o["_dump"], dtype=object))
            w(f"  로짓 {len(o['_dump'])}벌 → {o['save_logits']}")

        n = runs[0]["n"]
        if not n:
            raise CommandError("잴 것이 없다")
        r0 = runs[0]
        w(f"\n  {'합':<5}{len(days):>6}{n:>6}"
          f"{r0['knn1'] / n:>7.1%}{r0['knn5'] / n:>7.1%}{r0['knn10'] / n:>7.1%}"
          f"{r0['cls1'] / n:>8.1%}{r0['cls5'] / n:>7.1%}{r0['cls10'] / n:>7.1%}")
        w(self._top10_note(r0, n_ind))
        if o["calib"]:
            self._calib_note(r0, w)
        w(f"\n  **질의 {n}** — 한 문제가 {100 / n:.2f}%p "
          f"(고정 갈래 137문항이면 {100 / 137:.2f}%p 였다)")
        if o["seeds"] > 1:
            for key, nm in (("cls1", "분류기 top-1"), ("cls5", "분류기 top-5"),
                            ("cls10", "분류기 top-10")):
                v = np.array([r[key] / r["n"] for r in runs]) * 100
                w(f"  {nm:<12} 씨앗별 {' '.join(f'{x:.1f}' for x in v)}"
                  f"  →  **{v.mean():.1f} ± {v.std(ddof=0):.1f}**")
            w("  **폭이 씨앗 때문에 생긴 것이다** — 자료도 폴드도 같았다. "
              "두 자의 차이가 이 폭보다 작으면 그것은 아직 차이가 아니다.")

    def _calib_note(self, tot, w):
        """**확신이 읽히나.** 성적은 무엇을 골랐나만 재는데, 화면은 그 옆에
        퍼센트를 띄운다.

        둘을 가른다. `1등 확률` 은 화면이 실제로 보여 주는 값이고, `1·2등 차이`
        는 이 저장소가 대신 쓰기로 한 값이다 — `HANDOFF` 가 **"1등 점수는
        쓸모없다(38%)"** 고 적어 둔 그 자리다. 자를 붙여 그 말이 지금도 맞는지,
        손실을 바꾸면 달라지는지 본다.

        **ECE 는 "말한 것과 실제가 얼마나 벌어지나" 다.** 확신을 열 칸으로
        나눠 칸마다 `평균 확신 − 실제 정확도` 의 절대값을 무게 평균한다.
        0 이면 90%라고 할 때 정말 90% 맞는다는 뜻이다. **성적이 비겨도 이것이
        작으면 화면에는 이득이다** — 사람이 언제 믿을지 정할 수 있게 된다.
        """
        cal = np.concatenate(tot["cal"]) if tot["cal"] else None
        if cal is None or not len(cal):
            return
        p1, gap, hit = cal[:, 0], cal[:, 1], cal[:, 2]
        w(f"\n  {'확신 상위':<10}{'1등 확률':>12}{'1·2등 차이':>13}")
        for q in (0.10, 0.25, 0.50, 1.00):
            k = max(1, int(round(len(cal) * q)))
            a = hit[np.argsort(-p1)[:k]].mean()
            b = hit[np.argsort(-gap)[:k]].mean()
            lab = "전부" if q == 1 else f"{q:.0%}"
            w(f"  {lab:<10}{a:>12.1%}{b:>13.1%}")
        # 열 칸으로 나눠 잰다. **빈 칸은 안 센다** — 없는 칸을 0 으로 두면
        # 확신이 한쪽에 몰린 자가 거저 좋아 보인다
        edges = np.linspace(0, 1, 11)
        ece = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (p1 > lo) & (p1 <= hi) if lo else (p1 >= lo) & (p1 <= hi)
            if m.sum():
                ece += m.mean() * abs(hit[m].mean() - p1[m].mean())
        w(f"\n  **ECE {ece:.3f}** — 화면이 말한 퍼센트와 실제가 벌어진 정도"
          f" (0 이 맞는 것)")
        w(f"  평균 확신 {p1.mean():.1%} 대 실제 {hit.mean():.1%}"
          f" — {'부풀린다' if p1.mean() > hit.mean() else '낮춰 말한다'}")

    def _top10_note(self, tot, n_ind):
        """**top-10 을 다른 연구와 그대로 견주지 말 것.**

        finFindR 82% · CurvRank 83% · Kim et al. 84.8% 가 전부 top-10 이라 같은
        줄에 세우고 싶어지는데, **개체 수가 다르면 같은 자가 아니다.** 저쪽
        reference 는 79개체다 — 아무렇게나 찍어도 42마리에서 열을 고르면
        23.8%가 맞는데 79마리에서는 12.7%다.

        그래서 여기에 **찍어서 맞을 확률**을 함께 적는다. 숫자 옆에 그 값이
        없으면 다음에 보는 사람이 그대로 옮겨 적는다.

        **세어서 쓴다.** 2026-08-29 까지 `42마리 · 23.8%` 가 글에 박혀 있었는데,
        그날 개체는 65였다 — **"숫자를 그대로 옮겨 적지 말라" 고 말하는 문장이
        정작 제가 낡은 숫자를 옮기고 있었다.** 박아 둔 값은 자료가 자라는 만큼
        조용히 틀리고, 여기서 틀리면 성적을 남과 견줄 때 그대로 틀린다.
        """
        n = max(1, n_ind)
        return (f"  **top-10 을 남과 견줄 때는 개체 수를 함께 적을 것** — "
                f"우리 {n}마리에서 찍어 맞을 확률이 {10 / n:.1%}다 "
                f"(Kim et al. 은 79마리라 {10 / 79:.1%}).")

    def _save(self, o, X, ids, fac, lab, cat, w):
        """**화면이 쓸 분류기를 저장한다.** 날을 안 가르고 아는 것을 전부 쓴다 —
        성적을 재는 일과 실제로 쓰는 일은 다르다.

        `npz` 로 둔다. 추론은 `X @ W.T + b` 한 줄이라 화면 쪽에서 torch 를 들일
        이유가 없다.
        """
        # **화면은 선형 한 층만 읽는다** (`review/views.py` 의 `_score`).
        # 숨은 층을 넣은 것을 그대로 저장하면 칸 수가 안 맞아, 화면이 조용히
        # kNN 으로 떨어지거나 터진다 — 저장하는 자리에서 막는다.
        if o["unfreeze"]:
            raise CommandError(
                "`--unfreeze` 는 아직 재기만 한다 — 화면은 `X @ W.T + b` 로만 읽는데 "
                "(`review/views.py` 의 `_score`) 녹인 블록은 그 한 줄로 못 접힌다.\n"
                "**화면 쪽은 이제 준비돼 있다** — 임베딩·머리를 한 벌씩 짝지어 "
                "여러 벌 읽고 `reid.ens_logits` 로 합친다(`ensemble.json`). "
                "남은 것은 여기다: **녹여 배운 백본으로 격자 조각의 임베딩을 "
                "다시 내는 길**. 그러면 머리는 다시 선형 한 층이라 그대로 실린다")
        if o["hidden"]:
            raise CommandError(
                "`--hidden` 은 아직 재기만 한다 — 화면이 `X @ W.T + b` 로만 읽는다 "
                "(`review/views.py` 의 `_score`). 성적이 서면 화면 쪽을 먼저 고칠 것")
        out = {}
        for side in ("left", "right"):
            m = (fac == side) & (lab >= 0)
            idx = np.where(m)[0]
            classes = sorted(set(lab[idx].tolist()))
            if len(idx) < 10 or len(classes) < 2:
                w(f"  {side}: 배울 것이 모자란다 ({len(idx)}조각 · 개체 {len(classes)})")
                continue
            k = {c: i for i, c in enumerate(classes)}
            W_, B_ = self._fit(X[idx], [k[int(v)] for v in lab[idx]], len(classes),
                               o["epochs"], o["lr"], o["wd"], o["seed"],
                               o["l1"], o["no_bias_decay"], 0,
                               o["arcface"], o["arc_m"], o["arc_s"])
            out[f"{side}_W"] = W_
            out[f"{side}_b"] = B_
            out[f"{side}_cls"] = np.array(classes, dtype=np.int64)
            w(f"  {side:<6} 개체 {len(classes):>2} · 조각 {len(idx):>3}")
        if not out:
            raise CommandError("배운 것이 없다")
        out["emb"] = np.array([o["emb"]])
        out["dim"] = np.array([X.shape[1]])
        out["n_labeled"] = np.array([int((lab >= 0).sum())])
        np.savez_compressed(o["fit_all"], **out)
        w(f"\n{o['fit_all']}  — 아는 것을 전부 써서 배웠다.")
        w("**이것으로 성적을 재지 말 것** — 배운 것으로 배운 것을 재게 된다.")
