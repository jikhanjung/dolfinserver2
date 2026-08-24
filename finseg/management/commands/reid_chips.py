"""**상자 → re-ID 조각.** 스크립트로 돌리던 자리를 명령으로 뽑는다.

    python manage.py reid_chips --out reid/v2
    python manage.py reid_chips --out reid/v2 --auto-facing   # 검토 안 된 상자도

조각은 `seg` 갈래가 바뀌면 다시 만들어야 하고(마스크가 곧 윤곽이다), 정답이
아니라 **파생물**이라 다시 만들 수 있어야 한다. 스크립트로 두면 `seg-v4` 가
나올 때마다 손으로 하게 되고, 그 사이에 규칙이 조용히 달라진다.

## 무엇이 조각이 되나 — `reid.usable` 하나가 정한다

거르는 축이 전부 검토에서 남긴 것이다(`cls`·`edges`·`base_partial`·`facing`·
넓이). **여기서 그 규칙을 다시 쓰지 않는다** — 두 곳에 두면 화면이 거른 것과
자료로 나가는 것이 갈린다.

## `--auto-facing` — 검토 안 된 상자를 들일 때

새 사진에는 사람의 판정이 없다. 네 축 중 셋은 그럭저럭 메워진다 —
`cls` 는 검출기가 등지느러미만 내고, `edges`·`base_partial` 은 판정이 없으면
`usable` 이 통과값으로 읽는다. **`facing` 만 비면 그대로 걸린다.**

`baseline.propose_facing` 이 그 자리를 메운다 — 꼭대기 기준 넓이비로 앞쪽을
짐작하고, **애매하면 아무 말도 안 한다**(문턱 1.3). 사람이 찍은 156장에서
95%에 적용해 정확도 100%였다.

**다만 짐작은 짐작이라고 적는다.** `items.json` 의 `facing_src` 가 `human` 인지
`geom` 인지 들고 있어, 나중에 성적이 갈리면 그것부터 물을 수 있다.

## 못 만든 것을 세어서 말한다

`왜 빠졌나` 를 축별로 낸다. **조용히 적게 나오는 것이 가장 나쁘다** — 이
저장소가 "0/659" 하나로 실패 659건을 못 보고 지나간 적이 있다.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from finseg import baseline, geometry, reid, rules
from finseg.models import Box, Crop

LOOK = 320          # 사람이 보는 그림 한 변


class Command(BaseCommand):
    help = "상자에서 re-ID 조각·곡선·임베딩을 만든다"

    def add_arguments(self, p):
        p.add_argument("--out", default="reid/v2")
        p.add_argument("--auto-facing", action="store_true",
                       help="`facing` 이 없으면 기하로 짐작한다 (검토 안 된 상자)")
        p.add_argument("--auto-cls", action="store_true",
                       help="판정이 없으면 **분할 엔진이 말한 분류**를 쓴다. "
                            "`infer` 가 경고하듯 사람의 판정을 대신하지 않는다 — "
                            "아무도 안 본 상자에서만 쓸 값이다")
        p.add_argument("--min-area", type=int, default=reid.MIN_AREA,
                       help="**섞을 격자와 같은 값이어야 한다.** `reid/v1` 은 "
                            "15,000 으로 만들어졌는데 `reid.MIN_AREA` 는 그 뒤에 "
                            "30,000 이 되었다 — 그대로 다시 만들면 사람이 이미 "
                            "분류한 조각 1,034장이 격자에서 사라진다")
        p.add_argument("--boxes", nargs="+", type=int, help="이 상자들만")
        p.add_argument("--no-emb", action="store_true",
                       help="임베딩을 건너뛴다 (torch 가 없을 때)")
        p.add_argument("--backbone", choices=("resnet18", "dinov2"),
                       default="resnet18",
                       help="**조각은 그대로 두고 자만 바꾼다.** `resnet18` 은 "
                            "ImageNet 지도학습(512차원), `dinov2` 는 자기지도 "
                            "ViT-S/14(384차원)")
        p.add_argument("--emb-only", action="store_true",
                       help="이미 만든 조각으로 **임베딩만 다시 뽑는다** — 백본을 "
                            "갈아 볼 때. 조각·곡선·그림은 안 건드린다")
        p.add_argument("--emb-name", default="emb.npz",
                       help="임베딩 파일 이름. 백본마다 달리 두면 한 격자에서 "
                            "여러 자를 견줄 수 있다")
        p.add_argument("--dry-run", action="store_true")

    def handle(self, **o):
        w = self.stdout.write
        out = Path(o["out"])
        if o["emb_only"]:
            return self._emb_only(o, out, w)
        crops = {c.box_id: c for c in Crop.objects.all()}
        qs = Box.objects.prefetch_related("reviews", "masks").select_related("image")
        if o["boxes"]:
            qs = qs.filter(id__in=o["boxes"])
        boxes = [b for b in qs if b.id in crops]
        w(f"상자 {len(boxes):,} · 크롭 {settings.FIN_CROPS}")
        # **문턱을 자료에 적어 둔다.** 격자를 섞을 때 무엇으로 걸렀는지 모르면
        # 두 벌이 다른 규칙으로 만들어진 것을 알 길이 없다
        if o["min_area"] != reid.MIN_AREA:
            w(f"넓이 문턱 {o['min_area']:,} (`reid.MIN_AREA` 는 {reid.MIN_AREA:,})")
        reid.MIN_AREA = o["min_area"]

        why = Counter()
        rows, chips, looks, curves = [], [], [], []
        for b in sorted(boxes, key=lambda x: x.id):
            st = rules.resolve(b)
            crop = crops[b.id]
            src = "human" if st.get("facing") else ""
            cls_src = "human" if st.get("cls") else ""
            if not st.get("cls") and o["auto_cls"]:
                m = st.get("mask")
                if m is not None and getattr(m, "cls", ""):
                    st = dict(st); st["cls"] = m.cls
                    cls_src = "engine"
            if not st.get("facing") and o["auto_facing"]:
                pts = rules.final_points(st, crop)
                f, ratio = baseline.propose_facing(pts, crop.w)
                if f:
                    st = dict(st); st["facing"] = f
                    src = "geom"
            ok, bad = reid.usable(st)
            if not ok:
                why[bad.split("(")[0].strip()] += 1
                continue
            c = reid.chip(st, crop)
            look = reid.chip(st, crop, size=LOOK, color=True, cut=False)
            curve = reid.curve_of(st, crop)
            if c is None or look is None or not len(curve):
                why["조각·곡선을 못 만들었다"] += 1
                continue
            # `roughness` 는 못 재면 **None 을 돌려준다** — 0 으로 적으면
            # "매끈하다" 로 읽혀 정렬 맨 끝에 조용히 쌓인다
            rough = reid.roughness(st, crop)
            rows.append({"id": b.id, "day": str(b.image.obsdate),
                         "facing": st["facing"], "facing_src": src,
                         "cls_src": cls_src,
                         "rough": (round(float(rough["rear_max"]), 4)
                                   if rough else None)})
            chips.append(c); looks.append(look); curves.append(curve)

        w(f"\n조각 {len(rows):,} 개")
        if why:
            w(f"  {'왜 빠졌나':<28}{'수':>7}")
            for k, v in why.most_common():
                w(f"  {k:<28}{v:>7,}")
        if not rows:
            raise CommandError("조각이 하나도 안 나왔다 — 위의 이유를 볼 것")
        n_geom = sum(1 for r in rows if r["facing_src"] == "geom")
        n_eng = sum(1 for r in rows if r["cls_src"] == "engine")
        if n_geom:
            w(f"  앞쪽을 기하로 짐작한 것 {n_geom:,} 개 "
              f"({n_geom / len(rows):.0%}) — `facing_src=geom`")
        if n_eng:
            w(f"  분류를 엔진에서 읽은 것 {n_eng:,} 개 "
              f"({n_eng / len(rows):.0%}) — `cls_src=engine`")
        if o["dry_run"]:
            w("\n--dry-run 이라 아무것도 쓰지 않았다.")
            return

        # **한 격자가 두 규칙으로 만들어지면 아무도 모른다.** 이미 있는 것과
        # 문턱이 다르면 멎는다 — 섞인 뒤에는 어느 조각이 어느 규칙인지 못 가른다
        prev = out / "items.json"
        if prev.exists():
            was = json.loads(prev.read_text()).get("min_area")
            if was is not None and was != o["min_area"]:
                raise CommandError(
                    f"{out} 는 넓이 문턱 {was:,} 로 만들어졌는데 지금은 "
                    f"{o['min_area']:,} 다.\n"
                    f"  --min-area {was} 로 맞추거나 --out 을 새로 줄 것.")
        out.mkdir(parents=True, exist_ok=True)
        ids = np.array([r["id"] for r in rows])
        fac = np.array([r["facing"] for r in rows])
        np.savez_compressed(out / "chips.npz", box_id=ids, facing=fac,
                            chip=np.stack(chips).astype(np.float32))
        np.savez_compressed(out / "curves.npz", box_id=ids, facing=fac,
                            curve=np.stack(curves).astype(np.float32))
        (out / "items.json").write_text(json.dumps(
            {"n": len(rows), "min_area": o["min_area"],
             "auto_facing": bool(o["auto_facing"]), "auto_cls": bool(o["auto_cls"]),
             "items": rows}, ensure_ascii=False))
        # 사람이 보는 그림은 파일로 — 화면이 낱장으로 읽는다
        from PIL import Image as PImage
        (out / "look").mkdir(exist_ok=True)
        for r, im in zip(rows, looks):
            PImage.fromarray((np.clip(im, 0, 1) * 255).astype(np.uint8)).save(
                out / "look" / f"{r['id']:08d}.jpg", quality=92)
        w(f"{out}/chips.npz · curves.npz · items.json · look/")

        if not o["no_emb"]:
            emb = self._embed(np.stack(chips), o["backbone"])
            np.savez_compressed(out / o["emb_name"], box_id=ids, facing=fac, emb=emb)
            w(f"{out}/{o['emb_name']}  ({emb.shape[1]}차원 · {o['backbone']})")

    def _emb_only(self, o, out, w):
        """조각은 그대로 두고 임베딩만 다시 뽑는다.

        **자를 바꿔 보는 일은 조각을 다시 만드는 일이 아니다.** 조각을 다시
        만들면 그 사이에 마스크가 바뀌었을 수도 있어 **무엇이 성적을 움직였는지
        못 가른다** — 같은 조각에 다른 자를 대야 백본만의 몫이 나온다.
        """
        f = out / "chips.npz"
        if not f.exists():
            raise CommandError(f"{f} 가 없다 — 먼저 조각을 만들 것")
        z = np.load(f)
        w(f"{f} 조각 {len(z['box_id']):,} · 자 {o['backbone']}")
        emb = self._embed(z["chip"], o["backbone"])
        np.savez_compressed(out / o["emb_name"], box_id=z["box_id"],
                            facing=z["facing"], emb=emb)
        w(f"{out}/{o['emb_name']}  ({emb.shape[1]}차원 · {o['backbone']})")

    def _embed(self, X, backbone="resnet18"):
        """**학습 없는 사전학습 특징.** `resnet18` 은 `reid_train` 의 기준선과
        같은 식이어야 옛 조각과 새 조각을 한 통에 놓을 수 있다 — 다르면 새것만
        딴 데 모인다.

        `dinov2` 는 자기지도로 배운 ViT-S/14 다. 분류가 아니라 **닮음**을 배운
        특징이라 검색에서 더 낫다고 알려져 있다. 조각이 128px 이라 **224 로
        키워 넣는다** — 14의 배수라야 패치가 떨어진다.
        """
        import torch
        import torch.nn.functional as Fn
        if backbone == "dinov2":
            m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                               pretrained=True, verbose=False)
        else:
            import torchvision as tv
            m = tv.models.resnet18(weights=tv.models.ResNet18_Weights.IMAGENET1K_V1)
            m.fc = torch.nn.Identity()
        m.eval()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        outs = []
        X = torch.from_numpy(X)
        for i in range(0, len(X), 64):
            x = X[i:i + 64].unsqueeze(1).repeat(1, 3, 1, 1)
            x = Fn.interpolate(x, size=224, mode="bilinear", align_corners=False)
            with torch.no_grad():
                outs.append(m((x - mean) / std).numpy())
        return np.concatenate(outs)
