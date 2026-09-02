"""**백본 하나를 여는 자리.** DINOv2 와 v3 가 다른 것을 여기서만 안다.

    from finseg import backbone
    m = backbone.load("dinov3s")
    pre, tail = backbone.split(m, unfreeze=1)     # 캐시 경계

## 왜 한 자리에 두나

`reid_chips` 는 임베딩을 뽑고 `reid_cls --unfreeze` 는 마지막 블록을 녹인다.
둘이 **같은 그물에서 같은 값**을 내야 성적을 견줄 수 있는데, v2 와 v3 는
셋이 다르다 — 그것을 두 곳에 적으면 한쪽만 고쳐진다.

    DINOv2 ViT-S/14   블록 12 · 384차원 · 토큰 257 (CLS 1 + 패치 256)
                      위치를 patch embed 에서 더한다 → 블록이 토큰만 받는다
    DINOv3 ViT-S/16   블록 12 · 384차원 · 토큰 201 (CLS 1 + storage 4 + 패치 196)
                      **RoPE 를 블록마다 넘긴다** → `blk(x, rope)`

크기와 차원이 거의 같아(21.6M 대 22.1M · 둘 다 384) **사전학습만 다른 대조**가
된다 — ViT-B 처럼 "커져서 이겼나" 를 따로 물을 필요가 없다.

## 가중치

v2 는 `torch.hub` 가 받아 온다. **v3 는 라이선스에 동의해야 받을 수 있어**
받아 둔 파일을 가리킨다 (`FIN_MODELS`, 기본 `/nas/JikhanJung/models`).
없으면 어디에 무엇을 두라고 말하고 멈춘다 — 조용히 딴 백본으로 떨어지면
성적이 무엇으로 난 것인지 모른다.
"""
import glob
from pathlib import Path

from django.conf import settings

# 값: (갈래, torch.hub / timm 이름, v3 가중치 파일의 앞머리)
#   `tv` 는 torchvision · `v2`·`v3` 는 DINO · `timm` 은 timm 이 hf-hub 에서 연다.
BACKBONES = {
    "resnet18": ("tv", None, None),              # ImageNet 지도학습 · 512
    "dinov2":   ("v2", "dinov2_vits14", None),   # ViT-S/14 ·  384
    "dinov2b":  ("v2", "dinov2_vitb14", None),   # ViT-B/14 ·  768
    "dinov2l":  ("v2", "dinov2_vitl14", None),   # ViT-L/14 · 1024
    "dinov3s":  ("v3", "dinov3_vits16", "dinov3_vits16_pretrain"),        # 384
    "dinov3sp": ("v3", "dinov3_vits16plus", "dinov3_vits16plus_pretrain"),# 384
    "dinov3b":  ("v3", "dinov3_vitb16", "dinov3_vitb16_pretrain"),        # 768
    "dinov3l":  ("v3", "dinov3_vitl16", "dinov3_vitl16_pretrain"),        # 1024
    # 동물 re-ID 전용 사전학습 (WildlifeDatasets · Swin-L · 입력 384 · 1536차원).
    # **얼린 갈래 전용이다** — Swin 은 블록 구조가 달라 `split()` 이 못 가른다
    "megad":    ("timm", "hf-hub:BVRA/MegaDescriptor-L-384", None),
}


def kind(name):
    return BACKBONES[name][0] if name in BACKBONES else None


def input_size(name):
    """백본이 기대하는 입력 한 변. 조각을 이 크기로 늘려 넣는다."""
    return 384 if name == "megad" else 224


def load(name):
    """이름 → 얼린 모델. 못 열면 무엇이 없는지 말하고 멈춘다."""
    import torch
    from django.core.management.base import CommandError

    if name not in BACKBONES:
        raise CommandError(f"모르는 백본 {name!r} — {', '.join(BACKBONES)}")
    k, entry, stem = BACKBONES[name]
    if k == "tv":
        import torchvision as tv
        m = tv.models.resnet18(weights=tv.models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = torch.nn.Identity()
    elif k == "timm":
        import timm
        m = timm.create_model(entry, pretrained=True, num_classes=0)
    elif k == "v2":
        m = torch.hub.load("facebookresearch/dinov2", entry,
                           pretrained=True, verbose=False)
    else:
        root = Path(getattr(settings, "FIN_MODELS", "/nas/JikhanJung/models"))
        hit = sorted(glob.glob(str(root / "dinov3" / f"{stem}*.pth")))
        if not hit:
            raise CommandError(
                f"{name} 가중치가 없다 — {root}/dinov3/{stem}*.pth 를 찾았다.\n"
                f"  DINOv3 는 라이선스에 동의해야 받을 수 있다 "
                f"(dinov3.llamameta.net). 받아서 그 자리에 둘 것.\n"
                f"  다른 자리면 `FIN_MODELS` 로 대 줄 것")
        m = torch.hub.load("facebookresearch/dinov3", entry,
                           source="github", weights=hit[0])
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def split(m, unfreeze):
    """모델 → `(prefix, tail)`. **캐시 경계를 여기서 정한다.**

    `prefix(x)` 는 얼린 앞부분을 통과한 토큰을, `tail(h, blocks)` 는 그 뒤
    블록들을 태워 **`m(x)` 와 같은 값**을 낸다 (비트까지 같은 것을 확인했다).
    `tail` 이 블록을 인자로 받는 것은 **폴드마다 새로 만든 사본**을 태우기
    위해서다 — 모델 안의 것을 그대로 쓰면 앞 폴드가 배운 것이 새어 든다.
    """
    import torch

    if not hasattr(m, "blocks"):
        raise ValueError("이 백본은 블록을 못 가른다 (ViT 만 된다)")
    keep = m.blocks[-unfreeze:]

    if m.rope_embed is None if hasattr(m, "rope_embed") else True:
        # ---- DINOv2 — 블록이 토큰만 받는다
        def prefix(x):
            with torch.no_grad():
                h = m.prepare_tokens_with_masks(x)
                for blk in m.blocks[:-unfreeze]:
                    h = blk(h)
            return h, None

        def tail(h, blocks, _aux=None):
            for blk in blocks:
                h = blk(h)
            return m.norm(h)[:, 0]
    else:
        # ---- DINOv3 — **RoPE 를 블록마다 넘긴다.** 조각 크기가 한 가지라
        # rope 는 한 벌이면 되지만, 크기가 섞이면 여기서 갈린다
        def prefix(x):
            with torch.no_grad():
                h, (H, W) = m.prepare_tokens_with_masks(x)
                rope = m.rope_embed(H=H, W=W)
                for blk in m.blocks[:-unfreeze]:
                    h = blk(h, rope)
            return h, rope

        def tail(h, blocks, rope=None):
            for blk in blocks:
                h = blk(h, rope)
            # **CLS 는 0번**이고 그 뒤 `n_storage_tokens` 개는 register 다
            return m.norm(h)[:, 0]

    return prefix, tail, keep
