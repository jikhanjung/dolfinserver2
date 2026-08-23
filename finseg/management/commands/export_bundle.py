"""가중치와 **그것을 쓰는 규칙**을 한 벌로 묶어 낸다.

    python manage.py export_bundle --run 36 --out bundles/detect-v2
    python manage.py export_bundle --run 36 --out bundles/detect-v2 --seg-run 22

## 왜 묶나 — 규칙이 갈라지면 조용히 나빠진다

`.onnx` 하나를 세 곳이 쓴다: 이 저장소(`finseg/onnxdet.py`), 브라우저
(`/detect` 의 JS), 그리고 데스크톱 뷰어. **전처리가 어긋나도 예외가 안 난다** —
상자는 나오고 성적만 나빠진다. 실측으로 축소 필터 하나가 29×21px 상자의 확신을
0.36 → 0.17 로 반토막 냈고, 그것은 새 검출기가 여는 것의 3분의 2인 **끄트머리만
보이는 작은 지느러미**부터 먼저 지운다.

`.guides` 가 같은 것을 이미 말한다 — *"복사는 표류하고, 표류를 보고하지 않는다."*
그래서 **가중치와 `onnxdet.py` 와 상수를 한 디렉토리에 담아** 함께 움직이게 한다.
받는 쪽은 묶음을 통째로 쓰고, `MODEL.json` 의 `rules_sha256` 으로 자기가 든
규칙이 그 가중치의 짝인지 확인할 수 있다.

## 무엇이 들어가나

    detect-v2/
      model.onnx        검출기 (1280)
      seg.onnx          분할기 (640) — `--seg-run` 을 주면
      onnxdet.py        전처리 · NMS · 좌표 되돌리기 · 마스크 해독
      MODEL.json        상수와 출처 (run · 학습 자료 · 커밋 · 성적)

`MODEL.json` 이 **출처를 든다**. 어느 run 의 가중치인지, 어느 자료로 배웠는지,
어느 커밋의 규칙인지가 없으면 나중에 "이 상자가 어디서 나왔나" 를 물을 수 없다 —
`Box.source`·`Box.conf` 를 그래서 남기는 것과 같은 이유다.
"""
import hashlib
import json
import shutil
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from finseg import onnxdet, runs
from finseg.models import Run


def sha256(path, n=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(n):
            h.update(chunk)
    return h.hexdigest()


class Command(BaseCommand):
    help = "가중치와 그것을 쓰는 규칙을 한 벌로 묶는다"

    def add_arguments(self, p):
        p.add_argument("--run", type=int, required=True,
                       help="검출기 학습 run 번호")
        p.add_argument("--seg-run", type=int, help="분할 학습 run 번호 (없어도 된다)")
        p.add_argument("--det-weights", help="run 기록에 이름이 없을 때 직접 준다")
        p.add_argument("--seg-weights")
        p.add_argument("--out", required=True)
        p.add_argument("--name", help="기본은 --out 의 마지막 조각")
        p.add_argument("--dry-run", action="store_true")

    def _weights(self, run_id, kind, given=None):
        """학습 run → `.onnx` 경로. **`.pt` 옆에 있어야 한다.**

        run 기록의 `name` 으로 찾는다. **옛 run 에는 그것이 없다** — `train.py`
        가 `--name` 을 안 적었다(2026-08-24 에 고쳤다). 그때는 `--det-weights`
        / `--seg-weights` 로 직접 준다. 짐작으로 찾지 않는다 — 엉뚱한 가중치를
        묶어 내는 것이 못 찾는 것보다 나쁘다.
        """
        run = Run.objects.filter(id=run_id, kind="train").first()
        if run is None:
            raise CommandError(f"run {run_id} 이 학습 run 이 아니다")
        params = run.params or {}
        if params.get("task") != kind:
            raise CommandError(
                f"run {run_id} 은 `{params.get('task')}` 인데 `{kind}` 을 기대했다")
        if given:
            onnx = Path(given)
        elif params.get("name"):
            onnx = Path("runs") / params["name"] / "weights" / "best.onnx"
        else:
            raise CommandError(
                f"run {run_id} 의 기록에 `name` 이 없다 (2026-08-24 이전 run).\n"
                f"  `--{'det' if kind == 'detect' else 'seg'}-weights"
                f" runs/<이름>/weights/best.onnx` 로 직접 줄 것.\n"
                f"  자료는 `{Path(params.get('data','')).name}` 이었다.")
        name = onnx.parent.parent.name
        if not onnx.exists():
            raise CommandError(
                f"{onnx} 가 없다. 먼저 내보낼 것:\n"
                f"  yolo export model=runs/{name}/weights/best.pt format=onnx"
                f" imgsz={onnxdet.IMGSZ if kind == 'detect' else onnxdet.SEG_IMGSZ}"
                f" simplify=True opset=17 nms=False\n"
                f"  **`nms=False` 가 중요하다** — NMS 는 받는 쪽이 한다.")
        return onnx, run, params

    def handle(self, **o):
        w = self.stdout.write
        det_onnx, det_run, det_params = self._weights(
            o["run"], "detect", o["det_weights"])
        seg_onnx = seg_run = seg_params = None
        if o["seg_run"]:
            seg_onnx, seg_run, seg_params = self._weights(
                o["seg_run"], "segment", o["seg_weights"])

        rules = Path(onnxdet.__file__)
        out = Path(o["out"]).resolve()
        name = o["name"] or out.name

        # **상수를 손으로 안 적는다.** `onnxdet` 에서 읽어 온다 — 두 곳에 두면
        # 하나만 고쳐지고, 그 어긋남은 상자가 조금 밀린 모습으로만 나타난다
        meta = {
            "name": name,
            "rules": rules.name,
            "rules_sha256": sha256(rules),
            "detect": {
                "file": "model.onnx", "imgsz": onnxdet.IMGSZ,
                "conf": onnxdet.CONF, "iou": onnxdet.IOU, "pad": onnxdet.PAD,
                "nc": 1, "names": ["fin"],
                "run": det_run.id, "data": det_params.get("data"),
                "git_sha": det_run.git_sha, "sha256": sha256(det_onnx),
            },
            "crop": {"pad": onnxdet.CROP_PAD,
                     "rule": "정사각형 · 상자 긴 변의 pad 배 · 가장자리에서는 민다"},
            "git_sha": runs.git_sha(),
        }
        if seg_onnx:
            meta["segment"] = {
                "file": "seg.onnx", "imgsz": onnxdet.SEG_IMGSZ,
                "nc": onnxdet.SEG_NC, "names": ["fin", "dolphin", "nonfin"],
                "mask_thres": onnxdet.MASK_THRES,
                "run": seg_run.id, "data": seg_params.get("data"),
                "git_sha": seg_run.git_sha, "sha256": sha256(seg_onnx),
            }

        w(f"묶음 '{name}'")
        w(f"  검출 {det_onnx} ({det_onnx.stat().st_size/1e6:.1f}MB)"
          f" · run {det_run.id} · {Path(det_params.get('data','')).name}")
        if seg_onnx:
            w(f"  분할 {seg_onnx} ({seg_onnx.stat().st_size/1e6:.1f}MB)"
              f" · run {seg_run.id} · {Path(seg_params.get('data','')).name}")
        else:
            w("  분할 없음 — 받는 쪽은 상자만 낸다 (`--seg-run` 으로 넣는다)")
        w(f"  규칙 {rules.name} · sha256 {meta['rules_sha256'][:12]}…")
        if o["dry_run"]:
            w("\n--dry-run 이라 아무것도 쓰지 않았다.")
            return

        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(det_onnx, out / "model.onnx")
        if seg_onnx:
            shutil.copy2(seg_onnx, out / "seg.onnx")
        shutil.copy2(rules, out / rules.name)
        (out / "MODEL.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2))
        w(f"\n{out}")
        w("  받는 쪽은 이 디렉토리를 통째로 쓴다. `MODEL.json` 의 `rules_sha256`")
        w("  으로 자기가 든 규칙이 이 가중치의 짝인지 확인할 것 —")
        w("  **어긋나도 예외가 안 나고 성적만 나빠진다.**")
