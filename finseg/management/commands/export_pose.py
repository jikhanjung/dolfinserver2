"""밑동 두 점을 YOLO11-pose 자료로 내보낸다.

    python manage.py export_pose --out datasets/pose-v1 --dry-run
    python manage.py export_pose --out datasets/pose-v1

**왜 이것이 먼저인가.** 검토에서 `fix` 가 계속 85% 언저리다 — 열에 아홉은 사람이
두 점을 처음부터 찍는다. 상자 아래 두 모서리라는 첫 제안이 그만큼 쓸모없다는
뜻이고(`finseg/baseline.py` 3절), 그것을 바꿀 수 있는 것은 이 모델뿐이다.
**남은 검토가 있을 때 붙여야 값이 있다** — 다 끝난 뒤에 붙이면 이번 바퀴에는
아무 도움이 안 된다.

## 크롭 하나에 라벨 하나

`export_yolo` 는 크롭에 걸치는 이웃까지 전부 라벨한다. 여기는 **그 크롭의 상자
하나만** 낸다. 이웃은 검토가 안 됐을 수 있고, 밑동이 없는 지느러미를 라벨 없이
두면 모델이 그것을 배경이라고 배운다 — `rules` 가 "판정 없음" 을 배경으로 쓰지
않는 것과 같은 이유다. 그래서 **밑동 없는 지느러미 이웃이 크게 걸치는 크롭은
통째로 뺀다.**

## 두 점의 순서는 왼쪽·오른쪽이다

`base_partial` 은 앞·뒤(해부학적)로 말하지만 저장된 것은 화면 좌표다. 돌고래가
어느 쪽을 보느냐에 따라 앞이 왼쪽일 수도 오른쪽일 수도 있고 **그 방향은 어디에도
안 적혀 있다.** 그래서 여기서는 x 로 정렬해 **왼쪽 점이 늘 0번**이 되게 한다 —
모델이 배우려면 순서가 한결같아야 한다. (실제 자료도 이미 전부 그렇게 찍혀
있지만, 규칙으로 못 박아 두지 않으면 언젠가 어긋난다.)

`flip_idx: [1, 0]` 을 `data.yaml` 에 적는다. 좌우를 뒤집으면 왼쪽 점과 오른쪽
점이 자리를 바꾸므로, 이것이 없으면 `fliplr` 증강이 라벨을 망친다.

## 보임 표시(v)

`base_partial` 이 비어 있으면 2(보고 찍었다), 아니면 1(짐작해서 찍었다)이다.
**어느 쪽 점인지로는 안 가른다** — 위에 적은 이유로 앞·뒤를 점 번호에 이을 수
없기 때문이다. 짐작한 점도 자리는 대체로 맞으므로 버리지 않고 약하게 쓴다.
"""
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from finseg import geometry, rules, runs
from finseg.models import Box, Crop

MIN_VISIBLE = 0.10   # 이웃이 이만큼 넘게 걸치면 그 크롭을 막는다
NAMES = ["fin"]      # 밑동 현은 등지느러미에만 뜻이 있다


def _bbox(pts, size):
    xs = [min(max(x, 0), size) for x, _ in pts]
    ys = [min(max(y, 0), size) for _, y in pts]
    return min(xs), min(ys), max(xs), max(ys)


class Command(BaseCommand):
    help = "밑동 두 점을 YOLO11-pose 자료로 내보낸다"

    def add_arguments(self, p):
        p.add_argument("--crops", default=str(settings.FIN_CROPS))
        p.add_argument("--out", required=True)
        p.add_argument("--val-frac", type=float, default=0.2)
        p.add_argument("--val-date", help="통째로 뺄 관찰일 (기본은 씨앗으로)")
        p.add_argument("--seed", type=int, default=20260815)
        p.add_argument("--dry-run", action="store_true")

    def handle(self, **o):
        w = self.stdout.write
        boxes = list(Box.objects.select_related("image")
                     .prefetch_related("masks", "reviews"))
        states = {b.id: rules.resolve(b) for b in boxes}
        crops = {c.box_id: c for c in Crop.objects.all()}
        per_image = defaultdict(list)
        for b in boxes:
            per_image[b.image_id].append(b)

        kept, dropped = [], Counter()
        for box in boxes:
            crop = crops.get(box.id)
            if crop is None:
                continue
            st = states[box.id]
            if st["cls"] != "fin":
                dropped["등지느러미가 아니거나 아직 안 봤다"] += 1
                continue
            base = geometry.loads(st["base_line"])
            if len(base) != 2:
                dropped["밑동 두 점이 없다"] += 1
                continue
            pts = rules.final_points(st, crop)
            if not pts:
                dropped["마스크가 없다"] += 1
                continue
            # 밑동 없는 지느러미 이웃이 크게 걸치면 이 크롭은 못 쓴다
            blocked = False
            for other in per_image[box.image_id]:
                if other.id == box.id:
                    continue
                so = states[other.id]
                if so["cls"] != "fin" or geometry.loads(so["base_line"]):
                    continue
                op = rules.final_points(so, crop)
                if not op:
                    continue
                x0, y0, x1, y1 = _bbox(op, crop.w)
                area = max((max(p[0] for p in op) - min(p[0] for p in op))
                           * (max(p[1] for p in op) - min(p[1] for p in op)), 1)
                if (x1 - x0) * (y1 - y0) / area >= MIN_VISIBLE:
                    blocked = True
                    break
            if blocked:
                dropped["이웃 지느러미에 밑동 라벨이 없다"] += 1
                continue
            kp = sorted(geometry.to_crop(base, crop))   # **왼쪽 점이 늘 0번**
            kept.append({"box_id": box.id, "crop": crop, "poly": pts, "kp": kp,
                         "vis": 2 if not st["base_partial"] else 1,
                         "obsdate": str(box.image.obsdate)})

        w(f"크롭 {len(crops):,} 개 중 쓸 수 있는 것 {len(kept):,}")
        for k, v in dropped.most_common():
            w(f"  뺐다 — {k}: {v:,}")
        guessed = sum(1 for k in kept if k["vis"] == 1)
        w(f"  그중 짐작해서 찍은 점이 있는 것 {guessed:,} 개는 v=1 로 낸다")
        if len(kept) < 300:
            w(f"  ** {len(kept)} 개는 적다. 300 을 넘겨서 하는 편이 낫다")
        if not kept:
            raise CommandError("내보낼 것이 없다 — 밑동을 찍은 것이 없다.")

        by_date = defaultdict(list)
        for k in kept:
            by_date[k["obsdate"]].append(k)
        rng = random.Random(o["seed"])
        val_date = o["val_date"] or rng.choice(sorted(by_date))
        if val_date not in by_date:
            raise CommandError(f"그런 관찰일이 없다: {val_date}")
        split = {}
        for d, items in sorted(by_date.items()):
            if d == val_date:
                split.update({k["box_id"]: "val_date" for k in items})
                continue
            items = sorted(items, key=lambda k: k["box_id"])
            rng.shuffle(items)
            n_val = int(round(len(items) * o["val_frac"]))
            for i, k in enumerate(items):
                split[k["box_id"]] = "val" if i < n_val else "train"
        counts = Counter(split.values())
        w(f"\nval_date = {val_date} (통째로 뺀다)")
        w("  " + " · ".join(f"{s} {counts[s]:,}"
                            for s in ("train", "val", "val_date")))
        if o["dry_run"]:
            w("\n--dry-run 이라 아무것도 쓰지 않았다.")
            return

        out = Path(o["out"])
        for s in ("train", "val", "val_date"):
            (out / "images" / s).mkdir(parents=True, exist_ok=True)
            (out / "labels" / s).mkdir(parents=True, exist_ok=True)
        run = runs.start("export", params={"out": str(out), "task": "pose",
                                           "val_date": val_date,
                                           "val_frac": o["val_frac"],
                                           "seed": o["seed"]})
        crops_dir = Path(o["crops"])
        for k in kept:
            s = split[k["box_id"]]
            name = f"{k['box_id']:08d}"
            shutil.copyfile(crops_dir / k["crop"].path,
                            out / "images" / s / f"{name}.jpg")
            size = k["crop"].w
            x0, y0, x1, y1 = _bbox(k["poly"], size)
            n = lambda v: min(max(v / size, 0), 1)
            line = (f"0 {n((x0 + x1) / 2):.6f} {n((y0 + y1) / 2):.6f} "
                    f"{n(x1 - x0):.6f} {n(y1 - y0):.6f}")
            for x, y in k["kp"]:
                line += f" {n(x):.6f} {n(y):.6f} {k['vis']}"
            (out / "labels" / s / f"{name}.txt").write_text(line + "\n")

        # `kpt_shape` 와 `flip_idx` 는 ultralytics 가 여기서 읽는다. flip_idx 가
        # 없으면 fliplr 증강이 왼쪽·오른쪽 점을 안 바꾼 채 그림만 뒤집는다
        # **통째로 뺀 관찰일을 `test:` 로도 적는다.** ultralytics 는
        # `train`·`val`·`test` 만 절대경로로 풀어 준다 — 그 밖의 키는 상대경로로
        # 남고, 그러면 `images/…` → `labels/…` 치환이 안 먹어 **라벨을 하나도 못
        # 읽은 채 "92 backgrounds" 라고만 말한다.** 성적이 0 으로 나오는데 그것이
        # 모델 탓인지 자료 탓인지 화면이 구분해 주지 않는다.
        (out / "data.yaml").write_text(
            "path: .\ntrain: images/train\nval: images/val\n"
            "# 통째로 뺀 관찰일 — 성적은 이쪽으로 읽는다.\n"
            "# `test` 로 적어야 ultralytics 가 라벨을 찾는다 (val_date 는 이름표다)\n"
            "test: images/val_date\n"
            "val_date: images/val_date\n\n"
            "kpt_shape: [2, 3]\n"
            "# 좌우를 뒤집으면 왼쪽 점과 오른쪽 점이 자리를 바꾼다\n"
            "flip_idx: [1, 0]\n\nnames:\n  0: fin\n")
        (out / "MANIFEST.json").write_text(json.dumps({
            "git_sha": runs.git_sha(),
            "export_run": run.id,
            "task": "pose",
            "val_date": val_date, "val_frac": o["val_frac"], "seed": o["seed"],
            "names": NAMES,
            "keypoints": ["base_left", "base_right"],
            "counts": {**{s: counts[s] for s in ("train", "val", "val_date")},
                       "labels": len(kept), "guessed_points": guessed},
            "dropped": dict(dropped),
            "obsdates": sorted(by_date),
            "boxes": {s: sorted(b for b, v in split.items() if v == s)
                      for s in ("train", "val", "val_date")},
        }, ensure_ascii=False, indent=2))
        runs.finish(run)
        w(f"\nrun {run.id} · {out}")
        w("다음: manage.py train --data " + str(out))
