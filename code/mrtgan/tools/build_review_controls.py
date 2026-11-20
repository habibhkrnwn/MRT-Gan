"""Real train photographs as hidden controls inside a blind review.

    python tools/build_review_controls.py --crops runs/crops256 --out runs/review_controls --per-class 20

When every candidate in a review has the same target class, a reviewer can answer
by default. Real photographs of both classes, rendered exactly like the balancing
candidates (the fruit window of the full photograph at 512 px) and shuffled in with
them, keep the review blind and measure the reviewer's agreement with the frozen
labels. Controls never enter a detector arm.

Selection: the first `--per-class` train ids of each class in digest order, never a
score. DSC_0093 (disputed label) is skipped. Train split only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melon_cyclegan import config as cfg  # noqa: E402

REVIEW_SIZE = 512


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--crops", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args(argv)
    crops, out = Path(args.crops), Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; use a new directory")
    root = Path(args.dataset or cfg.DATASET_ROOT)
    rows = {r["id"]: r for r in csv.DictReader(open(root / "manifest.csv"))}
    manifest = json.loads((crops / "crops_manifest.json").read_text())
    by_class = {0: [], 1: []}
    for record in manifest["records"]:
        if record["split"] == "train" and record["id"] not in cfg.EXCLUDED_PARENTS:
            by_class[record["class_id"]].append(record)
    (out / "images").mkdir(parents=True)
    (out / "masks").mkdir()
    records = []
    for class_id, items in by_class.items():
        ordered = sorted(items, key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())[:args.per_class]
        for crop in ordered:
            left, top, side = crop["crop_window"]
            with Image.open(root / rows[crop["id"]]["image"]) as handle:
                full = handle.convert("RGB")
            view = full.crop((left, top, left + side, top + side)).resize((REVIEW_SIZE, REVIEW_SIZE),
                                                                         Image.Resampling.LANCZOS)
            candidate_id = f"control__{crop['id']}"
            view.save(out / "images" / f"{candidate_id}.png")
            Image.open(crops / crop["mask"]).convert("L").resize((REVIEW_SIZE, REVIEW_SIZE),
                                                                 Image.Resampling.NEAREST).save(
                out / "masks" / f"{candidate_id}.png")
            scale = REVIEW_SIZE / 256.0
            x, y, w, h = crop["crop_box"]
            records.append(dict(candidate_id=candidate_id, condition="review_control", method="real",
                                parent_id=crop["id"], class_id=class_id, parent_class_id=class_id,
                                direction="real_control", image=f"images/{candidate_id}.png",
                                mask=f"masks/{candidate_id}.png",
                                bbox=[int(x * scale), int(y * scale), int(w * scale), int(h * scale)],
                                status="needs_review", reviewer=""))
    (out / "records.json").write_text(json.dumps(records, indent=2) + "\n")
    print(f"{len(records)} controls -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
