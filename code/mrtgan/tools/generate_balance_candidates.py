"""Full-frame synthetic unripe images for the class-balancing experiment.

    python tools/generate_balance_candidates.py --checkpoints-dir runs/kaggle_ema/checkpoints \
        --name melon_cyclegan_g0x_ema --weights raw --crops runs/crops256 --out runs/balance_candidates

For every ripe TRAIN photograph (217, one candidate each):

1. take the fruit crop the generator was trained on (`crops256/trainB/<id>.png`,
   square window around the labelled box with a 15 percent margin, 256 px) and
   its GrabCut mask,
2. translate it with G_B (ripe -> unripe),
3. upsample the translation to the window size and paste it into the original
   full-resolution photograph through the upsampled mask; pixels outside the mask
   and outside the image stay the original bytes,
4. keep the original YOLO box and flip its class to unripe.

The result has the scale and scene of a real training photograph, so it can be
compared directly with oversampling real unripe photographs. A review crop of the
pasted window is written for the blind review; the candidate becomes usable only
when a named reviewer accepts it. Train split only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melon_cyclegan import config as cfg  # noqa: E402
from melon_cyclegan import generate  # noqa: E402

REVIEW_SIZE = 512
JPEG_QUALITY = 95


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paste(full, translated, mask, window):
    """Paste a square window translation into `full` through `mask`, clipped to the image."""
    left, top, side = window
    big = np.asarray(translated.resize((side, side), Image.Resampling.LANCZOS), np.uint8)
    big_mask = np.asarray(mask.resize((side, side), Image.Resampling.NEAREST)) > 127
    out = np.asarray(full, np.uint8).copy()
    height, width = out.shape[:2]
    x0, y0 = max(0, left), max(0, top)
    x1, y1 = min(width, left + side), min(height, top + side)
    region = big_mask[y0 - top:y1 - top, x0 - left:x1 - left]
    target = out[y0:y1, x0:x1]
    target[region] = big[y0 - top:y1 - top, x0 - left:x1 - left][region]
    out[y0:y1, x0:x1] = target
    return Image.fromarray(out), (x0, y0, x1, y1)


def flipped_label(label_text, to_class):
    lines = [l.split() for l in label_text.splitlines() if l.strip()]
    if len(lines) != 1:
        raise ValueError("expected exactly one melon box")
    return " ".join([str(to_class)] + lines[0][1:]) + "\n"


def run(checkpoints_dir, name, weights, crops, out, dataset_root=None, device="cpu"):
    crops, out = Path(crops), Path(out)
    if out.exists():
        raise ValueError(f"{out} exists; use a new directory")
    manifest = json.loads((crops / "crops_manifest.json").read_text())
    root = Path(dataset_root or cfg.DATASET_ROOT)
    import csv
    rows = {r["id"]: r for r in csv.DictReader(open(root / "manifest.csv"))}
    generators, meta = generate.load_generators(checkpoints_dir, name, device=device, weights=weights)
    for sub in ("full/images", "full/labels", "images", "masks"):
        (out / sub).mkdir(parents=True)
    records = []
    for crop in sorted(manifest["records"], key=lambda r: r["id"]):
        if crop["class_id"] != cfg.RIPE or crop["split"] != "train":
            continue
        row = rows[crop["id"]]
        if row["split"] != "train" or _sha(root / row["image"]) != crop["source_sha256"]:
            raise ValueError(f"{crop['id']}: source photograph differs from the crop manifest")
        source_crop = Image.open(crops / crop["image"]).convert("RGB")
        mask = Image.open(crops / crop["mask"]).convert("L")
        raw = generate.translate(generators["G_B"], source_crop, source_crop.size[0], device=device)
        with Image.open(root / row["image"]) as handle:
            full = handle.convert("RGB")
        synthetic, clip = paste(full, raw, mask, crop["crop_window"])
        candidate_id = f"balance__{crop['id']}"
        full_image = out / "full/images" / f"{candidate_id}.jpg"
        synthetic.save(full_image, quality=JPEG_QUALITY)
        (out / "full/labels" / f"{candidate_id}.txt").write_text(
            flipped_label((root / row["label"]).read_text(), cfg.UNRIPE))
        # Review view: the pasted window of the final photograph and its mask, same geometry.
        left, top, side = crop["crop_window"]
        view = synthetic.crop((left, top, left + side, top + side)).resize((REVIEW_SIZE, REVIEW_SIZE),
                                                                          Image.Resampling.LANCZOS)
        view.save(out / "images" / f"{candidate_id}.png")
        view_mask = mask.resize((REVIEW_SIZE, REVIEW_SIZE), Image.Resampling.NEAREST)
        view_mask.save(out / "masks" / f"{candidate_id}.png")
        scale = REVIEW_SIZE / 256.0
        x, y, w, h = crop["crop_box"]
        records.append(dict(
            candidate_id=candidate_id, condition=f"balance_{name}_{weights}", method="cyclegan",
            parent_id=crop["id"], class_id=cfg.UNRIPE, parent_class_id=cfg.RIPE, direction="ripe_to_unripe",
            image=f"images/{candidate_id}.png", mask=f"masks/{candidate_id}.png",
            bbox=[int(x * scale), int(y * scale), int(w * scale), int(h * scale)],
            full_image=f"full/images/{candidate_id}.jpg", full_label=f"full/labels/{candidate_id}.txt",
            full_image_sha256=_sha(full_image), paste_region=list(clip), crop_window=crop["crop_window"],
            status="needs_review", reviewer="", trace=dict(meta, crops_digest=manifest["crops_digest"]),
        ))
    (out / "records.json").write_text(json.dumps(records, indent=2) + "\n")
    (out / "run_summary.json").write_text(json.dumps(dict(
        version=cfg.VERSION, generator=meta, candidates=len(records), crops_digest=manifest["crops_digest"],
        dataset_manifest=cfg.EXPECTED_MANIFEST, split="train", test_used=False), indent=2) + "\n")
    return records


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--weights", choices=("raw", "ema"), default="raw")
    parser.add_argument("--crops", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    records = run(args.checkpoints_dir, args.name, args.weights, args.crops, args.out, args.dataset, args.device)
    print(f"{len(records)} candidates -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
