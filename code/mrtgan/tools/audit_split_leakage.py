"""Near-duplicate audit between splits of the frozen dataset. CPU only.

    python tools/audit_split_leakage.py --out runs/split_leakage_audit.json

Byte-identical and same-source overlaps are already excluded by the regrouped v4
manifest (SHA-256, photograph_group and `.rf.` source stem). This audit looks for
near-duplicates: the same fruit photographed again, or a re-encoded copy. For
every valid and test image it records the nearest train image under two
perceptual hashes (difference hash, 16 x 16 = 256 bits):

    whole   the full photograph
    fruit   the labelled fruit box, padded by 10 percent

Only image identity is computed. No model is run and nothing here may inform a
method, ratio or hyperparameter choice. A flagged pair is a candidate for human
inspection, not an automatic removal.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melon_cyclegan import config as cfg  # noqa: E402

HASH_SIZE = 16


def dhash_bits(image, size=HASH_SIZE):
    grey = np.asarray(image.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS), np.int16)
    return (grey[:, 1:] > grey[:, :-1]).reshape(-1)


def fruit_box(root, row, image):
    width, height = image.size
    lines = [l for l in (root / row["label"]).read_text().splitlines() if l.strip()]
    _, x, y, w, h = map(float, lines[0].split())
    pad_w, pad_h = w * 0.1, h * 0.1
    left = max(0, (x - w / 2 - pad_w) * width)
    top = max(0, (y - h / 2 - pad_h) * height)
    right = min(width, (x + w / 2 + pad_w) * width)
    bottom = min(height, (y + h / 2 + pad_h) * height)
    return image.crop((int(left), int(top), int(right), int(bottom)))


def audit(root, flag_fraction=0.10):
    root = Path(root)
    rows = list(csv.DictReader(open(root / "manifest.csv")))
    hashes = {}
    for row in rows:
        with Image.open(root / row["image"]) as handle:
            image = handle.convert("RGB")
        hashes[row["id"]] = dict(split=row["split"], class_id=row["class_id"], group=row["photograph_group"],
                                 whole=dhash_bits(image), fruit=dhash_bits(fruit_box(root, row, image)))
    train = [k for k, v in hashes.items() if v["split"] == "train"]
    train_whole = np.stack([hashes[k]["whole"] for k in train])
    train_fruit = np.stack([hashes[k]["fruit"] for k in train])
    bits = HASH_SIZE * HASH_SIZE
    limit = int(round(flag_fraction * bits))
    report, flagged = [], []
    for key, value in hashes.items():
        if value["split"] == "train":
            continue
        d_whole = (train_whole != value["whole"]).sum(axis=1)
        d_fruit = (train_fruit != value["fruit"]).sum(axis=1)
        i, j = int(d_whole.argmin()), int(d_fruit.argmin())
        entry = dict(id=key, split=value["split"], class_id=value["class_id"],
                     nearest_whole=train[i], whole_distance=int(d_whole[i]),
                     nearest_fruit=train[j], fruit_distance=int(d_fruit[j]))
        report.append(entry)
        if entry["whole_distance"] <= limit or entry["fruit_distance"] <= limit:
            flagged.append(entry)
    # Reference: nearest-neighbour distances inside train, excluding self.
    within = []
    for n in range(len(train)):
        d = (train_whole != train_whole[n]).sum(axis=1)
        d[n] = bits
        within.append(int(d.min()))
    summary = {}
    for split in ("valid", "test"):
        part = [e for e in report if e["split"] == split]
        summary[split] = dict(
            n=len(part),
            whole_distance_min=min(e["whole_distance"] for e in part),
            whole_distance_median=float(np.median([e["whole_distance"] for e in part])),
            fruit_distance_min=min(e["fruit_distance"] for e in part),
            fruit_distance_median=float(np.median([e["fruit_distance"] for e in part])),
            flagged=sum(1 for e in flagged if e["split"] == split))
    return dict(dataset_manifest=cfg.EXPECTED_MANIFEST, hash="dHash", hash_bits=bits,
                flag_rule=f"distance <= {limit} bits ({flag_fraction:.0%}) on whole image or fruit box",
                train_within_nearest=dict(min=min(within), median=float(np.median(within)),
                                          p5=float(np.percentile(within, 5))),
                summary=summary, flagged=sorted(flagged, key=lambda e: min(e["whole_distance"], e["fruit_distance"])),
                all=report, models_used=False)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(cfg.DATASET_ROOT))
    parser.add_argument("--flag-fraction", type=float, default=0.10)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = audit(args.dataset, args.flag_fraction)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(flag_rule=report["flag_rule"], train_within_nearest=report["train_within_nearest"],
                          summary=report["summary"]), indent=2))
    for e in report["flagged"][:30]:
        print(e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
