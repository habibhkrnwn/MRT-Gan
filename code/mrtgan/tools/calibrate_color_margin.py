"""Calibrate the margin m of the G0X colour hinge on real train crops. CPU only.

    python tools/calibrate_color_margin.py --crops runs/crops256

For every real crop, the separation `d(own class) - d(other class)` is measured
in the same z space the loss uses (`losses.margin_distances`). The margin is the
10th percentile of `d_other - d_own` in the class where that percentile is
smallest. So at least 90 percent of real crops of each class already satisfy the
hinge with zero loss, and the loss only asks a translation to be as clearly on
the target side as a typical real fruit.

The 10th percentile is a pre-registered tolerance for colour overlap, not proof
that the remaining crops have wrong labels. Recompute after any pool change.

Only `trainA` and `trainB` are read; the crop set itself is train split only.
Writes `color_margin.json` next to the crops.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melon_cyclegan import config as cfg  # noqa: E402
from melon_cyclegan.color import phi_numpy  # noqa: E402
from melon_cyclegan.losses import MARGIN_DIMS, margin_distances  # noqa: E402
from melon_cyclegan.model_impl import load_centroids, load_color_scale  # noqa: E402

PERCENTILE = 10.0
DOMAINS = ("A", "B")


def crop_phis(crops, domain):
    """phi of every crop of one domain, computed inside its mask."""
    folder, masks = Path(crops) / f"train{domain}", Path(crops) / f"train{domain}_mask"
    out = []
    for path in sorted(folder.glob("*.png")):
        with Image.open(path) as handle:
            image = np.asarray(handle.convert("RGB"), np.float64) / 255.0
        with Image.open(masks / path.name) as handle:
            mask = np.asarray(handle.convert("L")) > 127
        out.append(phi_numpy(image, mask))
    if not out:
        raise ValueError(f"no crops in {folder}")
    return torch.tensor(np.stack(out), dtype=torch.float32)


def calibrate(crops, percentile=PERCENTILE):
    crops = Path(crops)
    path = crops / "color_centroids.json"
    centroids, scale = load_centroids(path), load_color_scale(path)
    per_domain = {}
    for domain in DOMAINS:
        other = "B" if domain == "A" else "A"
        phi = crop_phis(crops, domain)
        d_own, d_other = margin_distances(phi, centroids[domain], centroids[other], scale)
        gap = (d_other - d_own).numpy()
        per_domain[domain] = dict(
            class_name=cfg.CLASS_NAMES[cfg.DOMAIN_CLASS[domain]], n=int(gap.size),
            gap_percentile=float(np.percentile(gap, percentile)),
            gap_median=float(np.median(gap)), gap_min=float(gap.min()),
            share_on_own_side=float((gap > 0).mean()),
            overlap_ids=[p.stem for p, value in zip(sorted((crops / f'train{domain}').glob('*.png')), gap)
                         if value <= 0],
        )
    margin = max(0.0, min(v["gap_percentile"] for v in per_domain.values()))
    for domain, values in per_domain.items():
        phi = crop_phis(crops, domain)
        other = "B" if domain == "A" else "A"
        d_own, d_other = margin_distances(phi, centroids[domain], centroids[other], scale)
        values["share_zero_loss_at_margin"] = float(((d_own - d_other + margin) <= 0).float().mean())
    return dict(
        version=cfg.VERSION, source="train split real crops", crops=str(crops),
        crops_digest=json.loads(path.read_text()).get("crops_digest"),
        centroids_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        dims=[["mean Cr", "std Cr", "mean a*/128", "std a*/128"][d] for d in MARGIN_DIMS],
        scale=[float(scale[d]) for d in MARGIN_DIMS], percentile=percentile,
        margin=round(margin, 4), per_domain=per_domain,
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--crops", required=True)
    parser.add_argument("--percentile", type=float, default=PERCENTILE)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    report = calibrate(args.crops, args.percentile)
    out = Path(args.out or Path(args.crops) / "color_margin.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
