"""Build the ROI crop domains for the translation model.

    python tools/build_crops.py --out runs/crops256

CPU only. Reads the frozen TRAIN split, nothing else.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melon_cyclegan import config as cfg  # noqa: E402
from melon_cyclegan import crops  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset", default=None, help="frozen dataset root; defaults to regrouped v4")
    parser.add_argument("--size", type=int, default=cfg.CROP_SIZE)
    parser.add_argument("--margin", type=float, default=cfg.CROP_MARGIN)
    parser.add_argument("--split", default="train", choices=["train"],
                        help="only the train split may be used as a parent source")
    args = parser.parse_args(argv)

    manifest, centroids = crops.build(args.out, args.dataset, args.size, args.margin, args.split)
    print("crops:", manifest["counts"])
    print("mask sources:", manifest["mask_sources"])
    for name, entry in centroids.items():
        values = ", ".join(f"{v:.4f}" for v in entry["centroid"])
        print(f"phi centroid {name} (n={entry['count']}): [{values}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
