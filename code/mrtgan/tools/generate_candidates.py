"""Translate the scope parents with a trained CycleGAN and write a review bundle.

    python tools/generate_candidates.py --checkpoints-dir runs/checkpoints \
        --name melon_cyclegan_g4 --condition cyclegan_g4 --out runs/bundle_g4 --slots 2

Then review the bundle blind and ingest the sheet:

    python tools/review.py export --bundle runs/bundle_g4
    python tools/review.py ingest --bundle runs/bundle_g4 --reviewer "<name>"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melon_cyclegan import config as cfg  # noqa: E402
from melon_cyclegan import generate  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", default="runs/checkpoints")
    parser.add_argument("--name", required=True, help="experiment name used during training")
    parser.add_argument("--condition", required=True, help="arm name, e.g. cyclegan_g0 or cyclegan_g4")
    parser.add_argument("--out", required=True)
    parser.add_argument("--epoch", default="latest")
    parser.add_argument("--weights", choices=("raw", "ema"), default="raw",
                        help="raw generator weights or their EMA shadows")
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--size", type=int, default=cfg.CROP_SIZE)
    parser.add_argument("--arm-size", type=int, default=cfg.ARM_CROP_SIZE)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--device", default="cuda" if _cuda() else "cpu")
    args = parser.parse_args(argv)

    cfg.benchmark_on_path()
    from common import Dataset  # noqa: E402

    dataset = Dataset(root=args.dataset or cfg.DATASET_ROOT)
    generators, meta = generate.load_generators(args.checkpoints_dir, args.name, args.epoch,
                                               device=args.device, weights=args.weights)
    records = generate.run(dataset, generators, args.out, args.condition,
                           slots=args.slots, size=args.size, arm_size=args.arm_size,
                           device=args.device, meta=meta)
    print(f"{len(records)} candidates written to {args.out}")
    for record in records:
        print(f"  {record['candidate_id']}: {record['direction']} -> class {record['class_id']}")
    return 0


def _cuda():
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
