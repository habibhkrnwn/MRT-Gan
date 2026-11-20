"""Automatic quality control over a generated bundle.

    python tools/run_qc.py --bundle runs/bundle_g4 --crops runs/crops256

Writes `quality_control.json` inside the bundle. CPU only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melon_cyclegan import config as cfg  # noqa: E402
from melon_cyclegan import qc  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--crops", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--threshold", type=float, default=cfg.QC["target_classifier_threshold"])
    parser.add_argument("--composite-trained", action="store_true",
                        help="rung trained with the mask composite in the forward pass (G0X); "
                             "r_bg is then reported but does not gate")
    args = parser.parse_args(argv)

    cfg.benchmark_on_path()
    from common import Dataset  # noqa: E402

    dataset = Dataset(root=args.dataset or cfg.DATASET_ROOT)
    report = qc.evaluate(args.bundle, args.crops, dataset, args.threshold,
                         composite_trained=args.composite_trained)
    print("counts:", report["counts"])
    print("per direction:", report["per_direction"])
    for row in report["rows"]:
        print(f"  {row['candidate_id']}: p={row['classifier_probability']:.3f} "
              f"r_bg={row['background_change_ratio']:.3f} "
              f"d_target={row['color_distance_target']:.3f} d_source={row['color_distance_source']:.3f} "
              f"ssim_in={row['ssim_inside']:.3f} -> {'keep' if row['recommended'] else 'flag'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
