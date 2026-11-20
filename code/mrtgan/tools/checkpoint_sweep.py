"""Partial-translation rate of every saved epoch checkpoint of one run.

    python tools/checkpoint_sweep.py --checkpoints-dir runs/checkpoints \
        --name melon_cyclegan_a0_short --crops runs/crops256 --out sweep.json

G0X-v2 showed that the rate swings between checkpoints (0.1 percent at epoch 60,
22 percent at epoch 120), so one final checkpoint is a noisy reading. This tool
translates the scope parents with every `<epoch>_net_G_*.pth`, pastes the result
into the parent through its reviewed mask exactly as generation does, and
reports the mean source-side share per direction per epoch, plus the mean over
the decay phase.

Diagnostic only. It must not be used to pick a checkpoint for a comparison; the
checkpoint of record is fixed before the run. Train-split parents only.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melon_cyclegan import config as cfg  # noqa: E402
from melon_cyclegan import generate, partial  # noqa: E402

ARM_SIZE = cfg.ARM_CROP_SIZE


def epochs_of(root, weights="raw"):
    suffix = "_ema" if weights == "ema" else ""
    pattern = re.compile(rf"(\d+)_net_G_A{suffix}\.pth$")
    found = sorted({int(m.group(1)) for p in Path(root).glob(f"*_net_G_A{suffix}.pth")
                    if (m := pattern.match(p.name))})
    if not found:
        raise FileNotFoundError(f"no epoch checkpoints in {root}")
    return found


def sweep(checkpoints_dir, name, crops, dataset=None, device="cpu", decay_start=None, weights="raw"):
    cfg.benchmark_on_path()
    from common import Dataset

    dataset = dataset or Dataset(root=cfg.DATASET_ROOT)
    midpoint = partial.cr_midpoint(Path(crops) / "color_centroids.json")
    parents = [p for p in dataset.scope_ids if p not in cfg.EXCLUDED_PARENTS]
    prepared = {p: dataset.prepare(p, ARM_SIZE) for p in parents}
    classes = {p: int(dataset.by_id[p]["class_id"]) for p in parents}
    rows = []
    for epoch in epochs_of(Path(checkpoints_dir) / name, weights):
        generators, _ = generate.load_generators(checkpoints_dir, name, epoch=str(epoch), device=device,
                                                 weights=weights)
        per_parent = {}
        for parent, (source, mask, _) in prepared.items():
            target = 1 - classes[parent]
            # Ripe parents go through G_B (ripe -> unripe), unripe parents through G_A.
            key = "G_B" if classes[parent] == cfg.RIPE else "G_A"
            raw = generate.translate(generators[key], source, cfg.CROP_SIZE, device=device)
            candidate = generate.composite(raw, source, mask)
            per_parent[parent] = partial.source_side_share(
                np.asarray(candidate, np.float64) / 255.0, np.asarray(mask) > 0, target, midpoint)
        by_direction = {}
        for label, cls in (("ripe_to_unripe", cfg.RIPE), ("unripe_to_ripe", cfg.UNRIPE)):
            values = [v for p, v in per_parent.items() if classes[p] == cls]
            by_direction[label] = round(float(np.mean(values)), 4) if values else None
        rows.append(dict(epoch=epoch, per_direction=by_direction,
                         per_parent={p: round(v, 4) for p, v in per_parent.items()}))
    last = rows[-1]["epoch"]
    start = decay_start if decay_start is not None else last // 2 + 1
    decay = [r["per_direction"]["ripe_to_unripe"] for r in rows if r["epoch"] >= start]
    return dict(version=cfg.VERSION, name=name, weights=weights, rule="source-side Cr share, reviewed-mask composite, identity slot",
                midpoint=midpoint, decay_start_epoch=start, test_used=False,
                ripe_to_unripe_final=rows[-1]["per_direction"]["ripe_to_unripe"],
                ripe_to_unripe_decay_mean=round(float(np.mean(decay)), 4) if decay else None,
                ripe_to_unripe_decay_max=round(float(np.max(decay)), 4) if decay else None,
                ripe_to_unripe_decay_std=round(float(np.std(decay)), 4) if decay else None,
                epochs=rows)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--crops", required=True, help="crop pool whose color_centroids.json defines the midpoint")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--decay-start", type=int, default=None)
    parser.add_argument("--weights", choices=("raw", "ema"), default="raw")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    dataset = None
    if args.dataset:
        cfg.benchmark_on_path()
        from common import Dataset
        dataset = Dataset(root=args.dataset)
    report = sweep(args.checkpoints_dir, args.name, args.crops, dataset, args.device, args.decay_start, args.weights)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    for row in report["epochs"]:
        print(row["epoch"], row["per_direction"])
    print("ripe_to_unripe final", report["ripe_to_unripe_final"], "decay mean", report["ripe_to_unripe_decay_mean"],
          "decay std", report["ripe_to_unripe_decay_std"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
