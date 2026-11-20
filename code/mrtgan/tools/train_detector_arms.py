"""Train the detector on every arm and seed, under the locked protocol.

    python tools/train_detector_arms.py --arms-dir runs/detector_arms \
        --arms real_only geometry cyclegan_g0 cyclegan_g4 --out runs/detector_cyclegan

Every value of the protocol is the one frozen on 2026-09-21: yolov8n, 760
optimizer updates, batch 8, lr 1e-4, letterbox 640, horizontal flip only, five
paired seeds, validation on the real validation split. Nothing here reads the
test split. `14_Benchmark_Kaggle/tools/train_detector.py` does the training; this
script only iterates and aggregates, so the numbers stay comparable with
`04_Hasil/detector_ablation_v2/`.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melon_cyclegan import config as cfg  # noqa: E402

UPDATES, BATCH, LR, SEEDS = 760, 8, 1e-4, (0, 1, 2, 3, 4)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms-dir", required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--no-export", action="store_true", default=True,
                        help="predictions and weights are not needed for the arm comparison")
    parser.add_argument('--export', dest='no_export', action='store_false',
                        help='save predictions and weights for operational recall/FP analysis')
    parser.add_argument("--eval-every", type=int, default=0,
                        help="periodic validation and best-by-validation checkpoint (0 = historical protocol)")
    args = parser.parse_args(argv)

    trainer = cfg.BENCHMARK / "tools" / "train_detector.py"
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    results, selected, started = {}, {}, time.time()
    for arm in args.arms:
        arm_dir = Path(args.arms_dir).resolve() / arm
        if not (arm_dir / "data.yaml").exists():
            raise SystemExit(f"missing arm dataset: {arm_dir}")
        for seed in args.seeds:
            command = [sys.executable, str(trainer), "--arm", str(arm_dir), "--seed", str(seed),
                       "--updates", str(args.updates), "--batch", str(args.batch),
                       "--lr", str(args.lr), "--out", str(out)]
            if args.no_export:
                command.append("--no-export")
            if args.eval_every:
                command += ["--eval-every", str(args.eval_every)]
            code = subprocess.run(command, cwd=str(cfg.BENCHMARK)).returncode
            if code != 0:
                raise SystemExit(f"detector training failed for {arm} seed {seed}")
            summary = json.loads((out / f"{arm}__seed{seed}" / "run_summary.json").read_text())
            results[f"{arm}@{seed}"] = summary["validation"]
            if summary.get("selected"):
                selected[f"{arm}@{seed}"] = dict(update=summary["selected"]["update"],
                                                 weights=summary["selected"]["weights"],
                                                 **summary["selected"]["validation"])

    rows = []
    for arm in args.arms:
        values = [results[f"{arm}@{s}"]["map50_95"] for s in args.seeds if f"{arm}@{s}" in results]
        unripe = [results[f"{arm}@{s}"]["per_class"]["unripe"]["ap50_95"] for s in args.seeds if f"{arm}@{s}" in results]
        # `recall` is a per-class list from Ultralytics; index 1 is unripe.
        recall = [results[f"{arm}@{s}"]["recall"][1] for s in args.seeds
                  if f"{arm}@{s}" in results and len(results[f"{arm}@{s}"].get("recall", [])) > 1]
        if not values:
            continue
        rows.append(dict(arm=arm, runs=len(values),
                         map50_95=round(statistics.fmean(values), 6),
                         sd=round(statistics.pstdev(values), 6),
                         ap_unripe=round(statistics.fmean(unripe), 6),
                         recall_unripe=round(statistics.fmean(recall), 6) if recall else None))
    base = next((r["map50_95"] for r in rows if r["arm"] == "real_only"), None)
    geometry = next((r["map50_95"] for r in rows if r["arm"] == "geometry_balanced"), None)
    if geometry is None:
        geometry = next((r["map50_95"] for r in rows if r["arm"] == "geometry"), None)
    for row in rows:
        row["delta_vs_real"] = round(row["map50_95"] - base, 6) if base is not None else None
        row["delta_vs_geometry"] = round(row["map50_95"] - geometry, 6) if geometry is not None else None

    (out / "results.json").write_text(json.dumps(dict(
        results=results, selected=selected, eval_every=args.eval_every,
        arms=args.arms, seeds=args.seeds, updates=args.updates,
        batch=args.batch, lr=args.lr, summary=rows,
        geometry_reference='geometry_balanced' if 'geometry_balanced' in args.arms else 'geometry',
        seconds=round(time.time() - started, 1), test_used=False), indent=2) + "\n")
    for row in rows:
        print(row)
    print("paired seeds, fixed update budget; an arm that does not beat geometry is not evidence for synthesis")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
