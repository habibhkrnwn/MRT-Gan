"""Freeze every decision of the balancing experiment BEFORE the test split is read.

    python tools/lock_final.py --results runs/kaggle_balance/balance_detector/results.json \
        --runs-root runs/kaggle_balance/balance_detector --out runs/FINAL_LOCK_balance.json

Records, from validation results only:

* the selected ratio: the `cyclegan_rXXX` arm with the highest mean validation
  mAP50:95 of its selected checkpoints (ties within 1e-6: the lower ratio);
* for every arm and seed, the selected checkpoint file and its SHA-256;
* SHA-256 of the evaluation code and of the dataset manifest, and the list of
  test ids (identity only, no pixels).

`tools/final_test_eval.py` refuses to run unless every hash still matches. The
lock is never rewritten: a second lock needs a new file name and a logged reason.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from melon_cyclegan import config as cfg  # noqa: E402

CODE_FILES = ("tools/final_test_eval.py", "melon_cyclegan/detection_metrics.py", "tools/lock_final.py")
CYCLEGAN_ARMS = ("cyclegan_r060", "cyclegan_r080", "cyclegan_r100")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose_ratio(selected, seeds, arms=CYCLEGAN_ARMS):
    arms = [arm for arm in arms if f"{arm}@{seeds[0]}" in selected]
    means = {arm: statistics.fmean(selected[f"{arm}@{s}"]["map50_95"] for s in seeds) for arm in arms}
    best = max(means.values())
    tied = [arm for arm in arms if best - means[arm] <= 1e-6]
    return sorted(tied)[0], means


def find_run(runs_root, arm, seed):
    hits = sorted(Path(runs_root).glob(f"group*/{arm}__seed{seed}"))
    if len(hits) != 1:
        raise ValueError(f"expected one run directory for {arm} seed {seed}, found {len(hits)}")
    return hits[0]


def build_lock(results_path, runs_root, dataset_root=None):
    results = json.loads(Path(results_path).read_text())
    if results.get("test_used"):
        raise ValueError("results already report test use")
    seeds, arms, selected = results["seeds"], results["arms"], results["results_selected"]
    chosen, means = choose_ratio(selected, seeds)
    runs = {}
    for arm in arms:
        for seed in seeds:
            key = f"{arm}@{seed}"
            run_dir = find_run(runs_root, arm, seed)
            weights = run_dir / selected[key]["weights"]
            runs[key] = dict(run_dir=str(run_dir.relative_to(Path(runs_root))), weights=weights.name,
                             weights_sha256=sha(weights), selected_update=selected[key]["update"],
                             valid_map50_95=selected[key]["map50_95"])
    root = Path(dataset_root or cfg.DATASET_ROOT)
    test_ids = sorted(r["id"] for r in csv.DictReader(open(root / "manifest.csv")) if r["split"] == "test")
    return dict(
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), test_opened=False,
        rule_ratio="highest mean validation mAP50:95 of selected checkpoints among cyclegan arms; ties -> lower ratio",
        selected_ratio_arm=chosen, selected_ratio_oversample_arm=chosen.replace("cyclegan", "oversample"),
        valid_means_cyclegan=means, arms=arms, seeds=seeds, runs=runs,
        results_sha256=sha(results_path), dataset_manifest_sha256=sha(root / "manifest.csv"),
        test_ids=test_ids, test_ids_sha256=hashlib.sha256("\n".join(test_ids).encode()).hexdigest(),
        code_sha256={name: sha(ROOT / name) for name in CODE_FILES},
        primary_comparisons=[[chosen, chosen.replace("cyclegan", "oversample")], [chosen, "real_only"]],
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; a lock is never rewritten")
    lock = build_lock(args.results, args.runs_root, args.dataset)
    out.write_text(json.dumps(lock, indent=2) + "\n")
    print("selected ratio:", lock["selected_ratio_arm"], {k: round(v, 4) for k, v in lock["valid_means_cyclegan"].items()})
    print("locked", len(lock["runs"]), "runs ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
