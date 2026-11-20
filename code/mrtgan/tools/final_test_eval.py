"""One evaluation of the locked balancing checkpoints on the real test split. CPU is enough.

    python tools/final_test_eval.py --lock runs/FINAL_LOCK_balance.json \
        --runs-root runs/kaggle_balance/balance_detector --out runs/final_test_balance

Refuses to run when the lock is missing, when any locked hash differs (weights,
code, dataset manifest, test ids), or when the output already holds results. The
only way to run twice is `--rerun-reason`, which is written into the results.

Per run: Ultralytics validation on the test split with `save_json`, which gives
the point metrics and the per-image predictions of the validator itself. Per
comparison: mean and SD of the per-seed differences, seeds won, and a 95 percent
bootstrap interval over test images (the seed-mean metric recomputed on each
resample with `melon_cyclegan.detection_metrics`, which reproduces the validator's
mAP to four decimals on the validation split).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from melon_cyclegan import config as cfg  # noqa: E402
from melon_cyclegan import detection_metrics as dm  # noqa: E402
import lock_final  # noqa: E402


def verify(lock, runs_root, dataset_root):
    problems = []
    for name, digest in lock["code_sha256"].items():
        if lock_final.sha(ROOT / name) != digest:
            problems.append(f"code changed: {name}")
    if lock_final.sha(Path(dataset_root) / "manifest.csv") != lock["dataset_manifest_sha256"]:
        problems.append("dataset manifest changed")
    ids = sorted(r["id"] for r in csv.DictReader(open(Path(dataset_root) / "manifest.csv")) if r["split"] == "test")
    if hashlib.sha256("\n".join(ids).encode()).hexdigest() != lock["test_ids_sha256"]:
        problems.append("test ids changed")
    for key, run in lock["runs"].items():
        path = Path(runs_root) / run["run_dir"] / run["weights"]
        if not path.exists() or lock_final.sha(path) != run["weights_sha256"]:
            problems.append(f"weights differ: {key}")
    return problems


def test_split_dir(dataset_root, out, split="test"):
    """YOLO folder of the evaluated split; for `test` the only place its pixels are read."""
    root = Path(dataset_root)
    target = Path(out) / "_test_split"
    (target / "images/test").mkdir(parents=True)
    (target / "labels/test").mkdir(parents=True)
    gt = dict(images=[], annotations=[])
    from PIL import Image
    for row in csv.DictReader(open(root / "manifest.csv")):
        if row["split"] != split:
            continue
        shutil.copyfile(root / row["image"], target / "images/test" / f"{row['id']}.jpg")
        shutil.copyfile(root / row["label"], target / "labels/test" / f"{row['id']}.txt")
        with Image.open(root / row["image"]) as handle:
            width, height = handle.size
        gt["images"].append(dict(id=row["id"]))
        for line in (root / row["label"]).read_text().splitlines():
            if line.strip():
                c, x, y, w, h = map(float, line.split())
                gt["annotations"].append(dict(image_id=row["id"], category_id=int(c) + 1,
                                              bbox=[(x - w / 2) * width, (y - h / 2) * height, w * width, h * height]))
    (target / "data.yaml").write_text(f"path: {target.resolve()}\ntrain: images/test\nval: images/test\n"
                                      "test: images/test\nnames:\n  0: ripe\n  1: unripe\n")
    return target, gt


def evaluate_run(weights, data_yaml, save_dir):
    import torch
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    net = DetectionModel(checkpoint["cfg"], nc=2, verbose=False)
    net.load_state_dict(checkpoint["state_dict"])
    wrapper = YOLO(checkpoint["cfg"])
    wrapper.model = net.eval()
    metrics = wrapper.val(data=str(data_yaml), split="test", verbose=False, plots=False, save_json=True,
                          project=str(save_dir.parent), name=save_dir.name, exist_ok=True)
    box = metrics.box
    point = dict(map50_95=float(box.map), map50=float(box.map50),
                 per_class={name: dict(ap50_95=float(box.maps[i])) for i, name in enumerate(["ripe", "unripe"])},
                 precision=[float(v) for v in box.p], recall=[float(v) for v in box.r])
    predictions = json.loads((Path(metrics.save_dir) / "predictions.json").read_text())
    return point, [dict(image_id=p["image_id"], category_id=p["category_id"], bbox=p["bbox"], score=p["score"])
                   for p in predictions]


def paired(a, b, seeds, table, metric):
    diffs = [table[f"{a}@{s}"][metric] - table[f"{b}@{s}"][metric] for s in seeds]
    return dict(mean=statistics.fmean(diffs), sd=statistics.pstdev(diffs), wins=sum(d > 0 for d in diffs),
                n=len(diffs), per_seed=diffs)


def bootstrap(per_image, arms, seeds, pairs, image_ids, resamples, seed):
    rng = np.random.default_rng(seed)
    draws = {tuple(p): [] for p in pairs}
    arm_draws = {arm: [] for arm in arms}
    for _ in range(resamples):
        sample = list(rng.choice(image_ids, size=len(image_ids), replace=True))
        means = {arm: statistics.fmean(dm.map_from(per_image[f"{arm}@{s}"], sample)["map50_95"] for s in seeds)
                 for arm in arms}
        for arm in arms:
            arm_draws[arm].append(means[arm])
        for a, b in pairs:
            draws[(a, b)].append(means[a] - means[b])
    ci = lambda values: [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
    return (dict(resamples=resamples, seed=seed,
                 pairs={f"{a} - {b}": dict(ci95=ci(v), share_positive=float(np.mean(np.asarray(v) > 0)))
                        for (a, b), v in draws.items()},
                 arms={arm: dict(ci95=ci(v)) for arm, v in arm_draws.items()}))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--rerun-reason", default=None)
    parser.add_argument("--split", choices=("test", "valid"), default="test",
                        help="valid = dry run of the whole procedure without reading the test split")
    args = parser.parse_args(argv)
    lock = json.loads(Path(args.lock).read_text())
    dataset_root = Path(args.dataset or cfg.DATASET_ROOT)
    out = Path(args.out)
    result_name = "test_results.json" if args.split == "test" else "valid_dryrun_results.json"
    if (out / result_name).exists() and not args.rerun_reason:
        raise SystemExit(f"{out / result_name} already exists; each split is evaluated once "
                         "(use --rerun-reason for a logged technical rerun)")
    problems = verify(lock, args.runs_root, dataset_root)
    if problems:
        raise SystemExit("lock verification failed: " + "; ".join(problems))
    out.mkdir(parents=True, exist_ok=True)
    if (out / "_test_split").exists():
        shutil.rmtree(out / "_test_split")
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    split, gt = test_split_dir(dataset_root, out, args.split)
    arms, seeds = lock["arms"], lock["seeds"]
    points, per_image = {}, {}
    (out / "predictions").mkdir(exist_ok=True)
    for key, run in lock["runs"].items():
        weights = Path(args.runs_root) / run["run_dir"] / run["weights"]
        point, predictions = evaluate_run(weights, split / "data.yaml", out / "val_runs" / key.replace("@", "__seed"))
        points[key] = point
        (out / "predictions" / f"{key.replace('@', '__seed')}.json").write_text(json.dumps(predictions) + "\n")
        per_image[key] = dm.prepare(gt, predictions)
        print(key, round(point["map50_95"], 4), flush=True)
    image_ids = [i["id"] for i in gt["images"]]
    own = {k: dm.map_from(v, image_ids) for k, v in per_image.items()}
    summary = []
    for arm in arms:
        keys = [f"{arm}@{s}" for s in seeds]
        summary.append(dict(arm=arm,
                            map50_95=statistics.fmean(points[k]["map50_95"] for k in keys),
                            sd=statistics.pstdev(points[k]["map50_95"] for k in keys),
                            map50=statistics.fmean(points[k]["map50"] for k in keys),
                            ap_ripe=statistics.fmean(points[k]["per_class"]["ripe"]["ap50_95"] for k in keys),
                            ap_unripe=statistics.fmean(points[k]["per_class"]["unripe"]["ap50_95"] for k in keys),
                            recall_unripe=statistics.fmean(points[k]["recall"][1] for k in keys),
                            precision_unripe=statistics.fmean(points[k]["precision"][1] for k in keys)))
    primary = [tuple(p) for p in lock["primary_comparisons"] if all(arm in arms for arm in p)]
    missing = [p for p in lock["primary_comparisons"] if tuple(p) not in primary]
    if missing and args.split == "test":
        raise SystemExit(f"primary comparisons reference arms that were not locked: {missing}")
    secondary = []
    for tag in ("r060", "r080", "r100"):
        for pair in ((f"cyclegan_{tag}", f"oversample_{tag}"), (f"cyclegan_{tag}", "real_only"),
                     (f"oversample_{tag}", "real_only")):
            if pair not in primary and all(arm in arms for arm in pair):
                secondary.append(pair)
    comparisons = {f"{a} - {b}": dict(primary=(a, b) in primary,
                                      map50_95=paired(a, b, seeds, points, "map50_95"))
                   for a, b in primary + secondary}
    boot = bootstrap(per_image, arms, seeds, primary + secondary, image_ids, args.bootstrap, args.bootstrap_seed)
    results = dict(lock=str(args.lock), lock_sha256=lock_final.sha(args.lock), started_at=started,
                   finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), rerun_reason=args.rerun_reason,
                   split=args.split, images=len(image_ids), test_used=args.split == "test", selected_ratio_arm=lock["selected_ratio_arm"],
                   points=points, own_metric_check={k: v["map50_95"] - points[k]["map50_95"] for k, v in own.items()},
                   summary=summary, comparisons=comparisons, bootstrap=boot)
    (out / result_name).write_text(json.dumps(results, indent=2) + "\n")
    shutil.rmtree(split)
    for row in summary:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
