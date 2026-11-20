"""YOLO arms of the class-balancing experiment on the FULL train split.

    python tools/build_balance_arms.py --bundle runs/balance_candidates --out runs/balance_arms

Every arm holds the real train photographs minus `--exclude` (default DSC_0093,
whose label is disputed; 217 ripe, 84 unripe remain) and the same 87 validation
photographs. Arms differ only in what is added:

    real_only            nothing                                   unripe:ripe = 84:217 = 0.39
    oversample_rXXX      k copies of real unripe train photographs
    cyclegan_rXXX        k accepted synthetic unripe photographs (full frame)

with k = round(r * ripe) - unripe for r in 0.6, 0.8, 1.0, i.e. k = 46, 90, 133 with
the default exclusion.

Both added sets are nested (the 45 are inside the 89, which are inside the 132)
and ordered by a digest of the id, never by a score. Oversampling cycles through
the 85 unripe photographs in that order, so 132 copies use some photographs twice.
The test split is not read.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melon_cyclegan import config as cfg  # noqa: E402

RATIOS = (0.6, 0.8, 1.0)
CLASSES = ["ripe", "unripe"]


def digest_order(ids):
    return sorted(ids, key=lambda i: (hashlib.sha256(i.encode()).hexdigest(), i))


def added_count(ratio, ripe, unripe):
    return int(round(ratio * ripe)) - unripe


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_arm(out, name, rows, root, added, exclude=()):
    """`added` is a list of (stem, image_path, label_text, source_id)."""
    arm = Path(out) / name
    for split in ("train", "val"):
        (arm / "images" / split).mkdir(parents=True)
        (arm / "labels" / split).mkdir(parents=True)
    real = [r for r in rows if r["split"] == "train" and r["id"] not in exclude]
    for row in real:
        shutil.copyfile(root / row["image"], arm / "images/train" / f"{row['id']}.jpg")
        shutil.copyfile(root / row["label"], arm / "labels/train" / f"{row['id']}.txt")
    manifest_added = []
    for stem, image, label, source in added:
        suffix = Path(image).suffix
        shutil.copyfile(image, arm / "images/train" / f"{stem}{suffix}")
        (arm / "labels/train" / f"{stem}.txt").write_text(label)
        class_id = int(label.split()[0])
        manifest_added.append(dict(id=stem, source=source, class_id=class_id))
    valid = [r for r in rows if r["split"] == "valid"]
    for row in valid:
        shutil.copyfile(root / row["image"], arm / "images/val" / f"{row['id']}.jpg")
        shutil.copyfile(root / row["label"], arm / "labels/val" / f"{row['id']}.txt")
    (arm / "data.yaml").write_text(f"path: {arm.resolve()}\ntrain: images/train\nval: images/val\n"
                                   f"names:\n  0: {CLASSES[0]}\n  1: {CLASSES[1]}\n")
    real_counts = collections.Counter(int(r["class_id"]) for r in real)
    added_counts = collections.Counter(a["class_id"] for a in manifest_added)
    total = real_counts + added_counts
    manifest = dict(arm=name, real=[dict(id=r["id"], class_id=int(r["class_id"])) for r in real],
                    added=manifest_added, validation=len(valid),
                    counts=dict(real=len(real), added=len(manifest_added), validation=len(valid),
                                real_per_class={str(k): v for k, v in sorted(real_counts.items())},
                                added_per_class={str(k): v for k, v in sorted(added_counts.items())},
                                total_per_class={str(k): v for k, v in sorted(total.items())},
                                unripe_to_ripe=round(total[1] / total[0], 4)),
                    excluded=sorted(exclude),
                    note="full train split plus added images; full frames only; test split not read")
    (arm / "arm_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def build_real_only(out, dataset_root=None, exclude=tuple(cfg.EXCLUDED_PARENTS)):
    """Only the `real_only` arm; used by the convergence pilot before any synthetic exists."""
    root = Path(dataset_root or cfg.DATASET_ROOT)
    rows = list(csv.DictReader(open(root / "manifest.csv")))
    out = Path(out)
    if out.exists():
        raise ValueError(f"{out} exists; use a new directory")
    counts = write_arm(out, "real_only", rows, root, [], set(exclude))["counts"]
    (out / "arms_summary.json").write_text(json.dumps(dict(excluded=sorted(exclude), arms=dict(real_only=counts)),
                                                      indent=2) + "\n")
    return counts


def build(bundle, out, dataset_root=None, ratios=RATIOS, exclude=tuple(cfg.EXCLUDED_PARENTS)):
    root = Path(dataset_root or cfg.DATASET_ROOT)
    rows = list(csv.DictReader(open(root / "manifest.csv")))
    out = Path(out)
    if out.exists():
        raise ValueError(f"{out} exists; use a new directory")
    exclude = set(exclude)
    train = [r for r in rows if r["split"] == "train" and r["id"] not in exclude]
    ripe = sum(1 for r in train if int(r["class_id"]) == cfg.RIPE)
    unripe = sum(1 for r in train if int(r["class_id"]) == cfg.UNRIPE)
    counts = {r: added_count(r, ripe, unripe) for r in ratios}
    need = max(counts.values())

    bundle = Path(bundle)
    records = json.loads((bundle / "records.json").read_text())
    if any(r["status"] == "needs_review" for r in records):
        raise ValueError("the balancing bundle still has unreviewed candidates")
    if any(not r.get("reviewer", "").strip() for r in records):
        raise ValueError("every candidate needs a named reviewer")
    accepted = {r["candidate_id"]: r for r in records
                if r["status"] == "accepted" and r["parent_id"] not in exclude}
    synthetic_order = digest_order(list(accepted))
    if len(synthetic_order) < need:
        raise ValueError(f"only {len(synthetic_order)} accepted synthetic images; {need} needed for ratio {max(ratios)}")
    unripe_order = digest_order([r["id"] for r in train if int(r["class_id"]) == cfg.UNRIPE])
    by_id = {r["id"]: r for r in rows}

    summary = dict(ripe=ripe, unripe=unripe, excluded=sorted(exclude), added_per_ratio={str(k): v for k, v in counts.items()},
                   accepted_synthetic=len(synthetic_order), arms={})
    summary["arms"]["real_only"] = write_arm(out, "real_only", rows, root, [], exclude)["counts"]
    for ratio, k in counts.items():
        tag = f"r{int(round(ratio * 100)):03d}"
        oversample = []
        for i in range(k):
            source = unripe_order[i % len(unripe_order)]
            row = by_id[source]
            oversample.append((f"{source}__dup{i // len(unripe_order)}", root / row["image"],
                               (root / row["label"]).read_text(), source))
        summary["arms"][f"oversample_{tag}"] = write_arm(out, f"oversample_{tag}", rows, root, oversample, exclude)["counts"]
        synthetic = []
        for cid in synthetic_order[:k]:
            record = accepted[cid]
            synthetic.append((cid, bundle / record["full_image"], (bundle / record["full_label"]).read_text(),
                              record["parent_id"]))
        summary["arms"][f"cyclegan_{tag}"] = write_arm(out, f"cyclegan_{tag}", rows, root, synthetic, exclude)["counts"]
    summary["synthetic_order_sha256"] = hashlib.sha256("\n".join(synthetic_order).encode()).hexdigest()
    (out / "arms_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default=None)
    parser.add_argument("--real-only", action="store_true", help="build only the real_only arm (no bundle needed)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--exclude", nargs="*", default=list(cfg.EXCLUDED_PARENTS),
                        help="train photographs left out of every arm (default: the disputed DSC_0093)")
    args = parser.parse_args(argv)
    if args.real_only:
        print("real_only", build_real_only(args.out, args.dataset, args.exclude))
        return 0
    if not args.bundle:
        parser.error("--bundle is required unless --real-only")
    summary = build(args.bundle, args.out, args.dataset, exclude=args.exclude)
    for name, counts in summary["arms"].items():
        print(name, counts["total_per_class"], counts["unripe_to_ripe"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
