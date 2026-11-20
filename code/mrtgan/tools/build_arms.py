"""Build the detector arms for the CycleGAN conditions.

    python tools/build_arms.py --bundle runs/bundle_g4 --arms cyclegan_g4 \
        --out ../04_Hasil/detector_arms_cyclegan

This is a thin wrapper over `14_Benchmark_Kaggle/tools/build_detector_arms.py`,
which is used unchanged: the bundle written by `melon_cyclegan.generate` already
has the record schema that script expects. `real_only` and `geometry` are rebuilt
by that script with the same deterministic seed, which is how the new arms stay
comparable with `04_Hasil/detector_ablation_v2/`.

Merging two bundles is supported so the G0 and G4 arms can be built in one call.
"""
from __future__ import annotations

import argparse
import json
import importlib.util
import collections
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melon_cyclegan import config as cfg  # noqa: E402


def balanced_parent_plan(parents, class_by_parent):
    """N added views such that N real + N views has N images of each class.

    Selection uses parent ids only, never a validation/model score. Multiple
    minority views are augmentation, not new independent fruit.
    """
    groups = {c: sorted(p for p in parents if int(class_by_parent[p]) == c) for c in (0, 1)}
    if not all(groups.values()):
        raise ValueError('balanced geometry requires real examples of both classes')
    plan = []
    for c in (0, 1):
        needed = len(parents) - len(groups[c])
        plan.extend((groups[c][i % len(groups[c])], i // len(groups[c])) for i in range(needed))
    return plan


def add_balanced_geometry(dataset_root, out, seed, size):
    cfg.benchmark_on_path()
    spec = importlib.util.spec_from_file_location('benchmark_arm_builder', cfg.BENCHMARK / 'tools/build_detector_arms.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    dataset = builder.Dataset(root=dataset_root)
    out = Path(out)
    summary = json.loads((out / 'arms_summary.json').read_text())
    parents = summary['parents']
    plan = balanced_parent_plan(parents, {p: dataset.by_id[p]['class_id'] for p in parents})
    added = []
    for pid, view in plan:
        # geometry_views uses parent and view in its stable seed.
        item = builder.geometry_views(dataset, [pid], seed=seed, views=view+1, size=size)[view]
        item['candidate_id'] = f'geometry_balanced__{pid}__{view}'
        added.append(item)
    manifest = builder.write_arm(dataset, 'geometry_balanced', added, parents, out, size)
    total = collections.Counter(int(r['class_id']) for r in manifest['real'] + manifest['added'])
    if total[0] != total[1]:
        raise ValueError('balanced geometry class count mismatch')
    summary['arms']['geometry_balanced'] = manifest['counts']
    (out / 'arms_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return manifest


def merge_bundles(bundles, target):
    """One directory holding the records and images of several bundles."""
    target = Path(target)
    for sub in ("images", "masks"):
        (target / sub).mkdir(parents=True, exist_ok=True)
    records = []
    for bundle in bundles:
        bundle = Path(bundle)
        part = json.loads((bundle / "records.json").read_text())
        for record in part:
            for key in ("image", "mask"):
                shutil.copyfile(bundle / record[key], target / record[key])
        records.extend(part)
    (target / "records.json").write_text(json.dumps(records, indent=2) + "\n")
    return records


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", nargs="+", required=True)
    parser.add_argument("--arms", nargs="+", required=True, help="condition names inside the bundles")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--exclude", nargs="*", default=list(cfg.EXCLUDED_PARENTS),
                        help="parents left out of every arm; the scope must be the same for all arms")
    parser.add_argument("--per-parent", type=int, default=1)
    parser.add_argument("--geometry-seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=cfg.ARM_CROP_SIZE)
    parser.add_argument('--balanced-geometry', action='store_true', help='also add a class-balanced geometry control')
    args = parser.parse_args(argv)

    script = cfg.BENCHMARK / "tools" / "build_detector_arms.py"
    with tempfile.TemporaryDirectory() as workspace:
        bundle = Path(args.bundle[0]) if len(args.bundle) == 1 else Path(workspace) / "merged"
        if len(args.bundle) > 1:
            merge_bundles(args.bundle, bundle)
        command = [sys.executable, str(script), "--bundle", str(Path(bundle).resolve()),
                   "--dataset", str(args.dataset or cfg.DATASET_ROOT),
                   "--out", str(Path(args.out).resolve()),
                   "--arms", *args.arms,
                   "--exclude", *args.exclude,
                   "--per-parent", str(args.per_parent),
                   "--geometry-seed", str(args.geometry_seed),
                   "--size", str(args.size)]
        result = subprocess.run(command, cwd=str(cfg.BENCHMARK)).returncode
        if result == 0 and args.balanced_geometry:
            add_balanced_geometry(args.dataset or cfg.DATASET_ROOT, args.out, args.geometry_seed, args.size)
        return result


if __name__ == "__main__":
    raise SystemExit(main())
