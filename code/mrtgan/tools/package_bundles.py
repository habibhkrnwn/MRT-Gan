"""Package the reviewed bundles as the Kaggle dataset the arms notebook reads.

    python tools/package_bundles.py --bundle runs/bundle_cyclegan_g0 runs/bundle_cyclegan_g4 \
        --out dist/melon-cyclegan-bundles

Refuses to package a bundle that still holds `needs_review` candidates or whose
reviewer field is empty. The blind review is the only path to `accepted`, and
this check is what keeps it that way once the files leave the laptop.
"""
from __future__ import annotations

import argparse
import collections
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

KEEP = ("records.json", "quality_control.json", "run_summary.json", "cost.json")


def check(bundle):
    bundle = Path(bundle)
    records = json.loads((bundle / "records.json").read_text())
    counts = collections.Counter(record["status"] for record in records)
    if counts.get("needs_review"):
        raise SystemExit(f"{bundle.name}: {counts['needs_review']} candidates still need review")
    reviewers = {record.get("reviewer", "").strip() for record in records}
    if not reviewers or "" in reviewers:
        raise SystemExit(f"{bundle.name}: every record needs a named reviewer")
    accepted = [r for r in records if r["status"] == "accepted"]
    parents = {r["parent_id"] for r in accepted}
    return dict(bundle=bundle.name, counts=dict(counts), reviewers=sorted(reviewers),
                accepted=len(accepted), parents=len(parents))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", nargs="+", required=True)
    parser.add_argument("--out", default="dist/melon-cyclegan-bundles")
    args = parser.parse_args(argv)

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    report = []
    for source in args.bundle:
        source = Path(source)
        summary = check(source)
        if not summary["parents"]:
            raise SystemExit(f"{source.name}: no accepted candidate, the arm cannot be built")
        target = out / source.name
        (target / "images").mkdir(parents=True)
        (target / "masks").mkdir(parents=True)
        records = json.loads((source / "records.json").read_text())
        for record in records:
            for key in ("image", "mask"):
                shutil.copyfile(source / record[key], target / record[key])
        for name in KEEP:
            if (source / name).exists():
                shutil.copyfile(source / name, target / name)
        report.append(summary)
        print(summary)

    (out / "dataset-metadata.json").write_text(json.dumps(dict(
        title="Melon CycleGAN bundles", id="mardiantiandini/melon-cyclegan-bundles",
        licenses=[dict(name="unknown")]), indent=1) + "\n")
    (out / "review_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"packaged {len(report)} bundles -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
