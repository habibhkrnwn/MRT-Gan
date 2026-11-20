"""One blind review across several bundles, then per-bundle ingest.

    python tools/merge_review.py build --bundles runs/x/bundle_a runs/x/bundle_b \
        --out runs/x/review_merged
    # fill runs/x/review_merged/review/review_tool.html, save the CSV, then:
    python tools/merge_review.py split --merged runs/x/review_merged \
        --sheet "runs/x/review_merged/review/<filled>.csv" --reviewer "<name>"

When bundles differ only in the condition under test (raw vs EMA weights, for
example) the folder name tells the reviewer which condition an image belongs to.
Merging shuffles every candidate of every bundle into one anonymous sheet, so the
reviewer sees neither method, parent, target class nor bundle.

`split` rewrites the filled rows into each bundle's own review ids and calls the
unchanged `pipeline.ingest_review`, so acceptance follows exactly the same rule
as a single-bundle review. Every bundle must still have its own exported
`review/key.json` (`tools/review.py export`).
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melon_cyclegan import config as cfg  # noqa: E402

FIELDS = ["review_id", "image", "label_class", "annotation_ok", "notes"]
SEED = 7


def build(bundles, out, seed=SEED):
    cfg.benchmark_on_path()
    import pipeline

    out = Path(out)
    if out.exists():
        raise ValueError(f"{out} exists; use a new directory so an earlier review is never overwritten")
    (out / "images").mkdir(parents=True)
    (out / "masks").mkdir()
    merged, sources = [], {}
    for index, bundle in enumerate(Path(b) for b in bundles):
        for record in json.loads((bundle / "records.json").read_text()):
            if record.get("status") != "needs_review":
                raise ValueError(f"{bundle.name}: {record['candidate_id']} is already {record.get('status')}")
            merged_id = f"b{index}_{record['candidate_id']}"
            image, mask = f"images/{merged_id}.png", f"masks/{merged_id}.png"
            shutil.copyfile(bundle / record["image"], out / image)
            shutil.copyfile(bundle / record["mask"], out / mask)
            merged.append(dict(candidate_id=merged_id, image=image, mask=mask, bbox=record["bbox"],
                               class_id=record["class_id"], status="needs_review"))
            sources[merged_id] = dict(bundle=str(bundle.resolve()), candidate_id=record["candidate_id"])
    (out / "records.json").write_text(json.dumps(merged, indent=2) + "\n")
    (out / "sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    report = pipeline.export_review(out, seed)
    subprocess.run([sys.executable, str(cfg.BENCHMARK / "make_review_tool.py"), str(out.resolve())], check=True)
    return report


def _read(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def split(merged, sheet, reviewer):
    cfg.benchmark_on_path()
    import pipeline

    merged = Path(merged)
    if not reviewer.strip():
        raise SystemExit("--reviewer is required; an unnamed review cannot accept candidates")
    rows = _read(sheet)
    key = json.loads((merged / "review" / "key.json").read_text())
    sources = json.loads((merged / "sources.json").read_text())
    if {r["review_id"] for r in rows} != set(key):
        raise ValueError("filled sheet does not match the merged key")
    per_bundle = {}
    for row in rows:
        source = sources[key[row["review_id"]]]
        per_bundle.setdefault(source["bundle"], {})[source["candidate_id"]] = row
    results = {}
    for bundle, by_candidate in per_bundle.items():
        bundle = Path(bundle)
        own_key = json.loads((bundle / "review" / "key.json").read_text())
        if set(own_key.values()) != set(by_candidate):
            raise ValueError(f"{bundle.name}: merged rows do not cover its candidates exactly")
        own_rows = []
        for review_id, candidate_id in sorted(own_key.items()):
            filled = by_candidate[candidate_id]
            own_rows.append(dict(review_id=review_id, image=f"blind/{review_id}.png",
                                 label_class=filled["label_class"], annotation_ok=filled["annotation_ok"],
                                 notes=filled.get("notes", "")))
        own_sheet = bundle / "review" / "review_sheet_from_merged.csv"
        with open(own_sheet, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(own_rows)
        shutil.copyfile(bundle / "records.json", bundle / "records.pre_review.json")
        results[bundle.name] = pipeline.ingest_review(bundle, reviewer, own_sheet)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    b = sub.add_parser("build")
    b.add_argument("--bundles", nargs="+", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--seed", type=int, default=SEED)
    s = sub.add_parser("split")
    s.add_argument("--merged", required=True)
    s.add_argument("--sheet", required=True)
    s.add_argument("--reviewer", required=True)
    args = parser.parse_args(argv)
    if args.action == "build":
        print(build(args.bundles, args.out, args.seed))
    else:
        print(json.dumps(split(args.merged, args.sheet, args.reviewer), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
