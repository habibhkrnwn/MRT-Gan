"""Blind review of a CycleGAN bundle, reusing the benchmark review contract.

    python tools/review.py export --bundle runs/bundle_g4
    python tools/review.py tool   --bundle runs/bundle_g4       # optional HTML helper
    python tools/review.py ingest --bundle runs/bundle_g4 --reviewer "<name>"

The reviewer sees the candidate only: no method, no parent, no target class. A
candidate becomes `accepted` when the reviewer reads it as the target maturity
class and accepts the annotation. Nothing else may promote a candidate.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melon_cyclegan import config as cfg  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["export", "tool", "ingest"])
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--reviewer", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sheet", default=None)
    args = parser.parse_args(argv)

    cfg.benchmark_on_path()
    import pipeline  # noqa: E402

    if args.action == "export":
        print(pipeline.export_review(args.bundle, args.seed))
    elif args.action == "tool":
        script = cfg.BENCHMARK / "make_review_tool.py"
        subprocess.run([sys.executable, str(script), str(Path(args.bundle).resolve())], check=True)
    else:
        if not args.reviewer.strip():
            raise SystemExit("--reviewer is required; an unnamed review cannot accept candidates")
        print(pipeline.ingest_review(args.bundle, args.reviewer, args.sheet))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
