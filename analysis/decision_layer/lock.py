"""Freeze the harvest-recommendation pipeline before the test split is opened.

    python harvest/lock.py --runs <kernel16 result dir> --validation results/validation.json \
        --manifest <dataset manifest.csv> --out results/PIPELINE_LOCK.json

Locks: detector arm (real + 133 best-ranked reviewed G3 unripe), weights of seeds 0-4 with
sha256, deployment seed 0 (pre-specified, not chosen by any score), the per-seed decision
parameters fitted on the full validation split, the decision-rule constants, the sha256 of every
harvest/*.py file and of the dataset manifest. Refuses to overwrite an existing lock.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import decision as D

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--validation', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f'{out} exists; a lock is never rewritten')
    validation = json.loads(Path(args.validation).read_text())
    seeds = {}
    for seed, s in sorted(validation['seeds'].items(), key=lambda kv: int(kv[0])):
        weights = Path(s['weights'])
        seeds[seed] = dict(weights=str(weights.resolve()), weights_sha256=sha(weights), decision=s['full_fit'])
    import ultralytics
    lock = dict(
        created_at=datetime.now(timezone.utc).isoformat(),
        detector=dict(arm='mrtgan_k22', architecture='yolov8n', train='301 real + 22 MRT-GAN (512 px) synthetic unripe, accepted in blinded review v2, best-ranked; kernel 27',
                      updates=3000, checkpoint='best validation mAP50:95', runs_dir=str(Path(args.runs).resolve())),
        deployment_seed='0',
        seeds=seeds,
        rule=dict(iou_merge=0.5, match_iou=0.5, precision_target=validation['target_precision'],
                  floor_high=0.90, ceil_low=0.10, calibration='Platt on ln(s_ripe)-ln(s_unripe)',
                  tau_det='median of F1-maximising grid points', inference=dict(imgsz=640, conf=0.001, max_det=100)),
        validation_json_sha256=sha(args.validation),
        dataset_manifest_sha256=sha(args.manifest),
        code_sha256={p.name: sha(p) for p in sorted(HERE.glob('*.py'))},
        ultralytics_local=ultralytics.__version__,
        test_used=False)
    out.write_text(json.dumps(lock, indent=2) + '\n')
    print(out, {k: v['decision'] for k, v in seeds.items()}['0'])


if __name__ == '__main__':
    main()
