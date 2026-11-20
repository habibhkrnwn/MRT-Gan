"""One-time evaluation of the locked pipeline on the test split.

    python harvest/test_once.py --lock results/PIPELINE_LOCK.json --dataset <regrouped v4> --out results/test.json

Verifies every hash in the lock, refuses to run when the output already exists, then runs the
five locked detector seeds on the 43 test photos and scores the calibrated rule, argmax and the
OMYON rule against the UPT labels. Nothing is refitted on test.
"""
import argparse
import csv
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import decision as D
import infer

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lock', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f'{out} exists; the test split is evaluated once')
    lock = json.loads(Path(args.lock).read_text())
    dataset = Path(args.dataset)
    bad = [n for n, h in lock['code_sha256'].items() if sha(HERE/n) != h]
    bad += [f'seed{s}' for s, v in lock['seeds'].items() if sha(v['weights']) != v['weights_sha256']]
    if sha(dataset/'manifest.csv') != lock['dataset_manifest_sha256']:
        bad.append('manifest')
    if bad:
        raise SystemExit(f'lock mismatch: {bad}')
    rows = [r for r in csv.DictReader(open(dataset/'manifest.csv')) if r['split'] == 'test']
    for r in rows:
        if sha(dataset/r['image']) != r['image_sha256'] or sha(dataset/r['label']) != r['label_sha256']:
            raise SystemExit(f'test file changed: {r["id"]}')
    gt = infer.groundtruth([(r['id'], dataset/r['image'], dataset/r['label']) for r in rows])
    ids = [r['id'] for r in rows]
    started = datetime.now(timezone.utc).isoformat()
    report = dict(started_at=started, lock_sha256=sha(args.lock), images=len(ids),
                  annotations=len(gt['annotations']), seeds={}, test_used=True,
                  note='single evaluation of the locked pipeline; no parameter was fitted on test')
    for seed, v in sorted(lock['seeds'].items(), key=lambda kv: int(kv[0])):
        model = infer.load_model(v['weights'])
        preds, seconds = infer.predict(model, [(r['id'], dataset/r['image']) for r in rows])
        p = v['decision']
        platt = D.Platt(p['platt_a'], p['platt_b'])
        rows_eval = D.collect(ids, gt, preds, p['tau_det'])
        seed_out = {rule: D.evaluate(rows_eval, platt, p['tau_low'], p['tau_high'], rule)
                    for rule in ('calibrated', 'argmax', 'omyon')}
        for m in seed_out.values():
            m['premature']['wilson95'] = D.wilson(m['premature']['k'], m['premature']['n'])
            m['missed_ripe']['wilson95'] = D.wilson(m['missed_ripe']['k'], m['missed_ripe']['n'])
        seed_out['seconds_per_image_median'] = statistics.median(seconds.values())
        seed_out['per_image'] = [dict(image_id=r['image_id'],
                                      gt=[g['category_id'] for g in r['gt']],
                                      fruits=[dict(p_ripe=round(platt(c['margin']), 6),
                                                   decision=D.decide(platt(c['margin']), p['tau_low'], p['tau_high']),
                                                   s_ripe=c['s_ripe'], s_unripe=c['s_unripe'], bbox=c['bbox'])
                                              for c in r['clusters']]) for r in rows_eval]
        report['seeds'][seed] = seed_out
        print(seed, seed_out['calibrated']['confusion'], seed_out['calibrated']['false_fruits'])
    report['finished_at'] = datetime.now(timezone.utc).isoformat()
    out.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
