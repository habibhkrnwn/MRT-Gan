"""Fit and measure the decision layer on validation predictions of the locked detector (5 seeds).

    python harvest/evaluate_validation.py --runs <kernel16 result dir> --manifest <dataset manifest.csv> \
        --out results/validation.json

Per seed: validation groups are split into a calibration half and a measurement half
(`split_groups`). tau_det, Platt calibration and (tau_low, tau_high) are chosen on the calibration
half and measured on the other half, next to the argmax and OMYON rules. A second fit on the full
validation set gives the parameters that are locked for the test run and for deployment.
Test images are not read.
"""
import argparse
import csv
import json
import statistics
from pathlib import Path

import decision as D


def load(run):
    return json.loads((run/'groundtruth.coco.json').read_text()), json.loads((run/'predictions.coco.json').read_text())


def fit(rows_all, tau_target):
    tau_det = D.choose_tau_det(rows_all)
    rows = [dict(r, clusters=[c for c in r['clusters'] if c['confidence'] >= tau_det]) for r in rows_all]
    rows = [dict(r, match=D.match(r['gt'], r['clusters'])) for r in rows]
    pairs = list(D.matched_pairs(rows))
    platt = D.Platt().fit([c['margin'] for c, _ in pairs], [int(y) for _, y in pairs])
    tau_low, tau_high = D.choose_thresholds([(platt(c['margin']), y) for c, y in pairs], tau_target)
    return dict(tau_det=tau_det, platt_a=platt.a, platt_b=platt.b, tau_low=tau_low, tau_high=tau_high,
                n_calibration_fruits=len(pairs)), platt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--target', type=float, default=0.98)
    args = ap.parse_args()
    groups = {r['id']: r['photograph_group'] for r in csv.DictReader(open(args.manifest)) if r['split'] == 'valid'}
    runs = sorted(Path(args.runs).glob('group*/*__seed*'), key=lambda p: int(p.name.split('seed')[-1]))
    report = dict(target_precision=args.target, split='valid', test_used=False, seeds={})
    for run in runs:
        seed = int(run.name.split('seed')[-1])
        gt, preds = load(run)
        classes = {}
        for a in gt['annotations']:
            classes.setdefault(a['image_id'], a['category_id'])
        ids = sorted(i['id'] for i in gt['images'])
        if set(ids) != set(groups):
            raise ValueError('validation images differ from the manifest')
        cal, meas = D.split_groups({i: groups[i] for i in ids}, classes)
        params, platt = fit(D.collect(cal, gt, preds, 0.0), args.target)
        rows_meas = D.collect(meas, gt, preds, params['tau_det'])
        seed_out = dict(calibration_images=len(cal), measurement_images=len(meas), half_fit=params,
                        measurement={rule: D.evaluate(rows_meas, platt, params['tau_low'], params['tau_high'], rule)
                                     for rule in ('calibrated', 'argmax', 'omyon')})
        full_params, full_platt = fit(D.collect(ids, gt, preds, 0.0), args.target)
        rows_full = D.collect(ids, gt, preds, full_params['tau_det'])
        seed_out['full_fit'] = full_params
        seed_out['full_validation_in_sample'] = D.evaluate(rows_full, full_platt, full_params['tau_low'],
                                                           full_params['tau_high'], 'calibrated')
        seed_out['weights'] = str(run/'weights_selected.pt')
        report['seeds'][seed] = seed_out
    summary = {}
    for rule in ('calibrated', 'argmax', 'omyon'):
        ms = [s['measurement'][rule] for s in report['seeds'].values()]
        summary[rule] = dict(
            premature=f"{sum(m['premature']['k'] for m in ms)}/{sum(m['premature']['n'] for m in ms)}",
            missed_ripe=f"{sum(m['missed_ripe']['k'] for m in ms)}/{sum(m['missed_ripe']['n'] for m in ms)}",
            coverage_mean=statistics.fmean(m['coverage'] for m in ms),
            selective_accuracy_mean=statistics.fmean(m['selective_accuracy'] for m in ms),
            manual_total=sum(m['manual'] for m in ms),
            false_fruits_total=sum(sum(m['false_fruits'].values()) for m in ms),
            ece_mean=statistics.fmean(m['ece'] for m in ms) if all(m['ece'] is not None for m in ms) else None)
    report['summary_measurement_all_seeds'] = summary
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(summary, indent=1))
    for seed, s in report['seeds'].items():
        print(seed, 'half', {k: round(v, 4) if isinstance(v, float) else v for k, v in s['half_fit'].items()},
              'full', {k: round(v, 4) if isinstance(v, float) else v for k, v in s['full_fit'].items()})


if __name__ == '__main__':
    main()
