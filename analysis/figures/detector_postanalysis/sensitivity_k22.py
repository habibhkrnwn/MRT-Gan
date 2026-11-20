"""Near-duplicate sensitivity for the kernel-27 detectors: final-checkpoint mAP50:95 and unripe AP recomputed from the exported
predictions with and without the two validation photographs flagged as near-duplicates of training fruit (oke6396, oke7065).
Evaluator copied from evidence/sensitivity_2026-10-01/sens.py (Ultralytics matching rule). Validation only."""
import glob, json, os, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
src = (HERE.parent / 'sensitivity_2026-10-01/sens.py').read_text()
import torch
from ultralytics.utils.metrics import ap_per_class, box_iou
exec(src[src.index('def xyxy'):src.index("EX=('oke6396','oke7065')")])
EX = ('oke6396', 'oke7065')
RUN = HERE.parents[3] / 'runs/19_mrtgan_detector/mrtgan_detector_v1/result'
out = {}
for arm in ('real_only', 'oversample_k22', 'colour_transfer_k22', 'mrtgan_k22'):
    runs = sorted(glob.glob(str(RUN / f'group*/{arm}__seed*')))
    rows = []
    for r in runs:
        full, sub = evaluate(r), evaluate(r, EX); ref = json.load(open(r + '/validation.json'))['map50_95']
        rows.append(dict(run=os.path.basename(r), all87=full[0], excl85=sub[0], apU_all=full[1], apU_excl=sub[1], validation_json=ref))
    out[arm] = dict(n=len(rows), all87=float(np.mean([x['all87'] for x in rows])), excl85=float(np.mean([x['excl85'] for x in rows])),
                    max_abs_change=float(max(abs(x['all87'] - x['excl85']) for x in rows)),
                    apU_max_abs_change=float(max(abs(x['apU_all'] - x['apU_excl']) for x in rows)),
                    evaluator_vs_validation_json_max_diff=float(max(abs(x['all87'] - x['validation_json']) for x in rows)), per_seed=rows)
    print(arm, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in out[arm].items() if k != 'per_seed'})
order = lambda key: sorted(out, key=lambda a: -out[a][key])
out['order_all87'], out['order_excl85'] = order('all87'), order('excl85')
json.dump(out, open(HERE / 'sensitivity_k22.json', 'w'), indent=1); print(out['order_all87'], out['order_excl85'])
