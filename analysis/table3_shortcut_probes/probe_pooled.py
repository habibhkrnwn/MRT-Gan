"""Round-2 CPU check (reviewer minor 8): synthetic consistency probe pooled over the full candidate pools (217 per
generator), from the per-group shares in evidence/detector_mrtgan_2026-10-04/derived_stats.json (shares are stored to
three decimals; pooled values are weighted by group size). Writes probe_pooled.json beside this script."""
import hashlib, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / 'detector_mrtgan_2026-10-04/derived_stats.json'
PKG = HERE.parents[4]
probe = json.load(open(SRC))['probe']
GROUPS = {'MRTGAN_s12': ['MRTGAN_s12_used', 'MRTGAN_s12_rejected'],
          'colour_transfer': ['colour_transfer_used', 'colour_transfer_accepted_unused', 'colour_transfer_rejected']}
out = dict(input=dict(path=str(SRC.relative_to(PKG)), sha256=hashlib.sha256(SRC.read_bytes()).hexdigest()), pooled={})
for det, g in probe.items():
    out['pooled'][det] = {}
    for pool, keys in GROUPS.items():
        n = sum(g[k]['n_per_seed'] for k in keys)
        out['pooled'][det][pool] = dict(n=n, share_unripe=sum(g[k]['n_per_seed'] * g[k]['share'] for k in keys) / n)
json.dump(out, open(HERE / 'probe_pooled.json', 'w'), indent=1)
for det, v in out['pooled'].items(): print(det, {k: round(x['share_unripe'], 3) for k, x in v.items()})
