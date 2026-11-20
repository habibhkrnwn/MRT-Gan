"""Round-2 CPU check CB2 (reviewer minor 2): per-seed Nyquist ratios, MRT-GAN without vs with RGB texture discriminators.
Reads evidence/generator_r512_2026-10-03/generator_table.json (raw per-seed values); writes nyquist_seed_ratios.json."""
import hashlib, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EV = HERE.parents[1]
PKG = EV.parents[1]
SRC = EV / 'generator_r512_2026-10-03/generator_table.json'
d = json.load(open(SRC))['raw']
PAIRS = {11: ('G3TI_noT_R512', 'G3TI_R512'), 12: ('G3TI_noT_R512_s12', 'G3TI_R512_s12'), 13: ('G3TI_noT_R512_s13', 'G3TI_R512_s13')}
out = dict(input=dict(path=str(SRC.relative_to(PKG)), sha256=hashlib.sha256(SRC.read_bytes()).hexdigest()), per_seed={})
for m in ('nyq_y', 'nyq_ry'):
    rat = []
    for s, (wo, w) in PAIRS.items():
        r = d[wo][m] / d[w][m]; rat.append(r)
        out['per_seed'].setdefault(str(s), {})[m] = dict(without_texture_D=d[wo][m], with_texture_D=d[w][m], ratio_without_over_with=r)
    mw = sum(d[wo][m] for wo, _ in PAIRS.values()) / 3; mt = sum(d[w][m] for _, w in PAIRS.values()) / 3
    out[m] = dict(ratios=rat, min_ratio=min(rat), max_ratio=max(rat), higher_without_in_every_seed=all(r > 1 for r in rat),
                  ratio_of_means=mw / mt, with_over_without_of_means=mt / mw)
json.dump(out, open(HERE / 'nyquist_seed_ratios.json', 'w'), indent=1)
for m in ('nyq_y', 'nyq_ry'): print(m, [round(x, 2) for x in out[m]['ratios']], round(out[m]['ratio_of_means'], 2), round(out[m]['with_over_without_of_means'], 2))
