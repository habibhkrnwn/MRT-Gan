"""Round-2 CPU check CB3a (reviewer minor 5): reproduce every field of bg_swap/bg_swap_paired.json from
bg_swap/bg_swap_results.json. The original paired test was run as an inline command on 21 Oct 2026 and was not saved as
a file; this script is its saved replacement. It also derives the per-row counts of Fig. 6 (reviewer minor 13) for the
three photographs shown, including colour transfer. Writes bg_swap_paired_reproduce.json beside this script.

Per-photograph statistic: for each unripe validation photograph, the fraction of the five seeds whose top-class
prediction on the swap image is correct; arm minus real-only; Wilcoxon signed-rank (scipy default, two-sided,
zero differences dropped)."""
import hashlib, json
from pathlib import Path
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
BG = HERE.parent / 'bg_swap'
PKG = HERE.parents[4]
RES, PAIRED = BG / 'bg_swap_results.json', BG / 'bg_swap_paired.json'
r = json.load(open(RES)); ref = json.load(open(PAIRED))
ARMS = ['real_only', 'oversample_k22', 'colour_transfer_k22', 'mrtgan_k22']


def det(arm, mode):
    return [{d['id']: d for d in r['runs'][f'{arm}@{s}']['details'][mode]} for s in range(5)]


unripe = sorted(d['id'] for d in r['runs']['real_only@0']['details']['swap'] if d['gt_class'] == 1)
assert len(unripe) == 25
frac = {a: {p: sum(s[p]['class_correct'] for s in det(a, 'swap')) / 5 for p in unripe} for a in ARMS}
per_seed = {a: [sum(s[p]['class_correct'] for p in unripe) for s in det(a, 'swap')] for a in ARMS}
out = dict(inputs={str(p.relative_to(PKG)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (RES, PAIRED)}, recomputed={})
ok = True
for a in ARMS[1:]:
    diff = [frac[a][p] - frac['real_only'][p] for p in unripe]
    rec = dict(mean_gain_unripe_photos=sum(diff) / 25, improved=sum(x > 0 for x in diff), worse=sum(x < 0 for x in diff),
               wilcoxon_p=float(wilcoxon(diff).pvalue), per_seed_unripe_correct=per_seed[a])
    out['recomputed'][a] = rec
    for k, v in rec.items():
        same = abs(v - ref[a][k]) < 1e-9 if isinstance(v, float) else v == ref[a][k]
        ok &= same
ok &= per_seed['real_only'] == ref['real_only_per_seed']
out['real_only_per_seed'] = per_seed['real_only']
out['all_fields_equal_bg_swap_paired_json'] = bool(ok)
# Fig. 6 rows: the three unripe photographs with the most real-only swap errors (ties by id), as in build_swap_figure.py
err = {p: 5 - round(5 * frac['real_only'][p]) for p in unripe}
pick = sorted(unripe, key=lambda p: (-err[p], p))[:3]
out['fig6_rows'] = [dict(row=i + 1, **{f'{a}_{m}_of5': sum(s[p]['class_correct'] for s in det(a, m)) for a in ARMS for m in ('control', 'swap')})
                    for i, p in enumerate(pick)]
json.dump(out, open(HERE / 'bg_swap_paired_reproduce.json', 'w'), indent=1)
print('equal:', ok)
for row in out['fig6_rows']: print(row)
