"""Run after both packs are filled: python analyze.py
Item 3: Cohen's kappa (accept + each criterion) between rater 1 (original session) and rater 2 on the 100 kappa items,
  bootstrap 95% CI (2000 resamples, seed 0), 2x2 table. Repeat consistency of rater 2 on the 6 duplicated images.
Item 4: acceptance of held-out vs train-derived MRT-GAN candidates per rater, Wilson 95% CI, Fisher exact test.
"""
import csv, json, math, random
from pathlib import Path
from scipy.stats import fisher_exact
OUT = Path(__file__).resolve().parent; PKG = OUT.parents[3]
R18 = PKG/'02_Hasil_Eksperimen/runs/18_review_mrtgan/review/review.csv'
CRIT = ['accepted', 'target_label', 'texture_realistic', 'shape_ok', 'boundary_ok', 'artifact_free']
def load(name):
    key = {k['blind_id']: k for k in json.load(open(OUT/f'{name}_key_PRIVATE.json'))}
    rev = {r['blind_id']: r for r in csv.DictReader(open(OUT/name/'review.csv'))}
    bad = [b for b in key if rev.get(b, {}).get('accepted') not in ('yes', 'no')]
    if bad: raise SystemExit(f'{name}: {len(bad)} items not filled')
    return key, rev
def kappa(a, b):
    n = len(a); cats = sorted(set(a) | set(b)); po = sum(x == y for x, y in zip(a, b))/n
    pe = sum((a.count(c)/n)*(b.count(c)/n) for c in cats)
    return 1.0 if pe == 1 else (po-pe)/(1-pe)
def boot(a, b, n=2000):
    rng = random.Random(0); idx = range(len(a)); v = []
    for _ in range(n):
        s = [rng.choice(idx) for _ in idx]; v.append(kappa([a[i] for i in s], [b[i] for i in s]))
    v.sort(); return v[int(.025*n)], v[int(.975*n)-1]
def wilson(k, n, z=1.96):
    p = k/n; d = 1+z*z/n; c = (p+z*z/(2*n))/d; h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return c-h, c+h
r1 = {r['blind_id']: r for r in csv.DictReader(open(R18))}
k2, v2 = load('pack_rater2'); res = dict(kappa={}, heldout={})
kap = [b for b, k in k2.items() if k['origin'] == 'kappa']
for c in CRIT:
    a = [r1[k2[b]['old_blind_id']][c] for b in kap]; bb = [v2[b][c] for b in kap]
    lo, hi = boot(a, bb); res['kappa'][c] = dict(kappa=kappa(a, bb), ci95=[lo, hi], agreement=sum(x == y for x, y in zip(a, bb))/len(a))
a = [r1[k2[b]['old_blind_id']]['accepted'] for b in kap]; bb = [v2[b]['accepted'] for b in kap]
res['kappa']['table_accept_r1_by_r2'] = {f'{x}/{y}': sum(p == x and q == y for p, q in zip(a, bb)) for x in ('yes', 'no') for y in ('yes', 'no')}
res['kappa']['by_arm_r2_accepted'] = {arm: sum(v2[b]['accepted'] == 'yes' for b in kap if k2[b]['arm'] == arm) for arm in sorted({k2[b]['arm'] for b in kap})}
dup = {}
for b, k in k2.items():
    if k['old_blind_id']: dup.setdefault(k['old_blind_id'], []).append(v2[b]['accepted'])
res['rater2_repeat_consistency'] = [v for v in dup.values() if len(v) == 2]
packs = {'rater2': (k2, v2)}
if all((OUT/'pack_heldout'/'review.csv').exists() for _ in [0]):
    try: packs['rater1'] = load('pack_heldout')
    except SystemExit as e: print(e)
for who, (k, v) in packs.items():
    t = {}
    for o in ('heldout_valid', 'train_derived'):
        ids = [b for b in k if k[b]['origin'] == o]; acc = sum(v[b]['accepted'] == 'yes' for b in ids)
        t[o] = dict(accepted=acc, n=len(ids), wilson95=wilson(acc, len(ids)))
    tab = [[t['heldout_valid']['accepted'], t['heldout_valid']['n']-t['heldout_valid']['accepted']],
           [t['train_derived']['accepted'], t['train_derived']['n']-t['train_derived']['accepted']]]
    t['fisher_p'] = fisher_exact(tab)[1]; res['heldout'][who] = t
(OUT/'kappa_heldout_results.json').write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
