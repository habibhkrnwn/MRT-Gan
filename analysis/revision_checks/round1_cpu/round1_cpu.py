"""Revision round 1, CPU-only re-analyses of existing result files (23 Oct 2026). No training, inference or ratings.

Usage: python round1_cpu.py            (runs every analysis; each writes <name>.json beside this script)

  swap_seed_level      CA1+CA2: background-swap inference at seed level (sign test, paired t CI, exact unpaired
                       permutation test, two-way cluster bootstrap) next to the per-photograph Wilcoxon test that was
                       fixed in PROTOCOL.md; localized and per-class counts for original, control and swap.
  same_period_agreement CA4: assessor 1 (session 2) vs assessor 2 on the 124 items both rated in the same period;
                       joint acceptance by origin.
  agreement_prevalence CA5: PABAK and Gwet's AC1 on the 100 kappa items (all and farmer-rated rows only);
                       inverse-probability-weighted (arm x session-1 decision strata) agreement and kappa.
  training_sources     CA7: file-name family of unripe training photographs and of each arm's additions.
  repeat_positions     CA8: display positions of the six repeated images in assessor 2's pack.
  table_trace          CA9: recompute Table 4 cells from the kernel-27 results.json and compare with the .tex.

The list of statistics in swap_seed_level was fixed before the script was first run; all are reported.
"""
import csv, hashlib, itertools, json, math, random, re
from pathlib import Path
import numpy as np
from scipy.stats import binomtest, t as tdist, wilcoxon

HERE = Path(__file__).resolve().parent
RV = HERE.parent
DRAFT = RV.parents[1]
PKG = DRAFT.parent
R18 = PKG / '02_Hasil_Eksperimen/runs/18_review_mrtgan'
DET = RV.parent / 'detector_mrtgan_2026-10-04'
ARMS = ['real_only', 'oversample_k22', 'colour_transfer_k22', 'mrtgan_k22']
CRIT = ['accepted', 'target_label', 'texture_realistic', 'shape_ok', 'boundary_ok', 'artifact_free']


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p): return str(Path(p).resolve().relative_to(PKG))
def rows(p): return list(csv.DictReader(open(p, newline='')))
def save(name, obj, inputs):
    obj = dict(inputs={rel(p): sha(p) for p in inputs}, **obj)
    (HERE / f'{name}.json').write_text(json.dumps(obj, indent=1))
    with open(HERE / 'round1_cpu.log', 'a') as fh: fh.write(f'{name}: written\n')


# ------------------------------------------------------------------------------------------------- agreement helpers
def kappa(a, b):  # identical to kappa/analyze.py
    n = len(a); cats = sorted(set(a) | set(b)); po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def boot(a, b, n=2000):  # identical to kappa/analyze.py
    rng = random.Random(0); idx = range(len(a)); v = []
    for _ in range(n):
        s = [rng.choice(idx) for _ in idx]; v.append(kappa([a[i] for i in s], [b[i] for i in s]))
    v.sort(); return [v[int(.025 * n)], v[int(.975 * n) - 1]]


def ac1(a, b):
    n = len(a); cats = sorted(set(a) | set(b)); q = len(cats)
    if q < 2: return 1.0
    po = sum(x == y for x, y in zip(a, b)) / n
    pi = [(a.count(c) + b.count(c)) / (2 * n) for c in cats]
    pe = sum(p * (1 - p) for p in pi) / (q - 1)
    return (po - pe) / (1 - pe)


def pabak(a, b, q=2):
    po = sum(x == y for x, y in zip(a, b)) / len(a); return (q * po - 1) / (q - 1)


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return [c - h, c + h]


def agree(a, b):
    q = 3 if any(x == 'ambiguous' for x in a + b) or any(x == 'ripe' for x in a + b) else 2
    return dict(n=len(a), agreement=sum(x == y for x, y in zip(a, b)) / len(a), kappa=kappa(a, b), kappa_ci95=boot(a, b),
                pabak=pabak(a, b, q), ac1=ac1(a, b),
                table_first_by_second={f'{x}/{y}': sum(p == x and q_ == y for p, q_ in zip(a, b))
                                       for x in sorted(set(a) | set(b)) for y in sorted(set(a) | set(b))})


# ------------------------------------------------------------------------------------------------------------- CA1/2
def swap_seed_level():
    src = RV / 'bg_swap/bg_swap_results.json'; paired = RV / 'bg_swap/bg_swap_paired.json'
    d = json.load(open(src))['runs']
    seeds = range(5)
    # per-photo unripe correctness matrix [arm][seed][photo]
    ids = sorted(x['id'] for x in d['real_only@0']['details']['swap'] if x['gt_class'] == 1)
    M = {}
    for a in ARMS:
        M[a] = np.array([[next(x['class_correct'] for x in d[f'{a}@{s}']['details']['swap'] if x['id'] == i) for i in ids]
                         for s in seeds], dtype=float)
    per_seed = {a: [int(v) for v in M[a].sum(1)] for a in ARMS}
    # localized and per-class counts
    loc = {}
    for a in ARMS:
        loc[a] = {}
        for cond in ('original', 'control', 'swap'):
            pc = {c: {k: [d[f'{a}@{s}']['summary'][cond]['per_class'][c][k] for s in seeds] for k in ('n', 'top_correct', 'localized_correct')}
                  for c in ('0', '1')}
            loc[a][cond] = {('ripe' if c == '0' else 'unripe'): dict(n=v['n'][0], top_correct=v['top_correct'], localized_correct=v['localized_correct'],
                                                                     mean_top=float(np.mean(v['top_correct'])), mean_localized=float(np.mean(v['localized_correct'])))
                            for c, v in pc.items()}
    rng = np.random.default_rng(0); B = 10000

    def compare(a, b):  # a minus b
        x = np.array(per_seed[a], float); y = np.array(per_seed[b], float); dlt = x - y
        nz = dlt[dlt != 0]; pos = int((nz > 0).sum())
        sign_p = binomtest(pos, len(nz), 0.5).pvalue if len(nz) else 1.0
        m = dlt.mean(); se = dlt.std(ddof=1) / math.sqrt(5); tq = tdist.ppf(.975, 4)
        pooled = np.concatenate([x, y]); obs = x.mean() - y.mean(); cnt = 0; tot = 0
        for comb in itertools.combinations(range(10), 5):
            g = np.zeros(10, bool); g[list(comb)] = True
            diff = pooled[g].mean() - pooled[~g].mean(); tot += 1; cnt += abs(diff) >= abs(obs) - 1e-12
        # two-way cluster bootstrap: seeds resampled independently per arm, photographs resampled jointly
        bs = np.empty(B)
        for i in range(B):
            p = rng.integers(0, 25, 25); sa = rng.integers(0, 5, 5); sb = rng.integers(0, 5, 5)
            bs[i] = M[a][np.ix_(sa, p)].sum(1).mean() - M[b][np.ix_(sb, p)].sum(1).mean()
        fa = M[a].mean(0); fb = M[b].mean(0)
        w = wilcoxon(fa, fb, zero_method='wilcox', alternative='two-sided') if np.any(fa != fb) else None
        return dict(per_seed_a=per_seed[a], per_seed_b=per_seed[b], per_seed_difference=[int(v) for v in dlt],
                    mean_difference=float(m), seeds_positive=int((dlt > 0).sum()), seeds_negative=int((dlt < 0).sum()),
                    seeds_tied=int((dlt == 0).sum()), sign_test_two_sided_p=float(sign_p),
                    paired_t_ci95=[float(m - tq * se), float(m + tq * se)] if se > 0 else [float(m), float(m)],
                    unpaired_exact_permutation_p=cnt / tot, permutation_splits=tot,
                    cluster_bootstrap_ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                    cluster_bootstrap_resamples=B,
                    photos_better=int((fa > fb).sum()), photos_worse=int((fa < fb).sum()),
                    photo_wilcoxon_p=float(w.pvalue) if w is not None else 1.0)
    comps = {f'{a}_vs_real_only': compare(a, 'real_only') for a in ARMS[1:]}
    comps['mrtgan_k22_vs_colour_transfer_k22'] = compare('mrtgan_k22', 'colour_transfer_k22')
    comps['mrtgan_k22_vs_oversample_k22'] = compare('mrtgan_k22', 'oversample_k22')
    comps['colour_transfer_k22_vs_oversample_k22'] = compare('colour_transfer_k22', 'oversample_k22')
    ref = json.load(open(paired))
    check = all(per_seed[a] == ref[a]['per_seed_unripe_correct'] for a in ARMS[1:]) and per_seed['real_only'] == ref['real_only_per_seed'] \
        and all(abs(comps[f'{a}_vs_real_only']['photo_wilcoxon_p'] - ref[a]['wilcoxon_p']) < 1e-9 for a in ARMS[1:])
    save('swap_seed_level', dict(unit_note='seed-level tests pair arms by seed index (same seed number, same kernel session); '
                                           'photo-level Wilcoxon on fraction of seeds correct as fixed in PROTOCOL.md 2C',
                                 per_seed_unripe_swap_correct=per_seed, comparisons=comps, counts=loc,
                                 matches_bg_swap_paired_json=bool(check)), [src, paired])
    return per_seed, comps, loc


# ---------------------------------------------------------------------------------------------------------------- CA4
def same_period_agreement():
    k1f = RV / 'kappa/pack_heldout_key_PRIVATE.json'; v1f = RV / 'kappa/pack_heldout/review.csv'
    k2f = RV / 'kappa/pack_rater2_key_PRIVATE.json'; v2f = RV / 'kappa/pack_rater2/review.csv'
    k1 = json.load(open(k1f)); v1 = {r['blind_id']: r for r in rows(v1f)}
    k2 = json.load(open(k2f)); v2 = {r['blind_id']: r for r in rows(v2f)}
    m2 = {}
    for k in k2:
        if k['origin'] in ('heldout_valid', 'train_derived'): m2[(k['origin'], k['image_sha256'])] = k
    pairs = [(k, m2[(k['origin'], k['image_sha256'])]) for k in k1]
    assert len(pairs) == 124
    res = {}
    for name, sel in (('all_124', lambda k: True), ('heldout_valid', lambda k: k['origin'] == 'heldout_valid'),
                      ('train_derived', lambda k: k['origin'] == 'train_derived')):
        s = [(a, b) for a, b in pairs if sel(a)]
        d = {c: agree([v1[a['blind_id']][c] for a, _ in s], [v2[b['blind_id']][c] for _, b in s]) for c in CRIT}
        both = sum(v1[a['blind_id']]['accepted'] == 'yes' and v2[b['blind_id']]['accepted'] == 'yes' for a, b in s)
        d['joint_accepted'] = dict(k=both, n=len(s), wilson95=wilson(both, len(s)))
        d['assessor1_accepted'] = sum(v1[a['blind_id']]['accepted'] == 'yes' for a, _ in s)
        d['assessor2_accepted'] = sum(v2[b['blind_id']]['accepted'] == 'yes' for _, b in s)
        d['both_unripe'] = sum(v1[a['blind_id']]['target_label'] == 'unripe' and v2[b['blind_id']]['target_label'] == 'unripe' for a, b in s)
        res[name] = d
    save('same_period_agreement', dict(match_rule='image_sha256 within origin', results=res), [k1f, v1f, k2f, v2f])
    return res


# ---------------------------------------------------------------------------------------------------------------- CA5
def agreement_prevalence():
    rf = R18 / 'review/review.csv'; kf = R18 / 'review_key_PRIVATE.json'
    k2f = RV / 'kappa/pack_rater2_key_PRIVATE.json'; v2f = RV / 'kappa/pack_rater2/review.csv'
    r1 = {r['blind_id']: r for r in rows(rf)}; key = {k['blind_id']: k for k in json.load(open(kf))}
    k2 = json.load(open(k2f)); v2 = {r['blind_id']: r for r in rows(v2f)}
    kap = [k for k in k2 if k['origin'] == 'kappa']
    out = {}
    for name, sel in (('all_100', lambda k: True), ('farmer_rows', lambda k: r1[k['old_blind_id']]['reviewer'] == 'petani_nilai')):
        s = [k for k in kap if sel(k)]
        out[name] = {c: agree([r1[k['old_blind_id']][c] for k in s], [v2[k['blind_id']][c] for k in s]) for c in CRIT}
    # inverse-probability weights: stratum = arm x session-1 decision (sampling design of kappa/build_packs.py)
    strat = lambda arm, dec: f'{arm}|{dec}'
    pop = {}
    for b, r in r1.items(): pop[strat(key[b]['arm'], r['accepted'])] = pop.get(strat(key[b]['arm'], r['accepted']), 0) + 1
    smp = {}
    for k in kap: smp[strat(k['arm'], r1[k['old_blind_id']]['accepted'])] = smp.get(strat(k['arm'], r1[k['old_blind_id']]['accepted']), 0) + 1
    def weighted(items):
        w = np.array([pop[strat(k['arm'], r1[k['old_blind_id']]['accepted'])] / smp[strat(k['arm'], r1[k['old_blind_id']]['accepted'])] for k in items])
        a = np.array([r1[k['old_blind_id']]['accepted'] == 'yes' for k in items]); b = np.array([v2[k['blind_id']]['accepted'] == 'yes' for k in items])
        W = w.sum(); po = (w * (a == b)).sum() / W; pa = (w * a).sum() / W; pb = (w * b).sum() / W
        pe = pa * pb + (1 - pa) * (1 - pb); return float(po), float((po - pe) / (1 - pe)), float(pa), float(pb)
    po, kw, pa, pb = weighted(kap)
    rng = random.Random(0); groups = {}
    for k in kap: groups.setdefault(strat(k['arm'], r1[k['old_blind_id']]['accepted']), []).append(k)
    bs = []
    for _ in range(2000):
        s = [rng.choice(g) for g in groups.values() for _ in g]; bs.append(weighted(s)[1])
    bs.sort()
    per_arm = {}
    for arm in sorted({k['arm'] for k in kap}):
        it = [k for k in kap if k['arm'] == arm]
        w = [pop[strat(arm, r1[k['old_blind_id']]['accepted'])] / smp[strat(arm, r1[k['old_blind_id']]['accepted'])] for k in it]
        per_arm[arm] = dict(n=len(it), assessor1_session1_accepted=sum(r1[k['old_blind_id']]['accepted'] == 'yes' for k in it),
                            assessor2_accepted=sum(v2[k['blind_id']]['accepted'] == 'yes' for k in it),
                            both_accepted=sum(r1[k['old_blind_id']]['accepted'] == 'yes' and v2[k['blind_id']]['accepted'] == 'yes' for k in it),
                            neither_accepted=sum(r1[k['old_blind_id']]['accepted'] != 'yes' and v2[k['blind_id']]['accepted'] != 'yes' for k in it),
                            assessor2_weighted_population_acceptance=sum(wi * (v2[k['blind_id']]['accepted'] == 'yes') for wi, k in zip(w, it)) / sum(w))
    out['weighted_population'] = dict(strata_population=pop, strata_sample=smp, agreement=po, kappa=kw,
                                      kappa_ci95_stratified_bootstrap=[bs[50], bs[1949]],
                                      assessor1_acceptance=pa, assessor2_acceptance=pb,
                                      note='strata use session-1 decisions, which include the 148 vision-language-model rows')
    out['per_arm'] = per_arm
    save('agreement_prevalence', out, [rf, kf, k2f, v2f, RV / 'kappa/build_packs.py'])
    return out


# ---------------------------------------------------------------------------------------------------------------- CA7
def training_sources():
    fam = lambda s: 'oke' if s.startswith('oke') else ('DSC' if s.startswith('DSC') else 'other')
    files = {a: DET / f'arm_manifest_{a.replace("_k22", "")}_k22.json' if a != 'real_only' else DET / 'arm_manifest_real_only.json' for a in ARMS}
    files['oversample_k22'] = DET / 'arm_manifest_oversample_k22.json'
    out = {}
    for a, f in files.items():
        m = json.load(open(f)); real_u = [x['id'] for x in m['real'] if x['class_id'] == 1]; real_r = [x['id'] for x in m['real'] if x['class_id'] == 0]
        out[a] = dict(real_unripe_by_family={k: sum(fam(i) == k for i in real_u) for k in ('oke', 'DSC', 'other')},
                      real_ripe_by_family={k: sum(fam(i) == k for i in real_r) for k in ('oke', 'DSC', 'other')},
                      added_source_by_family={k: sum(fam(x['source']) == k for x in m['added']) for k in ('oke', 'DSC', 'other')},
                      added=len(m['added']), counts=m['counts'])
    save('training_sources', out, list(files.values()))
    return out


# ---------------------------------------------------------------------------------------------------------------- CA8
def repeat_positions():
    kf = RV / 'kappa/pack_rater2_key_PRIVATE.json'; hf = RV / 'kappa/pack_rater2/review_tool.html'; cf = RV / 'kappa/pack_rater2/review.csv'
    k2 = json.load(open(kf)); html = hf.read_text()
    items = json.loads(re.search(r'const ITEMS=(\[[^\]]*\])', html).group(1))
    csv_order = [r['blind_id'] for r in rows(cf)]
    pos = {b: i + 1 for i, b in enumerate(items)}; byold = {}
    for k in k2:
        if k['old_blind_id']: byold.setdefault(k['old_blind_id'], []).append(pos[k['blind_id']])
    pairs = sorted(sorted(v) for v in byold.values() if len(v) == 2)
    save('repeat_positions', dict(n_items=len(items), display_order_equals_csv_order=items == csv_order,
                                  repeat_positions=pairs, gaps=[b - a for a, b in pairs],
                                  note='display order of review_tool.html; the order of the paper forms is not recorded'), [kf, hf, cf])
    return pairs


# ---------------------------------------------------------------------------------------------------------------- CA9
def table_trace():
    rf = DET / 'results.json'; rf2 = PKG / '02_Hasil_Eksperimen/runs/19_mrtgan_detector/mrtgan_detector_v1/result/results.json'
    tex = DRAFT / 'tables/detector_results.tex'
    d = json.load(open(rf)); out = dict(results_sha256=sha(rf), run_output_sha256=sha(rf2) if rf2.exists() else None,
                                        results_sha256_equal=(sha(rf) == sha(rf2)) if rf2.exists() else None)
    names = dict(real_only='Real-only', oversample_k22='Oversampling', colour_transfer_k22='Colour transfer', mrtgan_k22='MRT-GAN')
    texrows = {l.split('&')[0].strip(): l for l in tex.read_text().splitlines() if '&' in l}
    cells, mism = {}, []
    for a, nm in names.items():
        S, F = d['summary_selected'][a], d['summary_final'][a]
        sd = lambda v: float(np.std(v))
        rec = dict(sel=f"{np.mean(S['per_seed']):.4f} \\pm {sd(S['per_seed']):.4f}", fin=f"{np.mean(F['per_seed']):.4f} \\pm {sd(F['per_seed']):.4f}",
                   ap_u=f"{S['ap_unripe']:.4f}")
        if a != 'real_only':
            ds = np.array(S['per_seed']) - np.array(d['summary_selected']['real_only']['per_seed'])
            df = np.array(F['per_seed']) - np.array(d['summary_final']['real_only']['per_seed'])
            rec['d_sel'] = f"{ds.mean():.4f} ({int((ds > 0).sum())}/5)"
            rec['d_fin'] = f"{df.mean():.4f} ({int((df > 0).sum())}/5)"
        line = texrows.get(nm, '').replace(' ', '').replace('$', '')
        rec['all_cells_found_in_tex'] = all(v.replace(' ', '').replace('$', '') in line for k, v in rec.items() if k != 'all_cells_found_in_tex')
        if not rec['all_cells_found_in_tex']: mism.append(nm)
        cells[nm] = rec
    out.update(cells=cells, mismatches=mism, sd_type='population SD over seeds 0-4')
    save('table_trace', out, [rf, tex] + ([rf2] if rf2.exists() else []))
    return out


if __name__ == '__main__':
    (HERE / 'round1_cpu.log').write_text('')
    ps, comps, loc = swap_seed_level()
    print('per seed', ps)
    for k, v in comps.items():
        print(k, v['per_seed_difference'], 'sign p', round(v['sign_test_two_sided_p'], 4), 'tCI', [round(x, 2) for x in v['paired_t_ci95']],
              'perm p', round(v['unpaired_exact_permutation_p'], 4), 'boot', [round(x, 2) for x in v['cluster_bootstrap_ci95']],
              'photo', v['photos_better'], v['photos_worse'], round(v['photo_wilcoxon_p'], 5))
    for a in ARMS: print(a, {c: {cl: (x['mean_top'], x['mean_localized']) for cl, x in loc[a][c].items()} for c in loc[a]})
    sp = same_period_agreement()
    for n, d in sp.items(): print('same-period', n, {c: (round(d[c]['agreement'], 3), round(d[c]['kappa'], 3), round(d[c]['pabak'], 3), round(d[c]['ac1'], 3)) for c in CRIT}, d['joint_accepted'], d['assessor1_accepted'], d['assessor2_accepted'], d['both_unripe'])
    ap = agreement_prevalence()
    for n in ('all_100', 'farmer_rows'): print('prev', n, {c: (round(ap[n][c]['kappa'], 3), round(ap[n][c]['pabak'], 3), round(ap[n][c]['ac1'], 3)) for c in CRIT})
    print('weighted', {k: v for k, v in ap['weighted_population'].items() if 'strata' not in k}); print(ap['per_arm'])
    print('sources', json.dumps(training_sources()))
    print('repeats', repeat_positions())
    tt = table_trace(); print('trace', tt['results_sha256_equal'], tt['mismatches'], json.dumps(tt['cells']))
