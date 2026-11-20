"""Write pgfplots data files for the manuscript figures from frozen artefacts.

Only reads JSON/CSV already produced by the experiments; never trains, evaluates, or opens test data
beyond the locked test.json summary already reported in Table VI. Each series is cross-checked
against the values used by build_result_tables.py so figures and tables cannot drift apart.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parents[1]
FINAL = ROOT / 'FINAL_Melon_2026-09-28'
OUT = PROJECT / 'figures' / 'data'
SOURCES = []


def read(path):
    path = Path(path)
    SOURCES.append(path)
    return json.loads(path.read_text())


def write(name, header, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    lines = [' '.join(header)] + [' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in r) for r in rows]
    (OUT / name).write_text('\n'.join(lines) + '\n')


def check_summary(values, mean, sd, label):
    if abs(statistics.fmean(values) - mean) > 1e-6 or abs(statistics.pstdev(values) - sd) > 1e-6:
        raise ValueError(f'{label}: per-seed values do not reproduce the tabulated summary')


def wilson(k, n, z=1.959963984540054):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


# 1. Per-seed detector scores (Fig. detector seeds) ------------------------------------------
sub = read(FINAL / '03_Ablasi/B3_detektor_subset_25_50/results.json')
SUBSET_ARMS = ['real_only', 'oversample', 'geometry', 'synthetic_G3']
for pct, folder in [('25', 'p025'), ('50', 'p050')]:
    res = read(FINAL / f'03_Ablasi/B3_detektor_subset_25_50/{folder}/results.json')
    rows = []
    for x, arm in enumerate(SUBSET_ARMS, start=1):
        sel = [res['selected'][f'{arm}@{s}']['map50_95'] for s in res['seeds']]
        fin = [res['results'][f'{arm}@{s}']['map50_95'] for s in res['seeds']]
        ref = sub['report'][pct]
        check_summary(sel, ref['selected']['summary'][arm]['map50_95'], ref['selected']['summary'][arm]['sd'], f'{pct}% {arm} selected')
        check_summary(fin, ref['final']['summary'][arm]['map50_95'], ref['final']['summary'][arm]['sd'], f'{pct}% {arm} final')
        rows += [(x, s, a, b) for s, a, b in zip(res['seeds'], sel, fin)]
    write(f'detector_seeds_{folder}.dat', ['x', 'seed', 'selected', 'final'], rows)

g3 = read(FINAL / '03_Ablasi/B2_detektor_train_penuh/G3_ranked_r100_results.json')
controls = read(FINAL / '03_Ablasi/B2_detektor_train_penuh/pembanding_C_results.json')
cs = {d['arm']: d for d in controls['summary_selected']}
cf = {d['arm']: d for d in controls['summary_final']}
rows = []
for x, arm in enumerate(['real_only', 'oversample_r100', 'cyclegan_r100', 'g3'], start=1):
    if arm == 'g3':
        src, key, s_ref, f_ref = g3, 'texture_G3_ranked_r100', g3['summary_selected'], g3['summary_final']
    else:
        src, key, s_ref, f_ref = controls, arm, cs[arm], cf[arm]
    seeds = [0, 1, 2, 3, 4]
    sel = [src['results_selected'][f'{key}@{s}']['map50_95'] for s in seeds]
    fin = [src['results_final'][f'{key}@{s}']['map50_95'] for s in seeds]
    check_summary(sel, s_ref['map50_95'], s_ref['sd'], f'100% {arm} selected')
    check_summary(fin, f_ref['map50_95'], f_ref['sd'], f'100% {arm} final')
    rows += [(x, s, a, b) for s, a, b in zip(seeds, sel, fin)]
write('detector_seeds_p100.dat', ['x', 'seed', 'selected', 'final'], rows)

# 2. Validation curves (mean across seeds per update) -----------------------------------------
def curves(paths_by_arm, name, selected_lookup):
    table = {}
    for arm, paths in paths_by_arm.items():
        hist = [read(p) for p in paths]
        for h, p in zip(hist, paths):
            best = max(e['map50_95'] for e in h)
            if abs(best - selected_lookup(arm, p)) > 1e-9:
                raise ValueError(f'{p}: history maximum differs from selected checkpoint')
        updates = [e['update'] for e in hist[0]]
        if any([e['update'] for e in h] != updates for h in hist):
            raise ValueError(f'{arm}: update grids differ')
        table[arm] = (updates, [statistics.fmean(h[i]['map50_95'] for h in hist) for i in range(len(updates))])
    updates = next(iter(table.values()))[0]
    write(name, ['update'] + list(table), [(u, *[table[a][1][i] for a in table]) for i, u in enumerate(updates)])

SUBSET_RUNS = ROOT / 'runs/09_texture_subset_detectors/kaggle_v1_output/texture_subset_detectors_v1/result'
for folder in ['p025', 'p050']:
    res = json.loads((FINAL / f'03_Ablasi/B3_detektor_subset_25_50/{folder}/results.json').read_text())
    paths = {arm: [SUBSET_RUNS / folder / f'{arm}__seed{s}' / 'valid_history.json' for s in res['seeds']] for arm in SUBSET_ARMS}
    lookup = lambda arm, p, res=res: res['selected'][f"{arm}@{int(p.parent.name[-1])}"]['map50_95']
    curves(paths, f'valid_curve_{folder}.dat', lookup)

FULL_RUNS = ROOT / 'runs/03_balancing_full_train/detector_7_arms/balance_detector'
G3_RUNS = ROOT / 'runs/06_texture_ranked_detector/kaggle_v2_output/texture_ranked_v1/result'
paths = {arm: sorted(FULL_RUNS.glob(f'group*/{arm}__seed[0-4]/valid_history.json')) for arm in ['real_only', 'oversample_r100', 'cyclegan_r100']}
paths['g3'] = sorted(G3_RUNS.glob('group*/texture_G3_ranked_r100__seed[0-4]/valid_history.json'))
if any(len(v) != 5 for v in paths.values()):
    raise ValueError('expected five full-data seeds per arm')
def full_lookup(arm, p):
    seed = int(p.parent.name[-1])
    if arm == 'g3':
        return g3['results_selected'][f'texture_G3_ranked_r100@{seed}']['map50_95']
    return controls['results_selected'][f'{arm}@{seed}']['map50_95']
curves(paths, 'valid_curve_p100.dat', full_lookup)

# 3. Generator training losses (full-data G0-G3, seed 11) -------------------------------------
CKPT = ROOT / 'runs/05_texture_study/kaggle_v3_output/texture_study_v1/study/p100/checkpoints'
hist = {g: read(CKPT / f'p100_{g}_seed11/loss.json')['history'] for g in ['G0', 'G1', 'G2', 'G3']}
iters = [e['iters'] for e in hist['G0']]
if any([e['iters'] for e in h] != iters for h in hist.values()):
    raise ValueError('generator logs use different iteration grids')
# Domain A = unripe (81 crops), B = ripe (217 crops); G_B translates ripe -> unripe and is judged by D_A.
keys = ['G_B', 'D_A', 'cycle_B', 'idt_B']
header = ['iters'] + [f'{k}_{g}' for k in keys for g in hist] + [f'texture_{k}_{g}' for k in ['G_B', 'D_A'] for g in ['G2', 'G3']]
rows = []
for i, it in enumerate(iters):
    row = [it] + [float(hist[g][i][k]) for k in keys for g in hist]
    row += [float(hist[g][i][f'texture_{k}']) for k in ['G_B', 'D_A'] for g in ['G2', 'G3']]
    rows.append(row)
write('generator_loss.dat', header, rows)

# 4. Review acceptance with Wilson intervals ---------------------------------------------------
selection = read(FINAL / '02_Generator_G3/selection.json')
def review_counts(path):
    SOURCES.append(Path(path))
    rows = list(csv.DictReader(open(path, newline='')))
    return sum(r['accepted'] == 'yes' for r in rows), len(rows)
bars = [('G1 (100\\%)', selection['scores']['G1']['accepted'], selection['scores']['G1']['total']),
        ('G3 (100\\%)', selection['scores']['G3']['accepted'], selection['scores']['G3']['total'])]
for pct, folder in [('50', 'review_p050_G3'), ('25', 'review_p025_G3')]:
    k, n = review_counts(FINAL / f'02_Generator_G3/{folder}/review.csv')
    bars.append((f'G3 ({pct}\\%)', k, n))
if (bars[2][1], bars[2][2], bars[3][1], bars[3][2]) != (106, 108, 54, 54):
    raise ValueError('subset review counts differ from the manuscript text')
rows = []
for y, (label, k, n) in enumerate(bars, start=1):
    lo, hi = wilson(k, n)
    rows.append((y, k, n, 100 * k / n, 100 * (k / n - lo), 100 * (hi - k / n), f'{k}/{n}'))
write('acceptance.dat', ['y', 'k', 'n', 'rate', 'err_lo', 'err_hi', 'label'], rows)
(OUT / 'acceptance_labels.tex').write_text('\\def\\AcceptanceLabels{' + ','.join('{' + b[0] + '}' for b in bars) + '}\n')

# 5. Decision confusion counts on the reused test split (already reported in Table VI) --------
test = read(FINAL / '01_Pipeline_Final/results/test.json')
cols = ['Panen', 'Cek manual', 'Belum', 'Tidak terdeteksi']
rows = []
for seed, record in test['seeds'].items():
    c = record['calibrated']['confusion']
    for t in ['ripe', 'unripe']:
        rows.append((seed, t, *[c[f'{t}->{col}'] for col in cols]))
write('decision_confusion.dat', ['seed', 'truth', 'harvest', 'manual', 'notyet', 'undetected'], rows)
# TikZ cells: \DecisionCell{row}{column}{count}{percent of the row's fruit}; rows run seed-major.
cells = []
for r, (seed, truth, *counts) in enumerate(rows):
    total = sum(counts)
    cells += [f'\\DecisionCell{{{r}}}{{{c}}}{{{k}}}{{{round(100 * k / total)}}}' for c, k in enumerate(counts)]
(OUT / 'decision_cells.tex').write_text('\n'.join(cells) + '\n')

(PROJECT / 'evidence/figure_source_hashes.json').write_text(json.dumps(
    {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in dict.fromkeys(SOURCES)}, indent=2) + '\n')
print('Wrote figure data to', OUT.relative_to(PROJECT))
