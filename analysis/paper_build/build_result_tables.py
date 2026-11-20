"""Rebuild manuscript tables from the frozen final JSON; never trains or evaluates models."""
from pathlib import Path
import json
import hashlib
import shutil
import csv

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parents[1]
FINAL = ROOT / 'FINAL_Melon_2026-09-28'
SOURCES = []

def read(relative):
    path = FINAL / relative
    SOURCES.append(path)
    return json.loads(path.read_text())

def write(name, lines):
    (PROJECT / 'tables' / name).write_text('\n'.join(lines) + '\n')

diag = {a: read(f'03_Ablasi/A2_generator_G0-G3_p100/diagnostics/{a}.json') for a in ['G0','G1','G2','G3']}
selection = read('02_Generator_G3/selection.json')
# Post hoc unblinding (after the selection lock) with the private key from the notebook-13 Kaggle output.
key_path = ROOT / 'runs/05_texture_study/kaggle_v3_unblind/texture_study_v1/study/p100/review_key_PRIVATE.json'
review_path = FINAL / '02_Generator_G3/review_p100/review.csv'
SOURCES += [key_path, review_path]
if hashlib.sha256(review_path.read_bytes()).hexdigest() != selection['review_sha256']:
    raise ValueError('review.csv differs from the locked review')
review_rows = {r['blind_id']: r for r in csv.DictReader(review_path.open(newline=''))}
review_counts = {}
for k in json.loads(key_path.read_text()):
    a = review_counts.setdefault(k['arm'], {'accepted': 0, 'total': 0})
    a['total'] += 1
    a['accepted'] += review_rows[k['blind_id']]['accepted'] == 'yes'
for arm in ['G1', 'G3']:
    if (review_counts[arm]['accepted'], review_counts[arm]['total']) != (selection['scores'][arm]['accepted'], selection['scores'][arm]['total']):
        raise ValueError(f'{arm}: unblinded count differs from selection.json')
lines = [r'\begin{table*}[t]', r'\centering\footnotesize',
 r'\caption{Full-data generator diagnostics before candidate selection. Each generator produced 217 candidates; the real reference contains 81 eligible unripe crops. Accepted counts come from one blinded review of all 868 candidates; the recorded selection rule compared G1 and G3, and G0 and G2 were tabulated after the selection was locked.}',
 r'\label{tab:generator_results}', r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrr@{}}',r'\toprule',
 r'Variant & Nyquist $Y$ & Nyquist $R-Y$ & Luma energy & Chroma energy & LBP $\chi^2$ & Accepted \\',r'\midrule']
for name, d in [('Real unripe',diag['G3']['real_unripe'])]+[(a,diag[a]['synthetic']) for a in diag]:
    acc = review_counts.get(name)
    accepted = f"{acc['accepted']}/{acc['total']}" if acc else '--'
    vals = [d[k] for k in ['nyquist_luma','nyquist_r_minus_luma','luma_energy','chroma_energy']]
    lines.append(name+' & '+' & '.join(f'{x:.4f}' for x in vals)+f" & {d['lbp_chi2']:.3f} & {accepted}"+r' \\')
lines += [r'\bottomrule',r'\end{tabular*}',r'\end{table*}']
write('generator_results.tex',lines)

sub = read('03_Ablasi/B3_detektor_subset_25_50/results.json')
g3 = read('03_Ablasi/B2_detektor_train_penuh/G3_ranked_r100_results.json')
controls = read('03_Ablasi/B2_detektor_train_penuh/pembanding_C_results.json')
lines = [r'\begin{table*}[t]',r'\centering\footnotesize',
 r'\caption{Validation detector performance (mean $\pm$ population SD across seeds). Selected checkpoints maximize validation mAP$_{50:95}$; final checkpoints use update 3,000. AP$_U$ denotes unripe AP at IoU 0.50:0.05:0.95 for the selected checkpoint. Full-data controls are from a different session; comparisons are not external-test results.}',
 r'\label{tab:detector_results}',r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrr@{}}',r'\toprule',
 r'Training budget & Arm & Added & Selected mAP$_{50:95}$ & Final mAP$_{50:95}$ & AP$_U$ \\',r'\midrule']
fmt = lambda d: f"${d['map50_95']:.4f} \\pm {d['sd']:.4f}$"
for pct,n,k in [('25',75,33),('50',150,66)]:
    for j,(arm,name) in enumerate([('real_only','Real-only'),('oversample','Oversampling'),('geometry','Geometry'),('synthetic_G3','G3')]):
        s=sub['report'][pct]['selected']['summary'][arm];f=sub['report'][pct]['final']['summary'][arm]
        label=f'{pct}\\%: {n} images, 3 seeds' if j==0 else ''
        lines.append(f"{label} & {name} & {0 if j==0 else k} & {fmt(s)} & {fmt(f)} & {s['ap_unripe']:.4f}"+r' \\')
    lines.append(r'\midrule')
cs={d['arm']:d for d in controls['summary_selected']};cf={d['arm']:d for d in controls['summary_final']}
for j,(a,name) in enumerate([('real_only','Real-only'),('oversample_r100','Oversampling'),('cyclegan_r100','Earlier CycleGAN'),('g3','G3')]):
    s= g3['summary_selected'] if a=='g3' else cs[a]
    f= g3['summary_final'] if a=='g3' else cf[a]
    label='100\\%: 301 images, 5 seeds' if j==0 else ''
    lines.append(f"{label} & {name} & {0 if j==0 else 133} & {fmt(s)} & {fmt(f)} & {s['ap_unripe']:.4f}"+r' \\')
lines += [r'\bottomrule',r'\end{tabular*}',r'\end{table*}']
write('detector_results.tex',lines)

test=read('01_Pipeline_Final/results/test.json')
lines=[r'\begin{table}[t]',r'\centering\footnotesize',
 r'\caption{G3 decision outcomes on the reused 43-image test split. Each seed sees the same 31 ripe and 12 unripe fruit. PH: premature harvest; MR: missed ripe; Cov.: automatic-decision coverage; Acc.: selective accuracy. Seed 0 is the designated deployment model.}',
 r'\label{tab:decision_results}',r'\begin{tabular}{@{}lrrrrr@{}}',r'\toprule',r'Seed & PH & MR & Cov. & Acc. & ECE \\',r'\midrule']
for seed,record in test['seeds'].items():
    d=record['calibrated']
    lines.append(f"{seed} & {d['premature']['k']}/12 & {d['missed_ripe']['k']}/31 & {d['coverage']:.3f} & {d['selective_accuracy']:.3f} & {d['ece']:.4f}"+r' \\')
lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
write('decision_results.tex',lines)

(PROJECT/'figures').mkdir(exist_ok=True)
figure=FINAL/'03_Ablasi/A2_generator_G0-G3_p100/compare_ema_arms.png'
shutil.copyfile(figure,PROJECT/'figures/generator_comparison.png')
SOURCES.append(figure)
(PROJECT/'evidence/results_source_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES},indent=2)+'\n')
print('Built 3 result tables and copied the unmodified generator panel.')
