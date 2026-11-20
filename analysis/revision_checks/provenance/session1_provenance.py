"""Session-1 review provenance audit (revision round 1, 23 Oct 2026). CPU only, reads existing files, writes
session1_provenance.json beside this script. Output is aggregated (no blind-id-to-arm mapping).

Facts audited:
  * which rows of runs/18_review_mrtgan/review/review.csv carry reviewer `petani_nilai` (farmer, assessor 1) and
    which carry `codex_visual_ai` (a vision-language model); whether those rows were blank in the backup;
  * acceptance by arm and source, Fisher exact test per arm; farmer-only Wilson intervals;
  * reasons/target labels by source (for the review-reason statements of the paper);
  * the 22 MRT-GAN candidates used in detector training (linked by parent_id within arm MRTGAN_s12, using the
    `used` column of detector_mrtgan_2026-10-04/ranking_MRTGAN_s12.csv): source of their session-1 decision and later
    decisions of assessor 2 (pack_rater2) and assessor 1 in session 2 (pack_heldout);
  * the same for the 22 colour-transfer candidates used;
  * Cohen's kappa (assessor 1 session 1 vs assessor 2) on the 100 kappa items, overall and on farmer-rated rows only
    (same kappa and bootstrap code as kappa/analyze.py: 2000 resamples, random.Random(0));
  * assessor-1 test-retest (session 1 vs session 2) on the 62 train-derived items, split by session-1 source;
  * source of the review examples shown in the paper figures (smallest blind ids, postdetector figure script).
"""
import csv, hashlib, json, math, random
from pathlib import Path
from scipy.stats import fisher_exact

HERE = Path(__file__).resolve().parent
RV = HERE.parent
PKG = RV.parents[2]
R18 = PKG / '02_Hasil_Eksperimen/runs/18_review_mrtgan'
DET = RV.parent / 'detector_mrtgan_2026-10-04'
INPUTS = {
    'review_csv': R18 / 'review/review.csv',
    'backup_csv': R18 / 'review/backups/review_before_final148.csv',
    'review_key': R18 / 'review_key_PRIVATE.json',
    'ranking_mrtgan': DET / 'ranking_MRTGAN_s12.csv',
    'ranking_ct': DET / 'ranking_colour_transfer.csv',
    'manifest_mrtgan': DET / 'arm_manifest_mrtgan_k22.json',
    'manifest_ct': DET / 'arm_manifest_colour_transfer_k22.json',
    'rater2_key': RV / 'kappa/pack_rater2_key_PRIVATE.json',
    'rater2_csv': RV / 'kappa/pack_rater2/review.csv',
    'heldout_key': RV / 'kappa/pack_heldout_key_PRIVATE.json',
    'heldout_csv': RV / 'kappa/pack_heldout/review.csv',
    'rereview148_key': RV / 'rereview148/pack_rereview148_key_PRIVATE.json',
    'kappa_results': RV / 'kappa/kappa_heldout_results.json',
}
CRIT = ['accepted', 'target_label', 'texture_realistic', 'shape_ok', 'boundary_ok', 'artifact_free']
FARMER, VLM = 'petani_nilai', 'codex_visual_ai'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p): return list(csv.DictReader(open(p, newline='')))


def kappa(a, b):  # identical to kappa/analyze.py
    n = len(a); cats = sorted(set(a) | set(b)); po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def boot(a, b, n=2000):  # identical to kappa/analyze.py
    rng = random.Random(0); idx = range(len(a)); v = []
    for _ in range(n):
        s = [rng.choice(idx) for _ in idx]; v.append(kappa([a[i] for i in s], [b[i] for i in s]))
    v.sort(); return v[int(.025 * n)], v[int(.975 * n) - 1]


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return [c - h, c + h]


def agree(a, b):
    lo, hi = boot(a, b)
    t = {f'{x}/{y}': sum(p == x and q == y for p, q in zip(a, b)) for x in ('yes', 'no') for y in ('yes', 'no')}
    return dict(n=len(a), kappa=kappa(a, b), ci95=[lo, hi], agreement=sum(x == y for x, y in zip(a, b)) / len(a),
                table_first_by_second=t)


rev = rows(INPUTS['review_csv']); bak = rows(INPUTS['backup_csv'])
key = {k['blind_id']: k for k in json.load(open(INPUTS['review_key']))}
assert len(rev) == 868 and len(bak) == 868
r1 = {r['blind_id']: r for r in rev}
out = dict(inputs={k: dict(path=str(v.relative_to(PKG)), sha256=sha(v)) for k, v in INPUTS.items()})

# --- row ranges and backup ----------------------------------------------------------------------------------------
lab = [r['reviewer'] for r in rev]
ranges = {}
for i, l in enumerate(lab, start=1): ranges.setdefault(l, []).append(i)
out['row_ranges'] = {l: dict(n=len(v), first_row=min(v), last_row=max(v), contiguous=max(v) - min(v) + 1 == len(v))
                     for l, v in ranges.items()}
out['csv_sorted_by_blind_id'] = [r['blind_id'] for r in rev] == sorted(r['blind_id'] for r in rev)
bk = {r['blind_id']: r for r in bak}
blank = [b for b, r in bk.items() if not r['accepted'].strip()]
out['backup'] = dict(blank_rows=len(blank), blank_rows_all_vlm=all(r1[b]['reviewer'] == VLM for b in blank),
                     vlm_rows_all_blank_in_backup=all(not bk[r['blind_id']]['accepted'].strip() for r in rev if r['reviewer'] == VLM),
                     farmer_rows_unchanged=all(all(bk[r['blind_id']][c] == r[c] for c in CRIT)
                                               for r in rev if r['reviewer'] == FARMER))
out['reason_prefix_ai_all_rows'] = sum(r['reason'].startswith('Penilaian visual AI') for r in rev)

# --- acceptance by arm x source -------------------------------------------------------------------------------------
arms = sorted({k['arm'] for k in key.values()})
acc = {}
for a in arms:
    d = {}
    for src in (FARMER, VLM):
        it = [r for r in rev if key[r['blind_id']]['arm'] == a and r['reviewer'] == src]
        k = sum(r['accepted'] == 'yes' for r in it)
        tl = {t: sum(r['target_label'] == t for r in it) for t in sorted({r['target_label'] for r in it})}
        rej = [r for r in it if r['accepted'] != 'yes']
        fails = {c: sum(r[c] == 'no' for r in rej) for c in CRIT[2:]}
        d[src] = dict(accepted=k, n=len(it), wilson95=wilson(k, len(it)), target_label=tl, rejected=len(rej),
                      rejected_failing=fails,
                      rejected_failing_texture_and_artifact=sum(r['texture_realistic'] == 'no' and r['artifact_free'] == 'no' for r in rej))
    tot = sum(d[s]['accepted'] for s in d)
    d['total'] = dict(accepted=tot, n=217, wilson95=wilson(tot, 217))
    d['fisher_p_farmer_vs_vlm'] = fisher_exact([[d[FARMER]['accepted'], d[FARMER]['n'] - d[FARMER]['accepted']],
                                                [d[VLM]['accepted'], d[VLM]['n'] - d[VLM]['accepted']]])[1]
    acc[a] = d
out['acceptance_by_arm_and_source'] = acc

# --- used detector additions ----------------------------------------------------------------------------------------
k2 = json.load(open(INPUTS['rater2_key'])); v2 = {r['blind_id']: r for r in rows(INPUTS['rater2_csv'])}
kh = json.load(open(INPUTS['heldout_key'])); vh = {r['blind_id']: r for r in rows(INPUTS['heldout_csv'])}
by_old2, by_oldh = {}, {}
for k in k2:
    if k['old_blind_id']: by_old2.setdefault(k['old_blind_id'], []).append(v2[k['blind_id']]['accepted'])
for k in kh:
    if k['old_blind_id']: by_oldh.setdefault(k['old_blind_id'], []).append(vh[k['blind_id']]['accepted'])


def used_block(arm, ranking, manifest):
    used_parents = [r['parent_id'] for r in rows(ranking) if r['used'] == 'True']
    man = sorted(a['source'] for a in json.load(open(manifest))['added'])
    assert sorted(used_parents) == man, 'ranking/manifest mismatch'
    bid = {key[b]['parent_id']: b for b in key if key[b]['arm'] == arm}
    ids = [bid[p] for p in used_parents]
    res = dict(n=len(ids), linking_rule=f'parent_id of rows with used=True in {ranking.name}, matched to arm {arm} in review_key; equals manifest sources',
               all_accepted_session1=all(r1[b]['accepted'] == 'yes' for b in ids))
    for src in (FARMER, VLM):
        s = [b for b in ids if r1[b]['reviewer'] == src]
        r2 = [b for b in s if b in by_old2]; rh = [b for b in s if b in by_oldh]
        res[src] = dict(n=len(s),
                        assessor2_rated=len(r2), assessor2_accepted=sum(all(x == 'yes' for x in by_old2[b]) for b in r2),
                        assessor2_consistent_repeats=all(len(set(by_old2[b])) == 1 for b in r2),
                        assessor1_session2_rated=len(rh), assessor1_session2_accepted=sum(by_oldh[b][0] == 'yes' for b in rh))
    tot2 = [b for b in ids if b in by_old2]; toth = [b for b in ids if b in by_oldh]
    res['assessor2_rated'] = len(tot2); res['assessor2_accepted'] = sum(all(x == 'yes' for x in by_old2[b]) for b in tot2)
    res['assessor1_session2_rated'] = len(toth); res['assessor1_session2_accepted'] = sum(by_oldh[b][0] == 'yes' for b in toth)
    res['rated_by_neither_later'] = sum(b not in by_old2 and b not in by_oldh for b in ids)
    res['rated_by_either_later'] = sum(b in by_old2 or b in by_oldh for b in ids)
    res['accepted_by_every_later_rating'] = all(all(x == 'yes' for x in by_old2.get(b, []) + by_oldh.get(b, [])) for b in ids)
    return res


out['used_additions'] = dict(MRTGAN_s12=used_block('MRTGAN_s12', INPUTS['ranking_mrtgan'], INPUTS['manifest_mrtgan']),
                             colour_transfer=used_block('colour_transfer', INPUTS['ranking_ct'], INPUTS['manifest_ct']))

# --- kappa sample by source -----------------------------------------------------------------------------------------
kap = [k for k in k2 if k['origin'] == 'kappa']
assert len(kap) == 100
ks = {}
for name, sel in (('all_100', lambda k: True), ('farmer_rows', lambda k: r1[k['old_blind_id']]['reviewer'] == FARMER),
                  ('vlm_rows', lambda k: r1[k['old_blind_id']]['reviewer'] == VLM)):
    s = [k for k in kap if sel(k)]
    d = dict(n=len(s), by_arm={a: sum(k['arm'] == a for k in s) for a in arms})
    for c in CRIT:
        a = [r1[k['old_blind_id']][c] for k in s]; b = [v2[k['blind_id']][c] for k in s]
        d[c] = agree(a, b)
    d['assessor1_accepted_by_arm'] = {a: sum(r1[k['old_blind_id']]['accepted'] == 'yes' for k in s if k['arm'] == a) for a in arms}
    d['assessor2_accepted_by_arm'] = {a: sum(v2[k['blind_id']]['accepted'] == 'yes' for k in s if k['arm'] == a) for a in arms}
    ks[name] = d
ref = json.load(open(INPUTS['kappa_results']))['kappa']['accepted']['kappa']
ks['check_all_100_matches_kappa_heldout_results'] = abs(ks['all_100']['accepted']['kappa'] - ref) < 1e-12
out['kappa_sample_by_source'] = ks

# --- assessor-1 test-retest on the 62 train-derived items -----------------------------------------------------------
tr = [k for k in kh if k['origin'] == 'train_derived']
assert len(tr) == 62
tt = {}
for name, sel in (('all_62', lambda k: True), ('farmer_rows', lambda k: r1[k['old_blind_id']]['reviewer'] == FARMER),
                  ('vlm_rows', lambda k: r1[k['old_blind_id']]['reviewer'] == VLM)):
    s = [k for k in tr if sel(k)]
    d = dict(n=len(s), session1_accepted=sum(r1[k['old_blind_id']]['accepted'] == 'yes' for k in s),
             session2_accepted=sum(vh[k['blind_id']]['accepted'] == 'yes' for k in s))
    if s:
        for c in CRIT:
            d[c] = agree([r1[k['old_blind_id']][c] for k in s], [vh[k['blind_id']][c] for k in s])
    tt[name] = d
out['assessor1_test_retest'] = tt

# --- re-review pack ------------------------------------------------------------------------------------------------
rr = json.load(open(INPUTS['rereview148_key']))
out['rereview148_pack'] = dict(n=len(rr), same_items_as_vlm_rows=sorted(k['old_blind_id'] for k in rr) == sorted(r['blind_id'] for r in rev if r['reviewer'] == VLM),
                               by_arm={a: sum(k['arm'] == a for k in rr) for a in arms},
                               status='blank; farmer re-rating pending')

# --- examples shown in paper figures (postdetector_r512_2026-10-04/build_postdetector_figures.py rules) ---------------
def items(a): return sorted([r for r in rev if key[r['blind_id']]['arm'] == a], key=lambda r: r['blind_id'])
def fails(r): return {c for c in CRIT[2:] if r[c] == 'no'}
ex = {'mrtgan_accepted': [r for r in items('MRTGAN_s12') if r['accepted'] == 'yes'][:2],
      'mrtgan_rejected': [r for r in items('MRTGAN_s12') if r['accepted'] == 'no' and fails(r) == {'texture_realistic', 'artifact_free'}][:1],
      'mrtgan_boundary': [r for r in items('MRTGAN_s12') if r['accepted'] == 'no' and 'boundary_ok' in fails(r)][:1],
      'notex_rejected': [r for r in items('minus_textureD_s12') if r['accepted'] == 'no'][:1],
      'initial_rejected': [r for r in items('initial_config_s12') if r['accepted'] == 'no'][:1],
      'ct_accepted': [r for r in items('colour_transfer') if r['accepted'] == 'yes'][:1],
      'ct_ripe': [r for r in items('colour_transfer') if r['target_label'] == 'ripe'][:1],
      'difference_tiles': [r for r in items('MRTGAN_s12') if r['accepted'] == 'yes'][:3]}
out['figure_examples_source'] = {k: [r['reviewer'] for r in v] for k, v in ex.items()}

(HERE / 'session1_provenance.json').write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ('row_ranges', 'backup', 'used_additions', 'figure_examples_source')}, indent=1))
for a in arms: print(a, {s: (acc[a][s]['accepted'], acc[a][s]['n']) for s in (FARMER, VLM)}, round(acc[a]['fisher_p_farmer_vs_vlm'], 6))
for n in ('all_100', 'farmer_rows', 'vlm_rows'): print('kappa', n, ks[n]['n'], round(ks[n]['accepted']['kappa'], 3), [round(x, 3) for x in ks[n]['accepted']['ci95']])
for n in tt: print('retest', n, tt[n]['n'], tt[n]['session1_accepted'], tt[n]['session2_accepted'], tt[n].get('accepted', {}).get('kappa'))
