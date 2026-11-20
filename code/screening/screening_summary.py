"""Round-2 CPU check CB4: session-1 candidate screening reported as what it is, a vision-language-model screening.

All 868 decisions in runs/18_review_mrtgan/review/review.csv were written by an OpenAI Codex agent (model identifier
gpt-6-astra) in two sessions on 3 Oct 2026 (agent_log_audit.json: 720 + 148 rows, all matching the final CSV).
The reviewer identifiers in the CSV are relabelled here: `petani_nilai` = agent session 1 (rows 1-720),
`codex_visual_ai` = agent session 2 (rows 721-868). No human decision exists in this file.

Output (aggregated, no blind-id-to-arm mapping): per arm, combined screening outcome (accepted, Wilson 95% CI,
perceived-class counts, failing criteria among rejections), the split by agent session with Fisher's exact test, the
screening status of the 22 additions per arm used in detector training, and two descriptive model-screening
comparisons between MRT-GAN and colour transfer (Fisher). CPU only; writes screening_summary.json beside this script.
"""
import csv, hashlib, json, math
from pathlib import Path
from scipy.stats import fisher_exact

HERE = Path(__file__).resolve().parent
RV = HERE.parent
PKG = RV.parents[2]
R18 = PKG / '02_Hasil_Eksperimen/runs/18_review_mrtgan'
DET = RV.parent / 'detector_mrtgan_2026-10-04'
INPUTS = {'review_csv': R18 / 'review/review.csv', 'review_key': R18 / 'review_key_PRIVATE.json',
          'ranking_mrtgan': DET / 'ranking_MRTGAN_s12.csv', 'ranking_ct': DET / 'ranking_colour_transfer.csv',
          'agent_log_audit': HERE / 'agent_log_audit.json'}
CRIT = ['texture_realistic', 'shape_ok', 'boundary_ok', 'artifact_free']
SESSION = {'petani_nilai': 'agent_session_1', 'codex_visual_ai': 'agent_session_2'}
ARMS = ['MRTGAN_s12', 'minus_textureD_s12', 'initial_config_s12', 'colour_transfer']


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return [max(0.0, c - h), min(1.0, c + h)]


def block(rs):
    acc = sum(r['accepted'] == 'yes' for r in rs); rej = [r for r in rs if r['accepted'] != 'yes']
    lab = {}
    for r in rs: lab[r['target_label']] = lab.get(r['target_label'], 0) + 1
    return dict(n=len(rs), accepted=acc, wilson95=wilson(acc, len(rs)), target_label=lab, rejected=len(rej),
                rejected_failing={c: sum(r[c] == 'no' for r in rej) for c in CRIT},
                rejected_failing_texture_and_artifact=sum(r['texture_realistic'] == 'no' and r['artifact_free'] == 'no' for r in rej))


audit = json.load(open(INPUTS['agent_log_audit']))
assert audit['coverage']['rows_written_by_agent_save_calls'] == 868 and not audit['coverage']['rows_not_covered']
rev = list(csv.DictReader(open(INPUTS['review_csv'], newline='')))
key = {k['blind_id']: k for k in json.load(open(INPUTS['review_key']))}
assert len(rev) == 868 and set(SESSION) == {r['reviewer'] for r in rev}
for r in rev: r['arm'] = key[r['blind_id']]['arm']; r['parent_id'] = key[r['blind_id']]['parent_id']
out = dict(inputs={k: dict(path=str(v.relative_to(PKG)), sha256=sha(v)) for k, v in INPUTS.items()},
           relabel=SESSION, screening_model='OpenAI Codex agent, model identifier gpt-6-astra (both sessions)',
           human_decisions_in_file=0, by_arm={})
for arm in ARMS:
    rs = [r for r in rev if r['arm'] == arm]
    b = dict(combined=block(rs))
    for rid, s in SESSION.items(): b[s] = block([r for r in rs if r['reviewer'] == rid])
    s1, s2 = b['agent_session_1'], b['agent_session_2']
    b['fisher_p_session1_vs_session2'] = fisher_exact([[s1['accepted'], s1['n'] - s1['accepted']],
                                                       [s2['accepted'], s2['n'] - s2['accepted']]])[1]
    out['by_arm'][arm] = b
# additions used in detector training: all screening-accepted, split by agent session
for arm, rk in (('MRTGAN_s12', 'ranking_mrtgan'), ('colour_transfer', 'ranking_ct')):
    used = [r['parent_id'] for r in csv.DictReader(open(INPUTS[rk])) if r['used'] == 'True']
    rs = [r for r in rev if r['arm'] == arm and r['parent_id'] in used]
    assert len(rs) == len(used) == 22
    out['by_arm'][arm]['used_additions'] = dict(n=22, all_screening_accepted=all(r['accepted'] == 'yes' for r in rs),
                                                by_session={s: sum(r['reviewer'] == rid for r in rs) for rid, s in SESSION.items()})
m, c = out['by_arm']['MRTGAN_s12']['combined'], out['by_arm']['colour_transfer']['combined']
mu, cu = m['target_label'].get('unripe', 0), c['target_label'].get('unripe', 0)
out['model_screening_comparisons'] = dict(
    note='Descriptive comparisons of model-screening outcomes; not human or perceptual validation.',
    accepted_mrtgan_vs_ct=dict(mrtgan=[m['accepted'], m['n']], ct=[c['accepted'], c['n']],
                               fisher_p=fisher_exact([[m['accepted'], m['n'] - m['accepted']], [c['accepted'], c['n'] - c['accepted']]])[1]),
    labelled_unripe_mrtgan_vs_ct=dict(mrtgan=[mu, m['n']], ct=[cu, c['n']],
                                      fisher_p=fisher_exact([[mu, m['n'] - mu], [cu, c['n'] - cu]])[1]))
json.dump(out, open(HERE / 'screening_summary.json', 'w'), indent=1)
for arm in ARMS:
    b = out['by_arm'][arm]
    print(arm, b['combined']['accepted'], '/', b['combined']['n'], b['combined']['target_label'], b['combined']['rejected_failing'],
          'S1', b['agent_session_1']['accepted'], '/', b['agent_session_1']['n'], 'S2', b['agent_session_2']['accepted'], '/',
          b['agent_session_2']['n'], 'p=%.2g' % b['fisher_p_session1_vs_session2'], b.get('used_additions'))
print(out['model_screening_comparisons'])
