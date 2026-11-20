"""Balanced detector arm from the selected texture generator, best-ranked accepted candidates first.

    python tools/build_ranked_texture_arm.py --study <study> --dataset <regrouped v4> --out <arms dir>

Requested by the user on 2026-09-28 as a deviation from `texture_study.py build-detectors`
(which draws a hash-ordered k shared with G0 across five arms). This builds ONE arm:

- real train split of the p100 subset (301 photos) + validation (87), exactly as `write_arm`
  builds every other arm, so it is comparable to the balancing experiment's real_only,
  oversample_r100 and cyclegan_r100 (same data, same trainer);
- plus k = min(ripe - unripe deficit, accepted winner candidates) synthetic unripe photos;
- accepted candidates are ranked by how close their fruit-interior texture statistics are to real
  unripe TRAIN crops: sum over luma energy, chroma energy and Nyquist share (luma, R-luma, B-luma)
  of |ln(candidate / real mean)|, from p100/diagnostics/<winner>.json (EMA weights, the exported
  generator). Smaller is better; ties by candidate-id digest. Review decides eligibility, the
  diagnostics only decide order.

Validation only; the test split is never read.
"""
import argparse
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melon_cyclegan.texture_study import read, write, sha, digest, verify_scope, reviewed, tool  # noqa: E402

METRICS = ('luma_energy', 'chroma_energy', 'nyquist_luma', 'nyquist_r_minus_luma', 'nyquist_b_minus_luma')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    study, dataset, out = Path(args.study), Path(args.dataset), Path(args.out)
    scope, m = verify_scope(study, 100, dataset)
    selection = read(study/'selection.json')
    if selection['review_sha256'] != sha(scope/'review/review.csv'):
        raise ValueError('review changed after selection')
    winner = selection['winner']
    diag = read(scope/'diagnostics'/f'{winner}.json')
    binding = read(scope/'candidates'/winner/'binding.json')
    if diag['generator_sha256'] != binding['generator_sha256'] or diag['subset_digest'] != m['subset_digest']:
        raise ValueError('diagnostics do not belong to the exported candidates')
    ref = diag['real_unripe']
    per_image = {r['id']: r for r in diag['per_image']}

    accepted = [r for r in reviewed(study, 100)[winner] if r['accepted']]
    ranked = []
    for r in accepted:
        f = per_image[r['parent_id']]
        distance = sum(abs(math.log(max(f[k], 1e-12) / ref[k])) for k in METRICS)
        ranked.append(dict(r, texture_distance=distance))
    ranked.sort(key=lambda r: (r['texture_distance'], digest(r['candidate_id'])))

    counts = {c: sum(int(r['class_id']) == c for r in m['rows']) for c in (0, 1)}
    target = max(0, counts[0] - counts[1])
    k = min(target, len(ranked))
    if k <= 0:
        raise ValueError('no accepted candidates to add')
    bundle = scope/'candidates'/winner
    real = {r['id'] for r in m['rows']}
    additions = []
    for i, r in enumerate(ranked[:k]):
        if r['parent_id'] not in real:
            raise ValueError('synthetic parent outside subset')
        label = (bundle/r['full_label']).read_text()
        if label.split()[0] != '1':
            raise ValueError('synthetic label must be unripe')
        additions.append((f'add_{i:04d}', bundle/r['full_image'], label, r['parent_id']))

    name = f'texture_{winner}_ranked_r100'
    if (out/name).exists():
        raise ValueError(f'{out/name} exists')
    manifest = tool('build_balance_arms').write_arm(out, name, m['rows'] + m['validation'], dataset, additions)
    manifest.update(subset_digest=m['subset_digest'], selected_generator=winner, weights='ema',
                    selection=selection, review_sha256=sha(scope/'review/review.csv'),
                    accepted_winner=len(ranked), target_additions=target, added=k, full_balance=k == target,
                    ranking='ascending sum |ln(x/real unripe mean)| over ' + ', '.join(METRICS),
                    test_used=False, exploratory=True)
    manifest['files_sha256'] = {str(p.relative_to(out/name)): sha(p) for folder in ('images', 'labels')
                                for p in (out/name/folder).rglob('*') if p.is_file()}
    write(out/name/'arm_manifest.json', manifest)
    with (out/name/'ranking.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['rank', 'used', 'candidate_id', 'parent_id', 'texture_distance', *METRICS])
        for i, r in enumerate(ranked):
            p = per_image[r['parent_id']]
            w.writerow([i + 1, i < k, r['candidate_id'], r['parent_id'], round(r['texture_distance'], 5),
                        *[p[x] for x in METRICS]])
    print(name, manifest['counts'], f'accepted {len(ranked)}, added {k} of target {target}')


if __name__ == '__main__':
    main()
