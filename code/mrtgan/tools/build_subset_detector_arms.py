"""Detector arms for a data-limited subset, with the selected generator reviewed on its own.

    python tools/build_subset_detector_arms.py --study <study> --dataset <regrouped v4> \
        --fraction 25 --review <G3-only review.csv> --out <arms dir>

User decision 2026-09-28 (option A): on subsets only the selected generator's candidates are
reviewed, so the synthetic_G0 arm is dropped. Four arms, all on the subset's real train photos
plus the full validation split:

- real_only: nothing added;
- oversample: k copies of real unripe train photos, `order()` cycling (as `build_detectors`);
- geometry: the same k picks, alternately mirrored horizontally / vertically with exact bbox
  transforms (as `build_detectors`);
- synthetic_<winner>: the k best-ranked review-accepted candidates, ranked by the texture
  distance of `build_ranked_texture_arm.py` (diagnostics of the exported EMA generator).

k = min(ripe - unripe deficit of the subset, accepted candidates). The review file must contain
exactly the winner's rows of the scope's blind review; the private key is only read here.
Validation only; the test split is never read.
"""
import argparse
import collections
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melon_cyclegan.texture_study import read, write, sha, digest, order, verify_scope, tool  # noqa: E402

METRICS = ('luma_energy', 'chroma_energy', 'nyquist_luma', 'nyquist_r_minus_luma', 'nyquist_b_minus_luma')
CRITERIA = ('texture_realistic', 'shape_ok', 'boundary_ok', 'artifact_free')


def accepted_candidates(scope, winner, review_path, subset_rows):
    key = [k for k in read(scope/'review_key_PRIVATE.json') if k['arm'] == winner]
    with open(review_path, newline='') as f:
        rows = list(csv.DictReader(f))
    if sorted(r['blind_id'] for r in rows) != sorted(k['blind_id'] for k in key):
        raise ValueError('review file must hold exactly the winner rows of this scope')
    byid = {r['blind_id']: r for r in rows}
    bundle = scope/'candidates'/winner
    records = {x['candidate_id']: x for x in read(bundle/'records.json')}
    parents = {r['id'] for r in subset_rows}
    out = []
    for item in key:
        r = byid[item['blind_id']]
        if r['accepted'] not in ('yes', 'no') or not r['reviewer'].strip():
            raise ValueError('human review incomplete')
        if r['target_label'] not in ('ripe', 'unripe', 'ambiguous') or any(r[c] not in ('yes', 'no') for c in CRITERIA):
            raise ValueError('criteria incomplete')
        if r['accepted'] == 'yes' and (r['target_label'] != 'unripe' or any(r[c] != 'yes' for c in CRITERIA)):
            raise ValueError('accepted contradicts criteria')
        record = records[item['candidate_id']]
        if sha(bundle/record['full_image']) != item['image_sha256'] or sha(bundle/record['full_label']) != item['label_sha256']:
            raise ValueError('candidate changed after review')
        if record['parent_id'] not in parents:
            raise ValueError('candidate parent outside subset')
        label = (bundle/record['full_label']).read_text().split()
        if len(label) != 5 or label[0] != '1':
            raise ValueError('synthetic label must be one unripe bbox')
        if r['accepted'] == 'yes':
            out.append(dict(record, reviewer=r['reviewer']))
    return out, len(key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--fraction', type=int, choices=[25, 50], required=True)
    ap.add_argument('--review', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    study, dataset = Path(args.study), Path(args.dataset)
    scope, m = verify_scope(study, args.fraction, dataset)
    winner = read(study/'selection.json')['winner']
    out = Path(args.out)/f'p{args.fraction:03d}'
    if out.exists():
        raise ValueError(f'{out} exists')

    accepted, reviewed = accepted_candidates(scope, winner, args.review, m['rows'])
    diag = read(scope/'diagnostics'/f'{winner}.json')
    binding = read(scope/'candidates'/winner/'binding.json')
    if diag['generator_sha256'] != binding['generator_sha256'] or diag['subset_digest'] != m['subset_digest']:
        raise ValueError('diagnostics do not belong to the exported candidates')
    ref, per_image = diag['real_unripe'], {r['id']: r for r in diag['per_image']}
    for r in accepted:
        f = per_image[r['parent_id']]
        r['texture_distance'] = sum(abs(math.log(max(f[k], 1e-12) / ref[k])) for k in METRICS)
    accepted.sort(key=lambda r: (r['texture_distance'], digest(r['candidate_id'])))

    counts = collections.Counter(int(r['class_id']) for r in m['rows'])
    target = max(0, counts[0] - counts[1])
    k = min(target, len(accepted))
    if k <= 0:
        raise ValueError('no accepted candidates to add')
    real = {r['id']: r for r in m['rows']}
    minority = order([r['id'] for r in m['rows'] if int(r['class_id']) == 1])
    bundle = scope/'candidates'/winner
    geometry_dir = out/'geometry_files'
    geometry_dir.mkdir(parents=True)
    from PIL import Image, ImageOps

    arms = {}
    for arm in ('real_only', 'oversample', 'geometry', f'synthetic_{winner}'):
        additions = []
        for i in range(0 if arm == 'real_only' else k):
            if arm.startswith('synthetic'):
                r = accepted[i]
                additions.append((f'add_{i:04d}', bundle/r['full_image'], (bundle/r['full_label']).read_text(), r['parent_id']))
                continue
            pid = minority[i % len(minority)]
            row = real[pid]
            image, label = dataset/row['image'], (dataset/row['label']).read_text()
            if arm == 'geometry':
                horizontal = (i // len(minority)) % 2 == 0
                with Image.open(image) as im:
                    im = ImageOps.mirror(im) if horizontal else ImageOps.flip(im)
                    dest = geometry_dir/f'{i:04d}.png'
                    im.save(dest)
                parts = label.split()
                parts[1 if horizontal else 2] = str(1 - float(parts[1 if horizontal else 2]))
                label, image = ' '.join(parts) + '\n', dest
            additions.append((f'add_{i:04d}', image, label, pid))
        manifest = tool('build_balance_arms').write_arm(out, arm, m['rows'] + m['validation'], dataset, additions)
        manifest.update(subset_digest=m['subset_digest'], fraction=args.fraction, selected_generator=winner,
                        review_sha256=sha(Path(args.review)), reviewed=reviewed, accepted=len(accepted),
                        target_additions=target, common_additions=k, full_balance=k == target,
                        ranking='ascending sum |ln(x/real unripe mean)| over ' + ', '.join(METRICS),
                        exploratory=True, test_used=False)
        manifest['files_sha256'] = {str(p.relative_to(out/arm)): sha(p) for folder in ('images', 'labels')
                                    for p in (out/arm/folder).rglob('*') if p.is_file()}
        write(out/arm/'arm_manifest.json', manifest)
        arms[arm] = manifest['counts']
    with (out/f'ranking_{winner}.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['rank', 'used', 'candidate_id', 'parent_id', 'texture_distance'])
        for i, r in enumerate(accepted):
            w.writerow([i + 1, i < k, r['candidate_id'], r['parent_id'], round(r['texture_distance'], 5)])
    write(out/'arms_summary.json', dict(arms=arms, fraction=args.fraction, subset_digest=m['subset_digest'],
                                        winner=winner, reviewed=reviewed, accepted=len(accepted),
                                        target_additions=target, common_additions=k, test_used=False))
    print(args.fraction, {a: c['total_per_class'] for a, c in arms.items()}, f'k={k} accepted={len(accepted)}/{reviewed}')


if __name__ == '__main__':
    main()
