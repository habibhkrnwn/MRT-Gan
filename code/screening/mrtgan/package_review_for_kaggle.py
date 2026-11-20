"""Package the filled blinded review for the detector kernel (notebook 27) as a private Kaggle dataset folder.

    python package_review_for_kaggle.py <filled review.csv> <out dir> [--mock]

Checks that every item is rated and that accepted implies unripe plus all four criteria. Writes review.csv,
review_key_PRIVATE.json, review_thumbs.npz (64 px BOX thumbnails of the reviewed MRT-GAN and colour-transfer images,
used by the kernel to verify that its re-export matches what was reviewed), ct_stats.json and dataset-metadata.json.
--mock fills random answers into a copy (smoke tests only; never upload a mock)."""
import csv, json, random, shutil, sys
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
src, out = Path(sys.argv[1]), Path(sys.argv[2]); mock = '--mock' in sys.argv
CRIT = ('texture_realistic', 'shape_ok', 'boundary_ok', 'artifact_free')
key = json.load(open(HERE / 'review_key_PRIVATE.json'))
rows = list(csv.DictReader(open(src, newline='')))
if [r['blind_id'] for r in rows] != [r['blind_id'] for r in csv.DictReader(open(HERE / 'review/review.csv'))]:
    raise SystemExit('blind ids or order differ from the issued review.csv')
if mock:
    rng = random.Random(0)
    for r in rows:
        ok = rng.random() < 0.8
        r.update(target_label='unripe' if ok else rng.choice(['ripe', 'ambiguous']), accepted='yes' if ok else 'no',
                 reviewer='MOCK', **{c: 'yes' if ok else rng.choice(['yes', 'no']) for c in CRIT})
for r in rows:
    if r['accepted'] not in ('yes', 'no') or not r['reviewer'].strip(): raise SystemExit(f"incomplete: {r['blind_id']}")
    if r['target_label'] not in ('ripe', 'unripe', 'ambiguous') or any(r[c] not in ('yes', 'no') for c in CRIT):
        raise SystemExit(f"incomplete criteria: {r['blind_id']}")
    if r['accepted'] == 'yes' and (r['target_label'] != 'unripe' or any(r[c] != 'yes' for c in CRIT)):
        raise SystemExit(f"accepted contradicts criteria: {r['blind_id']}")
out.mkdir(parents=True, exist_ok=False)
with open(out / 'review.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
shutil.copy(HERE / 'review_key_PRIVATE.json', out / 'review_key_PRIVATE.json')
thumbs = {}
for k in key:
    if k['arm'] in ('MRTGAN_s12', 'colour_transfer'):
        im = Image.open(HERE / 'review/images' / f"{k['blind_id']}.png").convert('RGB').resize((64, 64), Image.Resampling.BOX)
        thumbs[f"{k['arm']}/{k['parent_id']}"] = np.asarray(im, dtype=np.float32)
np.savez_compressed(out / 'review_thumbs.npz', **thumbs)
shutil.copy(HERE.parent / '14_review_g3ti/candidates/CT/stats.json', out / 'ct_stats.json')
(out / 'dataset-metadata.json').write_text(json.dumps(dict(title='melon-mrtgan-review-v2' + ('-mock' if mock else ''),
    id='habibhammamkurniawan/melon-mrtgan-review-v2' + ('-mock' if mock else ''), licenses=[dict(name='CC0-1.0')]), indent=2))
acc = {}
for r in rows:
    k = next(x for x in key if x['blind_id'] == r['blind_id']); acc.setdefault(k['arm'], [0, 0])
    acc[k['arm']][0] += r['accepted'] == 'yes'; acc[k['arm']][1] += 1
print('packaged', out, {a: f'{v[0]}/{v[1]}' for a, v in acc.items()})
