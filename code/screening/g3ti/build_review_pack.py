"""Blind review pack for the G3TI round: every arm's 217 ripe-parent candidates under random identifiers.

    python build_review_pack.py ARM=path/to/images [ARM=path ...]

Writes review/images/<blind_id>.png, review/review.csv (same columns as the texture study) and
review_key_PRIVATE.json beside (not inside) review/. The order is a hash order, so arms and siblings
are interleaved. Then run code/cyclegan/tools/make_texture_review_tool.py --review review.
Train photographs only; the reviewer sees no arm, parent or generator information."""
import csv, hashlib, json, shutil, sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
FIELDS = ['blind_id', 'target_label', 'texture_realistic', 'shape_ok', 'boundary_ok', 'artifact_free',
          'accepted', 'reviewer', 'reason']
arms = dict(a.split('=', 1) for a in sys.argv[1:])
review = OUT / 'review'
if review.exists() or (OUT / 'review_key_PRIVATE.json').exists():
    raise SystemExit('review pack exists; never overwrite a review')
(review / 'images').mkdir(parents=True)
key = []
for arm, folder in arms.items():
    files = sorted(Path(folder).glob('balance__*.png'))
    if len(files) != 217:
        raise SystemExit(f'{arm}: expected 217 candidates, found {len(files)}')
    for f in files:
        pid = f.stem.replace('balance__', '')
        bid = hashlib.sha256(f'g3ti_review_v1:{arm}:{pid}'.encode()).hexdigest()[:20]
        shutil.copyfile(f, review / 'images' / f'{bid}.png')
        key.append(dict(blind_id=bid, arm=arm, parent_id=pid, source=str(f),
                        image_sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
if len({k['blind_id'] for k in key}) != len(key):
    raise SystemExit('blind id collision')
key.sort(key=lambda k: k['blind_id'])
json.dump(key, open(OUT / 'review_key_PRIVATE.json', 'w'), indent=1)
with open(review / 'review.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows({'blind_id': k['blind_id']} for k in key)
print(len(key), 'items from', list(arms))
