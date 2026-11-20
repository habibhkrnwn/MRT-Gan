"""Side-by-side V0 (vanilla CycleGAN) and G3 (MRT-GAN, seed 11) candidates from the same ripe parents.

Parents are drawn at random (seed 2026) from the 217 full-data parents; nothing is picked by eye.
Rows: source ripe crop, V0, G3, then a 2x detail patch (128 px at the fruit-mask centre) of V0 and G3.
Crops use the training crop window of each parent at 512 px, the same framing as the blinded review.
Only training photographs are read. Writes figures/img/v0g3_<k>_<row>.jpg and figures/data/v0g3.json.
"""
from pathlib import Path
import csv, hashlib, json, random, zipfile
import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent
G = OUT.parents[2]
POOL = G / 'runs/05_texture_study/p100/crops'
KEY = G / 'runs/05_texture_study/kaggle_v3_unblind/texture_study_v1/study/p100/review_key_PRIVATE.json'
REVIEW = G / 'runs/05_texture_study/kaggle_v3_output/texture_study_v1/study/p100/review/review.csv'
ZIPS = sorted((G / 'runs/05_texture_study/review_pack_download/review_pack').glob('review_part*.zip'))
V0 = G / 'runs/11_extra_generators/kaggle_v1_output/extra_generators_v1/result/candidates/V0/images'
BASE = G / 'melon_dataset_base_v1'
IMG, DATA = OUT / 'img', OUT
SIZE, PATCH, N = 512, 128, 6

records = {r['id']: r for r in json.load(open(POOL / 'crops_manifest.json'))['records']}
base = {r['id']: r for r in csv.DictReader(open(BASE / 'manifest.csv'))}
key = {k['blind_id']: k for k in json.load(open(KEY))}
accepted = {key[r['blind_id']]['parent_id']: r['accepted'] == 'yes'
            for r in csv.DictReader(open(REVIEW)) if key[r['blind_id']]['arm'] == 'G3'}
g3_blind = {k['parent_id']: b for b, k in key.items() if k['arm'] == 'G3'}
members = {}
for z in ZIPS:
    for n in zipfile.ZipFile(z).namelist():
        members[Path(n).stem] = (z, n)

parents = sorted(accepted)
assert len(parents) == 217
chosen = random.Random(2026).sample(parents, N)


def source_view(pid):
    rec = records[pid]
    path = BASE / base[pid]['image']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == rec['source_sha256']
    l, t, s = rec['crop_window']
    return Image.open(path).convert('RGB').crop((l, t, l + s, t + s)).resize((SIZE, SIZE), Image.Resampling.LANCZOS)


def patch_box(pid):
    m = np.asarray(Image.open(POOL / records[pid]['mask']).convert('L').resize((SIZE, SIZE), Image.Resampling.NEAREST)) > 127
    ys, xs = np.nonzero(m)
    cy, cx = int(ys.mean()), int(xs.mean())
    x0 = min(max(cx - PATCH // 2, 0), SIZE - PATCH); y0 = min(max(cy - PATCH // 2, 0), SIZE - PATCH)
    return (x0, y0, x0 + PATCH, y0 + PATCH)


info = []
for k, pid in enumerate(chosen, 1):
    src = source_view(pid)
    v0 = Image.open(V0 / f'balance__{pid}.png').convert('RGB')
    z, n = members[g3_blind[pid]]
    g3 = Image.open(zipfile.ZipFile(z).open(n)).convert('RGB')
    box = patch_box(pid)
    for name, im in (('source', src), ('v0', v0), ('g3', g3)):
        im.save(IMG / f'v0g3_{k}_{name}.jpg', quality=92)
    for name, im in (('v0', v0), ('g3', g3)):
        im.crop(box).resize((SIZE, SIZE), Image.Resampling.NEAREST).save(IMG / f'v0g3_{k}_{name}_detail.jpg', quality=92)
    info.append(dict(index=k, parent=pid, g3_accepted=accepted[pid], detail_box_512=box))
    print(k, pid, 'G3 accepted' if accepted[pid] else 'G3 rejected')
json.dump(dict(seed=2026, n=N, parents=info, note='V0 not reviewed; G3 = seed 11 review candidate'),
          open(DATA / 'v0g3.json', 'w'), indent=1)
