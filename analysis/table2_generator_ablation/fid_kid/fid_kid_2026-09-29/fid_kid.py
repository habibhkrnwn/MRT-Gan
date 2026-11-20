"""FID and KID of the G0-G3 full-data candidates against real unripe training crops (CPU).

Run with ~/.venvs/melon-bench/bin/python (torch-fidelity 0.4.0, standard Inception-v3 FID weights).
The generated crops are the blinded review images (crop window of the training crop, 512 px).
Real crops use the same crop window on the source photograph, so the framing matches.
Two views are measured: the whole crop, and the fruit only (pixels outside the GrabCut mask set to grey).
Only train photographs are read. Validation and test are never opened.
"""
from pathlib import Path
import csv, json, shutil, sys, hashlib
import numpy as np
from PIL import Image
import torch_fidelity

G = Path(__file__).resolve().parents[4]
POOL = G / 'runs/05_texture_study/p100/crops'
KEY = G / 'runs/05_texture_study/kaggle_v3_unblind/texture_study_v1/study/p100/review_key_PRIVATE.json'
REVIEW_ZIPS = sorted((G / 'runs/05_texture_study/review_pack_download/review_pack').glob('review_part*.zip'))
BASE = G / 'melon_dataset_base_v1'
WORK = Path(sys.argv[1])
OUT = Path(__file__).with_name('fid_kid.json')
SIZE = 512
GREY = (128, 128, 128)

manifest = json.load(open(POOL / 'crops_manifest.json'))
records = {r['id']: r for r in manifest['records']}
assert all(r['split'] == 'train' for r in records.values())
base = {r['id']: r for r in csv.DictReader(open(BASE / 'manifest.csv'))}


def fruit_only(img, mask_path):
    m = Image.open(mask_path).convert('L').resize((SIZE, SIZE), Image.Resampling.NEAREST)
    grey = Image.new('RGB', (SIZE, SIZE), GREY)
    return Image.composite(img, grey, m.point(lambda v: 255 if v > 127 else 0))


def save(img, rec, folder, name):
    (WORK / 'whole' / folder).mkdir(parents=True, exist_ok=True)
    (WORK / 'fruit' / folder).mkdir(parents=True, exist_ok=True)
    img.save(WORK / 'whole' / folder / f'{name}.png')
    fruit_only(img, POOL / rec['mask']).save(WORK / 'fruit' / folder / f'{name}.png')


def real_view(rec):
    full = Image.open(BASE / base[rec['id']]['image']).convert('RGB')
    assert hashlib.sha256((BASE / base[rec['id']]['image']).read_bytes()).hexdigest() == rec['source_sha256']
    l, t, s = rec['crop_window']
    return full.crop((l, t, l + s, t + s)).resize((SIZE, SIZE), Image.Resampling.LANCZOS)


# real unripe (domain A, 81) and real ripe sources (domain B, 217)
for rec in records.values():
    save(real_view(rec), rec, 'real_unripe' if rec['domain'] == 'A' else 'real_ripe_source', rec['id'])

# generated candidates from the blinded review pack
import zipfile
key = {k['blind_id']: k for k in json.load(open(KEY))}
for z in REVIEW_ZIPS:
    with zipfile.ZipFile(z) as zf:
        for n in zf.namelist():
            if not n.endswith('.png'):
                continue
            k = key[Path(n).stem]
            img = Image.open(zf.open(n)).convert('RGB')
            assert img.size == (SIZE, SIZE)
            save(img, records[k['parent_id']], k['arm'], k['parent_id'])

# kernel 22: vanilla CycleGAN (V0) and G3 seeds 12/13, exported the same way as G0-G3
EXTRA = G / 'runs/11_extra_generators/kaggle_v1_output/extra_generators_v1/result/candidates'
extra_arms = sorted(p.name for p in EXTRA.iterdir()) if EXTRA.exists() else []
for arm in extra_arms:
    files = sorted((EXTRA / arm / 'images').glob('balance__*.png'))
    assert len(files) == 217, (arm, len(files))
    for f in files:
        pid = f.stem.replace('balance__', '')
        img = Image.open(f).convert('RGB')
        assert img.size == (SIZE, SIZE)
        save(img, records[pid], arm, pid)

sets = ['real_ripe_source', 'G0', 'G1', 'G2', 'G3'] + extra_arms
counts = {v: {s: len(list((WORK / v / s).glob('*.png'))) for s in sets + ['real_unripe']} for v in ('whole', 'fruit')}
print(counts)
res = dict(counts=counts, reference='real_unripe (81 train crops)', kid_subset_size=80, kid_subsets=100,
           note='FID is biased upward at n=81; KID is the primary comparison. Absolute values are not comparable across datasets.')
for v in ('whole', 'fruit'):
    res[v] = {}
    for s in sets:
        m = torch_fidelity.calculate_metrics(input1=str(WORK / v / s), input2=str(WORK / v / 'real_unripe'),
                                             fid=True, kid=True, kid_subset_size=80, kid_subsets=100,
                                             cuda=False, verbose=False, cache=False, rng_seed=2026)
        res[v][s] = dict(fid=round(m['frechet_inception_distance'], 2),
                         kid_mean=round(m['kernel_inception_distance_mean'], 4),
                         kid_std=round(m['kernel_inception_distance_std'], 4))
        print(v, s, res[v][s], flush=True)
json.dump(res, open(OUT, 'w'), indent=1)
