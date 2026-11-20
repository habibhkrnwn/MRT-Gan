"""Incremental FID/KID for kernel 26 (R512 seeds 12/13), same method and rng seed as fid_kid_2026-10-02.

    python fid_kid_add.py <WORK of fid_kid_2026-10-02>

Reuses WORK/{whole,fruit}/real_unripe written by the 2026-10-02 run (81 real unripe train crops, 512 px, same
crop windows) and adds the new arms with the same whole/fruit views. Train photographs only."""
from pathlib import Path
import json, sys
from PIL import Image
import torch_fidelity

G = Path(__file__).resolve().parents[4]
POOL = G / 'runs/05_texture_study/p100/crops'
NEW = G / 'runs/17_g3ti_r512_seeds/kaggle_v1_output/g3ti_r512_seeds_v1/result/candidates'
WORK = Path(sys.argv[1]); SIZE = 512; GREY = (128, 128, 128)
records = {r['id']: r for r in json.load(open(POOL / 'crops_manifest.json'))['records']}
for v in ('whole', 'fruit'):
    assert len(list((WORK / v / 'real_unripe').glob('*.png'))) == 81

def fruit_only(img, mask_path):
    m = Image.open(mask_path).convert('L').resize((SIZE, SIZE), Image.Resampling.NEAREST)
    return Image.composite(img, Image.new('RGB', (SIZE, SIZE), GREY), m.point(lambda x: 255 if x > 127 else 0))

arms = sorted(p.name for p in NEW.iterdir())
res = {}
for arm in arms:
    files = sorted((NEW / arm / 'images').glob('balance__*.png')); assert len(files) == 217, (arm, len(files))
    for v in ('whole', 'fruit'): (WORK / v / arm).mkdir(parents=True, exist_ok=True)
    for f in files:
        pid = f.stem.replace('balance__', ''); img = Image.open(f).convert('RGB'); assert img.size == (SIZE, SIZE)
        img.save(WORK / 'whole' / arm / f'{pid}.png'); fruit_only(img, POOL / records[pid]['mask']).save(WORK / 'fruit' / arm / f'{pid}.png')
    for v in ('whole', 'fruit'):
        m = torch_fidelity.calculate_metrics(input1=str(WORK / v / arm), input2=str(WORK / v / 'real_unripe'), fid=True, kid=True,
                                             kid_subset_size=80, kid_subsets=100, cuda=False, verbose=False, cache=False, rng_seed=2026)
        res.setdefault(v, {})[arm] = dict(fid=round(m['frechet_inception_distance'], 2), kid_mean=round(m['kernel_inception_distance_mean'], 4),
                                          kid_std=round(m['kernel_inception_distance_std'], 4))
        print(v, arm, res[v][arm], flush=True)
json.dump(dict(reference='real_unripe (81 train crops), reused from fid_kid_2026-10-02 WORK', kid_subset_size=80, kid_subsets=100, **res),
          open(Path(__file__).with_name('fid_kid_add.json'), 'w'), indent=1)
