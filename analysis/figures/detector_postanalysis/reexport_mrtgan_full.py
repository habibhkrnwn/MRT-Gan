"""Re-export full-photograph MRT-GAN seed-12 candidates on CPU for Grad-CAM and figures (same EMA G_B as kernel 26/27).

Builds a local copy of the frozen regrouped-v4 dataset (manifest + symlinks to the byte-identical photographs of
melon_dataset_base_v1, sha256-checked), the 512-px pool (same windows/masks as kernels 25-27), and runs the project
exporter. G_A is not used for ripe->unripe; a copy of G_B stands in for it only because the loader requires both files.
Integrity: review views must match the reviewed images (64-px BOX thumbnails). Train photographs only."""
import csv, hashlib, json, os, shutil, sys
from pathlib import Path
import numpy as np
from PIL import Image
HERE = Path(__file__).resolve().parent; G = HERE.parents[3]
OUT = G / 'runs/19_mrtgan_detector/local_reexport'
CODE = G / 'code/cyclegan'; sys.path.insert(0, str(CODE)); sys.path.insert(0, str(CODE / 'tools'))
os.environ.setdefault('MELON_DATASET', str(OUT / 'dataset_v4'))
R = G.parents[2] / 'transfer_v4_to_v5/melon_outputs/melon_dataset_regrouped_v4'; B = G / 'melon_dataset_base_v1'
D = OUT / 'dataset_v4'; D.mkdir(parents=True, exist_ok=True); shutil.copy(R / 'manifest.csv', D / 'manifest.csv')
base = {r['id']: r for r in csv.DictReader(open(B / 'manifest.csv'))}
for r in csv.DictReader(open(R / 'manifest.csv')):
    b = base[r['id']]
    assert hashlib.sha256((B / b['image']).read_bytes()).hexdigest() == r['image_sha256']
    for k in ('image', 'label'):
        dst = D / r[k]; dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists(): os.symlink(B / b[k], dst)
exec((HERE / 'pool512_fn.py').read_text())
POOL = OUT / 'crops512'
if not POOL.exists(): build_pool512(G / 'runs/05_texture_study/p100/crops', POOL, D)
pool_manifest = json.loads((POOL / 'crops_manifest.json').read_text())
CK = OUT / 'checkpoints/p100_G3TI_R512_seed12'; CK.mkdir(parents=True, exist_ok=True)
src = G / 'runs/17_g3ti_r512_seeds/kaggle_v1_output/g3ti_r512_seeds_v1/result/checkpoints/p100_G3TI_R512_seed12'
for f in ('latest_net_G_B_ema.pth', 'melon_identity.json', 'loss.json'): shutil.copy(src / f, CK / f)
shutil.copy(src / 'latest_net_G_B_ema.pth', CK / 'latest_net_G_A_ema.pth')
from melon_cyclegan import texture_study as s
CAND = OUT / 'candidates/MRTGAN_s12'
if not CAND.exists(): s.tool('generate_balance_candidates').run(CK.parent, 'p100_G3TI_R512_seed12', 'ema', POOL, CAND, D, 'cpu')
key = json.load(open(G / 'runs/18_review_mrtgan/review_key_PRIVATE.json'))
diffs = []
for k in key:
    if k['arm'] != 'MRTGAN_s12': continue
    a = np.asarray(Image.open(G / 'runs/18_review_mrtgan/review/images' / f"{k['blind_id']}.png").convert('RGB').resize((64, 64), Image.Resampling.BOX), float)
    b = np.asarray(Image.open(CAND / 'images' / f"balance__{k['parent_id']}.png").convert('RGB').resize((64, 64), Image.Resampling.BOX), float)
    diffs.append(float(np.abs(a - b).mean()))
res = dict(n=len(diffs), max_thumb_diff=max(diffs), mean_thumb_diff=float(np.mean(diffs)), pool512=pool_manifest['crops_digest'], test_used=False)
json.dump(res, open(HERE / 'reexport_integrity.json', 'w'), indent=1); print(res)
