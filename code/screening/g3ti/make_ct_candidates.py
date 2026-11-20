"""Non-GAN baseline for the blind review: global CIELAB affine colour transfer (ripe -> unripe) of the fruit only.

One affine map per channel, fitted on pooled fruit pixels of the 217 ripe and 81 unripe training crops
(GrabCut masks of the texture-study pool), applied to every ripe parent. Each fruit keeps its own offset
from the ripe mean, so between-image diversity is preserved. Background pixels are the source pixels.
Output matches the review view of the GAN candidates: 512 px crop window of the training photograph.
Train split only. Writes candidates/CT/images/balance__<id>.png and stats.json."""
from pathlib import Path
import csv, hashlib, json
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb

OUT = Path(__file__).resolve().parent
G = OUT.parents[1]
POOL = G / 'runs/05_texture_study/p100/crops'
BASE = G / 'melon_dataset_base_v1'
SIZE = 512
man = json.load(open(POOL / 'crops_manifest.json'))
base = {r['id']: r for r in csv.DictReader(open(BASE / 'manifest.csv'))}

def view(rec):
    path = BASE / base[rec['id']]['image']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == rec['source_sha256']
    l, t, s = rec['crop_window']
    img = Image.open(path).convert('RGB').crop((l, t, l + s, t + s)).resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    mask = np.asarray(Image.open(POOL / rec['mask']).convert('L').resize((SIZE, SIZE), Image.Resampling.NEAREST)) > 127
    return np.asarray(img), mask

pix = {'A': [], 'B': []}
for rec in man['records']:
    a, m = view(rec); pix[rec['domain']].append(rgb2lab(a / 255.0)[m])
mu = {k: np.concatenate(v).mean(0) for k, v in pix.items()}; sd = {k: np.concatenate(v).std(0) for k, v in pix.items()}
d = OUT / 'candidates/CT/images'; d.mkdir(parents=True, exist_ok=True)
for rec in man['records']:
    if rec['domain'] != 'B':
        continue
    a, m = view(rec); lab = rgb2lab(a / 255.0)
    lab[m] = (lab[m] - mu['B']) / sd['B'] * sd['A'] + mu['A']
    rgb = (np.clip(lab2rgb(lab), 0, 1) * 255).round().astype(np.uint8); rgb[~m] = a[~m]
    Image.fromarray(rgb).save(d / f"balance__{rec['id']}.png")
json.dump(dict(method='global CIELAB affine, fruit only', ripe_mean=mu['B'].tolist(), ripe_sd=sd['B'].tolist(),
               unripe_mean=mu['A'].tolist(), unripe_sd=sd['A'].tolist(), n=len(list(d.glob('*.png'))), split='train'),
          open(OUT / 'candidates/CT/stats.json', 'w'), indent=1)
print(len(list(d.glob('*.png'))), 'CT candidates')
