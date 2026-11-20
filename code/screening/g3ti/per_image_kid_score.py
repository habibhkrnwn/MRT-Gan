"""Per-image proxy for KID: how close each candidate is to the real unripe distribution in the Inception-v3
pool features used by torch-fidelity's KID (fruit-only view, 512 px, grey background).

score(x) = mean_r k(x, r) - mean_r' k(r, r')_avg  with k(a,b) = (a.b/d + 1)^3, the KID kernel.
Higher (closer to 0 or positive) = more like real unripe. FID/KID themselves are set-level; this is only a ranking aid.
Usage: python per_image_kid_score.py <fidwork>   (fruit views written by evidence/fid_kid_2026-10-02)"""
import csv, json, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

W = Path(sys.argv[1]) / 'fruit'; OUT = Path(__file__).resolve().parent
fe = FeatureExtractorInceptionV3('inception-v3-compat', ['2048']).eval()
def feats(folder):
    names, out = [], []
    for f in sorted(folder.glob('*.png')):
        x = torch.from_numpy(np.asarray(Image.open(f).convert('RGB'))).permute(2, 0, 1)[None]
        with torch.no_grad(): out.append(fe(x)[0].numpy()[0])
        names.append(f.stem)
    return names, np.stack(out).astype(np.float64)
k = lambda a, b: (a @ b.T / a.shape[1] + 1) ** 3
_, R = feats(W / 'real_unripe'); rr = k(R, R); base = (rr.sum() - np.trace(rr)) / (len(R) * (len(R) - 1))
key = json.load(open(OUT / 'review_key_PRIVATE.json')); rows = list(csv.DictReader(open(OUT / 'review/review.csv')))
num = {r['blind_id']: i + 1 for i, r in enumerate(rows)}
arm_dir = {'G3TI_R512': 'G3TI_R512', 'G3TI_noT_R512': 'G3TI_noT_R512', 'CT': 'CT', 'V0': 'V0'}
res = []
for arm, d in arm_dir.items():
    names, X = feats(W / d); s = k(X, R).mean(1) - base
    for n, v in zip(names, s):
        b = next(x['blind_id'] for x in key if x['arm'] == arm and x['parent_id'] == n)
        res.append(dict(nomor=num[b], metode=arm, induk=n, skor=round(float(v), 4)))
res.sort(key=lambda r: -r['skor'])
with open(OUT / 'per_image_kid_score.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
print('top 15 overall'); [print(r) for r in res[:15]]
for arm in arm_dir:
    a = [r for r in res if r['metode'] == arm]; print(arm, 'top 10 nomor:', [r['nomor'] for r in a[:10]], 'median skor', round(float(np.median([r['skor'] for r in a])), 4))
