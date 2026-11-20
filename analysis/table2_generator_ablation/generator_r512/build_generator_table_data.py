"""Numbers for the generator ablation table (MRT-GAN = 512 px, export-only compositing, identity 0.5, RGB texture D).

    python build_generator_table_data.py <WORK>

WORK is the folder written by evidence/fid_kid_2026-10-02/fid_kid.py plus fid_kid_2026-10-03/fid_kid_add.py
(whole/ and fruit/ views at 512 px, grey background outside the GrabCut mask). FID/KID are read from those runs'
JSON files; Nyquist and colour diversity are computed here on the same fruit views. Train photographs only.
Nyquist: 32x32 interior patches (no grey), up to 8 per image on a stride-16 grid, FFT without window, power in the
highest row+column over total, mean removed; luma Y and R-Y. Colour diversity: between-image SD of the per-fruit
mean CIELAB a* divided by that of the 81 real unripe crops."""
import json, statistics as st, sys
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.color import rgb2lab

HERE = Path(__file__).resolve().parent; EV = HERE.parent
W = Path(sys.argv[1]) / 'fruit'
fk = json.load(open(EV / 'fid_kid_2026-10-02/fid_kid.json')); add = json.load(open(EV / 'fid_kid_2026-10-03/fid_kid_add.json'))
FK = {v: {**fk[v], **add[v]} for v in ('whole', 'fruit')}
ROWS = [('real_unripe', ['real_unripe']), ('real_ripe_source', ['real_ripe_source']),
        ('MRT-GAN', ['G3TI_R512', 'G3TI_R512_s12', 'G3TI_R512_s13']),
        ('minus_texture_D', ['G3TI_noT_R512', 'G3TI_noT_R512_s12', 'G3TI_noT_R512_s13']),
        ('minus_512', ['G3TI', 'G3TIs12', 'G3TIs13']),
        ('minus_identity_256', ['G3T']), ('minus_export_only_256', ['G3I']),
        ('initial_configuration', ['G3', 'G3s12', 'G3s13']), ('CT', ['CT'])]

def nyq(p):
    p = p - p.mean(); P = np.abs(np.fft.fft2(p)) ** 2; t = P.sum()
    return (P[16, :].sum() + P[:, 16].sum() - P[16, 16]) / t if t > 0 else 0.0

def per_set(s):
    ly, lr, a_means = [], [], []
    for f in sorted((W / s).glob('*.png')):
        a = np.asarray(Image.open(f).convert('RGB')).astype(float); m = ~np.all(a == 128, axis=2)
        Y = a @ [0.299, 0.587, 0.114]; RY = a[..., 0] - Y; n = 0
        for y in range(0, 512 - 32, 16):
            for x in range(0, 512 - 32, 16):
                if m[y:y + 32, x:x + 32].all():
                    ly.append(nyq(Y[y:y + 32, x:x + 32])); lr.append(nyq(RY[y:y + 32, x:x + 32])); n += 1
                    if n >= 8: break
            if n >= 8: break
        small = np.asarray(Image.fromarray(a.astype(np.uint8)).resize((256, 256), Image.Resampling.NEAREST)) / 255.0
        ms = ~np.all(np.asarray(Image.fromarray(a.astype(np.uint8)).resize((256, 256), Image.Resampling.NEAREST)) == 128, axis=2)
        a_means.append(rgb2lab(small)[ms][:, 1].mean())
    return dict(nyq_y=float(np.mean(ly)), nyq_ry=float(np.mean(lr)), a_sd=float(np.std(a_means)), n=len(a_means))

raw = {s: per_set(s) for _, sets in ROWS for s in sets}
ref_a = raw['real_unripe']['a_sd']
def agg(vals):
    return dict(mean=round(st.mean(vals), 4), sd=round(st.pstdev(vals), 4) if len(vals) > 1 else None, values=[round(v, 4) for v in vals])
table = {}
for name, sets in ROWS:
    row = dict(sets=sets, nyq_y=agg([raw[s]['nyq_y'] for s in sets]), nyq_ry=agg([raw[s]['nyq_ry'] for s in sets]),
               a_diversity_ratio=agg([raw[s]['a_sd'] / ref_a for s in sets]))
    for v in ('whole', 'fruit'):
        if all(s in FK[v] for s in sets):
            row[f'{v}_fid'] = agg([FK[v][s]['fid'] for s in sets]); row[f'{v}_kid'] = agg([FK[v][s]['kid_mean'] for s in sets])
    table[name] = row
    print(name, {k: (v['mean'], v['sd']) for k, v in row.items() if isinstance(v, dict)})
json.dump(dict(source=['fid_kid_2026-10-02/fid_kid.json', 'fid_kid_2026-10-03/fid_kid_add.json'], test_used=False,
               raw=raw, table=table), open(HERE / 'generator_table.json', 'w'), indent=1)
