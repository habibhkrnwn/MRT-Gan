"""Image figures for the MRT-GAN (512 px) revision. Train photographs only; candidates as exported (EMA, mask paste).
Writes figures/img/{arch_source,arch_output,arch_real}.png, generator_ablation_panel.png, checkerboard_zoom.png.
Parents DSC_0031 and DSC_0163 come from the seed-2026 random draw of runs/11_extra_generators/compare_v0_g3/v0g3.json."""
from pathlib import Path
import csv, hashlib, json
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent; M = HERE.parents[1]; G = M.parents[1]
OUT = M / 'figures/img'; OUT.mkdir(parents=True, exist_ok=True)
POOL = G / 'runs/05_texture_study/p100/crops'; BASE = G / 'melon_dataset_base_v1'
recs = {r['id']: r for r in json.load(open(POOL / 'crops_manifest.json'))['records']}
base = {r['id']: r for r in csv.DictReader(open(BASE / 'manifest.csv'))}
ARMS = [('MRT-GAN', 'runs/16_g3ti_r512/kaggle_v1_output/g3ti_r512_v1/result/candidates/G3TI_R512'),
        ('- texture D', 'runs/16_g3ti_r512/kaggle_v1_output/g3ti_r512_v1/result/candidates/G3TI_noT_R512'),
        ('- 512 px', 'runs/13_g3_tuned/kaggle_v1_output/g3_tuned_v1/result/candidates/G3TI'),
        ('Initial config.', 'runs/05_texture_study/kaggle_v3_unblind/texture_study_v1/study/p100/candidates/G3'),
        ('Colour transfer', 'runs/14_review_g3ti/candidates/CT')]
def source(pid):
    path = BASE / base[pid]['image']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == recs[pid]['source_sha256']
    l, t, s = recs[pid]['crop_window']
    return Image.open(path).convert('RGB').crop((l, t, l + s, t + s)).resize((512, 512), Image.Resampling.LANCZOS)
def cand(d, pid):
    d = G / d
    f = d / 'images' / f'balance__{pid}.png'
    if f.exists(): return Image.open(f).convert('RGB')
    # initial configuration: full photographs; cut the same window
    full = d / 'full/images' / f'balance__{pid}.jpg'
    l, t, s = recs[pid]['crop_window']
    return Image.open(full).convert('RGB').crop((l, t, l + s, t + s)).resize((512, 512), Image.Resampling.LANCZOS)
box = {c['parent']: tuple(c['detail_box_512']) for c in json.load(open(G / 'runs/11_extra_generators/compare_v0_g3/v0g3.json'))['parents']}
parents = ['DSC_0031', 'DSC_0163']
# architecture thumbnails
source('DSC_0031').resize((256, 256), Image.Resampling.LANCZOS).save(OUT / 'arch_source.png')
cand(ARMS[0][1], 'DSC_0031').resize((256, 256), Image.Resampling.LANCZOS).save(OUT / 'arch_output.png')
real = sorted((POOL / 'trainA').glob('*.png'))[0]
Image.open(real).convert('RGB').resize((256, 256), Image.Resampling.LANCZOS).save(OUT / 'arch_real.png')
# ablation panel: rows = parent full, parent detail; cols = source + arms
C = 300; cols = [('Source ripe', source)] + [(n, (lambda d: lambda p: cand(d, p))(d)) for n, d in ARMS]
panel = Image.new('RGB', (C * len(cols), C * 2 * len(parents)), 'white')
for i, pid in enumerate(parents):
    for j, (_, fn) in enumerate(cols):
        im = fn(pid)
        panel.paste(im.resize((C - 4, C - 4), Image.Resampling.LANCZOS), (j * C, (2 * i) * C))
        panel.paste(im.crop(box[pid]).resize((C - 4, C - 4), Image.Resampling.NEAREST), (j * C, (2 * i + 1) * C))
panel.save(OUT / 'generator_ablation_panel.png')
# checkerboard zoom: 48x48 px at the detail-box centre, 8x nearest
Z = 384; zoom = Image.new('RGB', (Z * 3, Z * len(parents)), 'white')
for i, pid in enumerate(parents):
    x0, y0, x1, y1 = box[pid]; cx, cy = (x0 + x1) // 2, (y0 + y1) // 2; b = (cx - 24, cy - 24, cx + 24, cy + 24)
    for j, im in enumerate([source(pid), cand(ARMS[0][1], pid), cand(ARMS[1][1], pid)]):
        zoom.paste(im.crop(b).resize((Z - 4, Z - 4), Image.Resampling.NEAREST), (j * Z, i * Z))
zoom.save(OUT / 'checkerboard_zoom.png')
json.dump(dict(parents=parents, arms=ARMS, seed='11 for multi-seed rows', real_thumbnail=real.name), open(HERE / 'figure_sources.json', 'w'), indent=1)
print('ok')
