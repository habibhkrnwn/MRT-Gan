"""Panel: source ripe, V0, G3TI (256), G3TI without texture D (256), both at 512, and the colour-transfer
baseline (CT), on the six parents of runs/11_extra_generators/compare_v0_g3 (random, seed 2026).
Full-crop rows, then 4x nearest-neighbour detail rows (128 px at the fruit-mask centre). Train photographs only."""
from pathlib import Path
import csv, hashlib, json
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
G = OUT.parents[2]
POOL = G / 'runs/05_texture_study/p100/crops'
BASE = G / 'melon_dataset_base_v1'
records = {r['id']: r for r in json.load(open(POOL / 'crops_manifest.json'))['records']}
base = {r['id']: r for r in csv.DictReader(open(BASE / 'manifest.csv'))}
chosen = json.load(open(G / 'runs/11_extra_generators/compare_v0_g3/v0g3.json'))['parents']
ARMS = [('V0', 'runs/11_extra_generators/kaggle_v1_output/extra_generators_v1/result/candidates/V0'),
        ('G3TI 256', 'runs/13_g3_tuned/kaggle_v1_output/g3_tuned_v1/result/candidates/G3TI'),
        ('G3TI noT 256', 'runs/15_g3ti_ablation/kaggle_v1_output/g3ti_ablation_v1/result/candidates/G3TI_noT'),
        ('G3TI 512', 'runs/16_g3ti_r512/kaggle_v1_output/g3ti_r512_v1/result/candidates/G3TI_R512'),
        ('G3TI noT 512', 'runs/16_g3ti_r512/kaggle_v1_output/g3ti_r512_v1/result/candidates/G3TI_noT_R512'),
        ('CT (no GAN)', 'runs/14_review_g3ti/candidates/CT')]

def source(pid):
    path = BASE / base[pid]['image']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == records[pid]['source_sha256']
    l, t, s = records[pid]['crop_window']
    return Image.open(path).convert('RGB').crop((l, t, l + s, t + s)).resize((512, 512), Image.Resampling.LANCZOS)

load = lambda d: (lambda p: Image.open(G / d / 'images' / f'balance__{p}.png').convert('RGB'))
rows = [('Source ripe', [source(c['parent']) for c in chosen])]
rows += [(n, [load(d)(c['parent']) for c in chosen]) for n, d in ARMS]
rows += [(f'{n} 4x', [load(d)(c['parent']).crop(tuple(c['detail_box_512'])).resize((512, 512), Image.Resampling.NEAREST)
                      for c in chosen]) for n, d in [('Source', None)] + ARMS if d] 
rows.insert(len(ARMS) + 1, ('Source 4x', [source(c['parent']).crop(tuple(c['detail_box_512'])).resize((512, 512), Image.Resampling.NEAREST) for c in chosen]))
CELL, LABEL = 260, 150
canvas = Image.new('RGB', (LABEL + CELL * len(chosen), 24 + CELL * len(rows)), 'white')
draw = ImageDraw.Draw(canvas); font = ImageFont.load_default(size=16)
for j, c in enumerate(chosen):
    draw.text((LABEL + j * CELL + 6, 4), c['parent'], fill='black', font=font)
for i, (name, ims) in enumerate(rows):
    y = 24 + i * CELL; draw.text((6, y + CELL // 2 - 8), name, fill='black', font=font)
    for j, im in enumerate(ims):
        canvas.paste(im.resize((CELL - 4, CELL - 4), Image.Resampling.LANCZOS), (LABEL + j * CELL, y))
canvas.save(OUT / 'compare_r512.png'); print([r[0] for r in rows])
