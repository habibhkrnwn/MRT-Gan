"""Panel: source ripe, V0, G3 (seed 11) and the kernel 23 arms on the same six parents as compare_v0_g3.

Parents come from runs/11_extra_generators/compare_v0_g3/v0g3.json (random, seed 2026), so nothing is
picked by eye. Each generator gets a full-crop row and a 4x nearest-neighbour detail row (128 px at the
fruit-mask centre). Only training photographs are read. Writes compare_g3_tuned.png beside this file.
"""
from pathlib import Path
import csv, hashlib, json, zipfile
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
G = OUT.parents[2]
POOL = G / 'runs/05_texture_study/p100/crops'
KEY = G / 'runs/05_texture_study/kaggle_v3_unblind/texture_study_v1/study/p100/review_key_PRIVATE.json'
ZIPS = sorted((G / 'runs/05_texture_study/review_pack_download/review_pack').glob('review_part*.zip'))
K22 = G / 'runs/11_extra_generators/kaggle_v1_output/extra_generators_v1/result/candidates'
K23 = G / 'runs/13_g3_tuned/kaggle_v1_output/g3_tuned_v1/result/candidates'
BASE = G / 'melon_dataset_base_v1'
CELL = 300

records = {r['id']: r for r in json.load(open(POOL / 'crops_manifest.json'))['records']}
base = {r['id']: r for r in csv.DictReader(open(BASE / 'manifest.csv'))}
key = json.load(open(KEY))
g3_blind = {k['parent_id']: k['blind_id'] for k in key if k['arm'] == 'G3'}
members = {Path(n).stem: (z, n) for z in ZIPS for n in zipfile.ZipFile(z).namelist()}
chosen = json.load(open(G / 'runs/11_extra_generators/compare_v0_g3/v0g3.json'))['parents']


def source(pid):
    path = BASE / base[pid]['image']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == records[pid]['source_sha256']
    l, t, s = records[pid]['crop_window']
    return Image.open(path).convert('RGB').crop((l, t, l + s, t + s)).resize((512, 512), Image.Resampling.LANCZOS)


def g3(pid):
    z, n = members[g3_blind[pid]]
    return Image.open(zipfile.ZipFile(z).open(n)).convert('RGB')


ARMS = [('V0', lambda p: Image.open(K22 / 'V0/images' / f'balance__{p}.png').convert('RGB')), ('G3', g3)]
ARMS += [(a, (lambda a: lambda p: Image.open(K23 / a / 'images' / f'balance__{p}.png').convert('RGB'))(a))
         for a in ('G3T', 'G3I', 'G3TI')]
rows = [('Source ripe', [source(c['parent']) for c in chosen])]
for name, load in ARMS:
    rows.append((name, [load(c['parent']) for c in chosen]))
for name, load in ARMS:
    rows.append((f'{name} detail 4x', [load(c['parent']).crop(tuple(c['detail_box_512'])).resize((512, 512), Image.Resampling.NEAREST)
                                       for c in chosen]))
LABEL = 150
W, H = LABEL + CELL * len(chosen), 24 + CELL * len(rows)
canvas = Image.new('RGB', (W, H), 'white')
draw = ImageDraw.Draw(canvas)
font = ImageFont.load_default(size=16)
for j, c in enumerate(chosen):
    draw.text((LABEL + j * CELL + 6, 4), c['parent'], fill='black', font=font)
for i, (name, ims) in enumerate(rows):
    y = 24 + i * CELL
    draw.text((6, y + CELL // 2 - 8), name, fill='black', font=font)
    for j, im in enumerate(ims):
        canvas.paste(im.resize((CELL - 4, CELL - 4), Image.Resampling.LANCZOS), (LABEL + j * CELL, y))
canvas.save(OUT / 'compare_g3_tuned.png')
print('rows', [r[0] for r in rows])
