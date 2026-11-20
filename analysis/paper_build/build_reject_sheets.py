"""Contact sheets of rejected full-data candidates, ranked by review severity, for manual selection.

Severity uses only the recorded review: first the number of failed criteria (more is worse), then,
where the reviewer scored on the 1-4 scale, the sum of the four scores (lower is worse). Candidates
reviewed with yes/no only follow those with scores at equal failure counts. Nothing here enters the
manuscript; the author chooses the examples for Fig. review_examples from these sheets.
"""
from pathlib import Path
import csv
import json
import re

from PIL import Image, ImageDraw, ImageFont

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parents[1]
KEY = ROOT / 'runs/05_texture_study/kaggle_v3_unblind/texture_study_v1/study/p100/review_key_PRIVATE.json'
IMAGES = ROOT / 'runs/05_texture_study/kaggle_v3_output/texture_study_v1/study/p100/review/images'
REVIEW = ROOT / 'FINAL_Melon_2026-09-28/02_Generator_G3/review_p100/review.csv'
OUT = PROJECT / 'review_candidates'
CRITERIA = [('texture_realistic', 'T'), ('shape_ok', 'S'), ('boundary_ok', 'B'), ('artifact_free', 'A')]
LIMIT = {'G0': 36, 'G1': 36, 'G2': None, 'G3': None}   # None: every rejected candidate
TILE, COLS, BAND = 240, 6, 46

rows = {r['blind_id']: r for r in csv.DictReader(REVIEW.open(newline=''))}
try:
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
    bold = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 17)
except OSError:
    font = bold = ImageFont.load_default()

OUT.mkdir(exist_ok=True)
index = []
for arm in ['G3', 'G2', 'G0', 'G1']:
    rejected = []
    for k in json.loads(KEY.read_text()):
        r = rows[k['blind_id']]
        if k['arm'] != arm or r['accepted'] != 'no':
            continue
        failed = [code for field, code in CRITERIA if r[field] == 'no']
        m = re.search(r'skor=(\d{4})', r['reason'])
        score = sum(int(c) for c in m.group(1)) if m else None
        rejected.append(dict(k, failed=failed, score=score, reviewer=r['reviewer'], label=r['target_label']))
    rejected.sort(key=lambda x: (-len(x['failed']), x['score'] if x['score'] is not None else 99, x['blind_id']))
    total = len(rejected)
    if LIMIT[arm]:
        rejected = rejected[:LIMIT[arm]]
    n = len(rejected)
    sheet = Image.new('RGB', (COLS * TILE, ((n + COLS - 1) // COLS) * (TILE + BAND) + 40), 'white')
    d = ImageDraw.Draw(sheet)
    d.text((8, 10), f'{arm}: {n} of {total} rejected candidates, worst first (T texture, S shape, B boundary, A artifact; score = sum of 1-4 ratings)', fill='black', font=bold)
    for i, item in enumerate(rejected, start=1):
        code = f'{arm}-R{i:02d}'
        x, y = ((i - 1) % COLS) * TILE, 40 + ((i - 1) // COLS) * (TILE + BAND)
        sheet.paste(Image.open(IMAGES / f"{item['blind_id']}.png").convert('RGB').resize((TILE - 6, TILE - 6), Image.LANCZOS), (x + 3, y))
        fails = ' '.join(item['failed']) or '-'
        extra = f"score {item['score']}" if item['score'] is not None else 'yes/no review'
        d.text((x + 4, y + TILE - 2), code, fill='black', font=bold)
        d.text((x + 4, y + TILE + 20), f'fail: {fails} | {extra}', fill='#52514e', font=font)
        index.append(dict(code=code, arm=arm, rank=i, blind_id=item['blind_id'], parent_id=item['parent_id'],
                          failed=''.join(item['failed']), score=item['score'] if item['score'] is not None else '',
                          reviewer=item['reviewer'], target_label=item['label']))
    sheet.save(OUT / f'rejected_{arm}.jpg', quality=90)

with (OUT / 'rejected_index.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(index[0]))
    w.writeheader()
    w.writerows(index)
print(f'{len(index)} tiles written to {OUT.relative_to(PROJECT)}')

# Author's choice for the manuscript figure (28 September 2026), exported at figure resolution.
CHOSEN = ['G0-R18', 'G1-R26', 'G2-R01', 'G3-R36']
by_code = {r['code']: r for r in index}
for code in CHOSEN:
    Image.open(IMAGES / f"{by_code[code]['blind_id']}.png").convert('RGB').resize((300, 300), Image.LANCZOS).save(
        PROJECT / f'figures/img/review_chosen_{code}.jpg', quality=92)
(PROJECT / 'figures/data/review_chosen.json').write_text(json.dumps([by_code[c] for c in CHOSEN], indent=1) + '\n')
print('chosen:', ', '.join(f"{c} {by_code[c]['parent_id']} fail {by_code[c]['failed']}" for c in CHOSEN))
