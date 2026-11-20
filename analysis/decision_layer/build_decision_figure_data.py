"""Decision-figure data from the locked MRT-GAN harvest pipeline (runs/20_harvest_mrtgan): writes
figures/data/decision_confusion.dat and decision_cells.tex from test.json (calibrated rule, seeds 0-4)."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent; DATA = HERE.parents[1] / 'figures/data'
t = json.load(open(HERE / 'test.json'))
cols = ['Panen', 'Cek manual', 'Belum', 'Tidak terdeteksi']
rows, cells = ['seed truth harvest manual notyet undetected'], []
for s in sorted(t['seeds'], key=int):
    c = t['seeds'][s]['calibrated']['confusion']
    for r, truth in enumerate(('ripe', 'unripe')):
        counts = [c[f'{truth}->{k}'] for k in cols]; n = sum(counts)
        rows.append(f"{s} {truth} " + ' '.join(map(str, counts)))
        for j, k in enumerate(counts):
            cells.append(f"\\DecisionCell{{{2 * int(s) + r}}}{{{j}}}{{{k}}}{{{round(100 * k / n)}}}")
(DATA / 'decision_confusion.dat').write_text('\n'.join(rows) + '\n')
(DATA / 'decision_cells.tex').write_text('\n'.join(cells) + '\n')
print('\n'.join(rows))
