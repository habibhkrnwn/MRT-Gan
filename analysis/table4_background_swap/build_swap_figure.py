"""Figure for the background-swap test. Rows: the three unripe validation photographs with the most real-only errors
on swap (ties by id). Columns: original, same-class control, opposite-class swap. Images are rebuilt with the exact
functions and donors of bg_swap.py; counts are read from bg_swap_results.json."""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
OUT = Path(__file__).resolve().parent
src = (OUT/'bg_swap.py').read_text().split('t0 = time.time()')[0]
ns = {'__file__': str(OUT/'bg_swap.py')}; exec(src, ns)
r = json.load(open(OUT/'bg_swap_results.json'))
det = lambda arm, mode, pid: [d for s in range(5) for d in r['runs'][f'{arm}@{s}']['details'][mode] if d['id'] == pid]
unripe = sorted(d['id'] for d in r['runs']['real_only@0']['details']['swap'] if d['gt_class'] == 1)
err = {p: sum(not d['class_correct'] for d in det('real_only', 'swap', p)) for p in unripe}
pick = sorted(unripe, key=lambda p: (-err[p], p))[:3]
rows = {x['id']: x for x in ns['rows']}
FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 26)
def tile(img, box, w=560, h=420):
    """Whole photograph, letterboxed, so the scene stays visible; annotation box drawn dashed-free in white."""
    im = Image.fromarray(img); s = min(w/im.width, h/im.height)
    im = im.resize((round(im.width*s), round(im.height*s)), Image.Resampling.LANCZOS)
    out = Image.new('RGB', (w, h), (255, 255, 255)); ox, oy = (w-im.width)//2, (h-im.height)//2; out.paste(im, (ox, oy))
    ImageDraw.Draw(out).rectangle([ox+box[0]*s, oy+box[1]*s, ox+box[2]*s, oy+box[3]*s], outline=(255, 255, 255), width=3)
    return out
def label(t, lines, strip=80):
    out = Image.new('RGB', (t.width, t.height+strip), (0, 0, 0)); out.paste(t, (0, strip)); d = ImageDraw.Draw(out)
    for k, s in enumerate(lines): d.text((12, 6+36*k), s, fill=(255, 255, 255), font=FONT)
    return out
tiles = []
for p in pick:
    im, box = ns['load'](rows[p]); fr = ns['grabcut'](im, box)
    row = [label(tile(im, box), ['Original'])]
    for mode in ('control', 'swap'):
        donor = det('real_only', mode, p)[0]['donor']
        dim, dbox = ns['load'](rows[donor]); img, nb = ns['paste'](fr, (ns['remove_fruit'](dim, dbox), dbox))
        c = {a: sum(d['class_correct'] for d in det(a, mode, p)) for a in ('real_only', 'mrtgan_k22')}
        name = 'Control: unripe scene' if mode == 'control' else 'Swap: ripe scene'
        row.append(label(tile(img, nb), [name, f"Real-only {c['real_only']}/5 | MRT-GAN {c['mrtgan_k22']}/5"]))
    tiles.append(row)
W, H = 560, 500; g = 8
fig = Image.new('RGB', (3*W+2*g, len(tiles)*H+(len(tiles)-1)*g), (255, 255, 255))
for i, row in enumerate(tiles):
    for j, t in enumerate(row): fig.paste(t, (j*(W+g), i*(H+g)))
dst = OUT.parents[2]/'figures/img/bg_swap_examples.jpg'
fig.save(dst, quality=92); print(pick, err, dst)
