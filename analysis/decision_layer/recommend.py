"""Harvest recommendation for new melon photos with the locked pipeline.

    python harvest/recommend.py --lock results/PIPELINE_LOCK.json --images <file or folder> ... --out <dir>

Writes <out>/recommendations.csv (one row per detected fruit), <out>/summary.json (counts per photo,
timing, lock hash) and <out>/annotated/<photo>.jpg (green = Panen, orange = Cek manual,
blue = Belum). Uses the deployment seed of the lock unless --seed is given. Refuses to run when
the code or weights differ from the lock.
"""
import argparse
import csv
import hashlib
import json
import platform
import statistics
from pathlib import Path

import decision as D
import infer

HERE = Path(__file__).resolve().parent
COLORS = {'Panen': (34, 160, 70), 'Cek manual': (235, 150, 20), 'Belum': (40, 110, 220)}
SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def images_from(args):
    out = []
    for item in map(Path, args):
        out += sorted(p for p in item.rglob('*') if p.suffix.lower() in SUFFIXES) if item.is_dir() else [item]
    return out


def annotate(path, fruits, target):
    from PIL import Image, ImageDraw, ImageFont
    with Image.open(path) as im:
        im = im.convert('RGB')
        draw = ImageDraw.Draw(im)
        width = max(3, im.width // 300)
        try:
            font = ImageFont.load_default(size=max(16, im.width // 45))
        except TypeError:  # Pillow < 10.1 has no sized default font
            font = ImageFont.load_default()
        for f in fruits:
            x, y, w, h = f['bbox']
            color = COLORS[f['decision']]
            draw.rectangle([x, y, x + w, y + h], outline=color, width=width)
            label = f" {f['decision']}  p(ripe)={f['p_ripe']:.2f} "
            left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
            ty = max(0, y - (bottom - top) - 2 * width)
            draw.rectangle([x, ty, x + right - left, ty + bottom - top + 2 * width], fill=color)
            draw.text((x, ty + width - top), label, fill=(255, 255, 255), font=font)
        im.save(target, quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lock', required=True)
    ap.add_argument('--images', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed')
    ap.add_argument('--no-annotate', action='store_true')
    args = ap.parse_args()
    lock = json.loads(Path(args.lock).read_text())
    seed = args.seed or lock['deployment_seed']
    entry = lock['seeds'][seed]
    bad = [n for n, h in lock['code_sha256'].items() if sha(HERE/n) != h]
    if sha(entry['weights']) != entry['weights_sha256']:
        bad.append('weights')
    if bad:
        raise SystemExit(f'pipeline differs from the lock: {bad}')
    paths = images_from(args.images)
    if not paths:
        raise SystemExit('no images found')
    out = Path(args.out)
    (out/'annotated').mkdir(parents=True, exist_ok=True)
    p = entry['decision']
    platt = D.Platt(p['platt_a'], p['platt_b'])
    model = infer.load_model(entry['weights'])
    items = [(f'{i:05d}_{path.stem}', path) for i, path in enumerate(paths)]
    preds, seconds = infer.predict(model, items)
    by = D.by_image(preds)
    summary, table = [], []
    for image_id, path in items:
        fruits = []
        for c in D.fruits(by.get(image_id, []), lock['rule']['iou_merge']):
            if c['confidence'] < p['tau_det']:
                continue
            prob = platt(c['margin'])
            fruits.append(dict(bbox=[round(v, 1) for v in c['bbox']], p_ripe=round(prob, 4),
                               decision=D.decide(prob, p['tau_low'], p['tau_high']),
                               s_ripe=c['s_ripe'], s_unripe=c['s_unripe'], detection_confidence=c['confidence']))
        for k, f in enumerate(fruits):
            table.append(dict(photo=str(path), fruit=k + 1, decision=f['decision'], p_ripe=f['p_ripe'],
                              detection_confidence=f['detection_confidence'], x=f['bbox'][0], y=f['bbox'][1],
                              w=f['bbox'][2], h=f['bbox'][3]))
        counts = {d: sum(f['decision'] == d for f in fruits) for d in D.DECISIONS}
        summary.append(dict(photo=str(path), fruits=len(fruits), **counts,
                            note='tidak ada buah terdeteksi' if not fruits else ''))
        if not args.no_annotate:
            annotate(path, fruits, out/'annotated'/f'{image_id}.jpg')
    fields = ['photo', 'fruit', 'decision', 'p_ripe', 'detection_confidence', 'x', 'y', 'w', 'h']
    with (out/'recommendations.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(table)
    timing = sorted(seconds.values())
    (out/'summary.json').write_text(json.dumps(dict(
        lock_sha256=sha(args.lock), seed=seed, thresholds=p, photos=summary,
        totals={d: sum(s[d] for s in summary) for d in D.DECISIONS},
        detector_seconds_per_photo=dict(median=statistics.median(timing), p90=timing[int(0.9 * (len(timing) - 1))]),
        hardware=dict(machine=platform.machine(), processor=platform.processor() or platform.platform(), device='cpu')),
        indent=2) + '\n')
    print(json.dumps({d: sum(s[d] for s in summary) for d in D.DECISIONS}), f'{len(paths)} photos ->', out)


if __name__ == '__main__':
    main()
