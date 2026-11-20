"""Build image-based figure assets and their statistics from frozen artefacts (CPU only).

Run with the melon-bench environment; umap-learn may come from an extra PYTHONPATH entry:
    PYTHONPATH=<dir with umap-learn> ~/.venvs/melon-bench/bin/python evidence/build_image_figures.py

Reads training crops and masks, the blind-review images and CSV, the review key downloaded from the
notebook-13 Kaggle output after selection was locked, training and validation photographs, and the
validation predictions of the full-data G3 detector (seed 0, final checkpoint). The test split is
never opened. No model is trained; ResNet-50 ImageNet weights are used only as a fixed feature
extractor for the embedding figure.
"""
from pathlib import Path
import csv
import hashlib
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parents[1]
STUDY = ROOT / 'runs/05_texture_study'
KEY = STUDY / 'kaggle_v3_unblind/texture_study_v1/study/p100/review_key_PRIVATE.json'
REVIEW_IMAGES = STUDY / 'kaggle_v3_output/texture_study_v1/study/p100/review/images'
REVIEW_CSV = ROOT / 'FINAL_Melon_2026-09-28/02_Generator_G3/review_p100/review.csv'
SELECTION = ROOT / 'FINAL_Melon_2026-09-28/02_Generator_G3/selection.json'
CROPS = STUDY / 'p100/crops'
DATASET = ROOT / 'melon_dataset_base_v1'
DETECTOR = ROOT / 'runs/06_texture_ranked_detector/kaggle_v2_output/texture_ranked_v1/result/group0/texture_G3_ranked_r100__seed0'
IMG = PROJECT / 'figures/img'
DATA = PROJECT / 'figures/data'
ARMS = ['G0', 'G1', 'G2', 'G3']
CRITERIA = ['texture_realistic', 'shape_ok', 'boundary_ok', 'artifact_free']
SOURCES = []
BLUE, ORANGE = (42, 120, 214), (235, 104, 52)


def src(path):
    SOURCES.append(Path(path))
    return Path(path)


def save_jpg(image, name, size=None):
    IMG.mkdir(parents=True, exist_ok=True)
    if size:
        image = image.resize((size, size) if isinstance(size, int) else size, Image.LANCZOS)
    image.convert('RGB').save(IMG / name, quality=92)


def font(px):
    for name in ['DejaVuSans.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
        try:
            return ImageFont.truetype(name, px)
        except OSError:
            pass
    return ImageFont.load_default()


# ---------------------------------------------------------------- review key and criteria ----
key = json.loads(src(KEY).read_text())
rows = {r['blind_id']: r for r in csv.DictReader(src(REVIEW_CSV).open(newline=''))}
if {k['blind_id'] for k in key} != set(rows):
    raise ValueError('review key and review.csv do not describe the same items')
selection = json.loads(src(SELECTION).read_text())
if hashlib.sha256(REVIEW_CSV.read_bytes()).hexdigest() != selection['review_sha256']:
    raise ValueError('review.csv differs from the file locked by selection.json')
items = {a: [] for a in ARMS}
for k in key:
    items[k['arm']].append(dict(k, **rows[k['blind_id']]))
stats = {}
for a in ARMS:
    n = len(items[a])
    stats[a] = {c: sum(r[c] == 'yes' for r in items[a]) for c in CRITERIA + ['accepted']}
    stats[a]['n'] = n
for a in ['G1', 'G3']:
    if (stats[a]['accepted'], stats[a]['n']) != (selection['scores'][a]['accepted'], selection['scores'][a]['total']):
        raise ValueError(f'{a}: unblinded counts differ from selection.json')
DATA.mkdir(parents=True, exist_ok=True)
with (DATA / 'review_criteria.dat').open('w') as f:
    f.write('x arm ' + ' '.join(CRITERIA + ['accepted']) + '\n')
    for x, a in enumerate(ARMS, start=1):
        f.write(f'{x} {a} ' + ' '.join(f"{100 * stats[a][c] / stats[a]['n']:.2f}" for c in CRITERIA + ['accepted']) + '\n')

# ---------------------------------------------------------------- helpers for crops ----------
def review_crop(blind_id):
    return Image.open(src(REVIEW_IMAGES / f'{blind_id}.png')).convert('RGB')


def source_crop(parent):
    return Image.open(src(CROPS / f'trainB/{parent}.png')).convert('RGB')


def source_mask(parent, folder='trainB_mask'):
    return np.asarray(Image.open(src(CROPS / f'{folder}/{parent}.png')).convert('L'), dtype=float) / 255 > 0.5


def valid_region(review256):
    """Exclude black padding added when the review crop extended past the photograph border."""
    dark = np.asarray(review256, dtype=float).max(axis=2) < 3
    valid = np.ones(dark.shape, bool)
    rows_dark, cols_dark = dark.all(axis=1), dark.all(axis=0)
    for axis_dark, setter in [(rows_dark, lambda i: valid.__setitem__((i, slice(None)), False)),
                              (cols_dark, lambda i: valid.__setitem__((slice(None), i), False))]:
        for i in range(len(axis_dark)):          # leading run
            if not axis_dark[i]:
                break
            setter(i)
        for i in range(len(axis_dark) - 1, -1, -1):  # trailing run
            if not axis_dark[i]:
                break
            setter(i)
    # Drop two more pixels next to the padding: bicubic resizing mixes padding into the border row.
    return ~dilate(~valid, 2)


def dilate(mask, px):
    return np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(2 * px + 1)), dtype=float) > 127


# ---------------------------------------------------------------- locality of change ---------
# Compositing dilates the mask by 4 px and feathers it; a 6 px dilation bounds the blending support
# at 256 px. Change outside that support should be near zero.
locality = {}
for a in ARMS:
    inside, outside, share = [], [], []
    for r in items[a]:
        cand = review_crop(r['blind_id']).resize((256, 256), Image.BICUBIC)
        diff = np.abs(np.asarray(cand, float) - np.asarray(source_crop(r['parent_id']), float)).mean(axis=2)
        support = dilate(source_mask(r['parent_id']), 6)
        valid = valid_region(cand)
        inside.append(diff[support & valid].mean())
        outside.append(diff[~support & valid].mean())
        share.append(diff[support & valid].sum() / diff[valid].sum())
    locality[a] = dict(mean_abs_inside=float(np.mean(inside)), mean_abs_outside=float(np.mean(outside)),
                       median_share_inside=float(np.median(share)), min_share_inside=float(np.min(share)))

# Cross-generator identity outside the blending support, at review resolution (512 px, so 12 px).
by_parent = {}
for a in ARMS:
    for r in items[a]:
        by_parent.setdefault(r['parent_id'], {})[a] = r['blind_id']
max_between_arms = 0
for parent, ids in by_parent.items():
    arrays = {a: np.asarray(review_crop(ids[a]), dtype=int) for a in ARMS}
    mask512 = np.asarray(Image.open(CROPS / f'trainB_mask/{parent}.png').convert('L').resize((512, 512), Image.NEAREST)) > 127
    outside = ~dilate(mask512, 12)
    max_between_arms = max(max_between_arms, *(int(np.abs(arrays['G0'] - arrays[a])[outside].max()) for a in ARMS[1:]))
locality['max_abs_difference_between_generators_outside_support'] = max_between_arms
locality['parents_checked'] = len(by_parent)

# Difference-map tiles for three accepted G3 candidates (deterministic: smallest blind ids).
g3_ok = sorted([r for r in items['G3'] if r['accepted'] == 'yes'], key=lambda r: r['blind_id'])[:3]
for i, r in enumerate(g3_ok, start=1):
    cand = review_crop(r['blind_id']).resize((256, 256), Image.BICUBIC)
    source = source_crop(r['parent_id'])
    diff = np.abs(np.asarray(cand, float) - np.asarray(source, float)).mean(axis=2)
    diff[~valid_region(cand)] = 0
    t = np.clip(diff / 60.0, 0, 1)[..., None]                      # 60/255 saturates the ramp
    heat = (255 * (1 - t) + np.array(BLUE) * t).astype(np.uint8)   # single-hue sequential ramp
    heat_img = Image.fromarray(heat)
    edge = Image.fromarray((dilate(source_mask(r['parent_id']), 6) * 255).astype(np.uint8)).filter(ImageFilter.FIND_EDGES)
    heat_img.paste(Image.new('RGB', heat_img.size, ORANGE), mask=edge)
    save_jpg(source, f'diff_{i}_source.jpg', 300)
    save_jpg(cand, f'diff_{i}_g3.jpg', 300)
    save_jpg(heat_img, f'diff_{i}_map.jpg', 300)

# ---------------------------------------------------------------- review examples -------------
def pick(arm, predicate, count):
    return sorted([r for r in items[arm] if predicate(r)], key=lambda r: r['blind_id'])[:count]

def failed_exactly(*crits):
    """Rejected candidates whose failed criteria are exactly `crits` (one distinct pattern each)."""
    return lambda r: r['accepted'] == 'no' and {c for c in CRITERIA if r[c] == 'no'} == set(crits)

examples = {
    'g3_accepted': pick('G3', lambda r: r['accepted'] == 'yes', 2),
    'g3_all': pick('G3', failed_exactly(*CRITERIA), 1),
    'g3_boundary': pick('G3', failed_exactly('boundary_ok'), 1),
    'g3_artifact': pick('G3', failed_exactly('artifact_free'), 1),
    'g3_shape': pick('G3', failed_exactly('shape_ok'), 1),
    'g0_rejected': pick('G0', failed_exactly('texture_realistic', 'artifact_free'), 1),
    'g1_rejected': pick('G1', failed_exactly('texture_realistic', 'artifact_free'), 1),
    'g2_rejected': pick('G2', failed_exactly('boundary_ok', 'artifact_free'), 1),
}
if any(len(v) != (2 if k == 'g3_accepted' else 1) for k, v in examples.items()):
    raise ValueError('a review-example pattern has no candidate')
example_meta = {}
for name, chosen in examples.items():
    for j, r in enumerate(chosen, start=1):
        save_jpg(review_crop(r['blind_id']), f'review_{name}_{j}.jpg', 300)
        example_meta[f'{name}_{j}'] = {c: r[c] for c in ['arm', 'parent_id'] + CRITERIA + ['accepted']}

# ---------------------------------------------------------------- dataset examples ------------
manifest = {r['id']: r for r in csv.DictReader(src(DATASET / 'manifest.csv').open(newline=''))}
train_ripe = sorted(p.stem for p in (CROPS / 'trainB').glob('*.png'))
train_unripe = sorted(p.stem for p in (CROPS / 'trainA').glob('*.png'))


def photo(pid):
    row = manifest[pid]
    return Image.open(src(DATASET / row['image'])).convert('RGB'), DATASET / row['label']


def draw_boxes(image, boxes, width):
    d = ImageDraw.Draw(image)
    f = font(max(28, image.width // 22))
    for (x0, y0, x1, y1), colour, text, dashed in boxes:
        if dashed:
            for k in range(int(x0), int(x1), 40):
                d.line([(k, y0), (min(k + 22, x1), y0)], fill=colour, width=width)
                d.line([(k, y1), (min(k + 22, x1), y1)], fill=colour, width=width)
            for k in range(int(y0), int(y1), 40):
                d.line([(x0, k), (x0, min(k + 22, y1))], fill=colour, width=width)
                d.line([(x1, k), (x1, min(k + 22, y1))], fill=colour, width=width)
        else:
            d.rectangle([x0, y0, x1, y1], outline=colour, width=width)
        if text:
            tw = d.textlength(text, font=f)
            top = y0 - f.size - 10 if y0 - f.size - 10 >= 0 else y0   # keep the tag inside the frame
            d.rectangle([x0, top, x0 + tw + 12, top + f.size + 10], fill=colour)
            d.text((x0 + 6, top + 3), text, fill='white', font=f)
    return image


dataset_meta = {}
for cls, ids, colour in [('ripe', train_ripe[:2], ORANGE), ('unripe', train_unripe[:2], BLUE)]:
    for j, pid in enumerate(ids, start=1):
        image, label = photo(pid)
        boxes = []
        for line in src(label).read_text().split('\n'):
            if line.strip():
                c, xc, yc, w, h = map(float, line.split())
                W, H = image.size
                boxes.append(((xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H))
        if len(boxes) != 1:
            raise ValueError(f'{pid}: expected one annotated fruit')
        draw_boxes(image, [(boxes[0], colour, cls, False)], width=max(8, image.width // 160))
        save_jpg(image, f'dataset_{cls}_{j}_photo.jpg', (600, round(600 * image.height / image.width)))
        folder = 'trainB' if cls == 'ripe' else 'trainA'
        save_jpg(Image.open(src(CROPS / f'{folder}/{pid}.png')), f'dataset_{cls}_{j}_crop.jpg', 300)
        save_jpg(Image.open(src(CROPS / f'{folder}_mask/{pid}.png')).convert('RGB'), f'dataset_{cls}_{j}_mask.jpg', 300)
        dataset_meta[f'{cls}_{j}'] = pid

# ---------------------------------------------------------------- validation detections -------
gt = json.loads(src(DETECTOR / 'groundtruth.coco.json').read_text())
preds = json.loads(src(DETECTOR / 'predictions.coco.json').read_text())
names = {c['id']: c['name'] for c in gt['categories']}
gt_by = {a['image_id']: a for a in gt['annotations']}
best = {}
for p in preds:
    if p['image_id'] not in best or p['score'] > best[p['image_id']]['score']:
        best[p['image_id']] = p
agreement = sum(names[best[i]['category_id']] == names[gt_by[i]['category_id']] for i in gt_by)
detection_meta = {'validation_images': len(gt_by), 'top_prediction_class_matches_label': agreement,
                  'min_top_score': min(best[i]['score'] for i in gt_by)}
chosen = []
for cls in ['ripe', 'unripe']:
    ids = sorted(i for i in gt_by if names[gt_by[i]['category_id']] == cls)
    chosen += ids[:3]
for j, pid in enumerate(chosen, start=1):
    image, _ = photo(pid)
    g, p = gt_by[pid], best[pid]
    if image.size != (next(x for x in gt['images'] if x['id'] == pid)['width'], next(x for x in gt['images'] if x['id'] == pid)['height']):
        raise ValueError(f'{pid}: photograph size differs from validation annotation')
    to_xyxy = lambda b: (b[0], b[1], b[0] + b[2], b[1] + b[3])
    colour = ORANGE if names[p['category_id']] == 'ripe' else BLUE
    draw_boxes(image, [(to_xyxy(g['bbox']), (255, 255, 255), None, True),
                       (to_xyxy(p['bbox']), colour, f"{names[p['category_id']]} {p['score']:.2f}", False)],
               width=max(8, image.width // 160))
    # Crop around the union of both boxes (40% margin, 4:3) so the fruit is legible at column width.
    bx = [min(g['bbox'][0], p['bbox'][0]), min(g['bbox'][1], p['bbox'][1]),
          max(g['bbox'][0] + g['bbox'][2], p['bbox'][0] + p['bbox'][2]), max(g['bbox'][1] + g['bbox'][3], p['bbox'][1] + p['bbox'][3])]
    cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
    h = max(bx[3] - bx[1], (bx[2] - bx[0]) * 3 / 4) * 1.4
    h = min(h, image.height, image.width * 3 / 4)
    w = h * 4 / 3
    x0 =min(max(0, cx - w / 2), max(0, image.width - w)); y0 = min(max(0, cy - h / 2), max(0, image.height - h))
    image = image.crop((round(x0), round(y0), round(min(image.width, x0 + w)), round(min(image.height, y0 + h))))
    save_jpg(image, f'detect_{j}.jpg', (480, 360))
    detection_meta[f'detect_{j}'] = dict(image=pid, label=names[g['category_id']], predicted=names[p['category_id']], score=p['score'])

# ---------------------------------------------------------------- embedding (UMAP) ------------
import torch
import umap
from torchvision.models import resnet50, ResNet50_Weights

torch.manual_seed(0)
weights = ResNet50_Weights.IMAGENET1K_V2
model = resnet50(weights=weights).eval()
model.fc = torch.nn.Identity()
norm = weights.transforms()


def features(images):
    out = []
    with torch.no_grad():
        for k in range(0, len(images), 32):
            batch = torch.stack([norm(im) for im in images[k:k + 32]])
            out.append(model(batch).numpy())
    return np.concatenate(out)


def fruit_only(image, mask):
    arr = np.asarray(image.resize((256, 256), Image.BICUBIC), dtype=np.uint8).copy()
    arr[~mask] = 128
    return Image.fromarray(arr)


groups, full_imgs, fruit_imgs = [], [], []
for pid in train_ripe:
    im = Image.open(src(CROPS / f'trainB/{pid}.png')).convert('RGB')
    groups.append('real_ripe'); full_imgs.append(im); fruit_imgs.append(fruit_only(im, source_mask(pid)))
for pid in train_unripe:
    im = Image.open(src(CROPS / f'trainA/{pid}.png')).convert('RGB')
    groups.append('real_unripe'); full_imgs.append(im); fruit_imgs.append(fruit_only(im, source_mask(pid, 'trainA_mask')))
for a in ARMS:
    for r in sorted(items[a], key=lambda r: r['parent_id']):
        im = review_crop(r['blind_id']).resize((256, 256), Image.BICUBIC)
        groups.append(a); full_imgs.append(im); fruit_imgs.append(fruit_only(im, source_mask(r['parent_id'])))
groups = np.array(groups)
embedding_meta = {}
for variant, imgs in [('full', full_imgs), ('fruit', fruit_imgs)]:
    f = features(imgs)
    f = f / np.linalg.norm(f, axis=1, keepdims=True)
    real = np.isin(groups, ['real_ripe', 'real_unripe'])
    # 1-NN label of each generated crop among real crops (cosine), computed in feature space.
    sims = f[~real] @ f[real].T
    nn_label = groups[real][sims.argmax(axis=1)]
    centroid = {g: f[groups == g].mean(axis=0) for g in ['real_ripe', 'real_unripe']}
    embedding_meta[variant] = {}
    for a in ARMS:
        sel = groups[~real] == a
        fa = f[groups == a]
        embedding_meta[variant][a] = dict(
            nearest_real_is_unripe=float((nn_label[sel] == 'real_unripe').mean()),
            cos_to_unripe_centroid=float((fa @ centroid['real_unripe']).mean() / np.linalg.norm(centroid['real_unripe'])),
            cos_to_ripe_centroid=float((fa @ centroid['real_ripe']).mean() / np.linalg.norm(centroid['real_ripe'])))
    xy = umap.UMAP(n_neighbors=15, min_dist=0.15, metric='cosine', random_state=0).fit_transform(f)
    with (DATA / f'umap_{variant}.dat').open('w') as out:
        out.write('x y group\n')
        for (x, y), g in zip(xy, groups):
            out.write(f'{x:.4f} {y:.4f} {g}\n')

(DATA / 'image_figure_stats.json').write_text(json.dumps(dict(
    review=stats, locality=locality, review_examples=example_meta, dataset_examples=dataset_meta,
    detection=detection_meta, embedding=embedding_meta,
    embedding_note='ResNet-50 IMAGENET1K_V2 global-pool features, L2-normalized; UMAP n_neighbors=15, min_dist=0.15, cosine, random_state=0'),
    indent=2) + '\n')
(PROJECT / 'evidence/image_figure_source_hashes.json').write_text(json.dumps(
    {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(SOURCES))}, indent=2) + '\n')
print(json.dumps(dict(review=stats, locality=locality, detection=detection_meta, embedding=embedding_meta), indent=1))
