"""Image figures for the MRT-GAN (512 px) results, CPU only, train and validation photographs only (test never opened).

Sections: locality of change and difference maps (review crops vs 512-px source crops), review examples from blinded
review v2, validation detections of the MRT-GAN detector (kernel 27, seed 0, exported predictions), and a ResNet-50 /
UMAP embedding of real crops and the four reviewed sets. Run with melon-bench plus umap-learn on PYTHONPATH."""
from pathlib import Path
import csv, hashlib, json
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parent; PROJECT = HERE.parents[1]; ROOT = PROJECT.parents[1]
REV = ROOT / 'runs/18_review_mrtgan'
POOL = ROOT / 'runs/19_mrtgan_detector/local_reexport/crops512'
DATASET = ROOT / 'runs/19_mrtgan_detector/local_reexport/dataset_v4'
DETECTOR = ROOT / 'runs/19_mrtgan_detector/mrtgan_detector_v1/result/group1/mrtgan_k22__seed0'
IMG, DATA = PROJECT / 'figures/img', PROJECT / 'figures/data'
ARMS = ['MRTGAN_s12', 'minus_textureD_s12', 'initial_config_s12', 'colour_transfer']
CRIT = ['texture_realistic', 'shape_ok', 'boundary_ok', 'artifact_free']
BLUE, ORANGE = (42, 120, 214), (235, 104, 52)
SOURCES = []
def src(p): SOURCES.append(Path(p)); return Path(p)
def save_jpg(im, name, size=None):
    if size: im = im.resize((size, size) if isinstance(size, int) else size, Image.LANCZOS)
    im.convert('RGB').save(IMG / name, quality=92)
def font(px):
    for n in ['DejaVuSans.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
        try: return ImageFont.truetype(n, px)
        except OSError: pass
    return ImageFont.load_default()
def dilate(m, px): return np.asarray(Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(2 * px + 1)), float) > 127

key = json.load(open(src(REV / 'review_key_PRIVATE.json')))
rows = {r['blind_id']: r for r in csv.DictReader(open(src(REV / 'review/review.csv'), newline=''))}
items = {a: [dict(rows[k['blind_id']], arm=a, parent_id=k['parent_id']) for k in key if k['arm'] == a] for a in ARMS}
pm = {r['id']: r for r in json.load(open(POOL / 'crops_manifest.json'))['records']}
def review_crop(r): return Image.open(src(REV / 'review/images' / f"{r['blind_id']}.png")).convert('RGB')
def source_crop(pid): return Image.open(src(POOL / pm[pid]['image'])).convert('RGB')
def mask512(pid): return np.asarray(Image.open(src(POOL / pm[pid]['mask'])).convert('L')) > 127
def valid_region(im):
    dark = np.asarray(im, float).max(axis=2) < 3; valid = np.ones(dark.shape, bool)
    for axis in (0, 1):
        d = dark.all(axis=1 - axis)
        for rng in (range(len(d)), range(len(d) - 1, -1, -1)):
            for i in rng:
                if not d[i]: break
                if axis == 0: valid[i, :] = False
                else: valid[:, i] = False
    return ~dilate(~valid, 3)

# ---- locality of change (512 px; export pastes through the binary mask, resampling blurs ~1-2 px; support = mask dilated 12 px)
locality = {}
for a in ARMS:
    inside, outside, share = [], [], []
    for r in items[a]:
        diff = np.abs(np.asarray(review_crop(r), float) - np.asarray(source_crop(r['parent_id']), float)).mean(axis=2)
        sup = dilate(mask512(r['parent_id']), 12); v = valid_region(review_crop(r))
        inside.append(diff[sup & v].mean()); outside.append(diff[~sup & v].mean()); share.append(diff[sup & v].sum() / diff[v].sum())
    locality[a] = dict(mean_abs_inside=float(np.mean(inside)), mean_abs_outside=float(np.mean(outside)),
                       median_share_inside=float(np.median(share)), min_share_inside=float(np.min(share)))
# ---- difference tiles: three accepted MRT-GAN candidates with the smallest blind ids
ok = sorted([r for r in items['MRTGAN_s12'] if r['accepted'] == 'yes'], key=lambda r: r['blind_id'])[:3]
diff_meta = []
for i, r in enumerate(ok, 1):
    cand, source = review_crop(r), source_crop(r['parent_id'])
    diff = np.abs(np.asarray(cand, float) - np.asarray(source, float)).mean(axis=2); diff[~valid_region(cand)] = 0
    t = np.clip(diff / 60.0, 0, 1)[..., None]
    heat = Image.fromarray((255 * (1 - t) + np.array(BLUE) * t).astype(np.uint8))
    edge = Image.fromarray((dilate(mask512(r['parent_id']), 12) * 255).astype(np.uint8)).filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(3))
    heat.paste(Image.new('RGB', heat.size, ORANGE), mask=edge)
    save_jpg(source, f'diff_{i}_source.jpg', 300); save_jpg(cand, f'diff_{i}_mrtgan.jpg', 300); save_jpg(heat, f'diff_{i}_map.jpg', 300)
    diff_meta.append(r['parent_id'])
# ---- review examples (deterministic: smallest blind id matching each pattern)
def pick(a, pred, n=1): return sorted([r for r in items[a] if pred(r)], key=lambda r: r['blind_id'])[:n]
fails = lambda r: {c for c in CRIT if r[c] == 'no'}
examples = {'mrtgan_accepted': pick('MRTGAN_s12', lambda r: r['accepted'] == 'yes', 2),
            'mrtgan_rejected': pick('MRTGAN_s12', lambda r: r['accepted'] == 'no' and fails(r) == {'texture_realistic', 'artifact_free'}),
            'mrtgan_boundary': pick('MRTGAN_s12', lambda r: r['accepted'] == 'no' and 'boundary_ok' in fails(r)),
            'notex_rejected': pick('minus_textureD_s12', lambda r: r['accepted'] == 'no'),
            'initial_rejected': pick('initial_config_s12', lambda r: r['accepted'] == 'no'),
            'ct_accepted': pick('colour_transfer', lambda r: r['accepted'] == 'yes'),
            'ct_ripe': pick('colour_transfer', lambda r: r['target_label'] == 'ripe')}
if any(len(v) != (2 if k == 'mrtgan_accepted' else 1) for k, v in examples.items()): raise ValueError('missing review example')
example_meta = {}
for name, chosen in examples.items():
    for j, r in enumerate(chosen, 1):
        save_jpg(review_crop(r), f'review_{name}_{j}.jpg', 300)
        example_meta[f'{name}_{j}'] = {c: r[c] for c in ['arm', 'parent_id', 'target_label'] + CRIT + ['accepted']}
# ---- validation detections of the MRT-GAN detector (seed 0, exported predictions)
manifest = {r['id']: r for r in csv.DictReader(open(src(DATASET / 'manifest.csv'), newline=''))}
gt = json.load(open(src(DETECTOR / 'groundtruth.coco.json'))); preds = json.load(open(src(DETECTOR / 'predictions.coco.json')))
names = {c['id']: c['name'] for c in gt['categories']}; gt_by = {a['image_id']: a for a in gt['annotations']}; best = {}
for p in preds:
    if p['image_id'] not in best or p['score'] > best[p['image_id']]['score']: best[p['image_id']] = p
detection_meta = dict(validation_images=len(gt_by), top_class_matches=sum(names[best[i]['category_id']] == names[gt_by[i]['category_id']] for i in gt_by),
                      min_top_score=min(best[i]['score'] for i in gt_by))
def draw_boxes(image, boxes, width):
    d = ImageDraw.Draw(image); f = font(max(28, image.width // 22))
    for (x0, y0, x1, y1), colour, text, dashed in boxes:
        if dashed:
            for k in range(int(x0), int(x1), 40):
                d.line([(k, y0), (min(k + 22, x1), y0)], fill=colour, width=width); d.line([(k, y1), (min(k + 22, x1), y1)], fill=colour, width=width)
            for k in range(int(y0), int(y1), 40):
                d.line([(x0, k), (x0, min(k + 22, y1))], fill=colour, width=width); d.line([(x1, k), (x1, min(k + 22, y1))], fill=colour, width=width)
        else: d.rectangle([x0, y0, x1, y1], outline=colour, width=width)
        if text:
            tw = d.textlength(text, font=f); top = y0 - f.size - 10 if y0 - f.size - 10 >= 0 else y0
            d.rectangle([x0, top, x0 + tw + 12, top + f.size + 10], fill=colour); d.text((x0 + 6, top + 3), text, fill='white', font=f)
chosen = []
for cls in ['ripe', 'unripe']: chosen += sorted(i for i in gt_by if names[gt_by[i]['category_id']] == cls)[:3]
for j, pid in enumerate(chosen, 1):
    image = Image.open(src(DATASET / manifest[pid]['image'])).convert('RGB'); g, p = gt_by[pid], best[pid]
    xy = lambda b: (b[0], b[1], b[0] + b[2], b[1] + b[3])
    colour = ORANGE if names[p['category_id']] == 'ripe' else BLUE
    draw_boxes(image, [(xy(g['bbox']), (255, 255, 255), None, True), (xy(p['bbox']), colour, f"{names[p['category_id']]} {p['score']:.2f}", False)], max(8, image.width // 160))
    bx = [min(g['bbox'][0], p['bbox'][0]), min(g['bbox'][1], p['bbox'][1]), max(g['bbox'][0] + g['bbox'][2], p['bbox'][0] + p['bbox'][2]), max(g['bbox'][1] + g['bbox'][3], p['bbox'][1] + p['bbox'][3])]
    cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2; h = min(max(bx[3] - bx[1], (bx[2] - bx[0]) * 3 / 4) * 1.4, image.height, image.width * 3 / 4); w = h * 4 / 3
    x0 = min(max(0, cx - w / 2), max(0, image.width - w)); y0 = min(max(0, cy - h / 2), max(0, image.height - h))
    save_jpg(image.crop((round(x0), round(y0), round(min(image.width, x0 + w)), round(min(image.height, y0 + h)))), f'detect_{j}.jpg', (480, 360))
    detection_meta[f'detect_{j}'] = dict(image=pid, label=names[g['category_id']], predicted=names[p['category_id']], score=p['score'])
# ---- embedding (ResNet-50 ImageNet features, cosine UMAP), whole crop and fruit only, 256 px
import torch, umap
from torchvision.models import resnet50
torch.manual_seed(0)
model = resnet50(); model.load_state_dict(torch.load(Path.home() / '.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth')); model.fc = torch.nn.Identity(); model.eval()
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]; STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
def feats(ims):
    out = []
    with torch.no_grad():
        for k in range(0, len(ims), 32):
            x = torch.stack([(torch.from_numpy(np.asarray(im.resize((224, 224), Image.BICUBIC), np.float32) / 255).permute(2, 0, 1) - MEAN) / STD for im in ims[k:k + 32]])
            out.append(model(x).numpy())
    return np.concatenate(out)
def fruit_only(im, m):
    a = np.asarray(im.resize((256, 256), Image.BICUBIC)).copy(); mm = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((256, 256), Image.NEAREST)) > 127
    a[~mm] = 128; return Image.fromarray(a)
groups, full, fruit = [], [], []
for pid, r in sorted(pm.items()):
    im = source_crop(pid).resize((256, 256), Image.BICUBIC); g = 'real_ripe' if r['domain'] == 'B' else 'real_unripe'
    groups.append(g); full.append(im); fruit.append(fruit_only(im, mask512(pid)))
for a in ARMS:
    for r in sorted(items[a], key=lambda r: r['parent_id']):
        im = review_crop(r).resize((256, 256), Image.BICUBIC); groups.append(a); full.append(im); fruit.append(fruit_only(im, mask512(r['parent_id'])))
groups = np.array(groups); embedding_meta = {}
for variant, ims in (('full', full), ('fruit', fruit)):
    f = feats(ims); f = f / np.linalg.norm(f, axis=1, keepdims=True); real = np.isin(groups, ['real_ripe', 'real_unripe'])
    nn = groups[real][(f[~real] @ f[real].T).argmax(axis=1)]; cen = {g: f[groups == g].mean(0) for g in ('real_ripe', 'real_unripe')}
    embedding_meta[variant] = {a: dict(nearest_real_is_unripe=float((nn[groups[~real] == a] == 'real_unripe').mean()),
                                       cos_to_unripe_centroid=float((f[groups == a] @ cen['real_unripe']).mean() / np.linalg.norm(cen['real_unripe'])),
                                       cos_to_ripe_centroid=float((f[groups == a] @ cen['real_ripe']).mean() / np.linalg.norm(cen['real_ripe']))) for a in ARMS}
    xy = umap.UMAP(n_neighbors=15, min_dist=0.15, metric='cosine', random_state=0).fit_transform(f)
    with open(DATA / f'umap_r512_{variant}.dat', 'w') as out:
        out.write('x y group\n'); [out.write(f'{x:.4f} {y:.4f} {g}\n') for (x, y), g in zip(xy, groups)]
json.dump(dict(locality=locality, difference_tiles=diff_meta, review_examples=example_meta, detection=detection_meta, embedding=embedding_meta,
               note='review v2 (runs/18_review_mrtgan); detector kernel 27 mrtgan_k22 seed 0 exported predictions; ResNet-50 ImageNet; UMAP n=15, min_dist=0.15, cosine, rs=0'),
          open(DATA / 'postdetector_figure_stats.json', 'w'), indent=1)
json.dump({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(SOURCES))}, open(HERE / 'figure_source_hashes.json', 'w'), indent=1)
print(json.dumps(dict(locality=locality, detection={k: v for k, v in detection_meta.items() if not k.startswith('detect_')}, embedding=embedding_meta), indent=1))
