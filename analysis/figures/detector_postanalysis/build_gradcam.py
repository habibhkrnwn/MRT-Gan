"""Grad-CAM of the fixed YOLOv8n detectors on validation photographs and on G3 candidates (CPU).

Run with ~/.venvs/melon-bench/bin/python. Uses the selected checkpoints of the full-data G3 detector
(seed 0, the designated deployment model) and, for comparison statistics only, the full-data
real-only detector (seed 0). The target is the class score of the highest-scoring anchor for the
annotated class; gradients are taken at the three feature maps that feed the Detect head (strides
8, 16 and 32), each map is normalised to [0, 1], and the maps are averaged. Inputs are letterboxed
to 640x640 with the Ultralytics LetterBox (pad 114) as in training. The test split is never opened.
"""
from pathlib import Path
import csv
import hashlib
import json

import numpy as np
import torch
import torch.nn.functional as nnf
from PIL import Image, ImageDraw
from ultralytics.data.augment import LetterBox
from ultralytics.nn.tasks import DetectionModel

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parents[1]
DATASET = ROOT / 'melon_dataset_base_v1'
DETECTORS = {
    'G3': ROOT / 'FINAL_Melon_2026-09-28/01_Pipeline_Final/models/seed0/weights_selected.pt',
    'real_only': ROOT / 'runs/03_balancing_full_train/detector_7_arms/balance_detector/group0/real_only__seed0/weights_selected.pt',
}
VALIDATION = ROOT / 'runs/06_texture_ranked_detector/kaggle_v2_output/texture_ranked_v1/result/group0/texture_G3_ranked_r100__seed0/groundtruth.coco.json'
CANDIDATES = ROOT / 'runs/05_texture_study/kaggle_v3_unblind/texture_study_v1/study/p100/candidates/G3/full/images'
KEY = ROOT / 'runs/05_texture_study/kaggle_v3_unblind/texture_study_v1/study/p100/review_key_PRIVATE.json'
REVIEW = ROOT / 'FINAL_Melon_2026-09-28/02_Generator_G3/review_p100/review.csv'
IMG = PROJECT / 'figures/img'
DATA = PROJECT / 'figures/data'
SOURCES = []
BLUE_DARK = np.array([13, 54, 107], float)
ORANGE = (235, 104, 52)
torch.set_grad_enabled(True)


def src(p):
    SOURCES.append(Path(p))
    return Path(p)


def load(weights):
    ck = torch.load(src(weights), map_location='cpu', weights_only=False)
    net = DetectionModel(ck['cfg'], nc=ck['nc'], verbose=False)
    net.load_state_dict(ck['state_dict'])
    return net.eval()


def gradcam(net, image, cls):
    """Return (cam at original resolution in [0,1], top score, predicted-top class)."""
    lb = LetterBox(new_shape=(640, 640), auto=False, stride=32)
    arr = np.asarray(image.convert('RGB'))
    boxed = lb(image=arr)
    x = torch.from_numpy(boxed).permute(2, 0, 1).float().unsqueeze(0) / 255
    acts = {}
    hooks = [net.model[i].register_forward_hook(lambda m, inp, out, i=i: acts.__setitem__(i, out))
             for i in net.model[-1].f]
    out = net(x)
    for h in hooks:
        h.remove()
    y = out[0] if isinstance(out, (tuple, list)) else out
    scores = y[0, 4:, :]                                   # [nc, anchors], sigmoid class scores
    top_class = int(scores.max(dim=1).values.argmax())
    anchor = int(scores[cls].argmax())
    target = scores[cls, anchor]
    grads = torch.autograd.grad(target, [acts[i] for i in net.model[-1].f])
    cams = []
    for i, g in zip(net.model[-1].f, grads):
        a = acts[i][0]
        w = g[0].mean(dim=(1, 2))
        cam = torch.relu((w[:, None, None] * a).sum(0))
        cam = nnf.interpolate(cam[None, None], size=(640, 640), mode='bilinear', align_corners=False)[0, 0]
        if cam.max() > 0:
            cam = cam / cam.max()
        cams.append(cam)
    cam = torch.stack(cams).mean(0).detach().numpy()
    # Undo the letterbox: remove symmetric padding, then resize to the photograph.
    h0, w0 = arr.shape[:2]
    r = min(640 / h0, 640 / w0)
    nh, nw = round(h0 * r), round(w0 * r)
    top, left = (640 - nh) // 2, (640 - nw) // 2
    cam = cam[top:top + nh, left:left + nw]
    cam = np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize((w0, h0), Image.BILINEAR), float) / 255
    return cam, float(target), top_class


def box_stats(cam, box):
    """Share of CAM mass inside the box, whether the peak lies inside, and the box's area share
    (the share a spatially uniform map would give)."""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0 = max(0, x0), max(0, y0)
    inside = cam[y0:y1, x0:x1].sum()
    peak = np.unravel_index(cam.argmax(), cam.shape)
    area = (min(y1, cam.shape[0]) - y0) * (min(x1, cam.shape[1]) - x0) / cam.size
    return float(inside / cam.sum()), bool(y0 <= peak[0] <= y1 and x0 <= peak[1] <= x1), float(area)


def colour(cam, name):
    """Display-scaled Grad-CAM through a Matplotlib colour map (statistics are scale-free)."""
    from matplotlib import colormaps
    rgb = colormaps[name](np.clip(cam / cam.max(), 0, 1))[..., :3]
    return Image.fromarray((rgb * 255).astype(np.uint8))


# Following the PMT-GAN presentation: 'hot' (black-red-yellow-white) for the activation image and
# 'jet' (blue-green-yellow-red) for the fusion image.
def activation(cam):
    return colour(cam, 'hot')


def fusion(image, cam):
    return Image.blend(image.convert('RGB'), colour(cam, 'jet'), 0.5)


def crop43(image, box):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    h = min(max(box[3] - box[1], (box[2] - box[0]) * 3 / 4) * 1.4, image.height, image.width * 3 / 4)
    w = h * 4 / 3
    x0 = min(max(0, cx - w / 2), image.width - w)
    y0 = min(max(0, cy - h / 2), image.height - h)
    return image.crop((round(x0), round(y0), round(x0 + w), round(y0 + h)))


manifest = {r['id']: r for r in csv.DictReader(src(DATASET / 'manifest.csv').open(newline=''))}
gt = json.loads(src(VALIDATION).read_text())
names = {c['id']: c['name'] for c in gt['categories']}
val = [(a['image_id'], a['category_id'] - 1, (a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]))
       for a in sorted(gt['annotations'], key=lambda a: a['image_id'])]

rows = {r['blind_id']: r for r in csv.DictReader(src(REVIEW).open(newline=''))}
g3 = sorted([k for k in json.loads(src(KEY).read_text()) if k['arm'] == 'G3'], key=lambda k: k['blind_id'])


def parent_box(pid, size):
    line = src(DATASET / manifest[pid]['label']).read_text().split()
    _, xc, yc, w, h = map(float, line[:5])
    W, H = size
    return ((xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H)


stats = {}
for det_name, weights in DETECTORS.items():
    net = load(weights)
    per = {'validation': [], 'g3_candidates': []}
    for pid, cls, box in val:
        image = Image.open(src(DATASET / manifest[pid]['image'])).convert('RGB')
        cam, score, top = gradcam(net, image, cls)
        share, hit, area = box_stats(cam, box)
        per['validation'].append(dict(id=pid, label=names[cls + 1], share=share, area=area, peak_in_box=hit, top_class_matches=top == cls))
    for k in g3:
        path = CANDIDATES / f"{k['candidate_id']}.jpg"
        image = Image.open(src(path)).convert('RGB')
        box = parent_box(k['parent_id'], image.size)
        cam, score, top = gradcam(net, image, 1)
        share, hit, area = box_stats(cam, box)
        per['g3_candidates'].append(dict(id=k['candidate_id'], accepted=rows[k['blind_id']]['accepted'] == 'yes',
                                         share=share, area=area, peak_in_box=hit, top_class_unripe=top == 1))
    summary = {}
    for split, items in per.items():
        groups = {'all': items}
        if split == 'validation':
            groups.update({lab: [i for i in items if i['label'] == lab] for lab in ['ripe', 'unripe']})
        else:
            groups['accepted'] = [i for i in items if i['accepted']]
        summary[split] = {g: dict(n=len(v), median_share_in_box=float(np.median([i['share'] for i in v])),
                                  median_box_area_share=float(np.median([i['area'] for i in v])),
                                  median_share_to_area=float(np.median([i['share'] / i['area'] for i in v])),
                                  share_above_area=sum(i['share'] > i['area'] for i in v),
                                  peak_in_box=sum(i['peak_in_box'] for i in v)) for g, v in groups.items()}
    stats[det_name] = dict(summary=summary, items=per)

# Figure tiles from the G3 (deployment) detector: first two ripe and unripe validation identifiers,
# and the two accepted G3 candidates with the smallest blind identifiers.
net = load(DETECTORS['G3'])
examples = [v for v in val if v[1] == 0][:2] + [v for v in val if v[1] == 1][:2]
tiles = []
for pid, cls, box in examples:
    tiles.append((f'val_{names[cls + 1]}', Image.open(DATASET / manifest[pid]['image']).convert('RGB'), cls, box, pid))
for k in [k for k in g3 if rows[k['blind_id']]['accepted'] == 'yes'][:2]:
    image = Image.open(CANDIDATES / f"{k['candidate_id']}.jpg").convert('RGB')
    tiles.append(('g3_candidate', image, 1, parent_box(k['parent_id'], image.size), k['candidate_id']))
IMG.mkdir(parents=True, exist_ok=True)
tile_meta = []
for j, (kind, image, cls, box, ident) in enumerate(tiles, start=1):
    cam, score, top = gradcam(net, image, cls)
    share, hit, area = box_stats(cam, box)
    plain = image.copy()
    ImageDraw.Draw(plain).rectangle(box, outline=ORANGE, width=max(6, image.width // 200))
    crop43(plain, box).resize((480, 360), Image.LANCZOS).save(IMG / f'gradcam_{j}_image.jpg', quality=92)
    crop43(activation(cam), box).resize((480, 360), Image.LANCZOS).save(IMG / f'gradcam_{j}_act.jpg', quality=92)
    crop43(fusion(image, cam), box).resize((480, 360), Image.LANCZOS).save(IMG / f'gradcam_{j}_fusion.jpg', quality=92)
    tile_meta.append(dict(tile=j, kind=kind, id=ident, share=share, box_area_share=area, peak_in_box=hit))

from matplotlib import colormaps
for cmap in ['hot', 'jet']:
    bar = (colormaps[cmap](np.linspace(1, 0, 256))[:, :3] * 255).astype(np.uint8)
    Image.fromarray(np.repeat(bar[:, None, :], 24, axis=1)).save(IMG / f'gradcam_colorbar_{cmap}.png')
DATA.mkdir(parents=True, exist_ok=True)
(DATA / 'gradcam_stats.json').write_text(json.dumps(dict(
    method='Grad-CAM on Detect inputs (strides 8/16/32), class score of the top anchor for the annotated class; per-map max-normalised then averaged; letterbox 640',
    summary={d: s['summary'] for d, s in stats.items()}, tiles=tile_meta,
    items={d: s['items'] for d, s in stats.items()}), indent=1) + '\n')
(PROJECT / 'evidence/gradcam_source_hashes.json').write_text(json.dumps(
    {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(SOURCES))}, indent=1) + '\n')
print(json.dumps(dict(summary={d: s['summary'] for d, s in stats.items()}, tiles=tile_meta), indent=1))
