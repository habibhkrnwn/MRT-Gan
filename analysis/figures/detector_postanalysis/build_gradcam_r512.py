"""Grad-CAM of the YOLOv8n detectors of kernel 27 (one session, k=22) on validation photographs and on the 217
MRT-GAN seed-12 candidates (CPU). Same method as evidence/build_gradcam.py: class score of the highest-scoring anchor for
the annotated class, gradients at the three Detect inputs (strides 8/16/32), per-map max-normalised then averaged,
letterbox 640. Detectors: selected seed-0 checkpoints of the MRT-GAN arm and of real-only. Test split never opened."""
from pathlib import Path
import csv, hashlib, json
import numpy as np
import torch
import torch.nn.functional as nnf
from PIL import Image, ImageDraw
from ultralytics.data.augment import LetterBox
from ultralytics.nn.tasks import DetectionModel

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
ROOT = PROJECT.parents[1]
DATASET = ROOT / 'runs/19_mrtgan_detector/local_reexport/dataset_v4'
RUN = ROOT / 'runs/19_mrtgan_detector/mrtgan_detector_v1/result'
DETECTORS = {'MRT-GAN': RUN / 'group1/mrtgan_k22__seed0/weights_selected.pt',
             'real_only': RUN / 'group0/real_only__seed0/weights_selected.pt'}
VALIDATION = RUN / 'group1/mrtgan_k22__seed0/groundtruth.coco.json'
CANDIDATES = ROOT / 'runs/19_mrtgan_detector/local_reexport/candidates/MRTGAN_s12/full/images'
KEY = ROOT / 'runs/18_review_mrtgan/review_key_PRIVATE.json'
REVIEW = ROOT / 'runs/18_review_mrtgan/review/review.csv'
IMG = PROJECT / 'figures/img'
DATA = PROJECT / 'figures/data'
SOURCES = []
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
cands = sorted([k for k in json.loads(src(KEY).read_text()) if k['arm'] == 'MRTGAN_s12'], key=lambda k: k['blind_id'])


def parent_box(pid, size):
    _, xc, yc, w, h = map(float, src(DATASET / manifest[pid]['label']).read_text().split()[:5])
    W, H = size
    return ((xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H)


stats = {}
for det_name, weights in DETECTORS.items():
    net = load(weights)
    per = {'validation': [], 'mrtgan_candidates': []}
    for pid, cls, box in val:
        image = Image.open(src(DATASET / manifest[pid]['image'])).convert('RGB')
        cam, score, top = gradcam(net, image, cls)
        share, hit, area = box_stats(cam, box)
        per['validation'].append(dict(id=pid, label=names[cls + 1], share=share, area=area, peak_in_box=hit, top_class_matches=top == cls))
    for k in cands:
        image = Image.open(src(CANDIDATES / f"balance__{k['parent_id']}.jpg")).convert('RGB')
        cam, score, top = gradcam(net, image, 1)
        share, hit, area = box_stats(cam, parent_box(k['parent_id'], image.size))
        per['mrtgan_candidates'].append(dict(id=k['parent_id'], accepted=rows[k['blind_id']]['accepted'] == 'yes',
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

net = load(DETECTORS['MRT-GAN'])
examples = [v for v in val if v[1] == 0][:2] + [v for v in val if v[1] == 1][:2]
tiles = [(f'val_{names[c + 1]}', Image.open(DATASET / manifest[p]['image']).convert('RGB'), c, b, p) for p, c, b in examples]
for k in [k for k in cands if rows[k['blind_id']]['accepted'] == 'yes'][:2]:
    image = Image.open(CANDIDATES / f"balance__{k['parent_id']}.jpg").convert('RGB')
    tiles.append(('mrtgan_candidate', image, 1, parent_box(k['parent_id'], image.size), k['parent_id']))
tile_meta = []
for j, (kind, image, cls, box, ident) in enumerate(tiles, start=1):
    cam, score, top = gradcam(net, image, cls)
    share, hit, area = box_stats(cam, box)
    plain = image.copy(); ImageDraw.Draw(plain).rectangle(box, outline=ORANGE, width=max(6, image.width // 200))
    crop43(plain, box).resize((480, 360), Image.LANCZOS).save(IMG / f'gradcam_{j}_image.jpg', quality=92)
    crop43(activation(cam), box).resize((480, 360), Image.LANCZOS).save(IMG / f'gradcam_{j}_act.jpg', quality=92)
    crop43(fusion(image, cam), box).resize((480, 360), Image.LANCZOS).save(IMG / f'gradcam_{j}_fusion.jpg', quality=92)
    tile_meta.append(dict(tile=j, kind=kind, id=ident, share=share, box_area_share=area, peak_in_box=hit))
(DATA / 'gradcam_stats.json').write_text(json.dumps(dict(
    method='Grad-CAM on Detect inputs (strides 8/16/32), class score of the top anchor for the annotated class; per-map max-normalised then averaged; letterbox 640; kernel 27 seed-0 selected checkpoints',
    summary={d: s['summary'] for d, s in stats.items()}, tiles=tile_meta, items={d: s['items'] for d, s in stats.items()}), indent=1) + '\n')
(HERE / 'gradcam_source_hashes.json').write_text(json.dumps(
    {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(SOURCES))}, indent=1) + '\n')
print(json.dumps(dict(summary={d: s['summary'] for d, s in stats.items()}, tiles=tile_meta), indent=1))
