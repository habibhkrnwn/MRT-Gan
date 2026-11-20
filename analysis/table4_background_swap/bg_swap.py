"""Revision item 2C: background-swap counterfactual on the 87 validation photographs.

Each real validation fruit (GrabCut mask, 5 iterations, seeded by its box) is pasted onto the background of a
validation photograph of the OPPOSITE class (swap) and, as a paste-artefact control, of a different photograph of the
SAME class (control). The donor's own fruit is removed first by inpainting its dilated box. The pasted fruit is scaled
to the donor box area (aspect kept) and centred on the donor box. Label = class of the pasted fruit.
Selected checkpoints of all four arms of kernel 27 (5 seeds each) are run on original / control / swap images.
CPU only; test split not read.
"""
from pathlib import Path
import csv, json, hashlib, time, sys
import numpy as np, cv2, torch
from PIL import Image
from ultralytics.data.augment import LetterBox
from ultralytics.nn.tasks import DetectionModel

OUT = Path(__file__).resolve().parent
PKG = OUT.parents[3]
DATA = PKG/'02_Hasil_Eksperimen/runs/19_mrtgan_detector/local_reexport/dataset_v4'
RUN = PKG/'02_Hasil_Eksperimen/runs/19_mrtgan_detector/mrtgan_detector_v1/result'
ARMS = {'real_only': 'group0', 'oversample_k22': 'group0', 'colour_transfer_k22': 'group1', 'mrtgan_k22': 'group1'}
torch.set_num_threads(4); torch.set_grad_enabled(False)
rows = sorted([r for r in csv.DictReader((DATA/'manifest.csv').open()) if r['split'] == 'valid'], key=lambda r: r['id'])
assert len(rows) == 87
h = lambda s: hashlib.sha256(s.encode()).hexdigest()

def load(r):
    im = np.asarray(Image.open(DATA/r['image']).convert('RGB')); H, W = im.shape[:2]
    c, cx, cy, bw, bh = map(float, (DATA/r['label']).read_text().split()); assert int(c) == int(r['class_id'])
    box = np.array([(cx-bw/2)*W, (cy-bh/2)*H, (cx+bw/2)*W, (cy+bh/2)*H])
    return im, box

def grabcut(im, box):
    x0, y0, x1, y1 = box; m = 0.15*max(x1-x0, y1-y0)
    L, T = int(max(0, x0-m)), int(max(0, y0-m)); R, B = int(min(im.shape[1], x1+m)), int(min(im.shape[0], y1+m))
    crop = im[T:B, L:R]; s = 256/max(crop.shape[:2]); small = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    rect = (int((x0-L)*s), int((y0-T)*s), max(1, int((x1-x0)*s)), max(1, int((y1-y0)*s)))
    mask = np.zeros(small.shape[:2], np.uint8); bg = np.zeros((1, 65)); fg = np.zeros((1, 65))
    cv2.grabCut(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), mask, rect, bg, fg, 5, cv2.GC_INIT_WITH_RECT)
    fgm = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    area = fgm.sum()/(rect[2]*rect[3])
    if not 0.05 <= area <= 0.98: fgm = np.zeros_like(fgm)  # same rejection rule as the paper; fallback below
    full = cv2.resize(fgm, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
    bi = np.array([x0-L, y0-T, x1-L, y1-T]).round().astype(int)
    return crop, full, bi, bool(fgm.any())

def remove_fruit(im, box):
    H, W = im.shape[:2]; x0, y0, x1, y1 = box; dx, dy = 0.1*(x1-x0), 0.1*(y1-y0)
    m = np.zeros((H, W), np.uint8); m[int(max(0, y0-dy)):int(min(H, y1+dy)), int(max(0, x0-dx)):int(min(W, x1+dx))] = 255
    s = 0.25; sm = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA); mm = cv2.resize(m, (sm.shape[1], sm.shape[0]), interpolation=cv2.INTER_NEAREST)
    inp = cv2.inpaint(cv2.cvtColor(sm, cv2.COLOR_RGB2BGR), mm, 9, cv2.INPAINT_TELEA)
    up = cv2.cvtColor(cv2.resize(inp, (W, H), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
    out = im.copy(); out[m > 0] = up[m > 0]; return out

def paste(fruit, donor):
    (crop, mask, bi, ok), (dim, dbox) = fruit, donor
    fx0, fy0, fx1, fy1 = bi; fw, fh = fx1-fx0, fy1-fy0
    if not ok: mask = np.zeros_like(mask); mask[fy0:fy1, fx0:fx1] = 1  # rectangle if GrabCut rejected
    dw, dh = dbox[2]-dbox[0], dbox[3]-dbox[1]; s = np.sqrt(dw*dh/(fw*fh))
    c = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    mk = cv2.resize(mask, (c.shape[1], c.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    mk = cv2.GaussianBlur(mk, (0, 0), 1.5)[..., None]
    b = (bi*s).round().astype(int); cx, cy = (dbox[0]+dbox[2])/2, (dbox[1]+dbox[3])/2
    ox, oy = int(round(cx-(b[0]+b[2])/2)), int(round(cy-(b[1]+b[3])/2))
    H, W = dim.shape[:2]; out = dim.astype(np.float32).copy()
    X0, Y0 = max(0, ox), max(0, oy); X1, Y1 = min(W, ox+c.shape[1]), min(H, oy+c.shape[0])
    sub = c[Y0-oy:Y1-oy, X0-ox:X1-ox].astype(np.float32); a = mk[Y0-oy:Y1-oy, X0-ox:X1-ox]
    out[Y0:Y1, X0:X1] = a*sub+(1-a)*out[Y0:Y1, X0:X1]
    nb = np.array([b[0]+ox, b[1]+oy, b[2]+ox, b[3]+oy], float); nb[[0, 2]] = nb[[0, 2]].clip(0, W); nb[[1, 3]] = nb[[1, 3]].clip(0, H)
    return out.round().astype(np.uint8), nb

t0 = time.time()
photos = {r['id']: load(r) for r in rows}
fruits = {r['id']: grabcut(*photos[r['id']]) for r in rows}
clean = {r['id']: remove_fruit(*photos[r['id']]) for r in rows}
by_class = {c: sorted([r['id'] for r in rows if int(r['class_id']) == c], key=h) for c in (0, 1)}
cases = []
for r in rows:
    i, c = r['id'], int(r['class_id'])
    opp = by_class[1-c]; same = [x for x in by_class[c] if x != i]
    k = int(h(i), 16)
    for mode, donor in (('swap', opp[k % len(opp)]), ('control', same[k % len(same)])):
        img, box = paste(fruits[i], (clean[donor], photos[donor][1]))
        cases.append(dict(id=i, mode=mode, donor=donor, gt_class=c, img=img, box=box))
    cases.append(dict(id=i, mode='original', donor=i, gt_class=c, img=photos[i][0], box=photos[i][1]))
ex = OUT/'examples'; ex.mkdir(exist_ok=True)
for cs in cases:
    if cs['id'] in (by_class[0][0], by_class[1][0]):
        Image.fromarray(cs['img']).resize((640, int(640*cs['img'].shape[0]/cs['img'].shape[1]))).save(ex/f"{cs['id']}__{cs['mode']}.jpg", quality=90)
print('built', len(cases), 'images; grabcut ok', sum(f[3] for f in fruits.values()), '/87; s', round(time.time()-t0), flush=True)

lb = LetterBox(new_shape=(640, 640), auto=False, stride=32)
def tensor(cs):
    H, W = cs['img'].shape[:2]; s = min(640/W, 640/H); nw, nh = round(W*s), round(H*s)
    left, top = round((640-nw)/2-.1), round((640-nh)/2-.1); b = cs['box']
    gt = np.array([b[0]*s+left, b[1]*s+top, b[2]*s+left, b[3]*s+top])
    return torch.from_numpy(lb(image=cs['img'])).permute(2, 0, 1).float().unsqueeze(0)/255, gt
inputs = [(cs, *tensor(cs)) for cs in cases]

def summarize(rr):
    loc = lambda x: x['class_correct'] and x['score'] >= .25 and x['iou_gt'] >= .5
    return dict(n=len(rr), top_class_correct=sum(x['class_correct'] for x in rr), localized_correct=sum(loc(x) for x in rr),
                follows_background=sum((not x['class_correct']) for x in rr),
                per_class={str(c): dict(n=sum(x['gt_class'] == c for x in rr), top_correct=sum(x['gt_class'] == c and x['class_correct'] for x in rr),
                                        localized_correct=sum(x['gt_class'] == c and loc(x) for x in rr)) for c in (0, 1)})
result = dict(protocol=__doc__, n_photos=87, grabcut_ok=int(sum(f[3] for f in fruits.values())), runs={})
for arm, g in ARMS.items():
    for seed in range(5):
        run = RUN/g/f'{arm}__seed{seed}'; w = run/'weights_selected.pt'
        ck = torch.load(w, map_location='cpu', weights_only=False)
        net = DetectionModel(ck['cfg'], nc=ck['nc'], verbose=False); net.load_state_dict(ck['state_dict']); net.eval()
        det = {m: [] for m in ('original', 'control', 'swap')}
        for cs, x, gt in inputs:
            raw = net(x); o = raw[0] if isinstance(raw, (list, tuple)) else raw
            sc = o[0, 4:]; a = int(sc.max(0).values.argmax()); c = int(sc[:, a].argmax()); score = float(sc[c, a])
            bb = o[0, :4, a].numpy(); p = np.array([bb[0]-bb[2]/2, bb[1]-bb[3]/2, bb[0]+bb[2]/2, bb[1]+bb[3]/2])
            inter = np.maximum(0, np.minimum(p[2:], gt[2:])-np.maximum(p[:2], gt[:2])).prod()
            iou = inter/((p[2:]-p[:2]).prod()+(gt[2:]-gt[:2]).prod()-inter)
            det[cs['mode']].append(dict(id=cs['id'], donor=cs['donor'], gt_class=cs['gt_class'], pred_class=c, class_correct=c == cs['gt_class'], score=score, iou_gt=float(iou)))
        result['runs'][f'{arm}@{seed}'] = dict(summary={m: summarize(v) for m, v in det.items()}, details=det,
                                              weights_sha256=hashlib.sha256(w.read_bytes()).hexdigest())
        (OUT/'bg_swap_results.json').write_text(json.dumps(result, indent=1))
        print(arm, seed, {m: (s['top_class_correct'], s['localized_correct']) for m, s in result['runs'][f'{arm}@{seed}']['summary'].items()}, round(time.time()-t0), flush=True)
print('DONE', round(time.time()-t0))
