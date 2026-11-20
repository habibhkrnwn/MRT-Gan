"""CPU shortcut diagnostics (source DSC/oke vs ripeness). Test split is never opened.

A. Detector probe: frozen detectors classify G3 candidates (green fruit on DSC background).
B. Feature probes: background-only, fruit-only and box-geometry logistic probes, train -> valid,
   plus grouped CV inside train scored on the 16 DSC-unripe photographs.
"""
from pathlib import Path
import csv, json, os, sys
import numpy as np
import torch
from PIL import Image
from ultralytics.data.augment import LetterBox
from ultralytics.nn.tasks import DetectionModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
import torchvision
from skimage.color import rgb2lab, rgb2hsv

# Project root of the original experiment layout (set MRTGAN_PROJECT_ROOT; default: current folder)
G = Path(os.environ.get('MRTGAN_PROJECT_ROOT', '.'))
MAN = Path(os.environ.get('MRTGAN_MANIFEST', 'melon_dataset_regrouped_v4/manifest.csv'))
BASE = G / 'melon_dataset_base_v1'
CAND = G / 'runs/05_texture_study/kaggle_v3_unblind/texture_study_v1/study/p100/candidates/G3/full'
KEY = G / 'runs/05_texture_study/kaggle_v3_unblind/texture_study_v1/study/p100/review_key_PRIVATE.json'
REVIEW = G / 'FINAL_Melon_2026-09-28/02_Generator_G3/review_p100/review.csv'
RANK = G / 'FINAL_Melon_2026-09-28/01_Pipeline_Final/models/ranking.csv'
B7 = G / 'runs/03_balancing_full_train/detector_7_arms/balance_detector/group0'
DET = {f'real_only_s{s}': B7 / f'real_only__seed{s}/weights_selected.pt' for s in (0, 1)}
DET |= {f'oversample_s{s}': B7 / f'oversample_r100__seed{s}/weights_selected.pt' for s in (0, 1)}
DET |= {f'earlier_cyclegan_s{s}': B7 / f'cyclegan_r100__seed{s}/weights_selected.pt' for s in (0, 1)}
DET |= {f'G3_s{s}': G / f'FINAL_Melon_2026-09-28/01_Pipeline_Final/models/seed{s}/weights_selected.pt' for s in range(5)}
OUT = Path(sys.argv[1])
torch.set_grad_enabled(False)

# ---------- manifest (train + valid only) ----------
rows = [r for r in csv.DictReader(open(MAN)) if r['split'] in ('train', 'valid') and r['id'] != 'DSC_0093']
assert all(r['split'] != 'test' for r in rows)
base = {r['id']: r for r in csv.DictReader(open(BASE / 'manifest.csv'))}
for r in rows:
    r['path'] = BASE / base[r['id']]['image']
    r['label_path'] = BASE / base[r['id']]['label']
print('photos', len(rows), {s: sum(r['split'] == s for r in rows) for s in ('train', 'valid')})


def yolo_box(path, w, h):
    c, cx, cy, bw, bh = open(path).read().split()[:5]
    cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
    return int(c), (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)


# ---------- A. detector probe ----------
def load(p):
    ck = torch.load(p, map_location='cpu', weights_only=False)
    net = DetectionModel(ck['cfg'], nc=ck['nc'], verbose=False)
    net.load_state_dict(ck['state_dict'])
    return net.eval()


LB = LetterBox(new_shape=(640, 640), auto=False, stride=32)


def predict(net, path):
    arr = np.asarray(Image.open(path).convert('RGB'))
    x = torch.from_numpy(LB(image=arr)).permute(2, 0, 1).float().unsqueeze(0) / 255
    y = net(x)
    y = y[0] if isinstance(y, (tuple, list)) else y
    sc = y[0, 4:, :]
    a = int(sc.max(0).values.argmax())
    return int(sc[:, a].argmax()), float(sc[:, a].max()), float(sc[0].max()), float(sc[1].max())


key = {k['blind_id']: k for k in json.load(open(KEY))}
acc = {}
for r in csv.DictReader(open(REVIEW)):
    k = key[r['blind_id']]
    if k['arm'] == 'G3':
        acc[k['parent_id']] = r['accepted'] == 'yes'
used = {r['parent_id']: r['used'] == 'True' for r in csv.DictReader(open(RANK))}
cand = sorted(CAND.glob('images/*.jpg'))
assert len(cand) == 217
group = {}
for p in cand:
    pid = p.stem.replace('balance__', '')
    group[p] = 'used_133' if used.get(pid) else ('accepted_unused' if acc[pid] else 'rejected')
byid = {r['id']: r for r in rows}
parents = [byid[p.stem.replace('balance__', '')]['path'] for p in cand]
valid = [r for r in rows if r['split'] == 'valid']
dsc_unripe_train = [r for r in rows if r['split'] == 'train' and r['id'].startswith('DSC') and r['class_id'] == '1']

detA = {}
for name, w in DET.items():
    net = load(w)
    res = {}
    cp = [predict(net, p) for p in cand]
    for g in ('used_133', 'accepted_unused', 'rejected'):
        sel = [c for c, p in zip(cp, cand) if group[p] == g]
        res[f'G3cand_{g}'] = dict(n=len(sel), pred_unripe=sum(c[0] == 1 for c in sel),
                                  conf_ge_025=sum(c[1] >= 0.25 for c in sel),
                                  mean_unripe_score=float(np.mean([c[3] for c in sel])),
                                  mean_ripe_score=float(np.mean([c[2] for c in sel])))
    pp = [predict(net, p) for p in parents]
    res['source_ripe_parents'] = dict(n=len(pp), pred_unripe=sum(c[0] == 1 for c in pp))
    vp = [(r, predict(net, r['path'])) for r in valid]
    res['valid'] = dict(n=len(vp), correct=sum(c[0] == int(r['class_id']) for r, c in vp))
    dp = [predict(net, r['path']) for r in dsc_unripe_train]
    res['train_DSC_unripe_seen_in_training'] = dict(n=len(dp), pred_unripe=sum(c[0] == 1 for c in dp))
    detA[name] = res
    print(name, json.dumps(res), flush=True)

# ---------- B. feature probes ----------
rn = torchvision.models.resnet50()
rn.load_state_dict(torch.load(Path.home() / '.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth'))
rn.fc = torch.nn.Identity(); rn.eval()
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]; STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def colour_feats(px):
    lab = rgb2lab(px[None])[0]; hsv = rgb2hsv(px[None])[0]
    f = [px.mean(0), px.std(0), lab.mean(0), lab.std(0), np.percentile(lab, [10, 50, 90], axis=0).ravel()]
    for i, b in ((0, 16), (1, 8), (2, 8)):
        f.append(np.histogram(hsv[:, i], bins=b, range=(0, 1))[0] / len(hsv))
    return np.concatenate([np.ravel(v) for v in f])


def cnn(img):
    x = torch.from_numpy(np.asarray(img.resize((224, 224)), np.float32) / 255).permute(2, 0, 1)
    return rn(((x - MEAN) / STD)[None])[0].numpy()


F = {k: [] for k in ('bg_colour', 'bg_cnn', 'fruit_colour', 'fruit_cnn', 'geometry')}
for r in rows:
    im = Image.open(r['path']).convert('RGB'); W, H = im.size
    c, (x0, y0, x1, y1) = yolo_box(r['label_path'], W, H)
    assert str(c) == r['class_id'], r['id']
    s = 512 / W; sm = im.resize((512, round(H * s)))
    a = np.asarray(sm, np.float64) / 255
    m = np.zeros(a.shape[:2], bool)
    m[max(0, int(y0 * s)):int(y1 * s) + 1, max(0, int(x0 * s)):int(x1 * s) + 1] = True
    F['bg_colour'].append(colour_feats(a[~m]))
    F['fruit_colour'].append(colour_feats(a[m]))
    bg = a.copy(); bg[m] = 114 / 255
    F['bg_cnn'].append(cnn(Image.fromarray((bg * 255).astype(np.uint8))))
    F['fruit_cnn'].append(cnn(im.crop((x0, y0, x1, y1))))
    F['geometry'].append([(x1 - x0) * (y1 - y0) / (W * H), (x1 - x0) / max(y1 - y0, 1), (x0 + x1) / 2 / W, (y0 + y1) / 2 / H])
F = {k: np.asarray(v) for k, v in F.items()}
y = np.array([int(r['class_id']) for r in rows])
src = np.array([int(r['id'].startswith('oke')) for r in rows])
tr = np.array([r['split'] == 'train' for r in rows]); va = ~tr
grp = np.array([r['photograph_group'] for r in rows])
dscu = np.array([r['id'].startswith('DSC') and r['class_id'] == '1' for r in rows])


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000, class_weight='balanced'))


probes = {}
for k, X in F.items():
    m = clf().fit(X[tr], y[tr])
    auc_valid = roc_auc_score(y[va], m.predict_proba(X[va])[:, 1])
    # grouped CV inside train: label target and source target; score the DSC-unripe photos
    oof_y = np.zeros(tr.sum()); oof_s = np.zeros(tr.sum())
    Xt, yt, st, gt = X[tr], y[tr], src[tr], grp[tr]
    for a, b in GroupKFold(5).split(Xt, yt, gt):
        oof_y[b] = clf().fit(Xt[a], yt[a]).predict_proba(Xt[b])[:, 1]
        oof_s[b] = clf().fit(Xt[a], st[a]).predict_proba(Xt[b])[:, 1]
    d = dscu[tr]; dsc = np.array([not s for s in st], bool)
    probes[k] = dict(
        dims=int(X.shape[1]),
        auc_label_train_to_valid=round(float(auc_valid), 3),
        auc_label_cv_train=round(float(roc_auc_score(yt, oof_y)), 3),
        auc_source_cv_train=round(float(roc_auc_score(st, oof_s)), 3),
        auc_label_cv_within_DSC=round(float(roc_auc_score(yt[dsc], oof_y[dsc])), 3),
        DSC_unripe_called_unripe_cv=f'{int((oof_y[d] >= 0.5).sum())}/{int(d.sum())}',
        oke_called_unripe_cv=f'{int((oof_y[st == 1] >= 0.5).sum())}/{int((st == 1).sum())}',
        DSC_ripe_called_unripe_cv=f'{int((oof_y[dsc & (yt == 0)] >= 0.5).sum())}/{int((dsc & (yt == 0)).sum())}',
    )
    print(k, probes[k], flush=True)

json.dump(dict(detector_probe=detA, feature_probes=probes,
               note='test split never opened; DSC_0093 excluded; valid label == source'),
          open(OUT, 'w'), indent=1)
