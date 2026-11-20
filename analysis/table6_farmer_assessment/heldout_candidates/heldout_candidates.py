"""Revision item 4: MRT-GAN seed-12 candidates from HELD-OUT ripe validation photographs.

Frozen generator (EMA G_B of p100_G3TI_R512_seed12, same weights as the reviewed train candidates). Valid crops are
built with the paper's crop rule (crops.build, split='valid': 15% margin, GrabCut seeded by the box, 256 px), then
re-cropped at 512 px with the 256 mask nearest-upsampled, exactly as for the train pool. Only the 62 ripe valid
photographs are translated. No detector, ranking or statistic is fitted on these images. Test split not read.
"""
from pathlib import Path
import json, csv, hashlib, sys, os
PKG = Path(__file__).resolve().parents[4]
CODE = Path(os.environ['MELON_CODE'])
DATA = Path(os.environ['MELON_DATA'])  # official regrouped dataset, manifest sha256 185e1422...
CKPT = PKG/'02_Hasil_Eksperimen/runs/19_mrtgan_detector/local_reexport/checkpoints'
os.environ['MELON_DATASET'] = str(DATA); sys.path.insert(0, str(CODE)); sys.path.insert(0, str(CODE/'tools'))
import numpy as np, cv2, torch
from PIL import Image
from melon_cyclegan import crops, generate, config as cfg
from melon_cyclegan.crops import crop_with_reflection
import generate_balance_candidates as gbc
OUT = Path(__file__).resolve().parent
P256, P512, CAND = OUT/'valid_crops256', OUT/'valid_crops512', OUT/'MRTGAN_s12_heldout'
if CAND.exists(): raise SystemExit(f'{CAND} exists')
torch.set_num_threads(4)
man = crops.build(P256, dataset_root=DATA, size=256, split='valid')
man = json.loads((P256/'crops_manifest.json').read_text())
rows = {r['id']: r for r in csv.DictReader(open(DATA/'manifest.csv'))}
G, meta = generate.load_generators(CKPT, 'p100_G3TI_R512_seed12', device='cpu', weights='ema')
for sub in ('full/images', 'full/labels', 'images', 'masks'): (CAND/sub).mkdir(parents=True)
recs = []
for r in sorted(man['records'], key=lambda r: r['id']):
    if int(r['class_id']) != cfg.RIPE: continue
    row = rows[r['id']]; assert row['split'] == 'valid'
    with Image.open(DATA/row['image']) as h: full = h.convert('RGB')
    left, top, side = r['crop_window']
    c512 = Image.fromarray(cv2.resize(crop_with_reflection(full, left, top, side), (512, 512), interpolation=cv2.INTER_AREA))
    m512 = Image.open(P256/r['mask']).convert('L').resize((512, 512), Image.Resampling.NEAREST)
    (P512/'images').mkdir(parents=True, exist_ok=True); c512.save(P512/'images'/f"{r['id']}.png"); m512.save(P512/'images'/f"{r['id']}_mask.png")
    raw = generate.translate(G['G_B'], c512, 512, device='cpu')
    syn, clip = gbc.paste(full, raw, m512, r['crop_window'])
    cid = f"balance__{r['id']}"
    syn.save(CAND/'full/images'/f'{cid}.jpg', quality=gbc.JPEG_QUALITY)
    (CAND/'full/labels'/f'{cid}.txt').write_text(gbc.flipped_label((DATA/row['label']).read_text(), cfg.UNRIPE))
    syn.crop((left, top, left+side, top+side)).resize((512, 512), Image.Resampling.LANCZOS).save(CAND/'images'/f'{cid}.png')
    m512.save(CAND/'masks'/f'{cid}.png')
    recs.append(dict(candidate_id=cid, parent_id=r['id'], split='valid', mask_source=r['mask_source'], crop_window=r['crop_window']))
    print(r['id'], flush=True)
(CAND/'records.json').write_text(json.dumps(recs, indent=1))
(CAND/'run_summary.json').write_text(json.dumps(dict(protocol=__doc__, generator=meta, candidates=len(recs),
    mask_sources=man.get('mask_sources'), test_used=False), indent=1, default=str))
print('candidates', len(recs))
