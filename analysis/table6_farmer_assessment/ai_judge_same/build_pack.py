"""AI-judge re-review: one blinded pack holding every image of pack_heldout and pack_rater2 (224 unique items).

Every model run gets a fresh copy of the same pack, the same AGENTS.md/CLAUDE.md and the same PROMPT.txt.
Images are re-encoded from pixels only (no PNG text chunks), renamed to fresh blind ids, and listed in a shuffled order.
The key stays here, never inside the pack.
"""
import csv, json, hashlib, random, shutil
from pathlib import Path
from PIL import Image
OUT = Path(__file__).resolve().parent; PKG = OUT.parents[3]
KAPPA = OUT.parent/'kappa'
PACK = PKG/'07_Paket_Review_Rater/AI_Judge_Sama_v1'; KEY = OUT/'key_PRIVATE.json'
FIELDS = ['blind_id', 'target_label', 'texture_realistic', 'shape_ok', 'boundary_ok', 'artifact_free', 'accepted', 'reviewer', 'reason']
if PACK.exists() or KEY.exists(): raise SystemExit('pack exists; never overwrite a review')
items = {}
for name in ('pack_heldout', 'pack_rater2'):
    for k in json.load(open(KAPPA/f'{name}_key_PRIVATE.json')):
        uid = f"{k['origin']}:{k['arm']}:{k['parent_id']}"
        it = items.setdefault(uid, dict(origin=k['origin'], arm=k['arm'], parent_id=k['parent_id'], old_blind_id_session1=k['old_blind_id'],
                                        source=k['source'], source_sha256=k['image_sha256']))
        assert it['source_sha256'] == k['image_sha256'], uid
        it[f'blind_id_{name}'] = k['blind_id']
assert len(items) == 224, len(items)
(PACK/'images').mkdir(parents=True)
for uid, it in items.items():
    b = hashlib.sha256(f'ai_judge_same_v1:{uid}'.encode()).hexdigest()[:20]
    Image.open(it['source']).convert('RGB').save(PACK/'images'/f'{b}.png')
    it.update(uid=uid, blind_id=b, image_sha256=hashlib.sha256((PACK/'images'/f'{b}.png').read_bytes()).hexdigest())
order = sorted(items.values(), key=lambda x: x['blind_id']); random.Random(20261024).shuffle(order)
json.dump(sorted(items.values(), key=lambda x: x['blind_id']), open(KEY, 'w'), indent=1)
with open(PACK/'review.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows({'blind_id': x['blind_id']} for x in order)
print(len(items), {o: sum(x['origin'] == o for x in items.values()) for o in ('kappa', 'heldout_valid', 'train_derived')})
