"""Revision items 3 and 4: blinded review packs.

pack_rater2 (second rater, item 3 + 4): 100 kappa items from the original 868-item session, stratified by arm (25 each)
  and by the first rater's decision (accepted items first, up to 12 per arm), PLUS every item of pack_heldout.
pack_heldout (first rater = farmer, item 4): 62 held-out MRT-GAN candidates (valid ripe parents) mixed with 62
  train-derived MRT-GAN seed-12 candidates drawn at random (seed 2026) from the 217 already reviewed.
Fresh blind ids per pack; order by hash; no arm/parent/origin shown. Keys are written beside, never inside, the packs.
"""
import csv, json, hashlib, random, shutil
from pathlib import Path
OUT = Path(__file__).resolve().parent; PKG = OUT.parents[3]
R18 = PKG/'02_Hasil_Eksperimen/runs/18_review_mrtgan'
HO = OUT.parent/'heldout_candidates/MRTGAN_s12_heldout/images'
FIELDS = ['blind_id', 'target_label', 'texture_realistic', 'shape_ok', 'boundary_ok', 'artifact_free', 'accepted', 'reviewer', 'reason']
key18 = {k['blind_id']: k for k in json.load(open(R18/'review_key_PRIVATE.json'))}
rev18 = {r['blind_id']: r for r in csv.DictReader(open(R18/'review/review.csv'))}
rng = random.Random(2026)
kappa = []
for arm in sorted({k['arm'] for k in key18.values()}):
    ids = sorted(b for b, k in key18.items() if k['arm'] == arm)
    acc = [b for b in ids if rev18[b]['accepted'] == 'yes']; rej = [b for b in ids if rev18[b]['accepted'] != 'yes']
    a = rng.sample(acc, min(12, len(acc))); kappa += a + rng.sample(rej, 25-len(a))
items = [dict(origin='kappa', arm=key18[b]['arm'], parent_id=key18[b]['parent_id'], old_blind_id=b,
              src=R18/'review/images'/f'{b}.png', rater1_accepted=rev18[b]['accepted']) for b in kappa]
mrt = sorted(b for b, k in key18.items() if k['arm'] == 'MRTGAN_s12')
train_pick = rng.sample(mrt, 62)
held = [dict(origin='heldout_valid', arm='MRTGAN_s12_heldout', parent_id=f.stem.replace('balance__', ''), old_blind_id='', src=f, rater1_accepted='')
        for f in sorted(HO.glob('balance__*.png'))]
assert len(held) == 62
train = [dict(origin='train_derived', arm='MRTGAN_s12', parent_id=key18[b]['parent_id'], old_blind_id=b,
              src=R18/'review/images'/f'{b}.png', rater1_accepted=rev18[b]['accepted']) for b in train_pick]
def write(name, its):
    pack = OUT/name; key = OUT/f'{name}_key_PRIVATE.json'
    if pack.exists() or key.exists(): raise SystemExit(f'{name} exists; never overwrite a review')
    (pack/'images').mkdir(parents=True); seen = {}
    for it in its:
        uid = f"{it['origin']}:{it['arm']}:{it['parent_id']}"
        if uid in seen: continue
        b = hashlib.sha256(f'revision_2026-10:{name}:{uid}'.encode()).hexdigest()[:20]
        shutil.copyfile(it['src'], pack/'images'/f'{b}.png')
        seen[uid] = dict({k: str(v) for k, v in it.items() if k != 'src'}, blind_id=b, source=str(it['src']),
                         image_sha256=hashlib.sha256(Path(it['src']).read_bytes()).hexdigest())
    k = sorted(seen.values(), key=lambda x: x['blind_id']); json.dump(k, open(key, 'w'), indent=1)
    with open(pack/'review.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows({'blind_id': x['blind_id']} for x in k)
    print(name, len(k), {o: sum(x['origin'] == o for x in k) for o in ('kappa', 'heldout_valid', 'train_derived')})
write('pack_heldout', held+train)
write('pack_rater2', items+held+train)
