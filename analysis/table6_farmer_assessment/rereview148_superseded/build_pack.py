"""Blinded re-review by the farmer of the 148 first-session items rated by a vision-language model (reviewer
`codex_visual_ai`, rows 721-868 of runs/18_review_mrtgan/review/review.csv). Fresh blind ids, hash order; the
farmer sees no arm, parent, earlier decision or reason. Key written beside, never inside, the pack."""
import csv, json, hashlib, shutil
from pathlib import Path
OUT = Path(__file__).resolve().parent; PKG = OUT.parents[3]
R18 = PKG/'02_Hasil_Eksperimen/runs/18_review_mrtgan'
FIELDS = ['blind_id', 'target_label', 'texture_realistic', 'shape_ok', 'boundary_ok', 'artifact_free', 'accepted', 'reviewer', 'reason']
key18 = {k['blind_id']: k for k in json.load(open(R18/'review_key_PRIVATE.json'))}
rev = [r for r in csv.DictReader(open(R18/'review/review.csv')) if r['reviewer'] == 'codex_visual_ai']
assert len(rev) == 148
pack, keyf = OUT/'pack_rereview148', OUT/'pack_rereview148_key_PRIVATE.json'
if pack.exists() or keyf.exists(): raise SystemExit('pack exists; never overwrite a review')
(pack/'images').mkdir(parents=True); key = []
for r in rev:
    k = key18[r['blind_id']]; b = hashlib.sha256(f"rereview148_2026-10-23:{r['blind_id']}".encode()).hexdigest()[:20]
    shutil.copyfile(R18/'review/images'/f"{r['blind_id']}.png", pack/'images'/f'{b}.png')
    key.append(dict(blind_id=b, old_blind_id=r['blind_id'], arm=k['arm'], parent_id=k['parent_id'], ai_accepted=r['accepted']))
key.sort(key=lambda x: x['blind_id']); json.dump(key, open(keyf, 'w'), indent=1)
with open(pack/'review.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows({'blind_id': x['blind_id']} for x in key)
print(len(key))
