"""Calibrated two-assessor review (1-5 scale) for the revision.

Calibration set (20 images, excluded from analysis): 5 real unripe and 3 real ripe training crops, 4 MRT-GAN,
4 colour-transfer and 4 GAN-ablation candidates whose first-session ratings came from the farmer. After rating them,
assessors open a key that reveals only image origin (real unripe / real ripe / computer-generated) and agree on the
meaning of each score. Main set (188 images, identical for both assessors, different order): the 148 first-session
items rated by a vision-language model, 20 real unripe and 5 real ripe crops not used in calibration, and 15 of the 148
shown a second time under another id (within-session repeat). Pre-registered acceptance rule for analysis:
target_label == unripe and all four criteria >= 4. Seeds: selection 2026, order A 1, order B 2.
"""
import csv, json, hashlib, random, shutil, html
from pathlib import Path
OUT = Path(__file__).resolve().parent; PKG = OUT.parents[3]
R18 = PKG/'02_Hasil_Eksperimen/runs/18_review_mrtgan'
CROPS = PKG/'02_Hasil_Eksperimen/runs/19_mrtgan_detector/local_reexport/crops512'
DEST = PKG/'07_Paket_Review_Rater/Kalibrasi_Penilaian_v2'
if DEST.exists() or (OUT/'key_PRIVATE.json').exists(): raise SystemExit('packs exist; never overwrite a review')
rng = random.Random(2026)
key18 = {k['blind_id']: k for k in json.load(open(R18/'review_key_PRIVATE.json'))}
rev = {r['blind_id']: r for r in csv.DictReader(open(R18/'review/review.csv'))}
ai = sorted(b for b, r in rev.items() if r['reviewer'] == 'codex_visual_ai'); assert len(ai) == 148
farmer = sorted(b for b, r in rev.items() if r['reviewer'] == 'petani_nilai')
unr = sorted(CROPS.glob('trainA/*.png')); rip = sorted(CROPS.glob('trainB/*.png'))
assert len(unr) == 81 and len(rip) == 217
pick_unr = rng.sample(unr, 25); pick_rip = rng.sample(rip, 8)
def arm_pick(arms, n): return rng.sample([b for b in farmer if key18[b]['arm'] in arms], n)
calib = ([('real_unripe', p) for p in pick_unr[:5]] + [('real_ripe', p) for p in pick_rip[:3]] +
         [('synthetic', R18/'review/images'/f'{b}.png') for b in arm_pick({'MRTGAN_s12'}, 4) + arm_pick({'colour_transfer'}, 4)
          + arm_pick({'initial_config_s12', 'minus_textureD_s12'}, 4)])
main = ([('ai_rated', R18/'review/images'/f'{b}.png', b) for b in ai] + [('real_unripe', p, p.stem) for p in pick_unr[5:]]
        + [('real_ripe', p, p.stem) for p in pick_rip[3:]])
dups = rng.sample(ai, 15); main += [('repeat', R18/'review/images'/f'{b}.png', b) for b in dups]
bid = lambda tag, s: hashlib.sha256(f'calibrated_v2:{tag}:{s}'.encode()).hexdigest()[:16]
TEMPLATE = (OUT/'tool_template.html').read_text()
def tool(folder, pack, title, ids):
    (folder/'review_tool.html').write_text(TEMPLATE.replace('__PACK__', pack).replace('__TITLE__', title).replace('__IDS__', json.dumps(ids)))
key = dict(protocol=__doc__, calibration=[], main=[])
cdir = DEST/'01_Kalibrasi'; (cdir/'images').mkdir(parents=True); cids = []
for j, (origin, src) in enumerate(calib):
    i = bid('calib', f'{j}:{src}'); shutil.copyfile(src, cdir/'images'/f'{i}.png'); cids.append(i)
    key['calibration'].append(dict(image_id=i, origin=origin, source=str(src)))
rng.shuffle(cids); tool(cdir, 'kalibrasi_v2', 'Kalibrasi (20 gambar latihan)', cids)
kdir = cdir/'KUNCI_buka_setelah_selesai'; kdir.mkdir(); (kdir/'images').mkdir()
lab = {'real_unripe': 'FOTO ASLI — belum matang', 'real_ripe': 'FOTO ASLI — matang', 'synthetic': 'BUATAN KOMPUTER'}
cells = ''.join(f'<figure><img src="../images/{i}.png" alt="gambar kalibrasi {n+1}"><figcaption>{n+1}. {html.escape(i)}<br><b>{lab[o["origin"]]}</b></figcaption></figure>'
                for n, (i, o) in enumerate((i, next(x for x in key['calibration'] if x['image_id'] == i)) for i in cids))
(kdir/'kunci_kalibrasi.html').write_text(f'''<!DOCTYPE html><html lang="id"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Kunci Kalibrasi</title>
<style>:root{{--bg:#fff;--ink:#15181d;--muted:#5d6675}}@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#14161a;--ink:#eceff4;--muted:#9aa4b2}}}}:root[data-theme="dark"]{{--bg:#14161a;--ink:#eceff4;--muted:#9aa4b2}}
body{{margin:0;padding:16px;background:var(--bg);color:var(--ink);font:15px/1.4 system-ui,sans-serif}}.g{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}}img{{width:100%;border-radius:6px}}figure{{margin:0}}figcaption{{color:var(--muted);font-size:13px}}figcaption b{{color:var(--ink)}}</style></head>
<body><h1>Kunci kalibrasi</h1><p>Buka hanya setelah KEDUA penilai selesai menilai 20 gambar latihan. Kunci hanya menunjukkan asal gambar, bukan skor yang benar. Diskusikan bersama arti setiap skor 1–5 dengan melihat contoh ini.</p><div class="g">{cells}</div></body></html>''')
order = []
for kind, src, ref in main:
    i = bid('main', f'{kind}:{ref}'); order.append(i); key['main'].append(dict(image_id=i, kind=kind, ref=ref, source=str(src),
        ai_decision=({c: rev[ref][c] for c in ('target_label', 'texture_realistic', 'shape_ok', 'boundary_ok', 'artifact_free', 'accepted')} if kind in ('ai_rated', 'repeat') else None),
        arm=key18[ref]['arm'] if kind in ('ai_rated', 'repeat') else kind, parent_id=key18[ref]['parent_id'] if kind in ('ai_rated', 'repeat') else ref))
assert len(set(order)) == len(order) == 188
for who, seed in (('A', 1), ('B', 2)):
    d = DEST/f'02_Penilaian_Penilai_{who}'; (d/'images').mkdir(parents=True)
    ids = order[:]; r = random.Random(seed)
    while True:
        r.shuffle(ids); pos = {x: n for n, x in enumerate(ids)}
        pairs = [(x['image_id'], next(y['image_id'] for y in key['main'] if y['kind'] == 'ai_rated' and y['ref'] == x['ref'])) for x in key['main'] if x['kind'] == 'repeat']
        if all(abs(pos[a]-pos[b]) >= 40 for a, b in pairs): break
    for x in key['main']: shutil.copyfile(x['source'], d/'images'/f"{x['image_id']}.png")
    tool(d, f'penilaian_v2_{who}', f'Penilaian utama — Penilai {who} (188 gambar)', ids)
    key[f'order_{who}'] = ids
(OUT/'key_PRIVATE.json').write_text(json.dumps(key, indent=1))
print('calibration', len(cids), 'main', len(order))
