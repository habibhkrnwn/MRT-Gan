# R512 pool: same crop windows, same reviewed-pool membership and same GrabCut masks (nearest-upsampled),
# images re-cropped from the original photographs at 512 px. Only resolution changes.
def build_pool512(src, dst, dataset, size=512, require_train=True):
    import csv, json, collections, hashlib
    from pathlib import Path
    import numpy as np, cv2
    from PIL import Image
    from melon_cyclegan import config as cfg
    from melon_cyclegan.crops import crop_with_reflection
    from melon_cyclegan.color import phi_numpy
    from melon_cyclegan.training_data import digest, verify, sha
    src, dst, dataset = Path(src), Path(dst), Path(dataset)
    if dst.exists(): raise ValueError(f'{dst} exists')
    man = json.loads((src/'crops_manifest.json').read_text())
    rows = {r['id']: r for r in csv.DictReader(open(dataset/'manifest.csv'))}
    k = size/float(man.get('size', 256)); records=[]; feats=collections.defaultdict(list); drift=[]
    for r in man['records']:
        row = rows[r['id']]
        if (require_train and row['split']!='train') or sha(dataset/row['image'])!=r['source_sha256']: raise ValueError(f"{r['id']}: source photo differs")
        with Image.open(dataset/row['image']) as h: img=h.convert('RGB')
        left, top, side = r['crop_window']
        crop = cv2.resize(crop_with_reflection(img,left,top,side),(size,size),interpolation=cv2.INTER_AREA)
        m256 = Image.open(src/r['mask']).convert('L')
        mask = np.asarray(m256.resize((size,size),Image.Resampling.NEAREST))
        old = np.asarray(Image.open(src/r['image']).convert('RGB'),np.float64)
        drift.append(np.abs(cv2.resize(crop,old.shape[1::-1],interpolation=cv2.INTER_AREA).astype(np.float64)-old).mean())
        rec = dict(r, crop_box=[float(v)*k for v in r['crop_box']])
        for key, arr in (('image', crop), ('mask', mask)):
            p = dst/r[key]; p.parent.mkdir(parents=True, exist_ok=True); Image.fromarray(arr).save(p); rec[f'{key}_sha256']=sha(p)
        feats[int(r['class_id'])].append(phi_numpy(crop.astype(np.float64)/255.0, mask>127)); records.append(rec)
    if max(drift) > 4.0: raise ValueError(f'512 crops do not match the 256 pool geometry (max mean abs diff {max(drift):.2f})')
    classes = {}
    for cid, v in sorted(feats.items()):
        x=np.stack(v); classes[cfg.CLASS_NAMES[cid]]=dict(class_id=cid,count=len(x),centroid=x.mean(0).tolist(),spread=x.std(0).tolist())
    new = {k2: v for k2, v in man.items() if k2 not in ('records','crops_digest','centroids_sha256')}
    new.update(records=records, size=size, crops_digest=digest(records), source_crops_digest_256=man['crops_digest'],
               note='R512: same windows, members and GrabCut masks as the 256 pool; images re-cropped at 512')
    (dst/'color_centroids.json').write_text(json.dumps(dict(version=cfg.VERSION, phi=man['phi'],
        source='curated train split real crops (R512)', crops_digest=new['crops_digest'], classes=classes), indent=2)+'\n')
    new['centroids_sha256'] = sha(dst/'color_centroids.json')
    (dst/'crops_manifest.json').write_text(json.dumps(new, indent=2)+'\n')
    verify(dst, man['training_policy'])
    return new, dict(max_drift=float(max(drift)), mean_drift=float(np.mean(drift)), n=len(records))


