"""Versioned, train-only texture study. Human review is a mandatory stage boundary."""
from __future__ import annotations
import collections
import csv
import hashlib
import html
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import config as cfg
from .training_data import sha, verify, curate

FRACTIONS = (25, 50, 100)
GEN_ARMS = ('G0', 'G1', 'G2', 'G3')
DET_ARMS = ('real_only', 'oversample', 'geometry', 'synthetic_G0', 'synthetic_improved')
ROOT = Path(__file__).resolve().parents[1]

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def read(path):
    return json.loads(Path(path).read_text())

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def order(values, seed=2026):
    return sorted(values, key=lambda x: (digest([seed, x]), x))

def tool(name):
    if str(ROOT/'tools') not in sys.path: sys.path.insert(0, str(ROOT/'tools'))
    import importlib
    return importlib.import_module(name)

def nested_subsets(rows, seed=2026):
    """Class-composition strata of whole photograph groups; nearest prefix size."""
    train = [r for r in rows if r['split']=='train' and r['id'] not in cfg.EXCLUDED_PARENTS]
    groups = collections.defaultdict(list)
    for row in train: groups[row['photograph_group']].append(row)
    valid_groups = {r['photograph_group'] for r in rows if r['split']=='valid'}
    if set(groups) & valid_groups: raise ValueError('train/validation photograph group overlap')
    strata = collections.defaultdict(list)
    for group, items in groups.items():
        strata[tuple(sorted({int(r['class_id']) for r in items}))].append(group)
    result = {}
    for fraction in FRACTIONS:
        chosen=[]
        for keys in strata.values():
            keys=order(keys,seed); sizes=[len(groups[k]) for k in keys]
            target=sum(sizes)*fraction/100
            prefix=[sum(sizes[:i]) for i in range(1,len(keys)+1)]
            n=min(range(len(prefix)),key=lambda i:(abs(prefix[i]-target),i))+1
            chosen.extend(r for key in keys[:n] for r in groups[key])
        if {int(r['class_id']) for r in chosen}!={0,1}: raise ValueError('subset lacks a class')
        result[fraction]=sorted(chosen,key=lambda r:r['id'])
    return result

def prepare(dataset, crops, out):
    dataset,crops,out=Path(dataset).resolve(),Path(crops).resolve(),Path(out).resolve()
    if out.exists(): raise ValueError('study directory exists; never overwrite a study')
    if sha(dataset/'manifest.csv') != cfg.EXPECTED_MANIFEST: raise ValueError('unfrozen dataset')
    original=verify(crops,'g0x_train_v2')
    with (dataset/'manifest.csv').open() as f: rows=list(csv.DictReader(f))
    subsets=nested_subsets(rows)
    out.mkdir(parents=True)
    protocol=dict(schema=1, fractions=list(FRACTIONS),generator_seed=11,detector_seeds=[0,1,2],
        epochs=20,weights='ema',updates=3000,eval_every=250,batch=8,imgsz=640,lr=.0001,
        bootstrap=1000,bootstrap_seed=2026,split='valid',test_used=False,exploratory=True,
        dataset_manifest_sha256=sha(dataset/'manifest.csv'),source_crops_digest=original['crops_digest'],
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*.py')},subsets={})
    for fraction, selected in subsets.items():
        ids={r['id'] for r in selected}; scope=out/f'p{fraction:03d}'; pool=scope/'crops'
        # Recompute all colour statistics only from the subset's real reviewed crops.
        policy=dict(name='texture_study_v1',source_crops_digest=original['crops_digest'],
            excluded_ids=sorted({r['id'] for r in original['records']}-ids),
            allowed_mask_sources=sorted({r['mask_source'] for r in original['records']}))
        m=curate(crops,pool,policy)
        report=tool('calibrate_color_margin').calibrate(pool)
        write(pool/'color_margin.json',report)
        if report['margin']<=0: raise ValueError(f'nonpositive train-only margin at {fraction}%')
        from .dataset_impl import load_pair
        from .texture import patch_indices
        for r in m['records']:
            _,mask=load_pair(pool/r['image'],pool/r['mask'])
            patch_indices(mask[None],32,16,4,.95,random=False)
        manifest=dict(fraction=fraction,rows=selected,validation=[r for r in rows if r['split']=='valid'],
            counts=dict(collections.Counter(r['class_id'] for r in selected)),
            generator_ids=sorted(r['id'] for r in m['records']),
            generator_excluded_ids=sorted(ids-{r['id'] for r in m['records']}),
            crops_digest=m['crops_digest'],dataset_manifest_sha256=sha(dataset/'manifest.csv'))
        manifest['subset_digest']=digest(manifest)
        write(scope/'subset.json',manifest);protocol['subsets'][str(fraction)]=manifest['subset_digest']
        for arm in GEN_ARMS:
            s=read(ROOT/'config/texture_baseline.json')
            s.update(experiment=f'p{fraction:03d}_{arm}_seed11',comparison_id='texture_study_v1',rung=arm,
                n_epochs=10,n_epochs_decay=10,seed=11,controlled_init_seed=11,texture_seed=11001,
                training_policy='texture_study_v1',expected_crops_digest=m['crops_digest'],
                subset_digest=manifest['subset_digest'],note='20-epoch exploratory study; EMA final primary')
            if arm=='G1': s['netG']='resize_resnet_6blocks'
            if arm in ('G2','G3'):
                s.update(model='melon_texture',lambda_texture=.5,texture_size=32,texture_stride=16,
                    texture_count=4,texture_coverage=.95,texture_gray=int(arm=='G2'),texture_ndf=32)
            write(scope/'configs'/f'{arm}.json',s)
    write(out/'study.json',protocol)
    return protocol

def verify_scope(study, fraction, dataset=None):
    study=Path(study);protocol=read(study/'study.json');scope=study/f'p{fraction:03d}'
    manifest=read(scope/'subset.json');d=manifest.pop('subset_digest')
    if digest(manifest)!=d or protocol['subsets'][str(fraction)]!=d: raise ValueError('subset manifest modified')
    manifest['subset_digest']=d
    pool=verify(scope/'crops','texture_study_v1')
    if pool['crops_digest']!=manifest['crops_digest']:raise ValueError('pool/subset mismatch')
    if set(manifest['generator_ids'])!={r['id'] for r in pool['records']}:raise ValueError('generator source leakage')
    if not set(manifest['generator_ids']) <= {r['id'] for r in manifest['rows']}:raise ValueError('generator outside detector subset')
    if dataset is not None:
        dataset=Path(dataset)
        if sha(dataset/'manifest.csv')!=protocol['dataset_manifest_sha256']:raise ValueError('dataset changed')
        for r in manifest['rows']+manifest['validation']:
            for key in ('image','label'):
                if sha(dataset/r[key])!=r[key+'_sha256']:raise ValueError(f'{key} changed: {r["id"]}')
    return scope,manifest

def run_generators(study,fraction,arms,budget_minutes):
    scope,m=verify_scope(study,fraction)
    if budget_minutes <= 0: raise ValueError('positive per-run budget required')
    protocol=read(Path(study)/'study.json')
    for path,h in protocol['code_sha256'].items():
        if not (ROOT/path).exists() or sha(ROOT/path)!=h:raise ValueError('code changed since prepare; create a new study')
    for arm in arms:
        if arm not in GEN_ARMS:raise ValueError('unknown generator arm')
        if fraction!=100:
            chosen=read(Path(study)/'selection.json')['winner']
            if arm not in ('G0',chosen):raise ValueError('only G0 and selected model on small subsets')
        s=read(scope/'configs'/f'{arm}.json')
        if s['subset_digest']!=m['subset_digest'] or s['expected_crops_digest']!=m['crops_digest']:raise ValueError('config scope mismatch')
        start=time.time()
        command=[sys.executable,str(ROOT/'tools/train_cyclegan.py'),'--config',str(scope/'configs'/f'{arm}.json'),
            '--dataroot',str(scope/'crops'),'--checkpoints-dir',str(scope/'checkpoints'),
            '--','--time_budget_minutes',str(budget_minutes)]
        scope.joinpath('logs').mkdir(exist_ok=True)
        with (scope/'logs'/f'{arm}.log').open('x') as log:
            proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in proc.stdout: print(line,end='',flush=True);log.write(line);log.flush()
            rc=proc.wait()
        write(scope/'logs'/f'{arm}_execution.json',dict(seconds=time.time()-start,returncode=rc,budget_minutes=budget_minutes))
        if rc:raise RuntimeError(f'{arm} training failed')
        loss=read(scope/'checkpoints'/s['experiment']/'loss.json')
        if loss['stopped'] is not None or loss['epochs']!=s['n_epochs']+s['n_epochs_decay']:raise RuntimeError('partial run; do not export')

def export_candidates(study,fraction,arms,dataset,device='cpu'):
    scope,m=verify_scope(study,fraction,dataset)
    for arm in arms:
        s=read(scope/'configs'/f'{arm}.json');folder=scope/'checkpoints'/s['experiment']
        identity=read(folder/'melon_identity.json')
        # Config plus crop binding checked against checkpoint metadata before export.
        trained=identity.get('config',{})
        if trained.get('subset_digest')!=m['subset_digest']:raise ValueError('checkpoint belongs to another subset')
        loss=read(folder/'loss.json')
        if loss['stopped'] is not None or loss['epochs']!=s['n_epochs']+s['n_epochs_decay']:raise ValueError('incomplete generator')
        tool('generate_balance_candidates').run(scope/'checkpoints',s['experiment'],'ema',scope/'crops',
            scope/'candidates'/arm,dataset,device)
        write(scope/'candidates'/arm/'binding.json',dict(arm=arm,subset_digest=m['subset_digest'],
            generator_sha256=sha(folder/'latest_net_G_B_ema.pth'),config_sha256=sha(scope/'configs'/f'{arm}.json'),weights='ema'))

def blind_review(study,fraction,arms):
    scope,m=verify_scope(study,fraction);private=[]
    public=scope/'review';public.mkdir(exist_ok=False);(public/'images').mkdir()
    for arm in arms:
        bundle=scope/'candidates'/arm;binding=read(bundle/'binding.json')
        if binding['subset_digest']!=m['subset_digest']:raise ValueError('wrong candidate scope')
        for record in read(bundle/'records.json'):
            key=digest([m['subset_digest'],arm,record['candidate_id']])[:20]
            shutil.copyfile(bundle/record['image'],public/'images'/f'{key}.png')
            private.append(dict(blind_id=key,arm=arm,candidate_id=record['candidate_id'],parent_id=record['parent_id'],
                image_sha256=sha(bundle/record['full_image']),label_sha256=sha(bundle/record['full_label'])))
    if len({r['blind_id'] for r in private})!=len(private):raise ValueError('blind ID collision')
    private.sort(key=lambda r:r['blind_id']);write(scope/'review_key_PRIVATE.json',private)
    fields=['blind_id','target_label','texture_realistic','shape_ok','boundary_ok','artifact_free','accepted','reviewer','reason']
    with (public/'review.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({'blind_id':r['blind_id']} for r in private)
    body=''.join(f'<figure><img width="384" src="images/{r["blind_id"]}.png"><figcaption>{r["blind_id"]}</figcaption></figure>' for r in private)
    (public/'index.html').write_text('<meta charset="utf-8"><h1>Review citra melon</h1><p>Isi CSV: target_label ripe/unripe/ambiguous; kriteria yes/no; accepted yes/no; reviewer wajib. Jangan melihat key PRIVATE.</p>'+body)
    return public

def reviewed(study,fraction):
    scope,m=verify_scope(study,fraction);key=read(scope/'review_key_PRIVATE.json')
    with (scope/'review/review.csv').open() as f:rows=list(csv.DictReader(f))
    if len(rows)!=len(key) or {r['blind_id'] for r in rows}!={r['blind_id'] for r in key}:raise ValueError('missing/duplicate/unknown reviews')
    byid={r['blind_id']:r for r in rows};out=collections.defaultdict(list)
    for item in key:
        r=byid[item['blind_id']]
        if r['accepted'] not in ('yes','no') or not r['reviewer'].strip():raise ValueError('human review incomplete')
        if r['target_label'] not in ('ripe','unripe','ambiguous'):raise ValueError('label review incomplete')
        if any(r[k] not in ('yes','no') for k in ('texture_realistic','shape_ok','boundary_ok','artifact_free')):raise ValueError('criteria incomplete')
        if r['accepted']=='yes' and (r['target_label']!='unripe' or any(r[k]!='yes' for k in ('texture_realistic','shape_ok','boundary_ok','artifact_free'))):raise ValueError('accepted contradicts criteria')
        bundle=scope/'candidates'/item['arm'];record=next(x for x in read(bundle/'records.json') if x['candidate_id']==item['candidate_id'])
        if sha(bundle/record['full_image'])!=item['image_sha256'] or sha(bundle/record['full_label'])!=item['label_sha256']:raise ValueError('candidate changed after review')
        if record['parent_id'] not in {x['id'] for x in m['rows']}:raise ValueError('candidate parent outside subset')
        label=(bundle/record['full_label']).read_text().split()
        if len(label)!=5 or label[0]!='1':raise ValueError('synthetic label must be one unripe bbox')
        if len({x['candidate_id'] for x in out[item['arm']]})!=len(out[item['arm']]):raise ValueError('duplicate candidate')
        out[item['arm']].append(dict(record,accepted=r['accepted']=='yes',reviewer=r['reviewer']))
    return out

def select(study):
    study=Path(study)
    if (study/'selection.json').exists():raise ValueError('selection already locked')
    groups=reviewed(study,100);scores={}
    for arm in ('G1','G3'):
        rs=groups[arm];metrics=read(study/'p100/diagnostics'/f'{arm}.json')
        if metrics['subset_digest']!=read(study/'p100/subset.json')['subset_digest']:raise ValueError('diagnostic scope mismatch')
        binding=read(study/'p100/candidates'/arm/'binding.json')
        if metrics['generator_sha256']!=binding['generator_sha256']:raise ValueError('diagnostics from different generator')
        scores[arm]=dict(accepted=sum(r['accepted'] for r in rs),total=len(rs),
            rate=sum(r['accepted'] for r in rs)/len(rs),chroma=metrics['synthetic']['chroma_energy'])
    if max(s['accepted'] for s in scores.values())==0:raise ValueError('no viable candidate; stop synthetic detector stage')
    winner=min(scores,key=lambda a:(-scores[a]['rate'],scores[a]['chroma'],a))
    result=dict(winner=winner,scores=scores,review_sha256=sha(study/'p100/review/review.csv'),
        rule='acceptance rate descending; chroma energy ascending; G1 tie',test_used=False)
    write(study/'selection.json',result);return result

def build_detectors(study,fraction,dataset):
    scope,m=verify_scope(study,fraction,dataset);dataset=Path(dataset);winner=read(Path(study)/'selection.json')['winner']
    selection=read(Path(study)/'selection.json')
    if selection['review_sha256']!=sha(Path(study)/'p100/review/review.csv'):raise ValueError('selection review changed')
    groups=reviewed(study,fraction);accepted={a:sorted((r for r in groups[a] if r['accepted']),key=lambda r:digest(r['candidate_id'])) for a in ('G0',winner)}
    counts=collections.Counter(int(r['class_id']) for r in m['rows']);target=max(0,counts[0]-counts[1])
    k=min(target,len(accepted['G0']),len(accepted[winner]))
    if k<=0:raise ValueError('no common accepted additions; no synthetic comparison')
    out=scope/'detector_arms'
    if out.exists():raise ValueError('detector arms already exist')
    real={r['id']:r for r in m['rows']};minority=order([r['id'] for r in m['rows'] if int(r['class_id'])==1])
    generated=scope/'geometry_files';generated.mkdir()
    from PIL import Image,ImageOps
    summaries={}
    for arm in DET_ARMS:
        additions=[]
        for i in range(k if arm!='real_only' else 0):
            if arm.startswith('synthetic'):
                a='G0' if arm=='synthetic_G0' else winner;r=accepted[a][i];bundle=scope/'candidates'/a
                if r['parent_id'] not in real:raise ValueError('synthetic parent outside subset')
                additions.append((f'add_{i:04d}',bundle/r['full_image'],(bundle/r['full_label']).read_text(),r['parent_id']))
            else:
                pid=minority[i%len(minority)];r=real[pid];image=dataset/r['image'];label=(dataset/r['label']).read_text()
                if arm=='geometry':
                    # Mirror alternates horizontal/vertical; exact bbox transforms, full-resolution.
                    horizontal=(i//len(minority))%2==0
                    with Image.open(image) as im: im=ImageOps.mirror(im) if horizontal else ImageOps.flip(im);dest=generated/f'{i:04d}.png';im.save(dest)
                    parts=label.split();parts[1 if horizontal else 2]=str(1-float(parts[1 if horizontal else 2]));label=' '.join(parts)+'\n';image=dest
                additions.append((f'add_{i:04d}',image,label,pid))
        manifest=tool('build_balance_arms').write_arm(out,arm,m['rows']+m['validation'],dataset,additions)
        manifest.update(subset_digest=m['subset_digest'],fraction=fraction,selected_generator=winner,
            review_sha256=sha(scope/'review/review.csv'),target_additions=target,common_additions=k,
            full_balance=k==target,exploratory=True)
        manifest['files_sha256']={str(p.relative_to(out/arm)):sha(p) for folder in ('images','labels') for p in (out/arm/folder).rglob('*') if p.is_file()}
        write(out/arm/'arm_manifest.json',manifest);summaries[arm]=manifest['counts']
    write(out/'arms_summary.json',dict(arms=summaries,subset_digest=m['subset_digest'],common_additions=k,target_additions=target,test_used=False))
    return summaries

def run_detectors(study,fraction):
    scope,m=verify_scope(study,fraction)
    for arm in DET_ARMS:
        root=scope/'detector_arms'/arm;manifest=read(root/'arm_manifest.json')
        if manifest['subset_digest']!=m['subset_digest'] or manifest['review_sha256']!=sha(scope/'review/review.csv'):raise ValueError('arm/review binding mismatch')
        current={str(p.relative_to(root)) for folder in ('images','labels') for p in (root/folder).rglob('*') if p.is_file()}
        if current!=set(manifest['files_sha256']):raise ValueError('extra/missing detector files')
        for path,h in manifest['files_sha256'].items():
            if sha(root/path)!=h:raise ValueError('detector files changed')
        # Kaggle mounts move: regenerate only the location field before consuming data.
        (root/'data.yaml').write_text(f'path: {root.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: ripe\n  1: unripe\n')
    if (scope/'detectors').exists():raise ValueError('detector output exists; no silent resume')
    command=[sys.executable,str(ROOT/'tools/train_detector_arms.py'),'--arms-dir',str(scope/'detector_arms'),
        '--arms',*DET_ARMS,'--out',str(scope/'detectors'),'--seeds','0','1','2',
        '--updates','3000','--batch','8','--lr','0.0001','--eval-every','250','--export']
    subprocess.run(command,check=True)
