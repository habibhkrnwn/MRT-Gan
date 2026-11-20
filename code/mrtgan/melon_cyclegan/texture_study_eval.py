"""Reproducible diagnostics and exploratory validation-only detector statistics."""
from __future__ import annotations
import collections
import csv
import json
import statistics
from pathlib import Path
import numpy as np
from .texture_study import read, write, verify_scope, sha, DET_ARMS


def spectral_share(patch):
    x=patch-patch.mean();power=np.abs(np.fft.fftshift(np.fft.fft2(x)))**2
    return float((power[0].sum()+power[:,0].sum()-power[0,0])/power.sum()) if power.sum()>1e-12 else 0.


def image_features(rgb, mask, positions):
    from scipy.ndimage import binary_erosion
    from skimage.color import rgb2lab
    from skimage.feature import local_binary_pattern
    inner=binary_erosion(mask,iterations=6)
    if not inner.any():raise ValueError('empty eroded ROI')
    lum=rgb@np.array([.299,.587,.114]);chroma=rgb-lum[...,None]
    def energy(arr):
        if arr.ndim==2:arr=arr[...,None]
        v=inner[1:]&inner[:-1];h=inner[:,1:]&inner[:,:-1]
        if not v.any() or not h.any():raise ValueError('empty ROI adjacent pairs')
        return float((np.abs(np.diff(arr,axis=0))[v].mean()+np.abs(np.diff(arr,axis=1))[h].mean())/2)
    channels=[lum,chroma[...,0],chroma[...,2]]
    spectra=[np.mean([spectral_share(x[y:y+32,z:z+32]) for y,z in positions]) for x in channels]
    lab=rgb2lab(rgb);ab=lab[...,1:][inner].mean(0)
    lbp=local_binary_pattern(np.rint(lum*255).clip(0,255).astype(np.uint8),8,1,'uniform')
    hist=np.bincount(lbp[inner].astype(int),minlength=10).astype(float);hist/=hist.sum()
    boundary=mask&~inner
    return dict(nyquist_luma=float(spectra[0]),nyquist_r_minus_luma=float(spectra[1]),
        nyquist_b_minus_luma=float(spectra[2]),luma_energy=energy(lum),chroma_energy=energy(chroma),
        red_share=float((lab[...,1][inner]>10).mean()),ab=ab.tolist(),lbp=hist.tolist(),
        boundary_red_share=float((lab[...,1][boundary]>10).mean()))


def summarize(rows,reference):
    keys=['nyquist_luma','nyquist_r_minus_luma','nyquist_b_minus_luma','luma_energy','chroma_energy','red_share','boundary_red_share']
    result={k:float(np.mean([r[k] for r in rows])) for k in keys};result['n']=len(rows)
    h=np.mean([r['lbp'] for r in rows],axis=0);rh=np.mean([r['lbp'] for r in reference],axis=0)
    result['lbp_chi2']=float(.5*np.sum((h-rh)**2/(h+rh+1e-12)))
    ab=np.array([r['ab'] for r in rows]);ref=np.mean([r['ab'] for r in reference],axis=0)
    result['mean_ab_distance']=float(np.linalg.norm(ab-ref,axis=1).mean())
    result['ab_diversity_rms']=float(np.sqrt(np.mean(np.sum((ab-ab.mean(0))**2,axis=1))))
    result['red_flag_count']=sum(r['red_share']>.05 for r in rows)
    return result


def diagnostics(study,fraction,arms,device='cpu'):
    import torch
    from .dataset_impl import MelonRoiData,load_pair
    from .texture import patch_indices
    from .generate import load_generators
    from .model_impl import composite
    scope,m=verify_scope(study,fraction);data=MelonRoiData(scope/'crops',no_flip=True)
    def sample(path,domain):
        x,mask=load_pair(path,data.mask_for(path,domain));ix=patch_indices(mask[None],32,16,4,.95,random=False)[0].tolist()
        cols=(x.shape[-1]-32)//16+1;positions=[(i//cols*16,i%cols*16) for i in ix]
        return x,mask,positions
    def feature(x,mask,pos):
        rgb=((x.detach().cpu().permute(1,2,0).numpy()+1)/2).clip(0,1)
        return image_features(rgb,mask[0].numpy()>.5,pos)
    reference=[feature(*sample(p,'A')) for p in data.A_paths]
    for arm in arms:
        settings=read(scope/'configs'/f'{arm}.json');folder=scope/'checkpoints'/settings['experiment']
        if read(folder/'melon_identity.json')['config']['subset_digest']!=m['subset_digest']:raise ValueError('wrong checkpoint subset')
        target=scope/'diagnostics'/f'{arm}.json'
        if target.exists():raise ValueError('diagnostics already exist')
        nets,meta=load_generators(scope/'checkpoints',settings['experiment'],device=device,weights='ema')
        rows=[]
        with torch.inference_mode():
            for path in data.B_paths:
                x,mask,pos=sample(path,'B');xx=x[None].to(device);mm=mask[None].to(device)
                fake=composite(nets['G_B'](xx),xx,mm)[0]
                rows.append(dict(id=path.stem,**feature(fake,mask,pos)))
        write(target,dict(subset_digest=m['subset_digest'],generator_sha256=sha(folder/'latest_net_G_B_ema.pth'),
            weights='ema',split='train',test_used=False,acceptance_metric=False,
            method='RGB in [0,1]; erode6 adjacent pair energy; four first eligible native 32px patches, stride16 coverage0.95; FFT Nyquist row/column fraction without window',
            synthetic=summarize(rows,reference),real_unripe=summarize(reference,reference),per_image=rows))


def grouped_draws(image_groups, resamples=1000, seed=2026):
    """Resample photograph clusters, retaining multiplicity and both classes."""
    groups=collections.defaultdict(list)
    for image,group in image_groups.items():groups[group].append(image)
    keys=sorted(groups);rng=np.random.default_rng(seed)
    for _ in range(resamples):
        yield [image for group in rng.choice(keys,len(keys),replace=True) for image in groups[group]]


def operating_recall(per_image, ids, threshold, cls=1):
    from . import detection_metrics as dm
    tp=fp=n=0
    for i in ids:
        v=per_image[i];keep=v['conf']>=threshold
        classes=v['pred_cls'][keep]
        correct=dm.match(classes,v['target_cls'],dm._iou(v['pred_boxes'][keep],v['gt_boxes']))[:,0]
        tp+=int((correct & (classes==cls)).sum());fp+=int((~correct & (classes==cls)).sum())
        n+=int((v['target_cls']==cls).sum())
    return dict(recall=tp/n if n else None,precision=tp/(tp+fp) if tp+fp else 0.,f1=2*tp/(n+tp+fp) if n+tp+fp else 0.,tp=tp,fp=fp,targets=n)


def evaluate(study,fraction,device='cpu',resamples=1000):
    """Re-evaluate SELECTED weights, never accidentally score final weights instead."""
    import torch
    from PIL import Image
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    from . import detection_metrics as dm
    scope,m=verify_scope(study,fraction);out=scope/'evaluation';out.mkdir(exist_ok=False)
    gt=dict(images=[],annotations=[]);image_groups={}
    real=scope/'detector_arms/real_only'
    for r in m['validation']:
        image_id=r['id'];image_groups[image_id]=r['photograph_group'];gt['images'].append(dict(id=image_id))
        with Image.open(real/'images/val'/f'{image_id}.jpg') as im:w,h=im.size
        for line in (real/'labels/val'/f'{image_id}.txt').read_text().splitlines():
            c,x,y,bw,bh=map(float,line.split());gt['annotations'].append(dict(image_id=image_id,category_id=int(c)+1,bbox=[(x-bw/2)*w,(y-bh/2)*h,bw*w,bh*h]))
    tables={};points={};provenance={};ids=sorted(image_groups)
    for arm in DET_ARMS:
        for seed in (0,1,2):
            key=f'{arm}@{seed}';run=scope/'detectors'/f'{arm}__seed{seed}';summary=read(run/'run_summary.json')
            selected=summary['selected'];weights=run/selected['weights']
            checkpoint=torch.load(weights,map_location='cpu',weights_only=False)
            net=DetectionModel(checkpoint['cfg'],nc=2,verbose=False);net.load_state_dict(checkpoint['state_dict'])
            wrapper=YOLO(checkpoint['cfg']);wrapper.model=net.eval()
            metrics=wrapper.val(data=str(real/'data.yaml'),split='val',imgsz=640,batch=8,device=device,
                conf=.001,iou=.7,max_det=300,verbose=False,plots=False,save_json=True,
                project=str(out/'validator'),name=key.replace('@','_'),exist_ok=False)
            predictions=read(Path(metrics.save_dir)/'predictions.json')
            tables[key]=dm.prepare(gt,predictions)
            for image_id,v in tables[key].items():
                v['pred_boxes']=np.array([dm._xyxy(x['bbox']) for x in predictions if x['image_id']==image_id]).reshape(-1,4)
                v['gt_boxes']=np.array([dm._xyxy(x['bbox']) for x in gt['annotations'] if x['image_id']==image_id]).reshape(-1,4)
            points[key]=dm.map_from(tables[key],ids)
            if abs(points[key]['map50_95']-float(metrics.box.map))>1e-4:raise ValueError('AP evaluator disagrees with validator')
            provenance[key]=dict(weights_sha256=sha(weights),selected_update=selected['update'])
    # Threshold calibration on half of validation groups, measurement on the other.
    # Checkpoint used full validation, so this remains exploratory, not an independent estimate.
    groups=sorted(set(image_groups.values()),key=lambda g:__import__('hashlib').sha256(('2026:'+g).encode()).hexdigest())
    strata=collections.defaultdict(list)
    for group in groups:
        classes=tuple(sorted({int(r['class_id']) for r in m['validation'] if r['photograph_group']==group}))
        strata[classes].append(group)
    calibration_groups={g for gs in strata.values() for g in gs[::2]};cal=[i for i in ids if image_groups[i] in calibration_groups];measurement=[i for i in ids if i not in cal]
    for key in points:
        candidates=[(t,operating_recall(tables[key],cal,float(t))) for t in np.arange(.05,.96,.05)]
        threshold=max(candidates,key=lambda pair:(pair[1]['f1'],pair[0]))[0]
        points[key]['operating']=dict(threshold=float(threshold),calibration_ids=cal,measurement_ids=measurement,
            **operating_recall(tables[key],measurement,float(threshold)))
    metrics=['map50_95','ap_unripe','ap_ripe']
    def value(point,metric):return point['map50_95'] if metric=='map50_95' else point['per_class'].get(1 if metric=='ap_unripe' else 0,float('nan'))
    pairs=[('synthetic_improved',b) for b in ('real_only','oversample','geometry','synthetic_G0')]
    boot={f'{a}-{b}':{metric:[] for metric in metrics} for a,b in pairs}
    valid_draws=0
    for draw in grouped_draws(image_groups,resamples,2026):
        # Undefined class AP in a small resample must not be turned into zero.
        pp={k:dm.map_from(t,draw) for k,t in tables.items()}
        if any(set(p['per_class'])!={0,1} for p in pp.values()):continue
        valid_draws+=1
        for a,b in pairs:
            for metric in metrics:boot[f'{a}-{b}'][metric].append(float(np.mean([value(pp[f'{a}@{s}'],metric)-value(pp[f'{b}@{s}'],metric) for s in (0,1,2)])))
    contrasts={}
    for a,b in pairs:
        key=f'{a}-{b}';contrasts[key]={}
        for metric in metrics:
            delta=[value(points[f'{a}@{s}'],metric)-value(points[f'{b}@{s}'],metric) for s in (0,1,2)]
            vals=boot[key][metric]
            contrasts[key][metric]=dict(mean=float(np.mean(delta)),seed_sd=float(np.std(delta,ddof=1)),per_seed=delta,
                ci95=np.percentile(vals,[2.5,97.5]).tolist() if vals else None,wins=sum(x>0 for x in delta))
    rows=[]
    for key,p in points.items():
        arm,seed=key.split('@');rows.append(dict(fraction=fraction,arm=arm,seed=int(seed),
            **{metric:value(p,metric) for metric in metrics},recall_unripe=p['operating']['recall']))
    with (out/'per_seed.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={}
    for arm in DET_ARMS:
        summary[arm]={}
        for metric in metrics+['recall_unripe']:
            vals=[r[metric] for r in rows if r['arm']==arm]
            summary[arm][metric]=dict(mean=float(np.mean(vals)),seed_sd=float(np.std(vals,ddof=1))) if all(v is not None for v in vals) else None
    write(out/'results.json',dict(points=points,summary=summary,contrasts=contrasts,provenance=provenance,subset_digest=m['subset_digest'],
        bootstrap_requested=resamples,bootstrap_valid=valid_draws,seed_sd_separate=True,test_used=False,exploratory=True,
        limitation='Validation used for checkpoint selection; operating calibration/measurement split does not remove checkpoint selection bias. One generator seed.'))
    return rows


def plot_study(study):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    study=Path(study);rows=[]
    for fraction in (25,50,100):
        with (study/f'p{fraction:03d}/evaluation/per_seed.csv').open() as f:rows.extend(csv.DictReader(f))
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,metric in zip(axes,('ap_unripe','map50_95')):
        for arm in DET_ARMS:
            vals=[[float(r[metric]) for r in rows if r['arm']==arm and int(r['fraction'])==p] for p in (25,50,100)]
            ax.errorbar([25,50,100],[np.mean(v) for v in vals],yerr=[np.std(v,ddof=1) for v in vals],marker='o',label=arm)
        ax.set(xlabel='Real train available (%)',ylabel=metric,title='Exploratory validation; bars = detector seed SD');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(study/'detection_data_curve.png',dpi=180);plt.close(fig)
