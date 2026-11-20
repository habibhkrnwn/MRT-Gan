"""Frozen-embedding crop classifier: cheap screening and end-to-end validation metrics.

The detector stays frozen, so a generator changes only the crop classifier. Fitting uses
train crops and their synthetic derivatives; calibration selects C and the operating
point; dev estimates development performance. Screening is not confirmation: a thesis
claim about detector training still needs the detector retrained per condition.
"""
from __future__ import annotations
import collections
import numpy as np
from PIL import Image
from common import *
from audit import unit_crop

BACKBONES={
 'mobilenet_v3_small':dict(source='torchvision',model='mobilenet_v3_small',
                           weights='MobileNet_V3_Small_Weights.IMAGENET1K_V1',strip='classifier'),
 'convnext_tiny':dict(source='torchvision',model='convnext_tiny',
                      weights='ConvNeXt_Tiny_Weights.IMAGENET1K_V1',strip='classifier.2'),
 'efficientnet_v2_s':dict(source='torchvision',model='efficientnet_v2_s',
                          weights='EfficientNet_V2_S_Weights.IMAGENET1K_V1',strip='classifier.1'),
 'mobilenet_v3_large':dict(source='torchvision',model='mobilenet_v3_large',
                           weights='MobileNet_V3_Large_Weights.IMAGENET1K_V2',strip='classifier'),
 'dinov2_vits14':dict(source='hub',repo='facebookresearch/dinov2',model='dinov2_vits14',size=224),
}
GRID=(0.01,0.1,1.)
THRESHOLDS=(0.05,0.10,0.15,0.25,0.40)


def _set_identity(model,path):
    import torch.nn as nn
    parts=path.split('.');holder=model
    for part in parts[:-1]:holder=holder[int(part)] if part.isdigit() else getattr(holder,part)
    last=parts[-1]
    if last.isdigit():holder[int(last)]=nn.Identity()
    else:setattr(holder,last,nn.Identity())


class Embedder:
    """Pretrained encoder used as a frozen feature extractor with its official preprocessing."""
    def __init__(self,name,device=None,batch=32):
        if name not in BACKBONES:raise ValueError(f'Unknown backbone {name}')
        import torch
        self.name=name;self.batch=batch;spec=BACKBONES[name]
        self.device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
        if spec['source']=='torchvision':
            import torchvision
            weights=torchvision.models.get_weight(spec['weights'])
            self.model=torchvision.models.get_model(spec['model'],weights=weights)
            _set_identity(self.model,spec['strip'])
            self.transform=weights.transforms()
            self.lock=dict(source='torchvision',weights=spec['weights'],url=weights.url,
                           library=torchvision.__version__)
        else:
            from torchvision import transforms
            self.model=torch.hub.load(spec['repo'],spec['model'])
            size=spec['size']
            if size%14:raise ValueError('DINOv2 input must be a multiple of the patch size')
            self.transform=transforms.Compose([
                transforms.Resize(size,interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(size),transforms.ToTensor(),
                transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
            self.lock=dict(source='hub',repo=spec['repo'],model=spec['model'],size=size)
        self.model.eval().to(self.device)
        for parameter in self.model.parameters():parameter.requires_grad_(False)
        self.dim=int(self(Image.new('RGB',(256,256))).shape[1])

    def __call__(self,images):
        import torch
        if isinstance(images,Image.Image):images=[images]
        if not images:raise ValueError('No crops to embed')
        out=[]
        with torch.inference_mode():
            for start in range(0,len(images),self.batch):
                stack=torch.stack([self.transform(im.convert('RGB')) for im in images[start:start+self.batch]])
                feature=self.model(stack.to(self.device))
                if feature.ndim!=2:raise ValueError(f'Encoder returned {feature.ndim}D features; strip the head')
                out.append(feature.float().cpu().numpy())
        return np.concatenate(out)


def jitter_for(seed,amount=.10,scale=.10):
    """Deterministic crop jitter. Frozen before confirmation; retuning belongs to calibration."""
    rng=np.random.default_rng(seed)
    return (float(rng.uniform(-amount,amount)),float(rng.uniform(-amount,amount)),
            float(rng.uniform(1-scale,1+scale)))


def real_crops(dataset,ids,size=224,repeats=1,jitter=True,seed=0):
    """GT crops from allowed splits; repeats add jittered views of the same annotation."""
    crops=[];labels=[];groups=[]
    for pid in ids:
        image=dataset.image(pid,('train','valid'))
        for index,box in enumerate(dataset.boxes(pid)):
            for repeat in range(repeats):
                offset=jitter_for(stable_seed(seed,pid,index,repeat)) if (jitter and repeat) else None
                crops.append(unit_crop(image,box['bbox'],size=size,jitter=offset)[0])
                labels.append(int(box['class_id']));groups.append(pid)
    return crops,labels,groups


def geometry_crops(dataset,parents,size=224,seed=0,views=1):
    """Plain geometric views of the same parents through the same crop path.

    The control for the synthetic arms: same parents, same count, same preprocessing, no
    generator. Without it, a gain from a synthetic arm cannot be separated from simply
    adding more geometric views of the same fruit.
    """
    crops=[];labels=[];groups=[]
    for pid in parents:
        image=dataset.image(pid,('train',))
        for index,box in enumerate(dataset.boxes(pid)):
            for view in range(views):
                offset=jitter_for(stable_seed('geometry',seed,pid,index,view),amount=.15,scale=.15)
                crop_image=unit_crop(image,box['bbox'],size=size,jitter=offset)[0]
                if stable_seed('flip',seed,pid,index,view)%2:
                    crop_image=crop_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                crops.append(crop_image);labels.append(int(box['class_id']));groups.append(pid)
    return crops,labels,groups


def synthetic_crops(out,records,size=224):
    """Crops of selected candidates, grouped under the parent so folds stay honest."""
    out=Path(out);crops=[];labels=[];groups=[]
    for record in records:
        if record.get('status')!='accepted':raise ValueError('Only reviewed accepted candidates may train')
        image=Image.open(out/record['image']).convert('RGB')
        mask=Image.open(out/record['mask']).convert('L')
        crops.append(unit_crop(image,record['bbox'],size=size,mask=mask)[0])
        labels.append(int(record['class_id']));groups.append(record['parent_id'])
    return crops,labels,groups


def fit_select(train_features,train_labels,cal_features,cal_labels,grid=GRID,seed=0):
    """Scaler and logistic regression fit on train only; C chosen on calibration macro-F1."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score
    if len(set(train_labels))<2:raise ValueError('Training crops must contain both classes')
    scaler=StandardScaler().fit(np.asarray(train_features,np.float64))
    scored=[]
    for C in grid:
        model=LogisticRegression(C=C,max_iter=5000,random_state=seed)
        model.fit(scaler.transform(np.asarray(train_features,np.float64)),train_labels)
        macro=f1_score(cal_labels,model.predict(scaler.transform(np.asarray(cal_features,np.float64))),
                       average='macro')
        scored.append((macro,C,model))
    macro,C,model=max(scored,key=lambda row:(row[0],-row[1]))
    return dict(scaler=scaler,model=model,C=C,calibration_macro_f1=float(macro),
                grid={str(c):float(m) for m,c,_ in scored},
                selection='C chosen on calibration; dev never used for fitting or selection')


def classify(fit,features):
    return fit['model'].predict(fit['scaler'].transform(np.asarray(features,np.float64)))


def predicted_crops(dataset,ids,proposals,size=224,score_threshold=.25):
    """Frozen detector boxes turned into comparable crops; misses stay missing."""
    crops=[];index=[]
    for pid in ids:
        rows=[r for r in proposals.get(pid,[]) if r['score']>=score_threshold]
        if not rows:continue
        image=dataset.image(pid,('valid',))
        for row in rows:
            crops.append(unit_crop(image,row['bbox'],size=size)[0]);index.append((pid,row))
    return crops,index


def score_sweep(dataset,ids,proposals,embed,fit,size=224,grid=THRESHOLDS):
    """Evaluate several detector thresholds in one pass.

    A higher threshold keeps a subset of the boxes of a lower one, so the crops are cut and
    embedded once at the lowest threshold and then filtered. Running the whole pipeline per
    threshold decodes the same images five times for the same answer.
    """
    grid=sorted(set(grid))
    crops,index=predicted_crops(dataset,ids,proposals,size,min(grid))
    classes=classify(fit,embed(crops)) if crops else []
    out={}
    for threshold in grid:
        predictions=collections.defaultdict(list)
        for (pid,row),class_id in zip(index,classes):
            if row['score']>=threshold:
                predictions[pid].append(dict(bbox=row['bbox'],score=row['score'],class_id=int(class_id)))
        metrics=detection_metrics(dataset,ids,dict(predictions))
        metrics.update(images=len(ids),boxes=sum(len(v) for v in predictions.values()),
                       score_threshold=threshold)
        out[threshold]=(metrics,dict(predictions))
    return out


def select_threshold(dataset,ids,proposals,embed,fit,size=224,grid=THRESHOLDS,sweep=None):
    """Detector operating point chosen on calibration images, never on dev."""
    sweep=sweep or score_sweep(dataset,ids,proposals,embed,fit,size,grid)
    scored=[(metrics['macro_f1'],threshold) for threshold,(metrics,_) in sweep.items()]
    macro,threshold=max(scored,key=lambda row:(row[0],-row[1]))
    return dict(score_threshold=threshold,calibration_macro_f1=float(macro),
                grid={str(t):round(m,6) for m,t in scored})


def gt_crop_metrics(dataset,ids,embed,fit,size=224):
    """Classification alone on GT crops: separates classifier quality from lost boxes."""
    from sklearn.metrics import f1_score
    crops,labels,_=real_crops(dataset,ids,size,1,False,0)
    predicted=classify(fit,embed(crops))
    return dict(macro_f1=float(f1_score(labels,predicted,average='macro')),
                accuracy=float(sum(int(a)==int(b) for a,b in zip(labels,predicted))/len(labels)),
                crops=len(labels))


def end_to_end_detail(dataset,ids,proposals,embed,fit,size=224,score_threshold=.25):
    """Detector localises, classifier decides the class; macro-F1 counts misses and FPs."""
    crops,index=predicted_crops(dataset,ids,proposals,size,score_threshold)
    predictions=collections.defaultdict(list)
    if crops:
        classes=classify(fit,embed(crops))
        for (pid,row),class_id in zip(index,classes):
            predictions[pid].append(dict(bbox=row['bbox'],score=row['score'],class_id=int(class_id)))
    metrics=detection_metrics(dataset,ids,dict(predictions))
    metrics.update(images=len(ids),boxes=len(crops),score_threshold=score_threshold)
    return metrics,dict(predictions)


def end_to_end(dataset,ids,proposals,embed,fit,size=224,score_threshold=.25):
    return end_to_end_detail(dataset,ids,proposals,embed,fit,size,score_threshold)[0]


def condition_metrics(dataset,proposals,embed,out,selected,seed=0,size=224,repeats=2,
                      score_threshold=.25,parents=None,geometry=True,tune_threshold=True,
                      keep_predictions=False,scope=None):
    """One screening row: real-only, real+geometry and real+synthetic on the same splits.

    `parents` limits the geometry control to the parents the synthetic arm actually covers,
    so the two added arms carry the same number of extra crops from the same fruit. `scope`
    limits the real training crops, which is how a parent with a disputed label is kept out
    of every arm instead of only out of the synthetic one.
    """
    calibration,dev=split_validation(dataset,seed)
    scope=list(scope or dataset.scope_ids)
    parents=list(parents or sorted({r['parent_id'] for r in selected})) or scope
    outside=[pid for pid in parents if pid not in scope]
    if outside:raise ValueError(f'Synthetic parents outside the real scope: {outside}')
    train_crops,train_labels,_=real_crops(dataset,scope,size,repeats,True,seed)
    cal_crops,cal_labels,_=real_crops(dataset,calibration,size,1,False,seed)
    train_features=embed(train_crops);cal_features=embed(cal_crops)
    arms={'real_only':(train_features,train_labels)}
    if geometry:
        views=max(1,round(len(selected)/max(1,len(parents)))) if selected else 1
        geometry_crop,geometry_labels,_=geometry_crops(dataset,parents,size,seed,views)
        arms['real_plus_geometry']=(np.concatenate([train_features,embed(geometry_crop)]),
                                    train_labels+geometry_labels)
    if selected:
        synth_crops,synth_labels,_=synthetic_crops(out,selected,size)
        arms['real_plus_synthetic']=(np.concatenate([train_features,embed(synth_crops)]),
                                     train_labels+synth_labels)
    rows={}
    for name,(features,labels) in arms.items():
        fit=fit_select(features,labels,cal_features,cal_labels,seed=seed)
        grid=tuple(sorted(set(THRESHOLDS)|{score_threshold}))
        point=(select_threshold(dataset,calibration,proposals,embed,fit,size,grid)
               if tune_threshold else dict(score_threshold=score_threshold,
                                           calibration_macro_f1=None,grid={}))
        dev_sweep=score_sweep(dataset,dev,proposals,embed,fit,size,grid)
        metrics,predictions=dev_sweep[point['score_threshold']]
        rows[name]=dict(C=fit['C'],calibration_macro_f1=fit['calibration_macro_f1'],
                        crops=len(labels),added_crops=len(labels)-len(train_labels),
                        operating_point=point,
                        gt_crop=gt_crop_metrics(dataset,dev,embed,fit,size),
                        dev=metrics,dev_at_025=dev_sweep[score_threshold][0])
        if keep_predictions:rows[name]['dev_predictions']=predictions
    for name in ('real_plus_geometry','real_plus_synthetic'):
        if name in rows:
            rows[f'delta_{name}']=round(rows[name]['dev']['macro_f1']-rows['real_only']['dev']['macro_f1'],6)
    if 'real_plus_synthetic' in rows and 'real_plus_geometry' in rows:
        rows['delta_synthetic_vs_geometry']=round(
            rows['real_plus_synthetic']['dev']['macro_f1']-rows['real_plus_geometry']['dev']['macro_f1'],6)
    rows['split_seed']=seed;rows['encoder']=embed.name
    rows['calibration_images']=len(calibration);rows['dev_images']=len(dev)
    rows['synthetic_parents']=parents;rows['real_scope']=scope
    rows['note']=('screening only; detector frozen, threshold chosen on calibration, '
                  'labels of calibration images are a real cost')
    return rows
