"""CPU audits and the shared evaluation unit: coverage, quota, diversity, domain separability.

Every feature that mixes RGB with a mask goes through `square_pad`; the historical
audit padded RGB and resized the mask, so on non-square crops the two disagreed.
"""
from __future__ import annotations
import collections
import numpy as np
from PIL import Image
from common import *

OFFSETS=[(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]


def _blank(mode,side):return Image.new(mode,(side,side),0 if mode=='L' else (0,0,0))


def square_pad(im,mask=None):
    """Pad image and mask onto the same square canvas with identical geometry."""
    w,h=im.size;side=max(w,h);off=((side-w)//2,(side-h)//2)
    out=_blank(im.mode,side);out.paste(im,off)
    if mask is None:return out,None
    if mask.size!=im.size:raise ValueError('Mask geometry differs from image')
    padded=_blank('L',side);padded.paste(mask.convert('L'),off)
    return out,padded


def unit_crop(im,box,size=224,pad=.10,jitter=None,mask=None):
    """One comparable unit for every method and every condition: crop, pad, resample."""
    region=crop(im,box,pad,jitter)
    region_mask=crop(mask,box,pad,jitter) if mask is not None else None
    region,region_mask=square_pad(region,region_mask)
    region=region.resize((size,size),Image.Resampling.LANCZOS)
    if region_mask is not None:region_mask=region_mask.resize((size,size),Image.Resampling.NEAREST)
    return region,region_mask


def coverage(scope_ids,records,key='parent_id'):
    """Coverage against the scope ID list. A hardcoded count reports false failures."""
    scope=list(scope_ids);inside=set(scope)
    seen=collections.Counter(r[key] for r in records if r.get(key) in inside)
    return dict(scope=len(scope),covered=sum(1 for p in scope if seen[p]),
                complete=all(seen[p] for p in scope),
                per_parent={p:seen[p] for p in scope},
                missing=[p for p in scope if not seen[p]],
                outside_scope=sorted({r[key] for r in records if r.get(key) not in inside}))


def quota(scope_ids,class_of,records,target=1):
    """Accepted per parent AND class against a pre-registered target; totals alone hide skew."""
    if target<1:raise ValueError('Target must be at least one accepted candidate per parent')
    got=collections.Counter()
    for r in records:
        pid=r['parent_id']
        if pid not in set(scope_ids):raise ValueError(f'Record outside scope: {pid}')
        got[(pid,str(class_of[pid]))]+=1
    keys=[(p,str(class_of[p])) for p in scope_ids]
    short={f'{p}/{c}':target-got[(p,c)] for p,c in keys if got[(p,c)]<target}
    per_class=collections.Counter()
    for (p,c),n in got.items():per_class[c]+=n
    return dict(target=target,met=not short,shortfall=short,
                per_parent_class={f'{p}/{c}':got[(p,c)] for p,c in keys},
                per_class=dict(per_class),total=int(sum(got.values())))


def yields(records):
    """Acceptance per condition at the same number of attempts; failed proposals stay in the cost."""
    attempts=collections.Counter(r['condition'] for r in records)
    accepted=collections.Counter(r['condition'] for r in records if r.get('status')=='accepted')
    return {c:dict(attempts=attempts[c],accepted=accepted[c],
                   yield_rate=accepted[c]/attempts[c],
                   seconds=round(sum(r.get('seconds',0.) for r in records if r['condition']==c),3))
            for c in sorted(attempts)}


def dhash(im,size=8):
    a=np.asarray(im.convert('L').resize((size+1,size),Image.Resampling.LANCZOS),np.int16)
    return ''.join('1' if b else '0' for b in (a[:,1:]>a[:,:-1]).reshape(-1))


def hamming(a,b):
    if len(a)!=len(b):raise ValueError('Hash length mismatch')
    return sum(x!=y for x,y in zip(a,b))


def effective_variants(hashes,parents,threshold=6):
    """Single-link clusters per parent. A redundancy diagnosis, not a universal gate."""
    if threshold<0:raise ValueError('Threshold must be non-negative')
    groups=collections.defaultdict(list)
    for cid,parent in parents.items():groups[parent].append(cid)
    per_parent={}
    for parent,ids in sorted(groups.items()):
        ids=sorted(ids);root={i:i for i in ids}
        def find(i):
            while root[i]!=i:root[i]=root[root[i]];i=root[i]
            return i
        for a in range(len(ids)):
            for b in range(a+1,len(ids)):
                if hamming(hashes[ids[a]],hashes[ids[b]])<=threshold:root[find(ids[a])]=find(ids[b])
        per_parent[parent]=len({find(i) for i in ids})
    per_parent_candidates={parent:len(ids) for parent,ids in sorted(groups.items())}
    informative=all(n>1 for n in per_parent_candidates.values())
    return dict(threshold=threshold,per_parent=per_parent,total=int(sum(per_parent.values())),
                candidates=len(hashes),per_parent_candidates=per_parent_candidates,
                informative=informative,
                note=('one candidate per parent: the cluster count only repeats the candidate count'
                      if not informative else 'clusters measured within each parent'))


def lab_lbp(im,mask=None,bins=16):
    """Colour and texture of the padded unit crop; foreground only when a mask is supplied."""
    lab=np.asarray(im.convert('LAB'),np.float32)
    grey=np.asarray(im.convert('L'),np.int16)
    select=np.ones(grey.shape,bool) if mask is None else (np.asarray(mask)>0)
    if select.sum()<64:raise ValueError('Too few selected pixels for features')
    parts=[]
    for channel in range(3):
        hist,_=np.histogram(lab[:,:,channel][select],bins=bins,range=(0,255))
        parts.append(hist/max(1,hist.sum()))
    code=np.zeros(grey.shape,np.uint8)
    for bit,(dy,dx) in enumerate(OFFSETS):
        code|=((np.roll(np.roll(grey,dy,0),dx,1)>=grey)<<bit).astype(np.uint8)
    inner=np.zeros(grey.shape,bool);inner[1:-1,1:-1]=True
    hist,_=np.histogram(code[select&inner],bins=256,range=(0,256))
    parts.append(hist/max(1,hist.sum()))
    return np.concatenate(parts).astype(np.float32)


def roc_auc(labels,scores):
    """Rank AUC with tied scores averaged."""
    y=np.asarray(labels,int);s=np.asarray(scores,float)
    if len(y)!=len(s):raise ValueError('Label/score length mismatch')
    positive=y==1;n_pos=int(positive.sum());n_neg=len(y)-n_pos
    if n_pos==0 or n_neg==0:raise ValueError('AUC needs both classes')
    order=np.argsort(s,kind='mergesort');sorted_scores=s[order]
    ranks=np.arange(1,len(s)+1,dtype=float);i=0
    while i<len(sorted_scores):
        j=i
        while j+1<len(sorted_scores) and sorted_scores[j+1]==sorted_scores[i]:j+=1
        ranks[i:j+1]=ranks[i:j+1].mean();i=j+1
    out=np.empty(len(s));out[order]=ranks
    return float((out[positive].sum()-n_pos*(n_pos+1)/2)/(n_pos*n_neg))


def domain_auc(features,labels,groups):
    """Leave-one-group-out separability of real vs synthetic.

    A parent and everything generated from it share a group, otherwise the classifier
    recognises the parent instead of the generator.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X=np.asarray(features,np.float64);y=np.asarray(labels,int);g=np.asarray([str(v) for v in groups])
    if not len(X)==len(y)==len(g):raise ValueError('Feature/label/group length mismatch')
    if set(np.unique(y).tolist())!={0,1}:raise ValueError('Need both real and synthetic examples')
    scores=np.zeros(len(y))
    for held in np.unique(g):
        held_out=g==held
        if len(set(y[~held_out].tolist()))<2:raise ValueError(f'Fold without {held} holds one class only')
        scaler=StandardScaler().fit(X[~held_out])
        model=LogisticRegression(max_iter=2000).fit(scaler.transform(X[~held_out]),y[~held_out])
        scores[held_out]=model.predict_proba(scaler.transform(X[held_out]))[:,1]
    return dict(auc=roc_auc(y,scores),n=int(len(y)),groups=int(len(np.unique(g))),
                note='separability of real vs synthetic, not evidence that synthesis harms detection')


def classifier_agreement(predicted,target):
    """Agreement between an automatic classifier and the target class.

    This is agreement, not reviewer-confirmed correctness; `class_fidelity` in the old
    audit was reported as if it were the latter.
    """
    if len(predicted)!=len(target):raise ValueError('Prediction/target length mismatch')
    if not predicted:raise ValueError('No candidates to score')
    hit=sum(1 for p,t in zip(predicted,target) if str(p)==str(t))
    return dict(agreement=hit/len(predicted),n=len(predicted),
                note='automatic agreement with the requested class; blind review decides acceptance')
