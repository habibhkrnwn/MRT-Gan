"""Colour-texture maturity head on detector crops.

The pilot measured where the errors are: at detection threshold 0.05 an oracle that only
fixes the class reaches macro-F1 1.0000, so localisation is not the bottleneck and the
remaining error is the maturity decision. This module implements the head from
`09_Arsitektur_dan_Spesifikasi/04_SPESIFIKASI_HEAD.md` so that decision can be studied on
its own: the detector proposes boxes, the head classifies the crop.

Three branches, as specified. MobileNetV3-Small carries the separation work (0.9593 alone
on GT crops), Lab colour and uniform LBP texture are complements (0.8227 and 0.6620), and a
learned gate mixes them. Colour and texture are NOT sufficient on their own: two ripe
parents sit on the unripe side of that feature space, which is a property of the data.

Inference needs an RGB image and detector proposals. No ground truth, no mask, no reviewer,
no SAM, no generator. The class mapping is internal 0=ripe, 1=unripe and COCO 1/2 on export.
"""
from __future__ import annotations
import math
import collections
import numpy as np
from PIL import Image

SIZE=224
PAD=.10
LBP_P,LBP_R=8,1
LAB_BINS=16
VISUAL_DIM=576
COLOUR_DIM=3*LAB_BINS
TEXTURE_DIM=LBP_P+2
PROJECTION_DIM=64
CLASS_NAMES=['ripe','unripe']
IMAGENET_MEAN=(0.485,0.456,0.406)
IMAGENET_STD=(0.229,0.224,0.225)
JS_EPSILON=1e-8
LAMBDA_JS=0.1


class CropRejected(ValueError):
    """A crop that cannot be taken. Counted in the log, never skipped in silence."""


def expand_box(box,width,height,pad=PAD):
    """Grow the box by `pad` on each side, clip to the image, floor left/top, ceil right/bottom.

    The detector box stays the exported box; this larger region is what the head looks at.
    A non-finite coordinate or an empty result is an error the caller must record, because a
    crop that vanishes silently would quietly change which objects the head ever sees.
    """
    values=[float(v) for v in box]
    if not all(math.isfinite(v) for v in values):raise CropRejected(f'non-finite box: {box}')
    x,y,w,h=values
    if w<=0 or h<=0:raise CropRejected(f'non-positive box size: {box}')
    left=math.floor(max(0.0,x-pad*w));top=math.floor(max(0.0,y-pad*h))
    right=math.ceil(min(float(width),x+w+pad*w));bottom=math.ceil(min(float(height),y+h+pad*h))
    if right<=left or bottom<=top:raise CropRejected(f'empty crop after clipping: {box}')
    return int(left),int(top),int(right),int(bottom)


def crop_for_head(image,box,size=SIZE,pad=PAD):
    """Expanded crop, resized to the head input. Bilinear with antialias, RGB, no mask.

    No mask at inference means no mask at training either: a head that needed one would
    depend on SAM and a reviewer at inference time, which the specification forbids.
    """
    left,top,right,bottom=expand_box(box,image.width,image.height,pad)
    region=image.convert('RGB').crop((left,top,right,bottom))
    return region.resize((size,size),Image.Resampling.BILINEAR)


def lab_histogram(rgb,bins=LAB_BINS):
    """16 bins per CIELAB channel, each normalised to sum 1. Ends are clipped, not dropped."""
    from skimage.color import rgb2lab
    lab=rgb2lab(np.asarray(rgb,np.float32)/255.0)
    parts=[]
    for channel,(low,high) in zip(range(3),((0,100),(-128,128),(-128,128))):
        values=np.clip(lab[:,:,channel],low,high)
        hist,_=np.histogram(values,bins=bins,range=(low,high))
        parts.append(hist/max(1,hist.sum()))
    return np.concatenate(parts).astype(np.float32)


def lbp_histogram(rgb):
    """Uniform LBP, P=8 R=1, 10 bins, normalised. The 1 px border is ignored."""
    from skimage.feature import local_binary_pattern
    array=np.asarray(rgb,np.float32)
    grey=np.round(0.299*array[:,:,0]+0.587*array[:,:,1]+0.114*array[:,:,2]).astype(np.uint8)
    code=local_binary_pattern(grey,LBP_P,LBP_R,method='uniform')
    inner=np.zeros(grey.shape,bool);inner[1:-1,1:-1]=True
    hist,_=np.histogram(code[inner],bins=TEXTURE_DIM,range=(0,TEXTURE_DIM))
    return (hist/max(1,hist.sum())).astype(np.float32)


def handcrafted(rgb):
    """Colour and texture features of one crop, as the two fixed branches read them."""
    return lab_histogram(rgb),lbp_histogram(rgb)


def normalise_batch(crops):
    """Crops to an ImageNet-normalised NCHW float tensor."""
    import torch
    stack=np.stack([np.asarray(c,np.float32)/255.0 for c in crops]).transpose(0,3,1,2)
    tensor=torch.from_numpy(stack)
    mean=torch.tensor(IMAGENET_MEAN).view(1,3,1,1);std=torch.tensor(IMAGENET_STD).view(1,3,1,1)
    return (tensor-mean)/std


def jitter_box(box,rng,width,height,scale=(0.90,1.10),shift=0.10,tries=10,min_iou=0.5):
    """Training-time box jitter calibrated to the IoU spread of predicted boxes.

    Training on predicted crops from the train images is not an option here: the detector
    memorises its ten training photos, so those boxes reach IoU 0.992-0.998 and the crops are
    near duplicates of the GT crops. Jitter is the mitigation that actually moved the number.
    After `tries` failures the original box is returned, so the sample is never dropped.
    """
    x,y,w,h=[float(v) for v in box]
    for _ in range(tries):
        sx=rng.uniform(*scale);sy=rng.uniform(*scale)
        nw=w*sx;nh=h*sy
        nx=x+rng.uniform(-shift,shift)*w+(w-nw)/2
        ny=y+rng.uniform(-shift,shift)*h+(h-nh)/2
        candidate=(nx,ny,nw,nh)
        if nw<=0 or nh<=0:continue
        if nx<0 or ny<0 or nx+nw>width or ny+nh>height:continue
        if box_iou(candidate,(x,y,w,h))>=min_iou:return candidate
    return (x,y,w,h)


def box_iou(a,b):
    ax,ay,aw,ah=a;bx,by,bw,bh=b
    x1,y1=max(ax,bx),max(ay,by)
    x2,y2=min(ax+aw,bx+bw),min(ay+ah,by+bh)
    inter=max(0.0,x2-x1)*max(0.0,y2-y1)
    union=aw*ah+bw*bh-inter
    return inter/union if union>0 else 0.0


def cross_class_nms(proposals,iou=0.6,max_det=100):
    """Class-agnostic NMS over decoded proposals, shared by every control.

    No-head and head must see the same boxes, otherwise a difference between them could come
    from a different proposal set rather than from the classification.
    """
    kept=[]
    for p in sorted(proposals,key=lambda z:(-z['score'],z.get('index',0))):
        if all(box_iou(p['bbox'],q['bbox'])<=iou for q in kept):
            kept.append(p)
        if len(kept)>=max_det:break
    return kept


def js_divergence(p,q,epsilon=JS_EPSILON):
    """Jensen-Shannon divergence between two probability rows, natural log, symmetric."""
    import torch
    p=p.clamp_min(epsilon);q=q.clamp_min(epsilon)
    m=(p+q)/2
    kl=lambda a,b:(a*(a.log()-b.log())).sum(-1)
    return 0.5*kl(p,m)+0.5*kl(q,m)


try:                                   # torch is optional for the feature-only helpers above
    import torch
    from torch import nn

    class VisualEncoder(nn.Module):
        """Frozen MobileNetV3-Small trunk, global average pool, 576 features.

        Frozen means frozen: eval mode so BatchNorm keeps its running statistics, and no
        parameter reaches the optimizer. With ten training crops a trainable backbone would
        memorise them long before it learned anything about maturity.
        """
        def __init__(self,weights='IMAGENET1K_V1'):
            super().__init__()
            from torchvision.models import mobilenet_v3_small
            model=mobilenet_v3_small(weights=weights)
            self.features=model.features
            self.pool=nn.AdaptiveAvgPool2d(1)
            for parameter in self.parameters():parameter.requires_grad_(False)
            self.eval()

        def train(self,mode=True):        # keep BatchNorm in eval whatever the caller does
            return super().train(False)

        @torch.no_grad()
        def forward(self,pixels):
            return self.pool(self.features(pixels)).flatten(1)

    class MaturityHead(nn.Module):
        """Three projections, a softmax gate over them, and a two-class classifier.

        `full` uses visual, colour and texture. `rgb` is the baseline that separates the
        benefit of classifying a crop at all from the benefit of colour and texture: one
        projection, no gate. `no_colour_texture` keeps the gate over two visual-only
        projections so the ablation changes the branches, not the architecture around them.
        """
        VARIANTS=('full','rgb','no_colour_texture')

        def __init__(self,variant='full',dropout=0.2):
            super().__init__()
            if variant not in self.VARIANTS:raise ValueError(f'Unknown variant: {variant}')
            self.variant=variant
            self.visual=nn.Linear(VISUAL_DIM,PROJECTION_DIM)
            if variant=='full':
                self.colour=nn.Linear(COLOUR_DIM,PROJECTION_DIM)
                self.texture=nn.Linear(TEXTURE_DIM,PROJECTION_DIM)
                self.gate=nn.Linear(3*PROJECTION_DIM,3)
            elif variant=='no_colour_texture':
                self.visual_second=nn.Linear(VISUAL_DIM,PROJECTION_DIM)
                self.gate=nn.Linear(2*PROJECTION_DIM,2)
            self.dropout=nn.Dropout(dropout)
            self.classifier=nn.Linear(PROJECTION_DIM,2)

        def branches(self,visual,colour=None,texture=None):
            relu=torch.relu
            if self.variant=='rgb':return [relu(self.visual(visual))]
            if self.variant=='no_colour_texture':
                return [relu(self.visual(visual)),relu(self.visual_second(visual))]
            if colour is None or texture is None:
                raise ValueError('The full variant needs colour and texture features')
            return [relu(self.visual(visual)),relu(self.colour(colour)),relu(self.texture(texture))]

        def forward(self,visual,colour=None,texture=None):
            parts=self.branches(visual,colour,texture)
            if len(parts)==1:
                fused=parts[0]
                gate=torch.ones(fused.shape[0],1,device=fused.device)
            else:
                gate=torch.softmax(self.gate(torch.cat(parts,dim=1)),dim=1)
                fused=sum(gate[:,i:i+1]*parts[i] for i in range(len(parts)))
            logits=self.classifier(self.dropout(fused))
            return logits,gate

    def head_loss(logits,targets,paired_logits=None,lam=LAMBDA_JS):
        """Cross entropy over the batch plus a consistency term over real-synthetic pairs.

        `lam` stays fixed across variants: changing the loss and the branches at the same
        time would leave the difference unattributable. Real-only batches have no pairs, so
        the JS term is exactly zero rather than absent.
        """
        ce=nn.functional.cross_entropy(logits,targets)
        if paired_logits is None:
            js=torch.zeros((),device=logits.device)
        else:
            p=torch.softmax(logits,dim=1);q=torch.softmax(paired_logits,dim=1)
            js=js_divergence(p,q).mean()
        return ce+lam*js,dict(ce=float(ce.detach()),js=float(js.detach()),
                              total=float((ce+lam*js).detach()))

except ImportError:                    # feature helpers still work without torch installed
    VisualEncoder=None
    MaturityHead=None
    head_loss=None


def build_head(variant='full',seed=0):
    """Every variant is initialised from the same seed, not carved out of a trained model.

    Deleting a branch from a trained full model would confound the ablation with whatever
    the remaining branches learned while that branch was present.
    """
    import torch
    if MaturityHead is None:raise RuntimeError('torch is required to build the head')
    torch.manual_seed(seed)
    return MaturityHead(variant=variant)


def expected_calibration_error(probabilities,labels,bins=10):
    """ECE over the predicted-class confidence. Valid here: head softmax IS a probability.

    Detector scores are not, which is why the calibration work on the detector reports an
    operating point instead of ECE.
    """
    probabilities=np.asarray(probabilities,float);labels=np.asarray(labels,int)
    confidence=probabilities.max(axis=1);predicted=probabilities.argmax(axis=1)
    correct=(predicted==labels).astype(float)
    edges=np.linspace(0.0,1.0,bins+1)
    total=len(labels);error=0.0;rows=[]
    for i in range(bins):
        low,high=edges[i],edges[i+1]
        inside=(confidence>low)&(confidence<=high) if i else (confidence>=low)&(confidence<=high)
        if not inside.any():
            rows.append(dict(bin=[round(low,3),round(high,3)],n=0,confidence=None,accuracy=None))
            continue
        gap=abs(correct[inside].mean()-confidence[inside].mean())
        error+=inside.sum()/total*gap
        rows.append(dict(bin=[round(low,3),round(high,3)],n=int(inside.sum()),
                         confidence=round(float(confidence[inside].mean()),4),
                         accuracy=round(float(correct[inside].mean()),4)))
    return dict(ece=round(float(error),4),bins=rows)


def brier_score(probabilities,labels):
    """Multiclass Brier score against the one-hot label."""
    probabilities=np.asarray(probabilities,float);labels=np.asarray(labels,int)
    onehot=np.zeros_like(probabilities);onehot[np.arange(len(labels)),labels]=1.0
    return round(float(((probabilities-onehot)**2).sum(axis=1).mean()),6)


def to_coco(boxes,labels,scores,image_id):
    """COCO export. category_id is 1/2 on paper and 0/1 in the tensors; never the other way."""
    rows=[]
    for box,label,score in zip(boxes,labels,scores):
        if int(label) not in (0,1):raise ValueError(f'Internal label must be 0 or 1: {label}')
        rows.append(dict(image_id=image_id,category_id=int(label)+1,
                         bbox=[round(float(v),2) for v in box],score=round(float(score),6)))
    return rows


def decision_table(proposals,probabilities):
    """The three postprocessing controls, kept apart so a gain cannot be misattributed.

    `common_nms/no_head` keeps the detector's own label and score on the shared proposal set;
    `common_nms/head` replaces the label with the head argmax and multiplies the scores. A
    difference between them is the head. A difference against `raw/native` is not: that one
    also carries the detector's own NMS.
    """
    if len(proposals)!=len(probabilities):
        raise ValueError('One probability row per proposal is required')
    no_head=[];head=[]
    for proposal,row in zip(proposals,probabilities):
        no_head.append(dict(bbox=proposal['bbox'],label=int(proposal['label']),
                            score=float(proposal['score'])))
        index=int(np.argmax(row))
        head.append(dict(bbox=proposal['bbox'],label=index,
                         score=float(proposal['score'])*float(row[index]),
                         probabilities=[round(float(v),6) for v in row]))
    return collections.OrderedDict([('common_nms/no_head',no_head),('common_nms/head',head)])
