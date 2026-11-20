"""Shared, split-bound data and artifact contracts for the two notebooks."""
from __future__ import annotations
import csv, hashlib, json, math, os, time, random, collections
from pathlib import Path
import numpy as np
from PIL import Image

# MELON_ASSETS lets a Kaggle notebook point at the uploaded assets dataset.
ASSETS = Path(os.environ.get('MELON_ASSETS') or Path(__file__).resolve().parent.parent / 'assets')
EXPECTED_MANIFEST = '185e14221e8882f7bcabb5fbb455d6a64aa09378f07234e1ccd4f855531df6ba'
VERSION = 'melon-benchmark-1.0'


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def read_json(path): return json.loads(Path(path).read_text())
def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def read_csv(path):
    with open(path,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def write_csv(path,rows,fields=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=fields or (list(rows[0]) if rows else [])
    tmp=path.with_suffix('.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    tmp.replace(path)

def safe(root,relative):
    root=Path(root).resolve();p=(root/relative).resolve()
    if not p.is_relative_to(root):raise ValueError(f'Path escapes bundle: {relative}')
    return p


class BudgetStop(RuntimeError):pass
class Budget:
    def __init__(self,hours=12,started=None,reserve=600):
        if not 0<hours<=12:raise ValueError('session_hours must be in (0,12]')
        self.started=float(started or time.time());self.deadline=self.started+hours*3600-reserve
    def check(self):
        if time.time()>=self.deadline:raise BudgetStop('Save reserve reached; partial output can be resumed')
    @property
    def elapsed(self):return time.time()-self.started


def stable_seed(*parts):return int(digest(list(parts))[:8],16)

def bind_identity(out,identity):
    path=Path(out)/'identity.json'
    if path.exists() and read_json(path)!=identity:raise ValueError('Output identity differs; use a new output directory or matching config')
    write_json(path,identity)


def code_digest(names=None):
    """Digest of the source. `names` limits it to the modules a run actually depends on,
    so editing an unrelated module does not invalidate a resumable output directory."""
    files=sorted(Path(__file__).parent.glob('*.py'))
    if names is not None:
        wanted=set(names);files=[p for p in files if p.name in wanted]
        missing=wanted-{p.name for p in files}
        if missing:raise ValueError(f'Unknown modules in code digest: {sorted(missing)}')
    return digest({p.name:sha(p) for p in files})


class Dataset:
    def __init__(self,root=None,input_root='/kaggle/input'):
        if root:
            self.root=Path(root)
        else:
            hits=[p.parent for p in Path(input_root).rglob('manifest.csv')
                  if (p.parent/'dataset.json').exists() and sha(p)==EXPECTED_MANIFEST]
            if len(hits)!=1:raise ValueError(f'Expected exactly one frozen dataset, found {hits}. Set dataset_root explicitly.')
            self.root=hits[0]
        if sha(self.root/'manifest.csv')!=EXPECTED_MANIFEST:raise ValueError('Frozen manifest mismatch; this release targets reviewed regrouped-v4 only')
        self.rows=read_csv(self.root/'manifest.csv');self.by_id={r['id']:r for r in self.rows}
        if len(self.by_id)!=len(self.rows):raise ValueError('Duplicate dataset IDs')
        if read_json(self.root/'dataset.json')['label_mapping']!={'0':'ripe','1':'unripe'}:raise ValueError('Class mapping mismatch')
        self.masks={r['id']:r for r in read_json(ASSETS/'reviewed_masks.json')}
        self.scope_ids=sorted(self.masks)
        self.sessions={r['photograph_group']:r['session'] for r in read_csv(self.root/'groups.csv')}
        if sha(self.root/'groups.csv')!='0c0b751bb5c6a05e249a923a1e76d737dfdf4781dd18cbf1f814d35b8f2195af':raise ValueError('Session grouping changed')
        for pid,m in self.masks.items():
            r=self.by_id[pid]
            if r['split']!='train' or m['status']!='accepted' or not m['reviewer'].strip():raise ValueError('Parent must be reviewed TRAIN')
            if r['image_sha256']!=m['source_sha256'] or r['class_id']!=m['class_id']:raise ValueError('Mask/image/class binding mismatch')
            if sha(ASSETS/m['mask'])!=m['reviewed_mask_sha256']:raise ValueError('Reviewed mask changed')
        # Metadata only: check session split integrity without opening any test file.
        groups=collections.defaultdict(set)
        for r in self.rows:groups[self.group(r)].add(r['split'])
        if any(len(v)>1 for v in groups.values()):raise ValueError('Capture session crosses dataset splits')
        self.verified=set()
        self.scope_hash=digest([self.masks[p] for p in self.scope_ids])

    def group(self,row):return self.sessions.get(row['photograph_group'],row['photograph_group'])

    def image(self,pid,allowed=('train','valid')):
        r=self.by_id[pid]
        if r['split'] not in allowed:raise ValueError('Held-out image access rejected')
        p=safe(self.root,r['image'])
        if pid not in self.verified:
            if sha(p)!=r['image_sha256']:raise ValueError(f'Image hash mismatch {pid}')
            if sha(safe(self.root,r['label']))!=r['label_sha256']:raise ValueError(f'Label hash mismatch {pid}')
            self.verified.add(pid)
        with Image.open(p) as im:return im.convert('RGB')

    def boxes(self,pid):
        r=self.by_id[pid]
        if r['split'] not in ('train','valid'):raise ValueError('No test annotation access')
        self.image(pid)
        w,h=int(r['width']),int(r['height']);out=[]
        for line in safe(self.root,r['label']).read_text().splitlines():
            c,x,y,bw,bh=map(float,line.split())
            if c not in (0,1) or not (0<=x<=1 and 0<=y<=1 and 0<bw<=1 and 0<bh<=1):raise ValueError('Invalid YOLO label')
            out.append({'bbox':[(x-bw/2)*w,(y-bh/2)*h,bw*w,bh*h],'class_id':int(c)})
        return out

    def prepare(self,pid,size=512):
        if pid not in self.masks:raise ValueError('Parent outside frozen scope')
        im=self.image(pid,('train',));mask=Image.open(ASSETS/self.masks[pid]['mask']).convert('L')
        if im.size!=mask.size:raise ValueError('Mask image geometry mismatch')
        a=np.asarray(mask)>0
        if a.sum()==0 or a.all():raise ValueError('Invalid foreground mask')
        yy,xx=np.where(a);cx=(xx.min()+xx.max()+1)/2;cy=(yy.min()+yy.max()+1)/2
        side=math.ceil(max(xx.max()-xx.min()+1,yy.max()-yy.min()+1)*1.6)
        left=math.floor(cx-side/2);top=math.floor(cy-side/2)
        # Pillow pads outside image identically; masks use zero, RGB fill uses black.
        box=(left,top,left+side,top+side)
        im=im.crop(box).resize((size,size),Image.Resampling.LANCZOS)
        mask=mask.crop(box).resize((size,size),Image.Resampling.NEAREST)
        return im,mask,dict(source_crop=list(box),source_size=[int(self.by_id[pid]['width']),int(self.by_id[pid]['height'])])

    def report(self):
        return dict(manifest_sha256=EXPECTED_MANIFEST,scope_hash=self.scope_hash,
                    counts={s:dict(collections.Counter(r['class_id'] for r in self.rows if r['split']==s)) for s in ('train','valid','test')},
                    scope_ids=self.scope_ids,scope_classes=dict(collections.Counter(self.by_id[p]['class_id'] for p in self.scope_ids)),
                    scope_policy='historical n10, embedded reviewed masks only',
                    split_unit='capture session where available; otherwise photograph group',
                    limitations=['biological fruit identity not fully verified','validation reused during historical development','test metadata only; never loaded for this benchmark'])


def mask_box(mask):
    a=np.asarray(mask)>0
    if not a.any():raise ValueError('Empty mask')
    y,x=np.where(a);return [int(x.min()),int(y.min()),int(x.max()+1-x.min()),int(y.max()+1-y.min())]


def crop(im,box,pad=.10,jitter=None):
    x,y,w,h=map(float,box)
    if not np.isfinite([x,y,w,h]).all() or w<=0 or h<=0:raise ValueError('Invalid bbox')
    if jitter:
        tx,ty,scale=jitter;x=x+w*tx+w*(1-scale)/2;y=y+h*ty+h*(1-scale)/2;w*=scale;h*=scale
    l=max(0,math.floor(x-w*pad));t=max(0,math.floor(y-h*pad))
    r=min(im.width,math.ceil(x+w*(1+pad)));b=min(im.height,math.ceil(y+h*(1+pad)))
    if r<=l or b<=t:raise ValueError('Empty crop')
    return im.crop((l,t,r,b))


def iou(a,b):
    ax,ay,aw,ah=a;bx,by,bw,bh=b
    inter=max(0,min(ax+aw,bx+bw)-max(ax,bx))*max(0,min(ay+ah,by+bh)-max(ay,by))
    return inter/max(1e-12,aw*ah+bw*bh-inter)


def nms(rows,threshold=.6):
    kept=[]
    for r in sorted(rows,key=lambda x:(-x['score'],x['bbox'])):
        if all(iou(r['bbox'],k['bbox'])<=threshold for k in kept):kept.append(r)
    return kept


def load_proposals(ds):
    obj=read_json(ASSETS/'baseline_proposals.json')
    if obj['manifest_sha256']!=EXPECTED_MANIFEST or sorted(obj['scope_ids'])!=ds.scope_ids:raise ValueError('Detector scope binding mismatch')
    grouped=collections.defaultdict(list)
    for r in obj['proposals']:
        if ds.by_id[r['parent_id']]['split']!='valid':raise ValueError('Proposal outside validation')
        grouped[r['parent_id']].append(r)
    return {pid:nms(rows) for pid,rows in grouped.items()}


def split_validation(ds,seed):
    groups=collections.defaultdict(list)
    for r in ds.rows:
        if r['split']=='valid':groups[ds.group(r)].append(r)
    rng=random.Random(seed);shuffled=list(sorted(groups));rng.shuffle(shuffled)
    order=sorted(shuffled,key=lambda g:-len(groups[g]));parts=[[],[]];counts=[collections.Counter(),collections.Counter()]
    totals=collections.Counter(r['class_id'] for r in ds.rows if r['split']=='valid')
    for g in order:
        cc=collections.Counter(r['class_id'] for r in groups[g]);dominant=max(cc,key=cc.get)
        j=min(range(2),key=lambda k:(counts[k][dominant]/max(1,totals[dominant]),len(parts[k])))
        parts[j].extend(r['id'] for r in groups[g]);counts[j].update(cc)
    if any(len(c)!=2 for c in counts):raise ValueError('Calibration/dev must each contain both classes')
    return parts


def detection_metrics(ds,ids,predictions):
    tp=[0,0];fp=[0,0];fn=[0,0]
    for pid in ids:
        gt=ds.boxes(pid);used=set()
        for p in sorted(predictions.get(pid,[]),key=lambda x:-x['score']):
            cand=[(iou(p['bbox'],g['bbox']),i) for i,g in enumerate(gt) if i not in used]
            overlap,idx=max(cand,default=(0,-1));pc=p['class_id']
            if overlap<.5:fp[pc]+=1
            else:
                used.add(idx);gc=gt[idx]['class_id']
                if pc==gc:tp[gc]+=1
                else:fp[pc]+=1;fn[gc]+=1
        for i,g in enumerate(gt):
            if i not in used:fn[g['class_id']]+=1
    out={}
    for c,name in enumerate(['ripe','unripe']):
        out.update({f'tp_{name}':tp[c],f'fp_{name}':fp[c],f'fn_{name}':fn[c],
                    f'f1_{name}':2*tp[c]/max(1,2*tp[c]+fp[c]+fn[c]),
                    f'recall_{name}':tp[c]/max(1,tp[c]+fn[c]),f'precision_{name}':tp[c]/max(1,tp[c]+fp[c])})
    out['macro_f1']=(out['f1_ripe']+out['f1_unripe'])/2
    return out
