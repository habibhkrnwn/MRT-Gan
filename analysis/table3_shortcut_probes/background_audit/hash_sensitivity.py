from pathlib import Path
import csv,json,hashlib
import numpy as np
from PIL import Image
import torch
from ultralytics.utils.metrics import ap_per_class,box_iou
OUT=Path(__file__).resolve().parent; G=OUT.parents[3]; DATA=G/'runs/19_mrtgan_detector/local_reexport/dataset_v4'; RUN=G/'runs/19_mrtgan_detector/mrtgan_detector_v1/result'
rows=[r for r in csv.DictReader((DATA/'manifest.csv').open()) if r['split'] in ('train','valid') and r['id']!='DSC_0093']
H=[]
for r in rows:
 im=Image.open(DATA/r['image']).convert('RGB'); W,HH=im.size
 _,x,y,w,h=map(float,(DATA/r['label']).read_text().split())
 crop=im.crop((int(max(0,(x-w/2-w*.1)*W)),int(max(0,(y-h/2-h*.1)*HH)),int(min(W,(x+w/2+w*.1)*W)),int(min(HH,(y+h/2+h*.1)*HH))))
 arr=np.asarray(crop.convert('L').resize((17,16),Image.Resampling.LANCZOS)); H.append((arr[:,1:]>arr[:,:-1]).ravel())
H=np.stack(H); train=np.array([r['split']=='train' for r in rows]); ids=np.array([r['id'] for r in rows]); flags=[]
for i in np.where(~train)[0]:
 d=(H[train]!=H[i]).sum(1); j=d.argmin()
 if d[j]<=26: flags.append(dict(valid=ids[i],train=ids[train][j],distance=int(d[j])))
tight=json.loads((OUT/'integrity.json').read_text())['near_duplicate_flags_threshold26']; exclude=sorted({p['valid'] for p in tight})

def evaluate(run,exclude):
 gt=json.loads((run/'groundtruth.coco.json').read_text()); pr=json.loads((run/'predictions.coco.json').read_text()); tp=[]; conf=[]; pc=[]; tc=[]
 def xyxy(b): x,y,w,h=b; return [x,y,x+w,y+h]
 for im in gt['images']:
  if im['id'] in exclude: continue
  aa=[a for a in gt['annotations'] if a['image_id']==im['id']]; pp=sorted([p for p in pr if p['image_id']==im['id']],key=lambda p:-p['score'])
  gc=np.array([a['category_id']-1 for a in aa]); tc.extend(gc)
  if not pp:continue
  cls=np.array([p['category_id']-1 for p in pp]); iou=box_iou(torch.tensor([xyxy(a['bbox']) for a in aa]),torch.tensor([xyxy(p['bbox']) for p in pp])).numpy()*(gc[:,None]==cls[None,:]); cor=np.zeros((len(pp),10),bool)
  for j,t in enumerate(np.linspace(.5,.95,10)):
   m=np.stack(np.nonzero(iou>=t),axis=1)
   if len(m):
    m=m[np.argsort(-iou[m[:,0],m[:,1]])]; m=m[np.unique(m[:,1],return_index=True)[1]]; m=m[np.unique(m[:,0],return_index=True)[1]]; cor[m[:,1],j]=True
  tp.append(cor);conf.extend([p['score'] for p in pp]);pc.extend(cls)
 ap=ap_per_class(np.concatenate(tp),np.array(conf),np.array(pc),np.array(tc))[5]
 return dict(map50_95=float(ap.mean()),ap_unripe=float(ap[1].mean()))
res={}
for arm in ['real_only','oversample_k22','colour_transfer_k22','mrtgan_k22']:
 per=[]
 for run in sorted(RUN.glob(f'group*/{arm}__seed*')):
  full=evaluate(run,[]); reduced=evaluate(run,exclude); archived=json.loads((run/'validation.json').read_text())
  per.append(dict(run=run.name,all87=full,exclude5_n82=reduced,reproduction_abs_error=abs(full['map50_95']-archived['map50_95'])))
 res[arm]={'per_seed':per,'all87_mean':float(np.mean([x['all87']['map50_95'] for x in per])),'exclude5_n82_mean':float(np.mean([x['exclude5_n82']['map50_95'] for x in per])),'max_reproduction_error':max(x['reproduction_abs_error'] for x in per)}
 print(arm,res[arm]['all87_mean'],res[arm]['exclude5_n82_mean'],flush=True)
(OUT/'hash_sensitivity.json').write_text(json.dumps(dict(padded10_flags=flags,tight_box_flags=tight,excluded_for_sensitivity_only=exclude,results=res,note='Post-hoc sensitivity, no original files or official split modified. Matching images are candidates, not definitive proof of source identity. Test not opened.'),indent=2))
print('padded',flags,flush=True)
