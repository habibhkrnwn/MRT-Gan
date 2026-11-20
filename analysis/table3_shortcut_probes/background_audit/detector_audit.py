from pathlib import Path
import csv,json,hashlib,time
import numpy as np
import torch
from PIL import Image
from ultralytics.data.augment import LetterBox
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.metrics import ap_per_class,box_iou
OUT=Path(__file__).resolve().parent; G=OUT.parents[3]
DATA=G/'runs/19_mrtgan_detector/local_reexport/dataset_v4'
RUN=G/'runs/19_mrtgan_detector/mrtgan_detector_v1/result'
torch.set_num_threads(4); torch.set_grad_enabled(False)
rows=[r for r in csv.DictReader((DATA/'manifest.csv').open()) if r['split']=='valid']
assert len(rows)==87
lb=LetterBox(new_shape=(640,640),auto=False,stride=32)
inputs=[]
for r in rows:
 im=np.asarray(Image.open(DATA/r['image']).convert('RGB')); H,W=im.shape[:2]
 c,cx,cy,bw,bh=map(float,(DATA/r['label']).read_text().split()); assert int(c)==int(r['class_id'])
 x0,y0,x1,y1=(cx-bw/2)*W,(cy-bh/2)*H,(cx+bw/2)*W,(cy+bh/2)*H
 ix0,iy0=max(0,int(np.floor(x0))),max(0,int(np.floor(y0)))
 ix1,iy1=min(W,int(np.ceil(x1))),min(H,int(np.ceil(y1)))
 bg=im.copy(); bg[iy0:iy1,ix0:ix1]=114
 fruit=np.full_like(im,114); fruit[iy0:iy1,ix0:ix1]=im[iy0:iy1,ix0:ix1]
 scale=min(640/W,640/H); nw,nh=round(W*scale),round(H*scale)
 left=round((640-nw)/2-.1); top=round((640-nh)/2-.1)
 gt=np.array([x0*scale+left,y0*scale+top,x1*scale+left,y1*scale+top])
 inputs.append((r,gt,{k:torch.from_numpy(lb(image=a)).permute(2,0,1).float().unsqueeze(0)/255 for k,a in [('original',im),('background_only_box_removed',bg),('fruit_box_only',fruit)]}))
def xyxy(b): x,y,w,h=b; return [x,y,x+w,y+h]
def evaluate_archive(run):
 gt=json.loads((run/'groundtruth.coco.json').read_text()); pred=json.loads((run/'predictions.coco.json').read_text())
 assert {im['id'] for im in gt['images']}=={r['id'] for r in rows}
 tp=[]; sc=[]; pc=[]; tc=[]
 for im in gt['images']:
  aa=[a for a in gt['annotations'] if a['image_id']==im['id']]
  pp=sorted([p for p in pred if p['image_id']==im['id']],key=lambda p:-p['score'])
  gb=torch.tensor([xyxy(a['bbox']) for a in aa]); gc=np.array([a['category_id']-1 for a in aa]); tc.extend(gc)
  if not pp: continue
  pb=torch.tensor([xyxy(p['bbox']) for p in pp]); cls=np.array([p['category_id']-1 for p in pp]); score=np.array([p['score'] for p in pp])
  iou=box_iou(gb,pb).numpy()*(gc[:,None]==cls[None,:]); cor=np.zeros((len(pp),10),bool)
  for j,t in enumerate(np.linspace(.5,.95,10)):
   m=np.stack(np.nonzero(iou>=t),axis=1)
   if len(m):
    m=m[np.argsort(-iou[m[:,0],m[:,1]])]; m=m[np.unique(m[:,1],return_index=True)[1]]; m=m[np.unique(m[:,0],return_index=True)[1]]
    cor[m[:,1],j]=True
  tp.append(cor); sc.extend(score); pc.extend(cls)
 ap=ap_per_class(np.concatenate(tp),np.array(sc),np.array(pc),np.array(tc))[5]
 return dict(map50_95=float(ap.mean()),ap_ripe=float(ap[0].mean()),ap_unripe=float(ap[1].mean()))
def summarize(rr):
 return dict(n=len(rr),top_class_correct=sum(x['class_correct'] for x in rr),top_class_correct_score025=sum(x['class_correct'] and x['score']>=.25 for x in rr),localized_correct_iou050_score025=sum(x['class_correct'] and x['score']>=.25 and x['iou_gt']>=.5 for x in rr),median_score=float(np.median([x['score'] for x in rr])),median_iou=float(np.median([x['iou_gt'] for x in rr])),per_class={str(c):dict(n=sum(x['gt_class']==c for x in rr),top_correct=sum(x['gt_class']==c and x['class_correct'] for x in rr),localized_correct=sum(x['gt_class']==c and x['class_correct'] and x['score']>=.25 and x['iou_gt']>=.5 for x in rr)) for c in [0,1]})
result={}; t0=time.time()
for seed in range(5):
 run=RUN/f'group0/real_only__seed{seed}'; weights=run/'weights_selected.pt'
 ck=torch.load(weights,map_location='cpu',weights_only=False)
 net=DetectionModel(ck['cfg'],nc=ck['nc'],verbose=False); net.load_state_dict(ck['state_dict']); net.eval()
 detail={k:[] for k in ['original','background_only_box_removed','fruit_box_only']}
 for r,gt,tensors in inputs:
  for mode,x in tensors.items():
   raw=net(x); out=raw[0] if isinstance(raw,(list,tuple)) else raw
   scores=out[0,4:]; a=int(scores.max(0).values.argmax()); c=int(scores[:,a].argmax()); score=float(scores[c,a])
   bb=out[0,:4,a].numpy(); pred=np.array([bb[0]-bb[2]/2,bb[1]-bb[3]/2,bb[0]+bb[2]/2,bb[1]+bb[3]/2])
   inter=np.maximum(0,np.minimum(pred[2:],gt[2:])-np.maximum(pred[:2],gt[:2])).prod(); union=(pred[2:]-pred[:2]).prod()+(gt[2:]-gt[:2]).prod()-inter
   detail[mode].append(dict(id=r['id'],gt_class=int(r['class_id']),pred_class=c,class_correct=c==int(r['class_id']),score=score,iou_gt=float(inter/union)))
 archive=evaluate_archive(run); summary=json.loads((run/'run_summary.json').read_text())
 result[str(seed)]={'fresh_selected':{k:summarize(v) for k,v in detail.items()},'details':detail,'archived_final_recomputed':archive,'archived_final_reported':summary['validation'],'selected_update':summary['selected']['update'],'weights_sha256':hashlib.sha256(weights.read_bytes()).hexdigest()}
 (OUT/'detector_results.json').write_text(json.dumps(result,indent=2))
 print('seed',seed,'seconds',round(time.time()-t0),json.dumps(result[str(seed)]['fresh_selected']),flush=True)
print('DONE',round(time.time()-t0),flush=True)
