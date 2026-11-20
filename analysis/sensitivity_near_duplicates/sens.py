import json,sys,glob,os
import numpy as np, torch
from ultralytics.utils.metrics import ap_per_class, box_iou
G=sys.argv[1]
def xyxy(b): x,y,w,h=b; return [x,y,x+w,y+h]
def evaluate(run,exclude=()):
    gt=json.load(open(run+'/groundtruth.coco.json')); pr=json.load(open(run+'/predictions.coco.json'))
    ious=torch.linspace(0.5,0.95,10)
    TP=[];CONF=[];PC=[];TC=[]
    by={}
    for p in pr: by.setdefault(p['image_id'],[]).append(p)
    for im in gt['images']:
        i=im['id']
        if i in exclude: continue
        g=[a for a in gt['annotations'] if a['image_id']==i]
        gb=torch.tensor([xyxy(a['bbox']) for a in g],dtype=torch.float32).reshape(-1,4); gc=torch.tensor([a['category_id']-1 for a in g])
        ps=sorted(by.get(i,[]),key=lambda p:-p['score'])
        TC+= gc.tolist()
        if not ps: continue
        pb=torch.tensor([xyxy(p['bbox']) for p in ps],dtype=torch.float32); pc=torch.tensor([p['category_id']-1 for p in ps]); sc=torch.tensor([p['score'] for p in ps])
        correct=np.zeros((len(ps),10),dtype=bool)
        if len(g):
            iou=box_iou(gb,pb)*(gc[:,None]==pc[None,:])
            iou=iou.numpy()
            for k,t in enumerate(ious.tolist()):
                m=np.nonzero(iou>=t)
                if m[0].shape[0]:
                    mt=np.array(m).T; mt=mt[iou[mt[:,0],mt[:,1]].argsort()[::-1]]
                    mt=mt[np.unique(mt[:,1],return_index=True)[1]]; mt=mt[np.unique(mt[:,0],return_index=True)[1]]
                    correct[mt[:,1].astype(int),k]=True
        TP.append(correct);CONF+=sc.tolist();PC+=pc.tolist()
    r=ap_per_class(np.concatenate(TP),np.array(CONF),np.array(PC),np.array(TC))
    ap=r[5]  # (nc,10)
    return float(ap.mean()), float(ap[1].mean()) if ap.shape[0]>1 else float('nan')
EX=('oke6396','oke7065')
arms={
 'p100':{'real_only':'runs/03_balancing_full_train/detector_7_arms/balance_detector/group*/real_only__seed*',
         'oversample':'runs/03_balancing_full_train/detector_7_arms/balance_detector/group*/oversample_r100__seed*',
         'earlier_cyclegan':'runs/03_balancing_full_train/detector_7_arms/balance_detector/group*/cyclegan_r100__seed*',
         'G3':'runs/06_texture_ranked_detector/kaggle_v2_output/texture_ranked_v1/result/group*/texture_G3_ranked_r100__seed*'},
}
for b in ('p025','p050'):
    arms[b]={a:f'runs/09_texture_subset_detectors/kaggle_v1_output/texture_subset_detectors_v1/result/{b}/{a}__seed*' for a in ('real_only','oversample','geometry','synthetic_G3')}
out={}
first=True
for b,d in arms.items():
    out[b]={}
    for a,pat in d.items():
        runs=sorted(glob.glob(os.path.join(G,pat)))
        full=[];sub=[];ref=[]
        for r in runs:
            f=evaluate(r);s=evaluate(r,EX);full.append(f);sub.append(s)
            ref.append(json.load(open(r+'/validation.json'))['map50_95'])
        if first: print('check evaluator vs validation.json:',[round(x[0],4) for x in full[:3]],[round(x,4) for x in ref[:3]]);first=False
        out[b][a]=dict(n=len(runs),all87=float(np.mean([x[0] for x in full])),excl85=float(np.mean([x[0] for x in sub])),
                       apU_all=float(np.mean([x[1] for x in full])),apU_excl=float(np.mean([x[1] for x in sub])),
                       per_seed=[dict(run=os.path.basename(r),all=f[0],excl=s[0]) for r,f,s in zip(runs,full,sub)])
        print(b,a,len(runs),'all %.4f excl %.4f | APu %.4f -> %.4f'%(out[b][a]['all87'],out[b][a]['excl85'],out[b][a]['apU_all'],out[b][a]['apU_excl']),flush=True)
json.dump(out,open(sys.argv[2],'w'),indent=1)
