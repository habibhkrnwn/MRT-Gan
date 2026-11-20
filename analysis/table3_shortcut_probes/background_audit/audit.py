from pathlib import Path
import csv,json,hashlib,collections,time,platform
import numpy as np
from PIL import Image
from skimage.color import rgb2lab,rgb2hsv
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score,balanced_accuracy_score,confusion_matrix

OUT=Path(__file__).resolve().parent
G=OUT.parents[3]
DATA=G/'runs/19_mrtgan_detector/local_reexport/dataset_v4'
OFF=G.parents[2]/'transfer_v4_to_v5/melon_outputs/melon_dataset_regrouped_v4'
BASE=G/'melon_dataset_base_v1'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False))
rows0=list(csv.DictReader((DATA/'manifest.csv').open()))
rows=[r for r in rows0 if r['id']!='DSC_0093']
active=[r for r in rows if r['split'] in ('train','valid')]
groups={r['photograph_group']:r for r in csv.DictReader((OFF/'groups.csv').open())}
base={r['id']:r for r in csv.DictReader((BASE/'manifest.csv').open())}
counts=collections.Counter((r['split'],r['class_id'],'DSC' if r['id'].startswith('DSC') else 'oke') for r in rows)
session_splits=collections.defaultdict(set)
for r in rows: session_splits[groups[r['photograph_group']]['session']].add(r['split'])
rep={'manifest_sha256':sha(DATA/'manifest.csv'),'official_manifest_sha256':sha(OFF/'manifest.csv'),'groups_sha256':sha(OFF/'groups.csv'),
 'manifest_rows':len(rows0),'excluded':['DSC_0093'],'effective_rows':len(rows),'images_opened':len(active),'test_images_opened':0,
 'split_class_source':[dict(split=s,class_id=int(c),source=f,n=n) for (s,c,f),n in sorted(counts.items())],
 'sessions_cross_split':{k:sorted(v) for k,v in session_splits.items() if len(v)>1},'issues':[], 'verified':collections.Counter()}
F={k:[] for k in ['bg_colour','bg_margin15_colour','fruit_box_colour','geometry','dimensions']}; dh=[]; dhfruit=[]; detail=[]
def colour(px):
 lab=rgb2lab(px[None])[0]; hsv=rgb2hsv(px[None])[0]
 z=[px.mean(0),px.std(0),lab.mean(0),lab.std(0),np.percentile(lab,[10,50,90],axis=0).ravel()]
 for c,b in [(0,16),(1,8),(2,8)]: z.append(np.histogram(hsv[:,c],bins=b,range=(0,1))[0]/len(px))
 return np.concatenate(z)
def dhash(im):
 a=np.asarray(im.convert('L').resize((17,16),Image.Resampling.LANCZOS))
 return np.packbits(a[:,1:]>a[:,:-1])
for i,r in enumerate(active):
 p=DATA/r['image']; lp=DATA/r['label']
 if not p.exists() or not lp.exists(): raise FileNotFoundError(r['id'])
 rep['verified']['symlink_images']+=int(p.is_symlink())
 ih,lh=sha(p),sha(lp)
 for key,val in [('image_sha256',ih),('label_sha256',lh)]:
  if r[key]!=val: rep['issues'].append([r['id'],key,'mismatch'])
  else: rep['verified'][key]+=1
 if p.resolve()==(BASE/base[r['id']]['image']).resolve(): rep['verified']['source_target_match']+=1
 lines=lp.read_text().strip().splitlines()
 assert len(lines)==1,(r['id'],lines)
 c,cx,cy,bw,bh=map(float,lines[0].split()); assert int(c)==int(r['class_id'])
 assert 0<bw<=1 and 0<bh<=1 and 0<=cx<=1 and 0<=cy<=1
 im=Image.open(p).convert('RGB'); W,H=im.size
 if (W,H)!=(int(r['width']),int(r['height'])): rep['issues'].append([r['id'],'dimensions_mismatch'])
 x0,y0,x1,y1=(cx-bw/2)*W,(cy-bh/2)*H,(cx+bw/2)*W,(cy+bh/2)*H
 if min(x0,y0)<-1 or x1>W+1 or y1>H+1: rep['issues'].append([r['id'],'box_outside'])
 sm=im.resize((512,round(H*512/W))); a=np.asarray(sm,dtype=float)/255; s=512/W
 for name,margin in [('bg_colour',0),('bg_margin15_colour',.15)]:
  dx=(x1-x0)*margin; dy=(y1-y0)*margin
  m=np.zeros(a.shape[:2],bool); m[max(0,int((y0-dy)*s)):min(a.shape[0],int((y1+dy)*s)+1),max(0,int((x0-dx)*s)):min(a.shape[1],int((x1+dx)*s)+1)]=True
  assert (~m).sum()>0
  F[name].append(colour(a[~m]))
  if margin==0: F['fruit_box_colour'].append(colour(a[m]))
 F['geometry'].append([bw*bh,bw*W/(bh*H),cx,cy]); F['dimensions'].append([W,H,W/H])
 dh.append(dhash(im)); dhfruit.append(dhash(im.crop((x0,y0,x1,y1))))
 detail.append(dict(id=r['id'],split=r['split'],class_id=int(c),session=groups[r['photograph_group']]['session'],image_sha256=ih,label_sha256=lh,width=W,height=H,box_area=bw*bh))
 if i%100==0: print('images',i,flush=True)
rep['verified']=dict(rep['verified'])
tr=np.array([r['split']=='train' for r in active]); va=~tr; y=np.array([int(r['class_id']) for r in active]); source=np.array([r['id'].startswith('oke') for r in active],int)
ids=np.array([r['id'] for r in active]); sessions=np.array([groups[r['photograph_group']]['session'] for r in active])
exact=collections.defaultdict(list)
for r in detail: exact[r['image_sha256']].append(r)
rep['exact_duplicates_train_valid']=[[r['id'] for r in rr] for rr in exact.values() if len({r['split'] for r in rr})>1]
near=[]
for name,arr in [('whole',dh),('fruit',dhfruit)]:
 a=np.stack(arr)
 for idx in np.where(va)[0]:
  dist=np.unpackbits(np.bitwise_xor(a[tr],a[idx]),axis=1).sum(1); j=int(np.argmin(dist))
  if int(dist[j])<=26: near.append(dict(view=name,valid=ids[idx],train=ids[tr][j],hamming=int(dist[j])))
rep['near_duplicate_flags_threshold26']=near
save('integrity.json',rep); save('train_valid_inventory.json',detail)
F={k:np.asarray(v) for k,v in F.items()}; np.savez_compressed(OUT/'features.npz',**F,ids=ids,y=y,train=tr,sessions=sessions)
def model(): return make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=5000,class_weight='balanced'))
def score(yy,p): return dict(n=len(yy),auc=float(roc_auc_score(yy,p)) if len(set(yy))==2 else None,balanced_accuracy=float(balanced_accuracy_score(yy,p>=.5)),confusion_matrix=confusion_matrix(yy,p>=.5,labels=[0,1]).tolist())
probes={}; dsc=tr&(source==0); ix=np.where(dsc)[0]; folds=list(GroupKFold(5).split(ix,y[ix],sessions[ix])); foldmeta=[]
for a,b in folds:
 assert not set(sessions[ix[a]])&set(sessions[ix[b]])
 foldmeta.append(dict(train_counts=np.bincount(y[ix[a]],minlength=2).tolist(),heldout_counts=np.bincount(y[ix[b]],minlength=2).tolist(),heldout_sessions=sorted(set(sessions[ix[b]]))))
for k,X in F.items():
 clf=model().fit(X[tr],y[tr]); p=clf.predict_proba(X[va])[:,1]
 oof=np.empty(len(ix)); individual=[]
 for a,b in folds:
  pp=model().fit(X[ix[a]],y[ix[a]]).predict_proba(X[ix[b]])[:,1]; oof[b]=pp
  individual.append(score(y[ix[b]],pp))
 probes[k]={'train_to_valid':score(y[va],p),'within_DSC_train_session_grouped_OOF':score(y[ix],oof),'folds':individual}
 print('probe',k,probes[k]['train_to_valid'],probes[k]['within_DSC_train_session_grouped_OOF'],flush=True)
probes['fold_definition']=foldmeta
probes['note']='Only DSC train used for DSC CV training and scoring. Pooled OOF AUC is exploratory; folds have unequal composition. Fruit-box features can include background. Background colour features exclude box pixels entirely.'
save('probes.json',probes)
# The name rule is a diagnostic of source/class confounding, not a deployable classifier.
save('source_rule.json',{'rule':'DSC -> ripe, oke -> unripe','train':score(y[tr],source[tr]),'valid':score(y[va],source[va])})
# Historical subset records: do not represent a new training run.
hist={}
runs=G/'runs/09_texture_subset_detectors/kaggle_v1_output/texture_subset_detectors_v1/result'
old=json.loads((runs/'results.json').read_text())
for pct,key in [('p025','25'),('p050','50')]:
 m=json.loads((runs/pct/'real_only_manifest.json').read_text()); real=m['real']
 assert all(r['id'] in set(ids[tr]) for r in real)
 hist[pct]={'n':len(real),'class_counts':dict(collections.Counter(str(r['class_id']) for r in real)),'manifest_sha256':sha(runs/pct/'real_only_manifest.json'),'report':old['report'][key]}
save('historical_subsets.json',hist)
save('runtime.json',dict(python=platform.python_version(),script_sha256=sha(Path(__file__)),protocol_sha256=sha(OUT/'PROTOCOL.md'),numpy=np.__version__,new_gpu_minutes=0))
print('DONE',rep['verified'],rep['issues'],hist['p025']['n'],hist['p050']['n'],flush=True)
