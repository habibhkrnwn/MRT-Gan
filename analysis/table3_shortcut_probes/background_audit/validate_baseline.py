from pathlib import Path
import csv,json,hashlib,shutil,time
import torch,ultralytics
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
OUT=Path(__file__).resolve().parent; G=OUT.parents[3]; DATA=G/'runs/19_mrtgan_detector/local_reexport/dataset_v4'; RUN=G/'runs/19_mrtgan_detector/mrtgan_detector_v1/result'
EVAL=OUT/'eval_dataset'; torch.set_num_threads(4)
for r in csv.DictReader((DATA/'manifest.csv').open()):
 if r['split'] not in ('train','valid') or r['id']=='DSC_0093': continue
 im=EVAL/'images'/r['split']/(r['id']+'.jpg'); lab=EVAL/'labels'/r['split']/(r['id']+'.txt')
 im.parent.mkdir(parents=True,exist_ok=True); lab.parent.mkdir(parents=True,exist_ok=True)
 if not im.exists(): im.symlink_to((DATA/r['image']).resolve())
 if not lab.exists(): shutil.copyfile(DATA/r['label'],lab)
# All generated caches live in EVAL, outside the user's source dataset.
yaml=EVAL/'data.yaml'; yaml.write_text(f'path: {json.dumps(str(EVAL))}\ntrain: images/train\nval: images/valid\nnames:\n  0: ripe\n  1: unripe\n')
res={};t0=time.time()
for seed in range(5):
 run=RUN/f'group0/real_only__seed{seed}'; summary=json.loads((run/'run_summary.json').read_text());res[str(seed)]={}
 for mode,fn in [('selected','weights_selected.pt'),('final','weights.pt')]:
  ck=torch.load(run/fn,map_location='cpu',weights_only=False); net=DetectionModel(ck['cfg'],nc=ck['nc'],verbose=False); net.load_state_dict(ck['state_dict']);net.eval()
  wrapper=YOLO(ck['cfg']);wrapper.model=net
  m=wrapper.val(data=str(yaml),split='val',device='cpu',batch=8,workers=0,imgsz=640,conf=.001,iou=.7,max_det=300,rect=True,plots=False,verbose=False,save_json=False,project=str(OUT/'validator_runs'),name=f'{mode}_s{seed}',exist_ok=True)
  res[str(seed)][mode]=dict(map50_95=float(m.box.map),map50=float(m.box.map50),per_class_ap50_95=m.box.maps.tolist(),reported_map50_95=summary['selected']['validation']['map50_95'] if mode=='selected' else summary['validation']['map50_95'])
  (OUT/'fresh_validator.json').write_text(json.dumps(dict(ultralytics=ultralytics.__version__,torch=torch.__version__,device='cpu',results=res,notes='Validation-only fresh evaluation. No training. Explicit batch8/rect640/maxdet300/conf0.001/iou0.7; historical CUDA evaluator used Ultralytics8.4.163, local8.4.157.'),indent=2))
  print('FINISHED',seed,mode,res[str(seed)][mode],round(time.time()-t0),flush=True)
