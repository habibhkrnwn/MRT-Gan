from pathlib import Path
import json,shutil,time
import torch,ultralytics
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
OUT=Path(__file__).resolve().parent;G=OUT.parents[3];RUN=G/'runs/19_mrtgan_detector/mrtgan_detector_v1/result';EVAL=OUT/'eval_dataset'
torch.set_num_threads(4)
excluded=set(json.loads((OUT/'hash_sensitivity.json').read_text())['excluded_for_sensitivity_only'])
for p in (EVAL/'images/valid').glob('*.jpg'):
 if p.stem in excluded: continue
 im=EVAL/'images/valid82'/p.name; lab=EVAL/'labels/valid82'/(p.stem+'.txt');im.parent.mkdir(parents=True,exist_ok=True);lab.parent.mkdir(parents=True,exist_ok=True)
 if not im.exists(): im.symlink_to(p.resolve())
 if not lab.exists(): shutil.copyfile(EVAL/'labels/valid'/(p.stem+'.txt'),lab)
yaml=EVAL/'data82.yaml';yaml.write_text(f'path: {json.dumps(str(EVAL))}\ntrain: images/train\nval: images/valid82\nnames:\n  0: ripe\n  1: unripe\n')
res={}
for seed in range(5):
 ck=torch.load(RUN/f'group0/real_only__seed{seed}/weights_selected.pt',map_location='cpu',weights_only=False)
 net=DetectionModel(ck['cfg'],nc=ck['nc'],verbose=False);net.load_state_dict(ck['state_dict']);net.eval();wrapper=YOLO(ck['cfg']);wrapper.model=net
 m=wrapper.val(data=str(yaml),split='val',device='cpu',batch=8,workers=0,imgsz=640,conf=.001,iou=.7,max_det=300,rect=True,plots=False,verbose=False,save_json=False,project=str(OUT/'validator_runs'),name=f'selected82_s{seed}',exist_ok=True)
 res[str(seed)]={'map50_95':float(m.box.map),'per_class_ap50_95':m.box.maps.tolist()}
 print('FINISHED',seed,res[str(seed)],flush=True)
 (OUT/'fresh_validator_sensitivity.json').write_text(json.dumps(dict(excluded=sorted(excluded),n=82,results=res,notes='Post-hoc removal of candidate near-duplicate images, fixed selected checkpoints. No official split modification.'),indent=2))
