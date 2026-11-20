"""Confirmation trainer: one detector per arm and seed, on a fixed optimizer-update budget.

    python tools/train_detector.py --arm ../04_Hasil/detector_arms/copy_medium --seed 0 \
        --updates 760 --out ../04_Hasil/detector_runs

An arm that adds images must not train longer than `real_only`, otherwise a gain confounds
better data with more optimizer steps. The loop therefore samples a fixed number of batches,
not a fixed number of passes, and every arm shares batch size, learning rate, schedule length,
augmentation and checkpoint rule. Only the training images differ.

Validation uses the frozen validation split. The test split is never opened.
"""
from __future__ import annotations
import argparse, json, math, random, sys, time
from pathlib import Path
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'src'))
from common import *

IMAGE_SIZE=640
EXPORT_CONF=0.001      # below every operating threshold under study; see export_predictions
EXPORT_MAX_DET=100


def letterbox(image,boxes,size=IMAGE_SIZE):
    """Resize keeping aspect ratio, pad to square, and move the boxes with the image."""
    width,height=image.size;scale=size/max(width,height)
    resized=image.resize((max(1,round(width*scale)),max(1,round(height*scale))),Image.Resampling.BILINEAR)
    canvas=Image.new('RGB',(size,size),(114,114,114))
    left=(size-resized.width)//2;top=(size-resized.height)//2
    canvas.paste(resized,(left,top))
    moved=[]
    for class_id,cx,cy,bw,bh in boxes:
        moved.append([class_id,(cx*width*scale+left)/size,(cy*height*scale+top)/size,
                      bw*width*scale/size,bh*height*scale/size])
    return canvas,moved


def read_labels(path):
    rows=[]
    for line in Path(path).read_text().splitlines():
        if not line.strip():continue
        values=line.split()
        rows.append([int(values[0])]+[float(v) for v in values[1:5]])
    return rows


class ArmDataset:
    def __init__(self,arm,split='train'):
        self.items=sorted((arm/'images'/split).glob('*'))
        self.labels=arm/'labels'/split
        if not self.items:raise ValueError(f'No images in {arm}/{split}')

    def __len__(self):return len(self.items)

    def load(self,index,flip=False):
        path=self.items[index]
        image=Image.open(path).convert('RGB')
        boxes=read_labels(self.labels/f'{path.stem}.txt')
        image,boxes=letterbox(image,boxes)
        if flip:
            image=image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            boxes=[[c,1-x,y,w,h] for c,x,y,w,h in boxes]
        return path.stem,image,boxes


def to_tensor(images):
    import torch
    stack=np.stack([np.asarray(im,np.float32)/255 for im in images]).transpose(0,3,1,2)
    return torch.from_numpy(stack)


ARCHITECTURES={'yolov8n':dict(cfg='yolov8n.yaml',weights='yolov8n.pt'),
               'yolov8s':dict(cfg='yolov8s.yaml',weights='yolov8s.pt'),
               'yolov8m':dict(cfg='yolov8m.yaml',weights='yolov8m.pt')}


def build_model(pretrained=True,architecture='yolov8n'):
    """Plain DetectionModel with the stock hyperparameters attached, as the loss expects.

    Every published number so far comes from one architecture, which is a stated limit of the
    benchmark. The name is a parameter so a second architecture needs a config change rather
    than a code change; the update budget, batch, schedule and augmentation stay identical,
    otherwise the comparison measures training length instead of architecture.

    RF-DETR is deliberately absent: it needs the `rfdetr` package and a different training
    loop, so it is a separate piece of work rather than another entry in this table.
    """
    import copy
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils import DEFAULT_CFG
    if architecture not in ARCHITECTURES:
        raise ValueError(f'Unknown architecture {architecture}; known: {sorted(ARCHITECTURES)}')
    spec=ARCHITECTURES[architecture]
    net=DetectionModel(spec['cfg'],nc=2,verbose=False)
    if pretrained:net.load(YOLO(spec['weights']).model)
    net.args=copy.deepcopy(DEFAULT_CFG)
    return net


def batches(dataset,seed,updates,batch):
    """Fixed number of batches, drawn with a seeded generator; arms train the same length."""
    rng=random.Random(seed)
    order=[]
    for _ in range(updates):
        while len(order)<batch:
            index=list(range(len(dataset)));rng.shuffle(index);order.extend(index)
        yield [order.pop(0) for _ in range(batch)]


def evaluate_copy(net,arm,architecture='yolov8n'):
    """Validation of a frozen copy of the training network, so the live one is untouched.

    Building the Ultralytics wrapper draws from the global random generators and the
    validator resets the CPU thread count, which changes floating-point reduction order.
    Both are saved and restored, so a run with periodic validation trains exactly like
    one without it.
    """
    import copy
    import torch
    from ultralytics import YOLO
    states=(random.getstate(),np.random.get_state(),torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    threads=torch.get_num_threads()
    try:
        wrapper=YOLO(ARCHITECTURES[architecture]['cfg']);wrapper.model=copy.deepcopy(net).eval()
        metrics=wrapper.val(data=str(Path(arm)/'data.yaml'),split='val',verbose=False,plots=False)
    finally:
        random.setstate(states[0]);np.random.set_state(states[1]);torch.set_rng_state(states[2])
        if states[3] is not None:torch.cuda.set_rng_state_all(states[3])
        torch.set_num_threads(threads)
    box=metrics.box
    return dict(map50_95=float(box.map),map50=float(box.map50),
                per_class={name:dict(ap50_95=float(box.maps[i])) for i,name in enumerate(['ripe','unripe'])},
                precision=[float(v) for v in box.p],recall=[float(v) for v in box.r])


def train(arm,seed,updates,batch,lr,out,pretrained=True,budget=None,architecture='yolov8n',eval_every=0):
    """Fixed-budget training. With `eval_every` > 0 the validation split is scored every
    `eval_every` updates and the best state by validation mAP50:95 is kept; the schedule,
    batches and final weights are identical to a run without it."""
    import torch
    arm=Path(arm);out=Path(out)/f'{arm.name}__seed{seed}';out.mkdir(parents=True,exist_ok=True)
    manifest=read_json(arm/'arm_manifest.json')
    identity=dict(version=VERSION,arm=arm.name,seed=seed,updates=updates,batch=batch,lr=lr,
                  architecture=architecture,
                  image_size=IMAGE_SIZE,counts=manifest['counts'],code=code_digest(),
                  scope=manifest.get('note',''))
    bind_identity(out,identity)
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    net=build_model(pretrained,architecture).to(device).train()
    optimizer=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=updates)
    dataset=ArmDataset(arm,'train')
    history=[];started=time.time()
    valid_history=[];best=None
    for step,index in enumerate(batches(dataset,seed,updates,batch)):
        if budget:budget.check()
        images=[];targets=[]
        for position,i in enumerate(index):
            flip=(stable_seed('flip',seed,step,position)%2==1)
            _,image,boxes=dataset.load(i,flip)
            images.append(image)
            for class_id,cx,cy,bw,bh in boxes:targets.append([position,class_id,cx,cy,bw,bh])
        pixels=to_tensor(images).to(device)
        target=torch.tensor(targets,dtype=torch.float32,device=device)
        payload=dict(img=pixels,batch_idx=target[:,0],cls=target[:,1:2],bboxes=target[:,2:6])
        loss,items=net(payload)
        optimizer.zero_grad(set_to_none=True)
        loss.sum().backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),10.0)
        optimizer.step();scheduler.step()
        if step%25==0 or step==updates-1:
            history.append(dict(step=step,loss=float(loss.sum().item()),
                                lr=scheduler.get_last_lr()[0],seconds=round(time.time()-started,1)))
            print(history[-1],flush=True)
        done=step+1
        if eval_every and done%eval_every==0 and done<updates:
            # The random stream of the batches is untouched: validation draws no samples.
            scores=evaluate_copy(net,arm,architecture);net.train()
            valid_history.append(dict(update=done,**scores))
            print(dict(update=done,valid_map50_95=scores['map50_95']),flush=True)
            if best is None or scores['map50_95']>best['map50_95']:
                best=dict(update=done,map50_95=scores['map50_95'],
                          state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()})
    write_json(out/'loss.json',history)
    train.valid_history=valid_history;train.best=best
    from ultralytics import YOLO
    wrapper=YOLO(ARCHITECTURES[architecture]['cfg']);wrapper.model=net.eval()
    return wrapper,out,identity,round(time.time()-started,1)


def validate(model,arm,out):
    """Ultralytics validation on the frozen validation split; test is never touched."""
    metrics=model.val(data=str(Path(arm)/'data.yaml'),split='val',verbose=False,plots=False)
    box=metrics.box
    summary=dict(map50_95=float(box.map),map50=float(box.map50),
                 per_class={name:dict(ap50_95=float(box.maps[i]))
                            for i,name in enumerate(['ripe','unripe'])},
                 precision=[float(v) for v in box.p],recall=[float(v) for v in box.r])
    write_json(Path(out)/'validation.json',summary)
    return summary


def export_predictions(model,arm,out,conf=EXPORT_CONF,max_det=EXPORT_MAX_DET):
    """Per-image validation predictions and ground truth in COCO form.

    Aggregate AP answers a different question from a chosen operating point, so the
    calibration work needs the raw boxes, not `validation.json`. The export keeps the
    detector's own class-aware NMS and stops there: class bias and cross-class NMS belong
    to the decision rule under study and must not be baked in here. `conf` is deliberately
    far below any operating threshold, otherwise the low-score boxes that a bias would
    promote are already gone.

    `image_id` is the file stem, which is the manifest id, so a reader can group images by
    `photograph_group` without a second lookup table.
    """
    arm=Path(arm);out=Path(out)
    images=sorted((arm/'images'/'val').glob('*'))
    if not images:raise ValueError(f'No validation images in {arm}')
    labels=arm/'labels'/'val'
    coco_images=[];annotations=[];predictions=[];ann_id=0
    for path in images:
        with Image.open(path) as im:width,height=im.size
        coco_images.append(dict(id=path.stem,file_name=path.name,width=width,height=height))
        for class_id,cx,cy,bw,bh in read_labels(labels/f'{path.stem}.txt'):
            ann_id+=1
            annotations.append(dict(id=ann_id,image_id=path.stem,category_id=class_id+1,
                                    bbox=[round((cx-bw/2)*width,2),round((cy-bh/2)*height,2),
                                          round(bw*width,2),round(bh*height,2)],
                                    area=round(bw*width*bh*height,2),iscrowd=0))
        result=model.predict(source=str(path),imgsz=IMAGE_SIZE,conf=conf,max_det=max_det,
                             verbose=False)[0]
        boxes=result.boxes
        for i in range(len(boxes)):
            x1,y1,x2,y2=[float(v) for v in boxes.xyxy[i]]
            predictions.append(dict(image_id=path.stem,category_id=int(boxes.cls[i])+1,
                                    bbox=[round(x1,2),round(y1,2),round(x2-x1,2),round(y2-y1,2)],
                                    score=round(float(boxes.conf[i]),6)))
    write_json(out/'groundtruth.coco.json',dict(
        images=coco_images,annotations=annotations,
        categories=[dict(id=1,name='ripe'),dict(id=2,name='unripe')]))
    write_json(out/'predictions.coco.json',predictions)
    return dict(images=len(coco_images),annotations=len(annotations),
                predictions=len(predictions),conf=conf,max_det=max_det,
                note='class-aware NMS only; class bias and cross-class NMS are left to the caller')


def save_weights(model,out):
    """Keep the trained weights so a later question does not need another GPU session."""
    import torch
    path=Path(out)/'weights.pt'
    torch.save(dict(state_dict=model.model.state_dict(),nc=2,names=['ripe','unripe'],
                    cfg=getattr(model,'cfg','yolov8n.yaml'),image_size=IMAGE_SIZE,
                    version=VERSION),path)
    return dict(path=path.name,bytes=path.stat().st_size)


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--arm',required=True)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--updates',type=int,default=760)
    parser.add_argument('--batch',type=int,default=8)
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--out',required=True)
    parser.add_argument('--hours',type=float,default=None)
    parser.add_argument('--architecture',default='yolov8n',choices=sorted(ARCHITECTURES))
    parser.add_argument('--no-export',action='store_true',
                        help='skip per-image predictions and weights')
    parser.add_argument('--eval-every',type=int,default=0,
                        help='score validation every N updates and keep the best state by mAP50:95 '
                             '(0 keeps the historical protocol: one validation at the end)')
    args=parser.parse_args(argv)
    budget=Budget(hours=args.hours) if args.hours else None
    model,out,identity,seconds=train(args.arm,args.seed,args.updates,args.batch,args.lr,args.out,
                                     budget=budget,architecture=args.architecture,eval_every=args.eval_every)
    summary=validate(model,args.arm,out)
    export=weights=None
    if not args.no_export:
        export=export_predictions(model,args.arm,out)
        weights=save_weights(model,out)
    selection=None
    if args.eval_every:
        # The final state competes with the periodic ones; ties keep the earlier update.
        history=train.valid_history+[dict(update=args.updates,**summary)]
        write_json(out/'valid_history.json',history)
        best=train.best
        if best is None or summary['map50_95']>best['map50_95']:
            selection=dict(update=args.updates,validation=summary,weights='weights.pt' if weights else None)
        else:
            import torch
            chosen=next(h for h in history if h['update']==best['update'])
            torch.save(dict(state_dict=best['state'],nc=2,names=['ripe','unripe'],
                            cfg=ARCHITECTURES[args.architecture]['cfg'],image_size=IMAGE_SIZE,version=VERSION,
                            update=best['update']),out/'weights_selected.pt')
            selection=dict(update=best['update'],validation={k:v for k,v in chosen.items() if k!='update'},
                           weights='weights_selected.pt')
        selection['rule']='best validation mAP50:95 over periodic evaluations and the final state'
    write_json(out/'run_summary.json',dict(identity=identity,training_seconds=seconds,
                                           validation=summary,selected=selection,export=export,weights=weights,
                                           eval_every=args.eval_every,test_used=False))
    print(json.dumps(dict(arm=Path(args.arm).name,seed=args.seed,seconds=seconds,**summary),indent=1))


if __name__=='__main__':main()
