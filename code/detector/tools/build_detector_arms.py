"""Build one YOLO dataset per confirmation arm from a reviewed benchmark bundle.

    python tools/build_detector_arms.py --bundle ../04_Hasil/benchmark_gen_v2 \
        --dataset ../10_Migrasi_Kaggle/dataset_regrouped --out ../04_Hasil/detector_arms \
        --arms copy_medium inpaint_s070

Every arm shares the same real images and the same validation split; they differ only in what
is added to training. `real_only` adds nothing. `geometry` adds plain geometric views of the
same parents through the same 512 crop path, which is the control the synthetic arms must
beat. A condition arm adds one accepted candidate per parent, chosen by the same deterministic
rule the screening used, never by a dev score.

The added images are 512 crops while the real images are full frames, exactly as in the v5.1
pilot export. Training uses a fixed update budget, so an arm with more images does not train
longer; that is enforced by the trainer, not here.
"""
from __future__ import annotations
import argparse, collections, json, shutil, sys
from pathlib import Path
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'src'))
import audit
from common import *

CLASSES=['ripe','unripe']


def pick(records,parents,per_parent=1):
    grouped=collections.defaultdict(list)
    for record in records:
        if record.get('status')=='accepted' and record['parent_id'] in parents:
            grouped[record['parent_id']].append(record)
    chosen=[]
    for pid in parents:
        ordered=sorted(grouped[pid],key=lambda r:(digest([r['candidate_id']]),r['candidate_id']))
        if not ordered:raise ValueError(f'No accepted candidate for parent {pid}')
        chosen.extend(ordered[:per_parent])
    return chosen


def geometry_views(dataset,parents,seed=0,views=1,size=512):
    """Affine views of the prepared crop. Image and mask move together; nothing is repainted."""
    import cv2
    out=[]
    for pid in parents:
        source,mask,geometry=dataset.prepare(pid,size)
        for view in range(views):
            rng=np.random.default_rng(stable_seed('geometry-arm',seed,pid,view))
            image=np.asarray(source);binary=np.asarray(mask)
            centre=(image.shape[1]/2,image.shape[0]/2)
            matrix=cv2.getRotationMatrix2D(centre,float(rng.uniform(-12,12)),float(rng.uniform(.9,1.1)))
            matrix[0,2]+=float(rng.uniform(-.1,.1))*image.shape[1]
            matrix[1,2]+=float(rng.uniform(-.1,.1))*image.shape[0]
            moved=cv2.warpAffine(image,matrix,(image.shape[1],image.shape[0]),flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)
            moved_mask=cv2.warpAffine(binary,matrix,(binary.shape[1],binary.shape[0]),flags=cv2.INTER_NEAREST)
            if not (moved_mask>0).any():raise ValueError(f'Geometry view lost the fruit: {pid}/{view}')
            out.append(dict(candidate_id=f'geometry__{pid}__{view}',parent_id=pid,
                            class_id=dataset.by_id[pid]['class_id'],
                            image=Image.fromarray(moved),mask=Image.fromarray(moved_mask),
                            trace=dict(affine=matrix.tolist(),source_geometry=geometry)))
    return out


def yolo_line(class_id,box,width,height):
    x,y,w,h=box
    return f'{int(class_id)} {(x+w/2)/width:.9f} {(y+h/2)/height:.9f} {w/width:.9f} {h/height:.9f}\n'


def write_arm(dataset,name,added,parents,out,size=512):
    """One arm directory: real train images, the arm's added images, and the shared validation."""
    root=Path(out)/name
    for split in ('train','val'):
        (root/'images'/split).mkdir(parents=True,exist_ok=True)
        (root/'labels'/split).mkdir(parents=True,exist_ok=True)
    manifest=dict(arm=name,real=[],added=[],validation=0,size=size)
    for pid in parents:
        row=dataset.by_id[pid]
        image=safe(dataset.root,row['image']);label=safe(dataset.root,row['label'])
        shutil.copyfile(image,root/'images'/'train'/f'{pid}.jpg')
        shutil.copyfile(label,root/'labels'/'train'/f'{pid}.txt')
        manifest['real'].append(dict(id=pid,class_id=row['class_id'],image_sha256=row['image_sha256']))
    for item in added:
        name_png=f"{item['candidate_id']}.png"
        image=item['image'];mask=item['mask']
        image.save(root/'images'/'train'/name_png)
        box=mask_box(mask)
        (root/'labels'/'train'/f"{item['candidate_id']}.txt").write_text(
            yolo_line(item['class_id'],box,image.width,image.height))
        manifest['added'].append(dict(id=item['candidate_id'],parent_id=item['parent_id'],
                                      class_id=item['class_id'],bbox=box,
                                      image_sha256=sha(root/'images'/'train'/name_png)))
    for row in dataset.rows:
        if row['split']!='valid':continue
        pid=row['id']
        shutil.copyfile(safe(dataset.root,row['image']),root/'images'/'val'/f'{pid}.jpg')
        shutil.copyfile(safe(dataset.root,row['label']),root/'labels'/'val'/f'{pid}.txt')
        manifest['validation']+=1
    (root/'data.yaml').write_text(
        f'path: {root.resolve()}\ntrain: images/train\nval: images/val\n'
        f'names:\n  0: {CLASSES[0]}\n  1: {CLASSES[1]}\n')
    manifest['counts']=dict(real=len(manifest['real']),added=len(manifest['added']),
                            validation=manifest['validation'],
                            added_per_class=dict(collections.Counter(a['class_id'] for a in manifest['added'])),
                            real_per_class=dict(collections.Counter(r['class_id'] for r in manifest['real'])))
    manifest['note']=('real images are full frames, added images are 512 crops; the update budget '
                      'is fixed by the trainer so more images never means longer training')
    write_json(root/'arm_manifest.json',manifest)
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--bundle',required=True)
    parser.add_argument('--dataset',default=None)
    parser.add_argument('--out',required=True)
    parser.add_argument('--arms',nargs='*',default=['copy_medium','inpaint_s070'])
    parser.add_argument('--exclude',nargs='*',default=['DSC_0093'])
    parser.add_argument('--per-parent',type=int,default=1)
    parser.add_argument('--geometry-seed',type=int,default=0)
    parser.add_argument('--size',type=int,default=512)
    args=parser.parse_args(argv)

    dataset=Dataset(root=args.dataset)
    parents=[pid for pid in dataset.scope_ids if pid not in set(args.exclude)]
    records=read_json(Path(args.bundle)/'records.json')
    if any(r['status']=='needs_review' for r in records):
        raise SystemExit('Unreviewed candidates present; run ingest_review first')
    summaries={}
    summaries['real_only']=write_arm(dataset,'real_only',[],parents,args.out,args.size)
    views=max(1,args.per_parent)
    summaries['geometry']=write_arm(dataset,'geometry',
                                    geometry_views(dataset,parents,args.geometry_seed,views,args.size),
                                    parents,args.out,args.size)
    for arm in args.arms:
        chosen=pick([r for r in records if r['condition']==arm],parents,args.per_parent)
        added=[dict(candidate_id=r['candidate_id'],parent_id=r['parent_id'],class_id=r['class_id'],
                    image=Image.open(Path(args.bundle)/r['image']).convert('RGB'),
                    mask=Image.open(Path(args.bundle)/r['mask']).convert('L'),trace=r.get('trace',{}))
               for r in chosen]
        summaries[arm]=write_arm(dataset,arm,added,parents,args.out,args.size)
    write_json(Path(args.out)/'arms_summary.json',
               dict(arms={k:v['counts'] for k,v in summaries.items()},parents=parents,
                    excluded=list(args.exclude),bundle=str(args.bundle),
                    scope=dataset.scope_hash,manifest=EXPECTED_MANIFEST,code=code_digest(),
                    test_used=False))
    for name,summary in summaries.items():print(name,summary['counts'])
    return summaries


if __name__=='__main__':main()
