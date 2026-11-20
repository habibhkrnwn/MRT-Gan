"""Actual generator adapters, compact configuration presets, immutable model locks."""
from __future__ import annotations
import gc, inspect, time, urllib.request
from pathlib import Path
import numpy as np
from PIL import Image
from common import *

REPOS={
 'sd':'stable-diffusion-v1-5/stable-diffusion-v1-5',
 'inpaint':'stable-diffusion-v1-5/stable-diffusion-inpainting',
 'canny':'lllyasviel/control_v11p_sd15_canny',
 'softedge':'lllyasviel/control_v11p_sd15_softedge',
 'hed':'lllyasviel/Annotators',
 'ip':'h94/IP-Adapter',
 'turbo':'stabilityai/sd-turbo',
 'dino':'IDEA-Research/grounding-dino-tiny',
}
NEGATIVE='watermelon, sliced fruit, cut open fruit, red flesh, pink flesh, pumpkin, deformed fruit, duplicate fruit, artificial texture, text, watermark'
CONTEXT='a photograph in a greenhouse, leaves, vines, natural daylight'

def prompt(cls,slot):
    appearance=['golden yellow ripe','pale green unripe'][cls]
    variants=['soft natural daylight','gentle leaf shadows']
    return f'a realistic photograph of a whole {appearance} melon hanging on the vine, natural fruit skin texture, '+variants[slot%len(variants)]


def plan(profile='core',custom=None):
    base=dict(steps=25,guidance=7.5,strength=.5,control_scale=1.,ip_scale=.6,translate=.10,scale_jitter=.10,rotation=8.,size=512)
    specifications=[
      ('copy_mild','copy_paste',dict(translate=.08,scale_jitter=.08,rotation=6.)),
      ('copy_medium','copy_paste',dict(translate=.15,scale_jitter=.15,rotation=12.)),
      ('canny_s035','canny',dict(strength=.35)),('canny_s055','canny',dict(strength=.55)),
      ('recon_s035','recon',dict(strength=.35)),('recon_s055','recon',dict(strength=.55)),
      ('inpaint_s045','inpaint_fg',dict(strength=.45)),('inpaint_s070','inpaint_fg',dict(strength=.70)),
      ('background_s075','inpaint_bg',dict(strength=.75)),
    ]
    if profile=='round2':
        # Narrow second round: the cheap control plus the three configurations that survived
        # screening round 1. ReCon stays for the mechanism question, not for a proven gain.
        keep={'copy_medium','canny_s035','inpaint_s070','recon_s055'}
        specifications=[s for s in specifications if s[0] in keep]
    elif profile=='extended':
        specifications += [('softedge_s035','softedge',dict(strength=.35)),('softedge_s055','softedge',dict(strength=.55)),
          ('ip_s070_a040','ip_inpaint',dict(strength=.70,ip_scale=.4)),('ip_s070_a070','ip_inpaint',dict(strength=.70,ip_scale=.7)),
          ('img2img_s035','img2img',dict(strength=.35)),('img2img_s055','img2img',dict(strength=.55)),
          ('turbo_s050','turbo_img2img',dict(strength=.5,steps=4,guidance=0.)),('turbo_s075','turbo_img2img',dict(strength=.75,steps=4,guidance=0.))]
    elif profile not in ('core','custom','cpu_smoke'):raise ValueError('Unknown profile')
    if profile=='custom':
        if not custom:raise ValueError('custom_conditions required')
        specs=[dict(base,**r) for r in custom]
    else:
        if profile=='cpu_smoke':specifications=specifications[:2]
        specs=[dict(base,id=name,method=method,**{})|changes for name,method,changes in specifications]
    ids=set();allowed={'copy_paste','canny','recon','inpaint_fg','inpaint_bg','softedge','ip_inpaint','img2img','turbo_img2img'}
    for c in specs:
        if not c['id'].replace('_','').isalnum() or c['id'] in ids:raise ValueError('Condition IDs must be unique alphanumeric/underscore')
        ids.add(c['id'])
        if c['method'] not in allowed:raise ValueError(f"Unsupported method {c['method']}")
        if c['size']!=512:raise ValueError('This release fixes generation at 512 for a fair benchmark')
        if not 0<c['strength']<=1 or int(c['steps']*c['strength'])<1:raise ValueError('Invalid strength/steps')
        if not 1<=c['steps']<=100 or not 0<=c['control_scale']<=2 or not 0<=c['ip_scale']<=1:raise ValueError('Invalid configuration range')
        if c['method']=='turbo_img2img' and c['guidance']!=0:raise ValueError('Turbo uses guidance=0')
        if c['method']!='turbo_img2img' and c['guidance']<=1:raise ValueError('SD1.5/ReCon require CFG > 1 in this release')
    return specs


def required_keys(conditions):
    keys=set()
    for c in conditions:
        m=c['method']
        if m in ('canny','recon','softedge','img2img'):keys.add('sd')
        if m in ('canny','recon'):keys.add('canny')
        if m=='recon':keys.add('dino')
        if m in ('inpaint_fg','inpaint_bg','ip_inpaint'):keys.add('inpaint')
        if m=='ip_inpaint':keys.add('ip')
        if m=='softedge':keys.update(['softedge','hed'])
        if m=='turbo_img2img':keys.add('turbo')
    return sorted(keys)


def lock_models(out,conditions,budget,input_root='/kaggle/input'):
    path=Path(out)/'model_lock.json'
    required=required_keys(conditions)
    lock=read_json(path) if path.exists() else {}
    for key in required:
        budget.check()
        if key not in lock:
            # Imported here so a copy-paste-only run needs no hub client at all.
            from huggingface_hub import model_info
            repo=REPOS[key];revision=model_info(repo).sha
            if not revision or len(revision)!=40:raise ValueError(f'Cannot resolve immutable revision: {repo}')
            lock[key]=dict(repo=repo,revision=revision);write_json(path,lock)
        elif lock[key]['repo']!=REPOS[key]:raise ValueError('Model repository mismatch on resume')
    if any(c['method']=='recon' for c in conditions):
        hits=sorted(Path(input_root).rglob('sam_vit_b_01ec64.pth'))
        sam=hits[0] if hits else Path('/tmp/melon_benchmark_weights/sam_vit_b_01ec64.pth')
        if not sam.exists():
            sam.parent.mkdir(parents=True,exist_ok=True)
            with urllib.request.urlopen('https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',timeout=60) as response, sam.with_suffix('.part').open('wb') as f:
                while True:
                    budget.check();b=response.read(1024*1024)
                    if not b:break
                    f.write(b)
            sam.with_suffix('.part').replace(sam)
        current=dict(name=sam.name,sha256=sha(sam))
        if 'sam' in lock and lock['sam']!=current:raise ValueError('SAM checkpoint hash mismatch')
        lock['sam']=current;write_json(path,lock)
        return lock,sam
    return lock,None


class Perception:
    def __init__(self,lock,sam_path):
        import torch
        from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
        from segment_anything import sam_model_registry,SamPredictor
        kw=lock['dino'];self.processor=AutoProcessor.from_pretrained(kw['repo'],revision=kw['revision'])
        self.net=AutoModelForZeroShotObjectDetection.from_pretrained(kw['repo'],revision=kw['revision']).to('cuda').eval()
        self.sam=SamPredictor(sam_model_registry['vit_b'](checkpoint=str(sam_path)).to('cuda').eval())
    def detect(self,im):
        import torch
        with torch.inference_mode():
            inp=self.processor(images=im,text='melon.',return_tensors='pt').to('cuda');out=self.net(**inp)
            post=self.processor.post_process_grounded_object_detection
            # Transformers changed threshold spelling across releases; inspect the actual API.
            key='box_threshold' if 'box_threshold' in inspect.signature(post).parameters else 'threshold'
            r=post(out,inp['input_ids'],text_threshold=.25,target_sizes=[im.size[::-1]],**{key:.25})[0]
        return r['boxes'].cpu().tolist(),r['scores'].cpu().tolist()
    def segment(self,im,boxes):
        import torch
        if not boxes:return []
        with torch.inference_mode():
            self.sam.set_image(np.asarray(im))
            bb=self.sam.transform.apply_boxes_torch(torch.tensor(boxes,device='cuda'),im.size[::-1])
            masks,_,_=self.sam.predict_torch(point_coords=None,point_labels=None,boxes=bb,multimask_output=False)
        return [m.squeeze().cpu().numpy().astype(np.uint8) for m in masks]


def copy_paste(source,mask,c,seed):
    import cv2
    rng=np.random.default_rng(seed);a=np.asarray(source);m=np.asarray(mask);h,w=m.shape
    # Remove the original fruit before placing the transformed same fruit; no hidden donor data.
    bg=cv2.inpaint(a,cv2.dilate(m,np.ones((9,9),np.uint8)),5,cv2.INPAINT_TELEA)
    x,y,bw,bh=mask_box(mask);cx=x+bw/2;cy=y+bh/2
    mat=cv2.getRotationMatrix2D((cx,cy),rng.uniform(-c['rotation'],c['rotation']),rng.uniform(1-c['scale_jitter'],1+c['scale_jitter']))
    mat[0,2]+=rng.uniform(-c['translate'],c['translate'])*bw;mat[1,2]+=rng.uniform(-c['translate'],c['translate'])*bh
    moved=cv2.warpAffine(a,mat,(w,h),flags=cv2.INTER_LINEAR)
    moved_mask=cv2.warpAffine(m,mat,(w,h),flags=cv2.INTER_NEAREST)
    # Feather only boundary, mask remains exact annotation proposal.
    alpha=cv2.GaussianBlur(moved_mask.astype(np.float32)/255,(3,3),.6)[:,:,None]
    result=np.clip(moved*alpha+bg*(1-alpha),0,255).astype(np.uint8)
    return Image.fromarray(result),Image.fromarray(moved_mask),{'affine':mat.tolist(),'mask_source':'transformed_reviewed_mask'}


class Adapter:
    def __init__(self,condition,lock,sam_path,budget,out):
        self.c=condition;self.budget=budget;self.out=Path(out);self.model=None;self.pipe=None;self.hed=None
        m=condition['method']
        # The copy-paste control is the cheap CPU baseline; it must not require torch.
        if m=='copy_paste':return
        import torch
        if not torch.cuda.is_available():raise RuntimeError('GPU required for diffusion; no silent CPU fallback')
        from diffusers import StableDiffusionImg2ImgPipeline,StableDiffusionInpaintPipeline,StableDiffusionControlNetImg2ImgPipeline,ControlNetModel,DDIMScheduler,AutoPipelineForImage2Image
        def kw(k):return dict(pretrained_model_name_or_path=lock[k]['repo'],revision=lock[k]['revision'],torch_dtype=torch.float16)
        if m in ('canny','recon','softedge'):
            control=ControlNetModel.from_pretrained(**kw('softedge' if m=='softedge' else 'canny'))
            self.pipe=StableDiffusionControlNetImg2ImgPipeline.from_pretrained(**kw('sd'),controlnet=control,safety_checker=None,requires_safety_checker=False)
            if m in ('canny','recon'):
                from recon_attention import RegionAlignedAttnProcessor
                self.processor=RegionAlignedAttnProcessor('per_region')
                self.pipe.unet.set_attn_processor(self.processor);self.pipe.controlnet.set_attn_processor(self.processor)
            else:
                from controlnet_aux import HEDdetector
                from huggingface_hub import hf_hub_download
                local=Path(hf_hub_download(lock['hed']['repo'],'ControlNetHED.pth',revision=lock['hed']['revision'])).parent
                self.hed=HEDdetector.from_pretrained(str(local))
        elif m in ('inpaint_fg','inpaint_bg','ip_inpaint'):
            self.pipe=StableDiffusionInpaintPipeline.from_pretrained(**kw('inpaint'),safety_checker=None,requires_safety_checker=False)
            if m=='ip_inpaint':
                self.pipe.load_ip_adapter(lock['ip']['repo'],revision=lock['ip']['revision'],subfolder='models',weight_name='ip-adapter_sd15.bin')
                self.pipe.set_ip_adapter_scale(condition['ip_scale'])
        elif m=='img2img':self.pipe=StableDiffusionImg2ImgPipeline.from_pretrained(**kw('sd'),safety_checker=None,requires_safety_checker=False)
        elif m=='turbo_img2img':self.pipe=AutoPipelineForImage2Image.from_pretrained(**kw('turbo'))
        if m!='turbo_img2img':self.pipe.scheduler=DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to('cuda');self.pipe.enable_vae_slicing();self.pipe.set_progress_bar_config(disable=True)
        if m=='recon':self.model=Perception(lock,sam_path)
        self.budget.check()

    def generate(self,source,mask,pid,cls,slot,seed):
        c=self.c;m=c['method'];self.budget.check()
        if m=='copy_paste':return copy_paste(source,mask,c,seed)
        import torch,cv2
        from recon_engine import _sample,match_detections,dilate_exclusive
        common=dict(image=source,prompt=prompt(cls,slot)+', '+CONTEXT,negative_prompt=NEGATIVE,
                    num_inference_steps=c['steps'],guidance_scale=c['guidance'],strength=c['strength'],
                    generator=torch.Generator('cpu').manual_seed(seed))
        def callback(pipe,i,t,kw):self.budget.check();return kw
        trace={}
        with torch.no_grad():
            if m in ('canny','recon'):
                edge=cv2.Canny(np.asarray(source),100,200);control=Image.fromarray(np.repeat(edge[:,:,None],3,2))
                box=mask_box(mask);xyxy=[box[0],box[1],box[0]+box[2],box[1]+box[3]]
                region=(np.asarray(mask)>0).astype(np.uint8)
                if m=='recon':
                    boxes,_=self.model.detect(source);_,fp,missed=match_detections(boxes,[xyxy],.5)
                    for pm in dilate_exclusive(self.model.segment(source,fp)):
                        edge[(pm>0)&(region==0)]=0
                    control=Image.fromarray(np.repeat(edge[:,:,None],3,2))
                    trace['fas_false_regions']=len(fp);trace['source_missed']=len(missed)
                dest=self.out/'perception'/'gt_masks'/f'{pid}.png';dest.parent.mkdir(parents=True,exist_ok=True);mask.save(dest)
                cfg=dict(modules={'raca':m=='recon','rgr':m=='recon'},num_inference_steps=c['steps'],strength=c['strength'],
                         guidance_scale=c['guidance'],control_scale=c['control_scale'],perception_steps=[.75,.5,.25],match_iou=.5,
                         run=str(self.out),deadline_check=self.budget.check)
                pair=dict(id=pid,prompt=prompt(cls,slot),context_prompt=CONTEXT,negative_prompt=NEGATIVE,gt_box=xyxy)
                result,steps=_sample(self.pipe,self.model,cfg,self.processor,source,control,torch.from_numpy(region).float(),pair,self.pipe.device,seed)
                trace['rectification']=steps;trace['raca_slots']=self.processor.routed_slots
            else:
                extra=dict(callback_on_step_end=callback)
                if m in ('inpaint_fg','ip_inpaint'):extra['mask_image']=mask
                if m=='inpaint_bg':extra['mask_image']=Image.fromarray(255-np.asarray(mask))
                if m=='ip_inpaint':extra['ip_adapter_image']=source
                if m=='softedge':extra.update(control_image=self.hed(source),controlnet_conditioning_scale=c['control_scale'])
                if m=='turbo_img2img':common.pop('negative_prompt')
                result=self.pipe(**common,**extra).images[0]
                # Exact protection of the unedited pixels, not approximate latent preservation.
                if m in ('inpaint_fg','ip_inpaint'):result=Image.composite(result,source,mask)
                if m=='inpaint_bg':result=Image.composite(source,result,mask)
        trace['mask_source']='parent_mask_PROPOSAL_REQUIRES_REVIEW'
        return result,mask,trace

    def close(self):
        self.pipe=None;self.model=None;self.hed=None;gc.collect()
        if self.c['method']=='copy_paste':return
        import torch
        if torch.cuda.is_available():torch.cuda.empty_cache()
