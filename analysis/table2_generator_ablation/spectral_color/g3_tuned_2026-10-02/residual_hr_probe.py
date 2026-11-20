"""CPU probe: does keeping the photograph's own high-frequency detail close the KID gap? (LPTN-style idea)
plain:    up512(G(x256))                       ~ current export
residual: x512 + up512(G(x256) - x256)         generator changes only what it can represent at 256 px
Ripe->unripe uses G_B (EMA). Fruit view = mask from the FID fruit view of the source (grey bg)."""
import sys, numpy as np, torch, torch_fidelity
from pathlib import Path
from PIL import Image
sys.path.insert(0, sys.argv[2])
from melon_cyclegan.generate import load_generators, _to_tensor
W=Path(sys.argv[1]); ck=Path(sys.argv[3]); name=sys.argv[4]; tag=sys.argv[5]
gens,_=load_generators(ck,name,weights='ema'); G=gens['G_B']
torch.set_num_threads(4)
for mode in ('plain','residual'):
    (W/'hr'/f'{tag}_{mode}').mkdir(parents=True,exist_ok=True)
for f in sorted((W/'whole'/'real_ripe_source').glob('*.png')):
    src=Image.open(f).convert('RGB'); x512=np.asarray(src,dtype=np.float32)
    small=src.resize((256,256),Image.Resampling.LANCZOS)
    with torch.no_grad(): out=G(_to_tensor(small,'cpu'))
    out=((out.squeeze(0).clamp(-1,1)+1)*127.5).numpy().transpose(1,2,0)
    up=lambda arr: np.stack([np.asarray(Image.fromarray(arr[...,c].astype(np.float32),mode='F').resize((512,512),Image.Resampling.BICUBIC)) for c in range(3)],-1)
    plain=up(out); resid=x512+up(out-np.asarray(small,dtype=np.float32))
    fruit=np.asarray(Image.open(W/'fruit'/'real_ripe_source'/f.name).convert('RGB')); m=~np.all(fruit==128,axis=2)
    for mode,arr in (('plain',plain),('residual',resid)):
        img=np.clip(arr,0,255).round().astype(np.uint8); img[~m]=128
        Image.fromarray(img).save(W/'hr'/f'{tag}_{mode}'/f.name)
for mode in ('plain','residual'):
    r=torch_fidelity.calculate_metrics(input1=str(W/'hr'/f'{tag}_{mode}'),input2=str(W/'fruit'/'real_unripe'),fid=True,kid=True,kid_subset_size=80,kid_subsets=100,cuda=False,verbose=False,cache=False,rng_seed=2026)
    print(tag,mode,'FID',round(r['frechet_inception_distance'],2),'KID',round(r['kernel_inception_distance_mean'],4),flush=True)
