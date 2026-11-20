"""Does between-image colour diversity drive fruit-only KID?  Two non-learned colour transfers of the ripe sources.
global: one Lab affine map (ripe->unripe pooled pixel stats) applied to every fruit -> keeps each fruit's offset.
collapse: per-image Reinhard to the pooled unripe stats -> every fruit gets the same mean/sd (no between-image spread).
Only the fruit pixels change; grey background is kept. Reference = real_unripe fruit view (81 train crops)."""
import sys, numpy as np, torch_fidelity
from pathlib import Path
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
W=Path(sys.argv[1])/'fruit'; OUT=Path(sys.argv[1])/'probe'
def load(s):
    for f in sorted((W/s).glob('*.png')):
        a=np.asarray(Image.open(f).convert('RGB')); m=~np.all(a==128,axis=2); yield f.name,a,m
def pooled(s):
    px=np.concatenate([rgb2lab(a/255.)[m] for _,a,m in load(s)]); return px.mean(0),px.std(0)
mu_r,sd_r=pooled('real_ripe_source'); mu_u,sd_u=pooled('real_unripe')
for mode in ('global','collapse'):
    d=OUT/mode; d.mkdir(parents=True,exist_ok=True)
    for name,a,m in load('real_ripe_source'):
        lab=rgb2lab(a/255.); px=lab[m]
        if mode=='global': px=(px-mu_r)/sd_r*sd_u+mu_u
        else: px=(px-px.mean(0))/px.std(0)*sd_u+mu_u
        lab[m]=px; rgb=(np.clip(lab2rgb(lab),0,1)*255).round().astype(np.uint8); rgb[~m]=128
        Image.fromarray(rgb).save(d/name)
    r=torch_fidelity.calculate_metrics(input1=str(d),input2=str(W/'real_unripe'),fid=True,kid=True,kid_subset_size=80,kid_subsets=100,cuda=False,verbose=False,cache=False,rng_seed=2026)
    print(mode,'FID',round(r['frechet_inception_distance'],2),'KID',round(r['kernel_inception_distance_mean'],4),flush=True)
