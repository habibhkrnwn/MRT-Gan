"""Per-image fruit colour statistics (CIELAB) for real and synthetic sets, from the FID fruit view (grey bg)."""
import sys, numpy as np
from pathlib import Path
from PIL import Image
from skimage.color import rgb2lab
W=Path(sys.argv[1])/'fruit'
sets=['real_unripe','real_ripe_source','V0','G3','G3T','G3I','G3TI']
out={}
for s in sets:
    rows=[]
    for f in sorted((W/s).glob('*.png')):
        a=np.asarray(Image.open(f).convert('RGB').resize((256,256),Image.Resampling.NEAREST))
        m=~np.all(a==128,axis=2)
        lab=rgb2lab(a/255.0)[m]
        rows.append([*lab.mean(0), *lab.std(0)])
    r=np.array(rows); out[s]=r
    print(f"{s:17s} n={len(r)} L {r[:,0].mean():5.1f}±{r[:,0].std():4.1f}  a {r[:,1].mean():6.1f}±{r[:,1].std():4.1f}  b {r[:,2].mean():5.1f}±{r[:,2].std():4.1f} | within-fruit sd L {r[:,3].mean():4.1f} a {r[:,4].mean():4.1f} b {r[:,5].mean():4.1f}")
ref=out['real_unripe']
for s in sets[1:]:
    r=out[s]; d=np.linalg.norm(r[:,:3].mean(0)-ref[:,:3].mean(0))
    print(f"{s:17s} centroid dist Lab {d:5.2f}   between-image spread ratio L/a/b {r[:,0].std()/ref[:,0].std():.2f} {r[:,1].std()/ref[:,1].std():.2f} {r[:,2].std()/ref[:,2].std():.2f}")
