"""Radially averaged luminance power spectrum of interior fruit patches (512px fruit view), by frequency band.
Patch 64x64 fully inside the fruit (no grey pixels), up to 6 per image, Hann window, mean-removed."""
import sys, numpy as np
from pathlib import Path
from PIL import Image
W=Path(sys.argv[1])/'fruit'
sets=['real_unripe','V0','G3TI','G3TIs12','G3TIs13','G3TI_noT','G3TI_R512','G3TI_noT_R512','CT']
P=64; win=np.outer(np.hanning(P),np.hanning(P))
fy,fx=np.meshgrid(np.fft.fftfreq(P),np.fft.fftfreq(P),indexing='ij'); rad=np.sqrt(fx**2+fy**2)
bands=[(0.02,0.06),(0.06,0.12),(0.12,0.25),(0.25,0.5)]  # cycles/pixel at 512 px: ~17-50px, 8-17px, 4-8px, 2-4px
res={}
for s in sets:
    acc=[];
    for f in sorted((W/s).glob('*.png')):
        a=np.asarray(Image.open(f).convert('RGB')).astype(float)
        m=~np.all(a==128,axis=2); Y=a@[0.299,0.587,0.114]
        H,Wd=Y.shape; n=0
        for y in range(0,H-P,48):
            for x in range(0,Wd-P,48):
                if m[y:y+P,x:x+P].all():
                    p=Y[y:y+P,x:x+P]; p=(p-p.mean())*win
                    ps=np.abs(np.fft.fft2(p))**2
                    acc.append([ps[(rad>=lo)&(rad<hi)].mean() for lo,hi in bands]); n+=1
                    if n>=6: break
            if n>=6: break
    res[s]=np.log10(np.array(acc).mean(0))
ref=res['real_unripe']
print('band (px period)      ', ['50-17','17-8','8-4','4-2'])
for s in sets: print(f"{s:12s} log10P", np.round(res[s],2), ' diff vs real', np.round(res[s]-ref,2))
