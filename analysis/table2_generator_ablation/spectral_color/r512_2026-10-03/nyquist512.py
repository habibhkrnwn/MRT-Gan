"""Nyquist-row/column power fraction of 32x32 interior patches at the 512 review view, luma and R-Y.
Same definition as the study diagnostics (FFT without window, highest row+column / total, mean removed)."""
import sys, numpy as np
from pathlib import Path
from PIL import Image
W=Path(sys.argv[1])/'fruit'
sets=['real_unripe','V0','G3TI','G3TI_noT','G3TI_R512','G3TI_noT_R512','CT']
def nyq(p):
    p=p-p.mean(); P=np.abs(np.fft.fft2(p))**2; tot=P.sum()
    return (P[16,:].sum()+P[:,16].sum()-P[16,16])/tot if tot>0 else 0
for s in sets:
    ly,lr=[],[]
    for f in sorted((W/s).glob('*.png')):
        a=np.asarray(Image.open(f).convert('RGB')).astype(float); m=~np.all(a==128,axis=2)
        Y=a@[0.299,0.587,0.114]; RY=a[...,0]-Y; n=0
        for y in range(0,512-32,16):
            for x in range(0,512-32,16):
                if m[y:y+32,x:x+32].all():
                    ly.append(nyq(Y[y:y+32,x:x+32])); lr.append(nyq(RY[y:y+32,x:x+32])); n+=1
                    if n>=8: break
            if n>=8: break
    print(f"{s:15s} Nyquist luma {np.mean(ly):.4f}  R-Y {np.mean(lr):.4f}")
