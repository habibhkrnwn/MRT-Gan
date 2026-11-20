"""Fixed TRAIN-only texture inspection. Diagnostics are not quality scores or acceptance."""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melon_cyclegan.dataset_impl import MelonRoiData, load_pair
from melon_cyclegan.generate import load_generators, _to_image
from melon_cyclegan.model_impl import dilate, feather, composite
from melon_cyclegan.texture import patch_indices, extract_patches


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--crops', required=True)
    p.add_argument('--checkpoints-dir', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--epoch', default='latest')
    p.add_argument('--weights', default='raw', choices=['raw','ema'])
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--count', type=int, default=12)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--patch-size', type=int, default=64)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    data = MelonRoiData(args.crops, no_flip=True)
    nets, meta = load_generators(args.checkpoints_dir, args.name, args.epoch, device=args.device, weights=args.weights)
    settings = meta['training_config']
    size, stride, coverage = args.patch_size, 16, .95  # identical diagnostic patches for every arm
    rng = random.Random(args.seed)
    sources = rng.sample(data.B_paths, min(args.count,len(data.B_paths)))
    refs = rng.choices(data.A_paths, k=len(sources))
    rows, panels = [], []
    for source, ref in zip(sources, refs):
        x, mask = load_pair(source, data.mask_for(source, 'B'))
        target, tm = load_pair(ref, data.mask_for(ref, 'A'))
        x, mask = x[None].to(args.device), mask[None].to(args.device)
        with torch.no_grad():
            raw = nets['G_B'](x)
            alpha = feather(dilate(mask, settings.get('mask_dilate',0)), settings.get('mask_feather',0))
            fake = composite(raw, x, alpha) if settings.get('composite_mask') else raw
        ix = patch_indices(mask, size,stride,1,coverage,random=False)
        it = patch_indices(tm[None],size,stride,1,coverage,random=False)
        # Full-frame exporter uses a hard mask; show its crop analogue too.
        hard = composite(raw, x, mask)
        xp=extract_patches(x,ix,size,stride,False)
        fp=extract_patches(hard,ix,size,stride,False)
        tp=extract_patches(target[None],it,size,stride,False)
        energy=lambda t: float(((t[:,:,:,1:]-t[:,:,:,:-1]).abs().mean()
                              +(t[:,:,1:,:]-t[:,:,:-1,:]).abs().mean()).detach().cpu()/2)
        rows.append(dict(source=source.stem,target_reference=ref.stem,
                         source_edge_energy=energy(xp),synthetic_edge_energy=energy(fp),
                         unpaired_real_edge_energy=energy(tp),
                         target_label='',texture_realistic='',boundary_ok='',accepted='',reviewer=''))
        row = Image.new('RGB',(256*6,290),'white')
        images=[_to_image(x),_to_image(fake),_to_image(hard),_to_image(xp),_to_image(fp),_to_image(tp)]
        labels=[source.stem+' ripe','synthetic soft mask','synthetic export mask',f'source patch {size}px',
                f'synthetic patch {size}px',ref.stem+' real unripe (unpaired)']
        draw=ImageDraw.Draw(row)
        for j,(im,label) in enumerate(zip(images,labels)):
            row.paste(im.resize((256,256),Image.Resampling.NEAREST),(j*256,30))
            draw.text((j*256+3,5),label,fill='black')
        row.save(out/f'{source.stem}.png')
        panels.append(row)
    for page,start in enumerate(range(0,len(panels),4)):
        batch=panels[start:start+4]
        sheet=Image.new('RGB',(1536,290*len(batch)),'white')
        for j,row in enumerate(batch): sheet.paste(row,(0,290*j))
        sheet.save(out/f'panel_{page:02d}.png')
    if rows:
        with (out/'review.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/'metadata.json').write_text(json.dumps(dict(generator=meta,seed=args.seed,split='train',
        diagnostic='Edge energy may rise with noise; not an acceptance criterion. References are unpaired.',
        test_used=False, patch_size=size, patch_stride=stride, patch_coverage=coverage),indent=2)+'\n')
    print(out)


if __name__ == '__main__':
    main()
