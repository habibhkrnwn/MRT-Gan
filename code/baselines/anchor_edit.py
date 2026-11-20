
import json, sys, torch
from pathlib import Path
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline as Pipe, DPMSolverMultistepScheduler as Sch
items, dst, seed, prompt, config, offset = json.load(open(sys.argv[1])), Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5], int(sys.argv[6])
STEPS, BETA, GT = 25, 0.01, 7.5
GI = 7.5 if config == 'A' else 1.5
dev, dt = 'cuda', torch.float16
ip = Pipe.from_pretrained('timbrooks/instruct-pix2pix', torch_dtype=dt, safety_checker=None).to(dev)
mb = Pipe.from_pretrained('vinesmsuic/magicbrush-jul7', torch_dtype=dt, safety_checker=None).to(dev)
sch_ip = Sch.from_config(ip.scheduler.config)
sch_mb = Sch.from_config(ip.scheduler.config)
dst.mkdir(parents=True, exist_ok=True)
def embed(text):
    ids = ip.tokenizer(text, padding='max_length', max_length=ip.tokenizer.model_max_length, truncation=True, return_tensors='pt').input_ids.to(dev)
    return ip.text_encoder(ids)[0]
with torch.no_grad():
    cond, unc = embed(prompt), embed('')
context = torch.cat([cond, unc, unc])
def branch(pipe, sch, z, t, image_latents, mask):
    x = sch.scale_model_input(torch.cat([z] * 3), t)
    x = torch.cat([x, image_latents], 1)
    e_text, e_img, e_unc = pipe.unet(x, t, encoder_hidden_states=context).sample.chunk(3)
    guided = e_unc + GT * (e_text - e_unc) + GI * (e_img - e_unc) if config == 'A' else e_unc + GT * (e_text - e_img) + GI * (e_img - e_unc)
    final = e_unc + mask * (guided - e_unc)
    return sch.step(final, t, z).prev_sample
for i, x in enumerate(items):
    im = Image.open(x['image']).convert('RGB').resize((512, 512), Image.Resampling.LANCZOS)
    mk = Image.open(x['mask']).convert('L').resize((64, 64), Image.Resampling.NEAREST)
    mask = torch.tensor(np.asarray(mk) > 127, device=dev, dtype=dt)[None, None]
    g = torch.Generator(dev).manual_seed(seed * 10000 + offset + i)
    with torch.no_grad():
        img = ip.image_processor.preprocess(im).to(dev, dt)
        lat = ip.vae.encode(img).latent_dist.mode()
        image_latents = torch.cat([lat, lat, torch.zeros_like(lat)])
        sch_ip.set_timesteps(STEPS, device=dev)
        sch_mb.set_timesteps(STEPS, device=dev)
        z0 = torch.randn((1, 4, 64, 64), generator=g, device=dev, dtype=dt) * sch_ip.init_noise_sigma
        z_ip, z_mb = z0.clone(), z0.clone()
        for t in sch_ip.timesteps:
            z_mb = branch(mb, sch_mb, z_mb, t, image_latents, mask)
            z_ip = branch(ip, sch_ip, z_ip, t, image_latents, mask)
            z_ip = (z_ip + BETA * z_mb) / (1 + BETA)
        out = ip.vae.decode(z_ip / ip.vae.config.scaling_factor).sample
    ip.image_processor.postprocess(out.float(), output_type='pil')[0].save(dst / f"{x['id']}.png")
print('peak GiB', round(torch.cuda.max_memory_allocated() / 2**30, 2), flush=True)
