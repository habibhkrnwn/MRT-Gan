
import json, sys, torch
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
lora, items, dst, seed, prompt = sys.argv[1], json.load(open(sys.argv[2])), Path(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
p = StableDiffusionInpaintPipeline.from_pretrained('stable-diffusion-v1-5/stable-diffusion-inpainting', torch_dtype=torch.float16, safety_checker=None).to('cuda')
p.load_lora_weights(lora); p.set_progress_bar_config(disable=True); dst.mkdir(parents=True, exist_ok=True)
for i, x in enumerate(items):
    im = Image.open(x['image']).convert('RGB'); mk = Image.open(x['mask']).convert('L').point(lambda v: 255 if v > 127 else 0)
    g = torch.Generator('cuda').manual_seed(seed*10000+i)
    out = p(prompt, image=im, mask_image=mk, height=512, width=512, num_inference_steps=30, guidance_scale=7.5, generator=g).images[0]
    out.save(dst/f"{x['id']}.png")
