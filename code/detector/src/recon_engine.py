# Adapted from the local v5.1 ReCon implementation; see README for differences.
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
SIZE=512
LATENT=64
def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.

def match_detections(boxes, gt_boxes, threshold=.5):
    """Class-agnostic greedy IoU matching. Returns (matched_gt, false_positive_boxes, missed_gt)."""
    taken = [False] * len(boxes)
    matched, missed = [], []
    for g, gt in enumerate(gt_boxes):
        best, best_iou = -1, 0.
        for i, box in enumerate(boxes):
            if taken[i]: continue
            score = iou(box, gt)
            if score > best_iou: best, best_iou = i, score
        if best >= 0 and best_iou >= threshold:
            taken[best] = True; matched.append((g, best, best_iou))
        else:
            missed.append(g)
    return matched, [boxes[i] for i, t in enumerate(taken) if not t], missed

def dilate_exclusive(masks, kernel=7):
    """Dilate each binary mask without letting it grow into another mask's pixels."""
    import cv2
    masks = [np.asarray(m, dtype=np.uint8) for m in masks]
    if not masks: return []
    occupied = np.clip(np.sum(masks, axis=0), 0, 1)
    k = np.ones((kernel, kernel), np.uint8)
    out = []
    for m in masks:
        others = np.clip(occupied - m, 0, 1)
        grown = cv2.dilate(m, k)
        keep = ((grown == 1) & (others == 0)).astype(np.uint8)
        occupied = np.clip(occupied + keep, 0, 1)
        out.append(keep)
    return out

@torch.no_grad()
def encode_prompt(pipe, texts, device):
    tokens = pipe.tokenizer(texts, padding='max_length', max_length=pipe.tokenizer.model_max_length,
                            truncation=True, return_tensors='pt').input_ids.to(device)
    return pipe.text_encoder(tokens)[0]

@torch.no_grad()
def _sample(pipe, model, cfg, processor, source, control, region, pair, device, seed):
    """One img2img pass with region-guided rectification at the configured checkpoints."""
    generator = torch.Generator(device='cpu').manual_seed(seed)
    dtype = pipe.unet.dtype
    n_region = 1

    # Slot 0 is the fruit's class prompt, slot 1 the scene context; the negative branch repeats the
    # same negative text in both slots so the classifier-free pair keeps a matching shape.
    positive = encode_prompt(pipe, [pair['prompt'], pair['prompt'] + ', ' + pair['context_prompt']], device)
    negative = encode_prompt(pipe, [pair['negative_prompt'], pair['negative_prompt']], device)
    stacked = torch.stack([negative, positive]).to(dtype)          # (2, 2, tokens, dim)
    context = stacked[:, -1]                                       # (2, tokens, dim)
    # diffusers only ever sees an ordinary tensor; the region routing rides on the processor.
    if cfg['modules']['raca']: processor.set_regions([region.to(device)], stacked)
    else: processor.clear_regions()
    embeds = context

    latents = pipe.vae.encode(
        (torch.from_numpy(np.asarray(source, np.float32) / 127.5 - 1).permute(2, 0, 1)[None]).to(device, dtype)
    ).latent_dist.sample(generator) * pipe.vae.config.scaling_factor
    pipe.scheduler.set_timesteps(cfg['num_inference_steps'], device=device)
    timesteps = pipe.scheduler.timesteps
    start = int(len(timesteps) * (1 - cfg['strength']))
    timesteps = timesteps[start:]
    noise = torch.randn(latents.shape, generator=generator, device='cpu').to(device, dtype)
    noised = torch.stack([pipe.scheduler.add_noise(latents, noise, t.reshape(1))[0] for t in timesteps])
    current = noised[:1].clone()
    # Checkpoints follow the steps actually scheduled, not num_inference_steps: strength < 1 trims the
    # schedule, and indices derived from the untrimmed count would silently never fire.
    total = len(timesteps)
    checkpoints = sorted({min(total - 1, max(0, total - int(total * p) - 1)) for p in cfg['perception_steps']})

    hint = torch.from_numpy(np.asarray(control, np.float32) / 255).permute(2, 0, 1)[None].to(device, dtype)
    hint = torch.cat([hint] * 2)
    trace = []
    for i, t in enumerate(timesteps):
        if cfg.get('deadline_check'): cfg['deadline_check']()
        model_input = torch.cat([current] * 2)
        model_input = pipe.scheduler.scale_model_input(model_input, t)
        # The ControlNet branch conditions on the scene context only; region routing is a UNet concern.
        processor.clear_regions()
        down, mid = pipe.controlnet(model_input, t, encoder_hidden_states=embeds,
                                    controlnet_cond=hint, conditioning_scale=cfg['control_scale'], return_dict=False)
        if cfg['modules']['raca']: processor.set_regions([region.to(device)], stacked)
        noise_pred = pipe.unet(model_input, t, encoder_hidden_states=embeds,
                               down_block_additional_residuals=down,
                               mid_block_additional_residual=mid, return_dict=False)[0]
        uncond, cond = noise_pred.chunk(2)
        noise_pred = uncond + cfg['guidance_scale'] * (cond - uncond)
        step = pipe.scheduler.step(noise_pred, t, current, return_dict=True)
        current = step.prev_sample

        if cfg['modules']['rgr'] and i in checkpoints and i + 1 < len(noised):
            decoded = pipe.vae.decode(step.pred_original_sample / pipe.vae.config.scaling_factor, return_dict=False)[0]
            preview = Image.fromarray(
                ((decoded[0].permute(1, 2, 0).float().clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy())
            boxes, _ = model.detect(preview)
            matched, false_positive, missed = match_detections(boxes, [pair['gt_box']], cfg['match_iou'])
            pieces = dilate_exclusive(model.segment(preview, false_positive)) if false_positive else []
            if missed:
                pieces.append(np.asarray(
                    Image.open(Path(cfg['run']) / 'perception' / 'gt_masks' / f"{pair['id']}.png").convert('L')) > 0)
            edit = np.zeros((SIZE, SIZE), np.uint8)
            for piece in pieces: edit |= np.asarray(piece, np.uint8)
            fraction = float(edit.mean())
            if edit.any():
                small = F.interpolate(torch.from_numpy(edit).float()[None, None], size=(LATENT, LATENT),
                                      mode='nearest').to(device, dtype)
                current = noised[i + 1][None] * small + current * (1 - small)
            trace.append({'step': i, 'matched': len(matched), 'false_positive': len(false_positive),
                          'missed': len(missed), 'rectified_fraction': fraction})

    decoded = pipe.vae.decode(current / pipe.vae.config.scaling_factor, return_dict=False)[0]
    image = Image.fromarray(((decoded[0].permute(1, 2, 0).float().clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy())
    return image, trace
