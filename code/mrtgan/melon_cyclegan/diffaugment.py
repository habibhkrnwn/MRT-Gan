"""Differentiable augmentation of discriminator inputs (module M4 of rung G0X).

Zhao et al., "Differentiable Augmentation for Data-Efficient GAN Training",
NeurIPS 2020. The same family of transforms is applied to real and fake images
right before the discriminator, in the discriminator step and in the generator
step, so the discriminator cannot memorise the 85 unripe crops and gradients
still reach the generator through the transform.

Only the geometric policies are provided. The `color` policy of the paper
jitters brightness, saturation and contrast, and on this task colour is the
maturity label, so it is refused rather than silently offered.

    translation  shift by up to `TRANSLATION_RATIO` of the side, zero padded
    cutout       zero one square of `CUTOUT_RATIO` of the side

Every call draws its own random parameters per sample.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

TRANSLATION_RATIO = 0.125
CUTOUT_RATIO = 0.5
POLICIES = ("translation", "cutout")
REFUSED = ("color",)


def parse_policy(policy):
    """'translation,cutout' -> ('translation', 'cutout'); empty -> ()."""
    names = tuple(p.strip() for p in (policy or "").split(",") if p.strip())
    for name in names:
        if name in REFUSED:
            raise ValueError(f"DiffAugment policy '{name}' changes colour, which is the maturity label")
        if name not in POLICIES:
            raise ValueError(f"unknown DiffAugment policy '{name}'")
    return names


def translation(x, ratio=TRANSLATION_RATIO):
    """Shift each sample by an integer offset; vacated pixels become zero."""
    n, _, h, w = x.shape
    sh, sw = int(h * ratio + 0.5), int(w * ratio + 0.5)
    dy = torch.randint(-sh, sh + 1, (n, 1, 1), device=x.device)
    dx = torch.randint(-sw, sw + 1, (n, 1, 1), device=x.device)
    grid_b, grid_y, grid_x = torch.meshgrid(
        torch.arange(n, device=x.device), torch.arange(h, device=x.device),
        torch.arange(w, device=x.device), indexing="ij")
    grid_y = torch.clamp(grid_y + dy + 1, 0, h + 1)
    grid_x = torch.clamp(grid_x + dx + 1, 0, w + 1)
    padded = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    return padded.permute(0, 2, 3, 1).contiguous()[grid_b, grid_y, grid_x].permute(0, 3, 1, 2)


def cutout(x, ratio=CUTOUT_RATIO):
    """Zero one square per sample; the square may overhang the border."""
    n, _, h, w = x.shape
    ch, cw = int(h * ratio + 0.5), int(w * ratio + 0.5)
    oy = torch.randint(0, h + (1 - ch % 2), (n, 1, 1), device=x.device)
    ox = torch.randint(0, w + (1 - cw % 2), (n, 1, 1), device=x.device)
    grid_b, grid_y, grid_x = torch.meshgrid(
        torch.arange(n, device=x.device), torch.arange(ch, device=x.device),
        torch.arange(cw, device=x.device), indexing="ij")
    grid_y = torch.clamp(grid_y + oy - ch // 2, 0, h - 1)
    grid_x = torch.clamp(grid_x + ox - cw // 2, 0, w - 1)
    keep = torch.ones(n, h, w, dtype=x.dtype, device=x.device)
    keep[grid_b, grid_y, grid_x] = 0
    return x * keep.unsqueeze(1)


FUNCTIONS = dict(translation=translation, cutout=cutout)


def diff_augment(x, policy):
    """Apply the named policies in order; an empty policy returns `x` unchanged."""
    for name in parse_policy(policy) if isinstance(policy, str) else policy:
        x = FUNCTIONS[name](x)
    return x
