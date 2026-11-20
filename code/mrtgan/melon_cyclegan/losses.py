"""Extra loss terms of the enhanced model (G1..G5 in the plan).

All terms take tensors in the CycleGAN convention, NxCxHxW in [-1, 1]; the
colour term converts to [0, 1] internally because phi is defined on [0, 1].

    L_cyc^SSIM  1 - SSIM(reconstruction, input)             structure, G1
    L_bg        || (1 - M) * (G(x) - x) ||_1                background, G3
    L_edge      || M * (grad G(x) - grad x) ||_1            contour, G5
    L_color     || phi(G(x)) - phi_target ||_1              colour, G4
    L_margin    relu(d_target - d_source + m), z-scaled     colour, G0X module M2
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .color import phi_torch

SSIM_WINDOW = 11
SSIM_SIGMA = 1.5
C1 = 0.01 ** 2
C2 = 0.03 ** 2

SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
SOBEL_Y = SOBEL_X.t().contiguous()


def to_unit(x):
    """[-1, 1] -> [0, 1]."""
    return (x + 1.0) * 0.5


def _gaussian_window(channels, device, dtype):
    coords = torch.arange(SSIM_WINDOW, device=device, dtype=dtype) - (SSIM_WINDOW - 1) / 2.0
    kernel = torch.exp(-(coords ** 2) / (2 * SSIM_SIGMA ** 2))
    kernel = kernel / kernel.sum()
    window = kernel.unsqueeze(1) @ kernel.unsqueeze(0)
    return window.expand(channels, 1, SSIM_WINDOW, SSIM_WINDOW).contiguous()


def ssim(a, b):
    """Mean SSIM between two NxCxHxW tensors in [-1, 1]."""
    a, b = to_unit(a), to_unit(b)
    channels = a.shape[1]
    window = _gaussian_window(channels, a.device, a.dtype)
    pad = SSIM_WINDOW // 2
    mu_a = F.conv2d(a, window, padding=pad, groups=channels)
    mu_b = F.conv2d(b, window, padding=pad, groups=channels)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sigma_a = F.conv2d(a * a, window, padding=pad, groups=channels) - mu_a2
    sigma_b = F.conv2d(b * b, window, padding=pad, groups=channels) - mu_b2
    sigma_ab = F.conv2d(a * b, window, padding=pad, groups=channels) - mu_ab
    numerator = (2 * mu_ab + C1) * (2 * sigma_ab + C2)
    denominator = (mu_a2 + mu_b2 + C1) * (sigma_a + sigma_b + C2)
    return (numerator / denominator).mean()


def ssim_cycle_loss(reconstruction, target):
    """1 - SSIM, the structural half of the cycle consistency."""
    return 1.0 - ssim(reconstruction, target)


def background_loss(generated, source, mask):
    """L1 change outside the mask; keeps the greenhouse context still."""
    outside = 1.0 - mask
    weight = outside.sum().clamp_min(1.0)
    return ((generated - source).abs() * outside).sum() / weight


def _gradients(x):
    channels = x.shape[1]
    kx = SOBEL_X.to(x.device, x.dtype).expand(channels, 1, 3, 3).contiguous()
    ky = SOBEL_Y.to(x.device, x.dtype).expand(channels, 1, 3, 3).contiguous()
    gx = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), kx, groups=channels)
    gy = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), ky, groups=channels)
    return gx, gy


def edge_loss(generated, source, mask):
    """L1 gradient difference inside the mask; keeps the fruit contour."""
    gx_g, gy_g = _gradients(generated)
    gx_s, gy_s = _gradients(source)
    weight = mask.sum().clamp_min(1.0)
    difference = (gx_g - gx_s).abs() + (gy_g - gy_s).abs()
    return (difference * mask).sum() / weight


def color_loss(generated, mask, target):
    """L1 distance from phi(G(x)) to the target-domain centroid.

    `target` is a 4-vector; the output is never compared against a single random
    image of the target domain, only against the domain statistic.
    """
    current = phi_torch(to_unit(generated).clamp(0.0, 1.0), mask)
    target = target.to(current.device, current.dtype).view(1, -1)
    return (current - target).abs().sum(dim=1).mean()


# Entries of phi used by the margin term: mean Cr and mean a*. The two spreads are
# left out on purpose; they describe texture, and pulling them to a class value
# would flatten the skin pattern the translation is supposed to keep.
MARGIN_DIMS = (0, 2)


def margin_distances(phi, target, source, scale):
    """Euclidean distances of phi to the target and source centroids, in z units.

    `phi` is Nx4; `target`, `source` and `scale` are 4-vectors. Only
    `MARGIN_DIMS` enter, each divided by its pooled class spread.
    """
    dims = list(MARGIN_DIMS)
    device, dtype = phi.device, phi.dtype
    scale = scale.to(device, dtype)[dims].view(1, -1)
    z = phi[:, dims] / scale
    z_target = target.to(device, dtype)[dims].view(1, -1) / scale
    z_source = source.to(device, dtype)[dims].view(1, -1) / scale
    return (z - z_target).norm(dim=1), (z - z_source).norm(dim=1)


def color_margin_loss(generated, mask, target, source, scale, margin):
    """Hinge that asks G(x) to sit closer to the target class than to the source.

        relu( d(z(phi(G(x))), z_target) - d(z(phi(G(x))), z_source) + margin )

    Zero once the output is on the target side by at least `margin` z units, so
    it does not collapse every output onto one centroid.
    """
    current = phi_torch(to_unit(generated).clamp(0.0, 1.0), mask)
    d_target, d_source = margin_distances(current, target, source, scale)
    return F.relu(d_target - d_source + margin).mean()
