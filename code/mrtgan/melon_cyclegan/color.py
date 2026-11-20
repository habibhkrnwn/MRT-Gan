"""Colour descriptor phi(x) used by the colour-aware loss and by the quality control.

The plan asks for statistics of Cr and CIELAB a* taken inside the fruit mask:

    phi(x) = [ mean(Cr), std(Cr), mean(a*), std(a*) ]

Both statistics are computed on pixels where the mask is positive. a* is divided
by 128 so that the four entries live on a comparable scale and a single weight
`lambda_color` is meaningful.

Two implementations are provided and are required to agree (see tests):

* `phi_numpy`  -- used when building the target centroids and in quality control.
* `phi_torch`  -- differentiable, used inside the training loss.

Both take RGB in [0, 1].
"""
from __future__ import annotations

import numpy as np

# ITU-R BT.601 chroma-red, expressed for RGB in [0, 1]; result is in [0, 1].
CR_COEFF = (0.5, -0.418688, -0.081312)
CR_OFFSET = 0.5

# sRGB (D65) -> XYZ, and the D65 white point.
RGB2XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)
WHITE_D65 = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)

A_SCALE = 128.0     # a* is divided by this before entering phi
EPS = 1e-8


def _srgb_to_linear_numpy(rgb):
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _f_lab_numpy(t):
    delta = 6.0 / 29.0
    return np.where(t > delta ** 3, np.cbrt(np.maximum(t, EPS)), t / (3 * delta ** 2) + 4.0 / 29.0)


def cr_numpy(rgb):
    """Chroma-red channel of an HxWx3 array in [0, 1]."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return CR_COEFF[0] * r + CR_COEFF[1] * g + CR_COEFF[2] * b + CR_OFFSET


def astar_numpy(rgb):
    """CIELAB a* of an HxWx3 array in [0, 1], divided by A_SCALE."""
    linear = _srgb_to_linear_numpy(np.asarray(rgb, dtype=np.float64))
    xyz = linear @ RGB2XYZ.T
    scaled = xyz / WHITE_D65
    fx = _f_lab_numpy(scaled[..., 0])
    fy = _f_lab_numpy(scaled[..., 1])
    return (500.0 * (fx - fy)) / A_SCALE


def phi_numpy(rgb, mask=None):
    """phi(x) for one HxWx3 image in [0, 1]. `mask` is a boolean HxW array."""
    rgb = np.asarray(rgb, dtype=np.float64)
    cr = cr_numpy(rgb)
    a = astar_numpy(rgb)
    if mask is None:
        selected_cr, selected_a = cr.reshape(-1), a.reshape(-1)
    else:
        m = np.asarray(mask).astype(bool)
        if m.shape != cr.shape:
            raise ValueError("mask geometry does not match the image")
        if not m.any():
            raise ValueError("empty mask; phi is undefined")
        selected_cr, selected_a = cr[m], a[m]
    return np.array(
        [selected_cr.mean(), selected_cr.std(), selected_a.mean(), selected_a.std()],
        dtype=np.float64,
    )


# --------------------------------------------------------------------------- torch


def _torch():
    import torch

    return torch


def _srgb_to_linear_torch(rgb):
    torch = _torch()
    rgb = rgb.clamp(0.0, 1.0)
    return torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _f_lab_torch(t):
    torch = _torch()
    delta = 6.0 / 29.0
    return torch.where(t > delta ** 3, t.clamp_min(EPS) ** (1.0 / 3.0), t / (3 * delta ** 2) + 4.0 / 29.0)


def cr_torch(rgb):
    """Chroma-red of an NxCxHxW tensor in [0, 1]; returns Nx1xHxW."""
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    return CR_COEFF[0] * r + CR_COEFF[1] * g + CR_COEFF[2] * b + CR_OFFSET


def astar_torch(rgb):
    """CIELAB a*/A_SCALE of an NxCxHxW tensor in [0, 1]; returns Nx1xHxW."""
    torch = _torch()
    linear = _srgb_to_linear_torch(rgb)
    matrix = torch.tensor(RGB2XYZ, dtype=linear.dtype, device=linear.device)
    white = torch.tensor(WHITE_D65, dtype=linear.dtype, device=linear.device).view(1, 3, 1, 1)
    flat = linear.permute(0, 2, 3, 1)
    xyz = (flat @ matrix.T).permute(0, 3, 1, 2)
    scaled = xyz / white
    fx = _f_lab_torch(scaled[:, 0:1])
    fy = _f_lab_torch(scaled[:, 1:2])
    return (500.0 * (fx - fy)) / A_SCALE


def phi_torch(rgb, mask=None):
    """phi(x) for an NxCxHxW tensor in [0, 1]; returns Nx4, differentiable.

    `mask` is Nx1xHxW with values in [0, 1]; weights are used directly, so a soft
    mask is allowed. A sample whose mask is empty falls back to the whole frame,
    which keeps the loss finite instead of producing NaN.
    """
    torch = _torch()
    cr = cr_torch(rgb)
    a = astar_torch(rgb)
    if mask is None:
        weights = torch.ones_like(cr)
    else:
        weights = mask
        if weights.shape[-2:] != cr.shape[-2:]:
            raise ValueError("mask geometry does not match the image")
        empty = weights.flatten(1).sum(1) <= EPS
        if bool(empty.any()):
            weights = torch.where(empty.view(-1, 1, 1, 1), torch.ones_like(weights), weights)
    total = weights.flatten(1).sum(1).clamp_min(EPS)

    def stats(channel):
        flat = channel.flatten(1)
        w = weights.flatten(1)
        mean = (flat * w).sum(1) / total
        var = (((flat - mean.unsqueeze(1)) ** 2) * w).sum(1) / total
        # sqrt'(0) is infinite. Even a loss using only the means can receive
        # NaN via a zero gradient through the unused std branch (0 * inf).
        return mean, var.clamp_min(1e-12).sqrt()

    cr_mean, cr_std = stats(cr)
    a_mean, a_std = stats(a)
    return torch.stack([cr_mean, cr_std, a_mean, a_std], dim=1)
