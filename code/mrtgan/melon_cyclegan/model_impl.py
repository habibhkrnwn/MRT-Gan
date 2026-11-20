"""Enhanced CycleGAN model: baseline objective plus the four melon-specific terms.

The class keeps every part of the upstream CycleGAN objective unchanged and adds

    lambda_ssim       * ( 1 - SSIM(rec, real) )            both directions
    lambda_background * || (1 - M) * (G(x) - x) ||_1       both directions
    lambda_edge       * || M * (grad G(x) - grad x) ||_1   both directions
    lambda_color      * || phi(G(x)) - phi_target ||_1     both directions

and the G0X modules, each off by default:

    lambda_color_margin * relu(d_target - d_source + m)    colour margin, M2
    composite_mask       G(x) pasted into x through the fruit mask, M1
    diffaug              DiffAugment policy on discriminator inputs, M4

Setting every weight to zero and leaving the switches off reproduces the
baseline objective exactly, so the same class serves G0..G5 and G0X.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from . import diffaugment, losses
from .config import CLASS_NAMES, RIPE, UNRIPE


def load_centroids(path):
    """phi targets as a dict {'A': tensor, 'B': tensor}; A is unripe, B is ripe."""
    data = json.loads(Path(path).read_text())["classes"]
    by_class = {entry["class_id"]: entry["centroid"] for entry in data.values()}
    missing = {RIPE, UNRIPE} - set(by_class)
    if missing:
        raise ValueError(f"colour centroids missing for classes {sorted(missing)}")
    return {
        "A": torch.tensor(by_class[UNRIPE], dtype=torch.float32),
        "B": torch.tensor(by_class[RIPE], dtype=torch.float32),
    }


def load_color_scale(path):
    """Pooled per-dimension spread of phi over both classes, used as z scale.

    Each class spread is weighted by its crop count when the file has counts.
    """
    data = json.loads(Path(path).read_text())["classes"]
    spreads, counts = [], []
    for entry in data.values():
        if "spread" not in entry:
            raise ValueError("color_centroids.json has no 'spread'; rebuild it with tools/build_crops.py")
        spreads.append(torch.tensor(entry["spread"], dtype=torch.float64))
        counts.append(float(entry.get("count", 1)))
    total = sum(counts)
    pooled = sum(c * s ** 2 for c, s in zip(counts, spreads)) / total
    scale = pooled.sqrt().float()
    if not torch.isfinite(scale).all() or bool((scale[list(losses.MARGIN_DIMS)] <= 0).any()):
        raise ValueError('colour scale must be finite and positive on the margin dimensions')
    return scale


def dilate(mask, pixels):
    """Grow an Nx1xHxW mask by `pixels` with a square max filter; 0 is a no-op."""
    if pixels <= 0:
        return mask
    return F.max_pool2d(mask, kernel_size=2 * pixels + 1, stride=1, padding=pixels)


def feather(mask, sigma):
    """Gaussian-blurred copy of an Nx1xHxW mask; sigma <= 0 returns it unchanged."""
    if sigma <= 0:
        return mask
    radius = max(1, int(3 * sigma + 0.5))
    coords = torch.arange(-radius, radius + 1, device=mask.device, dtype=mask.dtype)
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    padded = F.pad(mask, (radius, radius, radius, radius), mode="replicate")
    blurred = F.conv2d(padded, kernel.view(1, 1, 1, -1))
    blurred = F.conv2d(blurred, kernel.view(1, 1, -1, 1))
    return blurred.clamp(0.0, 1.0)


@torch.no_grad()
def ema_update(ema_net, net, decay):
    """ema <- decay * ema + (1 - decay) * net, parameters and floating buffers alike."""
    for shadow, value in zip(ema_net.parameters(), net.parameters()):
        shadow.lerp_(value.detach(), 1.0 - decay)
    for shadow, value in zip(ema_net.buffers(), net.buffers()):
        if shadow.dtype.is_floating_point:
            shadow.lerp_(value, 1.0 - decay)
        else:
            shadow.copy_(value)


def composite(raw, source, mask):
    """Translated fruit inside the mask, the untouched source outside it."""
    return mask * raw + (1.0 - mask) * source


def add_options(parser, is_train=True):
    """Options of the enhanced model, on top of the CycleGAN ones."""
    parser.set_defaults(no_dropout=True, netG="multiscale_resnet_6blocks",
                        dataset_mode="melon", preprocess="none", load_size=256, crop_size=256)
    if is_train:
        parser.add_argument("--lambda_A", type=float, default=10.0, help="weight for cycle loss (A -> B -> A)")
        parser.add_argument("--lambda_B", type=float, default=10.0, help="weight for cycle loss (B -> A -> B)")
        parser.add_argument("--lambda_identity", type=float, default=0.5,
                            help="scales the identity mapping loss, as in upstream CycleGAN")
        parser.add_argument("--lambda_ssim", type=float, default=1.0, help="weight for the SSIM cycle loss")
        parser.add_argument("--lambda_background", type=float, default=5.0,
                            help="weight for the L1 change outside the fruit mask")
        parser.add_argument("--lambda_edge", type=float, default=0.0,
                            help="weight for the gradient match inside the fruit mask")
        parser.add_argument("--lambda_color", type=float, default=1.0,
                            help="weight for the distance from phi(G(x)) to the target class centroid")
        parser.add_argument("--color_centroids", type=str, default="",
                            help="path to color_centroids.json; required when a colour term is on")
        parser.add_argument("--lambda_color_margin", type=float, default=0.0,
                            help="weight for the z-scaled hinge closer-to-target-than-source, G0X M2")
        parser.add_argument("--color_margin", type=float, default=0.0,
                            help="margin m of the colour hinge in z units, from tools/calibrate_color_margin.py")
        parser.add_argument("--diffaug", type=str, default="",
                            help="DiffAugment policy on discriminator inputs, e.g. 'translation,cutout', G0X M4")
        parser.add_argument("--ema_decay", type=float, default=0.0,
                            help="exponential moving average of generator weights (Yazici et al., ICLR 2019); "
                                 "0 disables. Saved beside the raw weights as *_net_G_A_ema.pth")
        parser.add_argument("--amp", type=int, default=0,
                            help="1 runs the networks in float16 autocast on CUDA; colour and SSIM terms stay "
                                 "float32. Off by default so historical runs reproduce unchanged")
    # Both phases, so a test-phase forward composites the way training did.
    parser.add_argument("--composite_mask", type=int, default=0,
                        help="1 pastes G(x) into x through the fruit mask in the forward pass, G0X M1")
    parser.add_argument("--mask_dilate", type=int, default=0,
                        help="pixels the training mask grows before compositing, so the generator also "
                             "learns the band where the reviewed generation mask may differ")
    parser.add_argument("--mask_feather", type=float, default=0.0,
                        help="Gaussian sigma in pixels applied to the mask before compositing")
    return parser


class MelonLossMixin:
    """The extra terms, factored out so they can be unit tested without a model."""

    def melon_setup(self, opt):
        self.lambda_ssim = float(getattr(opt, "lambda_ssim", 0.0))
        self.lambda_background = float(getattr(opt, "lambda_background", 0.0))
        self.lambda_edge = float(getattr(opt, "lambda_edge", 0.0))
        self.lambda_color = float(getattr(opt, "lambda_color", 0.0))
        self.lambda_color_margin = float(getattr(opt, "lambda_color_margin", 0.0))
        self.color_margin = float(getattr(opt, "color_margin", 0.0))
        self.composite_mask = bool(int(getattr(opt, "composite_mask", 0)))
        self.mask_dilate = int(getattr(opt, "mask_dilate", 0))
        self.mask_feather = float(getattr(opt, "mask_feather", 0.0))
        self.diffaug = diffaugment.parse_policy(getattr(opt, "diffaug", ""))
        self.ema_decay = float(getattr(opt, "ema_decay", 0.0))
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("--ema_decay must be in [0, 1)")
        # Mixed precision only exists on CUDA; a CPU run with --amp 1 trains in float32.
        self.amp = bool(int(getattr(opt, "amp", 0))) and torch.cuda.is_available()
        self.centroids = None
        self.color_scale = None
        if self.lambda_color > 0 or self.lambda_color_margin > 0:
            path = getattr(opt, "color_centroids", "")
            if not path:
                raise ValueError("--color_centroids is required when a colour term is on")
            self.centroids = load_centroids(path)
            if self.lambda_color_margin > 0:
                self.color_scale = load_color_scale(path)

    def melon_mask(self, mask):
        """The mask used for compositing: dilated, then feathered, when asked."""
        return feather(dilate(mask, self.mask_dilate), self.mask_feather)

    def melon_composite(self, raw, source, mask):
        """Composite when module M1 is on, the raw output otherwise."""
        if not self.composite_mask:
            return raw
        return composite(raw, source, self.melon_mask(mask))

    def melon_augment(self, x):
        """DiffAugment of a discriminator input when module M4 is on."""
        return diffaugment.diff_augment(x, self.diffaug) if self.diffaug else x

    def melon_autocast(self, enabled=True):
        """float16 autocast on CUDA when --amp is on; a no-op context otherwise."""
        return torch.autocast("cuda", dtype=torch.float16, enabled=bool(self.amp and enabled))

    def melon_terms(self, real_A, real_B, fake_A, fake_B, rec_A, rec_B, mask_A, mask_B):
        """Return a dict of the extra losses, already weighted, always in float32.

        The colour descriptor uses cube roots and square roots that lose too much
        precision in float16, so these terms leave autocast and upcast their inputs.
        """
        with self.melon_autocast(enabled=False):
            return self._melon_terms(*(t.float() for t in (real_A, real_B, fake_A, fake_B,
                                                         rec_A, rec_B, mask_A, mask_B)))

    def _melon_terms(self, real_A, real_B, fake_A, fake_B, rec_A, rec_B, mask_A, mask_B):
        zero = torch.zeros((), device=real_A.device, dtype=real_A.dtype)
        out = dict(ssim_A=zero, ssim_B=zero, bg_A=zero, bg_B=zero,
                   edge_A=zero, edge_B=zero, color_A=zero, color_B=zero,
                   color_margin_A=zero, color_margin_B=zero)
        if self.lambda_ssim > 0:
            out["ssim_A"] = losses.ssim_cycle_loss(rec_A, real_A) * self.lambda_ssim
            out["ssim_B"] = losses.ssim_cycle_loss(rec_B, real_B) * self.lambda_ssim
        if self.lambda_background > 0:
            out["bg_A"] = losses.background_loss(fake_B, real_A, mask_A) * self.lambda_background
            out["bg_B"] = losses.background_loss(fake_A, real_B, mask_B) * self.lambda_background
        if self.lambda_edge > 0:
            out["edge_A"] = losses.edge_loss(fake_B, real_A, mask_A) * self.lambda_edge
            out["edge_B"] = losses.edge_loss(fake_A, real_B, mask_B) * self.lambda_edge
        if self.lambda_color > 0:
            # G_A maps A (unripe) into B (ripe), so its target centroid is B.
            out["color_A"] = losses.color_loss(fake_B, mask_A, self.centroids["B"]) * self.lambda_color
            out["color_B"] = losses.color_loss(fake_A, mask_B, self.centroids["A"]) * self.lambda_color
        if self.lambda_color_margin > 0:
            a, b, m, s = self.centroids["A"], self.centroids["B"], self.color_margin, self.color_scale
            out["color_margin_A"] = losses.color_margin_loss(fake_B, mask_A, b, a, s, m) * self.lambda_color_margin
            out["color_margin_B"] = losses.color_margin_loss(fake_A, mask_B, a, b, s, m) * self.lambda_color_margin
        return out


DIRECTION_NOTE = (
    f"domain A = {CLASS_NAMES[UNRIPE]} (class {UNRIPE}), domain B = {CLASS_NAMES[RIPE]} (class {RIPE}); "
    "G_A translates unripe to ripe"
)
