"""Generator variant for the enhanced model: multi-scale feature extraction.

The reference paper inserts a parallel 3x3 / 5x5 / 7x7 block after the first 7x7
convolution of the CycleGAN ResNet generator, concatenates the branches and
fuses them with a 1x1 convolution. That is the only architectural change made
here. Channel width stays at `ngf`..`4*ngf` (256 at ngf=64), which the plan asks
for at 256x256 input; the 512-channel stage and the Swin blocks of the reference
belong to its 512x512 setting and are not part of this experiment.
"""
from __future__ import annotations

import functools

import torch
import torch.nn as nn

from .config import vendor_on_path

vendor_on_path()
from models.networks import ResnetBlock, get_norm_layer, init_net  # noqa: E402

MULTISCALE_KERNELS = (3, 5, 7)


class SelectiveKernelBlock(nn.Module):
    """Two receptive fields (3x3 and dilated 3x3), per-channel softmax fusion.

    Lightweight SK-inspired adaptation, not an exact Tea CycleGAN reproduction.
    InstanceNorm and no BatchNorm in the gate allow training with batch size 1.
    """
    def __init__(self, channels):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.ReflectionPad2d(d),
                          nn.Conv2d(channels, channels, 3, dilation=d, bias=True),
                          nn.InstanceNorm2d(channels), nn.ReLU(False)) for d in (1, 2)])
        self.reduce = nn.Sequential(nn.Conv2d(channels, max(channels // 16, 16), 1), nn.ReLU(False))
        self.select = nn.Conv2d(max(channels // 16, 16), channels * 2, 1)

    def forward(self, x):
        branches = torch.stack([branch(x) for branch in self.branches], dim=1)
        context = branches.sum(1).mean((2, 3), keepdim=True)
        weights = self.select(self.reduce(context)).reshape(x.shape[0], 2, x.shape[1], 1, 1).softmax(1)
        return x + (branches * weights).sum(1)


class MultiScaleBlock(nn.Module):
    """Parallel odd-kernel convolutions, concatenated and fused by a 1x1 conv."""

    def __init__(self, channels, norm_layer, use_bias, kernels=MULTISCALE_KERNELS):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.ReflectionPad2d(k // 2),
                nn.Conv2d(channels, channels, kernel_size=k, padding=0, bias=use_bias),
                norm_layer(channels),
                nn.ReLU(True),
            )
            for k in kernels
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * len(kernels), channels, kernel_size=1, bias=use_bias),
            norm_layer(channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.fuse(torch.cat([branch(x) for branch in self.branches], dim=1))


class MultiScaleResnetGenerator(nn.Module):
    """ResNet generator with a multi-scale block after the first convolution."""

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, n_blocks=6, padding_type="reflect"):
        super().__init__()
        if n_blocks < 0:
            raise ValueError("n_blocks must be non-negative")
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf), nn.ReLU(True)]
        model += [MultiScaleBlock(ngf, norm_layer, use_bias)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2,
                                         padding=1, output_padding=1, bias=use_bias),
                      norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


def define_G(input_nc, output_nc, ngf, netG, norm="instance", use_dropout=False,
             init_type="normal", init_gain=0.02):
    """`models.networks.define_G` plus the multiscale variants."""
    from models import networks as upstream

    if netG == 'resize_resnet_6blocks':
        net = upstream.ResnetGenerator(input_nc, output_nc, ngf,
                                       norm_layer=get_norm_layer(norm_type=norm),
                                       use_dropout=use_dropout, n_blocks=6)
        # Keep the outer indices stable; only the two upsampling operators change.
        for i, layer in enumerate(net.model):
            if isinstance(layer, nn.ConvTranspose2d):
                net.model[i] = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.ReflectionPad2d(1), nn.Conv2d(layer.in_channels, layer.out_channels,
                        3, bias=layer.bias is not None))
        return net
    if netG == 'sk_resnet_6blocks':
        net = upstream.ResnetGenerator(input_nc, output_nc, ngf,
                                       norm_layer=get_norm_layer(norm_type=norm),
                                       use_dropout=use_dropout, n_blocks=6)
        layers = list(net.model.children())
        first_residual = next(i for i, layer in enumerate(layers) if isinstance(layer, ResnetBlock))
        layers.insert(first_residual, SelectiveKernelBlock(ngf * 4))
        net.model = nn.Sequential(*layers)
        return net
    if not netG.startswith("multiscale_"):
        return upstream.define_G(input_nc, output_nc, ngf, netG, norm, use_dropout, init_type, init_gain)
    blocks = {"multiscale_resnet_6blocks": 6, "multiscale_resnet_9blocks": 9}
    if netG not in blocks:
        raise NotImplementedError(f"Generator model name [{netG}] is not recognized")
    net = MultiScaleResnetGenerator(input_nc, output_nc, ngf, norm_layer=get_norm_layer(norm_type=norm),
                                    use_dropout=use_dropout, n_blocks=blocks[netG])
    return init_net(net, init_type, init_gain)
