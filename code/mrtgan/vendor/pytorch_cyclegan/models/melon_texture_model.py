"""Experimental G0X + interior texture discriminators, independent of legacy model."""
import torch
from .melon_cyclegan_model import MelonCycleganModel
from . import networks
from melon_cyclegan.texture import patch_indices, extract_patches


class MelonTextureModel(MelonCycleganModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = MelonCycleganModel.modify_commandline_options(parser, is_train)
        parser.add_argument('--lambda_texture', type=float, default=0.5)
        parser.add_argument('--texture_size', type=int, default=64)
        parser.add_argument('--texture_stride', type=int, default=16)
        parser.add_argument('--texture_count', type=int, default=4)
        parser.add_argument('--texture_coverage', type=float, default=0.95)
        parser.add_argument('--texture_gray', type=int, default=1)
        parser.add_argument('--texture_seed', type=int, default=-1)
        parser.add_argument('--texture_ndf', type=int, default=32)
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        if not 0 <= opt.lambda_texture < float('inf'):
            raise ValueError('lambda_texture must be finite and nonnegative')
        self.texture_rng = (torch.Generator(device=self.device).manual_seed(opt.texture_seed)
                            if opt.texture_seed >= 0 else None)
        self.texture_on = self.isTrain and opt.lambda_texture > 0
        if self.texture_on:
            self.loss_names += ['texture_G_A', 'texture_G_B', 'texture_D_A', 'texture_D_B']
            for domain in ('A', 'B'):
                net = networks.define_D(1 if opt.texture_gray else 3, opt.texture_ndf,
                                       'n_layers', 2, 'instance', opt.init_type, opt.init_gain)
                setattr(self, 'netT_' + domain, net)
                self.model_names.append('T_' + domain)
                self.optimizer_D.add_param_group({'params': list(net.parameters())})

    def forward(self):
        super().forward()
        if not self.texture_on or not torch.is_grad_enabled():
            return
        o = self.opt
        # A=unripe; fake_A comes from B and must use B's mask/coordinates.
        ia = patch_indices(self.mask_A, o.texture_size, o.texture_stride, o.texture_count, o.texture_coverage, generator=self.texture_rng)
        ib = patch_indices(self.mask_B, o.texture_size, o.texture_stride, o.texture_count, o.texture_coverage, generator=self.texture_rng)
        extract = lambda x, i: extract_patches(x, i, o.texture_size, o.texture_stride, bool(o.texture_gray))
        self.texture_real_A, self.texture_real_B = extract(self.real_A, ia), extract(self.real_B, ib)
        self.texture_fake_A, self.texture_fake_B = extract(self.fake_A, ib), extract(self.fake_B, ia)

    def _generator_losses(self, *args):
        super()._generator_losses(*args)
        if self.texture_on:
            # No cutout/color augmentation on these small interior patches.
            # Raw texture signal complements the existing augmented global D.
            self.loss_texture_G_A = self.criterionGAN(self.netT_A(self.texture_fake_B).float(), True) * self.opt.lambda_texture
            self.loss_texture_G_B = self.criterionGAN(self.netT_B(self.texture_fake_A).float(), True) * self.opt.lambda_texture
            self.loss_G_A = self.loss_G_A + self.loss_texture_G_A
            self.loss_G_B = self.loss_G_B + self.loss_texture_G_B

    def _texture_D(self, net, real, fake):
        with self.melon_autocast():
            loss = (self.criterionGAN(net(real).float(), True)
                    + self.criterionGAN(net(fake.detach()).float(), False)) * 0.5 * self.opt.lambda_texture
        self._backward(loss, 'scaler_D')
        return loss

    def backward_D_A(self):
        super().backward_D_A()
        if self.texture_on:
            self.loss_texture_D_A = self._texture_D(self.netT_A, self.texture_real_B, self.texture_fake_B)

    def backward_D_B(self):
        super().backward_D_B()
        if self.texture_on:
            self.loss_texture_D_B = self._texture_D(self.netT_B, self.texture_real_A, self.texture_fake_A)

    def optimize_parameters(self):
        if not self.texture_on:
            return super().optimize_parameters()
        self.forward()
        discriminators = [self.netD_A, self.netD_B, self.netT_A, self.netT_B]
        self.set_requires_grad(discriminators, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self._step(self.optimizer_G, 'scaler_G')
        self._update_ema()
        self.set_requires_grad(discriminators, True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self._step(self.optimizer_D, 'scaler_D')
