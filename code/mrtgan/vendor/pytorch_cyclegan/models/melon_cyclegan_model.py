"""Bridge file: registers the enhanced melon model as `--model melon_cyclegan`.

Everything of the upstream CycleGAN objective is inherited; the extra terms come
from `melon_cyclegan/model_impl.py` and the multiscale generator from
`melon_cyclegan/networks.py`. With every extra weight at zero and the G0X
switches off (`composite_mask 0`, empty `diffaug`) the objective is identical
to the upstream one.
"""
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import copy
import itertools

import torch
from util.image_pool import ImagePool

from .cycle_gan_model import CycleGANModel
from melon_cyclegan import networks as melon_networks
from melon_cyclegan.model_impl import MelonLossMixin, add_options, ema_update


class MelonCycleganModel(MelonLossMixin, CycleGANModel):
    """CycleGAN plus SSIM-cycle, background, edge, colour and the G0X modules."""

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--controlled_init_seed', type=int, default=-1)
        return add_options(parser, is_train)

    def __init__(self, opt):
        # Build the networks with the melon `define_G`, which knows the
        # multiscale variants, then reuse everything else from upstream.
        from models.base_model import BaseModel

        BaseModel.__init__(self, opt)
        self.loss_names = ["D_A", "G_A", "cycle_A", "idt_A", "D_B", "G_B", "cycle_B", "idt_B",
                           "ssim_A", "ssim_B", "bg_A", "bg_B", "edge_A", "edge_B", "color_A", "color_B",
                           "color_margin_A", "color_margin_B"]
        visual_names_A = ["real_A", "fake_B", "rec_A"]
        visual_names_B = ["real_B", "fake_A", "rec_B"]
        if self.isTrain and opt.lambda_identity > 0.0:
            visual_names_A.append("idt_B")
            visual_names_B.append("idt_A")
        self.visual_names = visual_names_A + visual_names_B
        self.model_names = ["G_A", "G_B", "D_A", "D_B"] if self.isTrain else ["G_A", "G_B"]

        from models import networks

        self.netG_A = melon_networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
                                              opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain)
        self.netG_B = melon_networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG,
                                              opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain)
        if self.isTrain:
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D,
                                            opt.norm, opt.init_type, opt.init_gain)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D,
                                            opt.norm, opt.init_type, opt.init_gain)
            if opt.lambda_identity > 0.0:
                assert opt.input_nc == opt.output_nc
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(
                itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        self.melon_setup(opt)
        if self.isTrain and self.amp:
            self.scaler_G = torch.amp.GradScaler("cuda")
            self.scaler_D = torch.amp.GradScaler("cuda")

    def setup(self, opt):
        """Upstream setup, then the EMA shadows.

        The shadows are copied from the generators only after upstream setup has
        (re)initialised them, so both start from the same weights. They join
        `model_names` afterwards and are saved with every checkpoint.
        """
        super().setup(opt)
        if self.isTrain and self.ema_decay > 0:
            for key in ("G_A", "G_B"):
                shadow = copy.deepcopy(getattr(self, "net" + key))
                shadow.eval()
                for parameter in shadow.parameters():
                    parameter.requires_grad_(False)
                setattr(self, f"net{key}_ema", shadow)
                self.model_names.append(f"{key}_ema")

    def _update_ema(self):
        if self.ema_decay > 0 and hasattr(self, "netG_A_ema"):
            ema_update(self.netG_A_ema, self.netG_A, self.ema_decay)
            ema_update(self.netG_B_ema, self.netG_B, self.ema_decay)

    def _backward(self, loss, scaler_name):
        """Plain backward, or the scaled backward of mixed precision."""
        if self.amp:
            getattr(self, scaler_name).scale(loss).backward()
        else:
            loss.backward()

    def _step(self, optimizer, scaler_name):
        if self.amp:
            scaler = getattr(self, scaler_name)
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

    def optimize_parameters(self):
        """Upstream order of updates; only the backward and step go through the AMP scalers."""
        self.forward()
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self._step(self.optimizer_G, "scaler_G")
        self._update_ema()
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self._step(self.optimizer_D, "scaler_D")

    def set_input(self, input):
        AtoB = self.opt.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        self.real_B = input["B" if AtoB else "A"].to(self.device)
        self.mask_A = input["A_mask" if AtoB else "B_mask"].to(self.device)
        self.mask_B = input["B_mask" if AtoB else "A_mask"].to(self.device)
        self.image_paths = input["A_paths" if AtoB else "B_paths"]

    def forward(self):
        """Upstream forward; with M1 on, every translation is pasted into its source.

        The reconstruction is composited through the mask of the image it
        reconstructs, so outside the fruit it equals that image exactly and the
        cycle loss only sees the fruit.
        """
        with self.melon_autocast():
            self.fake_B = self.melon_composite(self.netG_A(self.real_A), self.real_A, self.mask_A)
            self.rec_A = self.melon_composite(self.netG_B(self.fake_B), self.fake_B, self.mask_A)
            self.fake_A = self.melon_composite(self.netG_B(self.real_B), self.real_B, self.mask_B)
            self.rec_B = self.melon_composite(self.netG_A(self.fake_A), self.fake_A, self.mask_B)

    def backward_D_basic(self, netD, real, fake):
        """Upstream discriminator loss; with M4 on, both inputs pass DiffAugment."""
        with self.melon_autocast():
            pred_real = netD(self.melon_augment(real))
            loss_D_real = self.criterionGAN(pred_real.float(), True)
            pred_fake = netD(self.melon_augment(fake.detach()))
            loss_D_fake = self.criterionGAN(pred_fake.float(), False)
            loss_D = (loss_D_real + loss_D_fake) * 0.5
        self._backward(loss_D, "scaler_D")
        return loss_D

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        with self.melon_autocast():
            self._generator_losses(lambda_idt, lambda_A, lambda_B)
        extra = self.melon_terms(self.real_A, self.real_B, self.fake_A, self.fake_B,
                                 self.rec_A, self.rec_B, self.mask_A, self.mask_B)
        for name, value in extra.items():
            setattr(self, f"loss_{name}", value)

        self.loss_G = (self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B
                       + self.loss_idt_A + self.loss_idt_B + sum(extra.values()))
        self._backward(self.loss_G, "scaler_G")

    def _generator_losses(self, lambda_idt, lambda_A, lambda_B):
        """Upstream adversarial, cycle and identity terms; float32 losses under AMP."""
        if lambda_idt > 0:
            self.idt_A = self.melon_composite(self.netG_A(self.real_B), self.real_B, self.mask_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A.float(), self.real_B) * lambda_B * lambda_idt
            self.idt_B = self.melon_composite(self.netG_B(self.real_A), self.real_A, self.mask_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B.float(), self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0
        self.loss_G_A = self.criterionGAN(self.netD_A(self.melon_augment(self.fake_B)).float(), True)
        self.loss_G_B = self.criterionGAN(self.netD_B(self.melon_augment(self.fake_A)).float(), True)
        self.loss_cycle_A = self.criterionCycle(self.rec_A.float(), self.real_A) * lambda_A
        self.loss_cycle_B = self.criterionCycle(self.rec_B.float(), self.real_B) * lambda_B
