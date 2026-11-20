import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from melon_cyclegan.config import vendor_on_path
from melon_cyclegan.texture import patch_indices, extract_patches
from melon_cyclegan.networks import define_G
vendor_on_path()
from options.train_options import TrainOptions
from models import create_model


class TextureTests(unittest.TestCase):
    def test_mask_alignment_gradient_and_empty(self):
        mask = torch.zeros(1, 1, 96, 96)
        mask[:, :, 16:80, 16:80] = 1
        idx = patch_indices(mask, 32, 16, 4, 1, random=False)
        x = torch.randn(1, 3, 96, 96, requires_grad=True)
        patches = extract_patches(x, idx, 32, 16, False)
        self.assertEqual(tuple(patches.shape), (4, 3, 32, 32))
        self.assertEqual(float(extract_patches(mask.expand(-1,3,-1,-1), idx,32,16,False).min()), 1)
        patches.sum().backward()
        self.assertEqual(float((x.grad * (1-mask)).abs().sum()), 0)
        with self.assertRaises(ValueError):
            patch_indices(torch.zeros_like(mask))

    def test_sk_batch_one_and_roundtrip(self):
        model = define_G(3, 3, 8, 'sk_resnet_6blocks')
        x = torch.randn(1, 3, 64, 64)
        out = model(x)
        self.assertEqual(out.shape, x.shape)
        out.square().mean().backward()
        sk = next(m for m in model.modules() if m.__class__.__name__ == 'SelectiveKernelBlock')
        self.assertGreater(float(sk.select.weight.grad.abs().sum()), 0)
        other = define_G(3, 3, 8, 'sk_resnet_6blocks')
        other.load_state_dict(model.state_dict())
        torch.testing.assert_close(out, other(x))

    def run_step(self, architecture, amp=False):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            previous = sys.argv
            try:
                sys.argv = ['test', '--dataroot', tmp, '--checkpoints_dir', tmp, '--name', 'smoke',
                            '--model', 'melon_texture', '--netG', architecture, '--ngf', '8', '--ndf', '8',
                            '--texture_ndf', '8', '--texture_size', '32', '--texture_stride', '16',
                            '--texture_count', '2', '--lambda_ssim', '0', '--lambda_background', '0',
                            '--lambda_color', '0', '--lambda_color_margin', '0', '--composite_mask', '1',
                            '--ema_decay', '0.9', '--amp', str(int(amp)), '--norm', 'instance']
                opt = TrainOptions().parse()
            finally:
                sys.argv = previous
            opt.device = torch.device('cuda' if amp else 'cpu')
            model = create_model(opt)
            model.setup(opt)
            data = dict(A=torch.randn(1,3,64,64), B=torch.randn(1,3,64,64),
                        A_mask=torch.ones(1,1,64,64), B_mask=torch.ones(1,1,64,64), A_paths=['a'], B_paths=['b'])
            # Deliberately unequal inputs: confirm fake_A uses source B coordinates.
            data['A_mask'][:,:,:, :16] = 0
            data['B_mask'][:,:,:, 48:] = 0
            model.set_input(data)
            before = next(model.netT_B.parameters()).detach().clone()
            g_weight = model.netG_B.model[1].weight
            # GradScaler starts at 2**16 and skips the whole optimizer step when fp16
            # gradients overflow (inf/nan grads), then halves the scale. scaler_G and
            # scaler_D do this independently. A skip is accepted only when that scaler's
            # scale dropped; the loop ends on a step where G stepped with finite grads.
            for _ in range(16 if amp else 1):
                scale_G = model.scaler_G.get_scale() if amp else None
                scale_D = model.scaler_D.get_scale() if amp else None
                t_before = next(model.netT_B.parameters()).detach().clone()
                model.optimize_parameters()
                g_finite = bool(torch.isfinite(g_weight.grad).all())
                if not g_finite:
                    self.assertTrue(amp and model.scaler_G.get_scale() < scale_G,
                                    'non-finite generator grads without an fp16 overflow')
                if torch.equal(t_before, next(model.netT_B.parameters())):
                    self.assertTrue(amp and model.scaler_D.get_scale() < scale_D,
                                    'texture D step skipped without an fp16 overflow')
                if g_finite and not torch.equal(before, next(model.netT_B.parameters())):
                    break
            self.assertFalse(torch.equal(before, next(model.netT_B.parameters())))
            self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in model.get_current_losses().values()))
            self.assertTrue(bool(torch.isfinite(g_weight.grad).all()))
            self.assertGreater(float(g_weight.grad.abs().sum()), 0)
            model.save_networks('latest')
            self.assertTrue((Path(tmp)/'smoke/latest_net_T_B.pth').exists())
            self.assertTrue((Path(tmp)/'smoke/latest_net_G_B_ema.pth').exists())
            from melon_cyclegan.generate import load_generators
            identity = dict(config=dict(netG=architecture, ngf=8, norm='instance'))
            (Path(tmp)/'smoke/melon_identity.json').write_text(json.dumps(identity))
            loaded, _ = load_generators(tmp, 'smoke', weights='ema')
            loaded['G_B'](torch.randn(1,3,64,64))

    def test_roi_training_and_inference(self):
        self.run_step('resnet_6blocks')

    def test_roi_sk_training_and_inference(self):
        self.run_step('sk_resnet_6blocks')

    @unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
    def test_amp(self):
        self.run_step('sk_resnet_6blocks', amp=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
