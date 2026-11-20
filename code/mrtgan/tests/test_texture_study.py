import contextlib
import io
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path
import numpy as np
import torch
from melon_cyclegan.config import vendor_on_path
from melon_cyclegan.texture_study import nested_subsets
from melon_cyclegan.texture_study_eval import grouped_draws,spectral_share,image_features
from melon_cyclegan.texture import patch_indices
from melon_cyclegan.networks import define_G
vendor_on_path()
from models import create_model
from options.train_options import TrainOptions

class StudyTests(unittest.TestCase):
    def test_grouped_nested_subsets(self):
        rows=[dict(id=f'{c}_{i}_{j}',photograph_group=f'{c}_{i}',class_id=str(c),split='train')
              for c,n in [(0,12),(1,6)] for i in range(n) for j in range(2)]
        subsets=nested_subsets(rows)
        for small,big in [(25,50),(50,100)]:
            self.assertLessEqual({r['id'] for r in subsets[small]},{r['id'] for r in subsets[big]})
        for rs in subsets.values():
            for r in rs:self.assertEqual(sum(x['photograph_group']==r['photograph_group'] for x in rs),2)
        self.assertEqual(subsets,nested_subsets(list(reversed(rows))))
        with self.assertRaises(ValueError):nested_subsets(rows+[dict(rows[0],split='valid')])

    def test_local_patch_rng(self):
        mask=torch.ones(1,1,96,96);torch.manual_seed(42);before=torch.get_rng_state().clone()
        a=patch_indices(mask,32,16,4,.95,generator=torch.Generator().manual_seed(11))
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        b=patch_indices(mask,32,16,4,.95,generator=torch.Generator().manual_seed(11))
        self.assertTrue(torch.equal(a,b))

    def test_resize_gradient_and_roundtrip(self):
        model=define_G(3,3,8,'resize_resnet_6blocks');x=torch.randn(1,3,64,64)
        self.assertFalse(any(isinstance(m,torch.nn.ConvTranspose2d) for m in model.modules()))
        y=model(x);self.assertEqual(y.shape,x.shape);y.square().mean().backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))
        other=define_G(3,3,8,'resize_resnet_6blocks');other.load_state_dict(model.state_dict());torch.testing.assert_close(y,other(x))

    def test_shared_initialization(self):
        states=[]
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            for typ,gray in [('melon_cyclegan',1),('melon_texture',1),('melon_texture',0)]:
                torch.manual_seed(11)
                args=['test','--dataroot',tmp,'--checkpoints_dir',tmp,'--model',typ,'--name',typ+str(gray),
                    '--ngf','8','--ndf','8','--netG','resnet_6blocks','--controlled_init_seed','11','--lambda_color','0','--lambda_color_margin','0']
                if typ=='melon_texture':args+=['--texture_gray',str(gray),'--texture_seed','11001']
                with unittest.mock.patch.object(sys,'argv',args):opt=TrainOptions().parse()
                opt.device=torch.device('cpu');model=create_model(opt);model.setup(opt)
                states.append({n:{k:v.detach().clone() for k,v in getattr(model,'net'+n).state_dict().items()} for n in ('G_A','G_B','D_A','D_B')})
            for s in states[1:]:
                for n in s:
                    for k in s[n]:self.assertTrue(torch.equal(states[0][n][k],s[n][k]),(n,k))

    def test_cluster_resampling_and_chroma(self):
        for sample in grouped_draws({'a':'g1','b':'g1','c':'g2'},20):self.assertEqual(sample.count('a'),sample.count('b'))
        checker=np.indices((32,32)).sum(0)%2
        self.assertGreater(spectral_share(checker),.99)
        self.assertEqual(spectral_share(np.ones((32,32))),0)
        rgb=np.full((64,64,3),.5);rgb[...,0]+=(np.indices((64,64)).sum(0)%2)*.1
        f=image_features(rgb,np.ones((64,64),bool),[(16,16)])
        self.assertGreater(f['chroma_energy'],0);self.assertTrue(np.isfinite(f['nyquist_r_minus_luma']))

if __name__=='__main__':
    from unittest import mock
    torch.set_num_threads(2);unittest.main()

class NewArmIntegrationTests(unittest.TestCase):
    def test_resize_and_rgb_training_checkpoint_inference(self):
        import json, os, subprocess
        from PIL import Image
        from melon_cyclegan.generate import load_generators
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);pool=tmp/'pool'
            for domain in ('A','B'):
                (pool/f'train{domain}').mkdir(parents=True);(pool/f'train{domain}_mask').mkdir()
                Image.fromarray(np.random.default_rng(ord(domain)).integers(0,256,(64,64,3),dtype=np.uint8)).save(pool/f'train{domain}/0.png')
                Image.new('L',(64,64),255).save(pool/f'train{domain}_mask/0.png')
            for arm,net,model,gray in [('G1','resize_resnet_6blocks','melon_cyclegan',1),('G3','resnet_6blocks','melon_texture',0)]:
                conf=json.loads((root/'config/texture_roi.json').read_text())
                for k in ('training_policy','expected_crops_digest','color_margin_file','color_margin_percentile'):conf.pop(k,None)
                conf.update(experiment=arm,netG=net,model=model,texture_gray=gray,texture_size=32,texture_ndf=8,
                    n_epochs=1,n_epochs_decay=0,controlled_init_seed=11,texture_seed=11001,lambda_color_margin=0,
                    amp=0,num_threads=0,monitor_samples=0,load_size=64,crop_size=64,ngf=8,ndf=8)
                path=tmp/f'{arm}.json';path.write_text(json.dumps(conf))
                command=[sys.executable,str(root/'tools/train_cyclegan.py'),'--config',str(path),'--dataroot',str(pool),
                    '--checkpoints-dir',str(tmp/'checkpoints'),'--','--ngf','8','--ndf','8']
                res=subprocess.run(command,env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'},capture_output=True,text=True)
                self.assertEqual(res.returncode,0,res.stdout[-1500:]+res.stderr)
                run=tmp/'checkpoints'/arm;self.assertTrue((run/'initial_weights.json').exists())
                nets,_=load_generators(tmp/'checkpoints',arm,weights='ema')
                with torch.no_grad():y=nets['G_B'](torch.randn(1,3,64,64))
                self.assertTrue(torch.isfinite(y).all())
                if arm=='G3':self.assertTrue((run/'latest_net_T_B.pth').exists())

    def test_operating_point_rematches_after_threshold(self):
        from melon_cyclegan.texture_study_eval import operating_recall
        table={'a':dict(conf=np.array([.9,.1]),pred_cls=np.array([1,1]),target_cls=np.array([1]),
            pred_boxes=np.array([[0,0,9,9],[0,0,10,10]]),gt_boxes=np.array([[0,0,10,10]]))}
        self.assertEqual(operating_recall(table,['a'],.5)['recall'],1.)
