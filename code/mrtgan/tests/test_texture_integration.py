import json, tempfile, subprocess, sys, os, unittest
from pathlib import Path
import numpy as np
from PIL import Image
root=Path(__file__).resolve().parents[1]
class IntegrationTest(unittest.TestCase):
    def test_wrapper_panel_and_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:

         t=Path(temp); crops=t/'pool'; out=t/'checkpoints'
         for domain in ('A','B'):
          (crops/f'train{domain}').mkdir(parents=True)
          (crops/f'train{domain}_mask').mkdir()
          for i in range(2):
           arr=np.random.default_rng(i).integers(0,256,(64,64,3),dtype=np.uint8)
           Image.fromarray(arr).save(crops/f'train{domain}/{i}.png')
           Image.fromarray(np.full((64,64),255,dtype=np.uint8)).save(crops/f'train{domain}_mask/{i}.png')
         cfg=json.loads((root/'config/texture_roi_sk.json').read_text())
         for k in ('training_policy','expected_crops_digest','color_margin_file','color_margin_percentile'):
          cfg.pop(k,None)
         cfg.update(experiment='smoke',ngf=8,ndf=8,texture_ndf=8,texture_size=32,n_epochs=1,n_epochs_decay=0,
                    lambda_color_margin=0,amp=0,load_size=64,crop_size=64,num_threads=0,
                    monitor_samples=1,sample_epoch_freq=1,save_epoch_freq=1)
         conf=t/'config.json';conf.write_text(json.dumps(cfg))
         cmd=[sys.executable,str(root/'tools/train_cyclegan.py'),'--config',str(conf),'--dataroot',str(crops),
              '--checkpoints-dir',str(out),'--','--ngf','8','--ndf','8','--print_freq','1']
         env={**os.environ,'PYTHONPATH':str(root),'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
         result=subprocess.run(cmd,env=env,text=True,capture_output=True)
         if result.returncode: print(result.stdout[-5000:],result.stderr);raise SystemExit(result.returncode)
         assert json.loads((out/'smoke/loss.json').read_text())['stopped'] is None
         cmd=[sys.executable,str(root/'tools/texture_panel.py'),'--crops',str(crops),'--checkpoints-dir',str(out),
              '--name','smoke','--weights','ema','--out',str(t/'panels'),'--count','2','--patch-size','32']
         result=subprocess.run(cmd,env=env,text=True,capture_output=True)
         if result.returncode: print(result.stdout,result.stderr);raise SystemExit(result.returncode)
         assert (t/'panels/panel_00.png').exists()
         # A second run must never overwrite existing weights.
         result=subprocess.run([sys.executable,str(root/'tools/train_cyclegan.py'),'--config',str(conf),
              '--dataroot',str(crops),'--checkpoints-dir',str(out)],env=env,text=True,capture_output=True)
         assert result.returncode != 0 and 'already exists' in result.stderr
         print('PASS: wrapper -> one CPU epoch -> local D/EMA checkpoints -> monitor -> inference panel -> overwrite guard')

if __name__ == "__main__":
    unittest.main()
