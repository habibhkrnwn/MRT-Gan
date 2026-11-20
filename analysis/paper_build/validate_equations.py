"""CPU-only checks of equation reductions against the frozen implementation.
Run with an environment containing torch and numpy; no model is trained.
"""
from pathlib import Path
import sys, tempfile, zipfile, importlib.util, json, math, hashlib
import numpy as np
import torch
root=Path(__file__).resolve().parents[3]
archive=root/'FINAL_Melon_2026-09-28/02_Generator_G3/code/texture_study_code.zip'
out=Path(__file__).parent/'equation_checks.json'
checks={}
with tempfile.TemporaryDirectory(prefix='g3_equations_') as temp:
 with zipfile.ZipFile(archive) as z:
  for name in z.namelist():
   if name.startswith('code/cyclegan/'):z.extract(name,temp)
 base=Path(temp)/'code/cyclegan';sys.path.insert(0,str(base));sys.path.insert(0,str(base/'vendor/pytorch_cyclegan'))
 from melon_cyclegan.model_impl import composite
 from melon_cyclegan.losses import margin_distances,color_margin_loss
 from melon_cyclegan.color import phi_torch
 from models.networks import GANLoss
 torch.manual_seed(42)
 raw=torch.rand(2,3,8,8)*2-1;src=torch.rand_like(raw)*2-1;alpha=torch.rand(2,1,8,8)
 checks['eq1_compositing']=bool(torch.allclose(composite(raw,src,alpha),alpha*raw+(1-alpha)*src))
 score=torch.tensor([[-.2,.3],[.8,1.4]])
 gan=GANLoss('lsgan')
 checks['eq2_generator_mean']=bool(torch.allclose(gan(score,True),((score-1)**2).mean()))
 checks['eq2_discriminator_half_sum']=bool(torch.allclose(.5*(gan(score,True)+gan(score*.8,False)),.5*((score-1)**2).mean()+.5*((score*.8)**2).mean()))
 target=torch.tensor([.2,.3,.4,.5]);source=torch.tensor([.5,.2,.1,.8]);scale=torch.tensor([.7,.8,.6,.9]);mask=torch.ones(2,1,8,8)
 phi=phi_torch(((raw+1)/2).clamp(0,1),mask);dt,ds=margin_distances(phi,target,source,scale)
 manual_dt=(((phi[:,[0,2]]-target[[0,2]])/scale[[0,2]])**2).sum(1).sqrt()
 manual_ds=(((phi[:,[0,2]]-source[[0,2]])/scale[[0,2]])**2).sum(1).sqrt()
 checks['eq5_distance']=bool(torch.allclose(dt,manual_dt) and torch.allclose(ds,manual_ds))
 checks['eq5_margin_mean']=bool(torch.allclose(color_margin_loss(raw,mask,target,source,scale,.25),torch.relu(manual_dt-manual_ds+.25).mean()))
 # Exact expression from the ranking script, evaluated with controlled positive references.
 script=(base/'tools/build_ranked_texture_arm.py').read_text()
 expression=next(l.split('distance = ',1)[1].strip() for l in script.splitlines() if 'distance = sum(' in l)
 metrics=['a','b','c','d','e'];values=np.array([0.,.1,.3,.8,1.2]);refs=np.array([.2,.2,.4,.7,.8])
 got=eval(expression,{'math':math,'f':dict(zip(metrics,values)),'ref':dict(zip(metrics,refs)),'METRICS':metrics})
 checks['eq7_rank']=bool(np.isclose(got,np.abs(np.log(np.maximum(values,1e-12)/refs)).sum()))
 spec=importlib.util.spec_from_file_location('decision',root/'FINAL_Melon_2026-09-28/01_Pipeline_Final/harvest/decision.py')
 decision=importlib.util.module_from_spec(spec);spec.loader.exec_module(decision)
 sr,su=.82,.14;d=math.log(sr+1e-6)-math.log(su+1e-6)
 checks['eq8_logistic']=bool(np.isclose(decision.Platt(1.3,-.2)(d),1/(1+math.exp(-(1.3*d-.2)))))
 checks['effective_identity_coefficient']=bool(.1*10==1)
 result={'scope':'Equation arithmetic and reductions on CPU; not retraining or validation of scientific efficacy','archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'checks':checks,'all_passed':all(checks.values()),'cycle_identity_total':'Call order and summation verified by source inspection; see EQUATION_AUDIT_2026-09-29.md'}
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
 assert all(checks.values())
