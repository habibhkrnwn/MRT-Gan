"""Stage-oriented CLI. No command reads the test split's images or labels."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from melon_cyclegan import texture_study as s
from melon_cyclegan import texture_study_eval as e


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','train','export','diagnose','review','select','build-detectors','train-detectors','evaluate','plot'])
    p.add_argument('--study',required=True);p.add_argument('--dataset');p.add_argument('--crops')
    p.add_argument('--fraction',type=int,choices=[25,50,100],default=100)
    p.add_argument('--arms',nargs='+',choices=s.GEN_ARMS,default=list(s.GEN_ARMS))
    p.add_argument('--device',default='cpu');p.add_argument('--budget-minutes',type=float,default=30)
    p.add_argument('--bootstrap',type=int,default=1000);a=p.parse_args(argv)
    if a.stage in ('prepare','export','build-detectors') and not a.dataset:p.error('--dataset required')
    if a.stage=='prepare':
        if not a.crops:p.error('--crops required')
        s.prepare(a.dataset,a.crops,a.study)
    elif a.stage=='train':s.run_generators(a.study,a.fraction,a.arms,a.budget_minutes)
    elif a.stage=='export':s.export_candidates(a.study,a.fraction,a.arms,a.dataset,a.device)
    elif a.stage=='diagnose':e.diagnostics(a.study,a.fraction,a.arms,a.device)
    elif a.stage=='review':print(s.blind_review(a.study,a.fraction,a.arms))
    elif a.stage=='select':print(s.select(a.study))
    elif a.stage=='build-detectors':print(s.build_detectors(a.study,a.fraction,a.dataset))
    elif a.stage=='train-detectors':s.run_detectors(a.study,a.fraction)
    elif a.stage=='evaluate':e.evaluate(a.study,a.fraction,a.device,a.bootstrap)
    elif a.stage=='plot':e.plot_study(a.study)

if __name__=='__main__':main()
