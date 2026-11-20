"""Screening runner for a reviewed bundle, with the provenance the notebook alone cannot give.

    python tools/run_screening.py --bundle ../04_Hasil/benchmark_gen_v1 \
        --dataset ../10_Migrasi_Kaggle/dataset_regrouped --seeds 0 1 2 3 4

Every arm is compared on the parents that all conditions share, so no condition carries a
parent the others lack. The real training crops still cover the whole scope; only the added
arms are restricted. The shortfall against the original S=N gate is reported, never hidden.
"""
from __future__ import annotations
import argparse, collections, json, platform, sys, time
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'src'))
import common, audit, pipeline, screening
from common import *


def shared_parents(accepted):
    return sorted(set.intersection(*[{r['parent_id'] for r in rows} for rows in accepted.values()]))


def pick(rows,parents,per_parent=1):
    """Deterministic selection, independent of any dev score."""
    grouped=collections.defaultdict(list)
    for record in rows:
        if record['parent_id'] in parents:grouped[record['parent_id']].append(record)
    picked=[]
    for pid in parents:
        ordered=sorted(grouped[pid],key=lambda r:(digest([r['candidate_id']]),r['candidate_id']))
        picked.extend(ordered[:per_parent])
    return picked


def versions():
    import sklearn,numpy,PIL
    out=dict(python=platform.python_version(),numpy=numpy.__version__,
             sklearn=sklearn.__version__,pillow=PIL.__version__)
    try:
        import torch,torchvision
        out.update(torch=torch.__version__,torchvision=torchvision.__version__)
    except Exception as error:out['torch']=f'unavailable: {error}'
    return out


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--bundle',required=True)
    parser.add_argument('--dataset',default=None)
    parser.add_argument('--encoder',default='mobilenet_v3_small')
    parser.add_argument('--seeds',type=int,nargs='+',default=[0,1,2,3,4])
    parser.add_argument('--per-parent',type=int,default=1)
    parser.add_argument('--size',type=int,default=224)
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--no-geometry',action='store_true')
    parser.add_argument('--exclude',nargs='*',default=[],
                        help='parents dropped from the real crops and from every added arm, '
                             'for example a parent whose source label is disputed')
    parser.add_argument('--fixed-threshold',action='store_true',
                        help='keep score_threshold at 0.25 instead of choosing it on calibration')
    parser.add_argument('--out',default=None,help='default: <bundle>/screening.json')
    args=parser.parse_args(argv)

    bundle=Path(args.bundle);dataset=Dataset(root=args.dataset)
    records=read_json(bundle/'records.json')
    if any(r['status']=='needs_review' for r in records):
        raise SystemExit('Unreviewed candidates present; run ingest_review first')
    conditions=sorted({r['condition'] for r in records})
    accepted={c:[r for r in records if r['condition']==c and r['status']=='accepted'] for c in conditions}
    scope=[pid for pid in dataset.scope_ids if pid not in set(args.exclude)]
    unknown=[pid for pid in args.exclude if pid not in dataset.scope_ids]
    if unknown:raise SystemExit(f'--exclude names parents outside the scope: {unknown}')
    parents=[pid for pid in shared_parents(accepted) if pid in scope]
    class_of={pid:dataset.by_id[pid]['class_id'] for pid in dataset.scope_ids}
    proposals=load_proposals(dataset)
    embed=screening.Embedder(args.encoder)
    started=time.time();results={};selection={}
    for condition in conditions:
        chosen=pick(accepted[condition],parents,args.per_parent)
        selection[condition]=dict(
            selected=[r['candidate_id'] for r in chosen],
            accepted=len(accepted[condition]),
            scope_gate=audit.quota(dataset.scope_ids,class_of,accepted[condition],args.per_parent))
        for seed in args.seeds:
            row=screening.condition_metrics(dataset,proposals,embed,bundle,chosen,seed=seed,
                                            size=args.size,repeats=args.repeats,parents=parents,
                                            geometry=not args.no_geometry,scope=scope,
                                            tune_threshold=not args.fixed_threshold,
                                            keep_predictions=True)
            results[f'{condition}@{seed}']=row
            print(condition,seed,'delta_synth',row.get('delta_real_plus_synthetic'),
                  'delta_geom',row.get('delta_real_plus_geometry'),
                  'synth vs geom',row.get('delta_synthetic_vs_geometry'),flush=True)
    splits={str(seed):dict(zip(('calibration','dev'),split_validation(dataset,seed))) for seed in args.seeds}
    payload=dict(
        results=results,selection=selection,shared_parents=parents,real_scope=scope,
        excluded=[pid for pid in dataset.scope_ids if pid not in parents],
        excluded_from_real=list(args.exclude),
        splits=splits,encoder=dict(name=embed.name,dim=embed.dim,lock=embed.lock),
        config=vars(args),versions=versions(),
        code=code_digest(),scope=dataset.scope_hash,manifest=EXPECTED_MANIFEST,
        elapsed_seconds=round(time.time()-started,1),test_used=False,
        note=('screening with a frozen detector and a frozen encoder; the scope gate against all '
              '10 parents is reported per condition in `selection`, the comparison itself runs on '
              'the shared parents'))
    out=Path(args.out or bundle/'screening.json');write_json(out,payload)
    print('wrote',out,'arms',len(results),'shared parents',len(parents))
    return payload


if __name__=='__main__':main()
