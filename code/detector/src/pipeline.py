"""Generation driver: locked scope, resumable runs, per-condition cost, blind review.

Candidates leave this module as `needs_review` proposals. Nothing here may promote a
candidate to accepted; only `ingest_review` with a named reviewer does that.
"""
from __future__ import annotations
import collections, random, shutil, time
from pathlib import Path
from PIL import Image
from common import *
import generators

CLASS_NAME={'0':'ripe','1':'unripe'}
GENERATION_MODULES=['common.py','generators.py','pipeline.py','recon_attention.py','recon_engine.py']
REVIEW_FIELDS=['review_id','image','label_class','annotation_ok','notes']


SEED_SCOPES=('condition','paired')


def identity_of(conditions,dataset,slots,seed_scope='condition',parents=None):
    if seed_scope not in SEED_SCOPES:raise ValueError(f'Unknown seed_scope: {seed_scope}')
    return dict(version=VERSION,code=code_digest(GENERATION_MODULES),manifest=EXPECTED_MANIFEST,
                scope=dataset.scope_hash,slots=slots,seed_scope=seed_scope,
                parents=list(parents or dataset.scope_ids),
                conditions=[{k:c[k] for k in sorted(c)} for c in conditions])


def seed_for(condition,pid,slot,seed_scope):
    """'paired' gives every condition the same noise per (parent, slot).

    With 'condition' a configuration change and a different random draw move together, so a
    Canny/ReCon or strength comparison mixes the two.
    """
    key=condition['id'] if seed_scope=='condition' else 'paired'
    return stable_seed(VERSION,key,pid,slot)


def _record(dataset,condition,pid,slot,seed,seconds,image_path,mask_path,mask,geometry,trace):
    return dict(candidate_id=image_path.stem,condition=condition['id'],method=condition['method'],
                parent_id=pid,slot=slot,class_id=dataset.by_id[pid]['class_id'],seed=seed,
                seconds=round(seconds,3),bbox=mask_box(mask),geometry=geometry,trace=trace,
                image=f'images/{image_path.name}',mask=f'masks/{mask_path.name}',
                image_sha256=sha(image_path),mask_sha256=sha(mask_path),
                status='needs_review',reviewer='',
                annotation_basis='parent mask proposal; review decides whether it still fits')


def run(dataset,conditions,out,budget,slots=1,input_root='/kaggle/input',seed_scope='condition',
        parents=None):
    """Generate every pending (condition, parent, slot). Safe to re-run after a stop.

    `parents` restricts generation to part of the frozen scope, for example when one parent's
    source label is disputed. It is bound into identity.json, so it cannot change silently.
    """
    parents=list(parents or dataset.scope_ids)
    outside=[pid for pid in parents if pid not in dataset.scope_ids]
    if outside:raise ValueError(f'Parents outside the frozen scope: {outside}')
    out=Path(out);bind_identity(out,identity_of(conditions,dataset,slots,seed_scope,parents))
    for sub in ('images','masks'):(out/sub).mkdir(parents=True,exist_ok=True)
    lock,sam=generators.lock_models(out,conditions,budget,input_root)
    path=out/'records.json';records=read_json(path) if path.exists() else []
    done={(r['condition'],r['parent_id'],r['slot']) for r in records}
    cost=read_json(out/'cost.json') if (out/'cost.json').exists() else {}
    stopped=None
    for condition in conditions:
        pending=[(pid,slot) for pid in parents for slot in range(slots)
                 if (condition['id'],pid,slot) not in done]
        if not pending or stopped:continue
        entry=cost.setdefault(condition['id'],dict(method=condition['method'],warmup_seconds=0.,
                                                   generate_seconds=0.,proposals=0))
        adapter=None
        try:
            started=time.time();adapter=generators.Adapter(condition,lock,sam,budget,out)
            entry['warmup_seconds']=round(entry['warmup_seconds']+time.time()-started,3)
            for pid,slot in pending:
                budget.check()
                source,mask,geometry=dataset.prepare(pid,condition['size'])
                seed=seed_for(condition,pid,slot,seed_scope)
                started=time.time()
                image,out_mask,trace=adapter.generate(source,mask,pid,int(dataset.by_id[pid]['class_id']),slot,seed)
                seconds=time.time()-started
                name=f"{condition['id']}__{pid}__{slot}"
                image_path=out/'images'/f'{name}.png';mask_path=out/'masks'/f'{name}.png'
                image.save(image_path);out_mask.convert('L').save(mask_path)
                records.append(_record(dataset,condition,pid,slot,seed,seconds,image_path,mask_path,
                                       out_mask,geometry,trace))
                entry['generate_seconds']=round(entry['generate_seconds']+seconds,3)
                entry['proposals']+=1
                entry['seconds_per_proposal']=round(entry['generate_seconds']/entry['proposals'],3)
                write_json(path,records);write_json(out/'cost.json',cost)
        except BudgetStop as stop:
            stopped=str(stop)
        finally:
            if adapter is not None:adapter.close()
    write_json(path,records);write_json(out/'cost.json',cost)
    progress=dict(version=VERSION,stopped=stopped,records=len(records),
                  expected=len(conditions)*len(parents)*slots,
                  per_condition={c['id']:sum(1 for r in records if r['condition']==c['id']) for c in conditions},
                  elapsed_seconds=round(budget.elapsed,1))
    write_json(out/'progress.json',progress)
    return records,progress


def export_review(out,seed=0):
    """Anonymous sheet. The reviewer sees no method, no parent and no target class."""
    out=Path(out);records=read_json(out/'records.json')
    if not records:raise ValueError('No candidates to review')
    blind=out/'review'/'blind';blind.mkdir(parents=True,exist_ok=True)
    order=sorted(records,key=lambda r:r['candidate_id'])
    random.Random(seed).shuffle(order)
    key={};rows=[]
    for i,record in enumerate(order):
        review_id=f'R{i:04d}';key[review_id]=record['candidate_id']
        shutil.copyfile(out/record['image'],blind/f'{review_id}.png')
        rows.append(dict(review_id=review_id,image=f'blind/{review_id}.png',
                         label_class='',annotation_ok='',notes=''))
    write_csv(out/'review'/'review_sheet.csv',rows,REVIEW_FIELDS)
    write_json(out/'review'/'key.json',key)
    return dict(sheet=str(out/'review'/'review_sheet.csv'),candidates=len(rows),
                fields=REVIEW_FIELDS,blind=True)


def ingest_review(out,reviewer,sheet=None):
    """Turn a filled sheet into accepted/rejected. Blank rows are an error, not a rejection."""
    out=Path(out)
    if not reviewer.strip():raise ValueError('Reviewer name required')
    rows=read_csv(sheet or out/'review'/'review_sheet.csv')
    key=read_json(out/'review'/'key.json')
    records=read_json(out/'records.json');by_id={r['candidate_id']:r for r in records}
    if {r['review_id'] for r in rows}!=set(key):raise ValueError('Review sheet does not match the key')
    stamp=time.strftime('%Y-%m-%dT%H:%M:%S%z')
    for row in rows:
        label=row['label_class'].strip().lower();annotation=row['annotation_ok'].strip().lower()
        if label not in ('ripe','unripe','unsure'):raise ValueError(f"Undecided label_class at {row['review_id']}")
        if annotation not in ('yes','no'):raise ValueError(f"Undecided annotation_ok at {row['review_id']}")
        record=by_id[key[row['review_id']]]
        target=CLASS_NAME[str(record['class_id'])]
        record.update(review_id=row['review_id'],reviewed_label=label,annotation_ok=annotation,
                      notes=row.get('notes',''),reviewer=reviewer,reviewed_at=stamp,
                      status='accepted' if (label==target and annotation=='yes') else 'rejected')
    write_json(out/'records.json',records)
    counts=collections.Counter(r['status'] for r in records)
    return dict(reviewer=reviewer,counts=dict(counts),n=len(records))


def select(dataset,records,per_parent_class=1):
    """S=N selection: the same accepted count for every parent and class, or a stated shortfall."""
    accepted=[r for r in records if r.get('status')=='accepted']
    outside=[r['parent_id'] for r in accepted if r['parent_id'] not in set(dataset.scope_ids)]
    if outside:raise ValueError(f'Accepted candidates outside scope: {sorted(set(outside))}')
    grouped=collections.defaultdict(list)
    for record in accepted:grouped[record['parent_id']].append(record)
    selected=[];shortfall={}
    for pid in dataset.scope_ids:
        pool=sorted(grouped[pid],key=lambda r:(digest([r['candidate_id']]),r['candidate_id']))
        selected.extend(pool[:per_parent_class])
        if len(pool)<per_parent_class:
            shortfall[f"{pid}/{dataset.by_id[pid]['class_id']}"]=per_parent_class-len(pool)
    per_class=collections.Counter(r['class_id'] for r in selected)
    return dict(selected=selected,shortfall=shortfall,met=not shortfall,
                per_class=dict(per_class),n=len(selected),per_parent_class=per_parent_class)
