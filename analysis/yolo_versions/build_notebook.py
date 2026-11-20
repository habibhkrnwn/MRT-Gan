"""Write notebook 21 (YOLO generation comparison) and its kernel metadata.

    python build_notebook.py --out ../../../dist/kaggle_yolo_versions --seeds 0 1 2 3 4

The helper scripts are embedded verbatim, with their sha256 printed in the notebook, so the
kernel needs no extra dataset beyond the code package and the frozen regrouped v4 dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
VERSIONS = ['yolov8n', 'yolov5nu', 'yolov9t', 'yolov10n', 'yolo11n', 'yolo12n', 'yolo26n']


def cell(kind, text):
    lines = text.strip('\n').splitlines(keepends=True)
    base = dict(cell_type=kind, metadata={}, source=lines)
    if kind == 'code':
        base.update(execution_count=None, outputs=[])
    return base


def build(seeds, workers_per_gpu, budget_minutes):
    helpers = {name: (HERE / name).read_text() for name in
               ('train_detector_versions.py', 'preletterbox_arm.py')}
    digests = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in helpers.items()}
    cells = [cell('markdown', f"""
# 21 — Perbandingan generasi YOLO (nano) pada protokol detektor terkunci

Arm `real_only` (301 foto train, DSC_0093 dikeluarkan; 87 validation). Tujuh generasi skala
nano/tiny: {', '.join(VERSIONS)}. Trainer `train_detector.py` (sha256 5b273050…) tidak diubah:
3000 update, batch 8, AdamW lr 1e-4, cosine, flip deterministik, letterbox 640, validasi tiap
250 update, checkpoint terbaik menurut mAP50:95 validation. Hanya jaringan (config + bobot COCO)
yang berbeda. Seed {seeds}. **Hanya validation; test tidak dibaca.**

Citra train di-letterbox sekali ke PNG 640 (diverifikasi identik piksel dan kotak dengan jalur
asli) agar pemuatan data tidak menjadi leher botol. {workers_per_gpu} proses per GPU.
Batas waktu peluncuran job baru: {budget_minutes} menit sejak sel scheduler mulai.
"""),
        cell('code', r"""
from pathlib import Path
import os, sys, json, shutil, zipfile, subprocess, hashlib, time
WORK = Path('/kaggle/working/yolo_versions_v1')
INPUT = Path('/kaggle/input')
WORK.mkdir(parents=True, exist_ok=False)
archives = list(INPUT.rglob('texture_study_code.zip'))
unpacked = [p.parent for p in INPUT.rglob('package_manifest.json')
            if (p.parent/'code/cyclegan/tools/texture_study.py').exists() and not (p.parent/'study').exists()]
if len(archives) + len(unpacked) != 1:
    raise ValueError(f'Attach exactly one texture study code dataset; zip={archives}, unpacked={unpacked}')
if archives:
    with zipfile.ZipFile(archives[0]) as z:
        for item in z.infolist():
            if not (WORK/item.filename).resolve().is_relative_to(WORK.resolve()): raise ValueError('unsafe zip')
        z.extractall(WORK)
else:
    shutil.copytree(unpacked[0], WORK, dirs_exist_ok=True)
    for p in [WORK, *WORK.rglob('*')]: p.chmod(p.stat().st_mode | 0o200)
manifest = json.loads((WORK/'package_manifest.json').read_text())
bad = [k for k,v in manifest.items() if hashlib.sha256((WORK/k).read_bytes()).hexdigest() != v]
if bad: raise ValueError(f'Package files differ from package_manifest.json: {bad[:5]}')
print('code package verified:', len(manifest), 'files')
HERE = WORK/'code/cyclegan'
TRAINER = WORK/'code/14_Benchmark_Kaggle/tools/train_detector.py'
def unique(marker):
    hits = sorted({p.parent for p in INPUT.rglob(marker)})
    if len(hits) != 1: raise ValueError(f'Expected one {marker}; found {hits}')
    return hits[0]
DATASET = unique('manifest.csv')
digest = hashlib.sha256((DATASET/'manifest.csv').read_bytes()).hexdigest()
if digest != '185e14221e8882f7bcabb5fbb455d6a64aa09378f07234e1ccd4f855531df6ba':
    raise ValueError(f'dataset manifest {digest} is not regrouped v4')
print('dataset', DATASET, digest[:16])
"""),
        cell('code', r"""
import importlib.util, importlib.metadata as meta
if importlib.util.find_spec('ultralytics') is None or meta.version('ultralytics') != '8.4.163':
    subprocess.run([sys.executable,'-m','pip','install','-q','ultralytics==8.4.163'],check=True)
import torch
for package in ('torch','torchvision','ultralytics'): print(package, meta.version(package))
if torch.cuda.device_count() < 1: raise RuntimeError('GPU required')
N_GPU = torch.cuda.device_count()
print('GPUs', N_GPU, [torch.cuda.get_device_name(i) for i in range(N_GPU)])
"""),
        cell('code', 'HELPERS = ' + json.dumps(helpers, indent=0) + '\nDIGESTS = ' + json.dumps(digests) + r"""
TOOLS = WORK/'yolo_tools'; TOOLS.mkdir()
for name, text in HELPERS.items():
    (TOOLS/name).write_text(text)
    assert hashlib.sha256(text.encode()).hexdigest() == DIGESTS[name]
    print(name, DIGESTS[name][:16])
"""),
        cell('code', r"""
# real_only arm (same builder and exclusion as balancing experiment C), then the 640 copy.
ENV = {**os.environ, 'PYTHONPATH': str(HERE), 'MELON_DATASET': str(DATASET), 'PYTHONUNBUFFERED': '1'}
subprocess.run([sys.executable, str(HERE/'tools/build_balance_arms.py'), '--real-only', '--out', str(WORK/'arms')],
               cwd=HERE, env=ENV, check=True)
subprocess.run([sys.executable, str(TOOLS/'preletterbox_arm.py'), '--trainer', str(TRAINER),
                '--arm', str(WORK/'arms/real_only'), '--out', str(WORK/'arms640/real_only'), '--verify'],
               cwd=TOOLS, env=ENV, check=True)
arm_manifest = json.loads((WORK/'arms640/real_only/arm_manifest.json').read_text())
print(arm_manifest['counts'])
assert arm_manifest['counts']['total_per_class'] == {'0': 217, '1': 84} and arm_manifest['counts']['validation'] == 87
"""),
        cell('code', f"VERSIONS = {VERSIONS!r}\nSEEDS = {list(seeds)!r}\nWORKERS_PER_GPU = {workers_per_gpu}\nBUDGET_MINUTES = {budget_minutes}\n" + r"""
# COCO weights fetched once; every job runs with this folder as working directory.
WEIGHTS = WORK/'weights'; WEIGHTS.mkdir()
from ultralytics import YOLO
for name in VERSIONS:
    spec = {'yolov5nu':'yolov5nu.pt','yolov8n':'yolov8n.pt','yolov9t':'yolov9t.pt','yolov10n':'yolov10n.pt',
            'yolo11n':'yolo11n.pt','yolo12n':'yolo12n.pt','yolo26n':'yolo26n.pt'}[name]
    cwd = os.getcwd(); os.chdir(WEIGHTS)
    try: YOLO(spec)
    finally: os.chdir(cwd)
print(sorted(p.name for p in WEIGHTS.glob('*.pt')))
# One private arm copy per worker slot, so concurrent validations never share a label cache.
SLOTS = [(g, w) for g in range(N_GPU) for w in range(WORKERS_PER_GPU)]
for g, w in SLOTS:
    dst = WORK/f'arm_slot{g}{w}/real_only'
    shutil.copytree(WORK/'arms640/real_only', dst)
    (dst/'data.yaml').write_text((dst/'data.yaml').read_text().replace(
        str((WORK/'arms640/real_only').resolve()), str(dst.resolve())))
print('slots', SLOTS)
"""),
        cell('code', r"""
# Seed-major order: if the budget stops new launches, every finished seed covers all generations.
import collections
jobs = collections.deque((name, seed) for seed in SEEDS for name in VERSIONS)
RUNS = WORK/'runs'; LOGS = WORK/'logs'; RUNS.mkdir(); LOGS.mkdir()
running, finished, failed, skipped = {}, [], [], []
started = time.time()
def launch(slot, name, seed):
    g, w = slot
    argv = [sys.executable, str(TOOLS/'train_detector_versions.py'), '--trainer', str(TRAINER),
            '--arm', str(WORK/f'arm_slot{g}{w}/real_only'), '--seed', str(seed), '--updates', '3000',
            '--eval-every', '250', '--architecture', name, '--out', str(RUNS/name)]
    env = {**ENV, 'CUDA_VISIBLE_DEVICES': str(g), 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    log = open(LOGS/f'{name}__seed{seed}.log', 'w')
    proc = subprocess.Popen(argv, cwd=WEIGHTS, env=env, stdout=log, stderr=subprocess.STDOUT)
    return dict(proc=proc, name=name, seed=seed, t0=time.time(), log=log)
free = list(SLOTS)
while jobs or running:
    while free and jobs:
        if (time.time() - started) / 60 > BUDGET_MINUTES:
            skipped.extend(jobs); jobs.clear(); break
        slot = free.pop(0); name, seed = jobs.popleft()
        running[slot] = launch(slot, name, seed)
    time.sleep(10)
    for slot, job in list(running.items()):
        code = job['proc'].poll()
        if code is None: continue
        job['log'].close(); minutes = round((time.time() - job['t0']) / 60, 1)
        (finished if code == 0 else failed).append(dict(name=job['name'], seed=job['seed'], gpu=slot[0],
                                                        code=code, minutes=minutes))
        print(time.strftime('%H:%M:%S'), job['name'], job['seed'], 'code', code, minutes, 'min', flush=True)
        del running[slot]; free.append(slot)
wall = round((time.time() - started) / 60, 1)
print('wall minutes', wall, 'finished', len(finished), 'failed', failed, 'skipped', list(skipped))
for f in failed: print((LOGS/f"{f['name']}__seed{f['seed']}.log").read_text()[-2000:])
"""),
        cell('code', r"""
# Summary per generation (selected checkpoint and final state) and paired differences vs YOLOv8n.
import statistics
def read(name, seed):
    path = RUNS/name/f'real_only__seed{seed}'/'run_summary.json'
    return json.loads(path.read_text()) if path.exists() else None
table = {}
for name in VERSIONS:
    runs = {s: read(name, s) for s in SEEDS}
    runs = {s: r for s, r in runs.items() if r}
    if not runs: continue
    sel = {s: r['selected']['validation'] for s, r in runs.items()}
    fin = {s: r['validation'] for s, r in runs.items()}
    def stats(d, key):
        values = [key(v) for v in d.values()]
        return dict(mean=statistics.fmean(values), sd=statistics.pstdev(values), per_seed=dict(zip(d, values)))
    table[name] = dict(seeds=sorted(runs),
        selected=dict(map50_95=stats(sel, lambda v: v['map50_95']), map50=stats(sel, lambda v: v['map50']),
                      ap_unripe=stats(sel, lambda v: v['per_class']['unripe']['ap50_95']),
                      recall_unripe=stats(sel, lambda v: v['recall'][1]),
                      update=dict(per_seed={s: r['selected']['update'] for s, r in runs.items()})),
        final=dict(map50_95=stats(fin, lambda v: v['map50_95']),
                   ap_unripe=stats(fin, lambda v: v['per_class']['unripe']['ap50_95'])),
        training_seconds=stats(runs, lambda r: r['training_seconds']))
ref = table.get('yolov8n')
for name, row in table.items():
    if not ref or name == 'yolov8n': continue
    common = [s for s in row['seeds'] if s in ref['seeds']]
    diffs = [row['selected']['map50_95']['per_seed'][s] - ref['selected']['map50_95']['per_seed'][s] for s in common]
    row['paired_vs_yolov8n'] = dict(seeds=common, mean_diff=statistics.fmean(diffs) if diffs else None,
                                    wins=sum(d > 0 for d in diffs), diffs=diffs)
OUT = WORK/'result'; OUT.mkdir()
report = dict(versions=VERSIONS, seeds=SEEDS, arm='real_only', updates=3000, eval_every=250, batch=8, lr=1e-4,
              optimizer='AdamW (identical for all; not each generation\'s default)', ultralytics=meta.version('ultralytics'),
              workers_per_gpu=WORKERS_PER_GPU, gpus=[torch.cuda.get_device_name(i) for i in range(N_GPU)],
              finished=finished, failed=failed, skipped=list(skipped), wall_minutes=wall, table=table,
              helper_sha256=DIGESTS, split='valid', test_used=False)
(OUT/'results.json').write_text(json.dumps(report, indent=2) + '\n')
for name, row in table.items():
    s = row['selected']
    print(f"{name:9s} n={len(row['seeds'])} mAP50:95 {s['map50_95']['mean']:.4f}±{s['map50_95']['sd']:.4f} "
          f"AP_unripe {s['ap_unripe']['mean']:.4f} R_unripe {s['recall_unripe']['mean']:.3f} "
          f"train {row['training_seconds']['mean']/60:.1f} min", row.get('paired_vs_yolov8n', {}).get('wins'))
# Keep run folders (summaries, histories, predictions, weights); drop bulky arm copies and code.
shutil.copytree(RUNS, OUT/'runs'); shutil.copytree(LOGS, OUT/'logs')
for p in [WORK/'arms', WORK/'arms640', WORK/'code', WORK/'weights', RUNS, LOGS, *WORK.glob('arm_slot*')]:
    shutil.rmtree(p, ignore_errors=True)
print(sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) / 1e6, 'MB in result')
"""),
    ]
    return dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                                           language_info=dict(name='python')), nbformat=4, nbformat_minor=5)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument('--workers-per-gpu', type=int, default=2)
    parser.add_argument('--budget-minutes', type=int, default=300)
    parser.add_argument('--kernel', default='habibhammamkurniawan/21-melon-yolo-version-comparison')
    args = parser.parse_args(argv)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    name = '21_yolo_version_comparison.ipynb'
    (out / name).write_text(json.dumps(build(args.seeds, args.workers_per_gpu, args.budget_minutes), indent=1) + '\n')
    (out / 'kernel-metadata.json').write_text(json.dumps(dict(
        id=args.kernel, title='21 Melon YOLO version comparison', code_file=name, language='python',
        kernel_type='notebook', is_private=True, enable_gpu=True, enable_tpu=False, enable_internet=True,
        dataset_sources=['habibhammamkurniawan/melon-texture-study-code',
                         'habibhammamkurniawan/melon-dataset-regrouped-v4'],
        competition_sources=[], kernel_sources=[], model_sources=[]), indent=2) + '\n')
    print(out / name)


if __name__ == '__main__':
    main()
