"""CPU preflight of notebook 21: run its own code cells locally with 4 updates and 2 seeds.

    python preflight_local.py --notebook ../../../dist/kaggle_yolo_versions/21_yolo_version_comparison.ipynb \
        --code-zip ../../../dist/kaggle_texture_study_code/texture_study_code.zip \
        --dataset ../../../../../10_Migrasi_Kaggle/dataset_regrouped --work <empty scratch dir>

Only paths, the update budget, the seed list and the GPU check are substituted; every other line
is the notebook's. A pass means the pipeline (package check, arm build, 640 copy + verify, weight
download, scheduler, summary, cleanup) runs end to end. It says nothing about GPU speed or results.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--notebook', required=True)
    parser.add_argument('--code-zip', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--work', required=True)
    args = parser.parse_args(argv)
    work = Path(args.work).resolve()
    if work.exists() and any(work.iterdir()):
        raise SystemExit(f'{work} is not empty')
    (work / 'input/code').mkdir(parents=True)
    (work / 'working').mkdir()
    shutil.copy(args.code_zip, work / 'input/code')
    shutil.copytree(Path(args.dataset).resolve(), work / 'input/dataset')  # rglob does not follow symlinks
    nb = json.loads(Path(args.notebook).read_text())
    source = '\n\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
    replace = {
        '/kaggle/working': str(work / 'working'), '/kaggle/input': str(work / 'input'),
        "'3000'": "'4'", "'250'": "'2'",
        "if torch.cuda.device_count() < 1: raise RuntimeError('GPU required')\nN_GPU = torch.cuda.device_count()": 'N_GPU = 1',
        '[torch.cuda.get_device_name(i) for i in range(N_GPU)]': "['cpu']",
        'SEEDS = [0, 1, 2, 3, 4]': 'SEEDS = [0, 1]',
        "subprocess.run([sys.executable,'-m','pip','install','-q','ultralytics==8.4.163'],check=True)": 'pass',
    }
    for old, new in replace.items():
        if old not in source:
            raise SystemExit(f'notebook no longer contains: {old!r}')
        source = source.replace(old, new)
    (work / 'run.py').write_text(source)
    code = subprocess.run([sys.executable, str(work / 'run.py')], cwd=work).returncode
    result = work / 'working/yolo_versions_v1/result/results.json'
    if code or not result.exists():
        raise SystemExit(f'preflight failed (exit {code})')
    report = json.loads(result.read_text())
    expected = len(report['versions']) * len(report['seeds'])
    if report['failed'] or len(report['finished']) != expected or len(report['table']) != len(report['versions']):
        raise SystemExit(f"preflight incomplete: finished {len(report['finished'])}/{expected}, failed {report['failed']}")
    print('PREFLIGHT PASS', len(report['finished']), 'jobs;', 'test_used', report['test_used'])


if __name__ == '__main__':
    main()
