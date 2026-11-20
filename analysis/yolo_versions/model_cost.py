"""Parameters, GFLOPs and CPU latency of each YOLO generation with the two-class melon head.

    python model_cost.py --trainer <train_detector.py> --out model_cost.json \
        [--weights-root <kaggle result dir>] [--images <validation image folder>]

Without `--weights-root` only parameters and GFLOPs are reported (architecture property).
With it, latency is measured on the trained seed-0 selected checkpoint of every generation,
through the same `YOLO.predict` path as the harvest pipeline (letterbox 640, conf 0.001,
NMS or end-to-end head as the model defines), on this machine's CPU, batch 1. The first
five images are warm-up and are not timed. Only validation images are used.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_detector_versions import VERSIONS, load_trainer  # noqa: E402


def gflops(net, size=640):
    from ultralytics.utils.torch_utils import get_flops
    return float(get_flops(net, size))


def latency(trainer, name, weights, images, warmup=5):
    import torch
    from ultralytics import YOLO
    state = torch.load(weights, map_location='cpu', weights_only=False)
    net = trainer.build_model(pretrained=False, architecture=name)
    net.load_state_dict(state['state_dict'])
    model = YOLO(VERSIONS[name]['cfg'])
    model.model = net.eval()
    times = []
    for i, path in enumerate(images):
        start = time.perf_counter()
        model.predict(source=str(path), imgsz=640, conf=0.001, max_det=100, verbose=False, device='cpu')
        if i >= warmup:
            times.append(time.perf_counter() - start)
    return dict(median_s=statistics.median(times), mean_s=statistics.fmean(times), images=len(times))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainer', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--weights-root', default=None)
    parser.add_argument('--images', default=None)
    parser.add_argument('--versions', nargs='*', default=list(VERSIONS))
    args = parser.parse_args(argv)
    import torch
    trainer = load_trainer(args.trainer)
    rows = {}
    for name in args.versions:
        net = trainer.build_model(pretrained=False, architecture=name).eval()
        row = dict(params_m=sum(p.numel() for p in net.parameters()) / 1e6, gflops_640=gflops(net))
        if args.weights_root:
            hits = sorted(Path(args.weights_root).rglob(f'{name}/*__seed0/weights_selected.pt')) or \
                sorted(Path(args.weights_root).rglob(f'{name}/*__seed0/weights.pt'))
            if not hits:
                raise SystemExit(f'no seed-0 weights for {name} under {args.weights_root}')
            images = sorted(Path(args.images).glob('*'))
            row.update(weights=str(hits[0]), **latency(trainer, name, hits[0], images))
        rows[name] = row
        print(name, row, flush=True)
    Path(args.out).write_text(json.dumps(dict(
        rows=rows, cpu=platform.processor() or platform.machine(), torch=torch.__version__,
        threads=torch.get_num_threads(), note='two-class head; validation images only'), indent=2) + '\n')


if __name__ == '__main__':
    main()
