"""Run the locked detector trainer with another YOLO generation.

    python train_detector_versions.py --trainer <14_Benchmark_Kaggle/tools/train_detector.py> \
        --arm <arm> --seed 0 --updates 3000 --eval-every 250 --architecture yolo11n --out <out>

`train_detector.py` (sha256 5b273050...) is the trainer behind every published detector number
and is not edited. This wrapper only adds entries to its `ARCHITECTURES` table and then calls its
`main`, so the loop, batches, flip rule, AdamW, cosine schedule, letterbox, update budget,
periodic validation and checkpoint rule are byte-for-byte the same for every generation.
Only the network (config + COCO-pretrained weights) differs. The test split is never read.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

# Nano / tiny scale of each generation shipped with Ultralytics 8.4. The 'u' variant of YOLOv5
# is the anchor-free Ultralytics port; the original anchor-based YOLOv5 is not trainable here.
VERSIONS = {
    'yolov5nu': dict(cfg='yolov5nu.yaml', weights='yolov5nu.pt'),
    'yolov8n': dict(cfg='yolov8n.yaml', weights='yolov8n.pt'),
    'yolov9t': dict(cfg='yolov9t.yaml', weights='yolov9t.pt'),
    'yolov10n': dict(cfg='yolov10n.yaml', weights='yolov10n.pt'),
    'yolo11n': dict(cfg='yolo11n.yaml', weights='yolo11n.pt'),
    'yolo12n': dict(cfg='yolo12n.yaml', weights='yolo12n.pt'),
    'yolo26n': dict(cfg='yolo26n.yaml', weights='yolo26n.pt'),
}
LOCKED_TRAINER_SHA256 = '5b27305000a529a0b0c375be26aee83aec3ffb408ca9f0ab811d2f1329b5d61b'


def load_trainer(path):
    path = Path(path).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != LOCKED_TRAINER_SHA256:
        raise SystemExit(f'{path} is not the locked trainer (sha256 {digest})')
    spec = importlib.util.spec_from_file_location('train_detector', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['train_detector'] = module
    spec.loader.exec_module(module)
    for name, entry in VERSIONS.items():
        module.ARCHITECTURES.setdefault(name, entry)
    return module


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--trainer' not in argv:
        raise SystemExit('--trainer is required')
    i = argv.index('--trainer')
    trainer = load_trainer(argv[i + 1])
    del argv[i:i + 2]
    # `choices` of --architecture was fixed when argparse was built inside main(); rebuild it
    # by calling main after the table is extended (choices=sorted(ARCHITECTURES) is evaluated
    # at call time, so the new names are accepted).
    trainer.main(argv)


if __name__ == '__main__':
    main()
