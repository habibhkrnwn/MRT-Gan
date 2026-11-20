"""Write a copy of an arm whose TRAIN images are already letterboxed to 640 (lossless PNG).

    python preletterbox_arm.py --trainer <train_detector.py> --arm <arm> --out <new arm> --verify

The locked trainer decodes every 2048x1366 JPEG and letterboxes it again at every update,
which makes a 3000-update run CPU-bound. Letterboxing a 640x640 canvas a second time is the
identity in the trainer (PIL returns a copy when the size is unchanged, and the offsets are 0),
so the tensors the network sees are unchanged. `--verify` checks this for every training image,
with and without the horizontal flip: pixels must be equal and box coordinates equal within
1e-9. Validation images are copied untouched, because Ultralytics validation reads them with
its own loader. The test split is not read.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_detector_versions import load_trainer  # noqa: E402


def build(trainer, arm, out):
    arm, out = Path(arm), Path(out)
    if out.exists():
        raise SystemExit(f'{out} exists; use a new directory')
    for split in ('train', 'val'):
        (out / 'images' / split).mkdir(parents=True)
        (out / 'labels' / split).mkdir(parents=True)
    source = trainer.ArmDataset(arm, 'train')
    for index in range(len(source)):
        stem, image, boxes = source.load(index, flip=False)
        image.save(out / 'images/train' / f'{stem}.png')
        (out / 'labels/train' / f'{stem}.txt').write_text(
            ''.join(f'{c} {x!r} {y!r} {w!r} {h!r}\n' for c, x, y, w, h in boxes))
    for path in sorted((arm / 'images/val').glob('*')):
        shutil.copyfile(path, out / 'images/val' / path.name)
    for path in sorted((arm / 'labels/val').glob('*')):
        shutil.copyfile(path, out / 'labels/val' / path.name)
    manifest = json.loads((arm / 'arm_manifest.json').read_text())
    manifest['preletterboxed_from'] = str(arm.resolve())
    (out / 'arm_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (out / 'data.yaml').write_text((arm / 'data.yaml').read_text().replace(
        f'path: {arm.resolve()}', f'path: {out.resolve()}'))
    return len(source)


def verify(trainer, arm, out):
    a, b = trainer.ArmDataset(Path(arm), 'train'), trainer.ArmDataset(Path(out), 'train')
    if [p.stem for p in a.items] != [p.stem for p in b.items]:
        raise SystemExit('train item order differs; batch indices would not match')
    worst = 0.0
    for index in range(len(a)):
        for flip in (False, True):
            sa, ia, ba = a.load(index, flip)
            sb, ib, bb = b.load(index, flip)
            if sa != sb or not np.array_equal(np.asarray(ia), np.asarray(ib)):
                raise SystemExit(f'pixels differ for {sa} flip={flip}')
            if len(ba) != len(bb) or any(x[0] != y[0] for x, y in zip(ba, bb)):
                raise SystemExit(f'boxes differ for {sa}')
            worst = max([worst] + [abs(u - v) for x, y in zip(ba, bb) for u, v in zip(x[1:], y[1:])])
    if worst > 1e-9:
        raise SystemExit(f'box coordinates differ by {worst}')
    return dict(images=len(a), max_box_difference=worst)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainer', required=True)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args(argv)
    trainer = load_trainer(args.trainer)
    print('written', build(trainer, args.arm, args.out))
    if args.verify:
        print('verified', verify(trainer, args.arm, args.out))


if __name__ == '__main__':
    main()
