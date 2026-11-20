"""Protocol amendment: give subset generators the same number of updates as the p100 generators.

    python tools/amend_subset_epochs.py --study <study> --fractions 25 50

Decided by the user on 2026-09-28 after kernel 17: with a fixed 20 epochs, one CycleGAN epoch is
max(|unripe|, |ripe|) crops, so p025 received about a quarter of the p100 updates and produced much
stronger checkerboard. That confounds "less labelled data" with "shorter training". This tool:

- counts the images per epoch of every scope (the same length rule as `MelonRoiData`);
- sets n_epochs = n_epochs_decay = round(p100 updates / subset epoch length / 2) in the subset
  configs of G0 and the selected generator (untrained configs only);
- records the amendment and the new code hashes in study.json, because the epoch check in
  `texture_study.py` now reads the config instead of a hard-coded 20.

It refuses scopes that already hold checkpoints. p100 is not changed.
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melon_cyclegan.texture_study import read, write, sha, verify_scope, ROOT  # noqa: E402
from melon_cyclegan.dataset_impl import MelonRoiData  # noqa: E402

CHANGED_CODE = ('melon_cyclegan/texture_study.py', 'tools/amend_subset_epochs.py')


def epoch_length(scope):
    return len(MelonRoiData(scope/'crops', no_flip=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', required=True)
    ap.add_argument('--fractions', type=int, nargs='+', choices=[25, 50], required=True)
    args = ap.parse_args()
    study = Path(args.study)
    protocol = read(study/'study.json')
    winner = read(study/'selection.json')['winner']
    p100, _ = verify_scope(study, 100)
    reference = read(p100/'configs'/f'{winner}.json')
    ref_epochs = reference['n_epochs'] + reference['n_epochs_decay']
    ref_updates = epoch_length(p100) * ref_epochs
    record = dict(at=datetime.now(timezone.utc).isoformat(), decided_by='user 2026-09-28',
                  reason='match generator updates to p100; fixed 20 epochs gave subsets fewer updates',
                  reference=dict(fraction=100, epoch_length=epoch_length(p100), epochs=ref_epochs, updates=ref_updates),
                  subsets={})
    for fraction in args.fractions:
        scope, _ = verify_scope(study, fraction)
        if (scope/'checkpoints').exists() and any((scope/'checkpoints').iterdir()):
            raise ValueError(f'p{fraction:03d} already has checkpoints; amend before training only')
        length = epoch_length(scope)
        half = max(1, round(ref_updates / length / 2))
        for arm in ('G0', winner):
            path = scope/'configs'/f'{arm}.json'
            config = read(path)
            config.update(n_epochs=half, n_epochs_decay=half,
                          note=f'{2 * half}-epoch subset run, updates matched to p100 ({ref_updates}); EMA final primary')
            write(path, config)
        record['subsets'][str(fraction)] = dict(epoch_length=length, epochs=2 * half, updates=length * 2 * half,
                                                arms=['G0', winner])
    for name in CHANGED_CODE:
        protocol['code_sha256'][name] = sha(ROOT/name)
    protocol.setdefault('amendments', []).append(record)
    write(study/'study.json', protocol)
    print(record)


if __name__ == '__main__':
    main()
