"""Prepare and calibrate the SAME train pool for G0-clean and G0X-v2 (CPU)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melon_cyclegan import config
from melon_cyclegan.training_data import curate, verify
from tools.calibrate_color_margin import calibrate


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--policy', default=str(config.ROOT / 'config/g0x_data_policy.json'))
    args = parser.parse_args(argv)
    policy = json.loads(Path(args.policy).read_text())
    root = Path(args.out)
    manifest = verify(root, policy['name']) if root.exists() else curate(args.source, root, policy)
    if manifest.get('curation_policy') != policy:
        raise ValueError('existing pool uses a different curation policy')
    calibration = calibrate(root)
    (root / 'color_margin.json').write_text(json.dumps(calibration, indent=2) + '\n')
    print(json.dumps(dict(counts=manifest['counts'], excluded=manifest['excluded'],
                         crops_digest=manifest['crops_digest'], margin=calibration['margin']), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
