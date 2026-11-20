"""Train one rung of the CycleGAN ladder from a config file.

    python tools/train_cyclegan.py --config config/g4_enhanced.json \
        --dataroot runs/crops256 --checkpoints-dir runs/checkpoints

The upstream training script is executed unchanged in its own directory; this
wrapper only translates the config into its command line, pins the colour
centroids and writes a small identity file next to the checkpoints so a run can
be traced back to its data and its code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import shlex
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from melon_cyclegan import config as cfg  # noqa: E402
from melon_cyclegan.training_data import verify, sha  # noqa: E402

FLAGS = (
    "model", "dataset_mode", "netG", "netD", "norm", "gan_mode", "preprocess",
    "load_size", "crop_size", "batch_size", "lr", "beta1", "lr_policy",
    "n_epochs", "n_epochs_decay", "pool_size", "lambda_A", "lambda_B",
    "lambda_identity", "lambda_ssim", "lambda_background", "lambda_color",
    "lambda_edge", "lambda_color_margin", "color_margin", "composite_mask",
    "mask_dilate", "mask_feather", "diffaug", "amp", "ema_decay", "num_threads", "save_epoch_freq",
    "lambda_texture", "texture_size", "texture_stride", "texture_count", "texture_coverage",
    "texture_gray", "texture_ndf", "texture_seed", "controlled_init_seed",
)
# Flags handled by melon_cyclegan.train_loop, not by the upstream option parser.
LOCAL_FLAGS = ("seed", "sample_epoch_freq", "monitor_samples")


def resolve_settings(settings, dataroot, centroids):
    """Bind v2 settings to the verified pool and its train-only calibration."""
    settings = dict(settings)
    if settings.get('training_policy'):
        manifest = verify(dataroot, settings['training_policy'])
        if manifest.get('dataset_manifest') != cfg.EXPECTED_MANIFEST:
            raise ValueError('dataset manifest differs from the frozen dataset')
        if sha(centroids) != manifest['centroids_sha256']:
            raise ValueError('requested centroids do not match the training pool')
        expected = settings.get('expected_crops_digest')
        if expected and expected != manifest['crops_digest']:
            raise ValueError('crop/mask bytes differ from the locked v2 pool')
    else:
        manifest = None
    if settings.get('color_margin_file'):
        if manifest is None:
            raise ValueError('automatic margin requires a verified curated pool')
        path = Path(dataroot) / settings['color_margin_file']
        report = json.loads(path.read_text())
        if report.get('crops_digest') != manifest['crops_digest'] or report.get('centroids_sha256') != sha(centroids):
            raise ValueError('colour margin calibration is stale or belongs to another pool')
        if report.get('percentile') != settings.get('color_margin_percentile', 10.0):
            raise ValueError('colour margin percentile differs from the registered rule')
        margin = float(report['margin'])
        if not math.isfinite(margin) or margin <= 0:
            raise ValueError('nonpositive/invalid class margin; inspect overlap before training')
        settings['color_margin'] = margin
        settings['color_margin_sha256'] = sha(path)
    return settings


def build_argv(settings, dataroot, checkpoints_dir, name, centroids=None, extra=()):
    argv = ["--dataroot", str(dataroot), "--name", name, "--checkpoints_dir", str(checkpoints_dir)]
    for key in FLAGS:
        if (key.startswith('texture_') or key == 'lambda_texture') and settings.get('model') != 'melon_texture':
            continue
        if key in settings:
            argv += [f"--{key}", str(settings[key])]
    colour_on = (float(settings.get("lambda_color", 0)) > 0
                 or float(settings.get("lambda_color_margin", 0)) > 0)
    if settings.get("model") in ("melon_cyclegan", "melon_texture") and colour_on:
        if not centroids:
            raise SystemExit("--color-centroids is required for this config")
        argv += ["--color_centroids", str(centroids)]
    for key in LOCAL_FLAGS:
        if key in settings:
            argv += [f"--{key}", str(settings[key])]
    argv += list(extra)
    return argv


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataroot", required=True, help="crop directory built by tools/build_crops.py")
    parser.add_argument("--checkpoints-dir", default="runs/checkpoints")
    parser.add_argument("--name", default=None)
    parser.add_argument("--color-centroids", default=None)
    parser.add_argument("--dry-run", action="store_true", help="print the command line and stop")
    parser.add_argument("extra", nargs="*", help="extra flags passed straight to the upstream trainer")
    args = parser.parse_args(argv)

    settings = json.loads(Path(args.config).read_text())
    name = args.name or settings["experiment"]
    dataroot = Path(args.dataroot).resolve()
    centroids = args.color_centroids or (dataroot / "color_centroids.json")
    checkpoints = Path(args.checkpoints_dir).resolve()
    requested_settings = settings
    settings = resolve_settings(settings, dataroot, centroids)

    command = ([sys.executable, "-m", "melon_cyclegan.train_loop"]
               + build_argv(settings, dataroot, checkpoints, name, centroids, args.extra))
    if args.dry_run:
        print(shlex.join(command))
        return 0

    if (checkpoints / name).exists():
        raise ValueError(f'run directory already exists: {checkpoints / name}; use a new run name')
    checkpoints.mkdir(parents=True, exist_ok=True)

    manifest = dataroot / "crops_manifest.json"
    identity = dict(
        version=cfg.VERSION, experiment=name, rung=settings.get("rung"),
        comparison_id=settings.get("comparison_id"), config=settings,
        requested_config=requested_settings,
        dataroot=str(dataroot), started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        crops_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.exists() else None,
        crops_digest=json.loads(manifest.read_text()).get("crops_digest") if manifest.exists() else None,
        dataset_manifest=cfg.EXPECTED_MANIFEST, test_used=False,
    )
    (checkpoints / name).mkdir(parents=True, exist_ok=True)
    (checkpoints / name / "melon_identity.json").write_text(json.dumps(identity, indent=2) + "\n")

    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(cfg.ROOT), environment.get("PYTHONPATH", "")]).strip(os.pathsep)
    started = time.time()
    result = subprocess.run(command, cwd=str(cfg.ROOT), env=environment)
    identity["seconds"] = round(time.time() - started, 1)
    identity["returncode"] = result.returncode
    (checkpoints / name / "melon_identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
