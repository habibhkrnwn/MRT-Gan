"""Translate the frozen scope parents and write a bundle the benchmark tools accept.

For every scope parent the generator of the opposite direction is applied, so the
maturity class of the candidate is the opposite of its parent:

    parent ripe (0)   -> G_B (B->A) -> candidate unripe (1)
    parent unripe (1) -> G_A (A->B) -> candidate ripe (0)

The parent's reviewed mask is reused, both for compositing and as the annotation
proposal, so the candidate box is exactly the parent box. Candidates leave this
module as `needs_review`; only `pipeline.ingest_review` with a named reviewer can
accept one, and a reviewer who does not read the candidate as the target class
rejects it. That is the label-preservation test of the hypothesis, not a
formality.

The bundle layout matches `14_Benchmark_Kaggle/src/pipeline.py`, so
`tools/build_detector_arms.py` consumes it without modification.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import config
from .config import CLASS_NAMES, RIPE, UNRIPE

SLOT_TRANSFORMS = ("identity", "hflip")


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_generators(checkpoints_dir, name, epoch="latest", netG=None, ngf=64, norm="instance", device="cpu",
                    weights="raw"):
    """Load G_A and G_B of a finished run as evaluation-mode modules.

    `weights="ema"` loads the exponential-moving-average shadows saved as
    `<epoch>_net_G_A_ema.pth` instead of the raw generator weights.
    """
    if weights not in ("raw", "ema"):
        raise ValueError("weights must be 'raw' or 'ema'")
    suffix = "_ema" if weights == "ema" else ""
    from . import networks

    root = Path(checkpoints_dir) / name
    identity_path = root / "melon_identity.json"
    settings = {}
    identity = {}
    if identity_path.exists():
        identity = json.loads(identity_path.read_text())
        settings = identity.get("config", {})
    netG = netG or settings.get("netG") or "resnet_6blocks"
    norm = settings.get("norm", norm)
    ngf = int(settings.get("ngf", ngf))
    out = {}
    for key in ("G_A", "G_B"):
        path = root / f"{epoch}_net_{key}{suffix}.pth"
        if not path.exists():
            raise FileNotFoundError(f"missing generator weights: {path}")
        net = networks.define_G(3, 3, ngf, netG, norm, False)
        state = torch.load(path, map_location="cpu", weights_only=True)
        if hasattr(state, "_metadata"):
            del state._metadata
        module = net.module if hasattr(net, "module") else net
        module.load_state_dict(state)
        module.to(device).eval()
        out[key] = module
    return out, dict(netG=netG, norm=norm, ngf=ngf, epoch=epoch, name=name, weights=weights,
                     comparison_id=identity.get('comparison_id'),
                     crops_digest=identity.get('crops_digest'),
                     training_policy=settings.get('training_policy'),
                     training_config=settings)


def _to_tensor(image, device):
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)
    return (tensor * 2.0 - 1.0).to(device)


def _to_image(tensor):
    array = ((tensor.squeeze(0).clamp(-1, 1) + 1.0) * 127.5).round().byte().cpu().numpy()
    return Image.fromarray(array.transpose(1, 2, 0))


def translate(generator, crop, size, transform="identity", device="cpu"):
    """Run one crop through one generator and return the raw translation."""
    source = crop if transform == "identity" else crop.transpose(Image.FLIP_LEFT_RIGHT)
    small = source.resize((size, size), Image.Resampling.LANCZOS)
    with torch.no_grad():
        output = generator(_to_tensor(small, device))
    result = _to_image(output).resize(crop.size, Image.Resampling.LANCZOS)
    return result if transform == "identity" else result.transpose(Image.FLIP_LEFT_RIGHT)


def composite(translated, source, mask):
    """Keep the generated fruit, keep the real background."""
    alpha = (np.asarray(mask) > 0)[:, :, None]
    blended = np.where(alpha, np.asarray(translated), np.asarray(source))
    return Image.fromarray(blended.astype(np.uint8))


def run(dataset, generators, out, condition, parents=None, slots=1, size=None,
        arm_size=None, device="cpu", meta=None):
    """Generate every (parent, slot) of one condition into a reviewable bundle."""
    size = int(size or config.CROP_SIZE)
    arm_size = int(arm_size or config.ARM_CROP_SIZE)
    parents = list(parents or [p for p in dataset.scope_ids if p not in config.EXCLUDED_PARENTS])
    outside = [pid for pid in parents if pid not in dataset.scope_ids]
    if outside:
        raise ValueError(f"parents outside the frozen scope: {outside}")
    out = Path(out)
    for sub in ("images", "masks", "raw"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    records, started = [], time.time()
    for pid in parents:
        source, mask, geometry = dataset.prepare(pid, arm_size)
        parent_class = int(dataset.by_id[pid]["class_id"])
        target_class = RIPE if parent_class == UNRIPE else UNRIPE
        key = "G_A" if parent_class == UNRIPE else "G_B"
        for slot in range(slots):
            transform = SLOT_TRANSFORMS[slot % len(SLOT_TRANSFORMS)]
            began = time.time()
            translated = translate(generators[key], source, size, transform, device)
            candidate = composite(translated, source, mask)
            seconds = time.time() - began
            candidate_id = f"{condition}__{pid}__{slot}"
            image_path = out / "images" / f"{candidate_id}.png"
            mask_path = out / "masks" / f"{candidate_id}.png"
            raw_path = out / "raw" / f"{candidate_id}.png"
            candidate.save(image_path)
            mask.save(mask_path)
            # The uncomposited translation is kept so the background change ratio
            # can be measured; compositing would otherwise make it zero by construction.
            translated.save(raw_path)
            box = _mask_box(mask)
            records.append(dict(
                candidate_id=candidate_id, condition=condition, method="cyclegan",
                parent_id=pid, slot=slot, class_id=target_class, parent_class_id=parent_class,
                direction=f"{CLASS_NAMES[parent_class]}_to_{CLASS_NAMES[target_class]}",
                generator=key, transform=transform, seed=0, seconds=round(seconds, 3),
                bbox=box, geometry=geometry,
                trace=dict(size=size, arm_size=arm_size, composite="mask", **(meta or {})),
                image=f"images/{image_path.name}", mask=f"masks/{mask_path.name}",
                raw=f"raw/{raw_path.name}",
                image_sha256=_sha(image_path), mask_sha256=_sha(mask_path),
                status="needs_review", reviewer="",
                annotation_basis="parent mask proposal; review decides whether it still fits",
            ))
    elapsed = time.time() - started
    (out / "records.json").write_text(json.dumps(records, indent=2) + "\n")
    (out / "cost.json").write_text(json.dumps({condition: dict(
        method="cyclegan", proposals=len(records),
        generate_seconds=round(elapsed, 3),
        seconds_per_proposal=round(elapsed / max(1, len(records)), 3))}, indent=2) + "\n")
    (out / "run_summary.json").write_text(json.dumps(dict(
        version=config.VERSION, condition=condition, parents=parents, slots=slots,
        excluded_parents=list(config.EXCLUDED_PARENTS), size=size, arm_size=arm_size,
        scope=dataset.scope_hash, manifest=config.EXPECTED_MANIFEST,
        per_parent_class={pid: int(dataset.by_id[pid]["class_id"]) for pid in parents},
        direction_note="candidate class is the opposite of its parent class",
        test_used=False, **(meta or {})), indent=2) + "\n")
    return records


def _mask_box(mask):
    array = np.asarray(mask) > 0
    if not array.any():
        raise ValueError("empty mask")
    y, x = np.where(array)
    return [int(x.min()), int(y.min()), int(x.max() + 1 - x.min()), int(y.max() + 1 - y.min())]
