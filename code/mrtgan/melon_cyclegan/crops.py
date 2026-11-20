"""Build the unpaired ROI crop domains that the translation model trains on.

Only the frozen TRAIN split is read. For every train image the melon bounding box
is expanded by a context margin, squared, cropped, padded by reflection when the
box leaves the frame, and resized to `CROP_SIZE`. A fruit mask is estimated with
GrabCut seeded by the same box; the mask is used for the background, edge and
colour terms and for compositing at generation time, never as a hard target.

Outputs (under `--out`):

    trainA/<id>.png        unripe crops   (class_id 1)
    trainB/<id>.png        ripe crops     (class_id 0)
    trainA_mask/<id>.png   fruit mask for the matching crop
    trainB_mask/<id>.png
    crops_manifest.json    per-crop geometry, mask area, sha256
    color_centroids.json   phi centroid per class, the colour-aware loss target
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from . import config
from .color import phi_numpy

GRABCUT_ITERATIONS = 5
GRABCUT_MIN_AREA = 0.05     # of the seed box; below this the mask is not trusted
GRABCUT_MAX_AREA = 0.98     # of the whole crop; above this the mask is not trusted


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_manifest(dataset_root=None):
    """Rows of the frozen manifest, after checking the manifest digest."""
    import csv

    root = Path(dataset_root or config.DATASET_ROOT)
    manifest = root / "manifest.csv"
    if _sha(manifest) != config.EXPECTED_MANIFEST:
        raise ValueError("frozen manifest mismatch; this experiment targets regrouped v4 only")
    with open(manifest, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    mapping = json.loads((root / "dataset.json").read_text())["label_mapping"]
    if mapping != {"0": "ripe", "1": "unripe"}:
        raise ValueError("class mapping mismatch")
    return root, rows


def read_boxes(root, row):
    """YOLO label of one row as pixel boxes [x, y, w, h] with the class id."""
    width, height = int(row["width"]), int(row["height"])
    out = []
    for line in (root / row["label"]).read_text().splitlines():
        if not line.strip():
            continue
        c, x, y, bw, bh = map(float, line.split())
        if int(c) not in config.CLASS_NAMES or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
            raise ValueError(f"invalid YOLO label in {row['label']}")
        out.append((int(c), [(x - bw / 2) * width, (y - bh / 2) * height, bw * width, bh * height]))
    return out


def square_box(box, margin):
    """Square window around `box` grown by `margin` on each side."""
    x, y, w, h = box
    side = max(w, h) * (1.0 + 2.0 * margin)
    cx, cy = x + w / 2.0, y + h / 2.0
    left = int(math.floor(cx - side / 2.0))
    top = int(math.floor(cy - side / 2.0))
    side = int(math.ceil(side))
    return left, top, side


def crop_with_reflection(image, left, top, side):
    """Crop a square window, reflecting the image where the window leaves it."""
    import cv2

    array = np.asarray(image)
    height, width = array.shape[:2]
    l, t = max(0, left), max(0, top)
    r, b = min(width, left + side), min(height, top + side)
    if r <= l or b <= t:
        raise ValueError("crop window does not intersect the image")
    inner = array[t:b, l:r]
    pad = (max(0, -top), max(0, top + side - height), max(0, -left), max(0, left + side - width))
    if any(pad):
        inner = cv2.copyMakeBorder(inner, pad[0], pad[1], pad[2], pad[3], cv2.BORDER_REFLECT_101)
    return inner


def grabcut_mask(crop, box):
    """Fruit mask from GrabCut seeded by `box` in crop coordinates.

    Falls back to the filled box when GrabCut collapses or floods, which keeps
    the pipeline deterministic instead of silently producing an unusable mask.
    """
    import cv2

    height, width = crop.shape[:2]
    x, y, w, h = box
    x = int(max(1, min(width - 3, round(x))))
    y = int(max(1, min(height - 3, round(y))))
    w = int(max(2, min(width - x - 1, round(w))))
    h = int(max(2, min(height - y - 1, round(h))))
    filled = np.zeros((height, width), dtype=np.uint8)
    filled[y:y + h, x:x + w] = 255
    state = np.zeros((height, width), np.uint8)
    background, foreground = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), state, (x, y, w, h),
                    background, foreground, GRABCUT_ITERATIONS, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return filled, "grabcut_error"
    mask = np.where((state == cv2.GC_FGD) | (state == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    area = float((mask > 0).sum())
    if area < GRABCUT_MIN_AREA * w * h:
        return filled, "grabcut_collapsed"
    if area > GRABCUT_MAX_AREA * width * height:
        return filled, "grabcut_flooded"
    return mask, "grabcut"


def build(out, dataset_root=None, size=None, margin=None, split="train"):
    """Write both crop domains plus their masks and the colour centroids."""
    import cv2

    size = int(size or config.CROP_SIZE)
    margin = float(config.CROP_MARGIN if margin is None else margin)
    out = Path(out)
    root, rows = read_manifest(dataset_root)
    for name in ("trainA", "trainB", "trainA_mask", "trainB_mask"):
        (out / name).mkdir(parents=True, exist_ok=True)

    records, features = [], collections.defaultdict(list)
    for row in sorted(rows, key=lambda r: r["id"]):
        if row["split"] != split:
            continue
        boxes = read_boxes(root, row)
        if len(boxes) != 1:
            raise ValueError(f"{row['id']}: expected exactly one melon box, found {len(boxes)}")
        class_id, box = boxes[0]
        if str(class_id) != str(row["class_id"]):
            raise ValueError(f"{row['id']}: label class disagrees with the manifest")
        with Image.open(root / row["image"]) as handle:
            image = handle.convert("RGB")
        left, top, side = square_box(box, margin)
        window = crop_with_reflection(image, left, top, side)
        scale = size / float(side)
        crop = cv2.resize(window, (size, size), interpolation=cv2.INTER_AREA)
        inner = [(box[0] - left) * scale, (box[1] - top) * scale, box[2] * scale, box[3] * scale]
        mask, source = grabcut_mask(crop, inner)

        domain = config.CLASS_DOMAIN[class_id]
        image_path = out / f"train{domain}" / f"{row['id']}.png"
        mask_path = out / f"train{domain}_mask" / f"{row['id']}.png"
        Image.fromarray(crop).save(image_path)
        Image.fromarray(mask).save(mask_path)

        selected = mask > 0
        features[class_id].append(phi_numpy(crop.astype(np.float64) / 255.0, selected))
        records.append(dict(
            id=row["id"], class_id=class_id, domain=domain, split=split,
            source_box=[float(v) for v in box], crop_window=[left, top, side],
            crop_box=[float(v) for v in inner], mask_source=source,
            mask_area_ratio=float(selected.mean()),
            image=str(image_path.relative_to(out)), mask=str(mask_path.relative_to(out)),
            image_sha256=_sha(image_path), source_sha256=row["image_sha256"],
        ))

    centroids = {}
    for class_id, rowset in sorted(features.items()):
        stacked = np.stack(rowset)
        centroids[config.CLASS_NAMES[class_id]] = dict(
            class_id=class_id, count=int(stacked.shape[0]),
            centroid=[float(v) for v in stacked.mean(axis=0)],
            spread=[float(v) for v in stacked.std(axis=0)],
        )
    # Path-independent digest: two machines that build the same crops get the same
    # value, so a run can pin the crop set without pinning a directory name.
    crops_digest = hashlib.sha256(json.dumps(
        [[r["id"], r["class_id"], r["image_sha256"], r["mask_source"]] for r in records],
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest = dict(
        version=config.VERSION, split=split, size=size, margin=margin,
        crops_digest=crops_digest,
        dataset_manifest=config.EXPECTED_MANIFEST, dataset_root=str(root),
        counts=dict(collections.Counter(r["domain"] for r in records)),
        mask_sources=dict(collections.Counter(r["mask_source"] for r in records)),
        phi="[mean Cr, std Cr, mean a*/128, std a*/128] inside the mask",
        records=records,
    )
    (out / "crops_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "color_centroids.json").write_text(json.dumps(dict(
        version=config.VERSION, phi=manifest["phi"], source=f"{split} split real crops",
        crops_digest=crops_digest,
        classes=centroids), indent=2) + "\n")
    return manifest, centroids
