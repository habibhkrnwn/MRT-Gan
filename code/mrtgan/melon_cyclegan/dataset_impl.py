"""Unaligned dataset that also serves the fruit mask of each crop.

Layout produced by `melon_cyclegan.crops.build`:

    <dataroot>/trainA, trainA_mask, trainB, trainB_mask

Domain A is unripe and domain B is ripe. Crops are already square and already at
`crop_size`, so no random resized crop is applied; the only augmentation is the
horizontal flip of the plain CycleGAN recipe, applied to image and mask together
so the mask keeps matching its image. Hue is part of the label here and is never
jittered.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def _listing(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"missing domain folder: {folder}")
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def _to_tensor(image):
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = array[:, :, None]
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


def load_pair(image_path, mask_path, flip=False):
    """One crop and its mask as tensors: image in [-1, 1], mask in {0, 1}."""
    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
    with Image.open(mask_path) as handle:
        mask = handle.convert("L")
    if image.size != mask.size:
        raise ValueError(f"mask geometry mismatch for {image_path}")
    if flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return _to_tensor(image) * 2.0 - 1.0, (_to_tensor(mask) > 0.5).float()


class MelonRoiData:
    """Backend of the vendored `MelonDataset`; kept here so it stays testable."""

    def __init__(self, dataroot, phase="train", serial_batches=False, no_flip=False, max_dataset_size=float("inf")):
        root = Path(dataroot)
        self.dir_A = root / f"{phase}A"
        self.dir_B = root / f"{phase}B"
        self.dir_A_mask = root / f"{phase}A_mask"
        self.dir_B_mask = root / f"{phase}B_mask"
        self.A_paths = _listing(self.dir_A)[: int(min(max_dataset_size, 1e18))]
        self.B_paths = _listing(self.dir_B)[: int(min(max_dataset_size, 1e18))]
        if not self.A_paths or not self.B_paths:
            raise ValueError("both domains must contain crops")
        self.A_size, self.B_size = len(self.A_paths), len(self.B_paths)
        self.serial_batches = serial_batches
        self.no_flip = no_flip

    def mask_for(self, path, domain):
        folder = self.dir_A_mask if domain == "A" else self.dir_B_mask
        mask = folder / f"{Path(path).stem}.png"
        if not mask.exists():
            raise ValueError(f"missing mask for {path}")
        return mask

    def __len__(self):
        return max(self.A_size, self.B_size)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        if self.serial_batches:
            B_path = self.B_paths[index % self.B_size]
        else:
            B_path = self.B_paths[random.randint(0, self.B_size - 1)]
        flip_A = (not self.no_flip) and random.random() < 0.5
        flip_B = (not self.no_flip) and random.random() < 0.5
        A, A_mask = load_pair(A_path, self.mask_for(A_path, "A"), flip_A)
        B, B_mask = load_pair(B_path, self.mask_for(B_path, "B"), flip_B)
        return dict(A=A, B=B, A_mask=A_mask, B_mask=B_mask, A_paths=str(A_path), B_paths=str(B_path))
