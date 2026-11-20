"""Partial-translation rate: the share of fruit pixels still on the source side.

Blind review of G4 and G0X-v2 rejected candidates whose fruit was only partly
translated, one contiguous region keeping the source colour. A mean colour
statistic cannot see this (a half-translated fruit can have a target-side mean),
so the check is made per pixel:

    source side = Cr on the source-class side of the midpoint between the two
                  class centroids of mean Cr (train crops)

For a ripe -> unripe candidate that is the share of fruit pixels with Cr above
the midpoint. Real unripe train crops average 4.5 percent by this rule. The rule
is a diagnostic, not a label; acceptance stays with the blind reviewer.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .color import cr_numpy
from .config import RIPE, UNRIPE


def cr_midpoint(centroids_path):
    classes = json.loads(Path(centroids_path).read_text())["classes"]
    by_class = {entry["class_id"]: entry["centroid"][0] for entry in classes.values()}
    return (by_class[RIPE] + by_class[UNRIPE]) / 2.0


def source_side_share(rgb, mask, target_class, midpoint):
    """Share of mask pixels whose Cr sits on the source-class side of `midpoint`.

    `rgb` is HxWx3 in [0, 1], `mask` HxW boolean, `target_class` the class the
    candidate should read as.
    """
    selected = np.asarray(mask, bool)
    if not selected.any():
        raise ValueError("empty mask")
    cr = cr_numpy(np.asarray(rgb, np.float64))
    # Ripe has the higher mean Cr, so an unripe target keeps source pixels above it.
    source = cr > midpoint if int(target_class) == UNRIPE else cr < midpoint
    return float((source & selected).sum() / selected.sum())
