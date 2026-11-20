"""Paths and frozen constants shared by every step of the CycleGAN experiment."""
from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # Ujicoba CycleGAN/
PLAN_ROOT = ROOT.parent                 # Rencana_Riset_Melon_2026-09-18/
VENDOR = ROOT / "vendor" / "pytorch_cyclegan"
BENCHMARK = PLAN_ROOT / "14_Benchmark_Kaggle"
BENCHMARK_SRC = BENCHMARK / "src"

# The frozen dataset every published melon result is bound to (regrouped v4).
DATASET_ROOT = Path(os.environ.get("MELON_DATASET") or PLAN_ROOT / "10_Migrasi_Kaggle" / "dataset_regrouped")
EXPECTED_MANIFEST = "185e14221e8882f7bcabb5fbb455d6a64aa09378f07234e1ccd4f855531df6ba"

# Class mapping of the frozen dataset. Do not reorder.
CLASS_NAMES = {0: "ripe", 1: "unripe"}
RIPE, UNRIPE = 0, 1

# CycleGAN domains. Upstream calls the source domain A and the target domain B.
DOMAIN_CLASS = {"A": UNRIPE, "B": RIPE}
CLASS_DOMAIN = {UNRIPE: "A", RIPE: "B"}

# Generation scope: the reviewed-mask train parents, minus the disputed label.
EXCLUDED_PARENTS = ("DSC_0093",)

# Detector scope of the budget-matched comparison. Blind review left G4 with no
# accepted candidate for DSC_0110 and DSC_0397, so those two parents are dropped
# from EVERY arm; an arm comparison is only meaningful on one shared parent set.
ARM_EXCLUDED_PARENTS = ("DSC_0093", "DSC_0110", "DSC_0397")

# Crop geometry for the translation model.
CROP_SIZE = 256          # generator input/output, PDF step 4
CROP_MARGIN = 0.15       # context margin per side, PDF step 3
ARM_CROP_SIZE = 512      # candidate size written into a detector arm, matches build_detector_arms.py

# Loss weights, PDF "Bobot loss: titik awal".
LAMBDA = dict(cycle_l1=10.0, ssim=1.0, identity=0.5, background=5.0, color=1.0, edge=0.0)

# Quality control thresholds, PDF "Aturan penerimaan otomatis" and the MVP block.
QC = dict(
    target_classifier_threshold=0.70,
    reject_top_background_change_percent=15.0,
    reject_color_distance_outliers=True,
    duplicate_filter="embedding_nearest_neighbor",
    visual_audit_per_direction=75,
)

VERSION = "melon-cyclegan-1.0"


def vendor_on_path():
    """Put the vendored upstream repository on sys.path and return its root.

    Upstream imports itself as top-level packages (`data`, `models`, `options`,
    `util`), so the repository root has to be on sys.path before those imports.
    """
    import sys

    root = str(VENDOR)
    if root not in sys.path:
        sys.path.insert(0, root)
    package = str(ROOT)
    if package not in sys.path:
        sys.path.insert(0, package)
    return VENDOR


def benchmark_on_path():
    """Put `14_Benchmark_Kaggle/src` on sys.path so `common` can be imported."""
    import sys

    root = str(BENCHMARK_SRC)
    if root not in sys.path:
        sys.path.insert(0, root)
    return BENCHMARK_SRC
