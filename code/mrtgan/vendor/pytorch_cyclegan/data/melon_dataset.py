"""Bridge file: exposes the melon ROI dataset to the upstream `--dataset_mode melon`.

The implementation lives in `melon_cyclegan/dataset_impl.py`; this file only
registers it with the upstream loader, which imports `data.<name>_dataset`.
"""
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from data.base_dataset import BaseDataset
from melon_cyclegan.dataset_impl import MelonRoiData


class MelonDataset(BaseDataset):
    """Unaligned ripe/unripe crops that also carry their fruit masks."""

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.data = MelonRoiData(
            opt.dataroot,
            phase=opt.phase,
            serial_batches=opt.serial_batches,
            no_flip=opt.no_flip,
            max_dataset_size=opt.max_dataset_size,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
