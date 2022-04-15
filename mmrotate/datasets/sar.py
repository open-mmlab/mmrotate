# Copyright (c) OpenMMLab. All rights reserved.
from .builder import ROTATED_DATASETS
from .dota import DOTADataset


@ROTATED_DATASETS.register_module()
class SARDataset(DOTADataset):
    """SAR ship dataset for detection (Support RSSDD and HRSID)."""
    CLASSES = ('ship', )
    PALETTE = [
        (0, 255, 0),
    ]
