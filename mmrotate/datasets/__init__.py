# Copyright (c) OpenMMLab. All rights reserved.
from .dior import DIORDataset  # noqa: F401, F403
from .dota import DOTAv2Dataset  # noqa: F401, F403
from .dota import DOTADataset, DOTAv15Dataset
from .hrsc import HRSCDataset  # noqa: F401, F403
from .ohd_sjtu import OHD_SJTUDataset_L, OHD_SJTUDataset_S
from .transforms import *  # noqa: F401, F403

__all__ = [
    'DOTADataset', 'DOTAv15Dataset', 'DOTAv2Dataset', 'HRSCDataset',
    'DIORDataset', 'OHD_SJTUDataset_S', 'OHD_SJTUDataset_L'
]
