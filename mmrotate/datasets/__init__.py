# Copyright (c) OpenMMLab. All rights reserved.
from .dior import DIORDataset  # noqa: F401, F403
from .dota import DOTADataset  # noqa: F401, F403
from .hrsc import HRSCDataset  # noqa: F401, F403
from .transforms import *  # noqa: F401, F403

__all__ = ['DOTADataset', 'HRSCDataset', 'DIORDataset']
