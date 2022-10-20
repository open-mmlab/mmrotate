# Copyright (c) OpenMMLab. All rights reserved.
from .bbox_heads import RotatedShared2FCBBoxHead
from .gv_ratio_roi_head import GVRatioRoIHead
from .roi_extractors import RotatedSingleRoIExtractor

__all__ = [
    'RotatedShared2FCBBoxHead', 'RotatedSingleRoIExtractor', 'GVRatioRoIHead'
]
