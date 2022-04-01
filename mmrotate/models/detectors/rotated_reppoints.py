# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import ROTATED_DETECTORS
from .single_stage import RotatedSingleStageDetector


@ROTATED_DETECTORS.register_module()
class RotatedRepPoints(RotatedSingleStageDetector):
    """Implementation of Rotated RepPoints."""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RotatedRepPoints, self).__init__(backbone, neck, bbox_head,
                                               train_cfg, test_cfg, pretrained)
