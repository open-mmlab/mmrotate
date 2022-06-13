# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import ROTATED_DETECTORS
from .single_stage import RotatedSingleStageDetector


@ROTATED_DETECTORS.register_module()
class RotatedFCOS(RotatedSingleStageDetector):
    """Implementation of Rotated `FCOS.`__

    __ https://arxiv.org/abs/1904.01355
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(RotatedFCOS, self).__init__(backbone, neck, bbox_head, train_cfg,
                                          test_cfg, pretrained, init_cfg)
