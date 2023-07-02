# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import ROTATED_DETECTORS
from .single_stage import RotatedSingleStageDetector


@ROTATED_DETECTORS.register_module()
class RotatedDalRetinaNet(RotatedSingleStageDetector):
    """Implementation of Rotated `RetinaNet.`__

    __ https://arxiv.org/abs/1708.02002
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(RotatedDalRetinaNet,
              self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg,
                             pretrained, init_cfg)

    def set_epoch(self, epoch, max_epoch):
        self.bbox_head.epoch = epoch
        self.bbox_head.max_epoch = max_epoch
