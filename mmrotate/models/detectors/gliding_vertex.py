# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import ROTATED_DETECTORS
from .two_stage import RotatedTwoStageDetector


@ROTATED_DETECTORS.register_module()
class GlidingVertex(RotatedTwoStageDetector):
    """Implementation of `Gliding Vertex on the Horizontal Bounding Box for
    Multi-Oriented Object Detection <https://arxiv.org/pdf/1911.09358.pdf>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(GlidingVertex, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
