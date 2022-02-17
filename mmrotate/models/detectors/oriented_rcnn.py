# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import ROTATED_DETECTORS
from .two_stage import RotatedTwoStageDetector


@ROTATED_DETECTORS.register_module()
class OrientedRCNN(RotatedTwoStageDetector):
    """Implementation of `Oriented R-CNN for Object Detection.

    <https://openaccess.thecvf.com/content/ICCV2021/papers
    /Xie_Oriented_R-CNN_for_Object_Detection_ICCV_2021_paper.pdf>`_
    """

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(OrientedRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
