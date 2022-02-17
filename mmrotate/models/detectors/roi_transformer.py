# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import ROTATED_DETECTORS
from .two_stage import RotatedTwoStageDetector


@ROTATED_DETECTORS.register_module()
class RoITransformer(RotatedTwoStageDetector):
    """Implementation of `Learning RoI Transformer for Oriented Object
    Detection in Aerial Images.

    <https://openaccess.thecvf.com/content_CVPR_2019/papers/Ding_
    Learning_RoI_Transformer_for_Oriented_Object_Detection_in_
    Aerial_Images_CVPR_2019_paper.pdf#:~:text=The%20core%20idea
    %20of%20RoI%20Transformer%20is%20to,embed-%20ded%20into%20
    detectors%20for%20oriented%20object%20detection.>`_
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
        super(RoITransformer, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
