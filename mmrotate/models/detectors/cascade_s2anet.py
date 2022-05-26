# Copyright (c) OpenMMLab. All rights reserved.
from mmrotate.core import rbbox2result
from ..builder import ROTATED_DETECTORS
from .single_stage import RotatedSingleStageDetector


@ROTATED_DETECTORS.register_module()
class CascadeS2ANet(RotatedSingleStageDetector):
    """Implementation of `Align Deep Features for Oriented Object Detection.`__

    __ https://ieeexplore.ieee.org/document/9377550
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(CascadeS2ANet,
              self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg,
                             pretrained, init_cfg)

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes. \
                The outer list corresponds to each image. The inner list \
                corresponds to each class.
        """
        x = self.extract_feat(img)
        outs, rois = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale, rois=rois)

        bbox_results = [
            rbbox2result(det_bboxes, det_labels,
                         self.bbox_head.heads[-1].num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results
