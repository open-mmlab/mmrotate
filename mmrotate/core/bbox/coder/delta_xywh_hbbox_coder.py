# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.task_modules.coders import DeltaXYWHBBoxCoder
from mmdet.models.task_modules.coders.delta_xywh_bbox_coder import bbox2delta
from mmdet.structures.bbox import HorizontalBoxes
from torch import Tensor

from mmrotate.core.bbox.structures import RotatedBoxes
from mmrotate.registry import TASK_UTILS


@TASK_UTILS.register_module()
class DeltaXYWHHBBoxCoder(DeltaXYWHBBoxCoder):
    """Delta XYWH HBBox coder.

    This coder is almost the same as `DeltaXYWHBBoxCoder`. Besides the
    gt_bboxes of encode is :obj:`RotatedBoxes`.
    """

    def encode(self, bboxes: HorizontalBoxes,
               gt_bboxes: RotatedBoxes) -> Tensor:
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.

        Args:
            bboxes (:obj:`HorizontalBoxes`): Source boxes, e.g.,
                object proposals.
            gt_bboxes (:obj:`RotatedBoxes`): Target of the
                transformation, e.g., ground-truth boxes.
        Returns:
            Tensor: Box transformation deltas
        """

        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == 4
        assert gt_bboxes.size(-1) == 5

        bboxes = bboxes.tensor

        if not isinstance(gt_bboxes, RotatedBoxes):
            gt_bboxes = RotatedBoxes(gt_bboxes)
        gt_bboxes = gt_bboxes.convert_to('hbox').tensor

        encoded_bboxes = bbox2delta(bboxes, gt_bboxes, self.means, self.stds)
        return encoded_bboxes
