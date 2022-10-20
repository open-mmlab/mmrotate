# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

from mmdet.models.task_modules.coders import DeltaXYWHBBoxCoder
from mmdet.models.task_modules.coders.delta_xywh_bbox_coder import bbox2delta
from mmdet.structures.bbox import HorizontalBoxes, get_box_tensor
from torch import Tensor

from mmrotate.registry import TASK_UTILS
from mmrotate.structures.bbox import RotatedBoxes


@TASK_UTILS.register_module()
class DeltaXYWHHBBoxCoder(DeltaXYWHBBoxCoder):
    """Delta XYWH HBBox coder.

    This coder is almost the same as `DeltaXYWHBBoxCoder`. Besides the
    gt_bboxes of encode is :obj:`RotatedBoxes`.
    """

    def encode(self, bboxes: Union[HorizontalBoxes, Tensor],
               gt_bboxes: Union[RotatedBoxes, Tensor]) -> Tensor:
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.

        Args:
            bboxes (:obj:`HorizontalBoxes` or Tensor): Source boxes, e.g.,
                object proposals.
            gt_bboxes (:obj:`RotatedBoxes` or Tensor): Target of the
                transformation, e.g., ground-truth boxes.
        Returns:
            Tensor: Box transformation deltas
        """

        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == 4
        assert gt_bboxes.size(-1) == 5

        bboxes = get_box_tensor(bboxes)

        if not isinstance(gt_bboxes, RotatedBoxes):
            gt_bboxes = RotatedBoxes(gt_bboxes)
        gt_bboxes = gt_bboxes.convert_to('hbox').tensor

        encoded_bboxes = bbox2delta(bboxes, gt_bboxes, self.means, self.stds)
        return encoded_bboxes
