# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.task_modules.coders import DeltaXYWHBBoxCoder
from mmdet.models.task_modules.coders.delta_xywh_bbox_coder import bbox2delta
from mmdet.structures.bbox import BaseBoxes

from mmrotate.core.bbox.structures import RotatedBoxes
from mmrotate.registry import TASK_UTILS


@TASK_UTILS.register_module()
class DeltaXYWHHBBoxCoder(DeltaXYWHBBoxCoder):
    """Delta XYWH HBBox coder.
    Following the practice in `R-CNN <https://arxiv.org/abs/1311.2524>`_, this
    coder encodes bbox (x1, y1, x2, y2) into delta (dx, dy, dw, dh) and decodes
    delta (dx, dy, dw, dh) back to original bbox (x1, y1, x2, y2).
    """

    def encode(self, bboxes, gt_bboxes):
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.
        Args:
            bboxes (:obj:`BaseBoxes` or Tensor): Source boxes, e.g.,
                object proposals.
            gt_bboxes (:obj:`RotatedBoxes` or Tensor): Target of the
                transformation, e.g., ground-truth boxes.
        Returns:
            torch.Tensor: Box transformation deltas
        """

        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == 4
        assert gt_bboxes.size(-1) == 5

        if isinstance(bboxes, BaseBoxes):
            bboxes = bboxes.tensor

        if not isinstance(gt_bboxes, RotatedBoxes):
            gt_bboxes = RotatedBoxes(gt_bboxes)
        gt_bboxes = gt_bboxes.convert_to('hbox').tensor

        encoded_bboxes = bbox2delta(bboxes, gt_bboxes, self.means, self.stds)
        return encoded_bboxes
