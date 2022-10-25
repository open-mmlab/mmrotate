# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

import torch
from mmdet.models.task_modules.coders.base_bbox_coder import BaseBBoxCoder
from mmdet.structures.bbox import HorizontalBoxes, get_box_tensor
from torch import Tensor

from mmrotate.registry import TASK_UTILS
from mmrotate.structures.bbox import QuadriBoxes


@TASK_UTILS.register_module()
class GVFixCoder(BaseBBoxCoder):
    """Gliding vertex fix coder.

    this coder encodes qbox (x1, y1, ..., x4, y4) into delta (dt, dr, dd, dl)
    and decodes delta (dt, dr, dd, dl) back to original qbox (x1, y1, ..., x4,
    y4).
    """

    def encode(self, qboxes: Union[QuadriBoxes, Tensor]):
        """Get box regression transformation deltas.

        Args:
            qboxes (:obj:`QuadriBoxes` or Tensor): Source boxes,
                e.g., object proposals.

        Returns:
            Tensor: Box transformation deltas.
        """
        assert qboxes.size(1) == 8

        qboxes = get_box_tensor(qboxes)

        max_x, max_x_idx = qboxes[:, ::2].max(1)
        min_x, min_x_idx = qboxes[:, ::2].min(1)
        max_y, max_y_idx = qboxes[:, 1::2].max(1)
        min_y, min_y_idx = qboxes[:, 1::2].min(1)
        hbboxes = torch.stack([min_x, min_y, max_x, max_y], dim=1)

        qboxes = qboxes.view(-1, 4, 2)
        num_qboxes = qboxes.size(0)
        qboxes_ordered = torch.zeros_like(qboxes)
        qboxes_ordered[:, 0] = qboxes[range(num_qboxes), min_y_idx]
        qboxes_ordered[:, 1] = qboxes[range(num_qboxes), max_x_idx]
        qboxes_ordered[:, 2] = qboxes[range(num_qboxes), max_y_idx]
        qboxes_ordered[:, 3] = qboxes[range(num_qboxes), min_x_idx]

        t_x = qboxes_ordered[:, 0, 0]
        r_y = qboxes_ordered[:, 1, 1]
        d_x = qboxes_ordered[:, 2, 0]
        l_y = qboxes_ordered[:, 3, 1]

        dt = (t_x - hbboxes[:, 0]) / (hbboxes[:, 2] - hbboxes[:, 0])
        dr = (r_y - hbboxes[:, 1]) / (hbboxes[:, 3] - hbboxes[:, 1])
        dd = (hbboxes[:, 2] - d_x) / (hbboxes[:, 2] - hbboxes[:, 0])
        dl = (hbboxes[:, 3] - l_y) / (hbboxes[:, 3] - hbboxes[:, 1])

        h_mask = (qboxes_ordered[:, 0, 1] - qboxes_ordered[:, 1, 1] == 0) | \
                 (qboxes_ordered[:, 1, 0] - qboxes_ordered[:, 2, 0] == 0)
        fix_deltas = torch.stack([dt, dr, dd, dl], dim=1)
        fix_deltas[h_mask, :] = 1
        return fix_deltas

    def decode(self, hboxes: Union[HorizontalBoxes, Tensor],
               fix_deltas: Tensor):
        """Apply transformation `fix_deltas` to `boxes`.

        Args:
            hboxes (:obj:`HorizontalBoxes` or Tensor): Basic boxes.
                Shape (B, N, 4) or (N, 4)
            fix_deltas (Tensor): Encoded offsets with respect to each
                roi. Has shape (B, N, num_classes * 4) or (B, N, 4) or
                (N, num_classes * 4) or (N, 4).

        Returns:
            Tensor: Decoded boxes.
        """
        assert hboxes.size(1) == 4

        hboxes = get_box_tensor(hboxes)

        x1 = hboxes[:, 0::4]
        y1 = hboxes[:, 1::4]
        x2 = hboxes[:, 2::4]
        y2 = hboxes[:, 3::4]
        w = hboxes[:, 2::4] - hboxes[:, 0::4]
        h = hboxes[:, 3::4] - hboxes[:, 1::4]

        pred_t_x = x1 + w * fix_deltas[:, 0::4]
        pred_r_y = y1 + h * fix_deltas[:, 1::4]
        pred_d_x = x2 - w * fix_deltas[:, 2::4]
        pred_l_y = y2 - h * fix_deltas[:, 3::4]

        qboxes = torch.stack(
            [pred_t_x, y1, x2, pred_r_y, pred_d_x, y2, x1, pred_l_y], dim=-1)
        qboxes = qboxes.flatten(1)

        return qboxes


@TASK_UTILS.register_module()
class GVRatioCoder(BaseBBoxCoder):
    """Gliding vertex ratio coder.

    this coder encodes qbox (x1, y1, ..., x4, y4) into delta (ratios).
    """

    def encode(self, qboxes: Union[QuadriBoxes, Tensor]):
        """Get box regression transformation deltas.

        Args:
            qboxes (:obj:`QuadriBoxes` or Tensor): Source boxes,
                e.g., object proposals.

        Returns:
            Tensor: Box transformation deltas
        """
        assert qboxes.size(1) == 8

        qboxes = get_box_tensor(qboxes)

        max_x, _ = qboxes[:, ::2].max(1)
        min_x, _ = qboxes[:, ::2].min(1)
        max_y, _ = qboxes[:, 1::2].max(1)
        min_y, _ = qboxes[:, 1::2].min(1)
        hboxes = torch.stack([min_x, min_y, max_x, max_y], dim=1)

        h_areas = (hboxes[:, 2] - hboxes[:, 0]) * \
                  (hboxes[:, 3] - hboxes[:, 1])

        qboxes = qboxes.view(qboxes.size(0), 4, 2)
        areas = qboxes.new_zeros(qboxes.size(0))
        for i in range(4):
            areas += 0.5 * (
                qboxes[:, i, 0] * qboxes[:, (i + 1) % 4, 1] -
                qboxes[:, (i + 1) % 4, 0] * qboxes[:, i, 1])
        areas = torch.abs(areas)

        ratios = areas / h_areas
        return ratios[:, None]

    def decode(self):
        """NotImplementedError."""
        raise NotImplementedError
