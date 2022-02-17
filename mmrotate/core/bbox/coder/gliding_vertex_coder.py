# Copyright (c) OpenMMLab. All rights reserved.
# Modified from jbwang1997: https://github.com/jbwang1997/OBBDetection
import torch
from mmdet.core.bbox.coder.base_bbox_coder import BaseBBoxCoder

from ..builder import ROTATED_BBOX_CODERS
from ..transforms import obb2poly, poly2obb


@ROTATED_BBOX_CODERS.register_module()
class GVFixCoder(BaseBBoxCoder):
    """Gliding vertex fix coder.

    this coder encodes bbox (cx, cy, w, h, a) into delta (dt, dr, dd, dl) and
    decodes delta (dt, dr, dd, dl) back to original bbox (cx, cy, w, h, a).
    Args:
        angle_range (str, optional): Angle representations. Defaults to 'oc'.
    """

    def __init__(self, angle_range='oc', **kwargs):
        self.version = angle_range

        super(GVFixCoder, self).__init__(**kwargs)

    def encode(self, rbboxes):
        """Get box regression transformation deltas.

        Args:
            rbboxes (torch.Tensor): Source boxes, e.g., object proposals.
        Returns:
            torch.Tensor: Box transformation deltas
        """
        assert rbboxes.size(1) == 5

        polys = obb2poly(rbboxes, self.version)

        max_x, max_x_idx = polys[:, ::2].max(1)
        min_x, min_x_idx = polys[:, ::2].min(1)
        max_y, max_y_idx = polys[:, 1::2].max(1)
        min_y, min_y_idx = polys[:, 1::2].min(1)
        hbboxes = torch.stack([min_x, min_y, max_x, max_y], dim=1)

        polys = polys.view(-1, 4, 2)
        num_polys = polys.size(0)
        polys_ordered = torch.zeros_like(polys)
        polys_ordered[:, 0] = polys[range(num_polys), min_y_idx]
        polys_ordered[:, 1] = polys[range(num_polys), max_x_idx]
        polys_ordered[:, 2] = polys[range(num_polys), max_y_idx]
        polys_ordered[:, 3] = polys[range(num_polys), min_x_idx]

        t_x = polys_ordered[:, 0, 0]
        r_y = polys_ordered[:, 1, 1]
        d_x = polys_ordered[:, 2, 0]
        l_y = polys_ordered[:, 3, 1]

        dt = (t_x - hbboxes[:, 0]) / (hbboxes[:, 2] - hbboxes[:, 0])
        dr = (r_y - hbboxes[:, 1]) / (hbboxes[:, 3] - hbboxes[:, 1])
        dd = (hbboxes[:, 2] - d_x) / (hbboxes[:, 2] - hbboxes[:, 0])
        dl = (hbboxes[:, 3] - l_y) / (hbboxes[:, 3] - hbboxes[:, 1])

        h_mask = (polys_ordered[:, 0, 1] - polys_ordered[:, 1, 1] == 0) | \
                 (polys_ordered[:, 1, 0] - polys_ordered[:, 2, 0] == 0)
        fix_deltas = torch.stack([dt, dr, dd, dl], dim=1)
        fix_deltas[h_mask, :] = 1
        return fix_deltas

    def decode(self, hbboxes, fix_deltas):
        """Apply transformation `fix_deltas` to `boxes`.
        Args:
            hbboxes (torch.Tensor): Basic boxes. Shape (B, N, 4) or (N, 4)
            fix_deltas (torch.Tensor): Encoded offsets with respect to each
                roi. Has shape (B, N, num_classes * 4) or (B, N, 4) or
               (N, num_classes * 4) or (N, 4). Note N = num_anchors * W * H
               when rois is a grid of anchors.Offset encoding follows [1]_.
        Returns:
            torch.Tensor: Decoded boxes.
        """
        x1 = hbboxes[:, 0::4]
        y1 = hbboxes[:, 1::4]
        x2 = hbboxes[:, 2::4]
        y2 = hbboxes[:, 3::4]
        w = hbboxes[:, 2::4] - hbboxes[:, 0::4]
        h = hbboxes[:, 3::4] - hbboxes[:, 1::4]

        pred_t_x = x1 + w * fix_deltas[:, 0::4]
        pred_r_y = y1 + h * fix_deltas[:, 1::4]
        pred_d_x = x2 - w * fix_deltas[:, 2::4]
        pred_l_y = y2 - h * fix_deltas[:, 3::4]

        polys = torch.stack(
            [pred_t_x, y1, x2, pred_r_y, pred_d_x, y2, x1, pred_l_y], dim=-1)
        polys = polys.flatten(1)
        rbboxes = poly2obb(polys, self.version)

        return rbboxes


@ROTATED_BBOX_CODERS.register_module()
class GVRatioCoder(BaseBBoxCoder):
    """Gliding vertex ratio coder.

    this coder encodes bbox (cx, cy, w, h, a) into delta (ratios).
    Args:
        angle_range (str, optional): Angle representations. Defaults to 'oc'.
    """

    def __init__(self, angle_range='oc', **kwargs):
        self.version = angle_range
        super(GVRatioCoder, self).__init__(**kwargs)

    def encode(self, rbboxes):
        """Get box regression transformation deltas.

        Args:
            rbboxes (torch.Tensor): Source boxes, e.g., object proposals.
        Returns:
            torch.Tensor: Box transformation deltas
        """
        assert rbboxes.size(1) == 5

        polys = obb2poly(rbboxes, self.version)
        max_x, _ = polys[:, ::2].max(1)
        min_x, _ = polys[:, ::2].min(1)
        max_y, _ = polys[:, 1::2].max(1)
        min_y, _ = polys[:, 1::2].min(1)
        hbboxes = torch.stack([min_x, min_y, max_x, max_y], dim=1)

        h_areas = (hbboxes[:, 2] - hbboxes[:, 0]) * \
                  (hbboxes[:, 3] - hbboxes[:, 1])

        polys = polys.view(polys.size(0), 4, 2)
        areas = polys.new_zeros(polys.size(0))
        for i in range(4):
            areas += 0.5 * (
                polys[:, i, 0] * polys[:, (i + 1) % 4, 1] -
                polys[:, (i + 1) % 4, 0] * polys[:, i, 1])
        areas = torch.abs(areas)

        ratios = areas / h_areas
        return ratios[:, None]

    def decode(self, bboxes, bboxes_pred):
        """Apply transformation `fix_deltas` to `boxes`.

        Args:
            bboxes (torch.Tensor)
            bboxes_pred (torch.Tensor)
        Returns:
            NotImplementedError
        """
        raise NotImplementedError
