# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Union

import numpy as np
import torch
from mmdet.models.task_modules.coders.base_bbox_coder import BaseBBoxCoder
from mmdet.structures.bbox import HorizontalBoxes, get_box_tensor
from torch import Tensor

from mmrotate.registry import TASK_UTILS
from mmrotate.structures.bbox import (RotatedBoxes, qbox2rbox, rbox2hbox,
                                      rbox2qbox)


@TASK_UTILS.register_module()
class MidpointOffsetCoder(BaseBBoxCoder):
    """Mid point offset coder. This coder encodes bbox (x1, y1, x2, y2) into
    delta (dx, dy, dw, dh, da, db) and decodes delta (dx, dy, dw, dh, da, db)
    back to original bbox (x1, y1, x2, y2).

    Args:
        target_means (Sequence[float]): Denormalizing means of target for
            delta coordinates
        target_stds (Sequence[float]): Denormalizing standard deviation of
            target for delta coordinates
        angle_version (str, optional): Angle representations.
            Defaults to 'oc'.
        use_box_type (bool): Whether to warp decoded boxes with the
            box type data structure. Defaults to True.
    """
    encode_size = 6

    def __init__(self,
                 target_means: Sequence[float] = (0., 0., 0., 0., 0., 0.),
                 target_stds: Sequence[float] = (1., 1., 1., 1., 1., 1.),
                 angle_version: str = 'oc',
                 use_box_type=True) -> None:
        super().__init__(use_box_type=use_box_type)
        self.means = target_means
        self.stds = target_stds
        self.angle_version = angle_version
        assert self.angle_version in ['oc', 'le135', 'le90']

    def encode(self, bboxes: Union[HorizontalBoxes, Tensor],
               gt_bboxes: Union[RotatedBoxes, Tensor]) -> Tensor:
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.

        Args:
            bboxes (:obj:`HorizontalBoxes` or Tensor): Source boxes,
                e.g.,object proposals.
            gt_bboxes (:obj:`RotatedBoxes` or Tensor): Target of the
                transformation, e.g., ground-truth boxes.

        Returns:
            torch.Tensor: Box transformation deltas.
        """
        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == 4
        assert gt_bboxes.size(-1) == 5
        bboxes = get_box_tensor(bboxes)
        gt_bboxes = gt_bboxes.regularize_boxes(self.angle_version)
        encoded_bboxes = bbox2delta(bboxes, gt_bboxes, self.means, self.stds)
        return encoded_bboxes

    def decode(
            self,
            bboxes: Union[HorizontalBoxes, Tensor],
            pred_bboxes: Tensor,
            max_shape: Optional[Sequence[int]] = None,
            wh_ratio_clip: float = 16 / 1000) -> Union[RotatedBoxes, Tensor]:
        """Apply transformation `pred_bboxes` to `bboxes`.

        Args:
            bboxes (:obj:`HorizontalBoxes` or Tensor): Basic boxes.
                Shape (B, N, 4) or (N, 4)
            pred_bboxes (Tensor): Encoded offsets with respect to each
                roi. Has shape (B, N, 6) or (N, 6).
                Note N = num_anchors * W * H when rois is a grid of anchors.
            max_shape (Sequence[int] or torch.Tensor or Sequence[
               Sequence[int]],optional): Maximum bounds for boxes, specifies
               (H, W, C) or (H, W). If bboxes shape is (B, N, 6), then
               the max_shape should be a Sequence[Sequence[int]]
               and the length of max_shape should also be B.
            wh_ratio_clip (float, optional): The allowed ratio between
                width and height.

        Returns:
            Union[:obj:`RotatedBoxes`, Tensor]: Decoded boxes.
        """
        assert pred_bboxes.size(0) == bboxes.size(0)
        assert bboxes.size(-1) == 4
        assert pred_bboxes.size(-1) == 6
        bboxes = get_box_tensor(bboxes)
        decoded_bboxes = delta2bbox(bboxes, pred_bboxes, self.means, self.stds,
                                    wh_ratio_clip)
        decoded_bboxes = RotatedBoxes(decoded_bboxes).regularize_boxes(
            self.angle_version)
        if self.use_box_type:
            assert decoded_bboxes.size(-1) == 5, \
                ('Cannot warp decoded boxes with box type when decoded'
                 'boxes have shape of (N, num_classes * 5)')
            decoded_bboxes = RotatedBoxes(decoded_bboxes)
        return decoded_bboxes


def bbox2delta(proposals: Tensor,
               gts: Tensor,
               means: Sequence[float] = (0., 0., 0., 0., 0., 0.),
               stds: Sequence[float] = (1., 1., 1., 1., 1., 1.)):
    """Compute deltas of proposals w.r.t. gt.

    We usually compute the deltas of x, y, w, h, a, b of proposals w.r.t ground
    truth bboxes to get regression target. This is the inverse function of
    :func:`delta2bbox`.

    Args:
        proposals (Tensor): Boxes to be transformed, shape (N, ..., 4)
        gt (Tensor): Gt bboxes to be used as base, shape (N, ..., 5)
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates.

    Returns:
        Tensor: deltas with shape (N, 6), where columns represent dx, dy,
        dw, dh, da, db.
    """
    proposals = proposals.float()
    gts = gts.float()

    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0]
    ph = proposals[..., 3] - proposals[..., 1]

    hbb, poly = rbox2hbox(gts), rbox2qbox(gts)
    gx = (hbb[..., 0] + hbb[..., 2]) * 0.5
    gy = (hbb[..., 1] + hbb[..., 3]) * 0.5
    gw = hbb[..., 2] - hbb[..., 0]
    gh = hbb[..., 3] - hbb[..., 1]

    x_coor, y_coor = poly[:, 0::2], poly[:, 1::2]
    y_min, _ = torch.min(y_coor, dim=1, keepdim=True)
    x_max, _ = torch.max(x_coor, dim=1, keepdim=True)

    _x_coor = x_coor.clone()
    _x_coor[torch.abs(y_coor - y_min) > 0.1] = -1000
    ga, _ = torch.max(_x_coor, dim=1)

    _y_coor = y_coor.clone()
    _y_coor[torch.abs(x_coor - x_max) > 0.1] = -1000
    gb, _ = torch.max(_y_coor, dim=1)

    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)
    da = (ga - gx) / gw
    db = (gb - gy) / gh
    deltas = torch.stack([dx, dy, dw, dh, da, db], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas


def delta2bbox(rois: Tensor,
               deltas: Tensor,
               means: Sequence[float] = (0., 0., 0., 0., 0., 0.),
               stds: Sequence[float] = (1., 1., 1., 1., 1., 1.),
               wh_ratio_clip: float = 16 / 1000):
    """Apply deltas to shift/scale base boxes.

    Typically the rois are anchor or proposed bounding boxes and the deltas
    are network outputs used to shift/scale those boxes. This is the inverse
    function of :func:`bbox2delta`.


    Args:
        rois (Tensor): Boxes to be transformed. Has shape (N, 4).
        deltas (Tensor): Encoded offsets relative to each roi.
            Has shape (N, num_classes * 6) or (N, 6). Note
            N = num_base_anchors * W * H, when rois is a grid of
            anchors.
        means (Sequence[float]): Denormalizing means for delta coordinates.
            Defaults to (0., 0., 0., 0., 0., 0.).
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates. Defaults to (1., 1., 1., 1., 1., 1.).
        wh_ratio_clip (float): Maximum aspect ratio for boxes. Defaults to
            16 / 1000.

    Returns:
        Tensor: Boxes with shape (N, num_classes * 5) or (N, 5), where 5
        represent cx, cy, w, h, a.
    """
    num_bboxes = deltas.size(0)
    if num_bboxes == 0:
        return deltas.new_zeros((0, 5))

    means = deltas.new_tensor(means).view(1, -1)
    stds = deltas.new_tensor(stds).view(1, -1)
    delta_shape = deltas.shape
    reshaped_deltas = deltas.view(delta_shape[:-1] + (-1, 6))
    denorm_deltas = reshaped_deltas * stds + means
    dx = denorm_deltas[..., 0::6]
    dy = denorm_deltas[..., 1::6]
    dw = denorm_deltas[..., 2::6]
    dh = denorm_deltas[..., 3::6]
    da = denorm_deltas[..., 4::6]
    db = denorm_deltas[..., 5::6]
    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)
    # Compute center of each roi
    px = ((rois[..., None, None, 0] + rois[..., None, None, 2]) * 0.5)
    py = ((rois[..., None, None, 1] + rois[..., None, None, 3]) * 0.5)
    # Compute width/height of each roi
    pw = (rois[..., None, None, 2] - rois[..., None, None, 0])
    ph = (rois[..., None, None, 3] - rois[..., None, None, 1])
    # Use exp(network energy) to enlarge/shrink each roi
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    # Use network energy to shift the center of each roi
    gx = px + pw * dx
    gy = py + ph * dy

    x1 = gx - gw * 0.5
    y1 = gy - gh * 0.5
    x2 = gx + gw * 0.5
    y2 = gy + gh * 0.5

    da = da.clamp(min=-0.5, max=0.5)
    db = db.clamp(min=-0.5, max=0.5)
    ga = gx + da * gw
    _ga = gx - da * gw
    gb = gy + db * gh
    _gb = gy - db * gh
    polys = torch.stack([ga, y1, x2, gb, _ga, y2, x1, _gb], dim=-1)

    center = torch.stack([gx, gy, gx, gy, gx, gy, gx, gy], dim=-1)
    center_polys = polys - center
    diag_len = torch.sqrt(center_polys[..., 0::2] * center_polys[..., 0::2] +
                          center_polys[..., 1::2] * center_polys[..., 1::2])
    max_diag_len, _ = torch.max(diag_len, dim=-1, keepdim=True)
    diag_scale_factor = max_diag_len / diag_len
    center_polys_shape = center_polys.shape
    center_polys = center_polys.view(*center_polys_shape[:3], 4,
                                     -1) * diag_scale_factor.view(
                                         *center_polys_shape[:3], 4, 1)
    center_polys = center_polys.view(center_polys_shape)
    rectpolys = center_polys + center
    rbboxes = qbox2rbox(rectpolys).view(delta_shape[:-1] + (5, ))

    return rbboxes
