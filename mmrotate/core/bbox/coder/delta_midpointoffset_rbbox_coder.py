# Copyright (c) OpenMMLab. All rights reserved.
# Modified from jbwang1997: https://github.com/jbwang1997/OBBDetection
import mmcv
import numpy as np
import torch
from mmdet.core.bbox.coder.base_bbox_coder import BaseBBoxCoder

from ..builder import ROTATED_BBOX_CODERS
from ..transforms import obb2poly, obb2xyxy, poly2obb


@ROTATED_BBOX_CODERS.register_module()
class MidpointOffsetCoder(BaseBBoxCoder):
    """Mid point offset coder. This coder encodes bbox (x1, y1, x2, y2) into \
    delta (dx, dy, dw, dh, da, db) and decodes delta (dx, dy, dw, dh, da, db) \
    back to original bbox (x1, y1, x2, y2).

    Args:
        target_means (Sequence[float]): Denormalizing means of target for
            delta coordinates
        target_stds (Sequence[float]): Denormalizing standard deviation of
            target for delta coordinates
        angle_range (str, optional): Angle representations. Defaults to 'oc'.
    """

    def __init__(self,
                 target_means=(0., 0., 0., 0., 0., 0.),
                 target_stds=(1., 1., 1., 1., 1., 1.),
                 angle_range='oc'):
        super(BaseBBoxCoder, self).__init__()
        self.means = target_means
        self.stds = target_stds
        self.version = angle_range

    def encode(self, bboxes, gt_bboxes):
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.

        Args:
            bboxes (torch.Tensor): Source boxes, e.g., object proposals.
            gt_bboxes (torch.Tensor): Target of the transformation, e.g.,
                ground-truth boxes.

        Returns:
            torch.Tensor: Box transformation deltas
        """
        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == 4
        assert gt_bboxes.size(-1) == 5
        encoded_bboxes = bbox2delta(bboxes, gt_bboxes, self.means, self.stds,
                                    self.version)
        return encoded_bboxes

    def decode(self,
               bboxes,
               pred_bboxes,
               max_shape=None,
               wh_ratio_clip=16 / 1000):
        """Apply transformation `pred_bboxes` to `bboxes`.

        Args:
            bboxes (torch.Tensor): Basic boxes. Shape (B, N, 4) or (N, 4)
            pred_bboxes (torch.Tensor): Encoded offsets with respect to each
                roi. Has shape (B, N, 5) or (N, 5).
                Note N = num_anchors * W * H when rois is a grid of anchors.

            max_shape (Sequence[int] or torch.Tensor or Sequence[
               Sequence[int]],optional): Maximum bounds for boxes, specifies
               (H, W, C) or (H, W). If bboxes shape is (B, N, 6), then
               the max_shape should be a Sequence[Sequence[int]]
               and the length of max_shape should also be B.
            wh_ratio_clip (float, optional): The allowed ratio between
                width and height.

        Returns:
            torch.Tensor: Decoded boxes.
        """
        assert pred_bboxes.size(0) == bboxes.size(0)
        assert bboxes.size(-1) == 4
        assert pred_bboxes.size(-1) == 6
        decoded_bboxes = delta2bbox(bboxes, pred_bboxes, self.means, self.stds,
                                    wh_ratio_clip, self.version)

        return decoded_bboxes


@mmcv.jit(coderize=True)
def bbox2delta(proposals,
               gt,
               means=(0., 0., 0., 0., 0., 0.),
               stds=(1., 1., 1., 1., 1., 1.),
               version='oc'):
    """Compute deltas of proposals w.r.t. gt.

    We usually compute the deltas of x, y, w, h, a, b of proposals w.r.t ground
    truth bboxes to get regression target. This is the inverse function of
    :func:`delta2bbox`.

    Args:
        proposals (torch.Tensor): Boxes to be transformed, shape (N, ..., 4)
        gt (torch.Tensor): Gt bboxes to be used as base, shape (N, ..., 5)
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates.
        version (str, optional): Angle representations. Defaults to 'oc'.

    Returns:
        Tensor: deltas with shape (N, 6), where columns represent dx, dy,
            dw, dh, da, db.
    """
    proposals = proposals.float()
    gt = gt.float()

    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0]
    ph = proposals[..., 3] - proposals[..., 1]

    hbb, poly = obb2xyxy(gt, version), obb2poly(gt, version)
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


@mmcv.jit(coderize=True)
def delta2bbox(rois,
               deltas,
               means=(0., 0., 0., 0., 0., 0.),
               stds=(1., 1., 1., 1., 1., 1.),
               wh_ratio_clip=16 / 1000,
               version='oc'):
    """Apply deltas to shift/scale base boxes.

    Typically the rois are anchor or proposed bounding boxes and the deltas
    are network outputs used to shift/scale those boxes. This is the inverse
    function of :func:`bbox2delta`.


    Args:
        rois (torch.Tensor): Boxes to be transformed. Has shape (N, 4).
        deltas (torch.Tensor): Encoded offsets relative to each roi.
            Has shape (N, num_classes * 4) or (N, 4). Note
            N = num_base_anchors * W * H, when rois is a grid of
            anchors.
        means (Sequence[float]): Denormalizing means for delta coordinates.
            Default (0., 0., 0., 0., 0., 0.).
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates. Default (1., 1., 1., 1., 1., 1.).
        wh_ratio_clip (float): Maximum aspect ratio for boxes. Default
            16 / 1000.
        version (str, optional): Angle representations. Defaults to 'oc'.

    Returns:
        Tensor: Boxes with shape (N, num_classes * 5) or (N, 5), where 5
           represent cx, cy, w, h, a.
    """
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 6)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 6)
    denorm_deltas = deltas * stds + means
    dx = denorm_deltas[:, 0::6]
    dy = denorm_deltas[:, 1::6]
    dw = denorm_deltas[:, 2::6]
    dh = denorm_deltas[:, 3::6]
    da = denorm_deltas[:, 4::6]
    db = denorm_deltas[:, 5::6]
    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)
    # Compute center of each roi
    px = ((rois[:, 0] + rois[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
    py = ((rois[:, 1] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)
    # Compute width/height of each roi
    pw = (rois[:, 2] - rois[:, 0]).unsqueeze(1).expand_as(dw)
    ph = (rois[:, 3] - rois[:, 1]).unsqueeze(1).expand_as(dh)
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
    center_polys = center_polys * diag_scale_factor.repeat_interleave(
        2, dim=-1)
    rectpolys = center_polys + center
    obboxes = poly2obb(rectpolys, version)
    return obboxes
